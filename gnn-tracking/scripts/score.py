#!/usr/bin/env python3

from postprocess.evaluate_tf import create_evaluator

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from trackml.dataset import load_event

from graph_nets import utils_np
from nx_graph.utils_data import merge_truth_info_to_hits



if __name__ == "__main__":
    import sys
    import argparse
    import os
    import glob
    from heptrkx import load_yaml

    parser = argparse.ArgumentParser(description='Get a score from postprocess')
    add_arg = parser.add_argument
    add_arg('train_config', nargs='?', default='configs/nxgraph_test_NOINT.yaml')
    add_agr('--evt', default=6000, type=int, help='event ID for evaluation')
    add_agr('--sec', default=-1,   type=int, help='section ID for evaluation, default use all sections')
    add_arg('--nEpoch',     default=7128, type=int, help='reload model from an epoch')
    add_arg('-c', '--ckpt', default='trained_results/nxgraph_big_test_NOINT/bak', help='path that stores checkpoint')
    add_arg('--trkml', default='/global/cscratch1/sd/xju/heptrkx/trackml_inputs/train_all', help='original tracking data')

    args = parser.parse_args()
    config_file = args.train_config
    input_ckpt = args.ckpt
    iteration = args.nEpoch

    # create the model
    model = create_evaluator(config_file, iteration, input_ckpt)
    config = load_yaml(config_file)

    # prepare for data
    file_dir = config['data']['output_nxgraph_dir']
    base_dir =  os.path.join(file_dir, "event00000{}_g{:09d}_INPUT.npz")
    evtid = args.evt
    isec = args.sec

    file_names = []
    section_list = []
    if isec < 0:
        section_patten = base_dir.format(evtid, 0).replace('_g{:09}'.format(0), '*')
        n_sections = int(len(glob.glob(section_patten)))
        file_names = [base_dir.format(evtid, ii) for ii in range(n_sections)]
        section_list = [ii for ii in range(n_sections)]
    else:
        file_names = [base_dir.format(evtid, isec)]
        section_list = [isec]


    batch_size = config['train']['batch_size']
    n_batches = len(file_names)//2 + 1
    split_inputs = np.array_split(file_names, n_batches)

    hits_graph_dir = config['data']['input_hitsgraph_dir']
    trk_dir = args.trkml
    dd = os.path.join(trk_dir, 'event{:09d}')
    hits, particles, truth = load_event(dd.format(evtid), parts=['hits', 'particles', 'truth'])
    hits = merge_truth_info_to_hits(hits, particles, truth)
    true_features = ['pt', 'particle_id', 'nhits']

    all_graphs = []
    # evaluate each graph
    for ibatch in range(n_batches):
        ## pad batch_size
        current_files = list(split_inputs[ibatch])
        if len(current_files) < batch_size:
            last_file = current_files[-1]
            current_files += [last_file] *(batch_size-len(current_files))

        input_graphs = []
        target_graphs = []
        for file_name in current_files:
            with np.load(file_name) as f:
                input_graphs.append(dict(f.items()))

            with np.load(file_name.replace("INPUT", "TARGET")) as f:
                target_graphs.append(dict(f.items()))

        graphs = model(utils_np.data_dicts_to_graphs_tuple(input_graphs),
                       utils_np.data_dicts_to_graphs_tuple(target_graphs))
        if len(graphs) != batch_size:
            raise ValueError("graph size not the same as batch-size")

        # decorate the graph with truth info
        for ii in range(batch_size):
            idx = ibatch*batch_size + ii
            id_name = os.path.join(hits_graph_dir, "event{:09d}_g{:03d}_ID.npz".format(evtid, idx))
            with np.load(id_name) as f:
                hit_ids = f['ID']

            for node in graphs[ii].nodes():
                hit_id = hit_ids[node]
                graphs[ii].node[node]['hit_id'] = hit_id
                graphs[ii].node[node]['info'] = hits[hits['hit_id'] == hit_id][true_features].values

        all_graphs += graphs


    # after get all_graphs...
    # start to do some performance checks
