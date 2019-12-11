"""Handle read and write objects"""

import numpy as np
import pandas as pd
import os

import h5py

from graph_nets import utils_np
import networkx as nx

ckpt_name = 'checkpoint_{:05d}.ckpt'

def read_pairs_input(file_name):
    pairs = []
    iline = 0
    with open(file_name) as f:
        for line in f:
            if iline == 0:
                n_pairs = int(line[:-1])
                iline += 1
                continue
            pairs.append([int(x) for x in line[:-1].split()])
    pairs = np.array(pairs)
    return pairs


def load_data_dicts(file_name):
    assert(os.path.exists(file_name))
    with np.load(file_name) as f:
        return dict(f.items())

def load_input_target_data_dicts(path, evtid, isec):
    base_name = os.path.join(path, 'event{:09d}_g{:09d}_{}.npz')
    input_dd = load_data_dicts(base_name.format(evtid, isec, "INPUT"))
    target_dd = load_data_dicts(base_name.format(evtid, isec, "TARGET"))
    return input_dd, target_dd

def save_nx_to_hdf5(graph, output_name):
    if os.path.exists(output_name):
        print(output_name, "is there")
        return
    else:
        os.makedirs(os.path.dirname(output_name), exist_ok=True)

    number_of_nodes = graph.number_of_nodes()
    node_idxs, node_attr = zip(*graph.nodes(data=True))
    node_features = next(iter(graph.nodes(data=True)))[1].keys()

    senders, receivers, edge_attr_dicts = zip(*graph.edges(data=True))
    edge_features = next(iter(graph.edges(data=True)))[2].keys()

    with h5py.File(output_name, 'w') as f:
        dset = f.create_dataset('node_index', data=np.array(node_idxs))

        node_group = f.create_group('nodes')
        for feature in node_features:
            data = [x[feature] for x in node_attr if x[feature] is not None]
            if len(data) != number_of_nodes:
                raise ValueError("Either all the nodes should have features, or none of them")
            dset = node_group.create_dataset(feature, data=np.array(data))

        dset = f.create_dataset('senders', data=np.array(senders))
        dset = f.create_dataset('receivers', data=np.array(receivers))

        edge_group = f.create_group('edges')
        for feature in edge_features:
            data = [x[feature] for x in edge_attr_dicts if x[feature] is not None]
            dset = edge_group.create_dataset(feature, data=np.array(data))


def read_hdf5_to_nx(hdf5_name, use_digraph=True, bidirection=False):
    with h5py.File(hdf5_name, 'r') as f:
        # nodes
        node_idx = f['node_index']
        node_info = f['nodes']
        node_features_data = [node_info[x] for x in node_info.keys()]
        node_features = [
            dict([(name, node_features_data[iname][ii]) for iname,name in enumerate(node_info.keys())]) for ii in range(len(node_idx))
        ]
        # edges
        senders = f['senders']
        receivers = f['receivers']
        edge_info = f['edges']
        edge_features_data = [edge_info[x] for x in edge_info.keys()]
        edge_features = [
            dict([(name, edge_features_data[iname][ii]) for iname,name in enumerate(edge_info.keys())]) for ii in range(len(senders))
        ]

        graph = nx.DiGraph() if use_digraph else nx.Graph()
        graph.add_nodes_from(zip(node_idx, node_features))
        graph.add_edges_from(zip(senders, receivers, edge_features))
        if bidirection:
            graph.add_edges_from(zip(receivers, senders, edge_features))

        return graph

