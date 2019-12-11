#!/usr/bin/env python
"""
Find best connections between hits of the secutive layers.
"""
import pandas as pd
import numpy as np
import torch
import pickle
import time
from collections import OrderedDict

import logging

logger = logging.getLogger(__name__)


vlids = [(8,2), (8,4), (8,6), (8,8),
         (13,2), (13,4), (13,6), (13,8),
         (17,2), (17,4)
        ]
n_det_layers = len(vlids)


def add_features(hit_ids, hits, truth):
    df_hits_used = pd.DataFrame(hit_ids, columns=['hit_id'])
    df = df_hits_used.merge(hits, on='hit_id', how='inner')
    df = df.merge(truth, on='hit_id', how='inner')
    vlid_groups = df.groupby(['volume_id', 'layer_id'])
    df = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i)
                    for i in range(n_det_layers)
                   ])

    r = np.sqrt(df.x**2 + df.y**2)
    phi = np.arctan2(df.y, df.x)
    df = df.assign(r=r, phi=phi)
    return df


def create_glue(graph, weights, hit_ids, hits, truth):
    df = add_features(hit_ids, hits, truth)

    # DF: hit_id and hit_idx in graph
    df_hit_ids = pd.DataFrame(hit_ids, columns=['hit_id'])
    df_hit_ids = df_hit_ids.assign(idx=df_hit_ids.index)

    def glue(outer_layer, inner_layer):
        # hit id in trkML data
        outer_hits = df[df.layer == outer_layer][['hit_id']]
        inner_hits = df[df.layer == inner_layer][['hit_id']]

        # hit index in Graph
        outer_hits_idx = df_hit_ids.merge(outer_hits, on='hit_id', how='inner')['idx']
        inner_hits_idx = df_hit_ids.merge(inner_hits, on='hit_id', how='inner')['idx']
        logger.debug("# outer hits: {}".format(outer_hits_idx.shape[0]))
        logger.debug("# inner hits: {}".format(inner_hits_idx.shape[0]))

        def extrapolate(hit_indexs, weight, g_start, g_end, reverse=False):
            """
            Find its pair in inner/outer layer for each hit in hit_indexs
            return the pairs and indexs of corresponding edges
            """
            cand_idx = []
            cand_edge_idx = []
            for idx in hit_indexs:
                hit_out = g_start[idx]
                w_edges = hit_out * weight
                next_hit_id = -1
                edge_idx = -1
                if w_edges.nonzero()[0].shape[0] > 0:
                    weighted_outgoing = np.argsort(w_edges)
                    weight_idx = weighted_outgoing[-1]
                    next_hit = g_end[:, weight_idx].nonzero()
                    if next_hit[0].shape[0] > 0:
                        next_hit_id = next_hit[0][0]
                        edge_idx = weight_idx

                cand_edge_idx.append(edge_idx)
                cand_idx.append(next_hit_id)

            if reverse:
                res_expolate = [(y,x) for x,y in zip(hit_indexs, cand_idx)]
            else:
                res_expolate = [(x,y) for x,y in zip(hit_indexs, cand_idx)]
            return res_expolate, cand_edge_idx

        # step 1: best choice of each other
        res_inner_expolate, sel_inner_edge_idx = extrapolate(outer_hits_idx, weights, graph.Ri, graph.Ro)
        res_outer_expolate, sel_outer_edge_idx = extrapolate(inner_hits_idx, weights, graph.Ro, graph.Ri, True)

        good_match = {}
        to_be_det_idx = OrderedDict()
        selected_edge_idx = []
        for idx,ii in enumerate(res_inner_expolate):
            if ii in res_outer_expolate:
                good_match[ii[0]] = ii[1]
                selected_edge_idx.append(sel_inner_edge_idx[idx])
            else:
                to_be_det_idx[ii[0]] = ii[1]

        logger.debug("# of reciprocal choice: {}".format(len(good_match.keys())))
        logger.debug("# of to-be-determined:  {}".format(len(to_be_det_idx.keys())))
        logger.debug("# of good outer exp.:   {}".format(len(res_outer_expolate)))

        values = np.array(list(to_be_det_idx.values()))
        vals, count = np.unique(values, return_counts=True)
        idx_vals_repeated = np.where(count > 1)[0]
        repeated_out_hits = vals[idx_vals_repeated]
        logger.debug("unique match:     {}".format( count.shape[0]) )
        logger.debug("duplicated match: {}".format( repeated_out_hits.shape[0]) )

        # step 2: user inner expolation to determine those to-be-determined
        used_hits = list(good_match.values())
        inner_can_tbd = []
        sel_edges_tbd = []
        for idx in to_be_det_idx.keys():
            hit_out = graph.Ri[idx]
            w_edges = hit_out * weights
            next_hit_id = -1
            edge_idx = -1
            if w_edges.nonzero()[0].shape[0] > 0:
                weighted_outgoing = np.argsort(w_edges)
                ii = -1
                while True:
                    weight_idx = weighted_outgoing[ii]
                    next_hit = graph.Ro[:, weight_idx].nonzero()
                    if next_hit[0].shape[0] > 0:
                        next_hit_id1 = next_hit[0][0]
                        if next_hit_id1 not in used_hits:
                            used_hits.append(next_hit_id1)
                            next_hit_id = next_hit_id1
                            edge_idx = weight_idx
                            break
                    ii -= 1

            inner_can_tbd.append(next_hit_id)
            sel_edges_tbd.append(edge_idx)

        # step 3: get truth edges, calculate precision
        res_truth_exp, truth_edge_idx = extrapolate(outer_hits_idx, graph.y, graph.Ri, graph.Ro)
        ## remove the ones with -1
        truth_edge_idx = [x for x in truth_edge_idx if x != -1]
        n_true_edges = int(np.sum(graph.y[truth_edge_idx]))
        all_edges    = selected_edge_idx + sel_edges_tbd
        all_edges = [x for x in all_edges if x != -1]
        vals, counts = np.unique(all_edges, return_counts=True)
        n_true_pos_edge = int(np.sum(graph.y[vals]))
        n_true_edge = int(np.sum(graph.y[truth_edge_idx]))
        precision = n_true_pos_edge*1.0/n_true_edges
        logger.debug("# of unique edges: {}".format( vals.shape[0]) )
        logger.debug("# of repeated    : {}".format( vals[np.where(counts > 1)].shape[0]) )
        logger.debug("# of total edges:  {}".format( len(all_edges)) )
        logger.debug("# of positive:     {}".format( n_true_pos_edge) )
        logger.debug("# of true edges:   {}".format( n_true_edge) )
        logger.debug("precision:         {}".format( precision ))
        if False:
            print("Edge info", precision)
            print(list(to_be_det_idx.keys()))
            print(inner_can_tbd)
            print(sel_edges_tbd)
            print(vals)
            print(truth_edge_idx)



        # setup 4: get final pairs
        final_pairs = [(x, y) for x,y in good_match.items()] \
                + [(x, y) for x,y in zip(to_be_det_idx.keys(), inner_can_tbd)]

        return final_pairs, precision

    return glue


def get_tracks(graph, weights, hit_ids, hits, truth):
    """
    pairs are a list of pairs for each pair of consective layers
    """
    glue_func = create_glue(graph, weights, hit_ids, hits, truth)
    all_pairs = []
    all_precisions = []
    for ilayer in range(n_det_layers-1):
        start_time = time.time()
        outer_layer = n_det_layers - 1 - ilayer
        inner_layer = outer_layer - 1
        layer_pairs, precision = glue_func(outer_layer, inner_layer)
        all_pairs.append(layer_pairs)
        all_precisions.append(precision)
        end_time = time.time()
        logger.debug("{} to {} takes: {:.1f} ms with precision {:.4f}".format(
            outer_layer, inner_layer, (end_time - start_time)*1000, precision)
        )

    list_pairs = []
    for ilay, pair in enumerate(all_pairs):
        pair_dict = {}
        for edges in pair:
            pair_dict[edges[0]] = edges[1]
        list_pairs.append(pair_dict)

    used_hits = []
    all_tracks = []
    for ilayer in range(n_det_layers-1):
        for hit,next_hit in list_pairs[ilayer].items():
            if hit in used_hits:
                continue
            if next_hit == -1:
                all_tracks.append([hit_ids[hit]])
                used_hits.append(hit)
                continue

            a_track = [hit_ids[hit], hit_ids[next_hit]]
            used_hits += [hit, next_hit]
            for ii in range(ilayer+1, n_det_layers-1, 1):
                try:
                    nn_hit = list_pairs[ii][next_hit]
                    if nn_hit == -1 or nn_hit in used_hits:
                        break
                except KeyError:
                    break
                a_track.append(hit_ids[nn_hit])
                used_hits.append(nn_hit)
                next_hit = nn_hit
            all_tracks.append(a_track)

    return all_tracks

if __name__ == "__main__":
    from datasets.graph import load_graph
    file_name = '/global/cscratch1/sd/xju/heptrkx/data/hitgraphs_001/event000001000_g000.npz'
    id_name   = '/global/cscratch1/sd/xju/heptrkx/data/hitgraphs_002/event000001000_g000_ID.npz'

    logging.basicConfig(filename='glue.log', level=logging.INFO)

    G = load_graph(file_name)
    with np.load(id_name) as f:
        hit_ids = f['ID']

    from score import load_model
    from score import load_config
    model_config_file = 'configs/segclf_small_new.yaml'

    model = load_model(load_config(model_config_file), reload_epoch=30).eval()
    batch_input = [torch.from_numpy(m[None]).float() for m in [G.X, G.Ri, G.Ro]]
    with torch.no_grad():
        weights = model(batch_input).flatten().numpy()

    logger.info("precision: {}".format( np.sum(weights*G.y)/G.y.nonzero()[0].shape[0] ) )


    event_input_name = '/global/cscratch1/sd/xju/heptrkx/trackml_inputs/train_all/event000001000'
    from trackml.dataset import load_event
    hits, cells, particles, truth = load_event(event_input_name)

    glue_func = create_glue(G, weights, hit_ids, hits, truth)
    all_pairs = []
    all_precisions = []
    for ilayer in range(n_det_layers-1):
        start_time = time.time()
        outer_layer = n_det_layers - 1 - ilayer
        inner_layer = outer_layer - 1
        layer_pairs, precision = glue_func(outer_layer, inner_layer)
        all_pairs.append(layer_pairs)
        all_precisions.append(precision)
        end_time = time.time()
        logger.info("{} to {} takes: {:.1f} ms with precision {:.4f}".format(
            outer_layer, inner_layer, (end_time - start_time)*1000, precision)
        )

