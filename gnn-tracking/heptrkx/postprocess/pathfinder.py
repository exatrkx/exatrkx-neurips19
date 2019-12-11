"""
Loop over all hits;
for each hit, find next hit that has maximum weight among all available edge candidates
"""
import numpy as np
import networkx as nx

from .utils_fit import poly_fit, poly_val

def get_tracks(graph, weights, hit_ids, weight_cutoff):
    hits_in_tracks = []
    hits_idx_in_tracks = []
    all_tracks = []

    n_hits = graph.X.shape[0]
    for idx in range(n_hits):
        # Loop over all hits
        # and save hits that are used in a track
        hit_id = hit_ids[idx]
        if hit_id not in hits_in_tracks:
            hits_in_tracks.append(hit_id)
            hits_idx_in_tracks.append(idx)
        else:
            continue

        a_track = [hit_id]
        while(True):
            # for this hit index (idx),
            # find its outgoing hits that could form a track
            hit_out = graph.Ro[idx]
            if hit_out.nonzero()[0].shape[0] < 1:
                break
            weighted_outgoing = np.argsort((hit_out * weights))
            if weights[weighted_outgoing[-1]] < weight_cutoff:
                break
            ii = -1
            has_next_hit = False
            while abs(ii) < 15:
                weight_idx = weighted_outgoing[ii]
                next_hit = graph.Ri[:, weight_idx].nonzero()
                if next_hit[0].shape[0] > 0:
                    next_hit_id = next_hit[0][0]
                    if next_hit_id != idx and next_hit_id not in hits_idx_in_tracks:
                        hits_in_tracks.append(hit_ids[next_hit_id])
                        hits_idx_in_tracks.append(next_hit_id)
                        a_track       .append(hit_ids[next_hit_id])
                        idx = next_hit_id
                        has_next_hit = True
                        break
                ii -= 1

            if not has_next_hit:
                # no more out-going tracks
                break
        all_tracks.append(a_track)
    return all_tracks


def get_tracks2(G, th=0.5, feature_name='solution'):
    used_nodes = []
    sub_graphs = []
    for node in G.nodes():
        if node in used_nodes:
            continue
        a_track = longest_track(
            G, node,
            used_nodes, th=th, feature_name=feature_name)
        if len(a_track) < 1:
            used_nodes.append(node)
            continue

        sub = nx.edge_subgraph(G, a_track)
        sub_graphs.append(sub)
        used_nodes += list(sub.nodes())

    n_tracks = len(sub_graphs)
    print("total tracks:", n_tracks)
    return sub_graphs
