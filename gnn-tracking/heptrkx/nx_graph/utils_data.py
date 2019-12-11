import networkx as nx

import numpy as np
import pandas as pd
import math
import numbers

import os
from collections import namedtuple

from graph_nets import utils_np

Point = namedtuple('Point', ['x', 'y', 'z'])
Pos = namedtuple('Pos', ['x', 'y', 'z', 'eta', 'phi', 'theta', 'r3', 'r'])

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    if dphi > np.pi:
        dphi -= 2*np.pi
    if dphi < -np.pi:
        dphi += 2*np.pi
    return dphi


def pos_transform(r, phi, z):
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    r3 = math.sqrt(r**2 + z**2)
    theta = math.acos(z/r3)
    eta = -math.log(math.tan(theta*0.5))
    return Pos(x, y, z, eta, phi, theta, r3, r)


def dist(x, y):
    return math.sqrt(x**2 + y**2)


def wdist(a, d, w):
    pp = a.x*a.x + a.y*a.y + a.z*a.z*w
    pd = a.x*d.x + a.y*d.y + a.z*d.z*w
    dd = d.x*d.x + d.y*d.y + d.z*d.z*w
    return math.sqrt(abs(pp - pd*pd/dd))


def wdistr(r1, dr, az, dz, w):
    pp = r1*r1+az*az*w
    pd = r1*dr+az*dz*w
    dd = dr*dr+dz*dz*w
    return math.sqrt(abs(pp-pd*pd/dd))

def circle(a, b, c):
    ax = a.x-c.x
    ay = a.y-c.y
    bx = b.x-c.x
    by = b.y-c.y
    aa = ax*ax + ay*ay
    bb = bx*bx + by*by
    idet = 0.5/(ax*by-ay*bx)
    p0 = Point(x=(aa*by-bb*ay)*idet, y=(ax*bb-bx*aa)*idet, z=0)
    r = math.sqrt(p0.x*p0.x + p0.y*p0.y)
    p = Point(x=p0.x+c.x, y=p0.y+c.y, z=p0.z)
    return p, r


def zdists(a, b):
    origin = Point(x=0, y=0, z=0)
    p, r = circle(origin, a, b)
    ang_ab = 2*math.asin(dist(a.x-b.x, a.y-b.y)*0.5/r)
    ang_a = 2*math.asin(dist(a.x, a.y)*0.5/r)
    return abs(b.z-a.z-a.z*ang_ab/ang_a)


def get_edge_features2(in_node, out_node, add_angles=False):
    # input are the features of incoming and outgoing nodes
    # they are ordered as [r, phi, z]
    v_in = pos_transform(*in_node)
    v_out = pos_transform(*out_node)

    deta = v_out.eta - v_in.eta
    dphi = calc_dphi(v_out.phi, v_in.phi)
    dR = np.sqrt(deta**2 + dphi**2)
    #dZ = v_out.z - v_in.z
    dZ = v_in.z - v_out.z #

    results = {"distance": np.array([deta, dphi, dR, dZ])}

    if add_angles:
        pa = Point(x=v_out.x, y=v_out.y, z=v_out.z)
        pb = Point(x=v_in.x, y=v_in.y, z=v_in.z)
        pd = Point(x=pa.x-pb.x, y=pa.y-pb.y, z=pa.z-pb.z)

        wd0 = wdist(pa, pd, 0)
        wd1 = wdist(pa, pd, 1)
        zd0 = zdists(pa, pb)
        wdr = wdistr(v_out.r, v_in.r-v_out.r, pa.z, pd.z, 1)

        results['angles'] = np.array([wd0, wd1, zd0, wdr])

    return results

def get_edge_features(in_node, out_node):
    # input are the features of incoming and outgoing nodes
    # they are ordered as [r, phi, z]
    in_r, in_phi, in_z    = in_node
    out_r, out_phi, out_z = out_node

    in_r3 = np.sqrt(in_r**2 + in_z**2)
    out_r3 = np.sqrt(out_r**2 + out_z**2)

    in_theta = np.arccos(in_z/in_r3)
    in_eta = -np.log(np.tan(in_theta/2.0))
    out_theta = np.arccos(out_z/out_r3)
    out_eta = -np.log(np.tan(out_theta/2.0))
    deta = out_eta - in_eta
    dphi = calc_dphi(out_phi, in_phi)
    dR = np.sqrt(deta**2 + dphi**2)
    dZ = in_z - out_z
    return np.array([deta, dphi, dR, dZ])


def data_dict_to_nx(dd_input, dd_target, use_digraph=True, bidirection=True):
    input_nx  = utils_np.data_dict_to_networkx(dd_input)
    target_nx = utils_np.data_dict_to_networkx(dd_target)

    G = nx.DiGraph() if use_digraph else nx.Graph()
    for node_index, node_features in input_nx.nodes(data=True):
        G.add_node(node_index, pos=node_features['features'])

    for sender, receiver, features in target_nx.edges(data=True):
        G.add_edge(sender, receiver, solution=features['features'])
        if use_digraph and bidirection:
            G.add_edge(receiver, sender, solution=features['features'])

    return G


def correct_networkx(Gi, isec, n_phi_sections=8, n_eta_sections=2):
    G = Gi.copy()

    phi_range = (-np.pi, np.pi)
    phi_edges = np.linspace(*phi_range, num=n_phi_sections+1)
    scale = [1000, np.pi/n_phi_sections, 1000]
    # update phi
    phi_min = phi_edges[isec//n_eta_sections]
    phi_max = phi_edges[isec//n_eta_sections+1]
    for node_id, features in G.nodes(data=True):
        new_feature = features['pos']*scale
        new_feature[1] = new_feature[1] + (phi_min + phi_max) / 2
        if new_feature[1] > np.pi:
            new_feature[1] -= 2*np.pi
        if new_feature[1] < -np.pi:
            new_feature[1]+= 2*np.pi

        G.node[node_id].update(pos=new_feature)
    return G


def networkx_graph_to_hitsgraph(G, is_digraph=True):
    n_nodes = len(G.nodes())
    n_edges = len(G.edges())//2 if is_digraph else len(G.edges())
    n_features = len(G.node[0]['pos'])

    X = np.zeros((n_nodes, n_features), dtype=np.float32)
    Ri = np.zeros((n_nodes, n_edges), dtype=np.uint8)
    Ro = np.zeros((n_nodes, n_edges), dtype=np.uint8)

    for node,features in G.nodes(data=True):
        X[node, :] = features['pos']

    ## build relations
    segments = []
    y = []
    for n, nbrsdict in G.adjacency():
        for nbr, eattr in nbrsdict.items():
            ## as hitsgraph is a directed graph from inner-most to outer-most
            ## so assume sender < receiver;
            if n > nbr and is_digraph:
                continue
            segments.append((n, nbr))
            y.append(int(eattr['solution'][0]))

    if len(y) != n_edges:
        print(len(y),"not equals to # of edges", n_edges)
    segments = np.array(segments)
    Ro[segments[:, 0], np.arange(n_edges)] = 1
    Ri[segments[:, 1], np.arange(n_edges)] = 1
    y = np.array(y, dtype=np.float32)
    return (X, Ri, Ro, y)


def is_diff_networkx(G1, G2):
    """
    G1,G2, networkx graphs
    Return True if they are different, False otherwise
    note that edge features are not checked!
    """
    # check node features first
    GRAPH_NX_FEATURES_KEY = 'pos'
    node_id1 = np.array([
        x[1][GRAPH_NX_FEATURES_KEY]
        for x in G1.nodes(data=True)
        if x[1][GRAPH_NX_FEATURES_KEY] is not None])
    node_id2 = np.array([
        x[1][GRAPH_NX_FEATURES_KEY]
        for x in G2.nodes(data=True)
        if x[1][GRAPH_NX_FEATURES_KEY] is not None])

    # check edges
    diff = np.any(node_id1 != node_id2)
    for sender, receiver, _ in G1.edges(data=True):
        try:
            _ = G2.edges[(sender, receiver)]
        except KeyError:
            diff = True
            break
    return diff
## predefined group info
vlids = [(7,2), (7,4), (7,6), (7,8), (7,10), (7,12), (7,14),
         (8,2), (8,4), (8,6), (8,8),
         (9,2), (9,4), (9,6), (9,8), (9,10), (9,12), (9,14),
         (12,2), (12,4), (12,6), (12,8), (12,10), (12,12),
         (13,2), (13,4), (13,6), (13,8),
         (14,2), (14,4), (14,6), (14,8), (14,10), (14,12),
         (16,2), (16,4), (16,6), (16,8), (16,10), (16,12),
         (17,2), (17,4),
         (18,2), (18,4), (18,6), (18,8), (18,10), (18,12)]
n_det_layers = len(vlids)

def merge_truth_info_to_hits(hits, particles, truth):
    if 'pt' not in particles.columns:
        px = particles.px
        py = particles.py
        pt = np.sqrt(px**2 + py**2)
        particles = particles.assign(pt=pt)

    hits = hits.merge(truth, on='hit_id', how='left')
    hits = hits.merge(particles, on='particle_id', how='left')
    # selective information
    # noise hits does not have particle info
    hits = hits.fillna(value=0)

    # Assign convenient layer number [0-47]
    vlid_groups = hits.groupby(['volume_id', 'layer_id'])
    hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i)
                      for i in range(n_det_layers)])

    # add new features
    x = hits.x
    y = hits.y
    z = hits.z
    absz = np.abs(z)
    r = np.sqrt(x**2 + y**2) # distance from origin in transverse plane
    r3 = np.sqrt(r**2 + z**2) # in 3D
    phi = np.arctan2(hits.y, hits.x)
    theta = np.arccos(z/r3)
    eta = -np.log(np.tan(theta/2.))

    tpx = hits.tpx
    tpy = hits.tpy
    tpt = np.sqrt(tpx**2 + tpy**2)

    hits = hits.assign(r=r, phi=phi, eta=eta, r3=r3, absZ=absz, tpt=tpt)

    # add hit indexes to column hit_idx
    hits = hits.rename_axis('hit_idx').reset_index()
    return hits


def pairs_to_df(pairs, hits):
    """pairs is np.array, each row is a pair. columns are incoming and outgoing nodes
    return a DataFrame with columns,
    ['hit_id_in', 'hit_idx_in', 'layer_in', 'hit_id_out', 'hit_idx_out', 'layer_out']
    """

    # form a DataFrame
    in_nodes  = pd.DataFrame(pairs[:, 0], columns=['hit_id'])
    out_nodes = pd.DataFrame(pairs[:, 1], columns=['hit_id'])

    # add hit features
    ins  = in_nodes.merge(hits, on='hit_id', how='left')
    outs = out_nodes.merge(hits, on='hit_id', how='left')
    pid1 = ins['particle_id'].values
    pid2 = outs['particle_id'].values
    y = np.zeros(ins.shape[0], dtype=np.float32)
    y[:] = (pid1 == pid2) & (pid1 != 0)
    true_pairs = pd.DataFrame(y, columns=['true'])

    # rename incoming nodes and outgoing nodes, concatenate them
    ins  = ins.rename(columns={'hit_id': 'hit_id_in', "hit_idx": 'hit_idx_in', 'layer': 'layer_in'})
    outs = outs.rename(columns={'hit_id': 'hit_id_out', "hit_idx": 'hit_idx_out', 'layer': 'layer_out'})
    edges = pd.concat([ins[['hit_id_in', 'hit_idx_in', 'layer_in']], outs[['hit_id_out', 'hit_idx_out', 'layer_out']], true_pairs], axis=1)
    return edges


def hitsgraph_to_nx2(G, IDs=None, use_digraph=True, bidirection=True):
    n_nodes, n_edges = G.Ri.shape

    graph = nx.DiGraph() if use_digraph else nx.Graph()

    ## it is essential to add nodes first
    # the node ID must be [0, N_NODES]
    if IDs is None:
        for i in range(n_nodes):
            graph.add_node(i, pos=G.X[i], solution=[0.0])
    else:
        for i in range(n_nodes):
            graph.add_node(i, pos=G.X[i],
                           hit_id=IDs.iloc[i],
                           solution=[0.0])

    for iedge in range(n_edges):
        """
        In_node:  node is a receiver, hits at outer-most layers can only be In-node
        Out-node: node is a sender, so hits in inner-most layer can only be Out-node
        """
        in_node_id  = G.Ri[:, iedge].nonzero()[0][0]
        out_node_id = G.Ro[:, iedge].nonzero()[0][0]
        solution = [G.y[iedge]]
        _add_edge(graph, out_node_id, in_node_id, solution, bidirection)

    # add global features, not used for now
    graph.graph['features'] = np.array([0.])
    return graph

def hitsgraph_to_nx(G, IDs=None, bidirection=True):
    n_nodes, n_edges = G.Ri.shape

    graph = nx.DiGraph()

    ## it is essential to add nodes first
    # the node ID must be [0, N_NODES]
    if IDs is None:
        for i in range(n_nodes):
            graph.add_node(i, pos=G.X[i], solution=0.0)
    else:
        for i in range(n_nodes):
            graph.add_node(i, pos=G.X[i],
                           hit_id=IDs.iloc[i],
                           solution=[0.0])

    for iedge in range(n_edges):
        in_node_id  = G.Ri[:, iedge].nonzero()[0][0]
        out_node_id = G.Ro[:, iedge].nonzero()[0][0]

        # distance as features
        in_node_features  = G.X[in_node_id]
        out_node_features = G.X[out_node_id]
        distance = get_edge_features(in_node_features, out_node_features)
        # add edges, bi-directions
        graph.add_edge(in_node_id, out_node_id, distance=distance, solution=G.y[iedge])
        graph.add_edge(out_node_id, in_node_id, distance=distance, solution=G.y[iedge])
        # add "solution" to nodes
        graph.node[in_node_id].update(solution=G.y[iedge])
        graph.node[out_node_id].update(solution=G.y[iedge])

    # add global features, not used for now
    graph.graph['features'] = np.array([0.])
    return graph

def segments_to_nx(hits, segments,
                   sender_hitid_name,
                   receiver_hitid_name,
                   solution_name,
                   use_digraph=True, bidirection=True):
    """only pairs with both hits presented in hits are used
    hits: nodes in the graphs
    segments: DataFrame, with columns ['sender_hit_id', 'receiver_hit_id', 'solution_name'], true edge or not
    """
    graph = nx.DiGraph() if use_digraph else nx.Graph()
    graph.graph['features'] = np.array([0.])

    feature_names = ['r', 'phi', 'z']
    truth_features = ['pt', 'particle_id', 'nhits']

    n_hits = hits.shape[0]
    hits_id_dict = {}
    for idx in range(n_hits):
        hit_id = int(hits.iloc[idx]['hit_id'])
        graph.add_node(idx,
                       pos=hits.iloc[idx][feature_names].values,
                       hit_id=hit_id,
                       info=hits.iloc[idx][truth_features],
                       solution=[0.0])
        hits_id_dict[hit_id] = idx

    # senders   = [hits_id_dict[x] for x in segments[sender_hitid_name].values]
    # receivers = [hits_id_dict[x] for x in segments[receiver_hitid_name].values]
    # edge_features = [{"solution": [x]} for x in segments[solution_name]]
    # edge_data = zip(senders, receivers, edge_features)
    # graph.add_edges_from(edge_data)
    n_edges = segments.shape[0]
    for idx in range(n_edges):
        in_hit_idx  = int(segments.iloc[idx][sender_hitid_name])
        out_hit_idx = int(segments.iloc[idx][receiver_hitid_name])

        in_node_idx  = hits_id_dict[in_hit_idx]
        out_node_idx = hits_id_dict[out_hit_idx]

        solution = [segments.iloc[idx][solution_name]]
        _add_edge(graph, in_node_idx, out_node_idx, solution, bidirection)
    return graph


def _add_edge(G, sender, receiver, solution, bidirection, edge_features=None):
    f1 = G.node[sender]['pos']   # (r, phi, z)
    f2 = G.node[receiver]['pos']
    if f1[0] > f2[0]:
        # sender should have smaller *r*
        # swap
        sender, receiver = receiver, sender
        f1, f2 = f2, f1

    this_edge_features = get_edge_features2(f1, f2)
    if edge_features is not None:
        for key,value in edge_features.items():
            this_edge_features[key] = value

    G.add_edge(sender,  receiver, solution=solution, **this_edge_features)
    if bidirection:
        edge_features = get_edge_features2(f2, f1)
        G.add_edge(receiver,  sender, solution=solution, **edge_features)

    G.node[sender].update(solution=solution)
    G.node[receiver].update(solution=solution)


def predicted_graphs_to_nxs(gnn_output, input_graphs, target_graphs, **kwargs):
    output_nxs = utils_np.graphs_tuple_to_networkxs(gnn_output)
    input_dds  = utils_np.graphs_tuple_to_data_dicts(input_graphs)
    target_dds = utils_np.graphs_tuple_to_data_dicts(target_graphs)

    total_graphs = len(output_nxs)
    print("total_graphs", total_graphs)
    graphs = []
    for ig in range(total_graphs):
        input_dd = input_dds[ig]
        target_dd = target_dds[ig]

        graph = data_dict_to_nx(input_dd, target_dd, **kwargs)

        ## update edge features with TF output
        for edge in graph.edges():
            graph.edges[edge]['predict'] = output_nxs[ig].edges[edge+(0,)]['features']

        graphs.append(graph)
    return graphs


def get_true_subgraph(G):
    true_edges = []
    for iedge,edge in enumerate(G.edges(data=True)):
        if int(edge[2]['solution'][0]) == 1:
            true_edges.append((edge[0], edge[1]))

    Gp = nx.edge_subgraph(G, true_edges)
    return Gp


def nx_to_pandas(nx_G, edge_feature=None):
    df_nodes = pd.DataFrame([(ii, nx_G.node[ii]['hit_id']) for ii in nx_G.nodes()],
                            columns=['node_idx', 'hit_id'])

    df_edges = pd.DataFrame([(ii, nx_G.node[ff[0]]['hit_id'], nx_G.node[ff[1]]['hit_id'])
                             for ii, ff in enumerate(nx_G.edges(data=True))],
                            columns=['edge_idx', 'sender', 'receiver'])
    if edge_feature:
        if type(edge_feature) is not list: edge_feature = [edge_feature]
        for feature in edge_feature:
            dict_feature = nx.get_edge_attributes(nx_G, feature)
            first_ele = next(iter(dict_feature))

            if isinstance(first_ele, numbers.Number):
                new_column = [dict_feature[edge] for edge in nx_G.edges()]
            elif type(first_ele) is np.array and first_ele.shape[0] == 1:
                new_column = [dict_feature[edge][0] for edge in nx_G.edges()]
            else:
                print("data associated with", feature, " are not supported")
                continue
            df_edges = df.edges.assign(feature=np.array(new_column))


    return df_nodes, df_edges


def split_list(input_list, frac_train=0.8, frac_val=0.1):
    if type(input_list) is not list:
        print("input has to be a list")
        return [None]*3

    n_total = len(input_list)
    n_train = int(n_total*frac_train)
    n_val   = int(n_total*frac_val)
    n_test  = n_total - n_train - n_val
    return [input_list[:n_train],
            input_list[n_train:n_train+n_val],
            input_list[n_train+n_val:]]


def segments_to_hitsgraph(hits, segments, feature_names, feature_scale,
                          in_name='hit_id_in', out_name='hit_id_out'):
    n_hits = hits.shape[0]
    hit_ids = hits.hit_id.values
    segments = segments[segments[in_name].isin(hit_ids) & segments[out_name].isin(hit_ids)]
    n_edges = segments.shape[0]

    X = (hits[feature_names].values/feature_scale).astype(np.float32)
    Ri = np.zeros((n_hits, n_edges), dtype=np.uint8)
    Ro = np.zeros((n_hits, n_edges), dtype=np.uint8)
    y = np.zeros(n_edges, dtype=np.float32)
    I = hits['hit_id']

    hit_idx = pd.Series(np.arange(n_hits), index=hits.index)
    seg_start = hit_idx.loc[segments[in_name]].values
    seg_end = hit_idx.loc[segments[out_name]].values

    # Now we can fill the association matrices.
    # Note that Ri maps hits onto their incoming edges,
    # which are actually segment endings.
    Ri[seg_end, np.arange(n_edges)] = 1
    Ro[seg_start, np.arange(n_edges)] = 1
    # Fill the segment labels
    pid1 = hits.particle_id.loc[segments[in_name]].values
    pid2 = hits.particle_id.loc[segments[out_name]].values
    y[:] = (pid1 == pid2)
    # Return a tuple of the results
    return Graph(X, Ri, Ro, y), I
