"""Algorithms to select doublets and triplets
"""
from __future__ import print_function

import pandas as pd
import numpy as np

def create_segments(hits, layer_pair, gid_keys='layer',
                    only_true=False, cluster_info=True,
                    origin_pos=True,
                    verbose=False
                   ):
    """Return all segments from all hits in the layer pair.
    only_true --  option to return only true segments, otherwise return all segments.
    """
    if 'particle_id' not in hits.columns:
        print('particle_id should be in hits,'
              'please merge hits and particles')
        return None

    if 'eta' not in hits.columns:
        # calculate eta, phi for each hit
        x = hits.x.values
        y = hits.y.values
        z = hits.z.values
        r = np.sqrt(x**2 + y**2)
        r3 = np.sqrt(r**2 + z**2)
        phi = np.arctan2(y, x)
        theta = np.arccos(z/r3)
        eta = -np.log(np.tan(theta/2.))
        hits = hits.assign(eta=eta, phi=phi)

    try:
        l1, l2 = layer_pair
    except:
        print("layer_pair should be a tuple")
        return None


    def calc_dphi(phi1, phi2):
        """Computes phi2-phi1 given in range [-pi,pi]"""
        dphi = phi2 - phi1
        dphi[dphi > np.pi] -= 2*np.pi
        dphi[dphi < -np.pi] += 2*np.pi
        return dphi

    hits1 = hits[hits[gid_keys] == l1]
    hits2 = hits[hits[gid_keys] == l2]

    hit_pairs = pd.merge(
        hits1.reset_index(), hits2.reset_index(),
        how='inner', on='evtid', suffixes=('_in', '_out'))

    # Identify the true pairs
    true_edges = (hit_pairs.particle_id_in == hit_pairs.particle_id_out) \
                 & (hit_pairs.particle_id_in != 0)

    if only_true:
        hit_pairs = hit_pairs[true_edges]

    # Calculate coordinate differences
    dphi = calc_dphi(hit_pairs.phi_in, hit_pairs.phi_out)
    dz = hit_pairs.z_out - hit_pairs.z_in
    dr = hit_pairs.r_out - hit_pairs.r_in
    phi_slope = dphi / dr
    z0 = hit_pairs.z_in - hit_pairs.r_in * dz / dr
    deta = hit_pairs.eta_in - hit_pairs.eta_out
    rz_slope = np.arctan2(dr, dz)

    selected_features = [
        'evtid', 'index_in', 'index_out',
        'hit_id_in', 'hit_id_out',
        'particle_id_in',
        'pt_in',
    ]
    if origin_pos:
        selected_features += [
            'x_in', 'x_out',
            'y_in', 'y_out',
            'z_in', 'z_out',
            'r_in', 'r_out',
            'phi_in', 'phi_out',
            'layer_in', 'layer_out',
        ]
    if cluster_info and 'lx_in' in hit_pairs.columns:
        selected_features += [
            'lx_in', 'lx_out',
            'ly_in', 'ly_out',
            'lz_in', 'lz_out']

    # Put the results in a new dataframe
    df_pairs = hit_pairs[selected_features].assign(
        dphi=dphi, dz=dz, dr=dr, true=true_edges,
        phi_slope=phi_slope, z0=z0, deta=deta, rz_slope=rz_slope
    )

    df_pairs = df_pairs.rename(
        columns={
            'index_in': 'hit_idx_in',
            "index_out": 'hit_idx_out',
            'particle_id_in':'particle_id',
            'pt_in':"pt",
        }
    )
    if cluster_info:
        try:
            deta1 = hit_pairs.geta_out - hit_pairs.geta_in
            dphi1 = hit_pairs.gphi_out - hit_pairs.gphi_in
        except (KeyError, AttributeError):
            pass
        else:
            df_pairs = df_pairs.assign(deta1=deta1, dphi1=dphi1)
    if verbose:
        n_true = df_pairs[df_pairs['true']].shape[0]
        n_fake = df_pairs[~df_pairs['true']].shape[0]
        print('Layer:({}, {}), {:,} ({:.3f}% of total edges) True edges and {:,} fake edges,'.format(
            l1, l2, n_true, 100*n_true/(n_true+n_fake), n_fake))

    return df_pairs
