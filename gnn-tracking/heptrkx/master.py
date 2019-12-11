"""Master class
"""
import os
import re
import glob

import numpy as np
import pandas as pd

from trackml.dataset import load_event

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

eventid_info = {}
def get_event(data_dir, n_events):
    if data_dir in eventid_info.keys():
        evt_ids = eventid_info[data_dir]
    else:
        all_files = glob.glob(os.path.join(data_dir, '*-hits.csv*'))
        evt_ids = np.sort([int(re.search('event0000([0-9]*)-hits.csv.gz',
                                         os.path.basename(x)).group(1))
                           for x in all_files])
    if n_events > len(evt_ids):
        print("Requested number of events {} larger than the total Events: {}".format(n_events, len(evt_ids)))
        n_events = len(evt_ids)
    return [Event(data_dir, evtid) for evtid in evt_ids[:n_events]]


class Event(object):
    """An object saving Event info, including hits, particles, truth and cell info"""
    def __init__(self, evtdir, evtid, blacklist_dir=None):
        self._evt_dir = evtdir
        self._blacklist_dir = blacklist_dir
        if not self.read(evtid):
            msg="Failed to read {} with event ID {}".format(evtdir, evtid)
            raise Exception('Event', msg)

    def read(self, evtid):
        prefix = os.path.join(os.path.expandvars(self._evt_dir),
                              'event{:09d}'.format(evtid))

        all_data = load_event(prefix, parts=['hits', 'particles', 'truth', 'cells'])
        if all_data is None:
            return False

        hits, particles, truth, cells = all_data
        hits = hits.assign(evtid=evtid)

        if self._blacklist_dir:
            prefix_bl = os.path.join(os.path.expandvars(self._blacklist_dir),
                                     'event{:09d}-blacklist_'.format(evtid))
            hits_exclude = pd.read_csv(prefix_bl+'hits.csv')
            particles_exclude = pd.read_csv(prefix_bl+'particles.csv')
            hits = hits[~hits['hit_id'].isin(hits_exclude['hit_id'])]
            particles = particles[~particles['particle_id'].isin(particles_exclude['particle_id'])]


        ## add pT to particles
        px = particles.px
        py = particles.py
        pt = np.sqrt(px**2 + py**2)
        particles = particles.assign(pt=pt)

        self._evtid = evtid
        self._hits = hits
        self._particles = particles
        self._truth = truth
        self._cells = cells

        self.merge_truth_info_to_hits()
        return True

    @property
    def particles(self):
        return self._particles

    @property
    def hits(self):
        return self._hits

    @property
    def cells(self):
        return self._cells

    @property
    def truth(self):
        return self._truth

    @property
    def evtid(self):
        return self._evtid


    def merge_truth_info_to_hits(self):
        hits = self._hits
        hits = hits.merge(self._truth, on='hit_id', how='left')
        hits = hits.merge(self._particles, on='particle_id', how='left')

        # noise hits does not have particle info
        # yielding NaN value
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
        self._hits = hits

    def reconstructable_pids(self, min_hits=4):
        truth_particles = self.particles.merge(self.truth, on='particle_id', how='left')
        reconstructable_particles = truth_particles[truth_particles.nhits > min_hits]
        return np.unique(reconstructable_particles.particle_id)

    def filter_hits(self, layers, inplace=True):
        """keep hits that are in the layers"""
        barrel_hits = self._hits[self._hits.layer.isin(layers)]
        if inplace:
            self._hits = barrel_hits
        return barrel_hits

    def remove_noise_hits(self, inplace=True):
        no_noise = self._hits[self._hits.particle_id > 0]
        if inplace:
            self._hits = no_noise
        return no_noise

    def remove_duplicated_hits(self, inplace=True):
        hits = self._hits.loc[
            self._hits.groupby(['particle_id', 'layer'], as_index=False).r.idxmin()
        ]
        if inplace:
            self._hits = hits
        return hits

    @staticmethod
    def _local_angle(cell, module):
        n_u = max(cell['ch0']) - min(cell['ch0']) + 1
        n_v = max(cell['ch1']) - min(cell['ch1']) + 1
        l_u = n_u * module.pitch_u.values   # x
        l_v = n_v * module.pitch_v.values   # y
        l_w = 2   * module.module_t.values  # z
        return (l_u, l_v, l_w)

    @staticmethod
    def _extract_rotation_matrix(module):
        rot_matrix = np.matrix( [[ module.rot_xu.values[0], module.rot_xv.values[0], module.rot_xw.values[0]],
                                 [  module.rot_yu.values[0], module.rot_yv.values[0], module.rot_yw.values[0]],
                                 [  module.rot_zu.values[0], module.rot_zv.values[0], module.rot_zw.values[0]]])
        return rot_matrix, np.linalg.inv(rot_matrix)

    @staticmethod
    def cartesion_to_spherical(x, y, z):
        r3 = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x)
        theta = np.arccos(z/r3)
        return r3, theta, phi

    @staticmethod
    def theta_to_eta(theta):
        return -np.log(np.tan(0.5*theta))

    def cluster_info(self, detector_dir, inplace=True):
        self._detector = pd.read_csv(os.path.expandvars(detector_dir))
        df_hits = self._hits
        cells = self._cells

        angles = []
        for ii in range(df_hits.shape[0]):
            hit = df_hits.iloc[ii]
            cell = cells[cells.hit_id == hit.hit_id]
            module = self._detector[(self.detector.volume_id == hit.volume_id)
                                    & (detector.layer_id == hit.layer_id)
                                    & (detector.module_id == hit.module_id)]
            l_x, l_y, l_z = self._local_angle(cell, module)
            # convert to global coordinates
            module_matrix, module_matrix_inv = self._extract_rotation_matrix(module)
            g_matrix = module_matrix * [l_x, l_y, l_z]
            _, g_theta, g_phi = cartesion_to_spherical(g_matrix[0][0], g_matrix[1][0], g_matrix[2][0])
            _, l_theta, l_phi = cartesion_to_spherical(l_x[0], l_y[0], l_z[0])
            # to eta and phi...
            l_eta = theta_to_eta(l_theta)
            g_eta = theta_to_eta(g_theta[0, 0])
            lx, ly, lz = l_x[0], l_y[0], l_z[0]
            angles.append([int(hit.hit_id), l_eta, l_phi, lx, ly, lz, g_eta, g_phi[0, 0]])
        df_angles = pd.DataFrame(angles, columns=['hit_id', 'leta', 'lphi', 'lx', 'ly', 'lz', 'geta', 'gphi'])
        if inplace:
            self._hits = self._hits.merge(df_angles, on='hit_id', how='left')
        return df_angles
