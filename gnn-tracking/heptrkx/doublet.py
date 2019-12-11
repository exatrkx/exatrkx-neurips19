"""
Class to make doublet
"""

from heptrkx.master import Event
from heptrkx import load_yaml, select_pair_layers, layer_pairs
from heptrkx import seeding

import os
import pandas as pd

class Segments(object):
    def create_segments(hits, layer_pair, gid_keys='layer'):
        raise NotImplementedError

class CutBasedSegments(Segments):
    def __init__(self):
        self._verbose = False

    def set_verbose(self, verbose):
        self._verbose = verbose

    def setup_from_config(self, config_dir):
        config = load_yaml(config_dir)
        evt_dir = config['track_ml']['dir']
        layers = config['doublets_from_cuts']['layers']
        phi_slope_max = config['doublets_from_cuts']['phi_slope_max']
        z0_max = config['doublets_from_cuts']['z0_max']
        min_hits = config['doublets_from_cuts']['min_hits']
        base_outdir = config['doublets_from_cuts']['selected']
        self.setup(evt_dir, phi_slope_max, z0_max, layers, min_hits, base_outdir)

    def setup(self, evt_dir, phi_slope_max, z0_max,
              layers=None, min_hits=3,
              outdir="doublets"):
        self._evt_dir = evt_dir
        self._phi_slope_max = phi_slope_max
        self._z0_max = z0_max
        self._layers = layers
        self._base_outdir = outdir
        if self._layers:
            self._sel_layer_id = select_pair_layers(layers)
        else:
            self._sel_layer_id = list(range(len(layer_pairs)))
        self._min_hits = min_hits

    @property
    def evt_dir(self):
        return self._evt_dir

    def get_outdir(self, evtid):
        return os.path.join(self._base_outdir, 'evt{}'.format(evtid))

    def is_exist(self, evtid):
        verbose = self._verbose
        res = False
        outdir = self.get_outdir(evtid)
        if os.path.exists(outdir):
            hits_outname = os.path.join(outdir, "event{:09d}-hits.h5".format(evtid))
            if os.path.exists(hits_outname):
                has_all_pairs = True
                for pair_idx in self._sel_layer_id:
                    outname = os.path.join(outdir, "pair{:03d}.h5".format(pair_idx))
                    if not os.path.exists(outname):
                        has_all_pairs = False
                        break
                if has_all_pairs:
                    if verbose:
                        print("Event {} has all output files".format(evtid))
                    res = True
        else:
            os.makedirs(outdir, exist_ok=True)

        return res

    def __call__(self, evtid, call_back=False):
        if self.is_exist(evtid):
            return (None, None, None)

        outdir = self.get_outdir(evtid)
        try:
            event = Event(self._evt_dir, evtid)
        except Exception as e:
            print(e)
            return (None, None, None)

        event.filter_hits(self._layers)
        event.remove_duplicated_hits()
        hits = event.hits
        verbose = self._verbose
        min_hits = self._min_hits
        phi_slope_max = self._phi_slope_max
        z0_max = self._z0_max

        ## particles having at least mininum number of hits associated
        cut = hits[hits.particle_id != 0].groupby('particle_id')['hit_id'].count() > min_hits
        pids = cut[cut].index
        if verbose:
            print("event {} has {} particles with at least {} hits".format(
                evtid, len(pids), min_hits))
        del cut

        if call_back:
            tot_list = []
            sel_true_list = []
            sel_list = []

        os.makedirs(outdir, exist_ok=True)
        hits_outname = os.path.join(outdir, "event{:09d}-hits.h5".format(evtid))
        if os.path.exists(hits_outname):
            if verbose:
                print("Found {}".format(hits_outname))
        else:
            with pd.HDFStore(hits_outname, 'w') as store:
                store['data'] = hits

        for pair_idx in self._sel_layer_id:
            outname = os.path.join(outdir, "pair{:03d}.h5".format(pair_idx))
            if os.path.exists(outname):
                if verbose:
                    print("Found {}".format(outname))
                continue

            layer_pair = layer_pairs[pair_idx]
            df = seeding.create_segments(hits, layer_pair)
            df.loc[~df.particle_id.isin(pids), 'true'] = False

            tot = df[df.true].pt.to_numpy()
            sel_true = df[
                (df.true)\
                & (df.phi_slope.abs() < phi_slope_max)\
                & (df.z0.abs() < z0_max)
            ].pt.to_numpy()
            df_sel = df[(df.phi_slope.abs() < phi_slope_max) &\
                        (df.z0.abs() < z0_max)]
            if call_back:
                tot_list.append(tot)
                sel_true_list.append(sel_true)
                sel = df_sel.pt.to_numpy()
                sel_list.append(sel)

            efficiency = sel_true.shape[0]/tot.shape[0]
            purity = sel_true.shape[0]/df_sel.shape[0]
            if verbose:
                print("event {}: pair ({}, {}), {} true segments, {} selected, {} true ones selected\n\
                      segment efficiency {:.2f}% and purity {:.2f}%".format(
                          evtid, layer_pair[0], layer_pair[1],
                          tot.shape[0], sel.shape[0], sel_true.shape[0],
                          100.*efficiency, 100.*purity
                      )
                     )

            with pd.HDFStore(outname, 'w') as store:
                store['data'] = df_sel
                store['info'] = pd.Series([efficiency, purity], index=['efficiency', 'purity'])

        if call_back:
            return (tot_list, sel_true_list, sel_list)
        else:
            return None

