"""Handle read and write objects"""

import os
import glob
import re

import numpy as np
import pandas as pd
import yaml
import os

import matplotlib.pyplot as plt

def evtids_at_disk(evt_dir):
    all_files = glob.glob(os.path.join(evt_dir, '*hits*'))
    evtids = np.sort([int(
        re.search('event([0-9]*)', os.path.basename(x).split('-')[0]).group(1))
        for x in all_files])
    return evtids


def load_yaml(file_name):
    assert(os.path.exists(file_name))
    with open(file_name) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def list_from_str(input_str):
    items = input_str.split(',')
    out = []
    for item in items:
        try:
            value = int(item)
            out.append(value)
        except ValueError:
            start, end = item.split('-')
            try:
                start, end = int(start), int(end)
                out += list(range(start, end+1))
            except ValueError:
                pass
    return out


layer_pairs = [
    (7, 8), (8, 9), (9, 10), (10, 24), (24, 25), (25, 26), (26, 27), (27, 40), (40, 41),
    (7, 6), (6, 5), (5, 4), (4, 3), (3, 2), (2, 1), (1, 0),
    (8, 6), (9, 6),
    (7, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17),
    (8, 11), (9, 11),
    (24, 23), (23, 22), (22, 21), (21, 19), (19, 18),
    (24, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33),
    (25, 23), (26, 23), (25, 28), (26, 28),
    (27, 39), (40, 39), (27, 42), (40, 42),
    (39, 38), (38, 37), (37, 36), (36, 35), (35, 34),
    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47),
    (19, 34), (20, 35), (21, 36), (22, 37), (23, 38),
    (28, 43), (29, 44), (30, 45), (31, 46), (32, 47),
    (0, 18), (0, 19), (1, 20), (1, 21), (2, 21), (2, 22), (3, 22), (4, 23),
    (17, 33), (17, 32), (17, 31), (16, 31), (16, 30), (15, 30), (15, 29), (14, 29), (14, 28), (13, 29), (13, 28),
    (11, 24), (12, 24), (6, 24), (5, 24), (4, 24)
]

layer_pairs_dict = dict([(ii, layer_pair)
                         for ii, layer_pair in enumerate(layer_pairs)])
pairs_layer_dict = dict([(layer_pair, ii)
                         for ii, layer_pair in enumerate(layer_pairs)])


def select_pair_layers(layers):
    return [ii for ii, layer_pair in enumerate(layer_pairs)
            if layer_pair[0] in layers and layer_pair[1] in layers
           ]


import time

def read_log(file_name):
    time_format = '%d %b %Y %H:%M:%S'
    get2nd = lambda x: x.split()[1]

    time_info = []
    data_info = []
    itime = -1
    with open(file_name) as f:
        for line in f:
            if line[0] != '#':
                tt = time.strptime(line[:-1], time_format)
                time_info.append(tt)
                data_info.append([])
                itime += 1
            else:
                items = line.split(',')
                try:
                    iteration = int(get2nd(items[0]))
                except ValueError:
                    continue
                time_consumption = float(get2nd(items[1]))
                loss_train = float(get2nd(items[2]))
                loss_test  = float(get2nd(items[3]))
                precision  = float(get2nd(items[4]))
                recall     = float(get2nd(items[5]))
                data_info[itime].append([iteration, time_consumption, loss_train,
                                      loss_test, precision, recall])
    return data_info, time_info


def plot_log(info, name, axs=None):
    fontsize = 16
    minor_size = 14
    if type(info) is not 'numpy.ndarray':
        info = np.array(info)
    df = pd.DataFrame(info, columns=['iteration', 'time', 'loss_train', 'loss_test', 'precision', 'recall'])

    # make plots
    if axs is None:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        axs = axs.flatten()

    y_labels = ['Time [s]', 'Training Loss', 'Precision', 'Recall']
    y_data   = ['time', 'loss_train', 'precision', 'recall']
    x_label = 'Iterations'
    x_data = 'iteration'
    for ib, values in enumerate(zip(y_data, y_labels)):
        ax = axs[ib]

        if 'loss_train' == values[0]:
            df.plot(x=x_data, y=values[0], ax=ax, label='Training')
            df.plot(x=x_data, y='loss_test', ax=ax, label='Testing')
            ax.set_ylabel("Losses", fontsize=fontsize)
            ax.legend(fontsize=fontsize)
        else:
            df.plot(x=x_data, y=values[0], ax=ax)
            ax.set_ylabel(values[1], fontsize=fontsize)

        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)

    return axs

def select_hits(event, no_noise, eta_cut=1.2):
    if no_noise:
        hits = event.hits[event.hits.particle_id != 0]
    else:
        hits = event.hits

    hits = hits[hits.layer.isin([7, 8, 9])]
    hits = hits[np.abs(hits.eta) < eta_cut]
    # a = hits.groupby('particle_id')['hit_idx'].count() > 2
    # n_particles = a[a].shape[0]
    return hits
