# Layer Information
__all__ = ['layer_pairs', 'load_yaml', 'select_pair_layers']

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

layer_pairs_dict = dict([(ii, layer_pair) for ii, layer_pair in enumerate(layer_pairs)])
pairs_layer_dict = dict([(layer_pair, ii) for ii, layer_pair in enumerate(layer_pairs)])

import numpy as np
def keep_finite(df):
    bad_list = []
    for column in df.columns:
        if not np.all(np.isfinite(df[column])):
            ss = df[column]
            bad_list += ss.loc[~np.isfinite(ss)].index.values.tolist()

    bad_list = list(set(bad_list))
    return df.drop(bad_list)


def select_pair_layers(layers):
    return [ii for ii, layer_pair in enumerate(layer_pairs) if layer_pair[0] in layers and layer_pair[1] in layers]

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

import yaml
import os
def load_yaml(file_name):
    assert(os.path.exists(file_name))
    with open(file_name) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

import itertools
def pairwise(iterable):
  """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
  a, b = itertools.tee(iterable)
  next(b, None)
  return zip(a, b)
