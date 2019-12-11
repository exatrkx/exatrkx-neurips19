from __future__ import absolute_import

from heptrkx.postprocess import wrangler
from heptrkx.postprocess import analysis

import numpy as np

def get_corrected_trks(nx_G, truth, hits_in_question,
                       feature_name='solution'
                      ):
    all_true_tracks = wrangler.get_tracks(nx_G, feature_name=feature_name, with_fit=False)
    true_df = analysis.graphs_to_df(all_true_tracks)
    n_total_predictions = len(np.unique(true_df['track_id']))
    print("total predictions:", n_total_predictions)
    res_truth = analysis.summary_on_prediction2(nx_G, hits_in_question, true_df)
    return res_truth


def print_info(res_pred):
    print("Correct {}, "
          "Wrong {}, ".format(
              res_pred['n_correct'],
              res_pred['n_wrong']
          ))
