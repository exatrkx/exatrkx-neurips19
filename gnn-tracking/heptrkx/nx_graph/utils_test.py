"""
utils for testing trained tf model
"""

import tensorflow as tf
from graph_nets import utils_tf

import sklearn.metrics
import os
import numpy as np

from . import utils_train, utils_io, prepare, get_model
from .utils_io import ckpt_name
from .. import load_yaml

import matplotlib.pyplot as plt

def create_trained_model(config_name, input_ckpt=None):
    """
    @config: configuration for train_nx_graph
    """
    # load configuration file
    config = load_yaml(config_name)
    config_tr = config['train']

    log_every_seconds       = config_tr['time_lapse']
    batch_size = n_graphs   = config_tr['batch_size']   # need optimization
    num_processing_steps_tr = config_tr['n_iters']      ## level of message-passing
    prod_name = config['prod_name']
    if input_ckpt is None:
        input_ckpt = os.path.join(config['output_dir'], prod_name)


    # generate inputs
    generate_input_target = prepare.inputs_generator(config['data']['output_nxgraph_dir'], n_train_fraction=0.8)

    # build TF graph
    tf.reset_default_graph()
    model = get_model(config['model']['name'])

    input_graphs, target_graphs = generate_input_target(n_graphs)
    input_ph  = utils_tf.placeholders_from_data_dicts(input_graphs, force_dynamic_num_graphs=True)
    target_ph = utils_tf.placeholders_from_data_dicts(target_graphs, force_dynamic_num_graphs=True)

    output_ops_tr = model(input_ph, num_processing_steps_tr)

    def evaluator(iteration, n_test_graphs=10):
        try:
            sess.close()
        except NameError:
            pass

        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(input_ckpt, ckpt_name.format(iteration)))
        odds = []
        tdds = []
        for _ in range(n_test_graphs):
            feed_dict = utils_train.create_feed_dict(generate_input_target, batch_size, input_ph, target_ph, is_trained=False)
            predictions = sess.run({
                "outputs": output_ops_tr,
                'target': target_ph
            }, feed_dict=feed_dict)
            output = predictions['outputs'][-1]
            target = predictions['target']
            odd, tdd = utils_train.eval_output(target, output)
            odds.append(odd)
            tdds.append(tdd)
        return np.concatenate(odds), np.concatenate(tdds)

    return evaluator
