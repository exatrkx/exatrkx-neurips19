"""Take a model configation and evaluate hitsgraph"""

import torch
from ..models import get_model
from ..datasets.graph import load_graph
from ..datasets.graph import collate_fn as hitsgraph_to_input_target

from ..nx_graph import utils_data, utils_io
from .. import load_yaml

import re
import glob
import yaml
import os



def load_model(config, reload_epoch, input_ckpt=None):
    model_config = config['model']
    model_type = model_config.pop('model_type')
    model_config.pop('optimizer', None)
    model_config.pop('learning_rate', None)
    model_config.pop('loss_func', None)
    model = get_model(name=model_type, **model_config)

    # Reload specified model checkpoint
    output_dir = input_ckpt if input_ckpt is not None else os.path.expandvars(config['experiment']['output_dir'])
    checkpoint_file = os.path.join(output_dir, 'checkpoints',
                                   'model_checkpoint_%03i.pth.tar' % reload_epoch)
    model.load_state_dict(torch.load(checkpoint_file, map_location='cpu')['model'])
    return model


def create_evaluator(config_file, reload_epoch, input_ckpt=None):
    """use training configrations to initialize models,
    return a function that could evaluate any event, or event section
    """
    config = load_yaml(config_file)
    model = load_model(config, reload_epoch, input_ckpt).eval()
    hitsgraph_dir = config['data']['input_dir']
    base_dir = os.path.join(hitsgraph_dir, 'event{:09d}_g{:03d}.npz')
    batch_size = config['data']['batch_size']

    def evaluate(graphs, use_digraph=False, bidirection=False):
        """G is hitsgraph,
        return: a list of graphs whose edges have a feature of __predict__
        and a feature of __solution__
        """
        input_graphs, target_graphs = hitsgraph_to_input_target(graphs)
        with torch.no_grad():
            weights = model(input_graphs).flatten().numpy()

        nx_G = utils_data.hitsgraph_to_nx(graphs, use_digraph=use_digraph,
                                          bidirection=bidirection)
        ## update edge features with the new weights
        _, n_edges = G.Ri.shape
        for iedge in range(n_edges):
            in_node_id  = G.Ri[:, iedge].nonzero()[0][0]
            out_node_id = G.Ro[:, iedge].nonzero()[0][0]
            try:
                nx_G.edges[(out_node_id, in_node_id)]['predict'] = [weights[iedge]]
            except KeyError:
                pass

            try:
                nx_G.edges[(in_node_id, out_node_id)]['predict'] = [weights[iedge]]
            except KeyError:
                pass

        return nx_G

    return evaluate
