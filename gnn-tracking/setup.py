from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from setuptools import find_packages
from setuptools import setup

description="Use Graph Network to reconstruct tracks"

setup(
    name="exatrkx",
    version="0.0.1",
    description="Library for building tracks with Graph Nural Networks.",
    long_description=description,
    author="Xiangyang Ju, Exa.Trkx",
    license="Apache License, Version 2.0",
    keywords=["graph networks", "track formation", "tracking", "machine learning"],
    url="https://github.com/exatrkx/exatrkx-work",
    packages=find_packages(),
    install_requires=[
        "graph_nets==1.0.5",
        'tensorflow-gpu<2',
        'gast==0.2.2',
        "future",
        "networkx==2.3",
        "numpy",
        "scipy",
        "pandas",
        "setuptools",
        "six",
        "matplotlib",
        "torch",
        "torchvision",
        'sklearn',
        'pyyaml>=5.1',
        'trackml@ https://github.com/LAL/trackml-library/tarball/master#egg=trackml-3',
        'tables',
        'h5py',
    ],
    setup_requires=['trackml'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    scripts=[
        'scripts/make_true_pairs_for_training_segments_mpi',
        'scripts/make_true_pairs',
        'scripts/merge_true_pairs',
        'scripts/make_pairs_for_training_segments',
        'scripts/select_pairs',
        'scripts/tf_train_pairs',
        'scripts/tf_train_pairs_all',
        'scripts/train_nx_graph',
        'scripts/evaluate_gnn_models',
        'scripts/evaluate_event',
        'scripts/hits_graph_to_tuple',
        'scripts/make_doublets_from_NNs',
        'scripts/make_doublets_from_cuts',
        'scripts/pairs_to_nx_graph',
        'scripts/get_total_segments',
        'scripts/make_graph',
        'scripts/plot_graph',
        'scripts/make_trkx',
        'scripts/duplicated_hits',
        'scripts/segment_eff_purity',
        'scripts/track_eff_purity',
        'scripts/create_seed_inputs',
        'scripts/eval_doublet_NN',
        'scripts/prepare_hitsgraph',
        'scripts/merge_true_fake_pairs',
        'scripts/seeding_eff_purity_comparison',
        'scripts/fit_hits',
        'scripts/peek_models',
        'scripts/train_infomax',
        'scripts/view_training_log',
        'scripts/acts_seeding_eff',
    ],
)
