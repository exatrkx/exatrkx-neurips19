#!/bin/bash
#SBATCH --qos=shared
#SBATCH --constraint=haswell
#SBATCH --time=2:00:00
#SBATCH --ntasks=1

which python
cd /global/homes/x/xju/track/gnn/code/gnn_networkx

srun python hits_graph_to_tuple.py configs/nxgraph_test_twolayers.yaml
