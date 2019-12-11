#!/bin/bash
#SBATCH --qos=shared
#SBATCH --constraint=haswell
#SBATCH --time 10:00:00
#SBATCH -c 10
#SBATCH -A m3253


which python
cd /global/homes/x/xju/track/gnn/code/gnn_networkx

srun python pairs_to_nx_graph.py

