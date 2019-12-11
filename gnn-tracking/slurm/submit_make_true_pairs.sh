#!/bin/bash
#SBATCH -J true-pairs 
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -A m3253
#SBATCH --nodes 64

set_root_py3

which python 
srun -n 64 -c 32 python make_true_pairs_for_training_segments_mpi.py configs/data.yaml

