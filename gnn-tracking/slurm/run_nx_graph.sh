#!/bin/bash
#SBATCH -C gpu
#SBATCH --gres gpu:1
#SBATCH -c 2
#SBATCH -t 2:00:00
#SBATCH -A atlas

if [ $# -lt 1 ]; then
	echo "provide configuration file please"
	exit 1
fi

CONFIG=$1

echo $CONFIG
which python
cd /global/homes/x/xju/track/gnn/code/gnn_networkx

srun python train_nx_graph.py $CONFIG
