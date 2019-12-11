# Example environment setup script for Cori
export OMP_NUM_THREADS=32
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
module load pytorch/v1.0.0-intel
export PYTHONPATH=$PYTHONPATH:/global/homes/x/xju/track/gnn/code/python_pkg/install/lib/python3.7/site-packages
