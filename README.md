# exatrkx-neurips19
Code and instructions to reproduce the results of our [NeurIPS 2019 paper](https://ml4physicalsciences.github.io/files/NeurIPS_ML4PS_2019_83.pdf).

## Setting up the exatrkx runtime environment with conda
These instructions assume you are running [miniconda](https://docs.conda.io/en/latest/miniconda.html) on a linux system.
1. Create an exatrkx conda environment
```bash
source [path_to_miniconda]/bin/activate
conda create --name exatrkx python=3.7
conda activate exatrkx
```
2. nb_conda_kernels allows to pick the exatrkx conda kernel from a traditional jupyter notebook. To install it (in your base environment or in the exatrkx environment)
```
conda install -c "conda-forge/label/cf201901" nb_conda_kernels
```
## Running the Exa.TrkX models

Refer to the intructions in the README.md of each subdirectory
