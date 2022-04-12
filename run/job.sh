#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH -N 1
#SBATCH -p gpu_titanrtx_shared
#SBATCH --gpus-per-node=titanrtx:1

python SWYFT.py \
    --config-name config_uniform_norm.yaml \
#     ++estimation.network.n_msc=12
