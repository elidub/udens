#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH -N 1
#SBATCH -p gpu_titanrtx_shared
#SBATCH --gpus-per-node=titanrtx:1

python SWYFT.py \
    --config-name config_blobs.yaml \
#     ++estimation.network.n_msc=12
