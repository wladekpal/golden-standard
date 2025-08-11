#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --account=plgcrlreason-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=experiment_out.txt

# Uncomment for first execution - uv has to be run on gpu node, NOT in entry node
# curl -LsSf https://astral.sh/uv/install.sh | sh

ml ML-bundle/24.06a

unset LD_LIBRARY_PATH

uv run src/train.py env:box-pushing \
    --env.grid-size 5  \
    --env.level-generator default \
    --exp.name helios_test\
    --exp.project helios_test \
    --exp.entity cl-probing \
    --exp.seed 0 \
    --exp.num_envs 2048 \
    --exp.gif_every 10 \
    --exp.epochs 25