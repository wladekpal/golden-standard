#!/bin/bash

GPU_ID=$1
grid_size=$2


exclude_dirs=( ".github" ".ruff_cache" "wandb" ".vscode" ".idea" "__pycache__" ".venv" "experiments" ".git" "notebooks" "runs" "notes" ".pytest")

# Experiment name
exp_name="many_boxes_crl_mc_grid_${grid_size}"

# Create the main experiments directory if it doesn't exist
mkdir -p ./experiments

# Create a temporary directory with the experiment name within ./experiments
# TODO: tmp dir should be created for every seed different configuration?
timestamp=$(date +"%Y%m%d_%H%M%S")
temp_dir="./experiments/${exp_name}_${timestamp}"
mkdir -p "$temp_dir"


# Create the rsync exclude options
exclude_opts=""
for dir in "${exclude_dirs[@]}"; do
  exclude_opts+="--exclude=${dir} "
done

# Copy all necessary files to the temporary directory, excluding specified directories
eval rsync -av $exclude_opts ./ "$temp_dir"

# Use the .venv from the original root directory, but do NOT copy it into the temp dir
VENV_PATH="$(cd "$(dirname "$0")/../.." && pwd)/.venv"
echo "VENV_PATH: $VENV_PATH"
if [ -d "$VENV_PATH" ]; then
  echo "VENV_PATH found"
  export VIRTUAL_ENV="$VENV_PATH"
  export PATH="$VENV_PATH/bin:$PATH"
  source "$VENV_PATH/bin/activate"
else
  echo "Warning: .venv not found in project root. Please ensure your virtual environment is set up."
fi

# Change to the temporary directory
cd "$temp_dir"
echo "Current path: '$(pwd)'"


echo "Running with grid_size: $grid_size, number_of_boxes_min: $number_of_boxes_min, number_of_boxes_max: $number_of_boxes_max"


for seed in 1 2
do
    for number_of_boxes in 2 3 4
    do
      for episode_length in 100 
      do
        CUDA_VISIBLE_DEVICES=$GPU_ID uv run --active src/train.py \
        env:box-pushing \
        --agent.agent_name crl_search \
        --exp.name norm_collection_evaluation_crl_softmax_all_moveable_no_target_entropy_mc_boxes_${number_of_boxes}_grid_${grid_size}_ep_len_${episode_length}_no_filtering \
        --env.number_of_boxes_max ${number_of_boxes} \
        --env.number_of_boxes_min ${number_of_boxes} \
        --env.number_of_moving_boxes_max ${number_of_boxes} \
        --env.grid_size ${grid_size} \
        --exp.gamma 0.99 \
        --env.episode_length ${episode_length} \
        --exp.seed ${seed} \
        --exp.project "crl_infinite_horizon" \
        --exp.epochs 50 \
        --exp.gif_every 10 \
        --agent.alpha 0.1 \
        --agent.expectile 0.5  \
        --exp.max_replay_size 10000 \
        --exp.batch_size 256 \
        --exp.eval_special \
        --env.level_generator quarter 
        done
    done
done