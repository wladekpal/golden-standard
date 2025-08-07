#!/bin/bash

GPU_ID=$1
grid_size=$2

number_of_boxes_min=3
number_of_boxes_max=7

exclude_dirs=( ".github" ".ruff_cache" "wandb" ".vscode" ".idea" "__pycache__" ".venv" "experiments" ".git" "notebooks")

# Experiment name
exp_name="test_generalization_sc"

# Create the main experiments directory if it doesn't exist
mkdir -p ./experiments

# Create a temporary directory with the experiment name within ./experiments
# TODO: tmp dir should be created for every seed different configuration?
temp_dir=$(mktemp -d "./experiments/${exp_name}_XXXXXX")

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

moving_boxes_max=5

for seed in 1 2 3
do
    for target_entropy in -1.1 -0.69 -1.38 # ln(1/3) and ln(1/2) and ln(1/4)
    do
        CUDA_VISIBLE_DEVICES=$GPU_ID uv run --active src/train.py \
        env:box-pushing \
        --agent.agent_name crl_search \
        --exp.name q_not_norm_mse_temp_tuning_${moving_boxes_max}_grid_${grid_size}_range_${number_of_boxes_min}_${number_of_boxes_max}_target_entropy_${target_entropy} \
        --env.number_of_boxes_max ${number_of_boxes_max} \
        --env.number_of_boxes_min ${number_of_boxes_min} \
        --env.number_of_moving_boxes_max ${moving_boxes_max} \
        --env.grid_size ${grid_size} \
        --exp.gamma 0.99 \
        --env.episode_length 100 \
        --exp.seed $seed \
        --exp.project "alpha_tuning" \
        --exp.epochs 50 \
        --agent.alpha 0.1 \
        --agent.target_entropy ${target_entropy} 
    done
done