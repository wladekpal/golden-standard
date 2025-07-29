#!/bin/bash

GPU_ID=$1
grid_size=$2

number_of_boxes_min=3
number_of_boxes_max=7

exclude_dirs=( ".github" ".ruff_cache" "wandb" ".vscode" ".idea" "__pycache__" ".venv" "experiments")

# Experiment name
exp_name="test_generalization"

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

for seed in 1 2 3 
do
    for moving_boxes_max in 3 5 7
    do
        CUDA_VISIBLE_DEVICES=$GPU_ID uv run --active src/train.py \
        env:box-pushing \
        --agent.agent_name gciql \
        --agent.discrete \
        --agent.actor_loss awr \
        --env.dense_rewards \
        --agent.alpha 0.3 \
        --agent.actor-hidden-dims 512 512 512 --agent.value-hidden-dims 512 512 512 \
        --exp.name moving_boxes_${moving_boxes_max}_grid_${grid_size}_range_${number_of_boxes_min}_${number_of_boxes_max} \
        --env.number_of_boxes_max ${number_of_boxes_max} \
        --env.number_of_boxes_min ${number_of_boxes_min} \
        --env.number_of_moving_boxes_max ${moving_boxes_max} \
        --env.grid_size ${grid_size} \
        --exp.gamma 0.99 \
        --env.episode_length 100 \
        --exp.seed $seed \
        --exp.project "generalization_proper_iql" \
        --exp.epochs 20 
    done
done