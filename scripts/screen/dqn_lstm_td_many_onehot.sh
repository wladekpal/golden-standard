#!/bin/bash

GPU_ID=$1



exclude_dirs=( ".github" ".ruff_cache" "wandb" ".vscode" ".idea" "__pycache__" ".venv" "experiments" ".git" "notebooks" "runs" "notes" ".pytest")

# Experiment name
exp_name="dqn_lstm_onehot_test"

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


number_of_boxes=6
min_number_of_boxes=6
grid_size=6
lstm_hidden_size=1024

# Option: With one-hot encoding + thinking (aggregate first, then thinking steps)
# This uses one-hot encoder (gc_encoder=OneHotEncoder), aggregates spatial positions, then thinking steps
for thinking_steps_test in 1
do
    for seed in 1 2
    do
        for number_of_moving_boxes_max in 1
        do
            CUDA_VISIBLE_DEVICES=$GPU_ID uv run --active src/train.py \
                env:box-moving \
                --agent.agent_name gcdqn_lstm \
                --exp.name g${grid_size}_aggregate_b${number_of_boxes}_bs1024_lr1e-4_UTD500_onehot_hidden${lstm_hidden_size}_m${number_of_moving_boxes_max}_t${thinking_steps_test} \
                --env.number_of_boxes_max ${number_of_boxes} \
                --env.number_of_boxes_min ${min_number_of_boxes} \
                --env.number_of_moving_boxes_max ${number_of_moving_boxes_max} \
                --env.grid_size ${grid_size} \
                --exp.gamma 0.99 \
                --env.episode_length 100 \
                --exp.seed ${seed} \
                --exp.project "dqn_lstm_aggregate_test" \
                --exp.epochs 50 \
                --exp.gif_every 10 \
                --agent.alpha 0.1 \
                --exp.max_replay_size 10000 \
                --exp.batch_size 1024 \
                --exp.updates_per_rollout 500 \
                --agent.lr 1e-4 \
                --exp.use_future_and_random_goals \
                --exp.eval_different_box_numbers \
                --agent.thinking_steps ${thinking_steps_test} \
                --agent.lstm_hidden_size ${lstm_hidden_size} \
                --agent.encoder onehot
        done
    done
done
