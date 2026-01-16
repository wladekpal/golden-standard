#!/usr/bin/env bash

set -euo pipefail

GPU_IDS_CSV="${1:-0}"
grid_size="${2:?Usage: $0 <gpu_ids_csv> <grid_size> <thinking_steps>}"
thinking_steps="${3:?Usage: $0 <gpu_ids_csv> <grid_size> <thinking_steps>}"

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="$ROOT_DIR/.venv"

exclude_dirs=( ".github" ".ruff_cache" "wandb" ".vscode" ".idea" "__pycache__" ".venv" "experiments" ".git" "notebooks" "runs" "notes" ".pytest" )

exclude_opts=()
for dir in "${exclude_dirs[@]}"; do
  exclude_opts+=("--exclude=${dir}")
done

# Experiment name
exp_name="stich_dqn_td_grid_${grid_size}_thinking_${thinking_steps}"

# Create the main experiments directory if it doesn't exist
EXPERIMENTS_DIR="$ROOT_DIR/experiments"
mkdir -p "$EXPERIMENTS_DIR"

timestamp=$(date +"%Y%m%d_%H%M%S")

SEEDS=(1 2 3)
MOVING_BOXES_MAX_VALUES=(4 3 2 1)

# Create an array of all job combinations
JOBS=()
for seed in "${SEEDS[@]}"; do
  for number_of_moving_boxes_max in "${MOVING_BOXES_MAX_VALUES[@]}"; do
    JOBS+=("$seed:$number_of_moving_boxes_max")
  done
done

run_job() {
  local gpu_id="$1"
  local seed="$2"
  local number_of_moving_boxes_max="$3"

  local job_id="seed${seed}_mov${number_of_moving_boxes_max}_gpu${gpu_id}"
  local temp_dir
  temp_dir="$(mktemp -d "$EXPERIMENTS_DIR/${exp_name}_${timestamp}_${job_id}_XXXX")"

  echo "Creating temp dir: $temp_dir"
  rsync -a "${exclude_opts[@]}" "$ROOT_DIR/" "$temp_dir/"

  (
    # Use the .venv from the original root directory, but do NOT copy it into the temp dir.
    if [ -d "$VENV_PATH" ]; then
      export VIRTUAL_ENV="$VENV_PATH"
      export PATH="$VENV_PATH/bin:$PATH"
      # shellcheck disable=SC1090
      source "$VENV_PATH/bin/activate"
    else
      echo "Warning: .venv not found in project root ($VENV_PATH)."
    fi

    cd "$temp_dir"
    echo "Starting job $job_id in '$(pwd)'"

    CUDA_VISIBLE_DEVICES="$gpu_id" uv run --active src/train.py \
      env:box-moving \
      --agent.agent_name gcdqn_lstm \
      --exp.name single_l_dqn_lstm_stable \
      --env.number_of_boxes_max "$number_of_moving_boxes_max" \
      --env.number_of_boxes_min "$number_of_moving_boxes_max" \
      --env.number_of_moving_boxes_max "$number_of_moving_boxes_max" \
      --env.grid_size "$grid_size" \
      --exp.gamma 0.99 \
      --env.episode_length 100 \
      --exp.seed "$seed" \
      --exp.project "dqn_lstm_exact" \
      --exp.epochs 50 \
      --exp.gif_every 10 \
      --agent.alpha 0.1 \
      --exp.max_replay_size 10000 \
      --exp.batch_size 256 \
      --exp.use_future_and_random_goals \
      --exp.eval_special \
      --agent.thinking_steps "$thinking_steps" \
      --env.level_generator variable 
  )
}

echo "Starting ${#JOBS[@]} jobs across GPUs: ${GPU_IDS[*]}"

GPU_INDEX=0
for JOB in "${JOBS[@]}"; do
  IFS=':' read -r seed number_of_moving_boxes_max <<< "$JOB"
  gpu_id="${GPU_IDS[$GPU_INDEX]}"

  run_job "$gpu_id" "$seed" "$number_of_moving_boxes_max" &

  GPU_INDEX=$(( (GPU_INDEX + 1) % ${#GPU_IDS[@]} ))
  if [ "$GPU_INDEX" -eq 0 ]; then
    wait
  fi
done

echo "Waiting for all instances to finish..."
wait
echo "All parallel instances have completed."