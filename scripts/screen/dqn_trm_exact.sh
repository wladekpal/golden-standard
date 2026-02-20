#!/usr/bin/env bash

set -euo pipefail

GPU_IDS_CSV="${1:-0}"
grid_size="${2:?Usage: $0 <gpu_ids_csv> <grid_size>}"

SEEDS=(1 2 3 4 5)
MOVING_BOXES_MAX_VALUES=(4 3)
thinking_steps=(2 5 10)

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="$ROOT_DIR/.venv"

exclude_dirs=( ".github" ".ruff_cache" "wandb" ".vscode" ".idea" "__pycache__" ".venv" "experiments" ".git" "notebooks" "runs" "notes" ".pytest" )

exclude_opts=()
for dir in "${exclude_dirs[@]}"; do
  exclude_opts+=("--exclude=${dir}")
done

# Experiment name
exp_name="exact_dqn_trm_grid_${grid_size}"

# Create the main experiments directory if it doesn't exist
EXPERIMENTS_DIR="$ROOT_DIR/experiments"
mkdir -p "$EXPERIMENTS_DIR"

timestamp=$(date +"%Y%m%d_%H%M%S")

# Create an array of all job combinations
JOBS=()
for seed in "${SEEDS[@]}"; do
  for number_of_moving_boxes_max in "${MOVING_BOXES_MAX_VALUES[@]}"; do
    for thinking_step in "${thinking_steps[@]}"; do
      JOBS+=("$seed:$number_of_moving_boxes_max:$thinking_step")
    done
  done
done

run_job() {
  local gpu_id="$1"
  local seed="$2"
  local number_of_moving_boxes_max="$3"
  local thinking_step="$4"

  local job_id="seed${seed}_mov${number_of_moving_boxes_max}_think${thinking_step}_gpu${gpu_id}"
  local temp_dir
  temp_dir="$(mktemp -d "$EXPERIMENTS_DIR/${exp_name}_${grid_size}_${thinking_step}_${timestamp}_${job_id}_XXXX")"

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
      --agent.agent_name gcdqn_trm \
      --exp.name dqn_trm_think_${thinking_step}_ts_1_layer_faster \
      --env.number_of_boxes_max "$number_of_moving_boxes_max" \
      --env.number_of_boxes_min "$number_of_moving_boxes_max" \
      --env.number_of_moving_boxes_max "$number_of_moving_boxes_max" \
      --env.grid_size "$grid_size" \
      --exp.gamma 0.99 \
      --env.episode_length 100 \
      --exp.seed "$seed" \
      --exp.project "dqn_trm_exact" \
      --exp.epochs 50 \
      --exp.gif_every 10 \
      --agent.alpha 0.1 \
      --exp.max_replay_size 10000 \
      --exp.batch_size 256 \
      --exp.use_future_and_random_goals \
      --exp.eval_special \
      --agent.thinking_steps "$thinking_step" \
      --env.level_generator variable
  )
}

echo "Starting ${#JOBS[@]} jobs across GPUs: ${GPU_IDS[*]}"

declare -a GPU_PIDS=()
for i in "${!GPU_IDS[@]}"; do
  GPU_PIDS[$i]=""
done

launch_job_on_gpu() {
  local gpu_index="$1"
  local seed="$2"
  local number_of_moving_boxes_max="$3"
  local thinking_step="$4"

  echo "[scheduler] Launching job ${job_index}/${#JOBS[@]} on GPU ${GPU_IDS[$gpu_index]} (seed=${seed}, mov=${number_of_moving_boxes_max}, think=${thinking_step})"
  run_job "${GPU_IDS[$gpu_index]}" "$seed" "$number_of_moving_boxes_max" "$thinking_step" &
  GPU_PIDS[$gpu_index]="$!"
}

job_index=0
while [ "$job_index" -lt "${#JOBS[@]}" ]; do
  free_gpu_index=""
  for i in "${!GPU_IDS[@]}"; do
    pid="${GPU_PIDS[$i]}"
    if [ -z "$pid" ]; then
      echo "[scheduler] GPU ${GPU_IDS[$i]} is free"
      free_gpu_index="$i"
      break
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || true
      GPU_PIDS[$i]=""
      echo "[scheduler] GPU ${GPU_IDS[$i]} job finished"
      free_gpu_index="$i"
      break
    fi
  done

  if [ -z "$free_gpu_index" ]; then
    echo "[scheduler] All GPUs busy; waiting for next job to finish"
    wait -n
    continue
  fi

  IFS=':' read -r seed number_of_moving_boxes_max thinking_step <<< "${JOBS[$job_index]}"
  launch_job_on_gpu "$free_gpu_index" "$seed" "$number_of_moving_boxes_max" "$thinking_step"
  job_index=$((job_index + 1))
done

echo "Waiting for all instances to finish..."
for pid in "${GPU_PIDS[@]}"; do
  if [ -n "$pid" ]; then
    wait "$pid"
  fi
done
echo "All parallel instances have completed."