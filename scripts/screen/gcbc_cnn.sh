#!/usr/bin/env bash

set -euo pipefail

GPU_IDS_CSV="${1:-0}"
use_discounted_mc_rewards="${2:?Usage: $0 <gpu_ids_csv> <use_discounted_mc_rewards:true|false>}"

case "${use_discounted_mc_rewards,,}" in
  true|1|yes|y)
    use_discounted_mc_rewards=true
    ;;
  false|0|no|n)
    use_discounted_mc_rewards=false
    ;;
  *)
    echo "Error: use_discounted_mc_rewards must be one of: true/false, 1/0, yes/no"
    exit 1
    ;;
esac

GRID_SIZES=(8 10 12)
NUMBER_OF_BOXES_VALUES=(6 8 10 12)
SEEDS=(1 2)
MOVING_BOXES_MAX_VALUES=(1 2)
LEARNING_RATES=(0.0003)
DISCOUNTS=(0.99)
BATCH_SIZES=(256)

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="$ROOT_DIR/.venv"

exclude_dirs=( ".github" ".ruff_cache" "wandb" ".vscode" ".idea" "__pycache__" ".venv" "experiments" ".git" "notebooks" "runs" "notes" ".pytest" )

exclude_opts=()
for dir in "${exclude_dirs[@]}"; do
  exclude_opts+=("--exclude=${dir}")
done

# Experiment name
exp_name="gcbc_cnn"

# Create the main experiments directory if it doesn't exist
EXPERIMENTS_DIR="$ROOT_DIR/experiments"
mkdir -p "$EXPERIMENTS_DIR"

timestamp=$(date +"%Y%m%d_%H%M%S")

# Create an array of all job combinations
JOBS=()
for grid_size in "${GRID_SIZES[@]}"; do
  for number_of_boxes in "${NUMBER_OF_BOXES_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for number_of_moving_boxes_max in "${MOVING_BOXES_MAX_VALUES[@]}"; do
        for learning_rate in "${LEARNING_RATES[@]}"; do
          for discount in "${DISCOUNTS[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
              JOBS+=("$grid_size:$number_of_boxes:$seed:$number_of_moving_boxes_max:$learning_rate:$discount:$batch_size")
            done
          done
        done
      done
    done
  done
done

run_job() {
  local gpu_id="$1"
  local grid_size="$2"
  local number_of_boxes="$3"
  local seed="$4"
  local number_of_moving_boxes_max="$5"
  local learning_rate="$6"
  local discount="$7"
  local batch_size="$8"

  local job_id="grid${grid_size}_boxes${number_of_boxes}_seed${seed}_mov${number_of_moving_boxes_max}_lr${learning_rate}_disc${discount}_bs${batch_size}_gpu${gpu_id}"
  local run_exp_name="gcbc_cnn_grid_${grid_size}_boxes${number_of_boxes}_mov${number_of_moving_boxes_max}_lr${learning_rate}_disc${discount}_bs${batch_size}"
  local discounted_mc_flag=()

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
      --agent.agent_name gcbc_cnn \
      --exp.name "$run_exp_name" \
      --env.number_of_boxes_max "$number_of_boxes" \
      --env.number_of_boxes_min "$number_of_boxes" \
      --env.number_of_moving_boxes_max "$number_of_moving_boxes_max" \
      --env.grid_size "$grid_size" \
      --agent.lr "$learning_rate" \
      --agent.discount "$discount" \
      --env.episode_length 100 \
      --exp.seed "$seed" \
      --exp.project "gcbc_cnn_generalized" \
      --exp.epochs 100 \
      --exp.gif_every 10 \
      --agent.alpha 0.1 \
      --exp.max_replay_size 10000 \
      --agent.batch_size "$batch_size" \
      --exp.num_envs 256 \
      --exp.input_representation "factored_flat" \
      --exp.eval-different-box-numbers
      # --env.level_generator "variable" \
      # --exp.eval_special
  )
}

echo "Starting ${#JOBS[@]} jobs across GPUs: ${GPU_IDS[*]}"

MAX_JOBS_PER_GPU=1

declare -a GPU_PID_LISTS=()
for i in "${!GPU_IDS[@]}"; do
  GPU_PID_LISTS[$i]=""
done

refresh_gpu_jobs() {
  local gpu_index="$1"
  local pid
  local active_pids=()

  for pid in ${GPU_PID_LISTS[$gpu_index]}; do
    if kill -0 "$pid" 2>/dev/null; then
      active_pids+=("$pid")
    else
      wait "$pid" || true
      echo "[scheduler] GPU ${GPU_IDS[$gpu_index]} job finished"
    fi
  done

  GPU_PID_LISTS[$gpu_index]="${active_pids[*]}"
}

gpu_job_count() {
  local gpu_index="$1"
  if [ -z "${GPU_PID_LISTS[$gpu_index]}" ]; then
    echo 0
    return
  fi

  local pids_array=()
  read -r -a pids_array <<< "${GPU_PID_LISTS[$gpu_index]}"
  echo "${#pids_array[@]}"
}

launch_job_on_gpu() {
  local gpu_index="$1"
  local grid_size="$2"
  local number_of_boxes="$3"
  local seed="$4"
  local number_of_moving_boxes_max="$5"
  local learning_rate="$6"
  local discount="$7"
  local batch_size="$8"

  local current_jobs
  current_jobs="$(gpu_job_count "$gpu_index")"

  echo "[scheduler] Launching job ${job_index}/${#JOBS[@]} on GPU ${GPU_IDS[$gpu_index]} (grid=${grid_size}, boxes=${number_of_boxes}, seed=${seed}, mov=${number_of_moving_boxes_max}, lr=${learning_rate}, discount=${discount}, bs=${batch_size}, slot=$((current_jobs + 1))/${MAX_JOBS_PER_GPU})"
  run_job "${GPU_IDS[$gpu_index]}" "$grid_size" "$number_of_boxes" "$seed" "$number_of_moving_boxes_max" "$learning_rate" "$discount" "$batch_size" &
  GPU_PID_LISTS[$gpu_index]="${GPU_PID_LISTS[$gpu_index]} $!"
  GPU_PID_LISTS[$gpu_index]="${GPU_PID_LISTS[$gpu_index]# }"
}

job_index=0
while [ "$job_index" -lt "${#JOBS[@]}" ]; do
  free_gpu_index=""
  for i in "${!GPU_IDS[@]}"; do
    refresh_gpu_jobs "$i"
    current_jobs="$(gpu_job_count "$i")"

    if [ "$current_jobs" -lt "$MAX_JOBS_PER_GPU" ]; then
      echo "[scheduler] GPU ${GPU_IDS[$i]} has capacity (${current_jobs}/${MAX_JOBS_PER_GPU})"
      free_gpu_index="$i"
      break
    fi
  done

  if [ -z "$free_gpu_index" ]; then
    echo "[scheduler] All GPUs at capacity (${MAX_JOBS_PER_GPU} jobs/GPU); waiting for next job to finish"
    wait -n
    continue
  fi

  IFS=':' read -r grid_size number_of_boxes seed number_of_moving_boxes_max learning_rate discount batch_size <<< "${JOBS[$job_index]}"
  launch_job_on_gpu "$free_gpu_index" "$grid_size" "$number_of_boxes" "$seed" "$number_of_moving_boxes_max" "$learning_rate" "$discount" "$batch_size"
  job_index=$((job_index + 1))
done

echo "Waiting for all instances to finish..."
for i in "${!GPU_IDS[@]}"; do
  for pid in ${GPU_PID_LISTS[$i]}; do
    wait "$pid"
  done
done
echo "All parallel instances have completed."