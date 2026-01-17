#!/bin/bash

# Path to the training script (from the root of the repo)
PYTHON_SCRIPT="src/train.py"

# --- CONFIGURATION ---
THINKING_STEPS=(1 2 4)
SEEDS=(1 2)
MOVING_BOXES_MAX_VALUES=(2 1)
GRID_SIZE=6
NUMBER_OF_BOXES=4

# Available GPUs
GPU_IDS=(0 1 2 3 4 5 6 7)

# --- JOB GENERATION ---
# Create a list of all combinations: ThinkingStep + Seed + MovingBoxes
JOBS=()
for step in "${THINKING_STEPS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for mov_box in "${MOVING_BOXES_MAX_VALUES[@]}"; do
            JOBS+=("$step:$seed:$mov_box")
        done
    done
done

NUM_JOBS=${#JOBS[@]}
NUM_GPUS=${#GPU_IDS[@]}

echo "=================================================="
echo "Total Jobs: $NUM_JOBS"
echo "Available GPUs: $NUM_GPUS"
echo "Job List (Step:Seed:MovBox): ${JOBS[*]}"
echo "=================================================="

# --- EXECUTION LOOP ---
GPU_IDX=0

for job in "${JOBS[@]}"; do
    # Parse the job string
    IFS=':' read -r THINKING_STEP SEED MOV_BOX <<< "$job"
    
    # Assign to the next available GPU
    GPU_ID=${GPU_IDS[$GPU_IDX]}

    (
        # Move to root directory
        cd .. || exit

        echo "[STARTED] GPU: $GPU_ID | Steps: $THINKING_STEP | Seed: $SEED | MovBox: $MOV_BOX"

        CUDA_VISIBLE_DEVICES=$GPU_ID uv run $PYTHON_SCRIPT \
            env:box-moving \
            --agent.agent_name gcdqn_interp \
            --exp.name single_l_dqn_interp_stable \
            --env.number_of_boxes_max "$NUMBER_OF_BOXES" \
            --env.number_of_boxes_min "$NUMBER_OF_BOXES" \
            --env.number_of_moving_boxes_max "$MOV_BOX" \
            --env.grid_size "$GRID_SIZE" \
            --exp.gamma 0.99 \
            --env.episode_length 100 \
            --exp.seed "$SEED" \
            --exp.project "dqn_interp_td_correct_init_generalized" \
            --exp.epochs 50 \
            --exp.gif_every 10 \
            --agent.alpha 0.1 \
            --exp.max_replay_size 10000 \
            --exp.batch_size 256 \
            --exp.use_future_and_random_goals \
            --exp.eval_different_box_numbers \
            --agent.thinking_steps "$THINKING_STEP"
        
        echo "[FINISHED] GPU: $GPU_ID | Steps: $THINKING_STEP | Seed: $SEED"
    ) &

    # Cycle to the next GPU
    GPU_IDX=$(( (GPU_IDX + 1) % NUM_GPUS ))

    # If we have filled all GPUs, wait for this batch to finish before continuing
    # (This is a simple way to prevent overloading GPU 0 with queued jobs)
    if [ "$GPU_IDX" -eq 0 ]; then
        wait
    fi

done

# Wait for any remaining stragglers
wait

echo "All $NUM_JOBS experiments completed."