#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=plgbro4ppo-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm_output/train-%j.out

ml ML-bundle/24.06a
unset LD_LIBRARY_PATH
ml libffi/3.4.4

export UV_PROJECT_ENVIRONMENT="$SCRATCH/golden-standard/.venv"
export UV_CACHE_DIR="$SCRATCH/.cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR="$SCRATCH/wandb"

exclude_dirs=( ".github" ".ruff_cache" "wandb" ".vscode" ".idea" "__pycache__" ".venv" "experiments" ".git" "notebooks" "runs" "notes" ".pytest")

# Go to project directory on SCRATCH
cd "$SCRATCH/golden-standard"
echo "Current path: '$(pwd)'"

echo "Running with grid_size: $grid_size, number_of_boxes_min: $number_of_boxes_min, number_of_boxes_max: $number_of_boxes_max"

number_of_boxes=4
min_number_of_boxes=4
grid_size=6
lstm_hidden_size=1024
thinking_steps_test=3

# Interpolation-thinking critic with Normalize encoder
for number_of_boxes in 6
do
    for seed in 1 2
    do
        for number_of_moving_boxes_max in 1
        do
            uv run src/train.py \
                env:box-moving \
                --agent.agent_name gcdqn_interp \
                --exp.name g${grid_size}_interp_normalize11_b${number_of_boxes}_norm_hidden${lstm_hidden_size}_m${number_of_moving_boxes_max}_t${thinking_steps_test} \
                --env.number_of_boxes_max ${number_of_boxes} \
                --env.number_of_boxes_min ${number_of_boxes} \
                --env.number_of_moving_boxes_max ${number_of_moving_boxes_max} \
                --env.grid_size ${grid_size} \
                --exp.gamma 0.99 \
                --env.episode_length 100 \
                --exp.seed ${seed} \
                --exp.project "dqn_lstm_aggregate_test" \
                --exp.save_dir "$SCRATCH/golden-standard/runs" \
                --exp.epochs 50 \
                --exp.gif_every 10 \
                --agent.alpha 0.1 \
                --exp.max_replay_size 10000 \
                --exp.batch_size 256 \
                --exp.updates_per_rollout 1000 \
                --exp.use_future_and_random_goals \
                --exp.eval_different_box_numbers \
                --agent.thinking_steps ${thinking_steps_test} \
                --agent.lstm_hidden_size ${lstm_hidden_size} \
                --agent.encoder normalize \
                --agent.encoder_normalize_value 11.0
        done
    done
done