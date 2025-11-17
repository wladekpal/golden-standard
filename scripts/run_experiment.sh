ml ML-bundle/24.06a

unset LD_LIBRARY_PATH

export UV_PROJECT_ENVIRONMENT=$SCRATCH/crl_subgoal/.venv
export UV_CACHE_DIR=$SCRATCH/.cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_API_KEY=$(cat ~/.wandb_key)

SEED=$1
AGENT=$2
BOX_NUM=$3


if [ $AGENT = "gcdqn" ]; then
        GOALS_FLAG=--exp.use_future_and_random_goals
else
        GOALS_FLAG=--exp.no_use_future_and_random_goals
fi

uv run src/train.py env:box-moving \
        --agent.agent_name $AGENT \
        --agent.action_sampling softmax \
        --exp.name "$AGENT"_tunable_ent_1.38_sampling_"$BOX_NUM"_boxes_6_grid \
        --env.number_of_boxes_min $BOX_NUM \
        --env.number_of_boxes_max $BOX_NUM \
        --env.number_of_moving_boxes_max $BOX_NUM \
        --env.grid_size 6 \
        --exp.gamma 0.99 \
        --env.episode_length 100 \
        --exp.seed $SEED \
        --exp.project obbt \
        --exp.entity cl-probing \
        --exp.epochs 50 \
        --exp.gif_every 10 \
        --agent.alpha 0.1 \
        --exp.max_replay_size 10000 \
        --exp.batch_size 256 \
        --exp.eval_special \
        --env.level-generator variable \
        --env.quarter_size 3 \
        --exp.save_dir $SCRATCH/crl_subgoal/runs \
        $GOALS_FLAG






