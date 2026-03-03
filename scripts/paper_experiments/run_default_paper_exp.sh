#!/bin/bash

GRID=6
BOX_NUM=4

for KIND in mc td;
do
    for ARCH in big small;
    do
        for AGENT in gcdqn clearn_search crl_search;
        do
            if [ $AGENT = "gcdqn" ]; then
                GOALS_FLAG=--exp.use_future_and_random_goals
            else
                GOALS_FLAG=--exp.no_use_future_and_random_goals
            fi

            if [ $KIND = "mc" ]; then
                KIND_FLAG=--agent.use_discounted_mc_rewards
                IS_TD_FLAG=--agent.no_is_td
            else
                KIND_FLAG=--agent.no_use_discounted_mc_rewards
                IS_TD_FLAG=--agent.is_td
                if [ $AGENT = "crl_search" ]; then
                    continue
                fi
            fi

            if [ $ARCH = "big" ]; then
                ARCH_FLAG=res_block
                SIZE_FLAG="1024 1024"
                TIME="10:00:00"
            else
                ARCH_FLAG=mlp
                SIZE_FLAG="256 256"
                TIME="02:00:00"
            fi

            sbatch <<EOT
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=$TIME
#SBATCH --account=plgcrlreason-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=experiment_out.txt


ml ML-bundle/24.06a

unset LD_LIBRARY_PATH

export UV_PROJECT_ENVIRONMENT=$SCRATCH/crl_subgoal/.venv
export UV_CACHE_DIR=$SCRATCH/.cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false

for SEED in 1 2 3; # Change So that it fits on a machine!
do
    uv run src/train.py env:box-pushing \
            --agent.agent_name $AGENT \
            --exp.name "$KIND"_"$AGENT"_"$ARCH"_"$BOX_NUM"_boxes_"$GRID"_grid \
            --env.number_of_boxes_min $BOX_NUM \
            --env.number_of_boxes_max $BOX_NUM \
            --env.number_of_moving_boxes_max $BOX_NUM \
            --env.grid_size $GRID \
            --exp.gamma 0.99 \
            --env.episode_length 100 \
            --exp.seed \$SEED \
            --exp.project paper_default \
            --exp.entity cl-probing \
            --exp.epochs 50 \
            --exp.gif_every 10 \
            --agent.alpha 0.1 \
            --exp.max_replay_size 10000 \
            --exp.batch_size 256 \
            --agent.value_hidden_dims $SIZE_FLAG \
            --agent.net_arch $ARCH_FLAG \
            --agent.target_entropy "-1.1" \
            --env.level-generator default \
            --exp.save_dir $SCRATCH/crl_subgoal/runs \
            $GOALS_FLAG \
            $IS_TD_FLAG \
            $KIND_FLAG &
done

wait
EOT
        done
    done
done
