import tyro
from config import Config
from envs import create_env
import functools
import os
import wandb
import dataclasses
import copy

from rb import TrajectoryUniformSamplingQueue, jit_wrap, flatten_batch

import jax
import jax.numpy as jnp
from jax import random

from impls.agents import create_agent
from envs.block_moving_env import AutoResetWrapper, TimeStep, GridStatesEnum
from config import ROOT_DIR
from impls.utils.checkpoints import save_agent
from utils import log_gif, sample_actions_critic


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def collect_data(agent, key, env, num_envs, episode_length, use_targets=False, critic_temp=None):
    def step_fn(carry, step_num):
        state, info, key = carry
        key, sample_key = jax.random.split(key)

        # Use jax.lax.cond instead of if statement to handle traced arrays
        state_agent = jax.lax.cond(
            use_targets,
            lambda: state.replace(),
            lambda: state.replace(
                grid=GridStatesEnum.remove_targets(state.grid), goal=GridStatesEnum.remove_targets(state.goal)
            ),
        )

        if critic_temp is None:
            actions = agent.sample_actions(
                state_agent.grid.reshape(num_envs, -1), state_agent.goal.reshape(num_envs, -1), seed=sample_key
            )
        else:
            actions = sample_actions_critic(
                agent,
                state_agent.grid.reshape(num_envs, -1),
                state_agent.goal.reshape(num_envs, -1),
                seed=sample_key,
                temperature=critic_temp,
            )

        new_state, reward, done, info = env.step(state, actions)
        timestep = TimeStep(
            key=state.key,
            grid=state.grid,
            agent_pos=state.agent_pos,
            agent_has_box=state.agent_has_box,
            steps=state.steps,
            number_of_boxes=state.number_of_boxes,
            action=actions,
            goal=state.goal,
            reward=reward,
            success=state.success,
            done=done,
        )
        return (new_state, info, key), timestep

    keys = jax.random.split(key, num_envs)
    state, info = env.reset(keys)
    (timestep, info, key), timesteps_all = jax.lax.scan(step_fn, (state, info, key), (), length=episode_length)
    return timestep, info, timesteps_all


@jax.jit
def extract_at_indices(data, indices):
    return jax.tree_util.tree_map(lambda x: x[jnp.arange(x.shape[0]), indices], data)


@functools.partial(jax.jit, static_argnums=(5,))
def get_single_pair_from_every_env(state, next_state, future_state, goal_index, key, use_double_batch_trick=False):
    """Sample two random indices and concatenate the results."""

    # # Sample two random indices for each batch
    # def double_batch_fn(key):
    #     subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    #     random_indices1 = jax.random.randint(subkey1, (state.grid.shape[0],), minval=0, maxval=state.grid.shape[1])
    #     random_indices2 = jax.random.randint(subkey2, (state.grid.shape[0],), minval=0, maxval=state.grid.shape[1])

    #     state1 = extract_at_indices(state, random_indices1)
    #     state2 = extract_at_indices(state, random_indices2)
    #     future_state1 = extract_at_indices(future_state, random_indices1)
    #     future_state2 = extract_at_indices(future_state, random_indices2)
    #     goal_index1 = extract_at_indices(goal_index, random_indices1)
    #     goal_index2 = extract_at_indices(goal_index, random_indices2)

    #     envs_to_take = jax.random.randint(subkey3, (state.grid.shape[0] // 2,), minval=0, maxval=state.grid.shape[0])
    #     state1 = jax.tree_util.tree_map(lambda x: x[envs_to_take], state1)
    #     state2 = jax.tree_util.tree_map(lambda x: x[envs_to_take], state2)
    #     future_state1 = jax.tree_util.tree_map(lambda x: x[envs_to_take], future_state1)
    #     future_state2 = jax.tree_util.tree_map(lambda x: x[envs_to_take], future_state2)
    #     goal_index1 = jax.tree_util.tree_map(lambda x: x[envs_to_take], goal_index1)
    #     goal_index2 = jax.tree_util.tree_map(lambda x: x[envs_to_take], goal_index2)

    #     # Concatenate the two samples
    #     state_concat = jax.tree_util.tree_map(
    #         lambda x1, x2: jnp.concatenate([x1, x2], axis=0), state1, state2
    #     )  # (batch_size, grid_size, grid_size)
    #     actions = jnp.concatenate([state1.action, state2.action], axis=0)
    #     future_state_concat = jax.tree_util.tree_map(
    #         lambda x1, x2: jnp.concatenate([x1, x2], axis=0), future_state1, future_state2
    #     )
    #     goal_index_concat = jnp.concatenate([goal_index1, goal_index2], axis=0)

    #     return state_concat, actions, future_state_concat, goal_index_concat

    def single_batch_fn(key):
        random_indices = jax.random.randint(key, (state.grid.shape[0],), minval=0, maxval=state.grid.shape[1])
        state_single = extract_at_indices(state, random_indices)  # (batch_size, grid_size, grid_size)
        next_state_single = extract_at_indices(next_state, random_indices)  # (batch_size, grid_size, grid_size)
        future_state_single = extract_at_indices(future_state, random_indices)
        goal_index_single = extract_at_indices(goal_index, random_indices)
        return state_single, state_single.action, next_state_single, future_state_single, goal_index_single

    return single_batch_fn(key)
    # return jax.lax.cond(use_double_batch_trick, double_batch_fn, single_batch_fn, key)


def evaluate_agent_in_specific_env(agent, key, jitted_flatten_batch, config, name, create_gif=False, critic_temp=None):
    env_eval = create_env(config.env)
    env_eval = AutoResetWrapper(env_eval)
    env_eval.step = jax.jit(jax.vmap(env_eval.step))
    env_eval.reset = jax.jit(jax.vmap(env_eval.reset))
    prefix = f"eval{name}"
    prefix_gif = f"gif{name}"

    data_key, double_batch_key = jax.random.split(key, 2)
    # Use collect_data for evaluation rollouts
    _, info, timesteps = collect_data(
        agent,
        data_key,
        env_eval,
        config.exp.num_envs,
        config.env.episode_length,
        use_targets=config.exp.use_targets,
        critic_temp=critic_temp,
    )
    timesteps = jax.tree_util.tree_map(lambda x: x.swapaxes(1, 0), timesteps)

    batch_keys = jax.random.split(data_key, config.exp.num_envs)
    state, next_state, future_state, goal_index = jitted_flatten_batch(config.exp.gamma, timesteps, batch_keys)

    # Sample and concatenate batch using the new function
    state, actions, next_state, future_state, goal_index = get_single_pair_from_every_env(
        state,
        next_state,
        future_state,
        goal_index,
        double_batch_key,
        use_double_batch_trick=config.exp.use_double_batch_trick,
    )  # state.grid is of shape (batch_size * 2, grid_size, grid_size)
    if not config.exp.use_targets:
        state = state.replace(grid=GridStatesEnum.remove_targets(state.grid))
        next_state = next_state.replace(grid=GridStatesEnum.remove_targets(next_state.grid))
        future_state = future_state.replace(grid=GridStatesEnum.remove_targets(future_state.grid))

    # Create valid batch
    valid_batch = {
        "observations": state.grid.reshape(state.grid.shape[0], -1),
        "next_observations": next_state.grid.reshape(next_state.grid.shape[0], -1),
        "actions": actions.squeeze(),
        "rewards": state.reward.reshape(state.reward.shape[0], -1),
        "masks": 1.0 - state.done.reshape(state.done.shape[0], -1),
        "value_goals": future_state.grid.reshape(future_state.grid.shape[0], -1),
        "actor_goals": future_state.grid.reshape(future_state.grid.shape[0], -1),
    }

    # Compute losses on example batch
    loss, loss_info = agent.total_loss(valid_batch, None)

    # Compile evaluation info
    # Only consider episodes that are done
    done_mask = timesteps.done
    eval_info_tmp = {
        f"{prefix}/mean_reward": timesteps.reward[done_mask].mean(),
        f"{prefix}/min_reward": timesteps.reward[done_mask].min(),
        f"{prefix}/max_reward": timesteps.reward[done_mask].max(),
        f"{prefix}/mean_success": timesteps.success[done_mask].mean(),
        f"{prefix}/mean_boxes_on_target": info["boxes_on_target"].mean(),
        f"{prefix}/total_loss": loss,
        f"{prefix}/actor_loss": loss_info["actor/actor_loss"],
    }
    if config.agent.agent_name == "crl":
        eval_info_tmp.update(
            {
                f"{prefix}/contrastive_loss": loss_info["critic/contrastive_loss"],
                f"{prefix}/cat_acc": loss_info["critic/categorical_accuracy"],
                f"{prefix}/v_mean": loss_info["critic/v_mean"],
            }
        )
    elif config.agent.agent_name == "gciql":
        eval_info_tmp.update(
            {
                f"{prefix}/critic_loss": loss_info["critic/critic_loss"],
                f"{prefix}/q_mean": loss_info["critic/q_mean"],
                f"{prefix}/v_mean": loss_info["value/v_mean"],
            }
        )
    else:
        raise ValueError(f"Unknown agent name {config.agent.agent_name}")

    if create_gif:
        log_gif(env_eval, config.env.episode_length, prefix_gif, timesteps, state)

    return eval_info_tmp, loss_info


def evaluate_agent(agent, key, jitted_flatten_batch, epoch, config):
    """Evaluate agent by running rollouts using collect_data and computing losses."""

    eval_configs = [config]
    eval_names_suff = [""]

    eval_info = {"epoch": epoch}
    create_gif = epoch > 0 and epoch % config.exp.gif_every == 0

    if config.exp.eval_mirrored:
        mirroring_config = copy.deepcopy(config)
        mirroring_config.env.generator_mirroring = True
        eval_configs.append(mirroring_config)
        eval_names_suff.append("_mirrored")

    if config.exp.eval_different_box_numbers:
        for number_of_boxes in range(1, 12, 2):
            new_config = copy.deepcopy(config)
            new_config.env = dataclasses.replace(
                new_config.env,
                number_of_boxes_min=number_of_boxes,
                number_of_boxes_max=number_of_boxes,
                number_of_moving_boxes_max=number_of_boxes,
            )
            eval_configs.append(new_config)
            eval_names_suff.append("_" + str(number_of_boxes))

    for eval_config, eval_name_suff in zip(eval_configs, eval_names_suff):
        eval_info_tmp, loss_info = evaluate_agent_in_specific_env(
            agent, key, jitted_flatten_batch, eval_config, eval_name_suff, create_gif=create_gif
        )
        eval_info.update(eval_info_tmp)
        # With critic softmax(Q) actions:
        eval_info_tmp, loss_info = evaluate_agent_in_specific_env(
            agent,
            key,
            jitted_flatten_batch,
            eval_config,
            eval_name_suff + "soft_q",
            create_gif=create_gif,
            critic_temp=1.0,
        )
        eval_info.update(eval_info_tmp)

        if eval_name_suff == "":
            eval_info.update(loss_info)

    wandb.log(eval_info)


def train(config: Config):
    wandb.init(
        project=config.exp.project,
        name=config.exp.name,
        config=config,
        entity=config.exp.entity,
        mode=config.exp.mode,
    )

    env = create_env(config.env)
    env = AutoResetWrapper(env)
    key = random.PRNGKey(config.exp.seed)
    env.step = jax.jit(jax.vmap(env.step))
    env.reset = jax.jit(jax.vmap(env.reset))
    partial_flatten = functools.partial(flatten_batch, get_next_obs=config.agent.use_next_obs)
    jitted_flatten_batch = jax.jit(jax.vmap(partial_flatten, in_axes=(None, 0, 0)), static_argnums=(0,))

    dummy_timestep = env.get_dummy_timestep(key)

    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=config.exp.max_replay_size,
            dummy_data_sample=dummy_timestep,
            sample_batch_size=config.exp.batch_size,
            num_envs=config.exp.num_envs,
            episode_length=config.env.episode_length,
        )
    )
    buffer_state = jax.jit(replay_buffer.init)(key)

    example_batch = {
        "observations": dummy_timestep.grid.reshape(1, -1),  # Add batch dimension
        "next_observations": dummy_timestep.grid.reshape(1, -1),
        "actions": jnp.ones((1,), dtype=jnp.int8)
        * (
            env._env.action_space - 1
        ),  # TODO: make sure it should be the maximal value of action space  # Single action for batch size 1
        "rewards": dummy_timestep.reward.reshape(1, -1),
        "masks": 1.0 - dummy_timestep.reward.reshape(1, -1),
        "value_goals": dummy_timestep.grid.reshape(1, -1),
        "actor_goals": dummy_timestep.grid.reshape(1, -1),
    }

    agent = create_agent(config.agent, example_batch, config.exp.seed)

    def make_batch(buffer_state, key):
        key, batch_key, double_batch_key = jax.random.split(key, 3)
        # Sample and process transitions
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        batch_keys = jax.random.split(batch_key, transitions.grid.shape[0])
        state, next_state, future_state, goal_index = jitted_flatten_batch(config.exp.gamma, transitions, batch_keys)

        state, actions, next_state, future_state, goal_index = get_single_pair_from_every_env(
            state,
            next_state,
            future_state,
            goal_index,
            double_batch_key,
            use_double_batch_trick=config.exp.use_double_batch_trick,
        )
        if not config.exp.use_targets:
            state = state.replace(grid=GridStatesEnum.remove_targets(state.grid))
            next_state = next_state.replace(grid=GridStatesEnum.remove_targets(next_state.grid))
            future_state = future_state.replace(grid=GridStatesEnum.remove_targets(future_state.grid))
        # Create valid batch
        batch = {
            "observations": state.grid.reshape(state.grid.shape[0], -1),
            "next_observations": next_state.grid.reshape(next_state.grid.shape[0], -1),
            "actions": actions.squeeze(),
            "rewards": state.reward.reshape(state.reward.shape[0], -1),
            "masks": 1.0 - state.reward.reshape(state.reward.shape[0], -1),  # TODO: add success and reward separately
            "value_goals": future_state.grid.reshape(future_state.grid.shape[0], -1),
            "actor_goals": future_state.grid.reshape(future_state.grid.shape[0], -1),
        }
        return buffer_state, batch

    @jax.jit
    def update_step(carry, _):
        buffer_state, agent, key = carry
        key, batch_key = jax.random.split(key, 2)
        buffer_state, batch = make_batch(buffer_state, batch_key)
        agent, update_info = agent.update(batch)
        return (buffer_state, agent, key), update_info

    @jax.jit
    def train_interval(buffer_state, agent, key):
        key, data_key, up_key = jax.random.split(key, 3)
        _, _, timesteps = collect_data(
            agent, data_key, env, config.exp.num_envs, config.env.episode_length, use_targets=config.exp.use_targets
        )
        buffer_state = replay_buffer.insert(buffer_state, timesteps)
        (buffer_state, agent, _), _ = jax.lax.scan(
            update_step, (buffer_state, agent, up_key), None, length=config.exp.updates_per_rollout
        )
        return buffer_state, agent, key

    if config.exp.save_dir is None:
        run_directory = os.path.join(ROOT_DIR, "runs", config.exp.name)
    else:
        run_directory = os.path.join(config.exp.save_dir, config.exp.name)

    os.makedirs(run_directory, exist_ok=True)

    # Evaluate before training
    evaluate_agent(agent, key, jitted_flatten_batch, 0, config)
    save_agent(agent, config, save_dir=run_directory, epoch=0)

    for epoch in range(config.exp.epochs):
        for _ in range(config.exp.intervals_per_epoch):
            buffer_state, agent, key = train_interval(buffer_state, agent, key)

        evaluate_agent(agent, key, jitted_flatten_batch, epoch + 1, config)
        save_agent(agent, config, save_dir=run_directory, epoch=epoch + 1)


if __name__ == "__main__":
    args = tyro.cli(Config, config=(tyro.conf.ConsolidateSubcommandArgs,))
    if args.exp.batch_size > args.exp.num_envs:
        raise ValueError("Batch size has to be less than or equal to number of environments")
    train(args)
