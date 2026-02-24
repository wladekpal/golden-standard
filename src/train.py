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
from envs.block_moving.block_moving_env import BoxMovingEnv
from envs.block_moving.wrappers import wrap_for_eval, wrap_for_training
from envs.block_moving.env_types import TimeStep, remove_targets
from envs.arm_envs.arm_envs import ArmEnvsConfig
from envs.arm_envs.wrappers import arm_wrap_for_eval, arm_wrap_for_training
from envs.arm_envs.env_types import ArmTimeStep
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
            lambda: state.replace(grid=remove_targets(state.grid), goal=remove_targets(state.goal)),
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
            success=new_state.success,
            done=done,
            truncated=info["truncated"],
            extras=state.extras,
        )
        return (new_state, info, key), timestep

    keys = jax.random.split(key, num_envs)
    state, info = env.reset(keys)
    (timestep, info, key), timesteps_all = jax.lax.scan(step_fn, (state, info, key), (), length=episode_length)
    return timestep, info, timesteps_all


@jax.jit
def extract_at_indices(data, indices):
    return jax.tree_util.tree_map(lambda x: x[jnp.arange(x.shape[0]), indices], data)


@functools.partial(jax.jit)
def get_single_pair_from_every_env(state, next_state, future_state, goal_index, key):
    """Sample two random indices and concatenate the results."""

    def single_batch_fn(key):
        random_indices = jax.random.randint(key, (state.grid.shape[0],), minval=0, maxval=state.grid.shape[1])
        state_single = extract_at_indices(state, random_indices)  # (batch_size, grid_size, grid_size)
        next_state_single = extract_at_indices(next_state, random_indices)  # (batch_size, grid_size, grid_size)
        future_state_single = extract_at_indices(future_state, random_indices)
        goal_index_single = extract_at_indices(goal_index, random_indices)
        return state_single, state_single.action, next_state_single, future_state_single, goal_index_single

    return single_batch_fn(key)


def create_batch(
    timesteps,
    key,
    gamma,
    use_targets,
    use_future_and_random_goals,
    jitted_flatten_batch,
    use_discounted_mc_rewards=False,
):
    batch_key, sampling_key = jax.random.split(key, 2)
    batch_keys = jax.random.split(batch_key, timesteps.grid.shape[0])
    state, next_state, future_state, goal_index = jitted_flatten_batch(
        gamma, use_discounted_mc_rewards, timesteps, batch_keys
    )

    state, actions, next_state, future_state, goal_index = get_single_pair_from_every_env(
        state,
        next_state,
        future_state,
        goal_index,
        sampling_key,
    )
    if not use_targets:
        state = state.replace(grid=remove_targets(state.grid), goal=remove_targets(state.goal))
        next_state = next_state.replace(grid=remove_targets(next_state.grid))
        future_state = future_state.replace(grid=remove_targets(future_state.grid))

    if use_future_and_random_goals:
        value_goals = jnp.concatenate(
            [
                jnp.roll(state.grid, shift=1, axis=(0))[: state.grid.shape[0] // 2],
                future_state.grid[state.grid.shape[0] // 2 :],
            ],
            axis=0,
        )
        actor_goals = jnp.concatenate(
            [
                jnp.roll(state.grid, shift=1, axis=(0))[: state.grid.shape[0] // 2],
                future_state.grid[state.grid.shape[0] // 2 :],
            ],
            axis=0,
        )
    else:
        value_goals = future_state.grid
        actor_goals = future_state.grid

    # TODO: this should be use only with dense reward/relabeling
    reward = jax.vmap(BoxMovingEnv.get_reward)(state.grid, next_state.grid, value_goals)

    # Create valid batch
    batch = {
        "observations": state.grid.reshape(state.grid.shape[0], -1),
        "next_observations": next_state.grid.reshape(next_state.grid.shape[0], -1),
        "actions": actions.squeeze(),
        "rewards": reward.reshape(reward.shape[0], -1).squeeze(),
        "masks": jnp.ones_like(reward.reshape(reward.shape[0], -1).squeeze()),  # Bootstrap always
        "value_goals": value_goals.reshape(value_goals.shape[0], -1),
        "actor_goals": actor_goals.reshape(actor_goals.shape[0], -1),
    }
    return batch


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def collect_data_arm(agent, key, env, num_envs, episode_length):
    """Collect rollouts for arm environments."""

    def step_fn(carry, _):
        state, key = carry
        key, sample_key = jax.random.split(key)

        # Split observation into state and goal components.
        obs_all = state.obs  # (num_envs, obs_dim)
        obs = obs_all[:, : env.state_dim]
        goal = obs_all[:, env.state_dim :]

        actions = agent.sample_actions(obs, goal, seed=sample_key)
        new_state = env.step(state, actions)

        timestep = ArmTimeStep(
            obs=obs,
            action=actions,
            goal=goal,
            reward=new_state.reward,
            success=new_state.metrics["success"],
            done=new_state.done,
            truncated=new_state.info["truncated"],
            steps=state.info["steps"],
        )
        return (new_state, key), timestep

    keys = jax.random.split(key, num_envs)
    state = env.reset(keys)
    (_, _), timesteps_all = jax.lax.scan(step_fn, (state, key), (), length=episode_length)
    return timesteps_all


def create_batch_arm(
    timesteps,
    key,
    gamma,
    jitted_flatten_batch,
    state_dim,
    goal_indices,
    completion_goal_indices,
    goal_reach_thresh,
    use_discounted_mc_rewards=False,
):
    """Create training batch for arm environments."""

    batch_key = key
    batch_keys = jax.random.split(batch_key, timesteps.obs.shape[0])
    state, next_state, future_state, goal_index = jitted_flatten_batch(
        gamma, use_discounted_mc_rewards, timesteps, batch_keys
    )

    # Observations and goals.
    obs = state.obs
    next_obs = next_state.obs
    goals = future_state.goal

    # Reward: recompute as goal-reaching indicator.
    current_pos = obs[..., completion_goal_indices]
    target_pos = goals[..., : current_pos.shape[-1]]
    dist = jnp.linalg.norm(current_pos - target_pos, axis=-1)
    reward = (dist < goal_reach_thresh).astype(jnp.float32)

    actions = state.action

    # Flatten env and time dimensions into a single batch.
    batch_size = obs.shape[0] * obs.shape[1]

    obs_flat = obs.reshape(batch_size, -1)
    next_obs_flat = next_obs.reshape(batch_size, -1)
    actions_flat = actions.reshape(batch_size, -1)
    reward_flat = reward.reshape(batch_size)
    value_goals_flat = goals.reshape(batch_size, -1)
    actor_goals_flat = value_goals_flat

    batch = {
        "observations": obs_flat,
        "next_observations": next_obs_flat,
        "actions": actions_flat,
        "rewards": reward_flat,
        "masks": jnp.ones_like(reward_flat),
        "value_goals": value_goals_flat,
        "actor_goals": actor_goals_flat,
    }
    return batch


def evaluate_agent_arm(agent, key, jitted_create_batch_arm, epoch, config):
    """Evaluate agent in an arm environment."""

    env_eval = create_env(config.env)
    env_eval = arm_wrap_for_eval(env_eval)
    env_eval.step = jax.jit(jax.vmap(env_eval.step))
    env_eval.reset = jax.jit(jax.vmap(env_eval.reset))

    data_key, batch_key = jax.random.split(key, 2)
    timesteps = collect_data_arm(
        agent,
        data_key,
        env_eval,
        config.exp.num_envs,
        config.env.episode_length,
    )
    # timesteps: (episode_length, num_envs, ...)
    timesteps = jax.tree_util.tree_map(lambda x: x.swapaxes(1, 0), timesteps)

    valid_batch = jitted_create_batch_arm(timesteps, batch_key)
    loss, loss_info = agent.total_loss(valid_batch, None)

    # For now, aggregate rewards and successes over the final timestep of each env.
    final_rewards = timesteps.reward[:, -1]
    final_success = timesteps.success[:, -1]

    eval_info = {
        "eval_arm/mean_reward": final_rewards.mean(),
        "eval_arm/min_reward": final_rewards.min(),
        "eval_arm/max_reward": final_rewards.max(),
        "eval_arm/mean_success": final_success.mean(),
        "eval_arm/total_loss": loss,
    }

    wandb.log({"epoch": epoch, **eval_info, **loss_info})
    return eval_info, loss_info


def evaluate_agent_in_specific_env(agent, key, jitted_create_batch, config, name, create_gif=False, critic_temp=None):
    env_eval = create_env(config.env)
    env_eval = wrap_for_eval(env_eval)  # Note: Wrap for eval is not using any quarter filtering
    env_eval.step = jax.jit(jax.vmap(env_eval.step))
    env_eval.reset = jax.jit(jax.vmap(env_eval.reset))
    prefix = f"eval{name}"
    prefix_gif = f"gif{name}"

    data_key, batch_key = jax.random.split(key, 2)
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
    timesteps = jax.tree_util.tree_map(lambda x: x.swapaxes(1, 0), timesteps)  # Returns N_envs x episode_length x ...

    valid_batch = jitted_create_batch(timesteps, batch_key)

    # Compute losses on example batch
    loss, loss_info = agent.total_loss(valid_batch, None)

    # Compile evaluation info
    # Only consider episodes that are done
    # truncated_mask = timesteps.truncated
    done_or_trunc = timesteps.done | timesteps.truncated  # bool, (N_envs, T)
    # first-occurrence mask: True at the first time (per row) where done_or_trunc is True
    truncated_mask = (jnp.cumsum(done_or_trunc.astype(jnp.int32), axis=1) == 1) & done_or_trunc

    eval_info_tmp = {
        f"{prefix}/mean_reward": timesteps.reward[truncated_mask].mean(),
        f"{prefix}/min_reward": timesteps.reward[truncated_mask].min(),
        f"{prefix}/max_reward": timesteps.reward[truncated_mask].max(),
        f"{prefix}/mean_success": timesteps.success[truncated_mask].mean(),
        f"{prefix}/mean_boxes_on_target": info["boxes_on_target"].mean(),
        f"{prefix}/mean_ep_len": timesteps.steps[truncated_mask].mean(),
        f"{prefix}/total_loss": loss,
    }
    if config.agent.agent_name == "crl" or config.agent.agent_name == "crl_search":
        eval_info_tmp.update(
            {
                f"{prefix}/contrastive_loss": loss_info["critic/contrastive_loss"],
                f"{prefix}/cat_acc": loss_info["critic/categorical_accuracy"],
            }
        )
    elif config.agent.agent_name == "gciql" or config.agent.agent_name == "gciql_search":
        eval_info_tmp.update(
            {
                f"{prefix}/critic_loss": loss_info["critic/critic_loss"],
                f"{prefix}/q_mean": loss_info["critic/q_mean"],
                f"{prefix}/q_min": loss_info["critic/q_min"],
                f"{prefix}/q_max": loss_info["critic/q_max"],
            }
        )
    elif config.agent.agent_name == "gcdqn":
        eval_info_tmp.update(
            {
                f"{prefix}/critic_loss": loss_info["critic/critic_loss"],
                f"{prefix}/q_mean": loss_info["critic/q_mean"],
                f"{prefix}/q_min": loss_info["critic/q_min"],
                f"{prefix}/q_max": loss_info["critic/q_max"],
            }
        )
    elif config.agent.agent_name == "clearn_search":
        eval_info_tmp.update(
            {
                f"{prefix}/critic_loss": loss_info["critic/critic_loss"],
                f"{prefix}/q_mean": loss_info["critic/q_mean"],
                f"{prefix}/q_min": loss_info["critic/q_min"],
                f"{prefix}/q_max": loss_info["critic/q_max"],
            }
        )
    else:
        raise ValueError(f"Unknown agent name {config.agent.agent_name}")

    if create_gif:
        log_gif(env_eval, config.env.episode_length, prefix_gif, timesteps)

    return eval_info_tmp, loss_info


def evaluate_agent(agent, key, jitted_create_batch, epoch, config):
    """Evaluate agent by running rollouts using collect_data and computing losses."""

    eval_configs = [config]
    eval_names_suff = [""]

    eval_info = {"epoch": epoch}
    create_gif = epoch > 0 and epoch % config.exp.gif_every == 0 and config.exp.num_gifs > 0

    if config.exp.eval_special:
        special_config = copy.deepcopy(config)
        special_config.env.generator_special = True
        eval_configs.append(special_config)
        eval_names_suff.append("_special")

    if config.exp.eval_different_box_numbers:
        for number_of_boxes in [config.env.number_of_boxes_max]:
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
            agent, key, jitted_create_batch, eval_config, eval_name_suff, create_gif=create_gif
        )
        eval_info.update(eval_info_tmp)
        if eval_name_suff == "":
            eval_info.update(loss_info)

        # With critic softmax(Q) actions:
        eval_info_tmp, loss_info = evaluate_agent_in_specific_env(
            agent,
            key,
            jitted_create_batch,
            eval_config,
            eval_name_suff + "soft_q",
            create_gif=create_gif,
            critic_temp=1.0,
        )
        eval_info.update(eval_info_tmp)

    wandb.log(eval_info)


def train(config: Config):
    # Create dirs, init wandb
    wandb_config = copy.deepcopy(config)
    wandb_config.agent = dict(wandb_config.agent)
    wandb.init(
        project=config.exp.project,
        name=config.exp.name,
        config=wandb_config,
        entity=config.exp.entity,
        mode=config.exp.mode,
    )
    if config.exp.save_dir is None:
        run_directory = os.path.join(ROOT_DIR, "runs", config.exp.name)
    else:
        run_directory = os.path.join(config.exp.save_dir, config.exp.name)
    os.makedirs(run_directory, exist_ok=True)

    # Create environment and batch functions
    is_arm = isinstance(config.env, ArmEnvsConfig)

    if is_arm:
        env = create_env(config.env)
        env = arm_wrap_for_training(env)
        key = random.PRNGKey(config.exp.seed)
        env.step = jax.jit(jax.vmap(env.step))
        env.reset = jax.jit(jax.vmap(env.reset))
        jitted_flatten_batch = jax.jit(jax.vmap(flatten_batch, in_axes=(None, None, 0, 0)), static_argnums=(0, 1))
        jitted_create_batch_arm = functools.partial(
            create_batch_arm,
            gamma=config.exp.gamma,
            jitted_flatten_batch=jitted_flatten_batch,
            state_dim=env.state_dim,
            goal_indices=env.goal_indices,
            completion_goal_indices=env.completion_goal_indices,
            goal_reach_thresh=env.goal_reach_thresh,
            use_discounted_mc_rewards=config.agent.use_discounted_mc_rewards,
        )

        # Create replay buffer and agent
        dummy_timestep = env.get_dummy_arm_timestep(key)

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
            "observations": dummy_timestep.obs.reshape(1, -1),
            "next_observations": dummy_timestep.obs.reshape(1, -1),
            "actions": jnp.zeros((1, env.action_size), dtype=jnp.float32),
            "rewards": jnp.ones((1,), dtype=jnp.float32),
            "masks": jnp.ones((1,), dtype=jnp.int8),
            "value_goals": dummy_timestep.goal.reshape(1, -1),
            "actor_goals": dummy_timestep.goal.reshape(1, -1),
        }
        agent = create_agent(config.agent, example_batch, config.exp.seed)
    else:
        env = create_env(config.env)
        env = wrap_for_training(config, env)
        key = random.PRNGKey(config.exp.seed)
        env.step = jax.jit(jax.vmap(env.step))
        env.reset = jax.jit(jax.vmap(env.reset))
        jitted_flatten_batch = jax.jit(jax.vmap(flatten_batch, in_axes=(None, None, 0, 0)), static_argnums=(0, 1))
        jitted_create_batch = functools.partial(
            create_batch,
            gamma=config.exp.gamma,
            use_targets=config.exp.use_targets,
            use_future_and_random_goals=config.exp.use_future_and_random_goals,
            jitted_flatten_batch=jitted_flatten_batch,
            use_discounted_mc_rewards=config.agent.use_discounted_mc_rewards,
        )

        # Create replay buffer and agent
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
            * (env._env.action_space - 1),  # it should be the maximal value of action space
            "rewards": jnp.ones((1,), dtype=jnp.float32),
            "masks": jnp.ones((1,), dtype=jnp.int8),
            "value_goals": dummy_timestep.grid.reshape(1, -1),
            "actor_goals": dummy_timestep.grid.reshape(1, -1),
        }
        agent = create_agent(config.agent, example_batch, config.exp.seed)

    if is_arm:

        @jax.jit
        def update_step(carry, _):
            buffer_state, agent, key = carry
            key, batch_key = jax.random.split(key, 2)
            buffer_state, transitions = replay_buffer.sample(buffer_state)
            batch = jitted_create_batch_arm(transitions, batch_key)
            agent, update_info = agent.update(batch)
            return (buffer_state, agent, key), update_info

        @jax.jit
        def train_interval(buffer_state, agent, key):
            key, data_key, up_key = jax.random.split(key, 3)
            timesteps = collect_data_arm(agent, data_key, env, config.exp.num_envs, config.env.episode_length)

            buffer_state = replay_buffer.insert(buffer_state, timesteps)
            (buffer_state, agent, _), _ = jax.lax.scan(
                update_step, (buffer_state, agent, up_key), None, length=config.exp.updates_per_rollout
            )
            return buffer_state, agent, key

        # Main training loop with evaluation for arm envs
        evaluate_agent_arm(agent, key, jitted_create_batch_arm, 0, config)
    else:

        @jax.jit
        def update_step(carry, _):
            buffer_state, agent, key = carry
            key, batch_key = jax.random.split(key, 2)
            buffer_state, transitions = replay_buffer.sample(buffer_state)
            batch = jitted_create_batch(transitions, batch_key)
            agent, update_info = agent.update(batch)
            return (buffer_state, agent, key), update_info

        @jax.jit
        def train_interval(buffer_state, agent, key):
            key, data_key, up_key = jax.random.split(key, 3)
            _, _, timesteps = collect_data(
                agent,
                data_key,
                env,
                config.exp.num_envs,
                config.env.episode_length,
                use_targets=config.exp.use_targets,
            )

            buffer_state = replay_buffer.insert(buffer_state, timesteps)
            (buffer_state, agent, _), _ = jax.lax.scan(
                update_step, (buffer_state, agent, up_key), None, length=config.exp.updates_per_rollout
            )
            return buffer_state, agent, key

        # Main training loop with evaluation for grid envs
        evaluate_agent(agent, key, jitted_create_batch, 0, config)
    save_agent(agent, config, save_dir=run_directory, epoch=0)

    for epoch in range(config.exp.epochs):
        for _ in range(config.exp.intervals_per_epoch):
            buffer_state, agent, key = train_interval(buffer_state, agent, key)

        if is_arm:
            evaluate_agent_arm(agent, key, jitted_create_batch_arm, epoch + 1, config)
        else:
            evaluate_agent(agent, key, jitted_create_batch, epoch + 1, config)
        save_agent(agent, config, save_dir=run_directory, epoch=epoch + 1)


if __name__ == "__main__":
    args = tyro.cli(Config, config=(tyro.conf.ConsolidateSubcommandArgs,))
    if args.exp.batch_size > args.exp.num_envs:
        raise ValueError("Batch size has to be less than or equal to number of environments")
    train(args)
