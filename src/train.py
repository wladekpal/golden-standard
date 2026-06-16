import tyro
from config import Config
from envs import create_env
import functools
import json
import os
import wandb
import dataclasses
import copy

from rb import TrajectoryUniformSamplingQueue, jit_wrap, flatten_batch
from rb import flatten_native_batch

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from impls.agents import create_agent
from envs.block_moving.wrappers import wrap_for_eval, wrap_for_training
from envs.block_moving.env_types import TimeStep, remove_targets
from envs.block_moving.input_features import encode_grid_inputs
from config import ROOT_DIR
from impls.utils.checkpoints import save_agent
from utils import log_gif, log_jumanji_gif


def env_uses_native_rewards(env_or_config) -> bool:
    return bool(
        getattr(env_or_config, "uses_native_rewards", False) or env_or_config.__class__.__name__ == "JumanjiConfig"
    )


def get_env_action_dim(env) -> int:
    if hasattr(env, "action_dim"):
        return int(env.action_dim)
    return int(env.action_space)


def agent_inputs(env, state, use_targets=False, input_representation="normalized_flat"):
    if hasattr(env, "agent_inputs"):
        return env.agent_inputs(state, use_targets, input_representation)

    state_agent = jax.lax.cond(
        use_targets,
        lambda: state.replace(),
        lambda: state.replace(grid=remove_targets(state.grid), goal=remove_targets(state.goal)),
    )
    return (
        encode_grid_inputs(state_agent.grid, input_representation),
        encode_grid_inputs(state_agent.goal, input_representation),
        None,
    )


def transition_from_step(env, state, actions, new_state, reward, done, info):
    if hasattr(env, "transition_from_step"):
        return env.transition_from_step(state, actions, new_state, reward, done, info)

    return TimeStep(
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


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def collect_data(agent, key, env, num_envs, episode_length, use_targets=False, input_representation="normalized_flat"):
    def step_fn(carry, step_num):
        state, info, key = carry
        key, sample_key = jax.random.split(key)

        observations, goals, action_masks = agent_inputs(env, state, use_targets, input_representation)
        actions = agent.sample_actions(
            observations,
            goals,
            seed=sample_key,
            action_masks=action_masks,
        )

        new_state, reward, done, info = env.step(state, actions)
        timestep = transition_from_step(env, state, actions, new_state, reward, done, info)
        return (new_state, info, key), timestep

    keys = jax.random.split(key, num_envs)
    state, info = env.reset(keys)
    (timestep, info, key), timesteps_all = jax.lax.scan(step_fn, (state, info, key), (), length=episode_length)
    return timestep, info, timesteps_all


def create_box_moving_batch(
    timesteps,
    key,
    gamma,
    use_targets,
    use_future_and_random_goals,
    jitted_flatten_batch,
    action_dim,
    use_discounted_mc_rewards=False,
    input_representation="normalized_flat",
):
    batch_keys = jax.random.split(key, timesteps.grid.shape[0])

    # Since flatten_batch is vmapped and we don't have access to whole batch inside, we need to roll the grids here,
    # to provide source of random targets.
    # rolling_mask is used to indicate whether for particular item in batch we should use future or random goals
    rolled_grids = jnp.roll(timesteps.grid, shift=1, axis=0)
    if use_future_and_random_goals:
        batch_half = timesteps.grid.shape[0] // 2
        rolling_mask = jnp.concatenate(
            [jnp.ones(batch_half, dtype=jnp.bool), jnp.zeros(timesteps.grid.shape[0] - batch_half, dtype=jnp.bool)]
        )
    else:
        rolling_mask = jnp.zeros(timesteps.grid.shape[0], dtype=jnp.bool)

    state, actions, next_state, value_goals, actor_goals, reward = jitted_flatten_batch(
        gamma, use_discounted_mc_rewards, use_targets, timesteps, rolled_grids, rolling_mask, batch_keys
    )
    flat_rewards = reward.reshape(reward.shape[0], -1).squeeze()

    # Create valid batch
    batch = {
        "observations": encode_grid_inputs(state.grid, input_representation),
        "next_observations": encode_grid_inputs(next_state.grid, input_representation),
        "actions": actions.squeeze(),
        "rewards": flat_rewards,
        "masks": jnp.ones_like(flat_rewards),  # Bootstrap always
        "value_goals": encode_grid_inputs(value_goals, input_representation),
        "actor_goals": encode_grid_inputs(actor_goals, input_representation),
        "action_masks": jnp.ones((flat_rewards.shape[0], action_dim), dtype=jnp.bool_),
        "next_action_masks": jnp.ones((flat_rewards.shape[0], action_dim), dtype=jnp.bool_),
    }
    return batch


def create_native_reward_batch(timesteps, key, jitted_flatten_native_batch):
    batch_keys = jax.random.split(key, timesteps.grid.shape[0])
    state, next_state = jitted_flatten_native_batch(timesteps, batch_keys)
    rewards = state.reward.reshape((state.reward.shape[0],))
    done_or_truncated = state.done | state.truncated

    return {
        "observations": state.grid.reshape(state.grid.shape[0], -1),
        "next_observations": next_state.grid.reshape(next_state.grid.shape[0], -1),
        "actions": state.action.reshape((state.action.shape[0],)).astype(jnp.int32),
        "rewards": rewards,
        "masks": (~done_or_truncated).astype(jnp.float32).reshape(rewards.shape),
        "value_goals": state.goal.reshape(state.goal.shape[0], -1),
        "actor_goals": state.goal.reshape(state.goal.shape[0], -1),
        "action_masks": state.action_mask,
        "next_action_masks": next_state.action_mask,
    }


CRITIC_LOSS_AGENTS = {"gciql", "gciql_search", "gcdqn", "clearn_search"}


def get_agent_specific_eval_metrics(prefix, loss_info, agent_name):
    if agent_name in {"crl", "crl_search"}:
        return {
            f"{prefix}/contrastive_loss": loss_info["critic/contrastive_loss"],
            f"{prefix}/cat_acc": loss_info["critic/categorical_accuracy"],
        }

    if agent_name in CRITIC_LOSS_AGENTS:
        return {
            f"{prefix}/critic_loss": loss_info["critic/critic_loss"],
            f"{prefix}/q_mean": loss_info["critic/q_mean"],
            f"{prefix}/q_min": loss_info["critic/q_min"],
            f"{prefix}/q_max": loss_info["critic/q_max"],
        }

    if agent_name in {"gcbc"}:
        return {
            f"{prefix}/actor_loss": loss_info["actor/actor_loss"],
            f"{prefix}/bc_log_prob": loss_info["actor/bc_log_prob"],
        }

    raise ValueError(f"Unknown agent name {agent_name}")


def json_ready(value):
    value = jax.device_get(value)
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.item() if value.shape == () else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def append_metrics_jsonl(run_directory, metrics):
    with open(os.path.join(run_directory, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(json_ready(metrics), sort_keys=True) + "\n")


def evaluate_agent_in_specific_env(agent, key, jitted_create_batch, config, name, create_gif=False):
    env_eval = create_env(config.env)
    if not getattr(env_eval, "is_jumanji", False):
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
        input_representation=config.exp.input_representation,
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
    first_end_mask = (jnp.cumsum(done_or_trunc.astype(jnp.int32), axis=1) == 1) & done_or_trunc
    last_step_mask = jnp.arange(timesteps.reward.shape[1])[None, :] == (timesteps.reward.shape[1] - 1)
    truncated_mask = jnp.where(first_end_mask.any(axis=1, keepdims=True), first_end_mask, last_step_mask)

    eval_info_tmp = {
        f"{prefix}/mean_reward": timesteps.reward[truncated_mask].mean(),
        f"{prefix}/min_reward": timesteps.reward[truncated_mask].min(),
        f"{prefix}/max_reward": timesteps.reward[truncated_mask].max(),
        f"{prefix}/mean_success": timesteps.success[truncated_mask].mean(),
        f"{prefix}/mean_boxes_on_target": info.get("boxes_on_target", jnp.array(0.0)).mean(),
        f"{prefix}/mean_ep_len": timesteps.steps[truncated_mask].mean(),
        f"{prefix}/total_loss": loss,
    }
    eval_info_tmp.update(get_agent_specific_eval_metrics(prefix, loss_info, config.agent.agent_name))

    if create_gif:
        if getattr(env_eval, "is_jumanji", False):
            gif_key = jax.random.fold_in(key, 1)
            log_jumanji_gif(
                env_eval,
                agent,
                gif_key,
                config.env.episode_length,
                prefix_gif,
                use_targets=config.exp.use_targets,
                input_representation=config.exp.input_representation,
            )
        else:
            log_gif(env_eval, config.env.episode_length, prefix_gif, timesteps)

    return eval_info_tmp, loss_info


def evaluate_agent(agent, key, jitted_create_batch, epoch, config):
    """Evaluate agent by running rollouts using collect_data and computing losses."""

    eval_configs = [config]
    eval_names_suff = [""]

    eval_info = {"epoch": epoch}
    create_gif = epoch > 0 and epoch % config.exp.gif_every == 0 and config.exp.num_gifs > 0

    if config.exp.eval_special and hasattr(config.env, "generator_special"):
        special_config = copy.deepcopy(config)
        special_config.env.generator_special = True
        eval_configs.append(special_config)
        eval_names_suff.append("_special")

    if config.exp.eval_different_box_numbers and hasattr(config.env, "number_of_boxes_max"):
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

    wandb.log(eval_info)
    return eval_info


def init_wandb_and_run_directory(config: Config):
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
    return run_directory


def build_training_env(config: Config):
    env = create_env(config.env)
    if not getattr(env, "is_jumanji", False):
        env = wrap_for_training(config, env)
    env.step = jax.jit(jax.vmap(env.step))
    env.reset = jax.jit(jax.vmap(env.reset))
    return env


def build_jitted_create_batch(config: Config):
    if env_uses_native_rewards(config.env):
        jitted_flatten_native_batch = jax.jit(jax.vmap(flatten_native_batch, in_axes=(0, 0)))
        return functools.partial(
            create_native_reward_batch,
            jitted_flatten_native_batch=jitted_flatten_native_batch,
        )

    jitted_flatten_batch = jax.jit(
        jax.vmap(flatten_batch, in_axes=(None, None, None, 0, 0, 0, 0)), static_argnums=(0, 1, 2)
    )
    return functools.partial(
        create_box_moving_batch,
        gamma=config.agent.discount,
        use_targets=config.exp.use_targets,
        use_future_and_random_goals=config.exp.use_future_and_random_goals,
        jitted_flatten_batch=jitted_flatten_batch,
        action_dim=getattr(config.env, "action_space", 6),
        use_discounted_mc_rewards=config.agent.use_discounted_mc_rewards,
        input_representation=config.exp.input_representation,
    )


def build_replay_buffer_and_agent(config: Config, env, key):
    dummy_timestep = env.get_dummy_timestep(key)
    action_dim = get_env_action_dim(env)
    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=config.exp.max_replay_size,
            dummy_data_sample=dummy_timestep,
            sample_batch_size=config.agent.batch_size,
            num_envs=config.exp.num_envs,
            episode_length=config.env.episode_length,
        )
    )
    buffer_state = jax.jit(replay_buffer.init)(key)

    if getattr(env, "is_jumanji", False):
        ex_observations = dummy_timestep.grid[None, ...]
        ex_goals = dummy_timestep.goal[None, ...]
    else:
        ex_observations = encode_grid_inputs(dummy_timestep.grid[None, ...], config.exp.input_representation)
        ex_goals = encode_grid_inputs(dummy_timestep.grid[None, ...], config.exp.input_representation)

    example_batch = {
        "observations": ex_observations,
        "next_observations": ex_observations,
        "actions": jnp.ones((1,), dtype=jnp.int32) * (action_dim - 1),
        "rewards": jnp.ones((1,), dtype=jnp.float32),
        "masks": jnp.ones((1,), dtype=jnp.float32),
        "value_goals": ex_goals,
        "actor_goals": ex_goals,
        "action_masks": jnp.ones((1, action_dim), dtype=jnp.bool_),
        "next_action_masks": jnp.ones((1, action_dim), dtype=jnp.bool_),
    }
    agent = create_agent(config.agent, example_batch, config.exp.seed)
    return replay_buffer, buffer_state, agent


def make_update_step(replay_buffer, jitted_create_batch):
    @jax.jit
    def update_step(carry, _):
        buffer_state, agent, key = carry
        key, batch_key = jax.random.split(key, 2)
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        batch = jitted_create_batch(transitions, batch_key)
        agent, update_info = agent.update(batch)
        return (buffer_state, agent, key), update_info

    return update_step


def make_train_interval(config: Config, env, replay_buffer, update_step):
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
            input_representation=config.exp.input_representation,
        )

        buffer_state = replay_buffer.insert(buffer_state, timesteps)
        (buffer_state, agent, _), _ = jax.lax.scan(
            update_step, (buffer_state, agent, up_key), None, length=config.exp.updates_per_rollout
        )
        return buffer_state, agent, key

    return train_interval


def make_train_epoch(config: Config, train_interval):
    @jax.jit
    def train_epoch(buffer_state, agent, key):
        def interval_step(carry, _):
            next_buffer_state, next_agent, next_key = train_interval(*carry)
            return (next_buffer_state, next_agent, next_key), None

        (buffer_state, agent, key), _ = jax.lax.scan(
            interval_step,
            (buffer_state, agent, key),
            None,
            length=config.exp.intervals_per_epoch,
        )
        return buffer_state, agent, key

    return train_epoch


def train(config: Config):
    run_directory = init_wandb_and_run_directory(config)

    # Create environment and training utilities
    key = random.PRNGKey(config.exp.seed)
    env = build_training_env(config)
    jitted_create_batch = build_jitted_create_batch(config)
    replay_buffer, buffer_state, agent = build_replay_buffer_and_agent(config, env, key)
    update_step = make_update_step(replay_buffer, jitted_create_batch)
    train_interval = make_train_interval(config, env, replay_buffer, update_step)
    train_epoch = make_train_epoch(config, train_interval)

    # Main training loop with evaluation
    eval_info = evaluate_agent(agent, key, jitted_create_batch, 0, config)
    append_metrics_jsonl(run_directory, eval_info)
    save_agent(agent, config, save_dir=run_directory, epoch=0)

    for epoch in range(config.exp.epochs):
        buffer_state, agent, key = train_epoch(buffer_state, agent, key)

        epoch_num = epoch + 1
        eval_info = evaluate_agent(agent, key, jitted_create_batch, epoch_num, config)
        append_metrics_jsonl(run_directory, eval_info)
        save_interval = config.exp.save_every
        if epoch_num == config.exp.epochs or (save_interval > 0 and epoch_num % save_interval == 0):
            save_agent(agent, config, save_dir=run_directory, epoch=epoch_num)


if __name__ == "__main__":
    args = tyro.cli(Config, config=(tyro.conf.ConsolidateSubcommandArgs,))
    if args.agent.batch_size > args.exp.num_envs:
        raise ValueError("Batch size has to be less than or equal to number of environments")
    train(args)
