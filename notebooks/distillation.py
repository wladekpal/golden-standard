# %%
import os
import sys
sys.path.append("/home/mbortkie/repos/crl_subgoal/src")
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# %%
import functools
import os
import distrax


import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Dict, Any
from dataclasses import dataclass
import chex
from flax import struct
from absl import app, flags
from ml_collections import config_flags
from impls.agents import agents
from config import SRC_ROOT_DIR
from envs.block_moving_env import *
from train import *
from impls.utils.checkpoints import restore_agent, save_agent
from config import Config, ExpConfig
from envs import legal_envs
import matplotlib.pyplot as plt
from impls.utils.networks import GCDiscreteActor
import copy
import numpy as np


def reset_actor(agent, seed, ex_observations, ex_goals):
    actor_def = GCDiscreteActor(
                hidden_dims=config.agent['actor_hidden_dims'],
                action_dim=6,
                gc_encoder=None
            )
    actor_args = (ex_observations, ex_goals)
    actor_init_rng = jax.random.PRNGKey(seed)

    new_actor_params = actor_def.init(actor_init_rng, *actor_args)['params']

    new_params = dict(agent.network.params)
    new_params['modules_actor'] = new_actor_params

    new_agent = copy.deepcopy(agent)
    new_network = new_agent.network.replace(params=new_params)
    return new_agent.replace(network=new_network)

def actor_loss(agent, batch, grad_params, rng=None):
    """Compute the actor loss (AWR or DDPG+BC)."""
    if agent.config['actor_log_q']:
        def value_transform(x):
            return jnp.log(jnp.maximum(x, 1e-6))
    else:
        def value_transform(x):
            return x

    # Maximize log Q if actor_log_q is True (which is default).
    all_actions = jnp.tile(jnp.arange(6), (batch['observations'].shape[0], 1))  # B x 6
    qs = jax.lax.stop_gradient(
        jax.vmap(agent.network.select("critic"), in_axes=(None, None, 1))(batch['observations'], batch['actor_goals'], all_actions)
    )  # 6 x 2 x B
    qs = qs.min(axis=1)  # 6 x B
    qs = value_transform(qs)
    qs = qs.transpose(1, 0)  # B x 6

    dist_q = distrax.Categorical(logits=qs / jnp.maximum(1e-6, 1))
    dist_pi = agent.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)

    if FORWARD_KL:
        actor_loss = dist_q.kl_divergence(dist_pi).mean()
    else:
        actor_loss = dist_pi.kl_divergence(dist_q).mean()

    actor_info = {
        'actor_loss': actor_loss,
    }
    return actor_loss, actor_info

@jax.jit
def total_loss(agent, batch, grad_params, rng=None):
    """Compute the total loss."""
    info = {}
    rng = rng if rng is not None else agent.rng

    rng, actor_rng = jax.random.split(rng)
    loss, actor_info = actor_loss(agent, batch, grad_params, actor_rng)
    for k, v in actor_info.items():
        info[f'actor/{k}'] = v
    return loss, info

@jax.jit
def update(agent, batch):
    """Update the agent and return a new agent with information dictionary."""
    new_rng, rng = jax.random.split(agent.rng)

    def loss_fn(grad_params):
        return total_loss(agent, batch, grad_params, rng=rng)

    new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)

    return agent.replace(network=new_network, rng=new_rng), info


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
    state, future_state, goal_index = jitted_flatten_batch(config.exp.gamma, timesteps, batch_keys)

    # Sample and concatenate batch using the new function
    state, actions, future_state, goal_index = get_single_pair_from_every_env(
        state, future_state, goal_index, double_batch_key, use_double_batch_trick=config.exp.use_double_batch_trick
    )  # state.grid is of shape (batch_size * 2, grid_size, grid_size)
    if not config.exp.use_targets:
        state = state.replace(grid=GridStatesEnum.remove_targets(state.grid))
        future_state = future_state.replace(grid=GridStatesEnum.remove_targets(future_state.grid))

    # Create valid batch
    valid_batch = {
        "observations": state.grid.reshape(state.grid.shape[0], -1),
        "next_observations": future_state.grid.reshape(future_state.grid.shape[0], -1),
        "actions": actions.squeeze(),
        "rewards": state.reward.reshape(state.reward.shape[0], -1),
        "masks": 1.0 - state.done.reshape(state.done.shape[0], -1),
        "value_goals": future_state.grid.reshape(future_state.grid.shape[0], -1),
        "actor_goals": future_state.grid.reshape(future_state.grid.shape[0], -1),
    }

    # Compute losses on example batch
    loss, loss_info = total_loss(agent, valid_batch, None)

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

    if create_gif:
        log_gif(env_eval, config.env.episode_length, prefix_gif, timesteps, state)

    return eval_info_tmp, loss_info

def eval_agent(agent, key, config, critic_temp=None, different_boxes=False):
    eval_configs = [config]
    eval_names_suff = [""]
    eval_info = {"epoch": 1}
    if different_boxes:
        for number_of_boxes in range(1, 12, 2):
            new_config = copy.deepcopy(config)
            new_config.env = dataclasses.replace(new_config.env, number_of_boxes_min=number_of_boxes, number_of_boxes_max=number_of_boxes, number_of_moving_boxes_max=number_of_boxes)
            eval_configs.append(new_config)
            eval_names_suff.append("_" + str(number_of_boxes))


    for eval_config, eval_name_suff in zip(eval_configs, eval_names_suff):
        eval_info_tmp, loss_info = evaluate_agent_in_specific_env(agent, key, jitted_flatten_batch, eval_config, eval_name_suff ,create_gif=False, critic_temp=critic_temp)
        eval_info.update(eval_info_tmp)

        if eval_name_suff == "":
            eval_info.update(loss_info)
    return eval_info


# %%
EPISODE_LENGTH = 100
NUM_ENVS = 1024
CHECKPOINT = 100
RUN_NAME = f"LONG_RUN_{CHECKPOINT}_ckpt_short"
MODEL_PATH = "/home/mbortkie/repos/crl_subgoal/experiments/test_generalization_sc_20250814_235903/runs/long_unbugged_check_moving_boxes_5_grid_5_range_3_7_alpha_0.1"
EPOCHS = 101
EVAL_EVERY = 10
FIGURES_PATH = f"/home/mbortkie/repos/crl_subgoal/notebooks/figures/{RUN_NAME}"
os.makedirs(FIGURES_PATH, exist_ok=True)

# %%
config = Config(
    exp=ExpConfig(seed=0, name="test"),
    env=BoxPushingConfig(
        grid_size=5,
        number_of_boxes_min=3,
        number_of_boxes_max=7,
        number_of_moving_boxes_max=5
    )
)

# %%
env = create_env(config.env)
env = AutoResetWrapper(env)
key = random.PRNGKey(config.exp.seed)
env.step = jax.jit(jax.vmap(env.step))
env.reset = jax.jit(jax.vmap(env.reset))
partial_flatten = functools.partial(flatten_batch, get_next_obs=config.agent.use_next_obs)
jitted_flatten_batch = jax.jit(jax.vmap(partial_flatten, in_axes=(None, 0, 0)), static_argnums=(0,))
dummy_timestep = env.get_dummy_timestep(key)

for RANDOM_GOALS in [True, False]:
    for FORWARD_KL in [True, False]:
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
            'observations':dummy_timestep.grid.reshape(1, -1),  # Add batch dimension 
            'next_observations': dummy_timestep.grid.reshape(1, -1),
            'actions': jnp.ones((1,), dtype=jnp.int8) * (env._env.action_space-1), # TODO: make sure it should be the maximal value of action space  # Single action for batch size 1
            'rewards': dummy_timestep.reward.reshape(1, -1),
            'masks': 1.0 - dummy_timestep.reward.reshape(1, -1), 
            'value_goals': dummy_timestep.grid.reshape(1, -1),
            'actor_goals': dummy_timestep.grid.reshape(1, -1),
        }

        # %%
        agent, config = restore_agent(example_batch, MODEL_PATH, CHECKPOINT)

        # %%
        keys = random.split(random.PRNGKey(0), NUM_ENVS)
        state, info = env.reset(keys)

        # %%
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

        # %%
        data_key = random.PRNGKey(0)
        _, _, timesteps = collect_data(agent, data_key, env, config.exp.num_envs, config.env.episode_length, use_targets=config.exp.use_targets)
        buffer_state = replay_buffer.insert(buffer_state, timesteps)

        # %%
        ex_goals = dummy_timestep.grid.reshape(1, -1)
        ex_observations = dummy_timestep.grid.reshape(1, -1)
        agent_new = reset_actor(agent, seed=0, ex_observations=ex_observations, ex_goals=ex_goals)

        #%%
        @jax.jit
        def make_batch(buffer_state, key):
            key, batch_key, double_batch_key = jax.random.split(key, 3)
            # Sample and process transitions
            buffer_state, transitions = replay_buffer.sample(buffer_state)
            batch_keys = jax.random.split(batch_key, transitions.grid.shape[0])
            state, future_state, goal_index = jitted_flatten_batch(config.exp.gamma, transitions, batch_keys)

            state, actions, future_state, goal_index = get_single_pair_from_every_env(state, future_state, goal_index, double_batch_key, use_double_batch_trick=config.exp.use_double_batch_trick)
            if not config.exp.use_targets:
                state = state.replace(grid=GridStatesEnum.remove_targets(state.grid))
                future_state = future_state.replace(grid=GridStatesEnum.remove_targets(future_state.grid))
            # Create valid batch
            batch = {
                'observations': state.grid.reshape(state.grid.shape[0], -1),
                'next_observations': future_state.grid.reshape(future_state.grid.shape[0], -1),
                'actions': actions.squeeze(),
                'rewards': state.reward.reshape(state.reward.shape[0], -1),
                'masks': 1.0 - state.reward.reshape(state.reward.shape[0], -1), # TODO: add success and reward separately
                'value_goals': future_state.grid.reshape(future_state.grid.shape[0], -1),
                'actor_goals': jnp.roll(future_state.grid.reshape(future_state.grid.shape[0], -1), shift=1, axis=0) if RANDOM_GOALS else future_state.grid.reshape(future_state.grid.shape[0], -1),
            }
            return buffer_state, batch

        @jax.jit
        def update_step(carry, _):
            buffer_state, agent, key = carry
            key, batch_key = jax.random.split(key, 2)
            buffer_state, batch = make_batch(buffer_state, batch_key)
            agent, update_info = update(agent, batch)
            return (buffer_state, agent, key), update_info

        @jax.jit
        def train_epoch(carry, _):
            buffer_state, agent, key = carry
            key, data_key, up_key = jax.random.split(key, 3)
            _, _, timesteps = collect_data(agent, data_key, env, config.exp.num_envs, config.env.episode_length, use_targets=config.exp.use_targets, critic_temp=1)
            buffer_state = replay_buffer.insert(buffer_state, timesteps)
            (buffer_state, agent, _), _ = jax.lax.scan(update_step, (buffer_state, agent, up_key), None, length=1000)
            return (buffer_state, agent, key), None


        @functools.partial(jax.jit, static_argnums=(3,))
        def train_n_epochs(buffer_state, agent, key, epochs=10):
            (buffer_state, agent, key), _ = jax.lax.scan(
                train_epoch,
                (buffer_state, agent, key),
                None,
                length=epochs,
            )
            return buffer_state, agent, key

        # %%
        eval_info_critic = eval_agent(agent, key, config, critic_temp=1)


        config = Config(
            exp=ExpConfig(seed=0, name="test"),
            env=BoxPushingConfig(
                grid_size=5,
                number_of_boxes_min=3,
                number_of_boxes_max=7,
                number_of_moving_boxes_max=5
            )
        )
        agent_new = reset_actor(agent, seed=0, ex_observations=ex_observations, ex_goals=ex_goals)

        # %% [markdown]
        # ####  Actual Distillation

        eval_infos = []

        eval_info = eval_agent(agent_new, key, config)
        eval_infos.append(eval_info)
        print(f"Mean reward: {eval_info['eval/mean_reward']:.2f}, actor loss: {eval_info['actor/actor_loss']:.2f}")

        for i in range(EPOCHS):
            key, new_key = jax.random.split(key, 2)
            buffer_state, agent_new, key = train_n_epochs(buffer_state, agent_new, key)
            if i%EVAL_EVERY==0 and i > 0:
                eval_info = eval_agent(agent_new, key, config)
                eval_infos.append(eval_info)
                print(f"Mean reward: {eval_info['eval/mean_reward']:.2f}, actor loss: {eval_info['actor/actor_loss']:.2f}")


        # %% Mean rewards plot
        mean_rewards = [info['eval/mean_reward'] for info in eval_infos]
        x_axis = jnp.linspace(0, EPOCHS*10_000, len(mean_rewards))
        plt.plot(x_axis, mean_rewards, label='Actor distilled')
        plt.hlines(eval_info_critic['eval/mean_reward'], xmin=x_axis.min(), xmax=x_axis.max(), colors='r', linestyles='dashed', label="Softmax(Q)")
        plt.legend()
        plt.ylabel('Mean reward')
        plt.xlabel('Training steps')
        plt.title(f'Mean reward: : {"forward" if FORWARD_KL else "backward"} KL, {"random" if RANDOM_GOALS else "future"} goals')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_PATH, f'mean_reward_distillation_training_KL_{"forward" if FORWARD_KL else "backward"}_{"random" if RANDOM_GOALS else "future"}_goals.png'))
        plt.close()
        # %% actor loss plot
        actor_losses = [info['actor/actor_loss'] for info in eval_infos]
        x_axis = jnp.linspace(0, EPOCHS*10_000, len(actor_losses))
        plt.plot(x_axis, actor_losses, label='Actor distilled')
        plt.legend()
        plt.ylabel('Actor loss')
        plt.xlabel('Training steps')
        plt.title(f'Actor loss: {"forward" if FORWARD_KL else "backward"} KL, {"random" if RANDOM_GOALS else "future"} goals')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_PATH, f'actor_loss_distillation_training_KL_{"forward" if FORWARD_KL else "backward"}_{"random" if RANDOM_GOALS else "future"}_goals.png'))
        plt.close()

        # %% [markdown]
        # ### Generalization tests

        # %%
        eval_general_actor = eval_agent(agent_new, key, config, None, True)
        eval_general_q =  eval_agent(agent_new, key, config, 1, True)

        mean_reward_general_actor = [eval_general_actor[f'eval_{i}/mean_reward'] for i in range(1,12,2)]
        mean_reward_general_q = [eval_general_q[f'eval_{i}/mean_reward'] for i in range(1,12,2)]

        x = np.array(range(1, 12, 2))
        width = 0.35  # bar width

        fig, ax = plt.subplots(figsize=(8,5))

        # Shift the bars so they don’t overlap
        ax.bar(x - width/2, mean_reward_general_actor, width, label='Actor distilled', alpha=0.8, color='tab:blue')
        ax.bar(x + width/2, mean_reward_general_q, width, label='Softmax(Q)', alpha=0.8, color='tab:orange')

        ax.set_xlabel('Number of boxes')
        ax.set_ylabel('Mean Reward')
        ax.set_title(f'Generalization: : {"forward" if FORWARD_KL else "backward"} KL, {"random" if RANDOM_GOALS else "future"} goals')
        ax.set_xticks(x)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_PATH, f'actor_distillation_generalization_KL_{"forward" if FORWARD_KL else "backward"}_{"random" if RANDOM_GOALS else "future"}_goals.png'))
        plt.close()

