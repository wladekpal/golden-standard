import tyro
from config import Config
from envs import create_env
import functools
import os
import wandb

from rb import TrajectoryUniformSamplingQueue, jit_wrap, flatten_batch

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib

from impls.agents import create_agent
from envs.block_moving_env import AutoResetWrapper, TimeStep, GridStatesEnum
from config import ROOT_DIR


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def collect_data(agent, key, env, num_envs, episode_length, use_targets=False):
    def step_fn(carry, step_num):
        state, info, key = carry
        key, sample_key = jax.random.split(key)

        # Use jax.lax.cond instead of if statement to handle traced arrays
        state_agent = jax.lax.cond(
            use_targets,
            lambda: state.replace(),
            lambda: state.replace(
                grid=GridStatesEnum.remove_targets(state.grid),
                goal=GridStatesEnum.remove_targets(state.goal)
            )
        )

        actions = agent.sample_actions(state_agent.grid.reshape(num_envs, -1), state_agent.goal.reshape(num_envs, -1),
                                       seed=sample_key)
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



@functools.partial(jax.jit, static_argnums=(4,))
def get_single_pair_from_every_env(state, future_state, goal_index, key, use_double_batch_trick=False):
    """Sample two random indices and concatenate the results."""
    # Sample two random indices for each batch
    def double_batch_fn(key):
        subkey1, subkey2, subkey3 = jax.random.split(key, 3)
        random_indices1 = jax.random.randint(subkey1, (state.grid.shape[0],), minval=0, maxval=state.grid.shape[1])
        random_indices2 = jax.random.randint(subkey2, (state.grid.shape[0],), minval=0, maxval=state.grid.shape[1])

        state1 = extract_at_indices(state, random_indices1) 
        state2 = extract_at_indices(state, random_indices2)
        future_state1 = extract_at_indices(future_state, random_indices1)
        future_state2 = extract_at_indices(future_state, random_indices2)
        goal_index1 = extract_at_indices(goal_index, random_indices1)
        goal_index2 = extract_at_indices(goal_index, random_indices2)

        envs_to_take = jax.random.randint(subkey3, (state.grid.shape[0]//2,), minval=0, maxval=state.grid.shape[0])
        state1 = jax.tree_util.tree_map(lambda x: x[envs_to_take], state1)
        state2 = jax.tree_util.tree_map(lambda x: x[envs_to_take], state2)
        future_state1 = jax.tree_util.tree_map(lambda x: x[envs_to_take], future_state1)
        future_state2 = jax.tree_util.tree_map(lambda x: x[envs_to_take], future_state2)
        goal_index1 = jax.tree_util.tree_map(lambda x: x[envs_to_take], goal_index1)
        goal_index2 = jax.tree_util.tree_map(lambda x: x[envs_to_take], goal_index2)
        
        # Concatenate the two samples
        state_concat = jax.tree_util.tree_map(lambda x1, x2: jnp.concatenate([x1, x2], axis=0), state1, state2) # (batch_size, grid_size, grid_size)
        actions = jnp.concatenate([state1.action, state2.action], axis=0)
        future_state_concat = jax.tree_util.tree_map(lambda x1, x2: jnp.concatenate([x1, x2], axis=0), future_state1, future_state2)
        goal_index_concat = jnp.concatenate([goal_index1, goal_index2], axis=0)
        
        return state_concat, actions, future_state_concat, goal_index_concat
    
    def single_batch_fn(key):
        random_indices = jax.random.randint(key, (state.grid.shape[0],), minval=0, maxval=state.grid.shape[1])
        state_single = extract_at_indices(state, random_indices) # (batch_size, grid_size, grid_size)
        future_state_single = extract_at_indices(future_state, random_indices)
        goal_index_single = extract_at_indices(goal_index, random_indices)
        return state_single, state_single.action, future_state_single, goal_index_single
    
    return jax.lax.cond(
        use_double_batch_trick,
        double_batch_fn,
        single_batch_fn,
        key
    )


def evaluate_agent(agent, env, key, jitted_flatten_batch, epoch, num_envs=1024, episode_length=100, use_double_batch_trick=False, gamma=0.99, use_targets=False):
    """Evaluate agent by running rollouts using collect_data and computing losses."""
    key, data_key, double_batch_key = jax.random.split(key, 3)
    # Use collect_data for evaluation rollouts
    _, info, timesteps = collect_data(agent, data_key, env, num_envs, episode_length, use_targets=use_targets)
    timesteps = jax.tree_util.tree_map(lambda x: x.swapaxes(1, 0), timesteps)

    batch_keys = jax.random.split(data_key, num_envs)
    state, future_state, goal_index = jitted_flatten_batch(gamma, timesteps, batch_keys)
    
    # Sample and concatenate batch using the new function
    state, actions, future_state, goal_index = get_single_pair_from_every_env(state, future_state, goal_index, double_batch_key, use_double_batch_trick=use_double_batch_trick) # state.grid is of shape (batch_size * 2, grid_size, grid_size)
    if not use_targets:
        state = state.replace(grid=GridStatesEnum.remove_targets(state.grid))
        future_state = future_state.replace(grid=GridStatesEnum.remove_targets(future_state.grid))

    # Create valid batch
    valid_batch = {
        'observations': state.grid.reshape(state.grid.shape[0], -1),
        'next_observations': future_state.grid.reshape(future_state.grid.shape[0], -1),
        'actions': actions.squeeze(),
        'rewards': state.reward.reshape(state.reward.shape[0], -1),
        'masks': 1.0 - state.reward.reshape(state.reward.shape[0], -1), # TODO: add success and reward separately
        'value_goals': future_state.grid.reshape(future_state.grid.shape[0], -1),
        'actor_goals': future_state.grid.reshape(future_state.grid.shape[0], -1),
    }

    # Compute losses on example batch
    loss, loss_info = agent.total_loss(valid_batch, None)
    
    # Compile evaluation info
    # Only consider episodes that are done
    done_mask = timesteps.done
    eval_info = {
        'eval/mean_reward': timesteps.reward[done_mask].mean(),
        'eval/min_reward': timesteps.reward[done_mask].min(),
        'eval/max_reward': timesteps.reward[done_mask].max(),
        'eval/mean_success': timesteps.success[done_mask].mean(),
        'eval/total_loss': loss,
        'eval/mean_boxes_on_target': info['boxes_on_target'].mean()
    }
    eval_info.update(loss_info)
    wandb.log(eval_info)

    # Create figure for GIF
    grid_size = state.grid.shape[-2:]
    fig, ax = plt.subplots(figsize=grid_size)
    
    animate = functools.partial(env.animate, ax, timesteps, img_prefix=os.path.join(ROOT_DIR, 'assets'))
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=episode_length, interval=80, repeat=False)
    
    # Save as GIF
    gif_path = f"/tmp/block_moving_epoch_{epoch}.gif"
    anim.save(gif_path, writer='pillow')
    plt.close()

    wandb.log({"gif": wandb.Video(gif_path, format="gif")})


def train(config: Config):
    wandb.init(
        project=config.exp.project,
        name=config.exp.name,
        config=config,
        entity=None,
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
        'observations':dummy_timestep.grid.reshape(1, -1),  # Add batch dimension 
        'next_observations': dummy_timestep.grid.reshape(1, -1),
        'actions': jnp.ones((1,), dtype=jnp.int8) * (env._env.action_space-1), # TODO: make sure it should be the maximal value of action space  # Single action for batch size 1
        'rewards': dummy_timestep.reward.reshape(1, -1),
        'masks': 1.0 - dummy_timestep.reward.reshape(1, -1), # TODO: add success and reward separately
        'value_goals': dummy_timestep.grid.reshape(1, -1),
        'actor_goals': dummy_timestep.grid.reshape(1, -1),
    }

    agent = create_agent(config.agent, example_batch, config.exp.seed)


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
            'actor_goals': future_state.grid.reshape(future_state.grid.shape[0], -1),
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
    def train_epoch(carry, _):
        buffer_state, agent, key = carry
        key, data_key, up_key = jax.random.split(key, 3)
        _, _, timesteps = collect_data(agent, data_key, env, config.exp.num_envs, config.env.episode_length, use_targets=config.exp.use_targets)
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

    # Evaluate before training
    evaluate_agent(agent, env, key, jitted_flatten_batch, 0, config.exp.num_envs, config.env.episode_length, config.exp.use_double_batch_trick, config.exp.gamma, config.exp.use_targets)
    
    for epoch in range(config.exp.epochs):
        for _ in range(10):
            buffer_state, agent, key = train_n_epochs(buffer_state, agent, key)

        evaluate_agent(agent, env, key, jitted_flatten_batch, epoch+1, config.exp.num_envs, config.env.episode_length,  config.exp.use_double_batch_trick, config.exp.gamma, config.exp.use_targets)


if __name__ == "__main__":
    args = tyro.cli(Config, config=(tyro.conf.ConsolidateSubcommandArgs,))
    train(args)
