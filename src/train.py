import tyro
from config import Config
from envs import create_env
import functools

import wandb

from rb import TrajectoryUniformSamplingQueue, jit_wrap, segment_ids_per_row, flatten_batch

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib

from impls.agents import create_agent
from envs.block_moving_env import AutoResetWrapper, TimeStep


@functools.partial(jax.jit, static_argnums=(2,3,4))
def collect_data(agent, key, env, num_envs, episode_length):
    def step_fn(carry, step_num):
        state, info, key = carry
        key, sample_key = jax.random.split(key)
        actions = agent.sample_actions(state.grid.reshape(num_envs, -1), state.goal.reshape(num_envs, -1), seed=sample_key)
        new_state, reward, done, info = env.step(state, actions)
        timestep = TimeStep(
            key=state.key,
            grid=state.grid,
            target_cells=state.target_cells,
            agent_pos=state.agent_pos,
            agent_has_box=state.agent_has_box,
            steps=state.steps,
            action=actions,
            goal=state.goal,
            reward=reward,
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


@jax.jit
def apply_double_batch_trick(state, future_state, goal_index, key):
    """Sample two random indices and concatenate the results."""
    # Sample two random indices for each batch
    subkey1, subkey2 = jax.random.split(key, 2)
    random_indices1 = jax.random.randint(subkey1, (state.grid.shape[0],), minval=0, maxval=state.grid.shape[1])
    random_indices2 = jax.random.randint(subkey2, (state.grid.shape[0],), minval=0, maxval=state.grid.shape[1])

    state1 = extract_at_indices(state, random_indices1)
    state2 = extract_at_indices(state, random_indices2)
    future_state1 = extract_at_indices(future_state, random_indices1)
    future_state2 = extract_at_indices(future_state, random_indices2)
    goal_index1 = extract_at_indices(goal_index, random_indices1)
    goal_index2 = extract_at_indices(goal_index, random_indices2)
    
    # Concatenate the two samples
    state = jax.tree_util.tree_map(lambda x1, x2: jnp.concatenate([x1, x2], axis=0), state1, state2)
    actions = jnp.concatenate([state1.action, state2.action], axis=0)
    future_state = jax.tree_util.tree_map(lambda x1, x2: jnp.concatenate([x1, x2], axis=0), future_state1, future_state2)
    goal_index = jnp.concatenate([goal_index1, goal_index2], axis=0)
    
    return state, actions, future_state, goal_index

def evaluate_agent(agent, env, key, jitted_flatten_batch, epoch, num_envs=1024, episode_length=100):
    """Evaluate agent by running rollouts using collect_data and computing losses."""
    key, data_key, double_batch_key = jax.random.split(key, 3)
    # Use collect_data for evaluation rollouts
    _, info, timesteps = collect_data(agent, data_key, env, num_envs, episode_length)
    timesteps = jax.tree_util.tree_map(lambda x: x.swapaxes(1, 0), timesteps)

    batch_keys = jax.random.split(data_key, num_envs)
    state, future_state, goal_index = jitted_flatten_batch(0.99, timesteps, batch_keys)
    
    # Sample and concatenate batch using the new function
    state, actions, future_state, goal_index = apply_double_batch_trick(state, future_state, goal_index, double_batch_key)
    
    # Create valid batch
    valid_batch = {
        'observations': state.grid.reshape(state.grid.shape[0], -1),
        'actions': actions.squeeze(),
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
        'eval/total_loss': loss,
        'eval/mean_boxes_on_target': info['boxes_on_target'].mean()
    }
    eval_info.update(loss_info)
    wandb.log(eval_info)

    # Create figure for GIF
    grid_size = state.grid.shape[-2:]
    fig, ax = plt.subplots(figsize=grid_size)
    
    animate = functools.partial(env.animate, ax, timesteps)
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=100, interval=80, repeat=False)
    
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
    jitted_flatten_batch = jax.jit(jax.vmap(flatten_batch, in_axes=(None, 0, 0)), static_argnums=(0,))

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
        'actions': jnp.ones((1,), dtype=jnp.int8) * (env._env.action_space-1), # TODO: make sure it should be the maximal value of action space  # Single action for batch size 1
        'value_goals': dummy_timestep.grid.reshape(1, -1),
        'actor_goals': dummy_timestep.grid.reshape(1, -1),
    }

    agent = create_agent(config.agent, example_batch, config.exp.seed)


    @jax.jit
    def update_step(carry, _):
        buffer_state, agent, key = carry
        key, batch_key, double_batch_key = jax.random.split(key, 3)
        # Sample and process transitions
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        batch_keys = jax.random.split(batch_key, transitions.grid.shape[0])
        state, future_state, goal_index = jitted_flatten_batch(config.agent.discount, transitions, batch_keys)

        state, actions, future_state, goal_index = apply_double_batch_trick(state, future_state, goal_index, double_batch_key)
        # Create valid batch
        valid_batch = {
            'observations': state.grid.reshape(state.grid.shape[0], -1),
            'actions': actions.squeeze(),
            'value_goals': future_state.grid.reshape(future_state.grid.shape[0], -1),
            'actor_goals': future_state.grid.reshape(future_state.grid.shape[0], -1),
        }

        # Update agent
        agent, update_info = agent.update(valid_batch)
        return (buffer_state, agent, key), update_info
    
    @jax.jit
    def train_epoch(carry, _):
        buffer_state, agent, key = carry
        key, data_key, up_key = jax.random.split(key, 3)
        _, _, timesteps = collect_data(agent, data_key, env, config.exp.num_envs, config.env.episode_length)
        buffer_state = replay_buffer.insert(buffer_state, timesteps)
        (buffer_state, agent, _), _ = jax.lax.scan(update_step, (buffer_state, agent, up_key), None, length=1000)
        return (buffer_state, agent, key), None

    @jax.jit
    def train_n_epochs(buffer_state, agent, key):
        (buffer_state, agent, key), _ = jax.lax.scan(
            train_epoch,
            (buffer_state, agent, key),
            None,
            length=10,
        )
        return buffer_state, agent, key
    

    for epoch in range(config.exp.epochs):
        evaluate_agent(agent, env, key, jitted_flatten_batch, epoch, config.exp.num_envs, config.env.episode_length)
        for _ in range(10):
            buffer_state, agent, key = train_n_epochs(buffer_state, agent, key)

        evaluate_agent(agent, env, key, jitted_flatten_batch, epoch, config.exp.num_envs, config.env.episode_length)


if __name__ == "__main__":
    args = tyro.cli(Config, config=(tyro.conf.ConsolidateSubcommandArgs,))
    train(args)
