import jax
import jax.numpy as jnp
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper
from xminigrid.types import TimeStep

def repeat_tree(tree, n: int):
    """Replicate every leaf `n` times on a new leading axis."""
    return jax.tree.map(
        lambda x: jnp.broadcast_to(x, (n,) + x.shape),  # cheap: stride-0 view
        tree,
    )

def get_concatenated_state(timestep):
    @jax.jit
    def _ravel_one(sample_tree):
        flat, _ = jax.flatten_util.ravel_pytree(sample_tree)   # 1-D feature vector
        return flat                           # shape (F,)

    if timestep.state.grid.ndim == 3:
        grid_state = timestep.state.grid.reshape(-1, timestep.state.grid.size)
        agent_state = jax.flatten_util.ravel_pytree(timestep.state.agent)[0].reshape(1, -1)
        return jnp.concatenate([grid_state, agent_state, timestep.state.step_num.reshape((-1, 1))], axis=1)
    elif timestep.state.grid.ndim == 4:
        grid_state = jax.tree_util.tree_map(lambda x: x.reshape(x.shape[0], x[0].size), timestep.state.grid)
        print(f"grid_state.shape: {grid_state.shape}")
        agent_state = jax.vmap(_ravel_one)(timestep.state.agent)
        print(f"agent_state.shape: {agent_state.shape}")
        return jnp.concatenate([grid_state, agent_state, timestep.state.step_num.reshape((-1, 1))], axis=1)


class TimeStepNew(TimeStep):
    action: jax.Array

    

def build_benchmark(env_id, num_envs, timesteps, view_size=3):
    env, env_params = xminigrid.make(env_id)
    env_params = env_params.replace(view_size=view_size)
    
    env = GymAutoResetWrapper(env)
    
    def benchmark_fn(key):
        def _step_fn(timestep, action):
            new_timestep = jax.vmap(env.step, in_axes=(None, 0, 0))(env_params, timestep, action)
            return new_timestep, TimeStepNew(
                observation=timestep.observation,
                reward=timestep.reward,
                discount=timestep.discount,
                step_type=timestep.step_type,
                state=timestep.state,
                action=action,
            )

        key, actions_key = jax.random.split(key)
        keys = jax.random.split(key, num=num_envs)
        actions = jax.random.randint(
            actions_key, shape=(timesteps, num_envs), minval=0, maxval=env.num_actions(env_params)
        )
        timestep = jax.vmap(env.reset, in_axes=(None, 0))(env_params, keys)
        timestep, timesteps_all = jax.lax.scan(_step_fn, timestep, actions, unroll=1)
        return timestep, timesteps_all

    return benchmark_fn

def collect_data(env_id, num_envs, goals, timesteps, view_size=3):
    env, env_params = xminigrid.make(env_id)
    env_params = env_params.replace(view_size=view_size)
    
    env = GymAutoResetWrapper(env)
    
    def benchmark_fn(agent, key):
        def _step_fn(carry, unused):
            timestep, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            concatenated_state = get_concatenated_state(timestep)
            print(f"concatenated_state.shape: {concatenated_state.shape}")
            concatenated_goals = get_concatenated_state(goals)
            print(f"concatenated_goals.shape: {concatenated_goals.shape}")
            action = agent.sample_actions(concatenated_state, concatenated_goals, seed=current_key)
            print(f"action.shape  benchmark_fn: {action.shape}")
            new_timestep = jax.vmap(env.step, in_axes=(None, 0, 0))(env_params, timestep, action)
            return (new_timestep, next_key), TimeStepNew(
                observation=timestep.observation,
                reward=timestep.reward,
                discount=timestep.discount,
                step_type=timestep.step_type,
                state=timestep.state,
                action=action,
            )

        keys = jax.random.split(key, num=num_envs)
        timestep = jax.vmap(env.reset, in_axes=(None, 0))(env_params, keys)
        timestep, timesteps_all = jax.lax.scan(_step_fn, (timestep, key), (), length=timesteps)
        return timestep, timesteps_all

    return benchmark_fn
