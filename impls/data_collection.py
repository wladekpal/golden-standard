import jax
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper
from xminigrid.types import TimeStep

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

