import os
import functools

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import jax
import flax.struct
import jax.numpy as jnp
from jax import flatten_util
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper
from xminigrid.types import TimeStep
from xminigrid.experimental.img_obs import RGBImgObservationWrapper




def build_benchmark(env_id, num_envs, timesteps, view_size=3):
    env, env_params = xminigrid.make(env_id)
    env_params = env_params.replace(view_size=view_size)
    
    env = GymAutoResetWrapper(env)
    
    def benchmark_fn(key):
        def _step_fn(timestep, action):
            new_timestep = jax.vmap(env.step, in_axes=(None, 0, 0))(env_params, timestep, action)
            return new_timestep, TimeStep(
                observation=timestep.observation,
                reward=timestep.reward,
                discount=timestep.discount,
                step_type=timestep.step_type,
                state=timestep.state,
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


# benchmark_fn = build_benchmark('MiniGrid-EmptyRandom-5x5', 1024, 3)

# # TypeError: Cannot determine dtype of key<fry> while using key = jax.random.key(0)
# key = jax.random.PRNGKey(0)
# env_step, timesteps_all = benchmark_fn(key)

# print(env_step.observation.shape)
# print(timesteps_all.observation.shape)