import os

from matplotlib import pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import jax
import jax.numpy as jnp
import xminigrid
from data_collection import TimeStepNew, build_benchmark
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from rb import TrajectoryUniformSamplingQueue, flatten_batch, flatten_batch_sanity_check, jit_wrap
from config import ROOT_DIR

# Environment parameters
VIEW_SIZE = 3
env_name = 'MiniGrid-EmptyRandom-8x8'

# Random seed TODO: Need to make sure that latter everything is correctly seeded
key = jax.random.PRNGKey(0)
buffer_key, reset_key = jax.random.split(key, 2)

# Just to create a dummy timestep
env, env_params = xminigrid.make(env_name)
env_params = env_params.replace(view_size=VIEW_SIZE)
timestep = jax.jit(env.reset)(env_params, reset_key)
timestep = TimeStepNew(
    state=timestep.state,
    step_type=timestep.step_type,
    reward=timestep.reward,
    discount=timestep.discount,
    observation=timestep.observation,
    action=jnp.zeros((1,), dtype=jnp.int32),
)
print(timestep.observation.shape)
print(timestep.state.step_num)

# Create a replay buffer
replay_buffer = jit_wrap(
    TrajectoryUniformSamplingQueue(
        max_replay_size=100,
        dummy_data_sample=timestep,
        sample_batch_size=256,
        num_envs=256,
        episode_length=10,
    )
)
buffer_state = jax.jit(replay_buffer.init)(buffer_key)
print(replay_buffer._data_shape)


# Create a benchmark function and get a batch of timesteps
benchmark_fn = build_benchmark(env_name, 256, 50, view_size=VIEW_SIZE)
env_step, timesteps_all = benchmark_fn(key)
print(f"timesteps_all.action.shape: {timesteps_all.action.shape}")

# RB operations
buffer_state = replay_buffer.insert(buffer_state, timesteps_all)
buffer_state, transitions = replay_buffer.sample(buffer_state) 

# Process transitions for training
batch_keys = jax.random.split(buffer_state.key, transitions.observation.shape[0])
print(f"batch_keys: {batch_keys.shape}")
state, action, future_state = jax.vmap(flatten_batch_sanity_check, in_axes=(None, 0, 0))((0.99, 171, 0), transitions, batch_keys)
print(f"state.shape: {state.shape}")
print(f"action.shape: {action.shape}")
print(f"future_state.shape: {future_state.shape}")

transition = jax.tree_util.tree_map(lambda x: x[0,0], transitions) # Take only one env and first timestep 
print(f"transition.observation.shape: {transition.observation.shape}")
plt.imshow(env.render(env_params, transition))
plt.savefig("render_transition.png")





