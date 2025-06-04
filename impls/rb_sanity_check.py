#%%
import copy
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
from rb import TrajectoryUniformSamplingQueue, flatten_batch, jit_wrap
from config import ROOT_DIR

# Environment parameters
VIEW_SIZE = 3
env_name = 'MiniGrid-EmptyRandom-5x5'

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
        episode_length=20,
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
state, future_state, goal_index = jax.vmap(flatten_batch, in_axes=(None, 0, 0))((0.99, 171, 0), transitions, batch_keys)
print(f"action.shape: {state.action.shape}")
print(f"future_state.observation.shape: {future_state.observation.shape}")
print(f"goal_index.shape: {goal_index.shape}")

transition = jax.tree_util.tree_map(lambda x: x[0,0], transitions) # Take only one env and first timestep 
print(f"transition.observation.shape: {transition.observation.shape}")
plt.imshow(env.render(env_params, transition))
plt.savefig("render_transition.png")


transition_from_state = jax.tree_util.tree_map(lambda x: x[0,0], state)
print(f"transition_from_state.observation.shape: {transition_from_state.observation.shape}")
plt.imshow(env.render(env_params, transition_from_state))
plt.savefig("render_transition_from_state.png")

print(f"goal_index: {goal_index[0]}")

transition_from_future_state = jax.tree_util.tree_map(lambda x: x[0,0], future_state)
print(f"transition_from_future_state.observation.shape: {transition_from_future_state.observation.shape}")
plt.imshow(env.render(env_params, transition_from_future_state))
plt.savefig("render_transition_from_future_state.png")

transition_from_state_goal_index = jax.tree_util.tree_map(lambda x: x[0,goal_index[0,0]], state)
print(f"transition_from_state_goal_index.observation.shape: {transition_from_state_goal_index.observation.shape}")
plt.imshow(env.render(env_params, transition_from_state_goal_index))
plt.savefig("render_transition_from_state_goal_index.png")


# Create a figure with subplots for each timestep
fig, axes = plt.subplots(2, state.observation.shape[1], figsize=(20, 4))
for i in range(state.observation.shape[1]):
    transition_from_state_i = jax.tree_util.tree_map(lambda x: x[0,i], state)
    axes[0,i].imshow(env.render(env_params, transition_from_state_i))
    axes[0,i].set_title(f'Timestep {i}\nAction: {state.action[0,i]}')
    axes[0,i].axis('off')

for i in range(future_state.observation.shape[1]):
    transition_from_future_state_i = jax.tree_util.tree_map(lambda x: x[0,i], future_state)
    axes[1,i].imshow(env.render(env_params, transition_from_future_state_i))
    axes[1,i].set_title(f'Future Timestep for {i}\nTimestep {goal_index[0,i]}')
    axes[1,i].axis('off')

plt.tight_layout()
plt.savefig("render_transition_from_state_all.png")
plt.close()

print(f"transition_from_state_i.state.grid[:,:,0]: {transition_from_state_i.state.grid[:,:,0]}")
print(f"transition_from_state_i.state.grid[:,:,1]: {transition_from_state_i.state.grid[:,:,1]}")


# Create a figure with subplots for each timestep
fig, axes = plt.subplots(2, state.observation.shape[1], figsize=(20, 4))
for i in range(state.observation.shape[1]):
    transition_from_state_i = jax.tree_util.tree_map(lambda x: x[1,i], state)
    axes[0,i].imshow(env.render(env_params, transition_from_state_i))
    axes[0,i].set_title(f'Timestep {i}\nAction: {state.action[1,i]}')
    axes[0,i].axis('off')

for i in range(future_state.observation.shape[1]):
    transition_from_future_state_i = jax.tree_util.tree_map(lambda x: x[1,i], future_state)
    axes[1,i].imshow(env.render(env_params, transition_from_future_state_i))
    axes[1,i].set_title(f'Future Timestep for {i}\nTimestep {goal_index[1,i]}')
    axes[1,i].axis('off')

plt.tight_layout()
plt.savefig("render_transition_from_state_all_diff_env.png")
plt.close()

#%%
plt.imshow(env.render(env_params, transition_from_future_state_i))

transition_from_future_state_i.state.grid[:,:,0]
transition_from_future_state_i.state.grid[:,:,1]

#%%
transition_from_future_state_i.state.agent

#%%
jax.tree_util.tree_flatten(transition_from_future_state_i.state.agent)

#%%
jax.flatten_util.ravel_pytree(transition_from_future_state_i.state.agent)[0]


#%%
goal_state = copy.deepcopy(transition_from_future_state_i)
goal_state = goal_state.replace(state=goal_state.state.replace(agent=goal_state.state.agent.replace(position=jnp.array([3,3]))))

plt.imshow(env.render(env_params, goal_state))

#%%
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
timestep = timestep.replace(state=timestep.state.replace(agent=timestep.state.agent.replace(position=jnp.array([3,3]))))
plt.imshow(env.render(env_params, timestep))
#%%
# get concatenated state and agent
timestep.state.grid.shape
#%%
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
        return jnp.concatenate([grid_state, agent_state], axis=1)
    elif timestep.state.grid.ndim == 4:
        grid_state = jax.tree_util.tree_map(lambda x: x.reshape(x.shape[0], x[0].size), timestep.state.grid)
        print(f"grid_state.shape: {grid_state.shape}")
        agent_state = jax.vmap(_ravel_one)(timestep.state.agent)
        print(f"agent_state.shape: {agent_state.shape}")
        return jnp.concatenate([grid_state, agent_state], axis=1)

        
print(f"timestep.state.grid.shape: {timestep.state.grid.shape}")
concatenated_state = get_concatenated_state(timestep)
print(f"concatenated_state.shape: {concatenated_state.shape}")


# batch_timestep = jax.tree_util.tree_map(lambda x: x[None], timestep)
batch_timestep = repeat_tree(timestep, 256)
print(f"batch_timestep.observation.shape: {batch_timestep.observation.shape}")
print(f"batch_timestep.state.grid.shape: {batch_timestep.state.grid.shape}")
print(f"batch_timestep.state.agent.position.shape: {batch_timestep.state.agent.position.shape}")
concatenated_state_2 = get_concatenated_state(batch_timestep)
print(f"concatenated_state_2.shape: {concatenated_state_2.shape}")


#%%
batch_timestep.state.agent.position.shape

# print(concatenated_state_2.shape)

#%%
# jax.tree_util.tree_map(lambda x: jax.flatten_util.ravel_pytree(x)[0], batch_timestep.state.agent)
print(f"batch_timestep.state.agent.position.shape: {batch_timestep.state.agent.position.shape}")
jax.flatten_util.ravel_pytree(batch_timestep.state.agent)[0]


#%%
# ----- helper that ravels ONE sample (a single slice along axis-0) -----
def _ravel_one(sample_tree):
    flat, _ = jax.flatten_util.ravel_pytree(sample_tree)   # 1-D feature vector
    return flat                           # shape (F,)

# ----- ravel the entire agent state while keeping the batch axis intact -----
# If the leaves all have shape (B, …), the result has shape (B, F)
agent_state = jax.vmap(_ravel_one)(batch_timestep.state.agent)
print(f"agent_state.shape: {agent_state.shape}")


#%%


batch_timestep = repeat_tree(timestep, 256)
print(f"batch_timestep.state.agent.position.shape: {batch_timestep.state.agent.position.shape}")
print(f"batch_timestep.state.grid.shape: {batch_timestep.state.grid.shape}")