import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import jax
import jax.numpy as jnp
import xminigrid
from data_collection import TimeStepNew, build_benchmark, collect_data
from absl import app, flags
from ml_collections import config_flags
from xminigrid.core.constants import NUM_ACTIONS
from agents import agents
from rb import TrajectoryUniformSamplingQueue, flatten_batch, jit_wrap
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
state, action, goal = jax.vmap(flatten_batch, in_axes=(None, 0, 0))((0.99, 171, 0), transitions, batch_keys)
print(f"state.shape: {state.shape}")
print(f"action.shape: {action.shape}")
print(f"goal.shape: {goal.shape}")

print(f"transitions.reward min, max: {transitions.reward.min()}, {transitions.reward.max()}")

# Take only first timestep (original state shape (256-BS, 9-T, 3-H, 3-W, 2-C))
state, action, goal = state[:,0].reshape(state.shape[0], -1), action[:,0].reshape(action.shape[0]), goal[:,0].reshape(goal.shape[0], -1) 

# Define the agent
FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 0, 'Random seed.')

config_flags.DEFINE_config_file('agent', ROOT_DIR + '/agents/crl.py', lock_config=False)


def main(_):
# Agent
    config = FLAGS.agent
    config['discrete'] = True
    agent_class = agents[config['agent_name']]
    example_batch = {
        'observations':timestep.observation.reshape(1, -1),  # Add batch dimension
        'actions': jnp.ones((1,), dtype=jnp.int32) * NUM_ACTIONS, # TODO: make sure it should be the maximal value of action space  # Single action for batch size 1
        # 'value_goals': timestep.state.goal_encoding.reshape(1, -1),
        # 'actor_goals': timestep.state.goal_encoding.reshape(1, -1),
        'value_goals': timestep.observation[:,timestep.observation.shape[1]//2,:].reshape(1, -1),
        'actor_goals': timestep.observation[:,timestep.observation.shape[1]//2,:].reshape(1, -1),
        'terminals': timestep.step_type.reshape(1,-1) == xminigrid.types.StepType.LAST,
    }
    print(f"example_batch['value_goals'].shape: {example_batch['value_goals'].shape}")
    print(f"timestep.observation[:,timestep.observation.shape[1]//2,:].reshape(1, -1): {timestep.observation[:,timestep.observation.shape[1]//2,:].reshape(1, -1)}")


    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
        example_batch['value_goals'],
    )

    valid_batch = {
        'observations':state,
        'actions':action,
        'value_goals':goal,
        'actor_goals':goal,
        'terminals':jnp.zeros((state.shape[0],), dtype=jnp.int32), # TODO: check if this is correct
    }
    print(f"agent.sample_actions(valid_batch['observations'], valid_batch['value_goals']): {agent.sample_actions(valid_batch['observations'], valid_batch['value_goals'], seed=key)}")
    agent, update_info = agent.update(valid_batch)
    
    print(f"update_info: {update_info}")

    exemplary_goals = jnp.array([6,2,2,6,-1,-1])
    exemplary_goals = exemplary_goals.reshape(1, -1)
    exemplary_goals = jnp.tile(exemplary_goals, (256, 1))
    print(f"exemplary_goals.shape: {exemplary_goals.shape}")

    collect_data_fn = collect_data(env_name, 256, agent, exemplary_goals, 50, view_size=VIEW_SIZE)
    env_step, timesteps_all = collect_data_fn(key)

    print(f"timesteps_all.reward min, max: {timesteps_all.reward.min()}, {timesteps_all.reward.max()}")
    # Hmm, seems like it's working.





if __name__ == "__main__":
    app.run(main)