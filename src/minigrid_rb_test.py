import os

from matplotlib import pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ['WANDB_MODE'] = 'offline'

import jax
import jax.numpy as jnp
import xminigrid
import wandb
from minigrid_data_collection import TimeStepNew, build_benchmark, collect_data, get_concatenated_state, repeat_tree
from absl import app, flags
from ml_collections import config_flags
from xminigrid.core.constants import NUM_ACTIONS
from impls.agents import agents
from rb import TrajectoryUniformSamplingQueue, flatten_batch, jit_wrap
from config import SRC_ROOT_DIR




FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 0, 'Random seed.')

config_flags.DEFINE_config_file('agent', SRC_ROOT_DIR + '/agents/crl.py', lock_config=False)

def main(_):
    # Wandb
    wandb.init(project="xminigrid-crl_full_state", name="fixed_index_test", config=FLAGS)
    # Environment parameters
    VIEW_SIZE = 3
    BATCH_SIZE = 512
    NUM_ENVS = 256
    MAX_REPLAY_SIZE = 10000
    EPISODE_LENGTH = 100
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
    exemplary_goal = timestep.replace(state=timestep.state.replace(agent=timestep.state.agent.replace(position=jnp.array([6,6]))))
    exemplary_goals = repeat_tree(exemplary_goal, 256)


    # Create a replay buffer (it stores full TimeStepNew objects)
    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=MAX_REPLAY_SIZE,
            dummy_data_sample=timestep,
            sample_batch_size=BATCH_SIZE,
            num_envs=NUM_ENVS,
            episode_length=EPISODE_LENGTH,
        )
    )
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    # Create a benchmark function and get a batch of timesteps
    benchmark_fn = build_benchmark(env_name, NUM_ENVS, EPISODE_LENGTH, view_size=VIEW_SIZE)
    env_step, timesteps_all = benchmark_fn(key)

    # RB operations
    buffer_state = replay_buffer.insert(buffer_state, timesteps_all)
    buffer_state, transitions = replay_buffer.sample(buffer_state) 

    # Process transitions for training
    batch_keys = jax.random.split(buffer_state.key, transitions.observation.shape[0])
    # state, future_state, goal_index = jax.vmap(flatten_batch, in_axes=(None, 0, 0))((0.99, None, None), transitions, batch_keys)
    # state, future_state, goal_index = state[:,0].reshape(state.shape[0], -1), future_state[:,0].reshape(future_state.shape[0]), goal_index[:,0].reshape(goal_index.shape[0], -1) 

    config = FLAGS.agent
    config['discrete'] = True
    agent_class = agents[config['agent_name']]

    example_batch = {
        'observations':get_concatenated_state(timestep) ,  # Add batch dimension 
        'actions': jnp.ones((1,), dtype=jnp.int32) * (NUM_ACTIONS-1), # TODO: make sure it should be the maximal value of action space  # Single action for batch size 1
        'value_goals': get_concatenated_state(timestep) ,
        'actor_goals': get_concatenated_state(timestep) ,
        'terminals': timestep.step_type.reshape(1,-1) == xminigrid.types.StepType.LAST,
    }

    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
        example_batch['value_goals'],
    )

    collect_data_fn = collect_data(env_name, NUM_ENVS, exemplary_goals, EPISODE_LENGTH, view_size=VIEW_SIZE)


    for epoch in range(1000):
        key, new_key = jax.random.split(key)
        # get new data
        env_step, timesteps_all = collect_data_fn(agent, new_key)

        # insert data into buffer
        buffer_state = replay_buffer.insert(buffer_state, timesteps_all)

        def update_step(carry, _):
            buffer_state, agent, key = carry
            
            # Sample and process transitions
            buffer_state, transitions = replay_buffer.sample(buffer_state)
            batch_keys = jax.random.split(buffer_state.key, transitions.observation.shape[0])
            state, future_state, goal_index = jax.vmap(flatten_batch, in_axes=(None, 0, 0))((0.99, None, None), transitions, batch_keys)

            # Get random index for each batch
            key, subkey = jax.random.split(key)
            random_indices = jax.random.randint(subkey, (state.observation.shape[0],), minval=0, maxval=state.observation.shape[1])
            
            # Extract data at random index
            state = jax.tree_util.tree_map(lambda x: x[jnp.arange(x.shape[0]), random_indices], state)
            future_state = jax.tree_util.tree_map(lambda x: x[jnp.arange(x.shape[0]), random_indices], future_state)
            goal_index = jax.tree_util.tree_map(lambda x: x[jnp.arange(x.shape[0]), random_indices], goal_index)
            actions = state.action

            # Create valid batch
            valid_batch = {
                'observations': get_concatenated_state(state),
                'actions': actions.squeeze(),
                'value_goals': get_concatenated_state(future_state),
                'actor_goals': get_concatenated_state(future_state),
            }

            # Update agent
            agent, update_info = agent.update(valid_batch)
            update_info.update({
                "eval/reward_min": timesteps_all.reward.min(),
                "eval/reward_max": timesteps_all.reward.max(), 
                "eval/reward_mean": timesteps_all.reward.mean()
            })

            return (buffer_state, agent, key), update_info

        # Run scan for updates
        (buffer_state, agent, key), update_infos = jax.lax.scan(
            update_step,
            (buffer_state, agent, key),
            None,
            length=1000
        )

        print(f"update_infos: {update_infos}")
        wandb.log(jax.tree_util.tree_map(lambda x: x[-1], update_infos))



if __name__ == "__main__":
    app.run(main)