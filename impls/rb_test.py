import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import jax
import jax.numpy as jnp
import xminigrid
import wandb
from data_collection import TimeStepNew, build_benchmark, collect_data
from absl import app, flags
from ml_collections import config_flags
from xminigrid.core.constants import NUM_ACTIONS
from agents import agents
from rb import TrajectoryUniformSamplingQueue, flatten_batch, jit_wrap
from config import ROOT_DIR

FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 0, 'Random seed.')

config_flags.DEFINE_config_file('agent', ROOT_DIR + '/agents/crl.py', lock_config=False)

def main(_):
    # Wandb
    wandb.init(project="xminigrid-crl", name="rb_test", config=FLAGS)
    # Environment parameters
    VIEW_SIZE = 3
    BATCH_SIZE = 256
    NUM_ENVS = 256
    MAX_REPLAY_SIZE = 1000
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
    print(timestep.observation.shape)
    print(timestep.state.step_num)

    # Create a replay buffer
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
    print(replay_buffer._data_shape)

    # Create a benchmark function and get a batch of timesteps
    benchmark_fn = build_benchmark(env_name, NUM_ENVS, EPISODE_LENGTH, view_size=VIEW_SIZE)
    env_step, timesteps_all = benchmark_fn(key)
    print(f"timesteps_all.action.shape: {timesteps_all.action.shape}")

    # RB operations
    buffer_state = replay_buffer.insert(buffer_state, timesteps_all)
    buffer_state, transitions = replay_buffer.sample(buffer_state) 

    # Process transitions for training
    batch_keys = jax.random.split(buffer_state.key, transitions.observation.shape[0])
    print(f"batch_keys: {batch_keys.shape}")
    state, action, goal = jax.vmap(flatten_batch, in_axes=(None, 0, 0))((0.99, None, None), transitions, batch_keys)
    print(f"state.shape: {state.shape}")
    print(f"action.shape: {action.shape}")
    print(f"goal.shape: {goal.shape}")

    print(f"transitions.reward min, max: {transitions.reward.min()}, {transitions.reward.max()}")

    # Take only first timestep (original state shape (256-BS, 9-T, 3-H, 3-W, 2-C))
    state, action, goal = state[:,0].reshape(state.shape[0], -1), action[:,0].reshape(action.shape[0]), goal[:,0].reshape(goal.shape[0], -1) 

    config = FLAGS.agent
    config['discrete'] = True
    agent_class = agents[config['agent_name']]
    example_batch = {
        'observations':timestep.observation.reshape(1, -1),  # Add batch dimension
        'actions': jnp.ones((1,), dtype=jnp.int32) * NUM_ACTIONS, # TODO: make sure it should be the maximal value of action space  # Single action for batch size 1
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

    exemplary_goals = jnp.array([6,2,2,6,1,7])
    exemplary_goals = exemplary_goals.reshape(1, -1)
    exemplary_goals = jnp.tile(exemplary_goals, (256, 1))
    print(f"exemplary_goals.shape: {exemplary_goals.shape}")

    collect_data_fn = collect_data(env_name, NUM_ENVS, exemplary_goals, EPISODE_LENGTH, view_size=VIEW_SIZE)


    for i in range(1000):
        key, new_key = jax.random.split(key)
        # get new data
        env_step, timesteps_all = collect_data_fn(agent, new_key)

        # insert data into buffer
        buffer_state = replay_buffer.insert(buffer_state, timesteps_all)

        for j in range(1):        
            buffer_state, transitions = replay_buffer.sample(buffer_state) 
            batch_keys = jax.random.split(buffer_state.key, transitions.observation.shape[0])
            state, action, goal = jax.vmap(flatten_batch, in_axes=(None, 0, 0))((0.99, None, None), transitions, batch_keys)
            state, action, goal = state[:,0].reshape(state.shape[0], -1), action[:,0].reshape(action.shape[0]), goal[:,0].reshape(goal.shape[0], -1) 
            # print(f"state.shape: {state.shape}")
            # print(f"action.shape: {action.shape}")
            # print(f"goal.shape: {goal.shape}")
            # jax.debug.print("state: {x}", x=state)
            # jax.debug.print("action: {x}", x=action)
            # jax.debug.print("goal: {x}", x=goal)

            valid_batch = {
                'observations':state,
                'actions':action,
                'value_goals':goal,
                'actor_goals':goal,
            }

            agent, update_info = agent.update(valid_batch)
            update_info.update({"eval/reward_min": timesteps_all.reward.min(), "eval/reward_max": timesteps_all.reward.max(), "eval/reward_mean": timesteps_all.reward.mean()})
        wandb.log(update_info)



if __name__ == "__main__":
    app.run(main)