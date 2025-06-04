import os

from matplotlib import pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['WANDB_MODE'] = 'offline'

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


FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 0, 'Random seed.')

config_flags.DEFINE_config_file('agent', ROOT_DIR + '/agents/crl.py', lock_config=False)

def main(_):
    # Wandb
    wandb.init(project="xminigrid-crl_full_state", name="rb_test", config=FLAGS)
    # Environment parameters
    VIEW_SIZE = 3
    BATCH_SIZE = 256
    NUM_ENVS = 256
    MAX_REPLAY_SIZE = 1000
    EPISODE_LENGTH = 20
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
    exemplary_goal = timestep.replace(state=timestep.state.replace(agent=timestep.state.agent.replace(position=jnp.array([3,3]))))
    # TODO Something wrong here
    exemplary_goals = repeat_tree(exemplary_goal, 256)
    print(f"exemplary_goals.shape: {exemplary_goals.observation.shape}")
    print(timestep.observation.shape)
    print(timestep.state.step_num)


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
    # state, future_state, goal_index = jax.vmap(flatten_batch, in_axes=(None, 0, 0))((0.99, None, None), transitions, batch_keys)
    # state, future_state, goal_index = state[:,0].reshape(state.shape[0], -1), future_state[:,0].reshape(future_state.shape[0]), goal_index[:,0].reshape(goal_index.shape[0], -1) 

    config = FLAGS.agent
    config['discrete'] = True
    agent_class = agents[config['agent_name']]
    print(f"jax.flatten_util.ravel_pytree(timestep.state.agent)[0].reshape(1, -1).shape {jax.flatten_util.ravel_pytree(timestep.state.agent)[0].reshape(1, -1).shape}")
    print(f"jnp.concatenate([timestep.state.grid.reshape(1, -1), jax.flatten_util.ravel_pytree(timestep.state.agent)[0].reshape(1, -1)], axis=1) {jnp.concatenate([timestep.state.grid.reshape(1, -1), jax.flatten_util.ravel_pytree(timestep.state.agent)[0].reshape(1, -1)], axis=1).shape}")
    # TODO: make sure to ravel only the agent 

    print(f"jnp.ones((1,), dtype=jnp.int32) * NUM_ACTIONS: {jnp.ones((1,), dtype=jnp.int32) * NUM_ACTIONS}")
    example_batch = {
        'observations':get_concatenated_state(timestep) ,  # Add batch dimension 
        'actions': jnp.ones((1,), dtype=jnp.int32) * (NUM_ACTIONS-1), # TODO: make sure it should be the maximal value of action space  # Single action for batch size 1
        'value_goals': get_concatenated_state(timestep) ,
        'actor_goals': get_concatenated_state(timestep) ,
        'terminals': timestep.step_type.reshape(1,-1) == xminigrid.types.StepType.LAST,
    }
    print(f"example_batch['value_goals'].shape: {example_batch['value_goals'].shape}")
    print(f"example_batch['observations'].shape: {example_batch['observations'].shape}")

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

        for j in range(1):        
            buffer_state, transitions = replay_buffer.sample(buffer_state) 
            batch_keys = jax.random.split(buffer_state.key, transitions.observation.shape[0])
            state, future_state, goal_index = jax.vmap(flatten_batch, in_axes=(None, 0, 0))((0.99, None, None), transitions, batch_keys)
            if epoch == 5:

                fig, axes = plt.subplots(4, state.observation.shape[1], figsize=(25, 10))

                for i in range(state.observation.shape[1]):
                    transition_from_state_i = jax.tree_util.tree_map(lambda x: x[0,i], state)
                    axes[0,i].imshow(env.render(env_params, transition_from_state_i))
                    axes[0,i].set_title(f'Timestep {i}\nAction: {state.action[0,i]}\nnum_steps: {state.state.step_num[0,i]}')
                    axes[0,i].axis('off')

                for i in range(future_state.observation.shape[1]):
                    transition_from_future_state_i = jax.tree_util.tree_map(lambda x: x[0,i], future_state)
                    axes[1,i].imshow(env.render(env_params, transition_from_future_state_i))
                    axes[1,i].set_title(f'Future for {i}\nTimestep {goal_index[0,i]}\nnum_steps: {future_state.state.step_num[0,i]}')
                    axes[1,i].axis('off')

                for i in range(state.observation.shape[1]):
                    transition_from_state_i = jax.tree_util.tree_map(lambda x: x[1,i], state)
                    axes[2,i].imshow(env.render(env_params, transition_from_state_i))
                    axes[2,i].set_title(f'Timestep {i}\nAction: {state.action[1,i]}\nnum_steps: {state.state.step_num[1,i]}')
                    axes[2,i].axis('off')

                for i in range(future_state.observation.shape[1]):
                    transition_from_future_state_i = jax.tree_util.tree_map(lambda x: x[1,i], future_state)
                    axes[3,i].imshow(env.render(env_params, transition_from_future_state_i))
                    axes[3,i].set_title(f'Future for {i}\nTimestep {goal_index[1,i]}\nnum_steps: {future_state.state.step_num[1,i]}')
                    axes[3,i].axis('off')

                plt.tight_layout()
                plt.savefig("render_transition_from_training.png")
                plt.close()

            state, future_state, goal_index = jax.tree_util.tree_map(lambda x: x[:,0], state), jax.tree_util.tree_map(lambda x: x[:,0], future_state), jax.tree_util.tree_map(lambda x: x[:,0], goal_index)
            actions = state.action
            print(f"actions.shape: {actions.shape}")
            print(f"state.state.grid.shape: {state.state.grid.shape}")



            # jax.debug.print("state: {x}", x=state)


            valid_batch = {
                'observations':get_concatenated_state(state),
                'actions':actions.squeeze(),
                'value_goals':get_concatenated_state(future_state),
                'actor_goals':get_concatenated_state(future_state),
            }

            agent, update_info = agent.update(valid_batch)
            update_info.update({"eval/reward_min": timesteps_all.reward.min(), "eval/reward_max": timesteps_all.reward.max(), "eval/reward_mean": timesteps_all.reward.mean()})
        wandb.log(update_info)



if __name__ == "__main__":
    app.run(main)