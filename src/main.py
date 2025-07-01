import functools

import wandb

from rb import TrajectoryUniformSamplingQueue, jit_wrap, segment_ids_per_row

import jax
import jax.numpy as jnp
from jax import random
from absl import app, flags
from ml_collections import config_flags
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib

from impls.agents import agents
from config import ROOT_DIR
from block_moving_env import BoxPushingEnv, AutoResetWrapper, TimeStep


def flatten_batch(gamma, transition, sample_key):
    # Because it's vmaped transition.obs.shape is of shape (episode_len, obs_dim)
    seq_len = transition.grid.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(
        arrangement[:, None] < arrangement[None], dtype=jnp.float32
    )  # upper triangular matrix of shape seq_len, seq_len where all non-zero entries are 1
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount
    # probs is an upper triangular matrix of shape seq_len, seq_len of the form:
    #    [[0.        , 0.99      , 0.98010004, 0.970299  , 0.960596 ],
    #    [0.        , 0.        , 0.99      , 0.98010004, 0.970299  ],
    #    [0.        , 0.        , 0.        , 0.99      , 0.98010004],
    #    [0.        , 0.        , 0.        , 0.        , 0.99      ],
    #    [0.        , 0.        , 0.        , 0.        , 0.        ]]
    # assuming seq_len = 5
    # the same result can be obtained using probs = is_future_mask * (gamma ** jnp.cumsum(is_future_mask, axis=-1))
    single_trajectories = segment_ids_per_row(transition.steps.squeeze())
    single_trajectories = jnp.concatenate(
            [single_trajectories[:, jnp.newaxis].T] * seq_len,
            axis=0,
    )
    # array of seq_len x seq_len where a row is an array of traj_ids that correspond to the episode index from which that time-step was collected
    # timesteps collected from the same episode will have the same traj_id. All rows of the single_trajectories are same.

    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
    # ith row of probs will be non zero only for time indices that
    # 1) are greater than i
    # 2) have the same traj_id as the ith time index

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state = jax.tree_util.tree_map(lambda x: jnp.take(x, goal_index[:-1], axis=0), transition)  # the last goal_index cannot be considered as there is no future.
    states = jax.tree_util.tree_map(lambda x: x[:-1], transition) # all states but the last one are considered


    return states, future_state, goal_index


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
    _, _, timesteps = collect_data(agent, data_key, env, num_envs, episode_length)
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
    }
    eval_info.update(loss_info)
    wandb.log(eval_info)

    # Create figure for GIF
    grid_size = state.grid.shape[-2:]
    fig, ax = plt.subplots(figsize=grid_size)
    
    def animate(frame):
        ax.clear()
        grid_state = timesteps.grid[0, frame]
        action = timesteps.action[0, frame]
        reward = timesteps.reward[0, frame]
        
        # Create color mapping for grid states
        imgs = {
            0: 'assets/floor.png',                                  # EMPTY
            1: 'assets/box.png',                                    # BOX
            2: 'assets/box_target.png',                             # TARGET
            3: 'assets/agent.png',                                  # AGENT
            4: 'assets/agent_carrying_box.png',                     # AGENT_CARRYING_BOX
            5: 'assets/agent_on_box.png',                           # AGENT_ON_BOX
            6: 'assets/agent_on_target.png',                        # AGENT_ON_TARGET
            7: 'assets/agent_on_target_carrying_box.png',           # AGENT_ON_TARGET_CARRYING_BOX
            8: 'assets/agent_on_target_with_box.png',               # AGENT_ON_TARGET_WITH_BOX
            9: 'assets/agent_on_target_with_box_carrying_box.png',  # AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX
            10: 'assets/box_on_target.png',                         # BOX_ON_TARGET
            11: 'assets/agent_on_box_carrying_box.png'              # AGENT_ON_BOX_CARRYING_BOX
        }
        
        # Plot grid
        for i in range(grid_state.shape[0]):
            for j in range(grid_state.shape[1]):
                img = matplotlib.image.imread(imgs[int(grid_state[i, j])])
                ax.imshow(img, extent = [i+1, i, j+1, j])
            
        
        ax.set_xlim(0, grid_state.shape[1])
        ax.set_ylim(0, grid_state.shape[0])
        ax.set_title(f'Step {frame} | Action: {action} | Reward: {reward:.2f}')
        ax.set_aspect('equal')
        ax.invert_yaxis()
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=100, interval=80, repeat=False)
    
    # Save as GIF
    gif_path = f"/tmp/block_moving_epoch_{epoch}.gif"
    anim.save(gif_path, writer='pillow')
    plt.close()

    wandb.log({"gif": wandb.Video(gif_path, format="gif")})

FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 0, 'Random seed.')

config_flags.DEFINE_config_file('agent', ROOT_DIR + '/impls/agents/crl.py', lock_config=False)


def main(_):
    # vmap environment
    NUM_ENVS = 1024
    MAX_REPLAY_SIZE = 10000
    BATCH_SIZE = 1024
    EPISODE_LENGTH = 100
    NUM_ACTIONS = 6
    GRID_SIZE = 5
    NUM_BOXES = 2
    SEED = 3
    SUFFIX = "reward_is_goal_reached"

    wandb.init(project="moving_blocks", name=f"{GRID_SIZE}x{GRID_SIZE}_grid_{NUM_BOXES}_boxes_{EPISODE_LENGTH}_el_{NUM_ENVS}_num_envs__{SUFFIX}", config=FLAGS)

    env = BoxPushingEnv(grid_size=GRID_SIZE, max_steps=EPISODE_LENGTH, number_of_boxes=NUM_BOXES)
    env = AutoResetWrapper(env)
    key = random.PRNGKey(SEED)
    env.step = jax.jit(jax.vmap(env.step))
    env.reset = jax.jit(jax.vmap(env.reset))
    jitted_flatten_batch = jax.jit(jax.vmap(flatten_batch, in_axes=(None, 0, 0)), static_argnums=(0,))

    # Replay buffer
    dummy_timestep = TimeStep(
        key=key,
        grid=jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int8),
        target_cells=jnp.zeros((NUM_BOXES, 2), dtype=jnp.int8),
        agent_pos=jnp.zeros((2,), dtype=jnp.int8),
        agent_has_box=jnp.zeros((1,), dtype=jnp.int8),
        steps=jnp.zeros((1,), dtype=jnp.int8),
        action=jnp.zeros((1,), dtype=jnp.int8),
        goal=jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int8),
        reward=jnp.zeros((1,), dtype=jnp.int8),
        done=jnp.zeros((1,), dtype=jnp.int8),
    )
    
    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=MAX_REPLAY_SIZE,
            dummy_data_sample=dummy_timestep,
            sample_batch_size=BATCH_SIZE,
            num_envs=NUM_ENVS,
            episode_length=EPISODE_LENGTH,
        )
    )
    buffer_state = jax.jit(replay_buffer.init)(key)

    # Agent
    config = FLAGS.agent
    config['discrete'] = True
    agent_class = agents[config['agent_name']]
    example_batch = {
        'observations':dummy_timestep.grid.reshape(1, -1),  # Add batch dimension 
        'actions': jnp.ones((1,), dtype=jnp.int8) * (NUM_ACTIONS-1), # TODO: make sure it should be the maximal value of action space  # Single action for batch size 1
        'value_goals': dummy_timestep.grid.reshape(1, -1),
        'actor_goals': dummy_timestep.grid.reshape(1, -1),
    }

    print("Testing agent creation")
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
        example_batch['value_goals'],
    )
    print("Agent created")


    @jax.jit
    def update_step(carry, _):
        buffer_state, agent, key = carry
        key, batch_key, double_batch_key = jax.random.split(key, 3)
        # Sample and process transitions
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        batch_keys = jax.random.split(batch_key, transitions.grid.shape[0])
        state, future_state, goal_index = jitted_flatten_batch(0.99, transitions, batch_keys)

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
        _, _, timesteps = collect_data(agent, data_key, env, NUM_ENVS, EPISODE_LENGTH)
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


    for epoch in range(10):
        evaluate_agent(agent, env, key, jitted_flatten_batch, epoch, NUM_ENVS, EPISODE_LENGTH)
        for _ in range(10):
            buffer_state, agent, key = train_n_epochs(buffer_state, agent, key)

        evaluate_agent(agent, env, key, jitted_flatten_batch, epoch, NUM_ENVS, EPISODE_LENGTH)


if __name__ == "__main__":
    app.run(main)

