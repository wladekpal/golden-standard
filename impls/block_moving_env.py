import functools
import os

from rb import TrajectoryUniformSamplingQueue, jit_wrap, segment_ids_per_row
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Dict, Any
from dataclasses import dataclass
import chex
from flax import struct
from absl import app, flags
from ml_collections import config_flags
from agents import agents
from config import ROOT_DIR


class BoxPushingState(struct.PyTreeNode):
    """State representation for the box pushing environment."""
    key: jax.Array
    grid: jax.Array  # N x N grid where 0=empty, 1=box
    agent_pos: jax.Array  # [row, col] position of agent
    agent_has_box: jax.Array  # Whether agent is carrying a box
    steps: jax.Array  # Current step count
    target_cells: jax.Array  # Target cell coordinates for boxes
    goal: jax.Array  # Goal cell coordinates for boxes
    reward: jax.Array  # Reward for the current step


class TimeStep(BoxPushingState):
    action: jax.Array


@dataclass
class GridStatesEnum:
    """Grid states representation for the box pushing environment."""
    EMPTY = 0
    BOX = 1
    TARGET = 2
    AGENT = 3
    AGENT_CARRYING_BOX = 4 # Agent is carrying a box
    AGENT_ON_BOX = 5 # Agent is on box
    AGENT_ON_TARGET = 6 # Agent is on target
    AGENT_ON_TARGET_CARRYING_BOX = 7 # Agent is on target carrying the box
    AGENT_ON_TARGET_WITH_BOX = 8 # Agent is on target on which there is a box
    AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX = 9 # Agent is on target on which there is a box and is carrying a box
    BOX_ON_TARGET = 10 # Box is on target
    AGENT_ON_BOX_CARRYING_BOX = 11 # Agent is on box carrying a box

ACTIONS = {
    0: (-1, 0),   # UP
    1: (1, 0),    # DOWN
    2: (0, -1),   # LEFT
    3: (0, 1),    # RIGHT
    4: None,      # PICK_UP
    5: None       # PUT_DOWN
}
        


class BoxPushingEnv:
    """JAX-based box pushing environment."""
    
    def __init__(self, grid_size: int = 20, max_steps: int = 2000, number_of_boxes: int = 3):

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.action_space = 6  # UP, DOWN, LEFT, RIGHT, PICK_UP, PUT_DOWN
        self.number_of_boxes = number_of_boxes
        
    def reset(self, key: jax.Array) -> Tuple[BoxPushingState, Dict[str, Any]]:
        """Reset environment to initial state."""
        key1, key2, key3, key4 = random.split(key, 4)
        
        # Initialize empty grid
        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        
        # Place exactly number_of_boxes boxes
        box_positions = random.choice(key1, self.grid_size * self.grid_size, shape=(self.number_of_boxes,), replace=False)
        box_rows = box_positions // self.grid_size
        box_cols = box_positions % self.grid_size
        
        # Place boxes at valid positions
        grid = grid.at[box_rows, box_cols].set(GridStatesEnum.BOX)
        
        # Generate target cells in right quarter
        target_cells = self._generate_target_cells(key3)
        
        # Check if there's already a box at target positions and use BOX_ON_TARGET if so
        # Use vectorized operations instead of for loop
        target_rows = target_cells[:, 0]
        target_cols = target_cells[:, 1]
        target_cell_values = grid[target_rows, target_cols]
        
        # Create mask for boxes at target positions
        box_at_target_mask = target_cell_values == GridStatesEnum.BOX
        
        # Update grid using vectorized operations
        grid = grid.at[target_rows, target_cols].set(
            jnp.where(box_at_target_mask, GridStatesEnum.BOX_ON_TARGET, GridStatesEnum.TARGET)
        )
        
        # Place agent randomly
        row = random.randint(key2, (), 0, self.grid_size)
        col = random.randint(key2, (), 0, self.grid_size)
        agent_pos = jnp.array([row, col])

        # Check what's at agent position and set appropriate state
        current_cell = grid[agent_pos[0], agent_pos[1]]
        agent_state = jax.lax.switch(
            current_cell,
            branches=[
                lambda _: GridStatesEnum.AGENT,  # Empty cell
                lambda _: GridStatesEnum.AGENT_ON_BOX,  # Box cell
                lambda _: GridStatesEnum.AGENT_ON_TARGET,  # Target cell
            ],
            operand=None
        )
        grid = grid.at[agent_pos[0], agent_pos[1]].set(agent_state)
        state = BoxPushingState(
            key=key4,
            grid=grid,
            agent_pos=agent_pos,
            agent_has_box=False,
            steps=0,
            target_cells=target_cells,
            goal=jnp.zeros_like(grid),
            reward=0
        )
        goal = self.create_solved_state(state)
        state = state.replace(goal=goal.grid)
        
        info = {
            'boxes_on_target': jnp.sum(grid == GridStatesEnum.BOX_ON_TARGET)
        }
        
        return state, info
    
    def _generate_target_cells(self, key: jax.Array) -> jax.Array:
        """Generate target cells in right quarter."""
        target_start_row = self.grid_size // 2
        target_start_col = self.grid_size // 2
        
        # Create candidate cells
        rows = jnp.arange(target_start_row, self.grid_size)
        cols = jnp.arange(target_start_col, self.grid_size)
        row_grid, col_grid = jnp.meshgrid(rows, cols, indexing='ij')
        candidates = jnp.stack([row_grid.flatten(), col_grid.flatten()], axis=1)
        
        # Shuffle and select
        shuffled_indices = random.permutation(key, len(candidates))
        candidates = candidates[shuffled_indices]
        
        return candidates[:self.number_of_boxes]
    
    def step(self, state: BoxPushingState, action: int) -> Tuple[BoxPushingState, float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        chex.assert_shape(action, ())
        
        # Increment steps
        new_steps = state.steps + 1        

        reward = 0 
        done = False

        # Use jax.lax.switch instead of if-elif to handle traced arrays
        action_result = jax.lax.switch(
            action,
            branches=[
                lambda _: self.handle_movement(state, 0),  # Movement action 0
                lambda _: self.handle_movement(state, 1),  # Movement action 1
                lambda _: self.handle_movement(state, 2),  # Movement action 2
                lambda _: self.handle_movement(state, 3),  # Movement action 3
                lambda _: (state.agent_pos, *self._handle_pickup(state)),  # PICK_UP
                lambda _: (state.agent_pos, *self._handle_putdown(state)),  # PUT_DOWN
            ],
            operand=None
        )
        
        new_pos, new_grid, new_agent_has_box = action_result
        
        # Check if done
        done = (new_steps >= self.max_steps) | self._is_goal_reached(new_grid)

        reward = self._get_reward(new_grid)
        
        new_state = BoxPushingState(
            key=state.key,
            grid=new_grid,
            agent_pos=new_pos,
            agent_has_box=new_agent_has_box,
            steps=new_steps,
            target_cells=state.target_cells,
            goal=state.goal,
            reward=reward
        )
        
        info = {
            'boxes_on_target': jnp.sum(new_grid == GridStatesEnum.BOX_ON_TARGET)
        }
        
        return new_state, reward, done, info
    
    def handle_movement(self, state: BoxPushingState, action: int) -> Tuple[jax.Array, bool]:
        row, col = state.agent_pos[0], state.agent_pos[1]
        grid = state.grid
        dr, dc = ACTIONS[action]
        new_row = row + dr
        new_col = col + dc
        
        # Check bounds and collision
        valid_move = (
            (new_row >= 0) & (new_row < self.grid_size) &
            (new_col >= 0) & (new_col < self.grid_size) 
            # ((grid[new_row, new_col] == GridStatesEnum.EMPTY) | (grid[new_row, new_col] == GridStatesEnum.TARGET) | (grid[new_row, new_col] == GridStatesEnum.BOX))
        )

        # Use jax.lax.cond instead of if statement to handle traced arrays
        def move_valid():
            # Check if agent was on box or target or box on target before clearing
            grid_after_clear = jax.lax.switch(
                jnp.array([
                    grid[row, col] == GridStatesEnum.AGENT_ON_BOX,
                    grid[row, col] == GridStatesEnum.AGENT_ON_TARGET,
                    grid[row, col] == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX,
                    grid[row, col] == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX,
                    grid[row, col] == GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX,
                    grid[row, col] == GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX,
                    grid[row, col] == GridStatesEnum.AGENT,
                    grid[row, col] == GridStatesEnum.AGENT_CARRYING_BOX,
                ]).astype(jnp.int32).argmax(),
                [
                    lambda: grid.at[row, col].set(GridStatesEnum.BOX),  # Leave box if agent was on box
                    lambda: grid.at[row, col].set(GridStatesEnum.TARGET),  # Leave target if agent was on target
                    lambda: grid.at[row, col].set(GridStatesEnum.BOX_ON_TARGET),  # Leave box on target if agent was on box on target
                    lambda: grid.at[row, col].set(GridStatesEnum.BOX_ON_TARGET),  # Leave box on target if agent was on box on target
                    lambda: grid.at[row, col].set(GridStatesEnum.TARGET),  # Leave target if agent was on target
                    lambda: grid.at[row, col].set(GridStatesEnum.BOX),  # Leave box on target if agent was on box on target
                    lambda: grid.at[row, col].set(GridStatesEnum.EMPTY),  # Clear if agent was just on empty cell
                    lambda: grid.at[row, col].set(GridStatesEnum.EMPTY),  # Clear if agent was just on empty cell
                ]
            )
            
            # Check if agent is now on box, target, or empty cell
            new_grid = jax.lax.cond(
                state.agent_has_box,
                lambda: jax.lax.switch(
                    jnp.array([
                        grid_after_clear[new_row, new_col] == GridStatesEnum.BOX,
                        grid_after_clear[new_row, new_col] == GridStatesEnum.TARGET,
                        grid_after_clear[new_row, new_col] == GridStatesEnum.BOX_ON_TARGET,
                        grid_after_clear[new_row, new_col] == GridStatesEnum.EMPTY,
                    ]).astype(jnp.int32).argmax(),
                    [
                        lambda: grid_after_clear.at[new_row, new_col].set(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX),  # Agent on box carrying box
                        lambda: grid_after_clear.at[new_row, new_col].set(GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX),  # Agent on target carrying box
                        lambda: grid_after_clear.at[new_row, new_col].set(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX),  # Agent on box on target carrying box
                        lambda: grid_after_clear.at[new_row, new_col].set(GridStatesEnum.AGENT_CARRYING_BOX),  # Agent on empty cell carrying box
                    ]
                ),
                lambda: jax.lax.switch(
                    jnp.array([
                        grid_after_clear[new_row, new_col] == GridStatesEnum.BOX,
                        grid_after_clear[new_row, new_col] == GridStatesEnum.TARGET,
                        grid_after_clear[new_row, new_col] == GridStatesEnum.BOX_ON_TARGET,
                        grid_after_clear[new_row, new_col] == GridStatesEnum.EMPTY,
                    ]).astype(jnp.int32).argmax(),
                    [
                        lambda: grid_after_clear.at[new_row, new_col].set(GridStatesEnum.AGENT_ON_BOX),  # Agent on box
                        lambda: grid_after_clear.at[new_row, new_col].set(GridStatesEnum.AGENT_ON_TARGET),  # Agent on target
                        lambda: grid_after_clear.at[new_row, new_col].set(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX),  # Agent on box on target
                        lambda: grid_after_clear.at[new_row, new_col].set(GridStatesEnum.AGENT),  # Agent on empty cell
                    ]
                )
            )
            return jnp.array([new_row, new_col]), new_grid, state.agent_has_box
        
        def move_invalid():
            return state.agent_pos, grid, state.agent_has_box
        
        new_pos, new_grid, new_agent_has_box = jax.lax.cond(
            valid_move,
            move_valid,
            move_invalid
        )
        
        return new_pos, new_grid, new_agent_has_box

    def _handle_putdown(self, state: BoxPushingState) -> Tuple[jax.Array, bool]:
        """Handle putdown action."""

        row, col = state.agent_pos[0], state.agent_pos[1]
        current_cell = state.grid[row, col]
        def putdown_valid():
            new_grid = state.grid.at[row, col].set(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX)
            return new_grid, False
        
        def putdown_invalid():
            return state.grid, state.agent_has_box
        
        new_grid, new_agent_has_box = jax.lax.cond(
            jnp.logical_or(
                current_cell == GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX,
                current_cell == GridStatesEnum.AGENT_CARRYING_BOX
            ),
            putdown_valid,
            putdown_invalid
        )
        return new_grid, new_agent_has_box

    def _is_goal_reached(self, grid: jax.Array) -> bool:
        """Check if all boxes are in target cells."""
        return jnp.sum(grid == GridStatesEnum.BOX_ON_TARGET) == self.number_of_boxes
    
    def _get_reward(self, grid: jax.Array) -> float:
        """Get reward for the current state."""
        return jnp.sum(grid == GridStatesEnum.BOX_ON_TARGET)

    def _handle_pickup(self, state: BoxPushingState) -> Tuple[jax.Array, bool]:
        """Handle pickup action."""
            
        row, col = state.agent_pos[0], state.agent_pos[1]
        current_cell = state.grid[row, col]
        def pickup_valid():
            new_grid = state.grid.at[row, col].set(GridStatesEnum.AGENT_CARRYING_BOX)
            return new_grid, True
        
        def pickup_invalid():
            return state.grid, state.agent_has_box
        
        new_grid, new_agent_has_box = jax.lax.cond(
            current_cell == GridStatesEnum.AGENT_ON_BOX,
            pickup_valid,
            pickup_invalid
        )
        return new_grid, new_agent_has_box

    def play_game(self, key: jax.Array):
        """Interactive game loop using input() for controls."""
        import time
        
        # Initialize the environment
        state, _ = self.reset(key)
        done = False
        total_reward = 0
        
        print("=== Box Pushing Game ===")
        print("Controls: w(up), s(down), a(left), d(right), e(pickup), r(drop)")
        print("Goal: Move boxes to target cells (marked with 'T')")
        print("Press 'q' to quit\n")
        
        while not done:
            # Display current state
            self._display_state(state)
            print(f"Steps: {state.steps}, Reward: {total_reward}")
            
            # Get user input
            action = None
            while action is None:
                try:
                    user_input = input("Enter action (w/s/a/d/e/r/q): ").lower().strip()
                    if user_input == 'w':
                        action = 0  # Up
                    elif user_input == 's':
                        action = 1  # Down
                    elif user_input == 'a':
                        action = 2  # Left
                    elif user_input == 'd':
                        action = 3  # Right
                    elif user_input == 'e':
                        action = 4  # Pickup
                    elif user_input == 'r':
                        action = 5  # Drop
                    elif user_input == 'q':
                        print("Game ended by user.")
                        return
                    else:
                        print("Invalid input. Use w/s/a/d/e/r/q")
                except (EOFError, KeyboardInterrupt):
                    print("\nGame ended by user.")
                    return
            
            # Take action
            state, reward, done, info = self.step(state, action)
            total_reward += reward
        
        # Final display
        self._display_state(state)
        print(f"\nGame Over! Final Reward: {total_reward}")
        print(f"Boxes moved to target: {info['boxes_moved']}/{info['total_boxes']}")
        
    def _display_state(self, state: BoxPushingState):
        """Display the current game state in ASCII."""
        print("\n" + "=" * (self.grid_size * 2 + 1))
        for row in range(self.grid_size):
            print("|", end="")
            for col in range(self.grid_size):
                cell_value = state.grid[row, col]
                print(f"{cell_value} ", end="")
            print("|")
        print("=" * (self.grid_size * 2 + 1))

    def create_solved_state(self, state: BoxPushingState) -> BoxPushingState:
        """Create a solved state."""
        # Change all target cells to box on target
        target_rows = state.target_cells[:, 0]
        target_cols = state.target_cells[:, 1]
        state = state.replace(
            grid=state.grid.at[target_rows, target_cols].set(GridStatesEnum.BOX_ON_TARGET)
        )

        # Change all boxes to empty - use where to avoid boolean indexing issue
        box_mask = state.grid == GridStatesEnum.BOX
        state = state.replace(
            grid=jnp.where(box_mask, GridStatesEnum.EMPTY, state.grid)
        )
        
        # Check what cell the agent is currently on and update accordingly
        agent_row, agent_col = state.agent_pos[0], state.agent_pos[1]
        current_cell = state.grid[agent_row, agent_col]
        
        # Update grid based on current cell type
        new_cell_value = jax.lax.cond(
            current_cell == GridStatesEnum.BOX_ON_TARGET,
            lambda: GridStatesEnum.AGENT_ON_TARGET_WITH_BOX,  # Agent on target with box
            lambda: jax.lax.cond(
                current_cell == GridStatesEnum.EMPTY,
                lambda: GridStatesEnum.AGENT,  # Agent on empty cell
                lambda: current_cell  # Keep current cell if it's already an agent state
            )
        )
        
        state = state.replace(
            grid=state.grid.at[agent_row, agent_col].set(new_cell_value),
            agent_has_box=jnp.array(False)
        )
        return state


class Wrapper(BoxPushingEnv):
    def __init__(self, env: BoxPushingEnv):
        self._env = env
        # Copy attributes from wrapped environment
        for attr in ['grid_size', 'max_steps', 'number_of_boxes']:
            if hasattr(env, attr):
                setattr(self, attr, getattr(env, attr))
    
    def reset(self, key: jax.Array) -> Tuple[BoxPushingState, Dict[str, Any]]:
        return self._env.reset(key)
    
    def step(self, state: BoxPushingState, action: int) -> Tuple[BoxPushingState, float, bool, Dict[str, Any]]:
        return self._env.step(state, action)

class AutoResetWrapper(Wrapper):
    def __init__(self, env: BoxPushingEnv):
        super().__init__(env)
    
    def step(self, state: BoxPushingState, action: int) -> Tuple[BoxPushingState, float, bool, Dict[str, Any]]:
        state, reward, done, info = self._env.step(state, action)
        key_new, _ = jax.random.split(state.key, 2)
        
        def reset_fn(key):
            reset_state, reset_info = self._env.reset(key)
            return reset_state, 0, False, reset_info
        
        state, reward, done, info = jax.lax.cond(
            done,
            lambda: reset_fn(key_new),
            lambda: (state, reward, done, info)
        )
        return state, reward, done, info



@functools.partial(jax.jit, static_argnames=("buffer_config"))
def flatten_batch(buffer_config, transition, sample_key):
    gamma, state_size, goal_indices = buffer_config

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
    single_trajectories = segment_ids_per_row(transition.steps)
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


FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 0, 'Random seed.')

config_flags.DEFINE_config_file('agent', ROOT_DIR + '/agents/crl.py', lock_config=False)


def main(_):
    
    # vmap environment
    NUM_ENVS = 4
    MAX_REPLAY_SIZE = 10000
    BATCH_SIZE = 128
    EPISODE_LENGTH = 100
    NUM_ACTIONS = 6

    env = BoxPushingEnv(grid_size=5, max_steps=200, number_of_boxes=5)
    env = AutoResetWrapper(env)
    key = random.PRNGKey(2)
    keys = random.split(key, NUM_ENVS)
    new_state, info = jax.vmap(env.reset)(keys)
    print(new_state.grid.shape)
    print(info)
    # vmap step for 5 consecutive steps using jax.lax.scan
    def step_fn(carry, step_num):
        state = carry
        actions = jnp.array([3] * NUM_ENVS)
        new_state, reward, done, info = jax.vmap(env.step)(state, actions)
        timestep = TimeStep(
            key=state.key,
            grid=state.grid,
            target_cells=state.target_cells,
            agent_pos=state.agent_pos,
            agent_has_box=state.agent_has_box,
            steps=state.steps,
            action=actions, 
            goal=state.goal,
            reward=reward
        )
        # return new_state, (new_state, reward, done, info)
        return new_state, (new_state, timestep)
    
    # final_state, (states, rewards, dones, infos) = jax.lax.scan(
    final_state, (states, timesteps) = jax.lax.scan(
        step_fn, 
        new_state, 
        jnp.arange(EPISODE_LENGTH)
    )
    new_state = final_state

    timestep = jax.tree_util.tree_map(lambda x: x[0, 0], timesteps)

    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=MAX_REPLAY_SIZE,
            dummy_data_sample=timestep,
            sample_batch_size=BATCH_SIZE,
            num_envs=NUM_ENVS,
            episode_length=EPISODE_LENGTH,
        )
    )
    buffer_state = jax.jit(replay_buffer.init)(key)
    buffer_state = replay_buffer.insert(buffer_state, timesteps)
    buffer_state, transitions = replay_buffer.sample(buffer_state)


    # Agent
    config = FLAGS.agent
    config['discrete'] = True
    agent_class = agents[config['agent_name']]


    example_batch = {
        'observations':transitions.grid[:,0,:].reshape(transitions.grid.shape[0], transitions.grid.shape[-1]*transitions.grid.shape[-2]),  # Add batch dimension 
        'actions': jnp.ones((transitions.grid.shape[0],), dtype=jnp.int32) * (NUM_ACTIONS-1), # TODO: make sure it should be the maximal value of action space  # Single action for batch size 1
        'value_goals': transitions.grid[:,0,:].reshape(transitions.grid.shape[0], transitions.grid.shape[-1]*transitions.grid.shape[-2]),
        'actor_goals': transitions.grid[:,0,:].reshape(transitions.grid.shape[0], transitions.grid.shape[-1]*transitions.grid.shape[-2]),
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

    agent, update_info = agent.update(example_batch)
    print(update_info)

    def collect_data(agent, key):
        env = BoxPushingEnv(grid_size=5, max_steps=200, number_of_boxes=5)
        env = AutoResetWrapper(env)
        
        def step_fn(carry, step_num):
            state, info, key = carry
            key, new_key = jax.random.split(key)
            print(f"state.grid.shape: {state.grid.shape}", flush=True)
            actions = agent.sample_actions(state.grid.reshape(NUM_ENVS, -1), state.goal.reshape(NUM_ENVS, -1), seed=key)
            new_state, reward, done, info = jax.vmap(env.step)(state, actions)
            timestep = TimeStep(
                key=state.key,
                grid=state.grid,
                target_cells=state.target_cells,
                agent_pos=state.agent_pos,
                agent_has_box=state.agent_has_box,
                steps=state.steps,
                action=actions,
                goal=state.goal,
                reward=reward
            )
            return (new_state, info, new_key), timestep
        
        keys = jax.random.split(key, NUM_ENVS)
        state, info = jax.vmap(env.reset)(keys)
        (timestep, info, key), timesteps_all = jax.lax.scan(step_fn, (state, info, key), (), length=EPISODE_LENGTH)
        return timestep, info, timesteps_all

    env_step, info, timesteps_all = collect_data(agent, key)


    # buffer_state, transitions = replay_buffer.sample(buffer_state)
    # batch_keys = jax.random.split(buffer_state.key, transitions.grid.shape[0])
    # state, future_state, goal_index = jax.vmap(flatten_batch, in_axes=(None, 0, 0))((0.99, None, None), transitions, batch_keys)

    # for epoch in range(1000):
    #     key, new_key = jax.random.split(key)
    #     env_step, info, timesteps_all = collect_data(agent, new_key)
    #     buffer_state = replay_buffer.insert(buffer_state, timesteps_all)
        
    #     def update_step(carry, _):
    #         buffer_state, agent, key = carry
            
    #         # Sample and process transitions
    #         buffer_state, transitions = replay_buffer.sample(buffer_state)
    #         batch_keys = jax.random.split(buffer_state.key, transitions.grid.shape[0])
    #         state, future_state, goal_index = jax.vmap(flatten_batch, in_axes=(None, 0, 0))((0.99, None, None), transitions, batch_keys)


if __name__ == "__main__":
    # Create and play the game
    # import jax.random as random
    # key = random.PRNGKey(2)
    # env = BoxPushingEnv(grid_size=5, max_steps=2000, number_of_boxes=5)
    # env.play_game(key)

    app.run(main)