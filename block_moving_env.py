import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Dict, Any
from dataclasses import dataclass
import chex
from flax import struct


class BoxPushingState(struct.PyTreeNode):
    """State representation for the box pushing environment."""
    grid: jax.Array  # N x N grid where 0=empty, 1=box
    agent_pos: jax.Array  # [row, col] position of agent
    agent_has_box: jax.Array  # Whether agent is carrying a box
    steps: jax.Array  # Current step count
    target_cells: jax.Array  # Target cell coordinates for boxes

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
        key1, key2, key3 = random.split(key, 3)
        
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
            grid=grid,
            agent_pos=agent_pos,
            agent_has_box=False,
            steps=0,
            target_cells=target_cells
        )
        
        info = {}
        
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
        
        new_state = BoxPushingState(
            grid=new_grid,
            agent_pos=new_pos,
            agent_has_box=new_agent_has_box,
            steps=new_steps,
            target_cells=state.target_cells
        )
        
        info = {
            'total_boxes': jnp.sum(new_grid)
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
        return jnp.sum(grid) == 0

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

        # Change all boxes to empty
        state = state.replace(
            grid=state.grid.at[state.grid == GridStatesEnum.BOX].set(GridStatesEnum.EMPTY)
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

if __name__ == "__main__":
    # Create and run the game
    # import jax.random as random
    # key = random.PRNGKey(2)
    # env = BoxPushingEnv(grid_size=5, max_steps=2000, number_of_boxes=5)
    # env.play_game(key)


    # vmap environment
    NUM_ENVS = 4
    env = BoxPushingEnv(grid_size=5, max_steps=2000, number_of_boxes=5)
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
        print(f"\n=== Step {step_num + 1} ===")
        return new_state, (new_state, reward, done, info)
    
    final_state, (states, rewards, dones, infos) = jax.lax.scan(
        step_fn, 
        new_state, 
        jnp.arange(5)
    )
    new_state = final_state
    print(new_state.grid.shape)
    print(states.grid.shape)

    # Show consecutive states from first environment
    print("\n=== Consecutive States from First Environment ===")
    for step in range(5):
        state = jax.tree_util.tree_map(lambda x: x[step, 0], states)
        print(f"\nStep {step + 1}:")
        env._display_state(state)

    # Create a solved state
    solved_state = env.create_solved_state(jax.tree_util.tree_map(lambda x: x[0, 0], states))
    print("\n=== Solved State ===")
    env._display_state(solved_state)