import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import chex

@dataclass
class BoxPushingState:
    """State representation for the box pushing environment."""
    grid: jax.Array  # N x N grid where 0=empty, 1=box
    agent_pos: jax.Array  # [row, col] position of agent
    agent_has_box: bool  # Whether agent is carrying a box
    steps: int  # Current step count
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

@dataclass
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
        for i in range(len(target_cells)):
            row, col = target_cells[i]
            if grid[row, col] == GridStatesEnum.BOX:
                grid = grid.at[row, col].set(GridStatesEnum.BOX_ON_TARGET)
            else:
                grid = grid.at[row, col].set(GridStatesEnum.TARGET)
        
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
        
        # Get current position
        row, col = state.agent_pos[0], state.agent_pos[1]
        
        # Define action mappings
        actions = {
            0: (-1, 0),   # UP
            1: (1, 0),    # DOWN
            2: (0, -1),   # LEFT
            3: (0, 1),    # RIGHT
            4: None,      # PICK_UP
            5: None       # PUT_DOWN
        }
        
        reward = -0.1  # Small penalty per step
        done = False

        grid = state.grid
        if action < 4:  # Movement action
            dr, dc = actions[action]
            new_row = row + dr
            new_col = col + dc
            
            # Check bounds and collision
            valid_move = (
                (new_row >= 0) & (new_row < self.grid_size) &
                (new_col >= 0) & (new_col < self.grid_size) &
                ((grid[new_row, new_col] == GridStatesEnum.EMPTY) | (grid[new_row, new_col] == GridStatesEnum.TARGET) | (grid[new_row, new_col] == GridStatesEnum.BOX))
            )
            
            new_pos = jax.lax.cond(
                valid_move,
                lambda: jnp.array([new_row, new_col]),
                lambda: state.agent_pos
            )
            
            # Check if agent was on box or target before clearing
            grid = jax.lax.cond(
                grid[row, col] == GridStatesEnum.AGENT_ON_BOX,
                lambda: grid.at[row, col].set(GridStatesEnum.BOX),  # Leave box if agent was on box
                lambda: jax.lax.cond(
                    grid[row, col] == GridStatesEnum.AGENT_ON_TARGET,
                    lambda: grid.at[row, col].set(GridStatesEnum.TARGET),  # Leave target if agent was on target
                    lambda: jax.lax.cond(
                        valid_move,
                        lambda: grid.at[row, col].set(GridStatesEnum.EMPTY),  # Clear if agent was just on empty cell
                        lambda: grid  # Leave agent there if not valid move
                    )
                )
            )
            
            # Check if agent is now on box, target, or empty cell
            new_grid = jax.lax.cond(
                valid_move,
                lambda: jax.lax.cond(
                    grid[new_row, new_col] == GridStatesEnum.BOX,
                    lambda: grid.at[new_row, new_col].set(GridStatesEnum.AGENT_ON_BOX),  # Agent on box
                    lambda: jax.lax.cond(
                        grid[new_row, new_col] == GridStatesEnum.TARGET,
                        lambda: grid.at[new_row, new_col].set(GridStatesEnum.AGENT_ON_TARGET),  # Agent on target
                        lambda: grid.at[new_row, new_col].set(GridStatesEnum.AGENT)  # Agent on empty cell
                    )
                ),
                lambda: grid
            )
            new_agent_has_box = state.agent_has_box
            
        elif action == 4:  # PICK_UP
            new_pos = state.agent_pos
            new_grid, new_agent_has_box, pickup_reward = self._handle_pickup(state)
            reward += pickup_reward
            
        elif action == 5:  # PUT_DOWN
            new_pos = state.agent_pos
            new_grid, new_agent_has_box, putdown_reward = self._handle_putdown(state)
            reward += putdown_reward
            
        else:
            new_pos = state.agent_pos
            new_grid = grid
            new_agent_has_box = state.agent_has_box
        
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
            'boxes_moved': self._count_boxes_in_target(new_grid),
            'total_boxes': jnp.sum(new_grid)
        }
        
        return new_state, reward, done, info
    
    def _handle_putdown(self, state: BoxPushingState) -> Tuple[jax.Array, bool, float]:
        """Handle putdown action."""
        if not state.agent_has_box:
            return state.grid, state.agent_has_box, 0.0
        
        row, col = state.agent_pos[0], state.agent_pos[1]

        # Check if target cell is available
        target_cell_available = (
            (row >= 0) & (row < self.grid_size) &
            (col >= 0) & (col < self.grid_size) &
            (state.grid[row, col] == 0)
        )

        # Check if target cell is in target cells
        target_cell_in_target_cells = any(jnp.array_equal(jnp.array([row, col]), target) for target in state.target_cells)

        if target_cell_available and target_cell_in_target_cells:
            return state.grid, state.agent_has_box, 1.0
        else:
            return state.grid, state.agent_has_box, 0.0


    def _is_goal_reached(self, grid: jax.Array) -> bool:
        """Check if all boxes are in target cells."""
        return jnp.sum(grid) == 0

    def _count_boxes_in_target(self, grid: jax.Array) -> int:
        """Count boxes in target cells."""
        return jnp.sum(grid)
    
    def _handle_pickup(self, state: BoxPushingState) -> Tuple[jax.Array, bool, float]:
        """Handle pickup action."""
        if state.agent_has_box: # Agent already has a box - invalid action
            return state.grid, state.agent_has_box, 0.0
        
        row, col = state.agent_pos[0], state.agent_pos[1]
        
        # Check if agent is standing on a box or on target with box
        current_cell = state.grid[row, col]
        can_pickup = (current_cell == GridStatesEnum.AGENT_ON_BOX or 
                     current_cell == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX)
        
        if can_pickup:
            # Pick up the box from current position
            # If on target with box, agent should remain on target
            if current_cell == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX:
                new_grid = state.grid.at[row, col].set(GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX)
            else:
                new_grid = state.grid.at[row, col].set(GridStatesEnum.EMPTY)
            return new_grid, True, 1.0
        else:
            return state.grid, False, 0.0

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


if __name__ == "__main__":
    # Create and run the game
    import jax.random as random
    key = random.PRNGKey(2)
    env = BoxPushingEnv(grid_size=6, max_steps=2000, number_of_boxes=3)
    env.play_game(key)
