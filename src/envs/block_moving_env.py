import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Dict, Any
from dataclasses import dataclass
import chex
from flax import struct
import matplotlib
import os


class BoxPushingState(struct.PyTreeNode):
    """State representation for the box pushing environment."""
    key: jax.Array
    grid: jax.Array  # N x N grid where 0=empty, 1=box
    agent_pos: jax.Array  # [row, col] position of agent
    agent_has_box: jax.Array  # Whether agent is carrying a box
    steps: jax.Array  # Current step count
    target_cells: jax.Array  # Target cell coordinates for boxes, it's a number_of_boxes_max
    number_of_boxes: jax.Array  # Number of boxes
    goal: jax.Array  # Goal cell coordinates for boxes
    reward: jax.Array  # Reward for the current step


class TimeStep(BoxPushingState):
    action: jax.Array
    done: jax.Array


@dataclass
class GridStatesEnum:
    """Grid states representation for the box pushing environment."""
    EMPTY = jnp.int8(0)
    BOX = jnp.int8(1)
    TARGET = jnp.int8(2)
    AGENT = jnp.int8(3)
    AGENT_CARRYING_BOX = jnp.int8(4) # Agent is carrying a box
    AGENT_ON_BOX = jnp.int8(5) # Agent is on box
    AGENT_ON_TARGET = jnp.int8(6) # Agent is on target
    AGENT_ON_TARGET_CARRYING_BOX = jnp.int8(7) # Agent is on target carrying the box
    AGENT_ON_TARGET_WITH_BOX = jnp.int8(8) # Agent is on target on which there is a box
    AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX = jnp.int8(9) # Agent is on target on which there is a box and is carrying a box
    BOX_ON_TARGET = jnp.int8(10) # Box is on target
    AGENT_ON_BOX_CARRYING_BOX = jnp.int8(11) # Agent is on box carrying a box

    @staticmethod
    @jax.jit
    def remove_targets(grid_state: jax.Array) -> jax.Array:
        """Project grid states with targets to states without targets."""
        # Create a mapping array for vectorized lookup
        # Map each state to its corresponding no-target state
        mapping_array = jnp.array([
            0,   # EMPTY -> EMPTY
            1,   # BOX -> BOX
            0,   # TARGET -> EMPTY
            3,   # AGENT -> AGENT
            4,   # AGENT_CARRYING_BOX -> AGENT_CARRYING_BOX
            5,   # AGENT_ON_BOX -> AGENT_ON_BOX
            3,   # AGENT_ON_TARGET -> AGENT
            4,   # AGENT_ON_TARGET_CARRYING_BOX -> AGENT_CARRYING_BOX
            5,   # AGENT_ON_TARGET_WITH_BOX -> AGENT_ON_BOX
            11,  # AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX -> AGENT_ON_BOX_CARRYING_BOX
            1,   # BOX_ON_TARGET -> BOX
            11,  # AGENT_ON_BOX_CARRYING_BOX -> AGENT_ON_BOX_CARRYING_BOX
        ], dtype=jnp.int8)
        
        # Apply the mapping
        return mapping_array[grid_state]


ACTIONS = {
    0: (jnp.int8(-1), jnp.int8(0)),   # UP
    1: (jnp.int8(1), jnp.int8(0)),    # DOWN
    2: (jnp.int8(0), jnp.int8(-1)),   # LEFT
    3: (jnp.int8(0), jnp.int8(1)),    # RIGHT
    4: None,      # PICK_UP
    5: None       # PUT_DOWN
}

@dataclass
class BoxPushingConfig:
    grid_size: int = 5
    number_of_boxes_min: int = 3
    number_of_boxes_max: int = 4
    episode_length: int = 100
    truncate_when_success: bool = False



class BoxPushingEnv:
    """JAX-based box pushing environment."""

    # TODO: I should define here a maximum and minimum number of boxes, so that every env during reset gets different number of them
    #  also, I need to add an argument that defines the number of boxes that need to be on target from start
    def __init__(self, grid_size: int = 20, episode_length: int = 2000, number_of_boxes_min: int = 3, number_of_boxes_max:int=4, truncate_when_success: bool = False, **kwargs):
        # logger.info(f"Initializing BoxPushingEnv with grid_size={grid_size}, episode_length={episode_length}, number_of_boxes={number_of_boxes_min}, {number_of_boxes_max}")
        self.grid_size = grid_size
        self.episode_length = episode_length
        self.action_space = 6  # UP, DOWN, LEFT, RIGHT, PICK_UP, PUT_DOWN
        self.number_of_boxes_min = number_of_boxes_min
        self.number_of_boxes_max = number_of_boxes_max
        self.truncate_when_success = truncate_when_success

    def reset(self, key: jax.Array) -> Tuple[BoxPushingState, Dict[str, Any]]:
        """Reset environment to initial state."""
        box_pos_key, number_of_boxes_key, agent_key, targets_key, state_key = random.split(key, 5)
        
        # Initialize empty grid
        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int8)

        number_of_boxes = jax.random.randint(number_of_boxes_key, (), self.number_of_boxes_min, self.number_of_boxes_max+1)

        # Place exactly number_of_boxes boxes
        box_positions = jax.random.choice(
            box_pos_key,
            self.grid_size * self.grid_size,
            shape=(self.number_of_boxes_max,),
            replace=False
        )
        box_rows = box_positions // self.grid_size
        box_cols = box_positions % self.grid_size

        # Workaround for dynamic indexing: use mask to set only the first number_of_boxes
        mask = jnp.arange(self.number_of_boxes_max) < number_of_boxes
        # Set boxes only where mask is True
        grid = jax.lax.fori_loop(
            0, self.number_of_boxes_max,
            lambda i, g: jax.lax.cond(
                mask[i],
                lambda _: g.at[box_rows[i], box_cols[i]].set(GridStatesEnum.BOX),
                lambda _: g,
                operand=None
            ),
            grid
        )

        target_cells = self._generate_target_cells(targets_key, number_of_boxes)
        # Only use target cells with non-negative indices (filter out any padding rows with -1)
        valid_mask = jnp.all(target_cells >= 0, axis=1)
        valid_target_cells = target_cells[valid_mask]
        target_rows = valid_target_cells[:, 0]
        target_cols = valid_target_cells[:, 1]
        target_cell_values = grid[target_rows, target_cols]

        # Create mask for boxes at target positions
        box_at_target_mask = target_cell_values == GridStatesEnum.BOX

        # Update grid using vectorized operations
        grid = grid.at[target_rows, target_cols].set(
            jnp.where(box_at_target_mask, GridStatesEnum.BOX_ON_TARGET, GridStatesEnum.TARGET)
        )
        # Place agent randomly
        agent_pos = random.randint(agent_key, (2,), 0, self.grid_size)

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
            key=state_key,
            grid=grid,
            agent_pos=agent_pos,
            agent_has_box=False,
            steps=0,
            target_cells=target_cells,
            number_of_boxes=number_of_boxes,
            goal=jnp.zeros_like(grid),
            reward=0
        )
        goal = self.create_solved_state(state)
        state = state.replace(goal=goal.grid)
        
        info = {
            'boxes_on_target': jnp.sum(grid == GridStatesEnum.BOX_ON_TARGET) + jnp.sum(grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX)
        }
        
        return state, info
    
    def _generate_target_cells(self, key: jax.Array, number_of_boxes: int) -> jax.Array:
        """Generate target cells in right quarter, padded with -1 to number_of_boxes_max.

        This version avoids using dynamic_slice with traced (non-static) shapes, which is not allowed under JAX's vmap/jit.
        Instead, it uses boolean masking and concatenation to select the first number_of_boxes rows.
        """
        # Create candidate cells
        rows = jnp.arange(0, self.grid_size)
        cols = jnp.arange(0, self.grid_size)
        row_grid, col_grid = jnp.meshgrid(rows, cols, indexing='ij')
        candidates = jnp.stack([row_grid.flatten(), col_grid.flatten()], axis=1)  # (N, 2)

        # Shuffle candidates
        shuffled_indices = random.permutation(key, len(candidates))
        candidates = candidates[shuffled_indices]

        # Select the first number_of_boxes rows using masking and concatenation
        # Create a mask for the first number_of_boxes elements
        mask = jnp.arange(self.number_of_boxes_max) < number_of_boxes
        # Pad candidates to at least number_of_boxes_max rows
        pad_len = self.number_of_boxes_max - candidates.shape[0]
        candidates_padded = jnp.pad(candidates, ((0, max(0, pad_len)), (0, 0)), mode='constant', constant_values=-1)
        # Take the first number_of_boxes_max rows
        candidates_max = candidates_padded[:self.number_of_boxes_max]
        # For rows >= number_of_boxes, set to -1
        selected_padded = jnp.where(mask[:, None], candidates_max, -jnp.ones_like(candidates_max))
        return selected_padded
    
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
        if self.truncate_when_success:
            done = (new_steps >= self.episode_length) | self._is_goal_reached(new_grid, state.number_of_boxes)
        else:
            done = new_steps >= self.episode_length

        reward = self._is_goal_reached(new_grid, state.number_of_boxes).astype(jnp.int32)
        
        new_state = BoxPushingState(
            key=state.key,
            grid=new_grid,
            agent_pos=new_pos,
            agent_has_box=new_agent_has_box,
            steps=new_steps,
            target_cells=state.target_cells,
            number_of_boxes=state.number_of_boxes,
            goal=state.goal,
            reward=reward
        )
        
        info = {
            'boxes_on_target': jnp.sum(new_grid == GridStatesEnum.BOX_ON_TARGET) + jnp.sum(new_grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX)
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
                ]).astype(jnp.int8).argmax(),
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
                    ]).astype(jnp.int8).argmax(),
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
                    ]).astype(jnp.int8).argmax(),
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
            current_cell == GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX,
            putdown_valid,
            putdown_invalid
        )
        return new_grid, new_agent_has_box

    def _is_goal_reached(self, grid: jax.Array, number_of_boxes: int) -> bool:
        """Check if all boxes are in target cells."""
        return jnp.sum(grid == GridStatesEnum.BOX_ON_TARGET) + jnp.sum(grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX) == number_of_boxes
    
    def _get_reward(self, grid: jax.Array) -> float:
        """Get reward for the current state."""
        return jnp.sum(grid == GridStatesEnum.BOX_ON_TARGET) + jnp.sum(grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX) 

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
                    user_input = input("Enter action (w/s/a/d/e/r/q/g): ").lower().strip()
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
                    elif user_input == 'g':
                        print("Game restarted by user.")
                        action= "restart"
                    else:
                        print("Invalid input. Use w/s/a/d/e/r/q/g")
                except (EOFError, KeyboardInterrupt):
                    print("\nGame ended by user.")
                    return
            
            # Take action
            if action=="restart":
                state, info = self.reset(state.key)
            else:
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
        valid_mask = jnp.all(state.target_cells >= 0, axis=1)
        valid_target_cells = state.target_cells[valid_mask]
        target_rows = valid_target_cells[:, 0]
        target_cols = valid_target_cells[:, 1]
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
    
    def get_dummy_timestep(self, key):
        dummy_timestep = TimeStep(
            key=key,
            grid=jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int8),
            target_cells=jnp.zeros((self.number_of_boxes_max, 2), dtype=jnp.int8),
            number_of_boxes=jnp.zeros((1,), dtype=jnp.int8),
            agent_pos=jnp.zeros((2,), dtype=jnp.int8),
            agent_has_box=jnp.zeros((1,), dtype=jnp.int8),
            steps=jnp.zeros((1,), dtype=jnp.int8),
            action=jnp.zeros((1,), dtype=jnp.int8),
            goal=jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int8),
            reward=jnp.zeros((1,), dtype=jnp.int8),
            done=jnp.zeros((1,), dtype=jnp.int8),
        )

        return dummy_timestep

    @staticmethod
    def animate(ax, timesteps, frame, img_prefix='assets'):
        ax.clear()
        grid_state = timesteps.grid[0, frame]
        action = timesteps.action[0, frame]
        reward = timesteps.reward[0, frame]

        # Create color mapping for grid states
        imgs = {
            0: 'floor.png',                                  # EMPTY
            1: 'box.png',                                    # BOX
            2: 'box_target.png',                             # TARGET
            3: 'agent.png',                                  # AGENT
            4: 'agent_carrying_box.png',                     # AGENT_CARRYING_BOX
            5: 'agent_on_box.png',                           # AGENT_ON_BOX
            6: 'agent_on_target.png',                        # AGENT_ON_TARGET
            7: 'agent_on_target_carrying_box.png',           # AGENT_ON_TARGET_CARRYING_BOX
            8: 'agent_on_target_with_box.png',               # AGENT_ON_TARGET_WITH_BOX
            9: 'agent_on_target_with_box_carrying_box.png',  # AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX
            10: 'box_on_target.png',                         # BOX_ON_TARGET
            11: 'agent_on_box_carrying_box.png'              # AGENT_ON_BOX_CARRYING_BOX
        }

        # Plot grid
        for i in range(grid_state.shape[0]):
            for j in range(grid_state.shape[1]):
                img_name = imgs[int(grid_state[i, j])]
                img_path = os.path.join(img_prefix, img_name)
                img = matplotlib.image.imread(img_path)
                ax.imshow(img, extent = [i+1, i, j+1, j])


        ax.set_xlim(0, grid_state.shape[1])
        ax.set_ylim(0, grid_state.shape[0])
        ax.set_title(f'Step {frame} | Action: {action} | Reward: {reward:.2f}')
        ax.set_aspect('equal')
        ax.invert_yaxis()


class Wrapper(BoxPushingEnv):
    def __init__(self, env: BoxPushingEnv):
        self._env = env
        # Copy attributes from wrapped environment
        for attr in ['grid_size', 'episode_length', 'number_of_boxes_min', 'number_of_boxes_max']:
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
            return reset_state
        
        state = jax.lax.cond(
            done,
            lambda: reset_fn(key_new),
            lambda: state
        )
        return state, reward, done, info


if __name__ == "__main__":
    env = BoxPushingEnv(grid_size=5, number_of_boxes_max=5, number_of_boxes_min=3)
    key = jax.random.PRNGKey(0)
    env.play_game(key)