import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Dict, Any
from dataclasses import dataclass
import chex
from flax import struct
import matplotlib
import os
import logging

class BoxPushingState(struct.PyTreeNode):
    """State representation for the box pushing environment."""
    key: jax.Array
    grid: jax.Array  # N x N grid where 0=empty, 1=box
    agent_pos: jax.Array  # [row, col] position of agent
    agent_has_box: jax.Array  # Whether agent is carrying a box
    steps: jax.Array  # Current step count
    number_of_boxes: jax.Array  # Number of boxes
    goal: jax.Array  # Goal cell coordinates for boxes
    reward: jax.Array  # Reward for the current step
    success: jax.Array # Whether all boxes are on targets


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
            0,   # 0 EMPTY -> EMPTY 0
            1,   # 1 BOX -> BOX 1
            0,   # 2 TARGET -> EMPTY 0
            3,   # 3 AGENT -> AGENT 3
            4,   # 4 AGENT_CARRYING_BOX -> AGENT_CARRYING_BOX 4
            5,   # 5 AGENT_ON_BOX -> AGENT_ON_BOX 5
            3,   # 6 AGENT_ON_TARGET -> AGENT 3
            4,   # 7 AGENT_ON_TARGET_CARRYING_BOX -> AGENT_CARRYING_BOX 4
            5,   # 8 AGENT_ON_TARGET_WITH_BOX -> AGENT_ON_BOX 5
            11,  # 9 AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX -> AGENT_ON_BOX_CARRYING_BOX
            1,   # 10 BOX_ON_TARGET -> BOX
            11,  # 11 AGENT_ON_BOX_CARRYING_BOX -> AGENT_ON_BOX_CARRYING_BOX
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
    number_of_moving_boxes_max: int = 2
    episode_length: int = 100
    truncate_when_success: bool = False
    dense_rewards: bool = False
    level_generator: str = 'default'
    generator_mirroring: bool = False

def create_solved_state(state: BoxPushingState) -> BoxPushingState:
    """Create a solved state."""
    # Change all target cells to box on target
    target_mask = (state.grid == GridStatesEnum.TARGET) | (state.grid == GridStatesEnum.BOX_ON_TARGET)
    state = state.replace(
        grid=jnp.where(target_mask, GridStatesEnum.BOX_ON_TARGET, state.grid)
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



class DefaultLevelGenerator:
    def __init__(self, grid_size, number_of_boxes_min, number_of_boxes_max, number_of_moving_boxes_max):
        self.grid_size = grid_size
        self.number_of_boxes_min = number_of_boxes_min
        self.number_of_boxes_max = number_of_boxes_max
        self.number_of_moving_boxes_max = number_of_moving_boxes_max

    def place_agent(self, grid, agent_key):
        agent_pos = random.randint(agent_key, (2,), 0, self.grid_size)
        current_cell = grid[agent_pos[0], agent_pos[1]]

        #TODO: This is ugly, maybe it can be refactored somehow ?
        agent_state = jax.lax.cond(
            current_cell == GridStatesEnum.EMPTY,
            lambda: GridStatesEnum.AGENT,
            lambda: jax.lax.cond(
                current_cell == GridStatesEnum.BOX,
                lambda: GridStatesEnum.AGENT_ON_BOX,
                lambda: jax.lax.cond(
                    current_cell == GridStatesEnum.TARGET,
                    lambda: GridStatesEnum.AGENT_ON_TARGET,
                    lambda: jax.lax.cond(
                        current_cell == GridStatesEnum.BOX_ON_TARGET,
                        lambda: GridStatesEnum.AGENT_ON_TARGET_WITH_BOX,
                        lambda: GridStatesEnum.AGENT_ON_TARGET_WITH_BOX, # This should never happen
                    )
                )
            )
        )

        new_grid = grid.at[agent_pos[0], agent_pos[1]].set(agent_state)
        return agent_pos, new_grid

    def generate(self, key) -> BoxPushingState:
        permutation_key, number_of_boxes_key, agent_key, state_key = random.split(key, 4)

        idxs = jnp.arange(self.grid_size * self.grid_size)

        number_of_boxes = jax.random.randint(number_of_boxes_key, (), self.number_of_boxes_min, self.number_of_boxes_max+1)
        number_of_boxes_on_target = jnp.maximum(0, number_of_boxes - self.number_of_moving_boxes_max)
        number_of_targets_without_boxes = number_of_boxes - number_of_boxes_on_target

        is_fixed = idxs < number_of_boxes_on_target
        is_box = (idxs >= number_of_boxes_on_target) & (idxs < number_of_boxes)
        is_target = (idxs >= number_of_boxes) & (idxs < number_of_boxes + number_of_targets_without_boxes)

        # TODO: this could be made more efficient by only looping over indicies up to 2 * number_of_boxes_max, and then concatenating with empty rest of the board
        # this shouldn't matter with board sizes we work now, but worth remembering in the future
        grid = jnp.piecewise(
            idxs,
            [is_fixed, is_box, is_target],
            [GridStatesEnum.BOX_ON_TARGET, GridStatesEnum.BOX, GridStatesEnum.TARGET, GridStatesEnum.EMPTY]
        ).astype(jnp.int8)

        grid = jax.random.permutation(permutation_key, grid)
        grid = grid.reshape((self.grid_size, self.grid_size))

        # Agent is placed at any field randomly
        agent_pos, grid = self.place_agent(grid, agent_key)

        state = BoxPushingState(
            key=state_key,
            grid=grid,
            agent_pos=agent_pos,
            agent_has_box=False,
            steps=0,
            number_of_boxes=number_of_boxes,
            goal=jnp.zeros_like(grid),
            reward=0,
            success=0,
        )


        goal = create_solved_state(state)
        state = state.replace(goal=goal.grid)

        return state


# This generator puts targets in one quarter of the board, and boxes in another quarter
class QuarterGenerator(DefaultLevelGenerator):
    def __init__(self, grid_size, number_of_boxes_min, number_of_boxes_max, number_of_moving_boxes_max, mirror=False):
        # This is mostly for convenience, without it there would have to be a lot of if statements
        assert grid_size % 2 == 0
        assert number_of_boxes_max <= grid_size * grid_size / 4
        self.mirror = mirror

        super().__init__(grid_size, number_of_boxes_min, number_of_boxes_max, number_of_moving_boxes_max)


    def generate_box_quarter(self, number_of_boxes, key):
        quarter_size = self.grid_size // 2
        quarter = jnp.full(quarter_size * quarter_size, GridStatesEnum.EMPTY)
        idxs = jnp.arange(quarter_size * quarter_size)

        is_box = idxs < number_of_boxes
    
        quarter = jnp.piecewise(
            idxs,
            [is_box],
            [GridStatesEnum.BOX, GridStatesEnum.EMPTY]
        ).astype(jnp.int8)

        quarter = jax.random.permutation(key, quarter)
        quarter = quarter.reshape((quarter_size, quarter_size))

        return quarter
    

    def generate_target_quarter(self, number_of_boxes_on_target, number_of_targets_without_boxes, key):
        quarter_size = self.grid_size // 2
        quarter = jnp.full(quarter_size * quarter_size, GridStatesEnum.EMPTY)
        number_of_targets = number_of_boxes_on_target + number_of_targets_without_boxes
        idxs = jnp.arange(quarter_size * quarter_size)

        is_box_on_target = idxs < number_of_boxes_on_target
        is_target_without_box = (idxs >= number_of_boxes_on_target) & (idxs < number_of_targets)

        quarter = jnp.piecewise(
            idxs,
            [is_box_on_target, is_target_without_box],
            [GridStatesEnum.BOX_ON_TARGET, GridStatesEnum.TARGET, GridStatesEnum.EMPTY],
        ).astype(jnp.int8)

        quarter = jax.random.permutation(key, quarter)
        quarter = quarter.reshape((quarter_size, quarter_size))

        return quarter

    def generate(self, key):
        box_quarter_key, target_quarter_key , permutation_3_key, number_of_boxes_key, agent_key, state_key = random.split(key, 6)

        number_of_boxes = jax.random.randint(number_of_boxes_key, (), self.number_of_boxes_min, self.number_of_boxes_max+1)
        number_of_boxes_on_target = jnp.maximum(0, number_of_boxes - self.number_of_moving_boxes_max)
        number_of_targets_without_boxes = number_of_boxes - number_of_boxes_on_target

        # We set quarter with boxes
        box_quarter = self.generate_box_quarter(number_of_targets_without_boxes, box_quarter_key)

        # We set quarter with targets     
        target_quarter = self.generate_target_quarter(number_of_boxes_on_target, number_of_targets_without_boxes, target_quarter_key)

        # 2 empty quarters
        empty_quarter_1 = jnp.full_like(box_quarter, GridStatesEnum.EMPTY)
        empty_quarter_2 = jnp.full_like(box_quarter, GridStatesEnum.EMPTY)

        # We shuffle all quarters and concatenate them into the full grid
        if self.mirror:
            block_grid = jnp.stack([empty_quarter_1, box_quarter, target_quarter, empty_quarter_2])
        else:
            block_grid = jnp.stack([box_quarter, empty_quarter_1, empty_quarter_2, target_quarter])
        
        permuted_grid = block_grid
        
        top = jnp.concatenate([permuted_grid[0], permuted_grid[1]], axis=1)
        bottom = jnp.concatenate([permuted_grid[2], permuted_grid[3]], axis=1)
        grid = jnp.concatenate([top, bottom], axis=0)

        agent_pos, grid = self.place_agent(grid, agent_key)

        state = BoxPushingState(
            key=state_key,
            grid=grid,
            agent_pos=agent_pos,
            agent_has_box=False,
            steps=0,
            number_of_boxes=number_of_boxes,
            goal=jnp.zeros_like(grid),
            reward=0,
            success=0,
        )


        goal = create_solved_state(state)
        state = state.replace(goal=goal.grid)

        return state
        

class BoxPushingEnv:
    """JAX-based box pushing environment."""

    # TODO: I should define here a maximum and minimum number of boxes, so that every env during reset gets different number of them
    #  also, I need to add an argument that defines the number of boxes that need to be on target from start
    def __init__(self, grid_size: int = 20, episode_length: int = 2000, number_of_boxes_min: int = 3, number_of_boxes_max:int=4, number_of_moving_boxes_max:int=2, truncate_when_success: bool = False, dense_rewards: bool = False, level_generator: str = 'default', **kwargs):
        logging.info(f"Initializing BoxPushingEnv with grid_size={grid_size}, episode_length={episode_length}, number_of_boxes={number_of_boxes_min}, number_of_boxes_max={number_of_boxes_max}, number_of_moving_boxes_max={number_of_moving_boxes_max}")
        self.grid_size = grid_size
        self.episode_length = episode_length
        self.action_space = 6  # UP, DOWN, LEFT, RIGHT, PICK_UP, PUT_DOWN
        self.number_of_boxes_min = number_of_boxes_min
        self.number_of_boxes_max = number_of_boxes_max
        self.number_of_moving_boxes_max = number_of_moving_boxes_max
        self.truncate_when_success = truncate_when_success
        self.dense_rewards = dense_rewards

        if level_generator == 'default':
            self.level_generator = DefaultLevelGenerator(grid_size, number_of_boxes_min, number_of_boxes_max, number_of_moving_boxes_max)
        elif level_generator == 'quarter':
            self.level_generator = QuarterGenerator(grid_size, number_of_boxes_min, number_of_boxes_max, number_of_moving_boxes_max, mirror=kwargs['generator_mirroring'])
        else:
            raise ValueError("Unknown level generator selected")

    def reset(self, key: jax.Array) -> Tuple[BoxPushingState, Dict[str, Any]]:
        """Reset environment to initial state."""
        state = self.level_generator.generate(key)
        grid = state.grid
        
        info = {
            'boxes_on_target': jnp.sum(grid == GridStatesEnum.BOX_ON_TARGET) + jnp.sum(grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX) + jnp.sum(grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX)
        }
        
        return state, info
    
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

        reward = self._get_reward(new_grid, state.number_of_boxes)
        success = self._is_goal_reached(new_grid, state.number_of_boxes).astype(jnp.int32)
        
        new_state = BoxPushingState(
            key=state.key,
            grid=new_grid,
            agent_pos=new_pos,
            agent_has_box=new_agent_has_box,
            steps=new_steps,
            number_of_boxes=state.number_of_boxes,
            goal=state.goal,
            reward=reward,
            success=success,
        )
        
        info = {
            'boxes_on_target': jnp.sum(new_grid == GridStatesEnum.BOX_ON_TARGET) + jnp.sum(new_grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX) + jnp.sum(new_grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX)
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
    
    def _get_reward(self, grid: jax.Array, number_of_boxes: int) -> float:
        """Get reward for the current state."""
        boxes_on_targets = jnp.sum(grid == GridStatesEnum.BOX_ON_TARGET) + jnp.sum(grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX) + jnp.sum(grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX)
        if self.dense_rewards:
            return boxes_on_targets
        else:
            return (boxes_on_targets == number_of_boxes).astype(jnp.int32)

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

    
    def get_dummy_timestep(self, key):
        dummy_timestep = TimeStep(
            key=key,
            grid=jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int8),
            number_of_boxes=jnp.zeros((1,), dtype=jnp.int8),
            agent_pos=jnp.zeros((2,), dtype=jnp.int8),
            agent_has_box=jnp.zeros((1,), dtype=jnp.int8),
            steps=jnp.zeros((1,), dtype=jnp.int8),
            action=jnp.zeros((1,), dtype=jnp.int8),
            goal=jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int8),
            reward=jnp.zeros((1,), dtype=jnp.int8),
            success=jnp.zeros((1,), dtype=jnp.int8),
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
        for attr in ['grid_size', 'episode_length', 'number_of_boxes_min', 'number_of_boxes_max', 'number_of_moving_boxes_max']:
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
    env = BoxPushingEnv(grid_size=6, number_of_boxes_max=3, number_of_boxes_min=3, number_of_moving_boxes_max=2, level_generator='quarter', generator_mirroring=False)
    key = jax.random.PRNGKey(0)
    env.play_game(key)
