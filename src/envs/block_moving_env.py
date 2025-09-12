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
    success: jax.Array  # Whether all boxes are on targets
    extras: Dict  # Field for extra information used by filtering wrappers


class TimeStep(BoxPushingState):
    action: jax.Array
    done: jax.Array
    truncated: jax.Array


@dataclass
class GridStatesEnum:
    """Grid states representation for the box pushing environment."""

    EMPTY = jnp.int8(0)
    BOX = jnp.int8(1)
    TARGET = jnp.int8(2)
    AGENT = jnp.int8(3)
    AGENT_CARRYING_BOX = jnp.int8(4)  # Agent is carrying a box
    AGENT_ON_BOX = jnp.int8(5)  # Agent is on box
    AGENT_ON_TARGET = jnp.int8(6)  # Agent is on target
    AGENT_ON_TARGET_CARRYING_BOX = jnp.int8(7)  # Agent is on target carrying the box
    AGENT_ON_TARGET_WITH_BOX = jnp.int8(8)  # Agent is on target on which there is a box
    AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX = jnp.int8(9)  # noqa: E501 Agent is on target on which there is a box and is carrying a box
    BOX_ON_TARGET = jnp.int8(10)  # Box is on target
    AGENT_ON_BOX_CARRYING_BOX = jnp.int8(11)  # Agent is on box carrying a box


REMOVE_TARGETS_DICT = {
    int(GridStatesEnum.EMPTY): int(GridStatesEnum.EMPTY),
    int(GridStatesEnum.BOX): int(GridStatesEnum.BOX),
    int(GridStatesEnum.TARGET): int(GridStatesEnum.EMPTY),  # map TARGET -> EMPTY
    int(GridStatesEnum.AGENT): int(GridStatesEnum.AGENT),
    int(GridStatesEnum.AGENT_CARRYING_BOX): int(GridStatesEnum.AGENT_CARRYING_BOX),
    int(GridStatesEnum.AGENT_ON_BOX): int(GridStatesEnum.AGENT_ON_BOX),
    int(GridStatesEnum.AGENT_ON_TARGET): int(GridStatesEnum.AGENT),  # example
    int(GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX): int(GridStatesEnum.AGENT_CARRYING_BOX),
    int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX): int(GridStatesEnum.AGENT_ON_BOX),
    int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX): int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX),
    int(GridStatesEnum.BOX_ON_TARGET): int(GridStatesEnum.BOX),
    int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX): int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX),
}

REMOVE_AGENT_DICT = {
    int(GridStatesEnum.EMPTY): int(GridStatesEnum.EMPTY),
    int(GridStatesEnum.BOX): int(GridStatesEnum.BOX),
    int(GridStatesEnum.TARGET): int(GridStatesEnum.TARGET),
    int(GridStatesEnum.AGENT): int(GridStatesEnum.EMPTY),  # agent standing on empty -> empty
    int(GridStatesEnum.AGENT_CARRYING_BOX): int(GridStatesEnum.EMPTY),  # carried box not on grid -> empty
    int(GridStatesEnum.AGENT_ON_BOX): int(GridStatesEnum.BOX),  # agent on box -> box remains
    int(GridStatesEnum.AGENT_ON_TARGET): int(GridStatesEnum.TARGET),  # agent on target -> target remains
    int(GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX): int(GridStatesEnum.TARGET),  # carrying box not on grid -> target
    int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX): int(GridStatesEnum.BOX_ON_TARGET),  # box on target stays
    int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX): int(GridStatesEnum.BOX_ON_TARGET),  # noqa: E501 same: box on target stays
    int(GridStatesEnum.BOX_ON_TARGET): int(GridStatesEnum.BOX_ON_TARGET),
    int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX): int(GridStatesEnum.BOX),  # noqa: E501 agent standing on a box (and also carrying one) -> box stays
}

ADD_AGENT_DICT = {
    int(GridStatesEnum.EMPTY): int(GridStatesEnum.AGENT),  # empty -> agent
    int(GridStatesEnum.BOX): int(GridStatesEnum.AGENT_ON_BOX),  # box -> agent on box
    int(GridStatesEnum.TARGET): int(GridStatesEnum.AGENT_ON_TARGET),  # target -> agent on target
    int(GridStatesEnum.BOX_ON_TARGET): int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX),  # noqa: E501 box on target -> agent on that box+target
}

EMPTY_VAL = int(GridStatesEnum.EMPTY)
max_key = max(REMOVE_AGENT_DICT.keys())
_REMOVE_AGENT_ARRAY = jnp.array([REMOVE_AGENT_DICT.get(i, EMPTY_VAL) for i in range(max_key + 1)], dtype=jnp.int8)
_ADD_AGENT_ARRAY = jnp.array([ADD_AGENT_DICT.get(i, EMPTY_VAL) for i in range(max_key + 1)], dtype=jnp.int8)


PICK_UP_DICT = {
    int(GridStatesEnum.AGENT_ON_BOX): int(GridStatesEnum.AGENT_CARRYING_BOX),
    int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX): int(GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX),
}

PUT_DOWN_DICT = {v: k for k, v in PICK_UP_DICT.items()}

EMPTY_VAL = int(GridStatesEnum.EMPTY)
max_key = max(REMOVE_TARGETS_DICT.keys())
# array size must be max_key + 1 so that numeric enum values are usable as indices
_MAPPING_ARRAY = jnp.array([REMOVE_TARGETS_DICT.get(i, EMPTY_VAL) for i in range(max_key + 1)], dtype=jnp.int8)


@jax.jit
def remove_targets(grid_state: jax.Array) -> jax.Array:
    """Project grid states with targets to states without targets using a pre-built mapping array."""
    # ensure integer indexing dtype
    return _MAPPING_ARRAY[grid_state.astype(jnp.int8)]


ACTIONS = {
    0: (jnp.int8(-1), jnp.int8(0)),  # UP
    1: (jnp.int8(1), jnp.int8(0)),  # DOWN
    2: (jnp.int8(0), jnp.int8(-1)),  # LEFT
    3: (jnp.int8(0), jnp.int8(1)),  # RIGHT
    4: None,  # PICK_UP
    5: None,  # PUT_DOWN
}


@dataclass
class BoxPushingConfig:
    grid_size: int = 5
    number_of_boxes_min: int = 3
    number_of_boxes_max: int = 4
    number_of_moving_boxes_max: int = 2
    episode_length: int = 100
    terminate_when_success: bool = False
    dense_rewards: bool = False
    negative_sparse: bool = False
    level_generator: str = "default"
    generator_special: bool = False
    quarter_size: int | None = None


def calculate_number_of_boxes(grid: jax.Array):
    return int(
        jnp.sum(grid == GridStatesEnum.BOX_ON_TARGET)
        + jnp.sum(grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX)
        + jnp.sum(grid == GridStatesEnum.AGENT_CARRYING_BOX)
        + jnp.sum(grid == GridStatesEnum.AGENT_ON_BOX)
        + jnp.sum(grid == GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX)
        + 2 * jnp.sum(grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX)
        + 2 * jnp.sum(grid == GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX)
        + jnp.sum(grid == GridStatesEnum.BOX)
    )


def create_solved_state(state: BoxPushingState) -> BoxPushingState:
    """Create a solved state."""
    # Change all target cells to box on target
    target_mask = (state.grid == GridStatesEnum.TARGET) | (state.grid == GridStatesEnum.BOX_ON_TARGET)
    state = state.replace(grid=jnp.where(target_mask, GridStatesEnum.BOX_ON_TARGET, state.grid))
    # Change all boxes to empty - use where to avoid boolean indexing issue
    box_mask = state.grid == GridStatesEnum.BOX
    state = state.replace(grid=jnp.where(box_mask, GridStatesEnum.EMPTY, state.grid))

    # Check what cell the agent is currently on and update accordingly
    agent_row, agent_col = state.agent_pos[0], state.agent_pos[1]
    current_cell = state.grid[agent_row, agent_col]

    # Update grid based on current cell type
    new_cell_value = jax.lax.cond(
        current_cell == GridStatesEnum.AGENT_ON_TARGET,
        lambda: GridStatesEnum.AGENT_ON_TARGET_WITH_BOX,
        lambda: jax.lax.cond(
            current_cell == GridStatesEnum.AGENT_ON_BOX,
            lambda: GridStatesEnum.AGENT,
            lambda: current_cell,  # noqa: E501 Here goes: AGENT_ON_TARGET_WITH_BOX -> AGENT_ON_TARGET_WITH_BOX and Agent -> Agent
        ),
    )
    state = state.replace(grid=state.grid.at[agent_row, agent_col].set(new_cell_value), agent_has_box=jnp.array(False))
    return state


def find_agent_position(grid) -> jnp.array:
    actor_states = jnp.isin(
        grid,
        jnp.array(
            [
                GridStatesEnum.AGENT,
                GridStatesEnum.AGENT_ON_BOX,
                GridStatesEnum.AGENT_ON_TARGET,
                GridStatesEnum.AGENT_ON_TARGET_WITH_BOX,
            ]
        ),
    )
    idx = jnp.argmax(actor_states)

    row_idx = idx // grid.shape[1]
    col_idx = idx % grid.shape[1]

    return jnp.array([row_idx, col_idx])


class DefaultLevelGenerator:
    def __init__(self, grid_size, number_of_boxes_min, number_of_boxes_max, number_of_moving_boxes_max):
        self.grid_size = grid_size
        self.number_of_boxes_min = number_of_boxes_min
        self.number_of_boxes_max = number_of_boxes_max
        self.number_of_moving_boxes_max = number_of_moving_boxes_max

    def place_agent(self, grid, agent_key):
        agent_pos = random.randint(agent_key, (2,), 0, grid.shape[0])
        current_cell = grid[agent_pos[0], agent_pos[1]]

        # TODO: This is ugly, maybe it can be refactored somehow ?
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
                        lambda: GridStatesEnum.AGENT_ON_TARGET_WITH_BOX,  # This should never happen
                    ),
                ),
            ),
        )

        new_grid = grid.at[agent_pos[0], agent_pos[1]].set(agent_state)
        return agent_pos, new_grid

    def generate(self, key) -> BoxPushingState:
        permutation_key, number_of_boxes_key, agent_key, state_key = random.split(key, 4)

        idxs = jnp.arange(self.grid_size * self.grid_size)

        number_of_boxes = jax.random.randint(
            number_of_boxes_key, (), self.number_of_boxes_min, self.number_of_boxes_max + 1
        )
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
            [GridStatesEnum.BOX_ON_TARGET, GridStatesEnum.BOX, GridStatesEnum.TARGET, GridStatesEnum.EMPTY],
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
            reward=0.0,
            success=0,
            extras={},
        )

        goal = create_solved_state(state)
        state = state.replace(goal=goal.grid)

        return state

    def get_dummy_timestep(self, key):
        return TimeStep(
            key=key,
            grid=jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int8),
            number_of_boxes=jnp.zeros((1,), dtype=jnp.int8),
            agent_pos=jnp.zeros((2,), dtype=jnp.int8),
            agent_has_box=jnp.zeros((1,), dtype=jnp.int8),
            steps=jnp.zeros((1,), dtype=jnp.int8),
            action=jnp.zeros((1,), dtype=jnp.int8),
            goal=jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int8),
            reward=jnp.zeros((1,), dtype=jnp.float32),
            success=jnp.zeros((1,), dtype=jnp.int8),
            done=jnp.zeros((1,), dtype=jnp.int8),
            truncated=jnp.zeros((1,), dtype=jnp.int8),
            extras={},
        )


class VariableQuarterGenerator(DefaultLevelGenerator):
    def __init__(
        self,
        grid_size,
        number_of_boxes_min,
        number_of_boxes_max,
        number_of_moving_boxes_max,
        quarter_size,
        special=False,
    ):
        # This is mostly for convenience, without it there would have to be a lot of if statements

        assert number_of_boxes_max <= quarter_size * quarter_size
        assert number_of_boxes_max == number_of_boxes_max, "In this generator we assume all boxes always move"
        assert number_of_boxes_min == number_of_boxes_max, (
            "In this generator we assume there is only one possible number of boxes"
        )
        self.special = special
        self.quarter_size = quarter_size

        super().__init__(grid_size, number_of_boxes_min, number_of_boxes_max, number_of_moving_boxes_max)

    def generate_box_quarter(self, number_of_boxes, key):
        quarter = jnp.full(self.quarter_size * self.quarter_size, GridStatesEnum.EMPTY)
        idxs = jnp.arange(self.quarter_size * self.quarter_size)

        is_box = idxs < number_of_boxes

        quarter = jnp.piecewise(idxs, [is_box], [GridStatesEnum.BOX, GridStatesEnum.EMPTY]).astype(jnp.int8)

        quarter = jax.random.permutation(key, quarter)
        quarter = quarter.reshape((self.quarter_size, self.quarter_size))

        return quarter

    def place_targets_in_slice(self, grid, number_of_targets, key):
        available_indices = jnp.nonzero(grid == GridStatesEnum.EMPTY, size=grid.shape[0] * grid.shape[1], fill_value=-1)
        available_indices = jnp.stack(available_indices, axis=1)
        available_indices = jax.random.permutation(key, available_indices)

        # This is ugly, but I don't see how to achieve the same without for loop
        def f(carry, index):
            _, current_num_targets = carry

            new_carry = jax.lax.cond(
                jnp.logical_and(current_num_targets > 0, index[0] >= 0),
                lambda _grid, _curr_num_targets: (
                    _grid.at[index[0], index[1]].set(GridStatesEnum.TARGET),
                    _curr_num_targets - 1,
                ),
                lambda x, y: (x, y),
                *carry,
            )
            return new_carry, None

        final_carry, _ = jax.lax.scan(f, (grid, number_of_targets), xs=available_indices)

        return final_carry[0]

    def generate(self, key):
        box_quarter_key, target_quarter_key, permutation_3_key, number_of_boxes_key, agent_key, state_key = (
            jax.random.split(key, 6)
        )

        number_of_boxes = self.number_of_boxes_max

        if self.special:
            corners = jnp.array(
                [
                    [0, 3],
                    [3, 0],
                    [1, 2],
                    [2, 1],
                ]
            )
        else:
            corners = jnp.array(
                [
                    [0, 1],
                    [1, 0],
                    [0, 2],
                    [2, 0],
                    [1, 3],
                    [3, 1],
                    [2, 3],
                    [3, 2],
                ]
            )

        corners = jax.random.choice(permutation_3_key, corners)
        corners_left_upper = jnp.array(
            [
                [0, 0],
                [0, self.grid_size - self.quarter_size],
                [self.grid_size - self.quarter_size, 0],
                [self.grid_size - self.quarter_size, self.grid_size - self.quarter_size],
            ]
        )

        box_corner = corners_left_upper[corners[0]]
        target_corner = corners_left_upper[corners[1]]

        grid = jnp.full(self.grid_size * self.grid_size, GridStatesEnum.EMPTY).reshape(self.grid_size, self.grid_size)
        box_slice = self.generate_box_quarter(self.number_of_boxes_max, box_quarter_key)

        grid = jax.lax.dynamic_update_slice(grid, box_slice, box_corner)
        target_slice = jax.lax.dynamic_slice(grid, target_corner, (self.quarter_size, self.quarter_size))
        target_slice = self.place_targets_in_slice(target_slice, self.number_of_boxes_max, target_quarter_key)

        grid = jax.lax.dynamic_update_slice(grid, target_slice, target_corner)

        possible_agent_placements = jax.lax.dynamic_slice(grid, box_corner, (self.quarter_size, self.quarter_size))
        agent_pos, updated_grid_slice = self.place_agent(possible_agent_placements, agent_key)
        grid = jax.lax.dynamic_update_slice(grid, updated_grid_slice, box_corner)

        agent_pos = agent_pos + box_corner

        fields_allowed = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.bool)
        allowed_mask = jnp.ones((self.quarter_size, self.quarter_size), dtype=jnp.bool)
        fields_allowed = jax.lax.dynamic_update_slice(fields_allowed, allowed_mask, box_corner)
        fields_allowed = jax.lax.dynamic_update_slice(fields_allowed, allowed_mask, target_corner)

        state = BoxPushingState(
            key=state_key,
            grid=grid,
            agent_pos=agent_pos,
            agent_has_box=False,
            steps=0,
            number_of_boxes=number_of_boxes,
            goal=jnp.zeros_like(grid),
            reward=0.0,
            success=0,
            extras={"fields_allowed": fields_allowed},
        )

        goal = create_solved_state(state)
        state = state.replace(goal=goal.grid)

        return state

    def get_dummy_timestep(self, key):
        default_dummy_timestep = super().get_dummy_timestep(key)
        return default_dummy_timestep.replace(
            extras={"fields_allowed": jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.bool)}
        )


class BoxPushingEnv:
    """JAX-based box pushing environment."""

    # TODO: I should define here a maximum and minimum number of boxes, so that every env during reset gets different number of them
    #  also, I need to add an argument that defines the number of boxes that need to be on target from start
    def __init__(
        self,
        grid_size: int = 20,
        episode_length: int = 2000,
        number_of_boxes_min: int = 3,
        number_of_boxes_max: int = 4,
        number_of_moving_boxes_max: int = 2,
        terminate_when_success: bool = False,
        dense_rewards: bool = False,
        negative_sparse: bool = True,
        level_generator: str = "default",
        quarter_size: int | None = None,
        **kwargs,
    ):
        logging.info(
            f"Initializing BoxPushingEnv with grid_size={grid_size}, episode_length={episode_length}, number_of_boxes={number_of_boxes_min}, number_of_boxes_max={number_of_boxes_max}, number_of_moving_boxes_max={number_of_moving_boxes_max}"
        )
        self.grid_size = grid_size
        self.episode_length = episode_length
        self.action_space = 6  # UP, DOWN, LEFT, RIGHT, PICK_UP, PUT_DOWN
        self.number_of_boxes_min = number_of_boxes_min
        self.number_of_boxes_max = number_of_boxes_max
        self.number_of_moving_boxes_max = number_of_moving_boxes_max
        self.terminate_when_success = terminate_when_success
        self.dense_rewards = dense_rewards
        self.negative_sparse = negative_sparse
        self.quarter_size = quarter_size or self.grid_size // 2

        if level_generator == "default":
            self.level_generator = DefaultLevelGenerator(
                grid_size, number_of_boxes_min, number_of_boxes_max, number_of_moving_boxes_max
            )
        elif level_generator == "variable":
            self.level_generator = VariableQuarterGenerator(
                grid_size,
                number_of_boxes_min,
                number_of_boxes_max,
                number_of_moving_boxes_max,
                self.quarter_size,
                special=kwargs["generator_special"],
            )
        else:
            raise ValueError("Unknown level generator selected")

    def reset(self, key: jax.Array) -> Tuple[BoxPushingState, Dict[str, Any]]:
        """Reset environment to initial state."""
        state = self.level_generator.generate(key)
        grid = state.grid

        info = {
            "truncated": jnp.bool_(False),
            "boxes_on_target": jnp.sum(grid == GridStatesEnum.BOX_ON_TARGET)
            + jnp.sum(grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX)
            + jnp.sum(grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX),
        }

        return state, info

    def step(self, state: BoxPushingState, action: int) -> Tuple[BoxPushingState, float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        chex.assert_shape(action, ())

        # Increment steps
        new_steps = state.steps + 1

        reward = 0.0
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
            operand=None,
        )

        new_pos, new_grid, new_agent_has_box = action_result

        truncated = new_steps >= self.episode_length
        # Checking if agent reaches the goal in next state (new_grid), if yes then success = 1
        reward = BoxPushingEnv.get_reward(state.grid, new_grid, state.goal).astype(jnp.float32)
        success = reward.astype(jnp.int32)
        if self.terminate_when_success:
            done = success.astype(bool)

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
            extras=state.extras,
        )

        info = {
            "truncated": truncated,
            "boxes_on_target": jnp.sum(new_grid == GridStatesEnum.BOX_ON_TARGET)
            + jnp.sum(new_grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX)
            + jnp.sum(new_grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX),
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
            (new_row >= 0) & (new_row < self.grid_size) & (new_col >= 0) & (new_col < self.grid_size)
            # ((grid[new_row, new_col] == GridStatesEnum.EMPTY) | (grid[new_row, new_col] == GridStatesEnum.TARGET) | (grid[new_row, new_col] == GridStatesEnum.BOX))
        )

        # Use jax.lax.cond instead of if statement to handle traced arrays
        def move_valid():
            # Check if agent was on box or target or box on target before clearing
            grid_after_clear = jax.lax.switch(
                jnp.array(
                    [
                        grid[row, col] == GridStatesEnum.AGENT_ON_BOX,
                        grid[row, col] == GridStatesEnum.AGENT_ON_TARGET,
                        grid[row, col] == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX,
                        grid[row, col] == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX,
                        grid[row, col] == GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX,
                        grid[row, col] == GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX,
                        grid[row, col] == GridStatesEnum.AGENT,
                        grid[row, col] == GridStatesEnum.AGENT_CARRYING_BOX,
                    ]
                )
                .astype(jnp.int8)
                .argmax(),
                [
                    lambda: grid.at[row, col].set(GridStatesEnum.BOX),  # Leave box if agent was on box
                    lambda: grid.at[row, col].set(GridStatesEnum.TARGET),  # Leave target if agent was on target
                    lambda: grid.at[row, col].set(
                        GridStatesEnum.BOX_ON_TARGET
                    ),  # Leave box on target if agent was on box on target
                    lambda: grid.at[row, col].set(
                        GridStatesEnum.BOX_ON_TARGET
                    ),  # Leave box on target if agent was on box on target
                    lambda: grid.at[row, col].set(GridStatesEnum.TARGET),  # Leave target if agent was on target
                    lambda: grid.at[row, col].set(
                        GridStatesEnum.BOX
                    ),  # Leave box on target if agent was on box on target
                    lambda: grid.at[row, col].set(GridStatesEnum.EMPTY),  # Clear if agent was just on empty cell
                    lambda: grid.at[row, col].set(GridStatesEnum.EMPTY),  # Clear if agent was just on empty cell
                ],
            )

            # Check if agent is now on box, target, or empty cell
            new_grid = jax.lax.cond(
                state.agent_has_box,
                lambda: jax.lax.switch(
                    jnp.array(
                        [
                            grid_after_clear[new_row, new_col] == GridStatesEnum.BOX,
                            grid_after_clear[new_row, new_col] == GridStatesEnum.TARGET,
                            grid_after_clear[new_row, new_col] == GridStatesEnum.BOX_ON_TARGET,
                            grid_after_clear[new_row, new_col] == GridStatesEnum.EMPTY,
                        ]
                    )
                    .astype(jnp.int8)
                    .argmax(),
                    [
                        lambda: grid_after_clear.at[new_row, new_col].set(
                            GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX
                        ),  # Agent on box carrying box
                        lambda: grid_after_clear.at[new_row, new_col].set(
                            GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX
                        ),  # Agent on target carrying box
                        lambda: grid_after_clear.at[new_row, new_col].set(
                            GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX
                        ),  # Agent on box on target carrying box
                        lambda: grid_after_clear.at[new_row, new_col].set(
                            GridStatesEnum.AGENT_CARRYING_BOX
                        ),  # Agent on empty cell carrying box
                    ],
                ),
                lambda: jax.lax.switch(
                    jnp.array(
                        [
                            grid_after_clear[new_row, new_col] == GridStatesEnum.BOX,
                            grid_after_clear[new_row, new_col] == GridStatesEnum.TARGET,
                            grid_after_clear[new_row, new_col] == GridStatesEnum.BOX_ON_TARGET,
                            grid_after_clear[new_row, new_col] == GridStatesEnum.EMPTY,
                        ]
                    )
                    .astype(jnp.int8)
                    .argmax(),
                    [
                        lambda: grid_after_clear.at[new_row, new_col].set(GridStatesEnum.AGENT_ON_BOX),  # Agent on box
                        lambda: grid_after_clear.at[new_row, new_col].set(
                            GridStatesEnum.AGENT_ON_TARGET
                        ),  # Agent on target
                        lambda: grid_after_clear.at[new_row, new_col].set(
                            GridStatesEnum.AGENT_ON_TARGET_WITH_BOX
                        ),  # Agent on box on target
                        lambda: grid_after_clear.at[new_row, new_col].set(GridStatesEnum.AGENT),  # Agent on empty cell
                    ],
                ),
            )
            return jnp.array([new_row, new_col]), new_grid, state.agent_has_box

        def move_invalid():
            return state.agent_pos, grid, state.agent_has_box

        new_pos, new_grid, new_agent_has_box = jax.lax.cond(valid_move, move_valid, move_invalid)

        return new_pos, new_grid, new_agent_has_box

    @staticmethod
    @jax.jit
    def get_reward(
        old_grid: jax.Array,
        new_grid: jax.Array,
        goal_grid: jax.Array,
    ) -> jax.Array:
        solved = jnp.all(_REMOVE_AGENT_ARRAY[new_grid] == _REMOVE_AGENT_ARRAY[goal_grid]).astype(jnp.float32)
        return solved

    def _handle_pickup(self, state: BoxPushingState) -> Tuple[jax.Array, bool]:
        """Handle pickup action using PICK_UP_DICT mapping (JAX-friendly)."""

        row, col = state.agent_pos[0], state.agent_pos[1]
        current_cell = state.grid[row, col]

        # prepare jax arrays of keys and values from the mapping
        keys = jnp.array(list(PICK_UP_DICT.keys()), dtype=state.grid.dtype)
        vals = jnp.array(list(PICK_UP_DICT.values()), dtype=state.grid.dtype)

        # boolean mask of which key (if any) matches current_cell
        matches = keys == current_cell
        any_match = jnp.any(matches)

        def pickup_valid():
            # index of the (first) matching key
            idx = jnp.argmax(matches)
            new_state_val = vals[idx]
            new_grid = state.grid.at[row, col].set(new_state_val)
            return new_grid, True

        def pickup_invalid():
            return state.grid, state.agent_has_box

        new_grid, new_agent_has_box = jax.lax.cond(any_match, pickup_valid, pickup_invalid)
        return new_grid, new_agent_has_box

    def _handle_putdown(self, state: BoxPushingState) -> Tuple[jax.Array, bool]:
        """Handle putdown action using PUT_DOWN_DICT mapping (JAX-friendly)."""

        row, col = state.agent_pos[0], state.agent_pos[1]
        current_cell = state.grid[row, col]

        # prepare jax arrays of keys and values from the mapping
        keys = jnp.array(list(PUT_DOWN_DICT.keys()), dtype=state.grid.dtype)
        vals = jnp.array(list(PUT_DOWN_DICT.values()), dtype=state.grid.dtype)

        # boolean mask of which key (if any) matches current_cell
        matches = keys == current_cell
        any_match = jnp.any(matches)

        def putdown_valid():
            idx = jnp.argmax(matches)
            new_state_val = vals[idx]
            new_grid = state.grid.at[row, col].set(new_state_val)
            return new_grid, False

        def putdown_invalid():
            return state.grid, state.agent_has_box

        new_grid, new_agent_has_box = jax.lax.cond(any_match, putdown_valid, putdown_invalid)
        return new_grid, new_agent_has_box

    def play_game(self, key: jax.Array):
        """Interactive game loop using input() for controls."""

        # Initialize the environment
        state, _ = self.reset(key)
        done = False
        info = {}
        total_reward = 0
        reward = 0

        print("=== Box Pushing Game ===")
        print("Controls: w(up), s(down), a(left), d(right), e(pickup), r(drop)")
        print("Goal: Move boxes to target cells (marked with 'T')")
        print("Press 'q' to quit\n")

        while True:
            # Display current state
            self._display_state(state)
            no_targets = remove_targets(state.grid)
            print(no_targets)
            print(f"Steps: {state.steps}, Return: {total_reward}, Reward: {reward}")
            print(f"Info: {info}, Done: {done}")

            # Get user input
            action = None
            while action is None:
                try:
                    user_input = input("Enter action (w/s/a/d/e/r/q/g): ").lower().strip()
                    if user_input == "w":
                        action = 0  # Up
                    elif user_input == "s":
                        action = 1  # Down
                    elif user_input == "a":
                        action = 2  # Left
                    elif user_input == "d":
                        action = 3  # Right
                    elif user_input == "e":
                        action = 4  # Pickup
                    elif user_input == "r":
                        action = 5  # Drop
                    elif user_input == "q":
                        print("Game ended by user.")
                        return
                    elif user_input == "g":
                        print("Game restarted by user.")
                        action = "restart"
                    else:
                        print("Invalid input. Use w/s/a/d/e/r/q/g")
                except (EOFError, KeyboardInterrupt):
                    print("\nGame ended by user.")
                    return

            # Take action
            if action == "restart":
                state, info = self.reset(state.key)
            else:
                state, reward, done, info = self.step(state, action)
                total_reward += reward

    def _display_state(self, state: BoxPushingState):
        """Display the current game state in ASCII."""
        print("\n" + "=" * (self.grid_size * 2 + 1))
        print(state.grid)
        print("=" * (self.grid_size * 2 + 1))

    def get_dummy_timestep(self, key):
        return self.level_generator.get_dummy_timestep(key)

    @staticmethod
    def animate(ax, timesteps, frame, img_prefix="assets"):
        ax.clear()
        grid_state = timesteps.grid[0, frame]
        action = timesteps.action[0, frame]
        reward = timesteps.reward[0, frame]

        # Create color mapping for grid states
        imgs = {
            int(GridStatesEnum.EMPTY): "floor.png",  # EMPTY
            int(GridStatesEnum.BOX): "box.png",  # BOX
            int(GridStatesEnum.TARGET): "box_target.png",  # TARGET
            int(GridStatesEnum.AGENT): "agent.png",  # AGENT
            int(GridStatesEnum.AGENT_CARRYING_BOX): "agent_carrying_box.png",  # AGENT_CARRYING_BOX
            int(GridStatesEnum.AGENT_ON_BOX): "agent_on_box.png",  # AGENT_ON_BOX
            int(GridStatesEnum.AGENT_ON_TARGET): "agent_on_target.png",  # AGENT_ON_TARGET
            int(GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX): "agent_on_target_carrying_box.png",  # noqa: E501 AGENT_ON_TARGET_CARRYING_BOX
            int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX): "agent_on_target_with_box.png",  # AGENT_ON_TARGET_WITH_BOX
            int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX): "agent_on_target_with_box_carrying_box.png",  # noqa: E501 AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX
            int(GridStatesEnum.BOX_ON_TARGET): "box_on_target.png",  # BOX_ON_TARGET
            int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX): "agent_on_box_carrying_box.png",  # AGENT_ON_BOX_CARRYING_BOX
        }

        # Plot grid
        for i in range(grid_state.shape[0]):
            for j in range(grid_state.shape[1]):
                img_name = imgs[int(grid_state[i, j])]
                img_path = os.path.join(img_prefix, img_name)
                img = matplotlib.image.imread(img_path)
                ax.imshow(img, extent=[i + 1, i, j + 1, j])

        ax.set_xlim(0, grid_state.shape[1])
        ax.set_ylim(0, grid_state.shape[0])
        ax.set_title(f"Step {frame} | Action: {action} | Reward: {reward:.2f}")
        ax.set_aspect("equal")
        ax.invert_yaxis()


class Wrapper(BoxPushingEnv):
    def __init__(self, env: BoxPushingEnv):
        self._env = env
        # Copy attributes from wrapped environment
        for attr in [
            "grid_size",
            "episode_length",
            "number_of_boxes_min",
            "number_of_boxes_max",
            "number_of_moving_boxes_max",
            "action_space",
            "level_generator",
        ]:
            if hasattr(env, attr):
                setattr(self, attr, getattr(env, attr))

    def reset(self, key: jax.Array) -> Tuple[BoxPushingState, Dict[str, Any]]:
        return self._env.reset(key)

    def step(self, state: BoxPushingState, action: int) -> Tuple[BoxPushingState, float, bool, Dict[str, Any]]:
        return self._env.step(state, action)


class AutoResetWrapper(Wrapper):
    def __init__(self, env: BoxPushingEnv):
        super().__init__(env)

    def reset_function(self, key):
        state, info = self._env.reset(key)
        extras_new = {**state.extras, "reset": jnp.bool_(False)}
        state = state.replace(extras=extras_new)
        return state, info

    def reset(self, key):
        state, info = self.reset_function(key)
        return state, info

    def step(self, state: BoxPushingState, action: int) -> Tuple[BoxPushingState, float, bool, Dict[str, Any]]:
        state, reward, done, info = self._env.step(state, action)
        key_new, _ = jax.random.split(state.key, 2)

        def reset_fn(key):
            reset_state, reset_info = self.reset_function(key)
            return reset_state, jnp.array(0.0).astype(jnp.float32), False, reset_info

        state, reward, done, info = jax.lax.cond(
            state.extras["reset"], lambda: reset_fn(key_new), lambda: (state, reward, done, info)
        )
        reset = jnp.logical_or(info["truncated"], done)
        extras_new = {**state.extras, "reset": reset}
        state = state.replace(extras=extras_new)
        return state, reward, done, info

    def get_dummy_timestep(self, key):
        default_dummy_timestep = super().get_dummy_timestep(key)
        return default_dummy_timestep.replace(extras={**default_dummy_timestep.extras, "reset": jnp.bool_(False)})


class SymmetryFilter(Wrapper):
    def __init__(self, env: BoxPushingEnv, axis="horizontal"):
        assert env.grid_size % 2 == 0  # For clarity and convenience
        self.axis = axis
        super().__init__(env)

    def check_symmetry_crossing(self, old_state: BoxPushingState, new_state: BoxPushingState):
        middle = self._env.grid_size // 2

        if self.axis == "horizontal":
            old_pos = old_state.agent_pos[0]
            new_pos = new_state.agent_pos[0]
        else:
            old_pos = old_state.agent_pos[1]
            new_pos = new_state.agent_pos[1]

        # If old_pos and new_pos are on different sides of the middle it means we've crossed the boundry,
        # and so we reset the environment
        return jnp.logical_xor(old_pos < middle, new_pos < middle)

    def step(self, state: BoxPushingState, action: int) -> Tuple[BoxPushingState, float, bool, Dict[str, Any]]:
        new_state, reward, done, info = self._env.step(state, action)
        is_truncated = jnp.logical_or(info["truncated"], self.check_symmetry_crossing(state, new_state))

        new_info = {**info, "truncated": is_truncated}

        return new_state, reward, done, new_info


class QuarterFilter(Wrapper):
    def __init__(self, env: BoxPushingEnv):
        assert env.grid_size % 2 == 0  # For clarity and convenience
        super().__init__(env)

    def check_wrong_quarter_crossing(self, new_state: BoxPushingState):
        fields_allowed = new_state.extras["fields_allowed"]
        agent_row, agent_col = new_state.agent_pos[0], new_state.agent_pos[1]

        return jnp.logical_not(fields_allowed[agent_row, agent_col])

    def step(self, state: BoxPushingState, action: int) -> Tuple[BoxPushingState, float, bool, Dict[str, Any]]:
        new_state, reward, done, info = self._env.step(state, action)
        is_truncated = jnp.logical_or(info["truncated"], self.check_wrong_quarter_crossing(new_state))

        new_info = {**info, "truncated": is_truncated}

        return new_state, reward, done, new_info


def wrap_for_training(config, env):
    if config.exp.filtering in ["horizontal", "vertical"]:
        env = SymmetryFilter(env, axis=config.exp.filtering)
    elif config.exp.filtering == "quarter":
        env = QuarterFilter(env)
    elif config.exp.filtering is None:
        pass
    else:
        raise ValueError(f"Unknown filtering type: {config.exp.filtering}")

    env = AutoResetWrapper(env)

    return env


def wrap_for_eval(env):
    env = AutoResetWrapper(env)

    return env


if __name__ == "__main__":
    env = BoxPushingEnv(
        grid_size=6,
        number_of_boxes_max=2,
        number_of_boxes_min=2,
        number_of_moving_boxes_max=2,
        level_generator="variable",
        generator_special=False,
        dense_rewards=False,
        terminate_when_success=True,
        episode_length=10,
        quarter_size=2,
    )
    env = QuarterFilter(env)
    env = AutoResetWrapper(env)
    key = jax.random.PRNGKey(0)
    env.play_game(key)
