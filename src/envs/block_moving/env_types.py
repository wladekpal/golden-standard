import jax
import jax.numpy as jnp
from flax import struct
from typing import Dict
from dataclasses import dataclass


class BoxMovingState(struct.PyTreeNode):
    """State representation for the box moving environment."""

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


class TimeStep(BoxMovingState):
    action: jax.Array
    done: jax.Array
    truncated: jax.Array


@dataclass
class GridStatesEnum:
    """Grid states representation for the box moving environment."""

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

# Placing an agent (not carrying a box) onto an agent-free cell.
ADD_AGENT_DICT = {
    int(GridStatesEnum.EMPTY): int(GridStatesEnum.AGENT),  # empty -> agent
    int(GridStatesEnum.BOX): int(GridStatesEnum.AGENT_ON_BOX),  # box -> agent on box
    int(GridStatesEnum.TARGET): int(GridStatesEnum.AGENT_ON_TARGET),  # target -> agent on target
    int(GridStatesEnum.BOX_ON_TARGET): int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX),  # noqa: E501 box on target -> agent on that box+target
}

# Placing an agent that *is* carrying a box onto an agent-free cell.
ADD_AGENT_CARRYING_DICT = {
    int(GridStatesEnum.EMPTY): int(GridStatesEnum.AGENT_CARRYING_BOX),
    int(GridStatesEnum.BOX): int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX),
    int(GridStatesEnum.TARGET): int(GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX),
    int(GridStatesEnum.BOX_ON_TARGET): int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX),
}

# Pick up the box the agent is standing on (only valid while not already carrying).
PICK_UP_DICT = {
    int(GridStatesEnum.AGENT_ON_BOX): int(GridStatesEnum.AGENT_CARRYING_BOX),
    int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX): int(GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX),
}

# Put the carried box down onto the current cell (inverse of pick up).
PUT_DOWN_DICT = {v: k for k, v in PICK_UP_DICT.items()}


_NUM_STATES = int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX) + 1


def _lookup_array(mapping: dict[int, int], identity: bool = False) -> jax.Array:
    """Build an int8 lookup table indexed by grid-state value.

    Cells absent from `mapping` map to themselves when `identity` is set,
    otherwise to EMPTY. This lets `array[grid]` replace nested cond/switch logic.
    """
    default = (lambda i: i) if identity else (lambda i: int(GridStatesEnum.EMPTY))
    return jnp.array([mapping.get(i, default(i)) for i in range(_NUM_STATES)], dtype=jnp.int8)


def _valid_mask(keys) -> jax.Array:
    """Boolean table that is True exactly for the given grid-state values."""
    keys = {int(k) for k in keys}
    return jnp.array([i in keys for i in range(_NUM_STATES)], dtype=jnp.bool_)


_REMOVE_AGENT_ARRAY = _lookup_array(REMOVE_AGENT_DICT)
_ADD_AGENT_ARRAY = _lookup_array(ADD_AGENT_DICT)
_ADD_AGENT_CARRYING_ARRAY = _lookup_array(ADD_AGENT_CARRYING_DICT)
_MAPPING_ARRAY = _lookup_array(REMOVE_TARGETS_DICT)

_PICK_UP_ARRAY = _lookup_array(PICK_UP_DICT, identity=True)
_PICK_UP_VALID = _valid_mask(PICK_UP_DICT.keys())
_PUT_DOWN_ARRAY = _lookup_array(PUT_DOWN_DICT, identity=True)
_PUT_DOWN_VALID = _valid_mask(PUT_DOWN_DICT.keys())


@jax.jit
def remove_targets(grid_state: jax.Array) -> jax.Array:
    """Project grid states with targets to states without targets using a pre-built mapping array."""
    return _MAPPING_ARRAY[grid_state.astype(jnp.int8)]


# Row/column deltas for the four movement actions; pick up / put down do not move.
_ACTION_DELTAS = jnp.array(
    [
        [-1, 0],  # UP
        [1, 0],  # DOWN
        [0, -1],  # LEFT
        [0, 1],  # RIGHT
    ],
    dtype=jnp.int8,
)


@dataclass
class BoxMovingConfig:
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
