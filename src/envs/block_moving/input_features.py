import jax
import jax.numpy as jnp

from envs.block_moving.env_types import GridStatesEnum


GRID_STATE_NORMALIZER = jnp.float32(int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX))
NUM_GRID_STATES = int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX) + 1

# Factored binary channels: [has_box, has_target, has_agent, agent_carrying]
# Each row corresponds to a GridStatesEnum value (0–11).
_FACTORED_CHANNELS = jnp.array(
    [
        # has_box, has_target, has_agent, agent_carrying
        [0, 0, 0, 0],  # 0  EMPTY
        [1, 0, 0, 0],  # 1  BOX
        [0, 1, 0, 0],  # 2  TARGET
        [0, 0, 1, 0],  # 3  AGENT
        [0, 0, 1, 1],  # 4  AGENT_CARRYING_BOX
        [1, 0, 1, 0],  # 5  AGENT_ON_BOX
        [0, 1, 1, 0],  # 6  AGENT_ON_TARGET
        [0, 1, 1, 1],  # 7  AGENT_ON_TARGET_CARRYING_BOX
        [1, 1, 1, 0],  # 8  AGENT_ON_TARGET_WITH_BOX
        [1, 1, 1, 1],  # 9  AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX
        [1, 1, 0, 0],  # 10 BOX_ON_TARGET
        [1, 0, 1, 1],  # 11 AGENT_ON_BOX_CARRYING_BOX
    ],
    dtype=jnp.float32,
)


def _to_one_hot(grid: jax.Array) -> jax.Array:
    return jax.nn.one_hot(grid.astype(jnp.int32), NUM_GRID_STATES, dtype=jnp.float32)


def _to_factored(grid: jax.Array) -> jax.Array:
    return _FACTORED_CHANNELS[grid.astype(jnp.int32)]


def encode_grid_inputs(grid: jax.Array, representation: str = "normalized_flat") -> jax.Array:
    if representation == "raw_flat":
        return grid.astype(jnp.float32).reshape(grid.shape[0], -1)

    if representation == "normalized_flat":
        return (grid.astype(jnp.float32) / GRID_STATE_NORMALIZER).reshape(grid.shape[0], -1)

    if representation == "one_hot_flat":
        return _to_one_hot(grid).reshape(grid.shape[0], -1)

    if representation == "factored_flat":
        return _to_factored(grid).reshape(grid.shape[0], -1)

    raise ValueError(
        f"Unknown input representation: {representation}. "
        "Supported: raw_flat, normalized_flat, one_hot_flat, factored_flat"
    )