import jax
import jax.numpy as jnp

from envs.block_moving.env_types import GridStatesEnum


GRID_STATE_NORMALIZER = jnp.float32(int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX))
NUM_GRID_STATES = int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX) + 1


def _to_one_hot(grid: jax.Array) -> jax.Array:
    return jax.nn.one_hot(grid.astype(jnp.int32), NUM_GRID_STATES, dtype=jnp.float32)



def encode_grid_inputs(grid: jax.Array, representation: str = "normalized_flat") -> jax.Array:
    if representation == "raw_flat":
        return grid.astype(jnp.float32).reshape(grid.shape[0], -1)

    if representation == "normalized_flat":
        return (grid.astype(jnp.float32) / GRID_STATE_NORMALIZER).reshape(grid.shape[0], -1)

    if representation == "one_hot_flat":
        return _to_one_hot(grid).reshape(grid.shape[0], -1)

    raise ValueError(
        f"Unknown input representation: {representation}. Supported: raw_flat, normalized_flat, one_hot_flat"
    )