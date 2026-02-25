import jax
import jax.numpy as jnp

try:
    from envs.block_moving.env_types import GridStatesEnum
except ModuleNotFoundError:
    from ...envs.block_moving.env_types import GridStatesEnum


GRID_STATE_NORMALIZER = jnp.float32(int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX))
NUM_GRID_STATES = int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX) + 1


def _to_one_hot(grid: jax.Array) -> jax.Array:
    return jax.nn.one_hot(grid.astype(jnp.int32), NUM_GRID_STATES, dtype=jnp.float32)


def _semantic_channels(grid: jax.Array) -> jax.Array:
    cell = grid.astype(jnp.int32)

    has_agent = (
        (cell == int(GridStatesEnum.AGENT))
        | (cell == int(GridStatesEnum.AGENT_CARRYING_BOX))
        | (cell == int(GridStatesEnum.AGENT_ON_BOX))
        | (cell == int(GridStatesEnum.AGENT_ON_TARGET))
        | (cell == int(GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX))
        | (cell == int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX))
        | (cell == int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX))
        | (cell == int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX))
    )

    has_box = (
        (cell == int(GridStatesEnum.BOX))
        | (cell == int(GridStatesEnum.AGENT_ON_BOX))
        | (cell == int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX))
        | (cell == int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX))
        | (cell == int(GridStatesEnum.BOX_ON_TARGET))
        | (cell == int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX))
    )

    has_target = (
        (cell == int(GridStatesEnum.TARGET))
        | (cell == int(GridStatesEnum.AGENT_ON_TARGET))
        | (cell == int(GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX))
        | (cell == int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX))
        | (cell == int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX))
        | (cell == int(GridStatesEnum.BOX_ON_TARGET))
    )

    carrying_box = (
        (cell == int(GridStatesEnum.AGENT_CARRYING_BOX))
        | (cell == int(GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX))
        | (cell == int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX))
        | (cell == int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX))
    )

    semantic = jnp.stack(
        [
            has_agent.astype(jnp.float32),
            has_box.astype(jnp.float32),
            has_target.astype(jnp.float32),
            carrying_box.astype(jnp.float32),
        ],
        axis=-1,
    )
    return semantic


def encode_grid_inputs(grid: jax.Array, representation: str = "normalized_flat") -> jax.Array:
    if representation == "normalized_flat":
        return (grid.astype(jnp.float32) / GRID_STATE_NORMALIZER).reshape(grid.shape[0], -1)

    if representation == "one_hot_flat":
        return _to_one_hot(grid).reshape(grid.shape[0], -1)

    if representation == "one_hot_semantic_flat":
        one_hot = _to_one_hot(grid)
        semantic = _semantic_channels(grid)
        return jnp.concatenate([one_hot, semantic], axis=-1).reshape(grid.shape[0], -1)

    raise ValueError(
        f"Unknown input representation: {representation}. Supported: normalized_flat, one_hot_flat, one_hot_semantic_flat"
    )