from dataclasses import fields, is_dataclass
from typing import Any

import jax
import jax.numpy as jnp


def get_named_field(tree: Any, name: str) -> Any | None:
    if isinstance(tree, dict):
        return tree.get(name)
    if hasattr(tree, name):
        return getattr(tree, name)
    if hasattr(tree, "_asdict"):
        return tree._asdict().get(name)
    if is_dataclass(tree):
        for field in fields(tree):
            if field.name == name:
                return getattr(tree, field.name)
    return None


def encode_observation(
    observation: Any,
    mode: str,
    grid_observation_key: str = "grid",
    grid_token_channels: int = 12,
) -> jax.Array:
    if mode == "flat":
        return _encode_flat_observation(observation)
    if mode == "grid":
        return _encode_grid_observation(observation, grid_observation_key, grid_token_channels)
    raise ValueError(f"Unknown Jumanji observation_mode={mode!r}. Supported: 'flat', 'grid'.")


def _encode_flat_observation(observation: Any) -> jax.Array:
    leaves = _numeric_observation_leaves(observation)
    if not leaves:
        raise ValueError("Could not find any numeric leaves in Jumanji observation.")
    flat = [jnp.asarray(leaf, dtype=jnp.float32).reshape(-1) for leaf in leaves]
    return jnp.concatenate(flat, axis=0)


def _numeric_observation_leaves(tree: Any, *, current_name: str | None = None) -> list[Any]:
    if current_name == "action_mask":
        return []

    if isinstance(tree, dict):
        leaves = []
        for key in sorted(tree.keys()):
            leaves.extend(_numeric_observation_leaves(tree[key], current_name=str(key)))
        return leaves

    if hasattr(tree, "_fields"):
        leaves = []
        for name in tree._fields:
            leaves.extend(_numeric_observation_leaves(getattr(tree, name), current_name=name))
        return leaves

    if is_dataclass(tree):
        leaves = []
        for field in fields(tree):
            leaves.extend(_numeric_observation_leaves(getattr(tree, field.name), current_name=field.name))
        return leaves

    try:
        arr = jnp.asarray(tree)
    except TypeError:
        return []
    if arr.dtype == jnp.dtype("O"):
        return []
    return [arr]


def _encode_grid_observation(observation: Any, grid_observation_key: str, grid_token_channels: int) -> jax.Array:
    grid = get_named_field(observation, grid_observation_key)
    if grid is None:
        raise ValueError(f"Jumanji observation_mode='grid' requires a '{grid_observation_key}' observation field.")

    grid = jnp.asarray(grid)
    if grid.ndim == 2:
        tokens = jax.nn.one_hot(grid.astype(jnp.int32), grid_token_channels, dtype=jnp.float32)
    elif grid.ndim == 3:
        tokens = grid.astype(jnp.float32)
        if tokens.shape[-1] > grid_token_channels:
            raise ValueError(
                f"Grid observation has {tokens.shape[-1]} channels, but grid_token_channels={grid_token_channels}."
            )
        pad_width = grid_token_channels - tokens.shape[-1]
        if pad_width > 0:
            tokens = jnp.pad(tokens, ((0, 0), (0, 0), (0, pad_width)))
    else:
        raise ValueError(
            "Jumanji observation_mode='grid' requires a 2D categorical grid or a 3D grid/image tensor, "
            f"got shape {grid.shape}."
        )

    return tokens.reshape(-1)
