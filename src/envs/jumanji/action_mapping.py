from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _shape_tuple(value) -> tuple[int, ...]:
    if value is None:
        return ()
    return tuple(int(v) for v in value)


def _get_attr(spec: Any, *names: str):
    for name in names:
        if hasattr(spec, name):
            return getattr(spec, name)
    return None


def _spec_class_name(spec: Any) -> str:
    return type(spec).__name__


def _num_values_from_spec(spec: Any) -> np.ndarray | None:
    num_values = _get_attr(spec, "num_values", "n")
    if num_values is not None:
        return np.asarray(num_values, dtype=np.int64)

    minimum = _get_attr(spec, "minimum", "min")
    maximum = _get_attr(spec, "maximum", "max")
    dtype = _get_attr(spec, "dtype")
    if minimum is not None and maximum is not None and dtype is not None and np.issubdtype(np.dtype(dtype), np.integer):
        return np.asarray(maximum, dtype=np.int64) - np.asarray(minimum, dtype=np.int64) + 1

    return None


def action_spec_factor_sizes(action_spec: Any) -> tuple[int, ...]:
    """Return finite discrete factor sizes for supported Jumanji-style specs."""
    class_name = _spec_class_name(action_spec)
    num_values = _num_values_from_spec(action_spec)
    spec_shape = _shape_tuple(_get_attr(action_spec, "shape"))

    if class_name == "DiscreteArray" or (num_values is not None and num_values.shape == () and spec_shape in [(), (1,)]):
        return (int(num_values),)

    if class_name == "MultiDiscreteArray" or (num_values is not None and spec_shape):
        if num_values.shape == ():
            num_values = np.full(spec_shape, int(num_values), dtype=np.int64)
        if tuple(num_values.shape) != spec_shape:
            num_values = np.broadcast_to(num_values, spec_shape)
        return tuple(int(v) for v in num_values.reshape(-1))

    raise ValueError(
        f"Unsupported Jumanji action spec {class_name}. Only finite scalar discrete and small MultiDiscrete "
        "action specs are currently supported."
    )


@dataclass(frozen=True)
class JumanjiActionMapper:
    factor_sizes: tuple[int, ...]
    action_shape: tuple[int, ...]
    dtype: Any
    flat_digits: jax.Array

    @classmethod
    def from_spec(cls, action_spec: Any, max_flattened_action_size: int) -> "JumanjiActionMapper":
        factor_sizes = action_spec_factor_sizes(action_spec)
        action_dim = int(np.prod(np.asarray(factor_sizes, dtype=np.int64)))
        if action_dim > max_flattened_action_size:
            raise ValueError(
                f"Flattened Jumanji action space has {action_dim} actions, which exceeds "
                f"max_flattened_action_size={max_flattened_action_size}."
            )

        spec_shape = _shape_tuple(_get_attr(action_spec, "shape"))
        if len(factor_sizes) == 1:
            action_shape = ()
        elif spec_shape and int(np.prod(spec_shape)) == len(factor_sizes):
            action_shape = spec_shape
        else:
            action_shape = (len(factor_sizes),)

        dtype = _get_attr(action_spec, "dtype") or jnp.int32
        flat_digits = cls._build_flat_digits(factor_sizes, action_dim)
        return cls(factor_sizes=factor_sizes, action_shape=action_shape, dtype=dtype, flat_digits=flat_digits)

    @staticmethod
    def _build_flat_digits(factor_sizes: tuple[int, ...], action_dim: int) -> jax.Array:
        ids = np.arange(action_dim, dtype=np.int64)
        digits_reversed = []
        remaining = ids.copy()
        for size in reversed(factor_sizes):
            digits_reversed.append(remaining % size)
            remaining //= size
        digits = np.stack(list(reversed(digits_reversed)), axis=-1)
        return jnp.asarray(digits, dtype=jnp.int32)

    @property
    def action_dim(self) -> int:
        return int(np.prod(np.asarray(self.factor_sizes, dtype=np.int64)))

    @property
    def is_scalar(self) -> bool:
        return len(self.factor_sizes) == 1

    def unflatten(self, flat_action: jax.Array) -> jax.Array:
        flat_action = flat_action.astype(jnp.int32)
        if self.is_scalar:
            return flat_action.astype(self.dtype)

        digits_reversed = []
        remaining = flat_action
        for size in reversed(self.factor_sizes):
            digits_reversed.append(remaining % size)
            remaining = remaining // size
        digits = jnp.stack(list(reversed(digits_reversed)), axis=-1)
        return digits.reshape((*flat_action.shape, *self.action_shape)).astype(self.dtype)

    def flatten_action_mask(self, action_mask: jax.Array | None) -> jax.Array:
        if action_mask is None:
            return jnp.ones((self.action_dim,), dtype=jnp.bool_)

        mask = jnp.asarray(action_mask, dtype=jnp.bool_)
        if self.is_scalar:
            mask = mask.reshape(-1)
            if mask.shape[0] != self.action_dim:
                raise ValueError(f"Expected scalar action mask of length {self.action_dim}, got shape {mask.shape}.")
            return _ensure_any_valid(mask)

        flat_factor_mask = mask.reshape((len(self.factor_sizes), -1))
        if flat_factor_mask.shape[0] != len(self.factor_sizes):
            raise ValueError(
                f"Expected action mask with {len(self.factor_sizes)} discrete factors, got shape {mask.shape}."
            )

        valid = jnp.ones((self.action_dim,), dtype=jnp.bool_)
        for factor_idx, factor_size in enumerate(self.factor_sizes):
            factor_valid = flat_factor_mask[factor_idx, :factor_size]
            valid = valid & factor_valid[self.flat_digits[:, factor_idx]]
        return _ensure_any_valid(valid)


def _ensure_any_valid(mask: jax.Array) -> jax.Array:
    has_valid = jnp.any(mask, axis=-1, keepdims=True)
    return jnp.where(has_valid, mask, jnp.ones_like(mask))
