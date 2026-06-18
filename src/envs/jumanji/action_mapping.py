from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np


ActionMaskMode = Literal["categorical", "factor", "joint"]
ActionMode = Literal["discrete", "multidiscrete"]


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


def _broadcast_array(value, shape: tuple[int, ...], fill_value: int = 0) -> np.ndarray:
    if value is None:
        return np.full(shape, fill_value, dtype=np.int64)
    array = np.asarray(value, dtype=np.int64)
    if array.shape == ():
        return np.full(shape, int(array), dtype=np.int64)
    if tuple(array.shape) != shape:
        return np.broadcast_to(array, shape).astype(np.int64)
    return array.astype(np.int64)


def _spec_bounds(action_spec: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    minimum = _get_attr(action_spec, "minimum", "min")
    maximum = _get_attr(action_spec, "maximum", "max")
    if minimum is None or maximum is None:
        return None, None
    return np.asarray(minimum, dtype=np.int64), np.asarray(maximum, dtype=np.int64)


def _num_values_from_spec(action_spec: Any) -> np.ndarray | None:
    num_values = _get_attr(action_spec, "num_values", "n")
    if num_values is not None:
        return np.asarray(num_values, dtype=np.int64)

    minimum, maximum = _spec_bounds(action_spec)
    dtype = _get_attr(action_spec, "dtype")
    if minimum is not None and maximum is not None and dtype is not None and np.issubdtype(np.dtype(dtype), np.integer):
        return maximum - minimum + 1

    return None


def action_spec_factor_sizes(action_spec: Any) -> tuple[int, ...]:
    """Return finite discrete factor sizes for supported Jumanji-style specs."""
    factor_sizes, _ = action_spec_factors_and_minimum(action_spec)
    return factor_sizes


def action_spec_factors_and_minimum(action_spec: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return zero-based factor sizes plus the environment minimum for each factor."""
    class_name = _spec_class_name(action_spec)
    num_values = _num_values_from_spec(action_spec)
    spec_shape = _shape_tuple(_get_attr(action_spec, "shape"))
    minimum, _ = _spec_bounds(action_spec)

    if class_name == "DiscreteArray" or (num_values is not None and num_values.shape == () and spec_shape in [(), (1,)]):
        factor_sizes = (int(num_values),)
        minimums = (int(np.asarray(minimum, dtype=np.int64)) if minimum is not None else 0,)
        return factor_sizes, minimums

    if class_name == "MultiDiscreteArray" or (num_values is not None and spec_shape):
        if num_values is None:
            raise ValueError(f"Unsupported Jumanji action spec {class_name}: missing finite integer bounds.")
        values = _broadcast_array(num_values, spec_shape)
        minimum_values = _broadcast_array(minimum, spec_shape, fill_value=0)
        return tuple(int(v) for v in values.reshape(-1)), tuple(int(v) for v in minimum_values.reshape(-1))

    raise ValueError(
        f"Unsupported Jumanji action spec {class_name}. Only finite scalar discrete and MultiDiscrete "
        "action specs are currently supported."
    )


@dataclass(frozen=True)
class JumanjiActionMapper:
    factor_sizes: tuple[int, ...]
    action_shape: tuple[int, ...]
    dtype: Any
    minimums: tuple[int, ...]

    @classmethod
    def from_spec(cls, action_spec: Any) -> "JumanjiActionMapper":
        factor_sizes, minimums = action_spec_factors_and_minimum(action_spec)

        spec_shape = _shape_tuple(_get_attr(action_spec, "shape"))
        if len(factor_sizes) == 1:
            action_shape = ()
        elif spec_shape and int(np.prod(spec_shape)) == len(factor_sizes):
            action_shape = spec_shape
        else:
            action_shape = (len(factor_sizes),)

        dtype = _get_attr(action_spec, "dtype") or jnp.int32
        return cls(factor_sizes=factor_sizes, action_shape=action_shape, dtype=dtype, minimums=minimums)

    @property
    def action_mode(self) -> ActionMode:
        return "discrete" if self.is_scalar else "multidiscrete"

    @property
    def action_dim(self) -> int:
        return self.factor_sizes[0] if self.is_scalar else max(self.factor_sizes)

    @property
    def num_action_factors(self) -> int:
        return len(self.factor_sizes)

    @property
    def is_scalar(self) -> bool:
        return len(self.factor_sizes) == 1

    @property
    def agent_action_shape(self) -> tuple[int, ...]:
        return () if self.is_scalar else (len(self.factor_sizes),)

    @property
    def max_agent_action(self) -> jax.Array:
        if self.is_scalar:
            return jnp.asarray(self.factor_sizes[0] - 1, dtype=jnp.int32)
        return jnp.asarray([size - 1 for size in self.factor_sizes], dtype=jnp.int32)

    @property
    def default_agent_action(self) -> jax.Array:
        return jnp.zeros(self.agent_action_shape, dtype=jnp.int32)

    @property
    def action_dims_array(self) -> jax.Array:
        return jnp.asarray(self.factor_sizes, dtype=jnp.int32)

    def to_env_action(self, agent_action: jax.Array) -> jax.Array:
        action = jnp.asarray(agent_action, dtype=jnp.int32)
        minimums = jnp.asarray(self.minimums, dtype=jnp.int32)
        if self.is_scalar:
            return (action + minimums[0]).astype(self.dtype)
        return (action.reshape(self.action_shape) + minimums.reshape(self.action_shape)).astype(self.dtype)

    def action_mask_mode(self, action_mask: jax.Array | None) -> ActionMaskMode:
        if self.is_scalar:
            return "categorical"
        if action_mask is None:
            return "factor"

        mask_shape = tuple(int(v) for v in jnp.asarray(action_mask).shape)
        factor_shape = (len(self.factor_sizes), self.action_dim)
        if mask_shape == factor_shape:
            return "factor"
        if mask_shape == self.factor_sizes:
            return "joint"
        if len(mask_shape) >= 2 and int(np.prod(mask_shape[:-1])) == len(self.factor_sizes):
            return "factor"
        if int(np.prod(mask_shape)) == int(np.prod(self.factor_sizes)):
            return "joint"
        raise ValueError(
            f"Unsupported Jumanji action mask shape {mask_shape} for MultiDiscrete action sizes "
            f"{self.factor_sizes}."
        )

    def agent_action_mask(self, action_mask: jax.Array | None) -> jax.Array:
        mode = self.action_mask_mode(action_mask)
        if mode == "categorical":
            return self._categorical_mask(action_mask)
        if mode == "factor":
            return self._factor_mask(action_mask)
        return self._joint_mask(action_mask)

    def _categorical_mask(self, action_mask: jax.Array | None) -> jax.Array:
        if action_mask is None:
            return jnp.ones((self.action_dim,), dtype=jnp.bool_)
        mask = jnp.asarray(action_mask, dtype=jnp.bool_).reshape(-1)
        if mask.shape[0] != self.action_dim:
            raise ValueError(f"Expected scalar action mask of length {self.action_dim}, got shape {mask.shape}.")
        return _ensure_any_valid(mask)

    def _factor_mask(self, action_mask: jax.Array | None) -> jax.Array:
        base = jnp.arange(self.action_dim)[None, :] < self.action_dims_array[:, None]
        if action_mask is None:
            return base.astype(jnp.bool_)

        mask = jnp.asarray(action_mask, dtype=jnp.bool_)
        flat_factor_mask = mask.reshape((len(self.factor_sizes), -1))
        if flat_factor_mask.shape[1] > self.action_dim:
            raise ValueError(
                f"Expected factor action mask width <= {self.action_dim}, got shape {mask.shape}."
            )
        pad_width = self.action_dim - flat_factor_mask.shape[1]
        if pad_width:
            flat_factor_mask = jnp.pad(flat_factor_mask, ((0, 0), (0, pad_width)), constant_values=False)
        valid = base & flat_factor_mask
        has_valid = jnp.any(valid, axis=-1, keepdims=True)
        return jnp.where(has_valid, valid, base)

    def _joint_mask(self, action_mask: jax.Array | None) -> jax.Array:
        if action_mask is None:
            raise ValueError("Internal error: joint action masks cannot be inferred from a missing mask.")
        mask = jnp.asarray(action_mask, dtype=jnp.bool_).reshape(self.factor_sizes)
        return _ensure_any_valid_joint(mask)


def _ensure_any_valid(mask: jax.Array) -> jax.Array:
    has_valid = jnp.any(mask, axis=-1, keepdims=True)
    return jnp.where(has_valid, mask, jnp.ones_like(mask))


def _ensure_any_valid_joint(mask: jax.Array) -> jax.Array:
    return jnp.where(jnp.any(mask), mask, jnp.ones_like(mask))
