import jax
import jax.numpy as jnp


def action_mode(config) -> str:
    return str(config.get("action_mode", "discrete"))


def discrete_action_dim(config) -> int:
    return int(config["action_dim"])


def multidiscrete_action_dims(config) -> tuple[int, ...]:
    return tuple(int(v) for v in config["action_dims"])


def all_actions(batch_size: int, action_dim: int) -> jax.Array:
    return jnp.broadcast_to(jnp.arange(action_dim, dtype=jnp.int32), (batch_size, action_dim))


def mask_logits(logits: jax.Array, action_masks: jax.Array | None) -> jax.Array:
    if action_masks is None:
        return logits
    action_masks = ensure_any_valid(action_masks.astype(jnp.bool_))
    return jnp.where(action_masks, logits, jnp.full_like(logits, -1.0e9))


def ensure_any_valid(action_masks: jax.Array) -> jax.Array:
    has_valid = jnp.any(action_masks, axis=-1, keepdims=True)
    return jnp.where(has_valid, action_masks, jnp.ones_like(action_masks))


def sample_uniform_actions(seed: jax.Array, action_masks: jax.Array | None, shape, action_dim: int) -> jax.Array:
    if action_masks is None:
        return jax.random.randint(seed, shape, 0, action_dim)
    logits = mask_logits(jnp.zeros_like(action_masks, dtype=jnp.float32), action_masks)
    return jax.random.categorical(seed, logits).astype(jnp.int32)


def default_factor_action_mask(batch_size: int, action_dims: tuple[int, ...]) -> jax.Array:
    max_action_dim = max(action_dims)
    return jnp.broadcast_to(
        jnp.arange(max_action_dim)[None, :] < jnp.asarray(action_dims, dtype=jnp.int32)[:, None],
        (batch_size, len(action_dims), max_action_dim),
    )


def factor_action_mask(action_masks: jax.Array | None, batch_size: int, action_dims: tuple[int, ...]) -> jax.Array:
    base = default_factor_action_mask(batch_size, action_dims)
    if action_masks is None:
        return base
    valid = base & action_masks.astype(jnp.bool_)
    has_valid = jnp.any(valid, axis=-1, keepdims=True)
    return jnp.where(has_valid, valid, base)


def mask_factor_logits(logits: jax.Array, action_masks: jax.Array | None, action_dims: tuple[int, ...]) -> jax.Array:
    if action_masks is None:
        masks = default_factor_action_mask(logits.shape[-3], action_dims)
    else:
        masks = factor_action_mask(action_masks, logits.shape[-3], action_dims)
    while masks.ndim < logits.ndim:
        masks = masks[None, ...]
    return jnp.where(masks, logits, jnp.full_like(logits, -1.0e9))


def select_multidiscrete_q(q_values: jax.Array, actions: jax.Array) -> jax.Array:
    actions = actions.astype(jnp.int32)
    indices = actions[..., None]
    while indices.ndim < q_values.ndim:
        indices = indices[None, ...]
    return jnp.take_along_axis(q_values, indices, axis=-1).squeeze(-1)


def multidiscrete_joint_q(q_values: jax.Array, action_dims: tuple[int, ...], reduction: str = "mean") -> jax.Array:
    """Convert factor Q-values (..., F, Amax) into additive joint Q-values (..., *action_dims)."""
    joint_q = None
    num_factors = len(action_dims)
    for factor_idx, factor_size in enumerate(action_dims):
        factor_q = q_values[..., factor_idx, :factor_size]
        reshape = factor_q.shape[:-1] + (1,) * factor_idx + (factor_size,) + (1,) * (num_factors - factor_idx - 1)
        factor_q = factor_q.reshape(reshape)
        joint_q = factor_q if joint_q is None else joint_q + factor_q
    if reduction == "mean":
        joint_q = joint_q / float(num_factors)
    elif reduction != "sum":
        raise ValueError(f"Unknown MultiDiscrete joint Q reduction {reduction!r}.")
    return joint_q


def mask_joint_logits(logits: jax.Array, action_masks: jax.Array | None) -> jax.Array:
    if action_masks is None:
        return logits
    masks = ensure_any_valid_joint(action_masks.astype(jnp.bool_))
    while masks.ndim < logits.ndim:
        masks = masks[None, ...]
    return jnp.where(masks, logits, jnp.full_like(logits, -1.0e9))


def ensure_any_valid_joint(action_masks: jax.Array) -> jax.Array:
    axes = tuple(range(1, action_masks.ndim))
    has_valid = jnp.any(action_masks, axis=axes, keepdims=True)
    return jnp.where(has_valid, action_masks, jnp.ones_like(action_masks))


def unravel_multidiscrete(flat_actions: jax.Array, action_dims: tuple[int, ...]) -> jax.Array:
    digits_reversed = []
    remaining = flat_actions.astype(jnp.int32)
    for size in reversed(action_dims):
        digits_reversed.append(remaining % size)
        remaining = remaining // size
    return jnp.stack(list(reversed(digits_reversed)), axis=-1).astype(jnp.int32)
