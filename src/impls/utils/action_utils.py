import jax
import jax.numpy as jnp


def discrete_action_dim(config) -> int:
    return int(config["action_dim"])


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
