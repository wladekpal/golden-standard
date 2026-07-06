from dataclasses import dataclass
from typing import Any, Literal

import flax
import jax


@dataclass
class JumanjiConfig:
    env_id: str = "Snake-v1"
    episode_length: int = 120
    observation_mode: Literal["flat", "grid"] = "flat"
    grid_observation_key: str = "grid"
    grid_observation_layout: Literal["spatial", "cube"] = "spatial"
    grid_token_channels: int = 12
    reward_reduction: Literal["sum", "mean"] = "sum"


class JumanjiEnvState(flax.struct.PyTreeNode):
    """Training-time state carried through JAX scans for a Jumanji environment."""

    key: jax.Array
    env_state: Any
    grid: jax.Array
    goal: jax.Array
    steps: jax.Array
    reward: jax.Array
    success: jax.Array
    done: jax.Array
    truncated: jax.Array
    action_mask: jax.Array
    extras: dict


class JumanjiTransition(flax.struct.PyTreeNode):
    """Replay-buffer transition emitted by the Jumanji adapter.

    The field names intentionally mirror the existing box-moving timestep fields where possible so
    the replay queue can store both environment families as ordinary pytrees.
    """

    key: jax.Array
    grid: jax.Array
    goal: jax.Array
    steps: jax.Array
    action: jax.Array
    reward: jax.Array
    success: jax.Array
    done: jax.Array
    truncated: jax.Array
    action_mask: jax.Array
    extras: dict
