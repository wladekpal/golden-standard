import jax
from flax import struct


@struct.dataclass
class ArmTimeStep:
    """Compact timestep representation for arm environments used in the replay buffer.

    Shapes are per single environment and single timestep; batching over
    (episode_length, num_envs, ...) is handled by the data collection code.
    """

    obs: jax.Array  # (state_dim,) e.g. 18 for ArmPushEasy (without goal appended)
    action: jax.Array  # (action_dim,) e.g. 5 for ArmPushEasy
    goal: jax.Array  # (goal_dim,) e.g. 3 for ArmPushEasy
    reward: jax.Array  # ()
    success: jax.Array  # ()
    done: jax.Array  # ()
    truncated: jax.Array  # ()
    steps: jax.Array  # () int32, step index within episode (used for trajectory segmentation)
