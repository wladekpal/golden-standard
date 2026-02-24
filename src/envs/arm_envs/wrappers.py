import jax
import jax.numpy as jnp
from brax.envs.base import State

from .arm_envs import ArmEnvs
from .env_types import ArmTimeStep


class ArmWrapper:
    """Lightweight wrapper that forwards to an ArmEnvs instance and exposes key attributes."""

    def __init__(self, env: ArmEnvs):
        self._env = env
        # Copy commonly used attributes for convenience.
        for attr in [
            "episode_length",
            "state_dim",
            "goal_indices",
            "completion_goal_indices",
            "goal_reach_thresh",
        ]:
            if hasattr(env, attr):
                setattr(self, attr, getattr(env, attr))

        # Action size is defined as a property on the underlying env.
        if hasattr(env, "action_size"):
            self.action_size = env.action_size

    def reset(self, key: jax.Array) -> State:
        return self._env.reset(key)

    def step(self, state: State, action: jax.Array) -> State:
        return self._env.step(state, action)


class ArmAutoResetWrapper(ArmWrapper):
    """Auto-reset wrapper for arm environments, mirroring BoxMoving AutoReset semantics.

    - Tracks episode steps and truncated flags in state.info.
    - Automatically resets the environment after a truncated step.
    """

    def __init__(self, env: ArmEnvs):
        super().__init__(env)

    def reset_function(self, key: jax.Array) -> State:
        state = self._env.reset(key)
        info = {
            **state.info,
            "reset": jnp.bool_(False),
            "truncated": jnp.bool_(False),
            "steps": jnp.array(0, dtype=jnp.int32),
            "key": key,
        }
        return state.replace(info=info)

    def reset(self, key: jax.Array) -> State:
        return self.reset_function(key)

    def step(self, state: State, action: jax.Array) -> State:
        key_new, _ = jax.random.split(state.info["key"], 2)

        def _reset_fn(key: jax.Array) -> State:
            return self.reset_function(key)

        def _step_fn(key: jax.Array) -> State:
            new_state = self._env.step(state, action)
            old_steps = state.info["steps"]
            new_steps = old_steps + 1
            truncated = new_steps >= self.episode_length

            # When an episode truncates, we start counting steps from 0 again.
            wrapped_steps = jax.lax.cond(
                truncated,
                lambda _: jnp.array(0, dtype=old_steps.dtype),
                lambda _: new_steps,
                operand=None,
            )

            info = {
                **new_state.info,
                "steps": wrapped_steps,
                "truncated": truncated,
                "reset": truncated,
                "key": key_new,
            }
            return new_state.replace(info=info)

        reset_flag = state.info["reset"]
        new_state = jax.lax.cond(reset_flag, _reset_fn, _step_fn, key_new)
        return new_state

    def get_dummy_arm_timestep(self, key: jax.Array) -> ArmTimeStep:
        """Return a dummy ArmTimeStep for replay buffer initialization."""
        del key  # Unused here; kept for API symmetry.
        obs = jnp.zeros((self.state_dim,), dtype=jnp.float32)
        action = jnp.zeros((self.action_size,), dtype=jnp.float32)
        goal_dim = int(self.goal_indices.shape[0])
        goal = jnp.zeros((goal_dim,), dtype=jnp.float32)

        zero_f = jnp.array(0.0, dtype=jnp.float32)
        zero_i = jnp.array(0, dtype=jnp.int32)

        return ArmTimeStep(
            obs=obs,
            action=action,
            goal=goal,
            reward=zero_f,
            success=zero_f,
            done=jnp.bool_(False),
            truncated=jnp.bool_(False),
            steps=zero_i,
        )


def arm_wrap_for_training(env: ArmEnvs) -> ArmAutoResetWrapper:
    """Wrap an arm environment with auto-reset semantics for training."""
    return ArmAutoResetWrapper(env)


def arm_wrap_for_eval(env: ArmEnvs) -> ArmAutoResetWrapper:
    """Wrap an arm environment with auto-reset semantics for evaluation."""
    return ArmAutoResetWrapper(env)
