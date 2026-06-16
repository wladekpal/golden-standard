from typing import Any, Tuple

import jax
import jax.numpy as jnp

from .action_mapping import JumanjiActionMapper
from .env_types import JumanjiConfig, JumanjiEnvState, JumanjiTransition
from .observation import encode_observation, get_named_field


class JumanjiDiscreteEnv:
    """Adapter exposing finite-discrete Jumanji environments to this repository's trainer."""

    is_jumanji = True
    uses_native_rewards = True

    def __init__(
        self,
        env_id: str = "Snake-v1",
        episode_length: int = 120,
        max_flattened_action_size: int = 4096,
        observation_mode: str = "flat",
        grid_observation_key: str = "grid",
        grid_token_channels: int = 12,
        reward_reduction: str = "sum",
        **kwargs,
    ):
        del kwargs
        try:
            import jumanji
        except ImportError as exc:
            raise ImportError(
                "The Jumanji environment suite is not installed. Run `uv sync` after the dependency update."
            ) from exc

        self.env_id = env_id
        self.episode_length = episode_length
        self.observation_mode = observation_mode
        self.grid_observation_key = grid_observation_key
        self.grid_token_channels = grid_token_channels
        self.reward_reduction = reward_reduction
        self._env = jumanji.make(env_id)
        self.action_mapper = JumanjiActionMapper.from_spec(
            self._env.action_spec,
            max_flattened_action_size=max_flattened_action_size,
        )
        self.action_dim = self.action_mapper.action_dim
        self.action_space = self.action_dim

    @classmethod
    def from_config(cls, config: JumanjiConfig) -> "JumanjiDiscreteEnv":
        return cls(**vars(config))

    def reset(self, key: jax.Array) -> Tuple[JumanjiEnvState, dict[str, Any]]:
        return self._reset_from_key(key)

    def _reset_from_key(self, key: jax.Array) -> Tuple[JumanjiEnvState, dict[str, Any]]:
        reset_key, carry_key = jax.random.split(key)
        env_state, timestep = self._env.reset(reset_key)
        observation = self._encode_observation(timestep.observation)
        action_mask = self._extract_action_mask(timestep.observation)
        reward = jnp.zeros((), dtype=jnp.float32)
        state = JumanjiEnvState(
            key=carry_key,
            env_state=env_state,
            grid=observation,
            goal=jnp.zeros_like(observation),
            steps=jnp.zeros((), dtype=jnp.int32),
            reward=reward,
            success=jnp.zeros((), dtype=jnp.int8),
            done=jnp.bool_(False),
            truncated=jnp.bool_(False),
            action_mask=action_mask,
            extras={"reset": jnp.bool_(False)},
        )
        return state, self._info(jnp.bool_(False), reward)

    def step(self, state: JumanjiEnvState, flat_action: jax.Array):
        def reset_branch():
            reset_state, reset_info = self._reset_from_key(state.key)
            return reset_state, jnp.zeros((), dtype=jnp.float32), jnp.bool_(False), reset_info

        def step_branch():
            action = self.action_mapper.unflatten(flat_action)
            env_state, timestep = self._env.step(state.env_state, action)
            observation = self._encode_observation(timestep.observation)
            action_mask = self._extract_action_mask(timestep.observation)
            reward = self._reduce_reward(timestep.reward)

            new_steps = state.steps + 1
            last = jnp.asarray(timestep.step_type == 2)
            terminated = last & jnp.all(jnp.asarray(timestep.discount) == 0)
            time_limit_truncated = new_steps >= self.episode_length
            truncated = (last & ~terminated) | time_limit_truncated
            reset = terminated | truncated

            new_state = JumanjiEnvState(
                key=state.key,
                env_state=env_state,
                grid=observation,
                goal=jnp.zeros_like(observation),
                steps=new_steps,
                reward=reward,
                success=terminated.astype(jnp.int8),
                done=terminated,
                truncated=truncated,
                action_mask=action_mask,
                extras={"reset": reset},
            )
            return new_state, reward, terminated, self._info(truncated, reward)

        return jax.lax.cond(state.extras["reset"], reset_branch, step_branch)

    def transition_from_step(
        self,
        state: JumanjiEnvState,
        action: jax.Array,
        new_state: JumanjiEnvState,
        reward: jax.Array,
        done: jax.Array,
        info: dict[str, Any],
    ) -> JumanjiTransition:
        return JumanjiTransition(
            key=state.key,
            grid=state.grid,
            goal=state.goal,
            steps=state.steps,
            action=action,
            reward=reward,
            success=new_state.success,
            done=done,
            truncated=info["truncated"],
            action_mask=state.action_mask,
            extras=state.extras,
        )

    def agent_inputs(self, state: JumanjiEnvState, use_targets: bool, input_representation: str):
        del use_targets, input_representation
        return state.grid, state.goal, state.action_mask

    def collect_render_states(
        self,
        agent: Any,
        key: jax.Array,
        episode_length: int | None = None,
        use_targets: bool = False,
        input_representation: str = "normalized_flat",
    ) -> list[Any]:
        """Collect native Jumanji states for viewer-based rollout rendering."""
        if not hasattr(self._env, "animate"):
            raise ValueError(
                f"Jumanji environment {self.env_id!r} does not expose animate(...). "
                "Disable GIF logging with --exp.num_gifs 0."
            )

        rollout_length = self.episode_length if episode_length is None else episode_length
        state, _ = self._reset_from_key(key)
        render_states = [state.env_state]
        step_key = key

        for _ in range(rollout_length):
            step_key, sample_key = jax.random.split(step_key)
            observations, goals, action_masks = self.agent_inputs(state, use_targets, input_representation)
            actions = agent.sample_actions(
                observations[None, ...],
                goals[None, ...],
                seed=sample_key,
                action_masks=action_masks[None, ...],
            )
            flat_action = jnp.asarray(actions).reshape(-1)[0]
            action = self.action_mapper.unflatten(flat_action)
            env_state, timestep = self._env.step(state.env_state, action)
            observation = self._encode_observation(timestep.observation)
            action_mask = self._extract_action_mask(timestep.observation)
            reward = self._reduce_reward(timestep.reward)

            new_steps = state.steps + 1
            last = jnp.asarray(timestep.step_type == 2)
            terminated = last & jnp.all(jnp.asarray(timestep.discount) == 0)
            time_limit_truncated = new_steps >= rollout_length
            truncated = (last & ~terminated) | time_limit_truncated
            reset = terminated | truncated

            state = JumanjiEnvState(
                key=state.key,
                env_state=env_state,
                grid=observation,
                goal=jnp.zeros_like(observation),
                steps=new_steps,
                reward=reward,
                success=terminated.astype(jnp.int8),
                done=terminated,
                truncated=truncated,
                action_mask=action_mask,
                extras={"reset": reset},
            )
            render_states.append(state.env_state)

            done_or_truncated = jax.device_get(terminated | truncated)
            if bool(done_or_truncated):
                break

        return render_states

    def get_dummy_timestep(self, key: jax.Array) -> JumanjiTransition:
        state, _ = self._reset_from_key(key)
        return JumanjiTransition(
            key=key,
            grid=jnp.zeros_like(state.grid),
            goal=jnp.zeros_like(state.goal),
            steps=jnp.zeros((), dtype=jnp.int32),
            action=jnp.zeros((), dtype=jnp.int32),
            reward=jnp.zeros((), dtype=jnp.float32),
            success=jnp.zeros((), dtype=jnp.int8),
            done=jnp.bool_(False),
            truncated=jnp.bool_(False),
            action_mask=jnp.ones((self.action_dim,), dtype=jnp.bool_),
            extras={"reset": jnp.bool_(False)},
        )

    def _encode_observation(self, observation: Any) -> jax.Array:
        return encode_observation(
            observation,
            self.observation_mode,
            grid_observation_key=self.grid_observation_key,
            grid_token_channels=self.grid_token_channels,
        )

    def _extract_action_mask(self, observation: Any) -> jax.Array:
        action_mask = get_named_field(observation, "action_mask")
        return self.action_mapper.flatten_action_mask(action_mask)

    def _reduce_reward(self, reward: jax.Array) -> jax.Array:
        reward = jnp.asarray(reward, dtype=jnp.float32)
        if reward.shape == ():
            return reward
        if self.reward_reduction == "sum":
            return reward.sum()
        if self.reward_reduction == "mean":
            return reward.mean()
        raise ValueError(f"Unknown reward_reduction={self.reward_reduction!r}.")

    @staticmethod
    def _info(truncated: jax.Array, reward: jax.Array) -> dict[str, jax.Array]:
        return {
            "truncated": truncated,
            "boxes_on_target": jnp.zeros((), dtype=jnp.float32),
            "native_reward": reward,
        }
