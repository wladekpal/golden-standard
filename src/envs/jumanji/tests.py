from typing import NamedTuple

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import pytest

from envs.jumanji.action_mapping import JumanjiActionMapper
from envs.jumanji.observation import encode_observation
from envs.jumanji.jumanji_env import JumanjiDiscreteEnv
from impls.agents import create_agent, default_config


class FakeDiscreteSpec:
    shape = ()
    dtype = jnp.int32
    num_values = 4


class FakeMultiDiscreteSpec:
    shape = (2,)
    dtype = jnp.int32
    num_values = np.array([2, 3])


class FakeJointMaskedMultiDiscreteSpec:
    shape = (2,)
    dtype = jnp.int32
    num_values = np.array([4, 10])


class FakeObservation(NamedTuple):
    grid: jnp.ndarray
    step_count: jnp.ndarray
    action_mask: jnp.ndarray


def test_scalar_discrete_action_mapping():
    mapper = JumanjiActionMapper.from_spec(FakeDiscreteSpec())

    assert mapper.action_mode == "discrete"
    assert mapper.action_dim == 4
    assert int(mapper.to_env_action(jnp.array(3))) == 3
    assert jnp.array_equal(
        mapper.agent_action_mask(jnp.array([True, False, True, False])),
        jnp.array([True, False, True, False]),
    )


def test_multidiscrete_action_mapping_and_factor_mask():
    mapper = JumanjiActionMapper.from_spec(FakeMultiDiscreteSpec())

    assert mapper.action_mode == "multidiscrete"
    assert mapper.action_dim == 3
    assert mapper.factor_sizes == (2, 3)
    assert mapper.agent_action_shape == (2,)
    assert jnp.array_equal(mapper.to_env_action(jnp.array([1, 2])), jnp.array([1, 2], dtype=jnp.int32))

    factor_mask = jnp.array(
        [
            [True, False, False],
            [False, True, True],
        ]
    )
    assert jnp.array_equal(
        mapper.agent_action_mask(factor_mask),
        factor_mask,
    )


def test_multidiscrete_joint_action_mask_passthrough():
    mapper = JumanjiActionMapper.from_spec(FakeJointMaskedMultiDiscreteSpec())
    joint_mask = jnp.zeros((4, 10), dtype=jnp.bool_).at[1, 3].set(True)

    assert mapper.action_mask_mode(joint_mask) == "joint"
    assert jnp.array_equal(
        mapper.agent_action_mask(joint_mask),
        joint_mask,
    )


def _multidiscrete_example_batch(action_mask_mode="factor"):
    if action_mask_mode == "joint":
        action_dims = (4, 10)
        action_masks = jnp.zeros((2, 4, 10), dtype=jnp.bool_).at[:, 1, 3].set(True)
        actions = jnp.array([[1, 3], [1, 3]], dtype=jnp.int32)
    else:
        action_dims = (2, 3)
        action_masks = jnp.array(
            [
                [[True, False, False], [False, True, True]],
                [[True, False, False], [True, False, False]],
            ],
            dtype=jnp.bool_,
        )
        actions = jnp.array([[0, 2], [0, 0]], dtype=jnp.int32)

    observations = jnp.ones((2, 4), dtype=jnp.float32)
    return {
        "observations": observations,
        "next_observations": observations,
        "actions": actions,
        "rewards": jnp.ones((2,), dtype=jnp.float32),
        "masks": jnp.ones((2,), dtype=jnp.float32),
        "value_goals": jnp.zeros_like(observations),
        "actor_goals": jnp.zeros_like(observations),
        "action_masks": action_masks,
        "next_action_masks": action_masks,
        "action_mode": "multidiscrete",
        "action_dims": action_dims,
        "action_mask_mode": action_mask_mode,
    }


def _scalar_example_batch(obs_dim=4):
    observations = jnp.ones((2, obs_dim), dtype=jnp.float32)
    return {
        "observations": observations,
        "next_observations": observations,
        "actions": jnp.array([0, 1], dtype=jnp.int32),
        "rewards": jnp.ones((2,), dtype=jnp.float32),
        "masks": jnp.ones((2,), dtype=jnp.float32),
        "value_goals": jnp.zeros_like(observations),
        "actor_goals": jnp.zeros_like(observations),
        "action_masks": jnp.array([[True, False, True, False], [False, True, True, False]], dtype=jnp.bool_),
        "next_action_masks": jnp.array([[True, False, True, False], [False, True, True, False]], dtype=jnp.bool_),
        "action_mode": "discrete",
        "action_dim": 4,
        "action_mask_mode": "categorical",
    }


def _without_action_metadata(batch):
    return {k: v for k, v in batch.items() if k not in {"action_mode", "action_dim", "action_dims", "action_mask_mode"}}


def _agent_config(agent_name):
    config = ml_collections.ConfigDict(default_config)
    config.agent_name = agent_name
    config.actor_hidden_dims = (8,)
    config.value_hidden_dims = (8,)
    if agent_name == "gciql":
        config.discrete = True
        config.actor_loss = "awr"
    return config


def _transformer_config(agent_name):
    config = _agent_config(agent_name)
    config.critic_arch = "universal_transformer"
    config.transformer_cell_dim = 1
    config.transformer_d_model = 8
    config.transformer_num_heads = 2
    config.transformer_mlp_dim = 16
    config.transformer_thinking_steps = 2
    return config


def test_dqn_multidiscrete_factor_mask_sampling_and_loss():
    config = _agent_config("gcdqn")
    batch = _multidiscrete_example_batch("factor")
    agent = create_agent(config, batch, seed=0)

    actions = agent.sample_actions(
        batch["observations"],
        batch["value_goals"],
        seed=jax.random.PRNGKey(1),
        action_masks=batch["action_masks"],
    )
    loss, info = agent.total_loss(_without_action_metadata(batch), None)

    assert actions.shape == (2, 2)
    assert jnp.all(batch["action_masks"][jnp.arange(2)[:, None], jnp.arange(2)[None, :], actions])
    assert jnp.isfinite(loss)
    assert "critic/critic_loss" in info


def test_dqn_multidiscrete_joint_mask_sampling_and_loss():
    config = _agent_config("gcdqn")
    batch = _multidiscrete_example_batch("joint")
    agent = create_agent(config, batch, seed=0)

    actions = agent.sample_actions(
        batch["observations"],
        batch["value_goals"],
        seed=jax.random.PRNGKey(2),
        action_masks=batch["action_masks"],
    )
    loss, _ = agent.total_loss(_without_action_metadata(batch), None)

    assert jnp.array_equal(actions, jnp.array([[1, 3], [1, 3]], dtype=jnp.int32))
    assert jnp.isfinite(loss)


def test_gciql_multidiscrete_factor_mask_sampling_and_loss():
    config = _agent_config("gciql")
    batch = _multidiscrete_example_batch("factor")
    agent = create_agent(config, batch, seed=0)

    actions = agent.sample_actions(
        batch["observations"],
        batch["value_goals"],
        seed=jax.random.PRNGKey(3),
        action_masks=batch["action_masks"],
    )
    loss, info = agent.total_loss(_without_action_metadata(batch), None)

    assert actions.shape == (2, 2)
    assert jnp.all(batch["action_masks"][jnp.arange(2)[:, None], jnp.arange(2)[None, :], actions])
    assert jnp.isfinite(loss)
    assert "actor/bc_log_prob" in info


def test_gciql_multidiscrete_joint_mask_sampling_and_loss():
    config = _agent_config("gciql")
    batch = _multidiscrete_example_batch("joint")
    agent = create_agent(config, batch, seed=0)

    actions = agent.sample_actions(
        batch["observations"],
        batch["value_goals"],
        seed=jax.random.PRNGKey(4),
        action_masks=batch["action_masks"],
    )
    loss, _ = agent.total_loss(_without_action_metadata(batch), None)

    assert jnp.array_equal(actions, jnp.array([[1, 3], [1, 3]], dtype=jnp.int32))
    assert jnp.isfinite(loss)


def test_dqn_transformer_scalar_discrete_loss_and_sampling():
    config = _transformer_config("gcdqn")
    batch = _scalar_example_batch()
    agent = create_agent(config, batch, seed=0)

    actions = agent.sample_actions(
        batch["observations"],
        batch["value_goals"],
        seed=jax.random.PRNGKey(5),
        action_masks=batch["action_masks"],
    )
    loss, _ = agent.total_loss(_without_action_metadata(batch), None)

    assert actions.shape == (2,)
    assert jnp.all(batch["action_masks"][jnp.arange(2), actions])
    assert jnp.isfinite(loss)


def test_dqn_transformer_multidiscrete_loss_and_sampling():
    config = _transformer_config("gcdqn")
    batch = _multidiscrete_example_batch("factor")
    agent = create_agent(config, batch, seed=0)

    actions = agent.sample_actions(
        batch["observations"],
        batch["value_goals"],
        seed=jax.random.PRNGKey(6),
        action_masks=batch["action_masks"],
    )
    loss, _ = agent.total_loss(_without_action_metadata(batch), None)

    assert actions.shape == (2, 2)
    assert jnp.all(batch["action_masks"][jnp.arange(2)[:, None], jnp.arange(2)[None, :], actions])
    assert jnp.isfinite(loss)


def test_gciql_transformer_scalar_actor_critic_loss_and_sampling():
    config = _transformer_config("gciql")
    config.actor_arch = "universal_transformer"
    batch = _scalar_example_batch()
    agent = create_agent(config, batch, seed=0)

    actions = agent.sample_actions(
        batch["observations"],
        batch["value_goals"],
        seed=jax.random.PRNGKey(7),
        action_masks=batch["action_masks"],
    )
    loss, _ = agent.total_loss(_without_action_metadata(batch), None)

    assert actions.shape == (2,)
    assert jnp.all(batch["action_masks"][jnp.arange(2), actions])
    assert jnp.isfinite(loss)


def test_gciql_transformer_multidiscrete_actor_critic_loss_and_sampling():
    config = _transformer_config("gciql")
    config.actor_arch = "universal_transformer"
    batch = _multidiscrete_example_batch("factor")
    agent = create_agent(config, batch, seed=0)

    actions = agent.sample_actions(
        batch["observations"],
        batch["value_goals"],
        seed=jax.random.PRNGKey(8),
        action_masks=batch["action_masks"],
    )
    loss, _ = agent.total_loss(_without_action_metadata(batch), None)

    assert actions.shape == (2, 2)
    assert jnp.all(batch["action_masks"][jnp.arange(2)[:, None], jnp.arange(2)[None, :], actions])
    assert jnp.isfinite(loss)


def test_transformer_rejects_non_grid_observation():
    config = _transformer_config("gcdqn")
    batch = _scalar_example_batch(obs_dim=5)

    with pytest.raises(ValueError, match="not a square grid"):
        create_agent(config, batch, seed=0)


def test_gciql_discrete_rejects_ddpgbc_actor_loss():
    config = _agent_config("gciql")
    config.actor_loss = "ddpgbc"

    with pytest.raises(ValueError, match="ddpgbc is continuous-only"):
        create_agent(config, _scalar_example_batch(), seed=0)


def test_multidiscrete_env_rejects_unsupported_agent():
    config = ml_collections.ConfigDict(default_config)
    config.agent_name = "gciql_search"

    with pytest.raises(ValueError, match="MultiDiscrete action spaces.*gcdqn and gciql"):
        create_agent(config, _multidiscrete_example_batch("factor"), seed=0)


def test_flat_observation_encoding_excludes_action_mask():
    observation = FakeObservation(
        grid=jnp.array([[1, 2], [3, 4]], dtype=jnp.int32),
        step_count=jnp.array(7, dtype=jnp.int32),
        action_mask=jnp.array([False, False, False, False]),
    )

    encoded = encode_observation(observation, "flat")

    assert jnp.array_equal(encoded, jnp.array([1, 2, 3, 4, 7], dtype=jnp.float32))


def test_grid_observation_encoding_pads_channels_for_transformer_tokens():
    observation = FakeObservation(
        grid=jnp.ones((2, 2, 5), dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32),
        action_mask=jnp.ones((4,), dtype=jnp.bool_),
    )

    encoded = encode_observation(observation, "grid", grid_token_channels=12)

    assert encoded.shape == (2 * 2 * 12,)
    assert jnp.all(encoded.reshape(2, 2, 12)[..., :5] == 1)
    assert jnp.all(encoded.reshape(2, 2, 12)[..., 5:] == 0)


class FirstValidActionAgent:
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0, action_masks=None):
        del observations, goals, seed, temperature
        return jnp.argmax(action_masks, axis=-1)


def test_jumanji_render_state_rollout_collects_native_states():
    env = JumanjiDiscreteEnv(env_id="Snake-v1", episode_length=3)
    states = env.collect_render_states(FirstValidActionAgent(), jax.random.PRNGKey(0), episode_length=3)

    assert 2 <= len(states) <= 4
    assert hasattr(states[0], "head_position")


def test_jumanji_gif_logger_logs_wandb_video(monkeypatch):
    from utils import log_jumanji_gif

    env = JumanjiDiscreteEnv(env_id="Snake-v1", episode_length=2)
    logged = {}

    def fake_log(payload):
        logged.update(payload)

    monkeypatch.setattr("wandb.log", fake_log)

    log_jumanji_gif(env, FirstValidActionAgent(), jax.random.PRNGKey(0), 2, "gif_test")

    assert "gif_test" in logged
