from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from envs.jumanji.action_mapping import JumanjiActionMapper
from envs.jumanji.observation import encode_observation


class FakeDiscreteSpec:
    shape = ()
    dtype = jnp.int32
    num_values = 4


class FakeMultiDiscreteSpec:
    shape = (2,)
    dtype = jnp.int32
    num_values = np.array([2, 3])


class FakeObservation(NamedTuple):
    grid: jnp.ndarray
    step_count: jnp.ndarray
    action_mask: jnp.ndarray


def test_scalar_discrete_action_mapping():
    mapper = JumanjiActionMapper.from_spec(FakeDiscreteSpec(), max_flattened_action_size=16)

    assert mapper.action_dim == 4
    assert int(mapper.unflatten(jnp.array(3))) == 3
    assert jnp.array_equal(
        mapper.flatten_action_mask(jnp.array([True, False, True, False])),
        jnp.array([True, False, True, False]),
    )


def test_multidiscrete_action_mapping_and_mask_flattening():
    mapper = JumanjiActionMapper.from_spec(FakeMultiDiscreteSpec(), max_flattened_action_size=16)

    assert mapper.action_dim == 6
    assert jnp.array_equal(mapper.unflatten(jnp.array(5)), jnp.array([1, 2], dtype=jnp.int32))

    factor_mask = jnp.array(
        [
            [True, False, False],
            [False, True, True],
        ]
    )
    assert jnp.array_equal(
        mapper.flatten_action_mask(factor_mask),
        jnp.array([False, True, True, False, False, False]),
    )


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
