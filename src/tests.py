from .rb import get_discounted_rewards
import jax.numpy as jnp
from .impls.utils.input_features import encode_grid_inputs


def compute_discounted_rewards_naive(rewards, gamma):
    final_rewards = []
    for episode_rewards in rewards:
        ep_final_rewards = []

        discounted_sum = 0.0

        for r in reversed(episode_rewards):
            discounted_sum = r + gamma * discounted_sum
            ep_final_rewards.append(discounted_sum)

        ep_final_rewards.reverse()
        final_rewards.extend(ep_final_rewards)

    return jnp.array(final_rewards)


def flatten_list(lst_of_lists):
    flat_list = []
    for lst in lst_of_lists:
        flat_list.extend(lst)
    return flat_list


def test_basic():
    rewards = [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 1.0]]

    gamma = 0.99

    correct_rewards = compute_discounted_rewards_naive(rewards, gamma)

    steps = [list(range(len(r))) for r in rewards]
    flat_steps = jnp.array(flatten_list(steps))

    flat_rewards = jnp.array(flatten_list(rewards))

    computed_rewards = get_discounted_rewards(flat_steps, flat_rewards, gamma)
    assert jnp.allclose(computed_rewards, correct_rewards, atol=1e-5), (
        f"Expected {correct_rewards}, but got {computed_rewards}"
    )


def test_negative_basic():
    rewards = [[-1.0, -1.0, 0.0, -1.0], [-1.0, -1.0, 0.0], [-1.0, -1.0, 0.0, -1.0, 0.0]]

    gamma = 0.99

    correct_rewards = compute_discounted_rewards_naive(rewards, gamma)

    steps = [list(range(len(r))) for r in rewards]
    flat_steps = jnp.array(flatten_list(steps))

    flat_rewards = jnp.array(flatten_list(rewards))

    computed_rewards = get_discounted_rewards(flat_steps, flat_rewards, gamma)
    assert jnp.allclose(computed_rewards, correct_rewards, atol=1e-5), (
        f"Expected {correct_rewards}, but got {computed_rewards}"
    )


def test_wrap_around():
    rewards = [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 1.0]]

    gamma = 0.99

    correct_rewards = compute_discounted_rewards_naive(rewards, gamma)

    correct_rewards = jnp.roll(correct_rewards, shift=2)

    steps = [list(range(len(r))) for r in rewards]
    flat_steps = jnp.array(flatten_list(steps))
    flat_steps = jnp.roll(flat_steps, shift=2)

    flat_rewards = jnp.array(flatten_list(rewards))
    flat_rewards = jnp.roll(flat_rewards, shift=2)

    computed_rewards = get_discounted_rewards(flat_steps, flat_rewards, gamma)
    assert jnp.allclose(computed_rewards, correct_rewards, atol=1e-5), (
        f"Expected {correct_rewards}, but got {computed_rewards}"
    )


def test_one_hot_semantic_flat_encoding():
    states = jnp.arange(12, dtype=jnp.int32).reshape(1, 3, 4)
    encoded = encode_grid_inputs(states, "one_hot_semantic_flat")

    assert encoded.shape == (1, 3 * 4 * 16)
    assert encoded.dtype == jnp.float32

    per_cell = encoded.reshape(1, 3, 4, 16)
    one_hot = per_cell[..., :12]
    semantic = per_cell[..., 12:]

    assert jnp.allclose(one_hot.sum(axis=-1), 1.0)
    assert jnp.all((one_hot == 0.0) | (one_hot == 1.0))
    assert jnp.all((semantic == 0.0) | (semantic == 1.0))

    expected = {
        0: [0, 0, 0, 0],
        1: [0, 1, 0, 0],
        2: [0, 0, 1, 0],
        3: [1, 0, 0, 0],
        4: [1, 0, 0, 1],
        5: [1, 1, 0, 0],
        6: [1, 0, 1, 0],
        7: [1, 0, 1, 1],
        8: [1, 1, 1, 0],
        9: [1, 1, 1, 1],
        10: [0, 1, 1, 0],
        11: [1, 1, 0, 1],
    }

    for s in range(12):
        r, c = divmod(s, 4)
        got = [int(x) for x in semantic[0, r, c].tolist()]
        assert got == expected[s], f"State {s}: expected {expected[s]}, got {got}"
