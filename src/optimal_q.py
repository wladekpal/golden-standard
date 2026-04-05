import jax
import jax.numpy as jnp
from ott.geometry import geometry
from ott.tools.unreg import hungarian

try:
    from envs.block_moving.block_moving_env import BoxMovingEnv
    from envs.block_moving.env_types import BoxMovingState, GridStatesEnum, TimeStep
except ModuleNotFoundError:
    from src.envs.block_moving.block_moving_env import BoxMovingEnv
    from src.envs.block_moving.env_types import BoxMovingState, GridStatesEnum, TimeStep

actor_states = jnp.array(
    [
        GridStatesEnum.AGENT,
        GridStatesEnum.AGENT_CARRYING_BOX,
        GridStatesEnum.AGENT_ON_BOX,
        GridStatesEnum.AGENT_ON_TARGET,
        GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX,
        GridStatesEnum.AGENT_ON_TARGET_WITH_BOX,
        GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX,
        GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX,
    ]
)

box_states = jnp.array(
    [
        GridStatesEnum.BOX,
        GridStatesEnum.AGENT_ON_BOX,
        GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX,
    ]
)

target_states = jnp.array(
    [
        GridStatesEnum.TARGET,
        GridStatesEnum.AGENT_ON_TARGET,
        GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX,
    ]
)

carrying_states = jnp.array(
    [
        GridStatesEnum.AGENT_CARRYING_BOX,
        GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX,
        GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX,
        GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX,
    ]
)
# EMPTY = jnp.int8(0)
# BOX = jnp.int8(1)
# TARGET = jnp.int8(2)
# AGENT = jnp.int8(3)
# AGENT_CARRYING_BOX = jnp.int8(4)  # Agent is carrying a box
# AGENT_ON_BOX = jnp.int8(5)  # Agent is on box
# AGENT_ON_TARGET = jnp.int8(6)  # Agent is on target
# AGENT_ON_TARGET_CARRYING_BOX = jnp.int8(7)  # Agent is on target carrying the box
# AGENT_ON_TARGET_WITH_BOX = jnp.int8(8)  # Agent is on target on which there is a box
# AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX = jnp.int8(9)  # noqa: E501 Agent is on target on which there is a box and is carrying a box
# BOX_ON_TARGET = jnp.int8(10)  # Box is on target
# AGENT_ON_BOX_CARRYING_BOX = jnp.int8(11)  # Agent is on box carrying a box


def find_boxes(state):
    return jnp.argwhere(jnp.isin(state, box_states))


def find_goals(state):
    return jnp.argwhere(jnp.isin(state, target_states))


def find_agent(state):
    return jnp.argwhere(jnp.isin(state, actor_states))[0]


def is_carrying(state: jnp.ndarray, agent_pos: jnp.ndarray) -> bool:
    cell = state[agent_pos[0], agent_pos[1]]
    return bool(jnp.isin(cell, carrying_states))


def manhattan_cost_matrix(points_a: jnp.ndarray, points_b: jnp.ndarray) -> jnp.ndarray:
    return jnp.abs(points_a[:, None, :] - points_b[None, :, :]).sum(axis=-1).astype(jnp.float32)


def ott_assignment(source_positions: jnp.ndarray, goal_positions: jnp.ndarray) -> jnp.ndarray:
    if source_positions.shape[0] == 0:
        return jnp.zeros((0,), dtype=jnp.int32)

    cost = manhattan_cost_matrix(source_positions, goal_positions)
    _, out = hungarian(geometry.Geometry(cost_matrix=cost))
    row_idx, col_idx = out.paired_indices
    assignment = jnp.zeros((source_positions.shape[0],), dtype=jnp.int32)
    return assignment.at[row_idx.astype(jnp.int32)].set(col_idx.astype(jnp.int32))


def translate_delta_to_action(delta):
    actions = []
    if delta[0] < 0:
        actions.extend([0] * int(-delta[0]))
    if delta[0] > 0:
        actions.extend([1] * int(delta[0]))
    if delta[1] < 0:
        actions.extend([2] * int(-delta[1]))
    if delta[1] > 0:
        actions.extend([3] * int(delta[1]))
    return actions


def solve_state(state: jnp.ndarray):
    agent_position = find_agent(state)
    box_positions = find_boxes(state)
    goal_positions = find_goals(state)

    if goal_positions.shape[0] == 0:
        return []

    carrying = is_carrying(state, agent_position)

    if carrying:
        # Treat currently carried box as a source at the agent position.
        source_positions = jnp.concatenate([agent_position[None, :], box_positions], axis=0)
    else:
        source_positions = box_positions

    num_sources = int(source_positions.shape[0])
    num_goals = int(goal_positions.shape[0])
    matched = min(num_sources, num_goals)

    if matched == 0:
        return []

    source_matched = source_positions[:matched]
    goal_matched = goal_positions[:matched]
    assignment = ott_assignment(source_matched, goal_matched)

    actions = []
    current_pos = agent_position

    if carrying:
        carried_goal = goal_matched[int(assignment[0])]
        actions.extend(translate_delta_to_action(carried_goal - current_pos))
        actions.append(5)  # put down carried box
        current_pos = carried_goal
        box_goal_pairs = [
            (source_matched[i + 1], goal_matched[int(assignment[i + 1])])
            for i in range(max(0, matched - 1))
        ]
    else:
        box_goal_pairs = [
            (source_matched[i], goal_matched[int(assignment[i])])
            for i in range(matched)
        ]

    # Execute pairings in nearest-next-box order to keep planning linear-time in the episode length.
    while box_goal_pairs:
        nearest_idx = min(
            range(len(box_goal_pairs)),
            key=lambda i: int(jnp.abs(current_pos - box_goal_pairs[i][0]).sum()),
        )
        box, goal = box_goal_pairs.pop(nearest_idx)
        actions.extend(translate_delta_to_action(box - current_pos))
        actions.append(4)
        actions.extend(translate_delta_to_action(goal - box))
        actions.append(5)
        current_pos = goal

    return actions


def calculate_optimal_q_value_and_traj(env: BoxMovingEnv, state: BoxMovingState, discount=0.99):
    best_moves = solve_state(state.grid)

    trajectory = []

    last_state = state

    rewards = []

    for action in best_moves:
        current_state, reward, _, _ = env.step(last_state, int(action))
        rewards.append(reward)
        timestep = TimeStep(
            **last_state.__dict__, action=jnp.array(action, dtype=jnp.int8), done=jnp.array(False), truncated=jnp.array(False)
        )
        trajectory.append(timestep)
        last_state = current_state

    final_action = jnp.array(best_moves[-1], dtype=jnp.int8) if best_moves else jnp.array(0, dtype=jnp.int8)
    timestep = TimeStep(
        **last_state.__dict__, action=final_action, done=jnp.array(False), truncated=jnp.array(False)
    )
    trajectory.append(timestep)

    q_value = 0.0

    for reward in rewards[::-1]:
        q_value = q_value * discount + reward

    return trajectory, q_value


if __name__ == "__main__":
    env = BoxMovingEnv(
        grid_size=3,
        episode_length=100,
        number_of_boxes_max=4,
        number_of_boxes_min=4,
        number_of_moving_boxes_max=4,
    )

    key = jax.random.PRNGKey(3)
    state, _ = env.reset(key)

    traj, q_value = calculate_optimal_q_value_and_traj(env, state)
    print(f"Planned actions: {len(traj) - 1}")
    print(f"Discounted return: {float(q_value):.5f}")
    print(f"Success: {bool(traj[-1].success)}")
