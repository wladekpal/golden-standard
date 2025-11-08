from envs.block_moving.env_types import GridStatesEnum, BoxMovingState
from envs.block_moving.block_moving_env import BoxMovingEnv
import jax.numpy as jnp
from itertools import permutations, product


def find_boxes(state):
    return jnp.argwhere(state == GridStatesEnum.BOX)


def find_goals(state):
    return jnp.argwhere(state == GridStatesEnum.TARGET)


def find_agent(state):
    return jnp.argwhere(state >= GridStatesEnum.AGENT)[0]


forbidden_states = jnp.array(
    [
        GridStatesEnum.AGENT_CARRYING_BOX,
        GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX,
        GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX,
        GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX,
    ]
)


def generate_permutations(positions):
    return list(jnp.array(p) for p in permutations(positions))


def compute_cost(agent_pos, box_pos, goal_pos):
    moves = 0
    moves += jnp.abs(agent_pos - box_pos[0]).sum()
    moves += jnp.abs(box_pos - goal_pos).sum()
    moves += box_pos.shape[0]

    return moves


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


def retrieve_moves(agent_pos, box_pos, goal_pos):
    actions = []

    current_pos = agent_pos

    for box, goal in zip(box_pos, goal_pos):
        delta = box - current_pos
        actions.extend(translate_delta_to_action(delta))
        actions.append(4)
        delta = goal - box
        actions.extend(translate_delta_to_action(delta))
        actions.append(5)
        current_pos = goal

    return actions


def solve_state(state):
    agent_position = find_agent(state)
    if (state[agent_position[0], agent_position[1]] == forbidden_states).any():
        raise ValueError("State with agent carrying box is not supported.")

    box_positions = find_boxes(state)
    goal_positions = find_goals(state)

    box_permutations = generate_permutations(box_positions)
    goal_permutations = generate_permutations(goal_positions)

    pairings = list(product(box_permutations, goal_permutations))

    min_cost = jnp.inf
    best_path = None

    for pairing in pairings:
        boxes, goals = pairing
        cost = compute_cost(agent_position, boxes, goals)
        if cost < min_cost:
            min_cost = cost
            best_path = (boxes, goals)

    best_moves = retrieve_moves(agent_position, best_path[0], best_path[1])

    return best_moves


def calculate_optimal_q_value_and_traj(env: BoxMovingEnv, state: BoxMovingState, discount=0.99):
    best_moves = solve_state(state.grid)

    trajectory = [state]

    last_state = state

    rewards = []

    for action in best_moves:
        last_state = trajectory[-1]
        last_state, reward, _, _ = env.step(last_state, action)
        rewards.append(reward)
        trajectory.append(last_state)

    q_value = 0

    for reward in rewards[::-1]:
        q_value = q_value * discount + reward

    return trajectory, q_value


if __name__ == "__main__":
    test_state = jnp.array(
        [
            [0, 0, 1, 0, 3],
            [0, 2, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    print(solve_state(test_state))

    # xd = jnp.array([
    #     [1, 2],
    #     [3, 4],
    #     [5, 6],
    #     [7, 8],
    # ])
    # print(xd)
    # perms = generate_permutations(xd)
    # print(len(perms))
    # for p in perms:
    #     print(p)
