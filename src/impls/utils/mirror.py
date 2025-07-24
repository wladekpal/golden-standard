from envs.block_moving_env import TimeStep, BoxPushingEnv, GridStatesEnum, ACTIONS
import jax.numpy as jnp
from jax.random import PRNGKey

def mirror_transition(timestep: TimeStep, symmetry='x') -> TimeStep:
    assert symmetry in ['x', 'y', 'xy']
    if symmetry == 'x':
        axes = -2
    elif symmetry == 'y':
        axes = -1
    else:
        axes = [-1, -2]


    new_grid = jnp.flip(timestep.grid, axis=axes)
    new_goal = jnp.flip(timestep.goal, axis=axes)
    
    grid_size = timestep.grid.shape[-1]

    agent_row, agent_col = timestep.agent_pos
    if 'x' in symmetry:
        agent_row = grid_size - agent_row - 1
    if 'y' in symmetry:
        agent_col = grid_size - agent_col - 1

    new_agent_pos = (agent_row, agent_col)


    action = timestep.action

    if 'x' in symmetry and action == 0:
        new_action = 1
    elif 'x' in symmetry and action == 1:
        new_action = 0
    elif 'y' in symmetry and action == 2:
        new_action = 3
    elif 'y' in symmetry and action == 3:
        new_action = 2
    else:
        new_action = action

    new_timestep = TimeStep(
        key=timestep.key,
        grid=new_grid,
        agent_pos=new_agent_pos,
        agent_has_box=timestep.agent_has_box,
        steps=timestep.steps,
        number_of_boxes=timestep.number_of_boxes,
        goal=new_goal,
        reward=timestep.reward,
        success=timestep.success,
        action=new_action,
        done=timestep.done,
    )

    return new_timestep

    
