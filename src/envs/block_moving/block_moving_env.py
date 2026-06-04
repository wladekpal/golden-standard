import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any
import chex
import matplotlib
import os
import logging
from .env_types import (
    BoxMovingState,
    GridStatesEnum,
    remove_targets,
    _ACTION_DELTAS,
    _REMOVE_AGENT_ARRAY,
    _ADD_AGENT_ARRAY,
    _ADD_AGENT_CARRYING_ARRAY,
    _PICK_UP_ARRAY,
    _PICK_UP_VALID,
    _PUT_DOWN_ARRAY,
    _PUT_DOWN_VALID,
)
from .generators import DefaultLevelGenerator, VariableQuarterGenerator


class BoxMovingEnv:
    """JAX-based box moving environment."""

    def __init__(
        self,
        grid_size: int = 20,
        episode_length: int = 2000,
        number_of_boxes_min: int = 3,
        number_of_boxes_max: int = 4,
        number_of_moving_boxes_max: int = 2,
        terminate_when_success: bool = False,
        dense_rewards: bool = False,
        negative_sparse: bool = True,
        level_generator: str = "default",
        quarter_size: int | None = None,
        **kwargs,
    ):
        logging.info(
            f"Initializing BoxMovingEnv with grid_size={grid_size}, episode_length={episode_length}, number_of_boxes={number_of_boxes_min}, number_of_boxes_max={number_of_boxes_max}, number_of_moving_boxes_max={number_of_moving_boxes_max}"
        )
        self.grid_size = grid_size
        self.episode_length = episode_length
        self.action_space = 6  # UP, DOWN, LEFT, RIGHT, PICK_UP, PUT_DOWN
        self.number_of_boxes_min = number_of_boxes_min
        self.number_of_boxes_max = number_of_boxes_max
        self.number_of_moving_boxes_max = number_of_moving_boxes_max
        self.terminate_when_success = terminate_when_success
        self.dense_rewards = dense_rewards
        self.negative_sparse = negative_sparse
        self.quarter_size = quarter_size or self.grid_size // 2

        if level_generator == "default":
            self.level_generator = DefaultLevelGenerator(
                grid_size, number_of_boxes_min, number_of_boxes_max, number_of_moving_boxes_max
            )
        elif level_generator == "variable":
            self.level_generator = VariableQuarterGenerator(
                grid_size,
                number_of_boxes_min,
                number_of_boxes_max,
                number_of_moving_boxes_max,
                self.quarter_size,
                special=kwargs["generator_special"],
            )
        else:
            raise ValueError("Unknown level generator selected")

    def reset(self, key: jax.Array) -> Tuple[BoxMovingState, Dict[str, Any]]:
        """Reset environment to initial state."""
        state = self.level_generator.generate(key)
        grid = state.grid

        info = {
            "truncated": jnp.bool_(False),
            "boxes_on_target": jnp.sum(grid == GridStatesEnum.BOX_ON_TARGET)
            + jnp.sum(grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX)
            + jnp.sum(grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX),
        }

        return state, info

    def step(self, state: BoxMovingState, action: int) -> Tuple[BoxMovingState, float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        chex.assert_shape(action, ())

        # Increment steps
        new_steps = state.steps + 1
        done = False

        # Use jax.lax.switch instead of if-elif to handle traced arrays
        action_result = jax.lax.switch(
            action,
            branches=[
                lambda _: self.handle_movement(state, 0),  # Movement action 0
                lambda _: self.handle_movement(state, 1),  # Movement action 1
                lambda _: self.handle_movement(state, 2),  # Movement action 2
                lambda _: self.handle_movement(state, 3),  # Movement action 3
                lambda _: (state.agent_pos, *self._handle_pickup(state)),  # PICK_UP
                lambda _: (state.agent_pos, *self._handle_putdown(state)),  # PUT_DOWN
            ],
            operand=None,
        )

        new_pos, new_grid, new_agent_has_box = action_result

        truncated = new_steps >= self.episode_length
        # Checking if agent reaches the goal in next state (new_grid), if yes then success = 1
        reward = BoxMovingEnv.get_reward(state.grid, new_grid, state.goal).astype(jnp.float32)
        success = reward.astype(jnp.int32)
        if self.terminate_when_success:
            done = success.astype(bool)

        new_state = BoxMovingState(
            key=state.key,
            grid=new_grid,
            agent_pos=new_pos,
            agent_has_box=new_agent_has_box,
            steps=new_steps,
            number_of_boxes=state.number_of_boxes,
            goal=state.goal,
            reward=reward,
            success=success,
            extras=state.extras,
        )

        info = {
            "truncated": truncated,
            "boxes_on_target": jnp.sum(new_grid == GridStatesEnum.BOX_ON_TARGET)
            + jnp.sum(new_grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX)
            + jnp.sum(new_grid == GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX),
        }

        return new_state, reward, done, info

    def handle_movement(self, state: BoxMovingState, action: int) -> Tuple[jax.Array, bool]:
        row, col = state.agent_pos[0], state.agent_pos[1]
        grid = state.grid
        new_row = row + _ACTION_DELTAS[action, 0]
        new_col = col + _ACTION_DELTAS[action, 1]

        # Moves off the board are rejected; the destination cell is always agent-free.
        valid_move = (new_row >= 0) & (new_row < self.grid_size) & (new_col >= 0) & (new_col < self.grid_size)

        def move_valid():
            # Remove the agent from its old cell, then drop it (with or without box) on the destination.
            grid_after_clear = grid.at[row, col].set(_REMOVE_AGENT_ARRAY[grid[row, col]])
            dest_cell = grid_after_clear[new_row, new_col]
            new_field_with_agent = jnp.where(
                state.agent_has_box, _ADD_AGENT_CARRYING_ARRAY[dest_cell], _ADD_AGENT_ARRAY[dest_cell]
            )
            new_grid = grid_after_clear.at[new_row, new_col].set(new_field_with_agent)
            return jnp.array([new_row, new_col]), new_grid, state.agent_has_box

        def move_invalid():
            return state.agent_pos, grid, state.agent_has_box

        return jax.lax.cond(valid_move, move_valid, move_invalid)

    @staticmethod
    @jax.jit
    def get_reward(
        old_grid: jax.Array,
        new_grid: jax.Array,
        goal_grid: jax.Array,
    ) -> jax.Array:
        # To make it work also when goal and new_grid have different target locations,
        # we first remove targets from both, and then we remove agent.
        new_grid_no_targets = remove_targets(new_grid)
        goal_grid_no_targets = remove_targets(goal_grid)

        new_final = _REMOVE_AGENT_ARRAY[new_grid_no_targets]
        goal_final = _REMOVE_AGENT_ARRAY[goal_grid_no_targets]

        solved = jnp.all(new_final == goal_final).astype(jnp.float32)
        return solved

    def _handle_pickup(self, state: BoxMovingState) -> Tuple[jax.Array, bool]:
        """Pick up the box under the agent (no-op unless standing on a box, not yet carrying)."""
        row, col = state.agent_pos[0], state.agent_pos[1]
        cell = state.grid[row, col]
        new_grid = state.grid.at[row, col].set(_PICK_UP_ARRAY[cell])
        new_agent_has_box = state.agent_has_box | _PICK_UP_VALID[cell]
        return new_grid, new_agent_has_box

    def _handle_putdown(self, state: BoxMovingState) -> Tuple[jax.Array, bool]:
        """Put the carried box down on the current cell (no-op unless carrying)."""
        row, col = state.agent_pos[0], state.agent_pos[1]
        cell = state.grid[row, col]
        new_grid = state.grid.at[row, col].set(_PUT_DOWN_ARRAY[cell])
        new_agent_has_box = state.agent_has_box & ~_PUT_DOWN_VALID[cell]
        return new_grid, new_agent_has_box

    def play_game(self, key: jax.Array):
        """Interactive game loop using input() for controls."""

        # Initialize the environment
        state, _ = self.reset(key)
        done = False
        info = {}
        total_reward = 0
        reward = 0

        print("=== Box Moving Game ===")
        print("Controls: w(up), s(down), a(left), d(right), e(pickup), r(drop)")
        print("Goal: Move boxes to target cells (marked with 'T')")
        print("Press 'q' to quit\n")

        while True:
            # Display current state
            self._display_state(state)
            no_targets = remove_targets(state.grid)
            print(no_targets)
            print(f"Steps: {state.steps}, Return: {total_reward}, Reward: {reward}")
            print(f"Info: {info}, Done: {done}")

            # Get user input
            action = None
            while action is None:
                try:
                    user_input = input("Enter action (w/s/a/d/e/r/q/g): ").lower().strip()
                    if user_input == "w":
                        action = 0  # Up
                    elif user_input == "s":
                        action = 1  # Down
                    elif user_input == "a":
                        action = 2  # Left
                    elif user_input == "d":
                        action = 3  # Right
                    elif user_input == "e":
                        action = 4  # Pickup
                    elif user_input == "r":
                        action = 5  # Drop
                    elif user_input == "q":
                        print("Game ended by user.")
                        return
                    elif user_input == "g":
                        print("Game restarted by user.")
                        action = "restart"
                    else:
                        print("Invalid input. Use w/s/a/d/e/r/q/g")
                except (EOFError, KeyboardInterrupt):
                    print("\nGame ended by user.")
                    return

            # Take action
            if action == "restart":
                state, info = self.reset(state.key)
            else:
                state, reward, done, info = self.step(state, action)
                total_reward += reward

    def _display_state(self, state: BoxMovingState):
        """Display the current game state in ASCII."""
        print("\n" + "=" * (self.grid_size * 2 + 1))
        print(state.grid)
        print("=" * (self.grid_size * 2 + 1))

    def get_dummy_timestep(self, key):
        return self.level_generator.get_dummy_timestep(key)

    @staticmethod
    def animate(ax, timesteps, frame, img_prefix="assets"):
        ax.clear()
        grid_state = timesteps.grid[0, frame]
        action = timesteps.action[0, frame]
        reward = timesteps.reward[0, frame]

        # Create color mapping for grid states
        imgs = {
            int(GridStatesEnum.EMPTY): "floor.png",  # EMPTY
            int(GridStatesEnum.BOX): "box.png",  # BOX
            int(GridStatesEnum.TARGET): "box_target.png",  # TARGET
            int(GridStatesEnum.AGENT): "agent.png",  # AGENT
            int(GridStatesEnum.AGENT_CARRYING_BOX): "agent_carrying_box.png",  # AGENT_CARRYING_BOX
            int(GridStatesEnum.AGENT_ON_BOX): "agent_on_box.png",  # AGENT_ON_BOX
            int(GridStatesEnum.AGENT_ON_TARGET): "agent_on_target.png",  # AGENT_ON_TARGET
            int(GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX): "agent_on_target_carrying_box.png",  # noqa: E501 AGENT_ON_TARGET_CARRYING_BOX
            int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX): "agent_on_target_with_box.png",  # AGENT_ON_TARGET_WITH_BOX
            int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX): "agent_on_target_with_box_carrying_box.png",  # noqa: E501 AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX
            int(GridStatesEnum.BOX_ON_TARGET): "box_on_target.png",  # BOX_ON_TARGET
            int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX): "agent_on_box_carrying_box.png",  # AGENT_ON_BOX_CARRYING_BOX
        }

        # Plot grid
        for i in range(grid_state.shape[0]):
            for j in range(grid_state.shape[1]):
                img_name = imgs[int(grid_state[i, j])]
                img_path = os.path.join(img_prefix, img_name)
                img = matplotlib.image.imread(img_path)
                ax.imshow(img, extent=[i + 1, i, j + 1, j])

        ax.set_xlim(0, grid_state.shape[1])
        ax.set_ylim(0, grid_state.shape[0])
        ax.set_title(f"Step {frame} | Action: {action} | Reward: {reward:.2f}")
        ax.set_aspect("equal")
        ax.invert_yaxis()
