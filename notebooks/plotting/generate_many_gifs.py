# %%
import os
import sys

from matplotlib import animation
sys.path.append("/home/mbortkie/repos/crl_subgoal/src")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
# %%
import functools
import os
import distrax


import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Dict, Any
from dataclasses import dataclass
import chex
from flax import struct
from absl import app, flags
from ml_collections import config_flags
from impls.agents import agents
from config import SRC_ROOT_DIR
from envs.block_moving_env import *
from train import *
from impls.utils.checkpoints import restore_agent, save_agent
from config import Config, ExpConfig
from envs import legal_envs
import matplotlib.pyplot as plt
from impls.utils.networks import GCDiscreteActor
import copy
import numpy as np
import IPython.display as display
import imageio
import numpy as np
import jax
import jax.numpy as jnp


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import imageio


# %%
RANGE_GENERALIZATION = [1,2,3,4,5,6,7,9,11]
EPISODE_LENGTH = 100
NUM_ENVS = 1024
CHECKPOINT = 50
RUN_NAME = f"DQN{CHECKPOINT}_investigation"
# MODEL_PATH = "/home/mbortkie/repos/crl_subgoal/experiments/stich_dqn_td_grid_4_20250917_044157/runs/dqn_1.38_3_grid_4_ep_len__filter_quarter"
MODEL_PATH = "/home/mbortkie/repos/crl_subgoal/experiments/stich_dqn_td_grid_6_20250923_015613/runs/dqn_te_-1.1_grid_6_boxes_3"

EPOCHS = 51
EVAL_EVERY = 10
FIGURES_PATH = f"/home/mbortkie/repos/crl_subgoal/notebooks/figures/{RUN_NAME}"
GIF_PATH = f"{FIGURES_PATH}/gifs"
os.makedirs(FIGURES_PATH, exist_ok=True)
# os.makedirs(GIF_PATH, exist_ok=True)


MODEL_PATHS =  [
    "/home/mbortkie/repos/crl_subgoal/experiments/stich_dqn_td_grid_6_20250923_015613/runs/dqn_te_-1.1_grid_6_boxes_2",
    "/home/mbortkie/repos/crl_subgoal/experiments/stich_dqn_td_grid_6_20250923_015613/runs/dqn_te_-1.1_grid_6_boxes_3",
    "/home/mbortkie/repos/crl_subgoal/experiments/stich_dqn_td_grid_6_20250923_015613/runs/dqn_te_-1.1_grid_6_boxes_4"]
for num_boxes, MODEL_PATH in zip([2,3,4],MODEL_PATHS):
    

    # %%
    config = Config(
        exp=ExpConfig(seed=0, name="test"),
        env=BoxPushingConfig(
            grid_size=6,
            number_of_boxes_min=1,
            number_of_boxes_max=1,
            number_of_moving_boxes_max=1
        )
    )

    # %%
    env = create_env(config.env)
    env = AutoResetWrapper(env)
    key = random.PRNGKey(config.exp.seed)
    env.step = jax.jit(jax.vmap(env.step))
    env.reset = jax.jit(jax.vmap(env.reset))
    partial_flatten = functools.partial(flatten_batch, get_next_obs=config.agent.use_next_obs)
    jitted_flatten_batch = jax.jit(jax.vmap(partial_flatten, in_axes=(None, 0, 0)), static_argnums=(0,))
    dummy_timestep = env.get_dummy_timestep(key)

    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=config.exp.max_replay_size,
            dummy_data_sample=dummy_timestep,
            sample_batch_size=config.agent.batch_size,
            num_envs=config.exp.num_envs,
            episode_length=config.env.episode_length,
        )
    )
    buffer_state = jax.jit(replay_buffer.init)(key)

    example_batch = {
        "observations": dummy_timestep.grid.reshape(1, -1),  # Add batch dimension
        "next_observations": dummy_timestep.grid.reshape(1, -1),
        "actions": jnp.ones((1,), dtype=jnp.int8)
        * (env._env.action_space - 1),  # it should be the maximal value of action space
        "rewards": jnp.ones((1,), dtype=jnp.int8),
        "masks": jnp.ones((1,), dtype=jnp.int8),
        "value_goals": dummy_timestep.grid.reshape(1, -1),
        "actor_goals": dummy_timestep.grid.reshape(1, -1),
    }

    # %%
    agent, config = restore_agent(example_batch, MODEL_PATH, CHECKPOINT)

    # Create env once again with correct config and collect data
    env = create_env(config.env)
    env = AutoResetWrapper(env)
    key = random.PRNGKey(config.exp.seed)


    def display_state(state, action_q_values=None, save_path=None):
    # def display_state(state, action_q_values=None):
        img_prefix="/home/mbortkie/repos/crl_subgoal/assets"
        fig, ax = plt.subplots(figsize=(5, 5))
        grid_state = state

        imgs = {
            int(GridStatesEnum.EMPTY): "floor.png",
            int(GridStatesEnum.BOX): "box.png",
            int(GridStatesEnum.TARGET): "box_target.png",
            int(GridStatesEnum.AGENT): "agent.png",
            int(GridStatesEnum.AGENT_CARRYING_BOX): "agent_carrying_box.png",
            int(GridStatesEnum.AGENT_ON_BOX): "agent_on_box.png",
            int(GridStatesEnum.AGENT_ON_TARGET): "agent_on_target.png",
            int(GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX): "agent_on_target_carrying_box.png",
            int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX): "agent_on_target_with_box.png",
            int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX): "agent_on_target_with_box_carrying_box.png",
            int(GridStatesEnum.BOX_ON_TARGET): "box_on_target.png",
            int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX): "agent_on_box_carrying_box.png",
        }

        rows, cols = grid_state.shape[0], grid_state.shape[1]

        # Plot grid: col -> x axis, row -> y axis
        for row in range(rows):
            for col in range(cols):
                img_name = imgs[int(grid_state[row, col])]
                img_path = os.path.join(img_prefix, img_name)
                if os.path.exists(img_path):
                    img = mpimg.imread(img_path)
                    ax.imshow(img, extent=[col, col + 1, row, row + 1])
                else:
                    # fallback: draw a rectangle if file missing
                    ax.add_patch(plt.Rectangle((col, row), 1, 1, fill=True, edgecolor='k', facecolor='0.05'))

        # Find agent position(s) by checking file names that contain "agent"
        agent_keys = [k for k, name in imgs.items() if 'agent' in name.lower()]
        agent_pos = None
        for row in range(rows):
            for col in range(cols):
                if int(grid_state[row, col]) in agent_keys:
                    agent_pos = (col, row)  # col -> x, row -> y
                    break
            if agent_pos is not None:
                break

        if action_q_values is not None and agent_pos is not None:
            # Accept dict {action: q} or list/array
            actions = ["Up", "Down", "Left", "Right", "Pickup", "Drop"]
            if isinstance(action_q_values, dict):
                qvals = np.array([action_q_values.get(a, 0.0) for a in actions], dtype=float)
            else:
                qvals = np.array(action_q_values, dtype=float)
                if qvals.shape[0] != len(actions):
                    raise ValueError(f"action_q_values length {qvals.shape[0]} != {len(actions)} actions")

            # Which action is best? (first argmax if ties)
            best_idx = int(np.argmax(qvals))

            # Normalize for arrow length (preserve sign for color mapping)
            q_min, q_max = qvals.min(), qvals.max()
            if q_max == q_min:
                norm_lengths = np.ones_like(qvals) * 0.6  # uniform small length
            else:
                # scale lengths to [0.15, 0.8]
                norm_lengths = 0.15 + 0.65 * (qvals - q_min) / (q_max - q_min)

            # Color map: map qvals to colors
            cmap = cm.get_cmap("coolwarm")
            norm = mcolors.Normalize(vmin=q_min, vmax=q_max)
            colors = cmap(norm(qvals))

            # center of the agent cell
            ax_x = agent_pos[0] + 0.5
            ax_y = agent_pos[1] + 0.5

            # mapping of actions to vector directions in plot coordinates
            # Note: Up has dy = -1 to visually go up after invert_yaxis()
            dir_vecs = {
                "Up": (0.0, -1.0),
                "Down": (0.0,  1.0),
                "Left": (-1.0, 0.0),
                "Right": (1.0, 0.0),
            }

            # Fixed label offsets (always the same place relative to agent)
            fixed_label_offsets = {
                "Up":    (0.0, -0.65),
                "Down":  (0.0,  0.65),
                "Left":  (-0.65, 0.0),
                "Right": (0.65, 0.0),
                "Pickup": (-0.6, -0.6),
                "Drop":   (0.6,  -0.6),  # choose positions that don't overlap too much
            }

            # Draw directional arrows (Up/Down/Left/Right)
            for i, name in enumerate(["Up", "Down", "Left", "Right"]):
                dx_unit, dy_unit = dir_vecs[name]
                # length = norm_lengths[i]
                length = 0.6
                dx = dx_unit * length
                dy = dy_unit * length
                color = colors[i]

                # highlight style for best action
                is_best = (i == best_idx)

                # start slightly offset so arrow doesn't start exactly at center
                start_x = ax_x - 0.12 * dx_unit
                start_y = ax_y - 0.12 * dy_unit
                end_x = start_x + dx
                end_y = start_y + dy

                arr = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                    arrowstyle='-|>', mutation_scale=15 * (0.9 + 0.4 * length),
                                    linewidth=(3.5 if is_best else 2.0),
                                    color=color, alpha=(1.0 if is_best else 0.95), zorder=4)
                ax.add_patch(arr)

                # place Q-value text at fixed position relative to agent (not moving with Q)
                lbl_off = fixed_label_offsets[name]
                txt_x = ax_x + lbl_off[0]
                txt_y = ax_y + lbl_off[1]
                bbox = dict(boxstyle="round,pad=0.15", fc='white', alpha=0.75, linewidth=0)
                # give a subtle highlight box for the best action
                if is_best:
                    bbox['edgecolor'] = 'gold'
                    bbox['linewidth'] = 3.0
                ax.text(txt_x, txt_y, f"{int(qvals[i])}", fontsize=9, ha='center', va='center', bbox=bbox, zorder=5)

            # Draw Pickup / Drop as small circles (with fixed positions) and labels
            for i, name in enumerate(["Pickup", "Drop"], start=4):
                is_best = (i == best_idx)
                offset = fixed_label_offsets[name]
                cx = ax_x + offset[0]
                cy = ax_y + offset[1]
                radius = 0.18 + 0.12 * norm_lengths[i]
                # if best, draw a thicker gold outline behind
                if is_best:
                    # outline circle
                    outline = Circle((cx, cy), radius=radius + 0.06, linewidth=3.0, edgecolor='gold', facecolor='none', zorder=3)
                    ax.add_patch(outline)

                circ = Circle((cx, cy), radius=radius, linewidth=(2.5 if is_best else 1.3),
                            edgecolor='k', facecolor=colors[i], alpha=(1.0 if is_best else 0.9), zorder=4)
                ax.add_patch(circ)
                ax.text(cx, cy, f"{name[0]}\n{int(qvals[i])}", fontsize=7, ha='center', va='center', color='white', zorder=5)

            # Add small colorbar legend
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Q value')

        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # keep (0,0) at top-left like array indexing; remove if you prefer origin at bottom-left
        plt.tight_layout()
        plt.show()


        # convert to array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))

        plt.close(fig)
        if save_path:
            plt.imsave(save_path, frame)
        return frame



    def create_gif(
        env,
        agent,
        state,
        steps: int = 20,
        filename: str = "trajectory.gif",
        duration: float = 0.5,
        policy: str = "argmax",         # "argmax" or "softmax"
        rng_key = jax.random.PRNGKey(0),  # optional JAX PRNGKey for reproducible softmax sampling
    ):
        """
        Run the environment from `state`, compute Qs each step, choose actions according to `policy`,
        capture frames with display_state(...), and save a GIF.

        Args:
        policy: "argmax" (choose highest-Q) or "softmax" (sample from softmax(Q/temperature)).
        temperature: positive float used when policy == "softmax".
        rng_key: optional JAX PRNGKey. If provided and policy == "softmax", sampling uses JAX and the key
                is split each step for reproducibility. If not provided, NumPy's RNG is used.
        Returns:
        filename (str) where GIF was saved.
        """
        frames = []
        state_local = state  # avoid mutating caller's reference unintentionally
        key = rng_key

        for t in range(steps):
            # --- compute Q-values (exactly like your snippet) ---
            grid = remove_targets(state_local.grid)
            goal = remove_targets(state_local.goal)
            all_actions = jnp.arange(6).reshape(1, -1)  # B x 6 (1 x 6)
            qs = jax.lax.stop_gradient(
                jax.vmap(agent.network.select("critic"), in_axes=(None, None, 1))(
                    jnp.expand_dims(grid.flatten(), 0),
                    jnp.expand_dims(goal.flatten(), 0),
                    all_actions,
                )
            )
            qs = qs.mean(axis=1)  # shape: (6, B)
            q_col = qs[:, 0]      # shape: (6,)
            qs = qs.transpose(1, 0) # B x 6

            # --- choose action according to requested policy ---
            if policy == "argmax":
                action = int(jnp.argmax(q_col))
            elif policy == "softmax":
                temperature = agent.network.select('alpha_temp')()
                # Use JAX sampling if a key was provided (reproducible), otherwise fallback to numpy
                assert key is not None, "rng_key must be provided for softmax policy"
                key, subkey = jax.random.split(key)
                dist = distrax.Categorical(logits=qs / jnp.maximum(1e-6, temperature))
                action = dist.sample(seed=subkey)[0]
                print(action)
            else:
                raise ValueError(f"Unknown policy '{policy}'; choose 'argmax' or 'softmax'")

            # --- render frame and step env ---
            q_col_np = np.array(q_col)  # convert to numpy for display_state (if it expects numpy)
            # If your display_state returns a frame (as in prior snippets), use it directly:
            frame = display_state(state_local.grid, q_col_np)  # returns uint8 RGBA or RGB frame
            frames.append(frame)

            # step environment with chosen action
            state_local, reward, done, info = env.step(state_local, action)

            # optionally print / debug
            print(f"[t={t:02d}] policy={policy}, chosen_action={action}, q={list(map(float, q_col))}")

            if done:
                # render the final state too (optional)
                q_col_np = np.array(q_col)  # last q's (or recompute if you prefer)
                final_frame = display_state(state_local.grid, q_col_np)
                frames.append(final_frame)
                print("Episode finished at step", t)
                break

        if len(frames) == 0:
            raise RuntimeError("No frames captured; GIF not created.")

        # save GIF (duration = seconds per frame)
        imageio.mimsave(filename, frames, duration=duration)
        print(f"Saved GIF to {filename}")

        # return filename (and optionally updated key if you want reproducibility)
        # If you passed rng_key and want the final key back, return (filename, key).
        if rng_key is not None:
            return filename, key
        return filename

    state, _ = env.reset(key)
    out_path = create_gif(env, agent, state, filename=f"agent_q_rollout_dqn_{num_boxes}_boxes.gif", policy="argmax", steps=100)
