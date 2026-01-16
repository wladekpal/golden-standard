import copy
import functools
import os
import tempfile

import distrax
import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from flax.traverse_util import flatten_dict
from ml_collections import ConfigDict

import wandb
from config import ROOT_DIR
from impls.agents import create_agent
from impls.agents.crl import CRLAgent
from impls.agents.crl_search import CRLSearchAgent


def calculate_params(example_batch, config):
    config_copy = copy.deepcopy(config)
    agent_config = ConfigDict(config_copy.agent.to_dict())
    agent_config.ensemble = False
    agent = create_agent(agent_config, example_batch, config.exp.seed)

    # Log number of parameters.
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(agent.network.params))
    wandb.config.update({"num_params": int(num_params)}, allow_val_change=True)
    print(f"Number of parameters: {num_params}")

    flat_params = flatten_dict(agent.network.params)
    layer_counts = {}
    for path, value in flat_params.items():
        layer_key = "/".join(str(part) for part in path[:-1]) or "/".join(str(part) for part in path)
        layer_counts[layer_key] = layer_counts.get(layer_key, 0) + value.size
    if layer_counts:
        max_name_len = max(len(name) for name in layer_counts)
        print("Parameters by layer:")
        for name, count in sorted(layer_counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {name.ljust(max_name_len)} : {count}")

    # Log effective compute when using LSTM thinking steps (parameter reuse).
    if config.agent.agent_name == "gcdqn_lstm":
        lstm_params = sum(
            x.size
            for path, x in flatten_dict(agent.network.params).items()
            if any("lstm_layers" in key for key in path)
        )
        effective_params = int(num_params - lstm_params) + int(lstm_params) * int(config.agent["thinking_steps"])
        wandb.config.update(
            {
                "thinking_steps": int(config.agent["thinking_steps"]),
                "effective_params_reapplied": effective_params,
            },
            allow_val_change=True,
        )
        print(
            "Effective compute (params reapplied) with thinking_steps "
            f"={int(config.agent['thinking_steps'])}: {effective_params}"
        )


def log_gif(original_env, episode_length, prefix_gif, timesteps):
    grid_size = timesteps.grid.shape[-2:]
    fig, ax = plt.subplots(figsize=grid_size)

    animate = functools.partial(original_env.animate, ax, timesteps, img_prefix=os.path.join(ROOT_DIR, "assets"))

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=episode_length, interval=80, repeat=False)

    # Save as GIF

    gif_file = tempfile.NamedTemporaryFile(suffix=".gif")
    gif_path = gif_file.name
    anim.save(gif_path, writer="pillow")
    plt.close()

    wandb.log({f"{prefix_gif}": wandb.Video(gif_path, format="gif")})


@jax.jit
def sample_actions_critic(
    agent,
    observations,
    goals=None,
    seed=None,
    temperature=1.0,
):
    """Sample action based on the critic q value."""

    def value_transform(x):
        return jnp.log(jnp.maximum(x, 1e-6))

    all_actions = jnp.tile(jnp.arange(6), (observations.shape[0], 1))  # B x 6
    qs = jax.lax.stop_gradient(
        jax.vmap(agent.network.select("critic"), in_axes=(None, None, 1))(observations, goals, all_actions)
    )  # 6 x 2 x B

    if len(qs.shape) == 2:  # Non-ensemble.
        qs = qs[:, None, ...]

    qs = qs.min(axis=1)  # 6 x B
    qs = qs.transpose(1, 0)  # B x 6
    if isinstance(agent, CRLAgent) or isinstance(agent, CRLSearchAgent):
        qs = value_transform(qs)
    dist = distrax.Categorical(logits=qs / jnp.maximum(1e-6, temperature))
    actions = dist.sample(seed=seed)
    return actions
