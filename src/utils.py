import functools
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wandb
from config import ROOT_DIR
import jax 
import jax.numpy as jnp
import distrax

def log_gif(original_env, episode_length, prefix_gif, timesteps, state):
    grid_size = state.grid.shape[-2:]
    fig, ax = plt.subplots(figsize=grid_size)
        
    animate = functools.partial(original_env.animate, ax, timesteps, img_prefix=os.path.join(ROOT_DIR, 'assets'))
        
        # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=episode_length, interval=80, repeat=False)
        
        # Save as GIF
    gif_path = f"/tmp/block_moving_epoch.gif"
    anim.save(gif_path, writer='pillow')
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
    """Sample based on the critic q value."""
    def value_transform(x):
        return jnp.log(jnp.maximum(x, 1e-6))
    all_actions = jnp.tile(jnp.arange(6), (observations.shape[0], 1))  # B x 6
    qs = jax.lax.stop_gradient(jax.vmap(agent.network.select('critic'), in_axes=(None, None, 1))(observations, goals, all_actions)) # 6 x 2 x B
    qs = qs.min(axis=1) # 6 x B
    qs = value_transform(qs)
    qs = qs.transpose(1, 0) # B x 6
    qs = (qs-qs.mean(axis=1, keepdims=True)) / jnp.maximum(1e-6, qs.std(axis=1, keepdims=True))  # Normalize logits.
    dist = distrax.Categorical(logits=qs / jnp.maximum(1e-6, temperature))
    actions = dist.sample(seed=seed)
    return actions
