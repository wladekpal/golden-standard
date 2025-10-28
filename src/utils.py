import functools
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wandb
from config import ROOT_DIR


def log_gif(original_env, episode_length, prefix_gif, timesteps):
    grid_size = timesteps.grid.shape[-2:]
    fig, ax = plt.subplots(figsize=grid_size)

    animate = functools.partial(original_env.animate, ax, timesteps, img_prefix=os.path.join(ROOT_DIR, "assets"))

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=episode_length, interval=80, repeat=False)

    # Save as GIF
    gif_path = "/tmp/block_moving_epoch.gif"
    anim.save(gif_path, writer="pillow")
    plt.close()

    wandb.log({f"{prefix_gif}": wandb.Video(gif_path, format="gif")})
