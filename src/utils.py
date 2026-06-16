import functools
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wandb
from config import ROOT_DIR
import tempfile


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


def log_jumanji_gif(
    env, agent, key, episode_length, prefix_gif, use_targets=False, input_representation="normalized_flat"
):
    render_states = env.collect_render_states(
        agent,
        key,
        episode_length=episode_length,
        use_targets=use_targets,
        input_representation=input_representation,
    )

    gif_file = tempfile.NamedTemporaryFile(suffix=".gif")
    gif_path = gif_file.name
    anim = env._env.animate(render_states, interval=80, save_path=gif_path)
    plt.close(anim._fig)

    wandb.log({f"{prefix_gif}": wandb.Video(gif_path, format="gif")})
