
# Overview

Reinforcement learning (RL) promises to solve long-horizon tasks even when training data contains only short fragments of the behaviors. This quality is called stitching, and is a crucial prerequisite for more general, foundational RL models. Conventional wisdom dictates, that only temporal difference (TD) methods are able stitch fragments of experiences gathered during the training and use them to solve more complex tasks. We show that, while on simple, low-dimensional settings TD methods can indeed stitch experiences, this does not transfer to more complex, high-dimensional tasks. Additionally we show that Monte Carlo (MC) methods, while they still fall behind TD methods, are able to exhibit some stitching behavior as well. Furthermore we determine that scaling the network sizes plays more of a critical role in closing the generalization gap than previously thought, and is a promising avenue of research, especially in the age of larger models in RL.


<p align="center">
    <video autoplay muted loop playsinline preload="metadata" style="width:45%; height:auto; object-fit:contain; border-radius:0;" aria-label="TD stitching success view 1">
        <source src="assets/stitching_1_success.mp4" type="video/mp4">
    </video>
    <video autoplay muted loop playsinline preload="metadata" style="width:45%; height:auto; object-fit:contain; border-radius:0;" aria-label="TD stitching failure view 2">
        <source src="assets/stitching_4_failure.mp4" type="video/mp4">
    </video>
</p>

# Installation $ Setup

This repo uses uv. To install uv, please follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
To install all dependencies and create the virual environment run:
```
uv sync
```
> [!NOTE]
> We are using `wandb` for experiment tracking by default, you may be prompted to login to wandb when running first experiment. If you don't want to use wandb, you can use `--exp.mode disabled` flag to skip wandb logging.

# Running experiments
To run a simple training with CRL and default environment configuration use:
> [!WARNING]
> Our repository is optimized for GPU, running line below without decent GPU may take a very long time.
```bash
uv run src/train.py env:box-moving --exp.name test
```

Current version of the code only supports `box-moving` environment, thus in each experiment you should specify `env:box-moving` flag first.

# For development

Install pre-commit globally (you can [follow this article](https://adamj.eu/tech/2025/05/07/pre-commit-install-uv/)):

```bash
uv tool install pre-commit --with pre-commit-uv
```

You can also run following command to repair automatically most of the formatting/linting problems:
```bash
uv run ruff check --fix
```