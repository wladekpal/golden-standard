# Abstract
Reinforcement learning (RL) promises to solve long-horizon tasks even when training data contains only short fragments of the behaviors. This experience stitching capability is often viewed as the purview of temporal difference (TD) methods. However, outside of small tabular settings, trajectories never intersect, calling into question this conventional wisdom. Moreover, the common belief is that Monte Carlo (MC) methods should not be able to recombine experience, yet it remains unclear whether function approximation could result in a form of implicit stitching. We empirically study whether the conventional wisdom about stitching actually holds in settings where function approximation is used. The experiments demonstrate that Monte Carlo methods can achieve experience stitching. While TD methods retain a slight edge, the gap is substantially smaller than the gap between small and large neural networks, even on simple tasks. Increasing critic capacity reduces the generalization gap for both MC and TD methods, suggesting that the traditional TD inductive bias for stitching may be less necessary in the era of large models and that scaling alone can deliver stitching in RL. 

# Installation

Using uv, there is installation happening while running the code.

```bash
uv run src/train.py 
```

Running tests:
```bash
uv run pytest src/envs/tests.py
```

# For development

Install pre-commit globally (you can [follow this article](https://adamj.eu/tech/2025/05/07/pre-commit-install-uv/)):

```bash
uv tool install pre-commit --with pre-commit-uv
```

You can also run following command to repair automatically most of the formatting/linting problems:
```bash
uv run ruff check --fix
```