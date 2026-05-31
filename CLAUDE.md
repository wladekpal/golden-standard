# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for the paper "Is Temporal Difference Learning the Gold Standard for Stitching in RL?" — investigates whether TD methods are superior at stitching (combining short trajectory fragments for long-horizon tasks) compared to Monte Carlo methods. Built with JAX/Flax for GPU-accelerated goal-conditioned offline RL.

## Common Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run training (GPU recommended, CPU will be very slow)
uv run src/train.py env:box-moving --exp.name test

# Run training without wandb
uv run src/train.py env:box-moving --exp.name test --exp.mode disabled

# View all hyperparameters
uv run src/train.py --help

# Run tests
pytest src/

# Run a single test file
pytest src/tests.py
pytest src/envs/block_moving/tests.py

# Lint and format (also runs automatically via pre-commit hooks)
ruff check src/
ruff format src/

# Collect expert datasets
uv run scripts/gather_expert_dataset.py --help
```

## Architecture

### Configuration System (Tyro + ml_collections)

Training uses Tyro for CLI parsing with three config namespaces:
- **`exp.*`** — Experiment settings (wandb, seeds, epochs). Defined in `src/config.py::ExpConfig`.
- **`env:box-moving`** — Environment config via Tyro subcommand. Defined in `src/envs/block_moving/env_types.py::BoxMovingConfig`.
- **`agent.*`** — Algorithm hyperparameters. Uses `ml_collections.FrozenConfigDict` (not a dataclass). Defaults in `src/impls/agents/__init__.py::default_config`.

The top-level `Config` dataclass in `src/config.py` combines all three. Agent config is a flat dict accessed like `config.agent_name`, `config.lr`, etc.

### Agent System

Agents are registered in `src/impls/agents/__init__.py`. Each agent is a `flax.struct.PyTreeNode` with `create()` and `update()` methods. Instantiation goes through `create_agent()` which dispatches by `agent_name`.

Available agents: `crl`, `crl_search`, `clearn_search`, `gciql`, `gciql_search`, `gcdqn`, `gcbc`, `qrl`, `sac`. The core paper algorithms are CRL (MC), C-Learning (TD), GCDQN, and GCIQL (each with TD/MC variants). Actions are sampled directly from Q-functions via softmax (no separate policy network for paper algorithms).

### Training Loop

`src/train.py` is the single entrypoint. The loop: collect rollouts with `collect_data()` (JIT-compiled, vmapped over `num_envs`) → store in `TrajectoryUniformSamplingQueue` (`src/rb.py`) → train agent with `update()` batches → evaluate periodically.

### Environment

`src/envs/block_moving/` implements a JAX-native grid-world (BoxMovingEnv). State is a `TimeStep` pytree with grid, agent position, goal grid. Input encoding happens via `src/envs/block_moving/input_features.py::encode_grid_inputs()` with modes: `raw_flat`, `normalized_flat`, `one_hot_flat`, `factored_flat`.

Level generators (`default` = random, `variable` = corner-based) control train/eval distribution splits for testing generalization.

### Neural Networks

Defined in `src/impls/utils/networks.py` using Flax. Key modules: `GCBilinearValue` (contrastive Q-function with dot-product energy), `GCDiscreteCritic` (standard Q-network). Architecture options: MLP or res-blocks, with optional layer norm. `src/impls/utils/flax_utils.py` provides `ModuleDict` and `TrainState` utilities.

## Code Quality

- **Ruff** handles linting and formatting (line length: 120)
- **Ruff excludes** `src/impls/` and `notebooks/` — these directories are not linted
- Pre-commit hooks run Ruff automatically on commit
- Python 3.11 required
