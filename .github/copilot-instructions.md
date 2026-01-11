# Copilot Instructions for crl_subgoal

## Project Overview
Research codebase investigating whether TD (Temporal Difference) methods are truly superior to MC (Monte Carlo) methods for "stitching" in reinforcement learning. Implements goal-conditioned RL algorithms in JAX/Flax on a Box Moving grid-world environment.

## Architecture

### Entry Point & Configuration
- **[src/train.py](src/train.py)** - Main training loop with `tyro` CLI parsing
- **[src/config.py](src/config.py)** - Dataclass configs: `ExpConfig`, `Config`
- Run: `uv run src/train.py env:box_moving --exp.name <name> --agent.<param>=<value>`

### Agent Structure (`src/impls/agents/`)
All agents are `flax.struct.PyTreeNode` subclasses with:
- `create()` classmethod for initialization
- `update()` for training step (returns new agent + info dict)
- `sample_actions()` for action selection
- `total_loss()` combining component losses

Key agents:
| Agent | Type | Notes |
|-------|------|-------|
| `crl` / `crl_search` | MC | Contrastive RL, no rewards |
| `clearn_search` | TD | C-Learning, no rewards |
| `gcdqn` | TD/MC | DQN with goal-conditioning |
| `gciql` / `gciql_search` | TD/MC | IQL with goal-conditioning |

### Network Patterns (`src/impls/utils/networks.py`)
- Use `ensemblize()` wrapper for critic ensembles
- `GCEncoder` for goal-conditioned input processing
- Networks: `GCValue`, `GCDiscreteCritic`, `GCDiscreteActor`, `GCBilinearValue`

### Environment (`src/envs/block_moving/`)
- **[block_moving_env.py](src/envs/block_moving/block_moving_env.py)** - JAX-based grid-world
- **[env_types.py](src/envs/block_moving/env_types.py)** - `BoxMovingState`, `GridStatesEnum` (12 cell states)
- Actions: UP(0), DOWN(1), LEFT(2), RIGHT(3), PICK_UP(4), PUT_DOWN(5)

## Code Conventions

### JAX/Flax Patterns
```python
# Use jax.lax.cond/switch for traced arrays, never Python if
action_result = jax.lax.switch(action, branches=[...], operand=None)

# Agent as immutable PyTreeNode - return new instance
return self.replace(network=new_network, rng=new_rng), info

# Config as FrozenConfigDict for JIT compatibility
config: Any = nonpytree_field()  # Exclude from pytree
```

### Batch Dictionary Format
```python
batch = {
    "observations": jnp.ndarray,      # (B, grid_size*grid_size)
    "next_observations": jnp.ndarray,
    "actions": jnp.ndarray,           # Integer indices for discrete
    "rewards": jnp.ndarray,
    "masks": jnp.ndarray,             # 1.0 for bootstrap
    "value_goals": jnp.ndarray,       # Flattened goal grid
    "actor_goals": jnp.ndarray,
}
```

### Loss Functions
- MC mode: `use_discounted_mc_rewards=True` → target = rewards (already discounted)
- TD mode: `use_discounted_mc_rewards=False` → target = r + γ * mask * next_v

## Development

### Commands
```bash
uv sync                              # Install dependencies
uv run src/train.py --help           # List all hyperparameters
uv run src/train.py env:box_moving --exp.name test --exp.mode disabled  # No wandb
uv run pytest src/tests.py           # Run tests
```

### Adding New Agent
1. Create `src/impls/agents/<name>.py` with agent class
2. Add `get_config()` returning `ml_collections.ConfigDict`
3. Register in `src/impls/agents/__init__.py`: add to imports and `create_agent()`

### Key Files to Understand
- [src/rb.py](src/rb.py) - `TrajectoryUniformSamplingQueue` replay buffer
- [src/impls/utils/flax_utils.py](src/impls/utils/flax_utils.py) - `ModuleDict`, `TrainState`
- [src/impls/utils/networks.py](src/impls/utils/networks.py) - All network architectures

## Linting
Ruff configured with 120 char line length, excludes `src/impls/` and `notebooks/`.
