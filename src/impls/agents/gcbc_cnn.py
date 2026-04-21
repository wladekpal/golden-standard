from typing import Any, Sequence, cast

import distrax
from flax import core, struct
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from ml_collections import config_dict

from impls.utils.flax_utils import ModuleDict, TrainState, nonpytree_field


FACTORED_BITS = 4


def _get_int(cfg: ml_collections.ConfigDict, key: str, default: int) -> int:
    value = cfg.get(key, default)
    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating, str)):
        return int(value)
    return default


def _get_float(cfg: ml_collections.ConfigDict, key: str, default: float) -> float:
    value = cfg.get(key, default)
    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating, str)):
        return float(value)
    return default


def _get_bool(cfg: ml_collections.ConfigDict, key: str, default: bool) -> bool:
    value = cfg.get(key, default)
    return default if value is None else bool(value)


def _get_int_tuple(
    cfg: ml_collections.ConfigDict,
    key: str,
    default: Sequence[int],
) -> tuple[int, ...]:
    value = cfg.get(key, default)
    if value is None:
        return tuple(int(v) for v in default)

    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            return tuple(int(v) for v in default)
        return tuple(int(p) for p in parts)

    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(int(v) for v in value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return (int(value),)

    return tuple(int(v) for v in default)


def _infer_grid_size_from_factored_flat(flat_dim: int) -> int:
    if flat_dim % FACTORED_BITS != 0:
        raise ValueError(f"Observation dim {flat_dim} is not divisible by {FACTORED_BITS} for factored_flat input.")

    num_cells = flat_dim // FACTORED_BITS
    grid_size = int(round(float(num_cells) ** 0.5))
    if grid_size * grid_size != num_cells:
        raise ValueError(
            "Factored flattened observations must correspond to a square grid: "
            f"got {num_cells} cells (flat_dim={flat_dim})."
        )

    return grid_size


class FactorizedGridCNNPolicy(nn.Module):
    action_dim: int
    grid_size: int
    conv_channels: Sequence[int] = (64, 128, 128)
    kernel_size: int = 3
    dense_dim: int = 256
    layer_norm: bool = True
    use_goals: bool = True

    def _to_grid(self, flat_inputs: jax.Array) -> jax.Array:
        grid = flat_inputs.reshape(flat_inputs.shape[0], self.grid_size, self.grid_size, FACTORED_BITS)
        return jnp.clip(grid, 0.0, 1.0)

    @nn.compact
    def __call__(self, observations, goals=None):
        obs_grid = self._to_grid(observations)

        if self.use_goals:
            if goals is None:
                goal_grid = jnp.zeros_like(obs_grid)
            else:
                goal_grid = self._to_grid(goals)
            x = jnp.concatenate([obs_grid, goal_grid], axis=-1)
        else:
            x = obs_grid

        for idx, channels in enumerate(self.conv_channels):
            x = nn.Conv(
                features=int(channels),
                kernel_size=(self.kernel_size, self.kernel_size),
                padding="SAME",
                name=f"conv_{idx}",
            )(x)
            if self.layer_norm:
                x = nn.LayerNorm(name=f"conv_ln_{idx}")(x)
            x = nn.gelu(x)

        x = x.mean(axis=(1, 2))
        x = nn.Dense(self.dense_dim, name="dense_head")(x)
        if self.layer_norm:
            x = nn.LayerNorm(name="dense_ln")(x)
        x = nn.gelu(x)
        logits = nn.Dense(self.action_dim, name="policy_head")(x)
        return logits


class CNNDiscreteActor(nn.Module):
    action_dim: int
    grid_size: int
    conv_channels: Sequence[int] = (64, 128, 128)
    kernel_size: int = 3
    dense_dim: int = 256
    layer_norm: bool = True
    use_goals: bool = True

    def setup(self):
        self.policy_net = FactorizedGridCNNPolicy(
            action_dim=self.action_dim,
            grid_size=self.grid_size,
            conv_channels=self.conv_channels,
            kernel_size=self.kernel_size,
            dense_dim=self.dense_dim,
            layer_norm=self.layer_norm,
            use_goals=self.use_goals,
        )

    def __call__(self, observations, goals=None, temperature=1.0):
        logits = self.policy_net(observations, goals)
        temperature = jnp.maximum(jnp.asarray(temperature), 1e-6)
        return distrax.Categorical(logits=logits / temperature)


class GCBCCNNAgent(struct.PyTreeNode):
    """Goal-conditioned BC agent with a factorized-grid CNN actor."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss(self, batch, grad_params, rng=None):
        del rng
        dist = self.network.select("actor")(batch["observations"], batch["actor_goals"], params=grad_params)
        log_prob = dist.log_prob(batch["actions"])

        actor_loss = -log_prob.mean()
        return actor_loss, {
            "actor_loss": actor_loss,
            "bc_log_prob": log_prob.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        del rng
        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        info = {f"actor/{k}": v for k, v in actor_info.items()}
        return actor_loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        dist = self.network.select("actor")(observations, goals, temperature=temperature)
        return dist.sample(seed=seed)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        cfg = ml_collections.ConfigDict(dict(config))
        if not _get_bool(cfg, "discrete", True):
            raise ValueError("GCBCCNNAgent supports only discrete action spaces.")

        flat_dim = int(ex_observations.shape[-1])
        grid_size = _infer_grid_size_from_factored_flat(flat_dim)

        action_dim = int(ex_actions.max() + 1)
        conv_channels = _get_int_tuple(cfg, "cnn_conv_channels", (64, 128, 128))
        if len(conv_channels) == 0:
            raise ValueError("cnn_conv_channels must contain at least one channel size.")

        kernel_size = _get_int(cfg, "cnn_kernel_size", 3)
        if kernel_size <= 0:
            raise ValueError(f"cnn_kernel_size must be > 0, got {kernel_size}.")

        dense_dim = _get_int(cfg, "cnn_dense_dim", 256)
        layer_norm = _get_bool(cfg, "layer_norm", True)
        use_goals = _get_bool(cfg, "cnn_use_goals", True)
        cfg["cnn_grid_size"] = grid_size

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        actor_def = CNNDiscreteActor(
            action_dim=action_dim,
            grid_size=grid_size,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            dense_dim=dense_dim,
            layer_norm=layer_norm,
            use_goals=use_goals,
        )

        network_info = {
            "actor": (actor_def, (ex_observations, ex_goals)),
        }
        networks: dict[str, nn.Module] = {k: v[0] for k, v in network_info.items()}
        network_args: dict[str, Any] = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=_get_float(cfg, "lr", 3e-4))
        network_params = cast(Any, network_def).init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=core.freeze(dict(cfg)))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="gcbc_cnn",
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            discount=0.99,
            const_std=True,
            discrete=True,
            encoder=config_dict.placeholder(str),
            dataset_class="GCDataset",
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            value_geom_sample=False,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=config_dict.placeholder(int),
            cnn_use_goals=True,
            cnn_conv_channels=(8, 16, 64),
            cnn_kernel_size=3,
            cnn_dense_dim=256,
        )
    )
    return config