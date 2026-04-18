import copy
from typing import Any, Sequence

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
from impls.utils.networks import LogParam, ensemblize


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


class FactorizedGridCNNQ(nn.Module):
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
        q_values = nn.Dense(self.action_dim, name="q_head")(x)
        return q_values


class CNNDiscreteCritic(nn.Module):
    action_dim: int
    grid_size: int
    conv_channels: Sequence[int] = (64, 128, 128)
    kernel_size: int = 3
    dense_dim: int = 256
    layer_norm: bool = True
    use_goals: bool = True
    ensemble: bool = True
    ensemble_size: int = 2

    def setup(self):
        q_module = FactorizedGridCNNQ
        if self.ensemble and self.ensemble_size > 1:
            q_module = ensemblize(q_module, self.ensemble_size)

        self.q_net = q_module(
            action_dim=self.action_dim,
            grid_size=self.grid_size,
            conv_channels=self.conv_channels,
            kernel_size=self.kernel_size,
            dense_dim=self.dense_dim,
            layer_norm=self.layer_norm,
            use_goals=self.use_goals,
        )

    def __call__(self, observations, goals=None, actions=None):
        q_values = self.q_net(observations, goals)

        if actions is None:
            return q_values

        actions = actions.astype(jnp.int32)
        if q_values.ndim == 3:
            selected_q = jnp.take_along_axis(q_values, actions[None, :, None], axis=-1).squeeze(-1)
        else:
            selected_q = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze(-1)

        return selected_q


class GCDQNCNNAgent(struct.PyTreeNode):
    """Goal-conditioned DQN agent with a CNN critic over factorized grid channels."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params):
        q_values = self.network.select("critic")(batch["observations"], batch["value_goals"], params=grad_params)
        if q_values.ndim == 2:
            q_values = q_values[None, ...]

        actions = batch["actions"].astype(jnp.int32)
        q_taken = jnp.take_along_axis(q_values, actions[None, :, None], axis=-1).squeeze(-1)

        target_q_values = jax.lax.stop_gradient(
            self.network.select("target_critic")(batch["next_observations"], batch["value_goals"])
        )
        if target_q_values.ndim == 2:
            target_q_values = target_q_values[None, ...]

        max_next_q = jnp.max(target_q_values.mean(axis=0), axis=-1)
        if self.config["use_discounted_mc_rewards"]:
            target = batch["rewards"]
        else:
            target = batch["rewards"] + self.config["discount"] * batch["masks"] * max_next_q

        critic_loss = ((q_taken - target[None, :]) ** 2).mean()

        alpha_temp = self.network.select("alpha_temp")(params=grad_params)
        q_logits = jax.lax.stop_gradient(q_values.mean(axis=0))
        dist = distrax.Categorical(logits=q_logits / jnp.maximum(1e-6, alpha_temp))
        entropy = dist.entropy()
        alpha_temp_loss = ((entropy + self.config["target_entropy"]) ** 2).mean()

        total_loss = critic_loss + alpha_temp_loss
        return total_loss, {
            "critic_loss": critic_loss,
            "q_mean": target.mean(),
            "q_max": target.max(),
            "q_min": target.min(),
            "q.std": target.std(),
            "entropy": entropy.mean(),
            "entropy_std": entropy.std(),
            "alpha_temp": alpha_temp,
            "alpha_temp_loss": alpha_temp_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        info = {f"critic/{k}": v for k, v in critic_info.items()}
        return critic_loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        grads, info = jax.grad(loss_fn, has_aux=True)(self.network.params)
        new_network = self.network.apply_gradients(grads=grads)

        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            new_network.params["modules_critic"],
            new_network.params["modules_target_critic"],
        )
        if isinstance(new_network.params, core.FrozenDict):
            updated_params = new_network.params.copy(add_or_replace={"modules_target_critic": new_target_params})
        else:
            updated_params = dict(new_network.params)
            updated_params["modules_target_critic"] = new_target_params
        new_network = new_network.replace(params=updated_params)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        del temperature
        q_values = jax.lax.stop_gradient(self.network.select("critic")(observations, goals))
        if q_values.ndim == 3:
            q_values = q_values.mean(axis=0)

        action_dim = q_values.shape[-1]

        if self.config["action_sampling"] == "softmax":
            alpha_temp = jax.lax.stop_gradient(self.network.select("alpha_temp")())
            dist = distrax.Categorical(logits=q_values / jnp.maximum(1e-6, alpha_temp))
            if seed is None:
                actions = jnp.argmax(q_values, axis=-1)
            else:
                actions = dist.sample(seed=seed)
        elif self.config["action_sampling"] == "epsilon_greedy":
            greedy_actions = jnp.argmax(q_values, axis=-1)
            if seed is None:
                actions = greedy_actions
            else:
                epsilon = self.config.get("epsilon", 0.1)
                rng_actions, rng_probs = jax.random.split(seed)
                random_actions = jax.random.randint(rng_actions, greedy_actions.shape, 0, action_dim)
                probs = jax.random.uniform(rng_probs, greedy_actions.shape)
                actions = jnp.where(probs < epsilon, random_actions, greedy_actions)
        else:
            raise ValueError(f"Unknown action sampling type {self.config['action_sampling']}")

        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        cfg = ml_collections.ConfigDict(dict(config))
        if not _get_bool(cfg, "discrete", True):
            raise ValueError("GCDQNCNNAgent supports only discrete action spaces.")

        flat_dim = int(ex_observations.shape[-1])
        grid_size = _infer_grid_size_from_factored_flat(flat_dim)

        action_dim = int(ex_actions.max() + 1)
        if cfg.get("target_entropy") is None:
            target_entropy_multiplier = _get_float(cfg, "target_entropy_multiplier", 0.5)
            cfg["target_entropy"] = -target_entropy_multiplier * action_dim / 2

        conv_channels = _get_int_tuple(cfg, "cnn_conv_channels", (64, 128, 128))
        if len(conv_channels) == 0:
            raise ValueError("cnn_conv_channels must contain at least one channel size.")

        kernel_size = _get_int(cfg, "cnn_kernel_size", 3)
        if kernel_size <= 0:
            raise ValueError(f"cnn_kernel_size must be > 0, got {kernel_size}.")

        dense_dim = _get_int(cfg, "cnn_dense_dim", 256)
        critic_ensemble_size = _get_int(cfg, "critic_ensemble_size", 2)
        if critic_ensemble_size < 1:
            raise ValueError(f"critic_ensemble_size must be >= 1, got {critic_ensemble_size}.")

        layer_norm = _get_bool(cfg, "layer_norm", True)
        use_goals = _get_bool(cfg, "cnn_use_goals", True)
        cfg["cnn_grid_size"] = grid_size

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        critic_def = CNNDiscreteCritic(
            action_dim=action_dim,
            grid_size=grid_size,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            dense_dim=dense_dim,
            layer_norm=layer_norm,
            use_goals=use_goals,
            ensemble=critic_ensemble_size > 1,
            ensemble_size=critic_ensemble_size,
        )
        alpha_temp_def = LogParam()

        network_info = {
            "critic": (critic_def, (ex_observations, ex_goals)),
            "target_critic": (copy.deepcopy(critic_def), (ex_observations, ex_goals)),
            "alpha_temp": (alpha_temp_def, ()),
        }
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=_get_float(cfg, "lr", 3e-4))
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        if isinstance(network.params, core.FrozenDict):
            updated_params = network.params.copy(add_or_replace={"modules_target_critic": network.params["modules_critic"]})
        else:
            updated_params = dict(network.params)
            updated_params["modules_target_critic"] = network.params["modules_critic"]
        network = network.replace(params=updated_params)

        return cls(rng, network=network, config=core.freeze(dict(cfg)))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="gcdqn_cnn",
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            expectile=0.9,
            actor_loss="ddpgbc",
            alpha=0.3,
            const_std=True,
            discrete=True,
            encoder=config_dict.placeholder(str),
            dataset_class="GCDataset",
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=config_dict.placeholder(int),
            target_entropy_multiplier=0.5,
            target_entropy=None,
            use_discounted_mc_rewards=False,
            action_sampling="softmax",
            epsilon=0.1,
            cnn_use_goals=True,
            cnn_conv_channels=(8, 16, 64),
            cnn_kernel_size=3,
            cnn_dense_dim=256,
            critic_ensemble_size=2,
        )
    )
    return config
