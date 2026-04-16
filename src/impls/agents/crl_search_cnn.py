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
from impls.utils.networks import GCDiscreteActor, LogParam, create_network, ensemblize


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


def _get_int_tuple(cfg: ml_collections.ConfigDict, key: str, default: Sequence[int]) -> tuple[int, ...]:
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


class FactorizedGridCNNFeature(nn.Module):
    grid_size: int
    conv_channels: Sequence[int] = (8, 16, 16)
    kernel_size: int = 3
    dense_dim: int = 256

    def _to_grid(self, flat_inputs: jax.Array) -> jax.Array:
        grid = flat_inputs.reshape(flat_inputs.shape[0], self.grid_size, self.grid_size, FACTORED_BITS)
        # Drop has_target channel to avoid leaking target occupancy into both state and goal streams.
        grid = jnp.concatenate([grid[..., :1], grid[..., 2:]], axis=-1)
        return jnp.clip(grid, 0.0, 1.0)

    @nn.compact
    def __call__(self, flat_inputs):
        x = self._to_grid(flat_inputs)

        for idx, channels in enumerate(self.conv_channels):
            x = nn.Conv(
                features=int(channels),
                kernel_size=(self.kernel_size, self.kernel_size),
                padding="SAME",
                name=f"conv_{idx}",
            )(x)
            x = nn.gelu(x)

        x = x.mean(axis=(1, 2))
        x = nn.Dense(self.dense_dim, name="dense_head")(x)
        x = nn.gelu(x)
        return x


class GCDiscreteBilinearCNNCritic(nn.Module):
    action_dim: int
    grid_size: int
    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    ensemble_size: int = 2
    value_exp: bool = True
    conv_channels: Sequence[int] = (8, 16, 16)
    kernel_size: int = 3
    dense_dim: int = 256
    net_arch: str = "mlp"

    def setup(self):
        self.shared_encoder = FactorizedGridCNNFeature(
            grid_size=self.grid_size,
            conv_channels=self.conv_channels,
            kernel_size=self.kernel_size,
            dense_dim=self.dense_dim,
            name="shared_encoder",
        )

        mlp_module = create_network(self.net_arch)
        if self.ensemble and self.ensemble_size > 1:
            mlp_module = ensemblize(mlp_module, self.ensemble_size)

        self.phi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.psi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, actions=None, info=False):
        obs_repr = self.shared_encoder(observations)
        goal_repr = self.shared_encoder(goals)

        if actions is None:
            action_one_hot = jnp.zeros((obs_repr.shape[0], self.action_dim), dtype=obs_repr.dtype)
        else:
            action_one_hot = jax.nn.one_hot(actions.astype(jnp.int32), self.action_dim, dtype=obs_repr.dtype)

        phi_inputs = jnp.concatenate([obs_repr, action_one_hot], axis=-1)
        phi = self.phi(phi_inputs)
        psi = self.psi(goal_repr)

        q = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if self.value_exp:
            q = jnp.exp(q)

        if info:
            return q, phi, psi
        return q


class CRLSearchCNNAgent(struct.PyTreeNode):
    """CRL-search agent with a factorized-grid CNN critic."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def contrastive_loss(self, batch, grad_params, module_name='critic'):
        """Compute the contrastive value loss for the Q or V function."""
        batch_size = batch['observations'].shape[0]
        actions = batch['actions']

        q, phi, psi = self.network.select(module_name)(
            batch['observations'],
            batch['value_goals'],
            actions=actions,
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 2:  # Non-ensemble.
            phi = phi[None, ...]
            psi = psi[None, ...]
        logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])
        # logits.shape is (B, B, e) with one term for positive pair and (B - 1) terms for negative pairs in each row.
        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        # Compute additional statistics.
        logits = jnp.mean(logits, axis=-1) # (B, B)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        # Update target entropy
        def value_transform(x):
            return jnp.log(jnp.maximum(x, 1e-6))
        
        all_actions = jnp.tile(jnp.arange(6), (batch['observations'].shape[0], 1))  # B x 6
        qs = jax.lax.stop_gradient(
            jax.vmap(self.network.select("critic"), in_axes=(None, None, 1))(batch['observations'], 
                                                                            #  jnp.roll(batch['next_observations'], shift=1, axis=0),
                                                                             batch['value_goals'],
                                                                               all_actions)
        )  # 6 x 2 x B
        if len(qs.shape) == 2:  # Non-ensemble.
            qs = qs[:, None, ...]
        qs = qs.mean(axis=1)  # 6 x B
        qs = qs.transpose(1, 0) # B x 6
        qs = value_transform(qs)

        alpha_temp = self.network.select('alpha_temp')(params=grad_params)
        dist = distrax.Categorical(logits=qs / jnp.maximum(1e-6, alpha_temp))
        entropy = dist.entropy()
        alpha_temp_loss = ((entropy + self.config['target_entropy'])**2).mean()  

        total_loss = contrastive_loss + alpha_temp_loss
        return total_loss, {
            'contrastive_loss': contrastive_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
            'entropy': entropy.mean(),
            'alpha_temp': alpha_temp,
            'entropy_std': dist.entropy().std(),
            'alpha_temp_loss': alpha_temp_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        del rng
        critic_loss, critic_info = self.contrastive_loss(batch, grad_params, "critic")
        info = {f"critic/{k}": v for k, v in critic_info.items()}
        return critic_loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """
        Returns integer action indices. Continuous actions are not supported here.
        """
        if not self.config['discrete']:
            raise NotImplementedError("ClearnSearchAgent.sample_actions supports only discrete action spaces.")
        
        def value_transform(x):
            return jnp.log(jnp.maximum(x, 1e-6))
        

        all_actions = jnp.tile(jnp.arange(6), (observations.shape[0], 1))  # B x 6
        qs = jax.lax.stop_gradient(jax.vmap(self.network.select('critic'), in_axes=(None, None, 1))(observations, goals, all_actions)) # 6 x 2 x B
        qs = qs.mean(axis=1) # 6 x B
        qs = value_transform(qs)
        qs = qs.transpose(1, 0) # B x 6
        
        if self.config['action_sampling'] == 'softmax':
            # Use critic to get Q-values (use first/ensemble as appropriate). Prefer the minimum head for conservative action,
            # or average — here we average the two heads and pick argmax.

            # Softmax actions
            alpha_temp = jax.lax.stop_gradient(self.network.select('alpha_temp')())
            dist = distrax.Categorical(logits=qs / jnp.maximum(1e-6, alpha_temp))
            actions = dist.sample(seed=seed)
        elif self.config['action_sampling'] == 'epsilon_greedy':
            greedy_actions = jnp.argmax(qs, axis=-1)  # B
            # random actions
            rng, rng_uniform = jax.random.split(seed)
            random_actions = jax.random.randint(rng, greedy_actions.shape, 0, 6)

            # ε-greedy: pick random with prob ε, else greedy
            probs = jax.random.uniform(rng_uniform, greedy_actions.shape)
            actions = jnp.where(probs < 0.1, random_actions, greedy_actions)
        else:
            raise ValueError(f"Unknown action sampling type {self.config['action_sampling']}")

        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, ex_goals=None):
        cfg = ml_collections.ConfigDict(dict(config))

        if ex_goals is None:
            ex_goals = ex_observations

        if not cfg["discrete"]:
            raise ValueError("CRLSearchCNNAgent supports only discrete action spaces.")

        flat_dim = int(ex_observations.shape[-1])
        grid_size = _infer_grid_size_from_factored_flat(flat_dim)

        action_dim = int(ex_actions.max() + 1)
        cfg["action_dim"] = action_dim
        cfg["cnn_grid_size"] = grid_size

        if cfg.get("target_entropy") is None:
            target_entropy_multiplier = _get_float(cfg, "target_entropy_multiplier", 0.5)
            cfg["target_entropy"] = -target_entropy_multiplier * action_dim / 2

        conv_channels = _get_int_tuple(cfg, "cnn_conv_channels", (8, 16, 16))
        if len(conv_channels) == 0:
            raise ValueError("cnn_conv_channels must contain at least one channel size.")

        kernel_size = _get_int(cfg, "cnn_kernel_size", 3)
        if kernel_size <= 0:
            raise ValueError(f"cnn_kernel_size must be > 0, got {kernel_size}.")

        dense_dim = _get_int(cfg, "cnn_dense_dim", 256)
        critic_ensemble_size = _get_int(cfg, "critic_ensemble_size", 2)
        if critic_ensemble_size < 1:
            raise ValueError(f"critic_ensemble_size must be >= 1, got {critic_ensemble_size}.")
        value_hidden_dims = _get_int_tuple(cfg, "value_hidden_dims", (256, 256))
        actor_hidden_dims = _get_int_tuple(cfg, "actor_hidden_dims", (256, 256))
        net_arch = str(cfg.get("net_arch", "mlp"))

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        critic_def = GCDiscreteBilinearCNNCritic(
            action_dim=action_dim,
            grid_size=grid_size,
            hidden_dims=value_hidden_dims,
            latent_dim=_get_int(cfg, "latent_dim", 64),
            layer_norm=bool(cfg["layer_norm"]),
            ensemble=critic_ensemble_size > 1,
            ensemble_size=critic_ensemble_size,
            value_exp=True,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            dense_dim=dense_dim,
            net_arch=net_arch,
        )

        actor_def = GCDiscreteActor(
            hidden_dims=actor_hidden_dims,
            action_dim=action_dim,
            gc_encoder=cast(Any, None),
            net_arch=net_arch,
        )
        alpha_temp_def = LogParam()

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
            alpha_temp=(alpha_temp_def, ()),
        )
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
            agent_name="crl_search_cnn",
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(256, 256),
            value_hidden_dims=(256, 256),
            latent_dim=64,
            net_arch="mlp",
            layer_norm=True,
            discount=0.99,
            actor_loss="awr",
            alpha=0.1,
            actor_log_q=True,
            const_std=True,
            discrete=True,
            encoder=config_dict.placeholder(str),
            dataset_class="GCDataset",
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=False,
            p_aug=0.0,
            frame_stack=config_dict.placeholder(int),
            target_entropy_multiplier=0.5,
            target_entropy=-jnp.log(6),
            action_sampling="softmax",
            epsilon=0.1,
            cnn_conv_channels=(8, 16, 64),
            cnn_kernel_size=3,
            cnn_dense_dim=256,
            critic_ensemble_size=2,
        )
    )
    return config
