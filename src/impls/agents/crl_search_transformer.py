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


NUM_GRID_STATES = 12
FACTORED_BITS = 4
_FACTORED_VALID_CODES = np.array([0, 8, 4, 2, 3, 10, 6, 7, 14, 15, 12, 11], dtype=np.int32)
_FACTORED_CODE_TO_STATE = jnp.array(
    [
        0,  # 0000 -> EMPTY
        0,
        3,  # 0010 -> AGENT
        4,  # 0011 -> AGENT_CARRYING_BOX
        2,  # 0100 -> TARGET
        0,
        6,  # 0110 -> AGENT_ON_TARGET
        7,  # 0111 -> AGENT_ON_TARGET_CARRYING_BOX
        1,  # 1000 -> BOX
        0,
        5,  # 1010 -> AGENT_ON_BOX
        11,  # 1011 -> AGENT_ON_BOX_CARRYING_BOX
        10,  # 1100 -> BOX_ON_TARGET
        0,
        8,  # 1110 -> AGENT_ON_TARGET_WITH_BOX
        9,  # 1111 -> AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX
    ],
    dtype=jnp.int32,
)
_TARGET_MASKED_STATE = jnp.array(
    [
        0,  # EMPTY -> EMPTY
        1,  # BOX -> BOX
        0,  # TARGET -> EMPTY
        3,  # AGENT -> AGENT
        4,  # AGENT_CARRYING_BOX -> AGENT_CARRYING_BOX
        5,  # AGENT_ON_BOX -> AGENT_ON_BOX
        3,  # AGENT_ON_TARGET -> AGENT
        4,  # AGENT_ON_TARGET_CARRYING_BOX -> AGENT_CARRYING_BOX
        5,  # AGENT_ON_TARGET_WITH_BOX -> AGENT_ON_BOX
        11,  # AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX -> AGENT_ON_BOX_CARRYING_BOX
        1,  # BOX_ON_TARGET -> BOX
        11,  # AGENT_ON_BOX_CARRYING_BOX -> AGENT_ON_BOX_CARRYING_BOX
    ],
    dtype=jnp.int32,
)


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


def _get_str(cfg: ml_collections.ConfigDict, key: str, default: str) -> str:
    value = cfg.get(key, default)
    return default if value is None else str(value)


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


def _infer_token_representation(observations: np.ndarray, num_grid_states: int) -> str:
    """Infer token representation from flattened observation shape and value pattern."""
    flat_dim = observations.shape[-1]
    obs = observations.reshape(-1, flat_dim)

    if flat_dim % num_grid_states == 0:
        reshaped = obs.reshape(obs.shape[0], -1, num_grid_states)
        valid_range = np.all((reshaped >= -1e-3) & (reshaped <= 1.0 + 1e-3))
        row_sums = reshaped.sum(axis=-1)
        if valid_range and np.allclose(row_sums, 1.0, atol=1e-3):
            return "one_hot_flat"

    if flat_dim % FACTORED_BITS == 0:
        reshaped = obs.reshape(obs.shape[0], -1, FACTORED_BITS)
        valid_range = np.all((reshaped >= -1e-3) & (reshaped <= 1.0 + 1e-3))
        if valid_range:
            bits = (reshaped > 0.5).astype(np.int32)
            codes = bits[..., 0] * 8 + bits[..., 1] * 4 + bits[..., 2] * 2 + bits[..., 3]
            if np.all(np.isin(codes, _FACTORED_VALID_CODES)):
                return "factored_flat"

    if obs.min() >= -1e-3 and obs.max() <= 1.0 + 1e-3:
        return "normalized_flat"

    return "raw_flat"


def _num_tokens_for_representation(flat_dim: int, representation: str, num_grid_states: int) -> int:
    if representation in {"raw_flat", "normalized_flat"}:
        return flat_dim

    if representation == "one_hot_flat":
        if flat_dim % num_grid_states != 0:
            raise ValueError(
                f"Observation dim {flat_dim} is not divisible by num_grid_states={num_grid_states} for one_hot_flat."
            )
        return flat_dim // num_grid_states

    if representation == "factored_flat":
        if flat_dim % FACTORED_BITS != 0:
            raise ValueError(f"Observation dim {flat_dim} is not divisible by {FACTORED_BITS} for factored_flat.")
        return flat_dim // FACTORED_BITS

    raise ValueError(f"Unknown token input representation: {representation}")


class TransformerEncoderBlock(nn.Module):
    d_model: int
    num_heads: int
    mlp_hidden_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=self.dropout_rate,
        )(y, deterministic=deterministic)
        x = x + y

        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_hidden_dim)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.d_model)(y)
        return x + y


class GridTokenTransformerEncoder(nn.Module):
    sequence_length: int
    num_grid_states: int = NUM_GRID_STATES
    token_input_representation: str = "raw_flat"
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_hidden_dim: int = 512
    dropout_rate: float = 0.0
    mask_targets: bool = True

    def _to_tokens(self, flat_inputs: jax.Array) -> jax.Array:
        if self.token_input_representation == "raw_flat":
            tokens = jnp.rint(flat_inputs).astype(jnp.int32)
        elif self.token_input_representation == "normalized_flat":
            tokens = jnp.rint(flat_inputs * (self.num_grid_states - 1)).astype(jnp.int32)
        elif self.token_input_representation == "one_hot_flat":
            reshaped = flat_inputs.reshape(flat_inputs.shape[0], -1, self.num_grid_states)
            tokens = jnp.argmax(reshaped, axis=-1).astype(jnp.int32)
        elif self.token_input_representation == "factored_flat":
            reshaped = flat_inputs.reshape(flat_inputs.shape[0], -1, FACTORED_BITS)
            bits = (reshaped > 0.5).astype(jnp.int32)
            codes = bits[..., 0] * 8 + bits[..., 1] * 4 + bits[..., 2] * 2 + bits[..., 3]
            tokens = _FACTORED_CODE_TO_STATE[codes]
        else:
            raise ValueError(f"Unknown token input representation: {self.token_input_representation}")

        tokens = jnp.clip(tokens, 0, self.num_grid_states - 1)
        if self.mask_targets:
            # Remove target occupancy information from both state and goal encodings.
            tokens = _TARGET_MASKED_STATE[tokens]
        return tokens

    @nn.compact
    def __call__(self, flat_inputs):
        tokens = self._to_tokens(flat_inputs)
        cls_id = self.num_grid_states
        cls_tokens = jnp.full((tokens.shape[0], 1), cls_id, dtype=jnp.int32)
        token_ids = jnp.concatenate([cls_tokens, tokens], axis=1)

        token_embedding = nn.Embed(num_embeddings=self.num_grid_states + 1, features=self.d_model)(token_ids)
        pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.sequence_length, self.d_model),
        )
        x = token_embedding + pos_embedding[:, : token_ids.shape[1], :]

        for layer_idx in range(self.num_layers):
            x = TransformerEncoderBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                dropout_rate=self.dropout_rate,
                name=f"encoder_block_{layer_idx}",
            )(x, deterministic=True)

        return nn.LayerNorm()(x[:, 0, :])


class GCDiscreteBilinearTransformerCritic(nn.Module):
    action_dim: int
    sequence_length: int
    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    ensemble_size: int = 2
    value_exp: bool = True
    num_grid_states: int = NUM_GRID_STATES
    token_input_representation: str = "raw_flat"
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_hidden_dim: int = 512
    dropout_rate: float = 0.0
    mask_targets: bool = True
    net_arch: str = "mlp"

    def setup(self):
        self.shared_encoder = GridTokenTransformerEncoder(
            sequence_length=self.sequence_length,
            num_grid_states=self.num_grid_states,
            token_input_representation=self.token_input_representation,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_hidden_dim=self.mlp_hidden_dim,
            dropout_rate=self.dropout_rate,
            mask_targets=self.mask_targets,
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


class CRLSearchTransformerAgent(struct.PyTreeNode):
    """CRL-search agent with a token-transformer bilinear critic."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def contrastive_loss(self, batch, grad_params, module_name="critic"):
        """Compute the contrastive value loss for the Q or V function."""
        batch_size = batch["observations"].shape[0]
        actions = batch["actions"]

        q, phi, psi = self.network.select(module_name)(
            batch["observations"],
            batch["value_goals"],
            actions=actions,
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 2:  # Non-ensemble.
            phi = phi[None, ...]
            psi = psi[None, ...]
        logits = jnp.einsum("eik,ejk->ije", phi, psi) / jnp.sqrt(phi.shape[-1])
        # logits.shape is (B, B, e) with one term for positive pair and (B - 1) terms for negative pairs in each row.
        identity = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=identity),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        # Compute additional statistics.
        logits = jnp.mean(logits, axis=-1)  # (B, B)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(identity, axis=1)
        logits_pos = jnp.sum(logits * identity) / jnp.sum(identity)
        logits_neg = jnp.sum(logits * (1 - identity)) / jnp.sum(1 - identity)

        # Update target entropy.
        def value_transform(x):
            return jnp.log(jnp.maximum(x, 1e-6))

        action_dim = int(self.config["action_dim"])
        all_actions = jnp.tile(jnp.arange(action_dim), (batch["observations"].shape[0], 1))  # B x A
        qs = jax.lax.stop_gradient(
            jax.vmap(self.network.select("critic"), in_axes=(None, None, 1))(
                batch["observations"],
                batch["value_goals"],
                all_actions,
            )
        )  # A x E x B or A x B
        if len(qs.shape) == 2:  # Non-ensemble.
            qs = qs[:, None, ...]
        qs = qs.mean(axis=1)  # A x B
        qs = qs.transpose(1, 0)  # B x A
        qs = value_transform(qs)

        alpha_temp = self.network.select("alpha_temp")(params=grad_params)
        dist = distrax.Categorical(logits=qs / jnp.maximum(1e-6, alpha_temp))
        entropy = dist.entropy()
        alpha_temp_loss = ((entropy + self.config["target_entropy"]) ** 2).mean()

        total_loss = contrastive_loss + alpha_temp_loss
        return total_loss, {
            "contrastive_loss": contrastive_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
            "binary_accuracy": jnp.mean((logits > 0) == identity),
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logits": logits.mean(),
            "entropy": entropy.mean(),
            "alpha_temp": alpha_temp,
            "entropy_std": dist.entropy().std(),
            "alpha_temp_loss": alpha_temp_loss,
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
        """Returns integer action indices."""
        del temperature
        if not self.config["discrete"]:
            raise NotImplementedError("CRLSearchTransformerAgent.sample_actions supports only discrete action spaces.")

        def value_transform(x):
            return jnp.log(jnp.maximum(x, 1e-6))

        action_dim = int(self.config["action_dim"])
        all_actions = jnp.tile(jnp.arange(action_dim), (observations.shape[0], 1))  # B x A
        qs = jax.lax.stop_gradient(
            jax.vmap(self.network.select("critic"), in_axes=(None, None, 1))(observations, goals, all_actions)
        )  # A x E x B or A x B
        if len(qs.shape) == 2:  # Non-ensemble.
            qs = qs[:, None, ...]
        qs = qs.mean(axis=1)  # A x B
        qs = value_transform(qs)
        qs = qs.transpose(1, 0)  # B x A

        if self.config["action_sampling"] == "softmax":
            alpha_temp = jax.lax.stop_gradient(self.network.select("alpha_temp")())
            dist = distrax.Categorical(logits=qs / jnp.maximum(1e-6, alpha_temp))
            if seed is None:
                actions = jnp.argmax(qs, axis=-1)
            else:
                actions = dist.sample(seed=seed)
        elif self.config["action_sampling"] == "epsilon_greedy":
            greedy_actions = jnp.argmax(qs, axis=-1)
            if seed is None:
                actions = greedy_actions
            else:
                epsilon = self.config.get("epsilon", 0.1)
                rng_actions, rng_uniform = jax.random.split(seed)
                random_actions = jax.random.randint(rng_actions, greedy_actions.shape, 0, action_dim)
                probs = jax.random.uniform(rng_uniform, greedy_actions.shape)
                actions = jnp.where(probs < epsilon, random_actions, greedy_actions)
        else:
            raise ValueError(f"Unknown action sampling type {self.config['action_sampling']}")

        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, ex_goals=None):
        cfg = ml_collections.ConfigDict(dict(config))

        if ex_goals is None:
            ex_goals = ex_observations

        if not _get_bool(cfg, "discrete", True):
            raise ValueError("CRLSearchTransformerAgent supports only discrete action spaces.")

        num_grid_states = _get_int(cfg, "num_grid_states", NUM_GRID_STATES)
        token_representation = _get_str(cfg, "token_input_representation", "auto")
        if token_representation == "auto":
            token_representation = _infer_token_representation(np.asarray(ex_observations), num_grid_states)

        flat_dim = int(ex_observations.shape[-1])
        num_tokens = _num_tokens_for_representation(flat_dim, token_representation, num_grid_states)
        sequence_length = 1 + num_tokens

        action_dim = int(ex_actions.max() + 1)
        cfg["action_dim"] = action_dim

        if cfg.get("target_entropy") is None:
            target_entropy_multiplier = _get_float(cfg, "target_entropy_multiplier", 0.5)
            cfg["target_entropy"] = -target_entropy_multiplier * action_dim / 2

        cfg["token_input_representation"] = token_representation
        cfg["num_grid_states"] = num_grid_states
        cfg["transformer_num_tokens"] = num_tokens

        value_hidden_dims = _get_int_tuple(cfg, "value_hidden_dims", (256, 256))
        actor_hidden_dims = _get_int_tuple(cfg, "actor_hidden_dims", (256, 256))
        net_arch = _get_str(cfg, "net_arch", "mlp")

        d_model = _get_int(cfg, "transformer_d_model", 128)
        num_heads = _get_int(cfg, "transformer_num_heads", 4)
        critic_ensemble_size = _get_int(cfg, "critic_ensemble_size", 2)
        if critic_ensemble_size < 1:
            raise ValueError(f"critic_ensemble_size must be >= 1, got {critic_ensemble_size}.")
        if d_model % num_heads != 0:
            raise ValueError(
                f"transformer_d_model ({d_model}) must be divisible by transformer_num_heads ({num_heads})."
            )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        critic_def = GCDiscreteBilinearTransformerCritic(
            action_dim=action_dim,
            sequence_length=sequence_length,
            hidden_dims=value_hidden_dims,
            latent_dim=_get_int(cfg, "latent_dim", 64),
            layer_norm=_get_bool(cfg, "layer_norm", True),
            ensemble=critic_ensemble_size > 1,
            ensemble_size=critic_ensemble_size,
            value_exp=True,
            num_grid_states=num_grid_states,
            token_input_representation=token_representation,
            d_model=d_model,
            num_layers=_get_int(cfg, "transformer_num_layers", 1),
            num_heads=num_heads,
            mlp_hidden_dim=_get_int(cfg, "transformer_mlp_dim", 2 * d_model),
            dropout_rate=_get_float(cfg, "transformer_dropout", 0.0),
            mask_targets=_get_bool(cfg, "transformer_mask_targets", True),
            net_arch=net_arch,
        )

        actor_def = GCDiscreteActor(
            hidden_dims=actor_hidden_dims,
            action_dim=action_dim,
            gc_encoder=cast(Any, None),
            net_arch=net_arch,
        )
        alpha_temp_def = LogParam()

        network_info = {
            "critic": (critic_def, (ex_observations, ex_goals, ex_actions)),
            "actor": (actor_def, (ex_observations, ex_goals)),
            "alpha_temp": (alpha_temp_def, ()),
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
            agent_name="crl_search_transformer",
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
            token_input_representation="auto",
            transformer_d_model=128,
            transformer_num_layers=1,
            transformer_num_heads=4,
            transformer_mlp_dim=256,
            transformer_dropout=0.0,
            transformer_mask_targets=True,
            critic_ensemble_size=2,
            num_grid_states=NUM_GRID_STATES,
        )
    )
    return config