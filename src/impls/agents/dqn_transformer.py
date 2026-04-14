import copy
from typing import Any

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


class GridTokenTransformerQ(nn.Module):
    action_dim: int
    sequence_length: int
    num_grid_states: int = NUM_GRID_STATES
    token_input_representation: str = "raw_flat"
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_hidden_dim: int = 512
    dropout_rate: float = 0.0
    use_goals: bool = True

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

        return jnp.clip(tokens, 0, self.num_grid_states - 1)

    @nn.compact
    def __call__(self, observations, goals=None):
        obs_tokens = self._to_tokens(observations)

        cls_id = 2 * self.num_grid_states
        cls_tokens = jnp.full((obs_tokens.shape[0], 1), cls_id, dtype=jnp.int32)

        if self.use_goals:
            if goals is None:
                goal_tokens = jnp.zeros_like(obs_tokens)
            else:
                goal_tokens = self._to_tokens(goals)
            goal_tokens = goal_tokens + self.num_grid_states
            token_ids = jnp.concatenate([cls_tokens, obs_tokens, goal_tokens], axis=1)
        else:
            token_ids = jnp.concatenate([cls_tokens, obs_tokens], axis=1)

        token_embedding = nn.Embed(num_embeddings=2 * self.num_grid_states + 1, features=self.d_model)(token_ids)
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

        cls_repr = nn.LayerNorm()(x[:, 0, :])
        q_values = nn.Dense(self.action_dim)(cls_repr)
        return q_values


class TransformerDiscreteCritic(nn.Module):
    action_dim: int
    sequence_length: int
    num_grid_states: int = NUM_GRID_STATES
    token_input_representation: str = "raw_flat"
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_hidden_dim: int = 512
    dropout_rate: float = 0.0
    use_goals: bool = True
    ensemble: bool = True
    ensemble_size: int = 2

    def setup(self):
        q_module = GridTokenTransformerQ
        if self.ensemble and self.ensemble_size > 1:
            q_module = ensemblize(q_module, self.ensemble_size)

        self.q_net = q_module(
            action_dim=self.action_dim,
            sequence_length=self.sequence_length,
            num_grid_states=self.num_grid_states,
            token_input_representation=self.token_input_representation,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_hidden_dim=self.mlp_hidden_dim,
            dropout_rate=self.dropout_rate,
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


class GCDQNTransformerAgent(struct.PyTreeNode):
    """Goal-conditioned DQN agent with a transformer critic over flattened grid tokens."""

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

        # Use a lean update path to avoid expensive grad-tree diagnostics every step.
        grads, info = jax.grad(loss_fn, has_aux=True)(self.network.params)
        new_network = self.network.apply_gradients(grads=grads)

        # Keep target update purely functional inside jit to avoid Python-side mutation overhead.
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
            raise ValueError("GCDQNTransformerAgent supports only discrete action spaces.")

        num_grid_states = _get_int(cfg, "num_grid_states", NUM_GRID_STATES)
        token_representation = _get_str(cfg, "token_input_representation", "auto")
        if token_representation == "auto":
            token_representation = _infer_token_representation(np.asarray(ex_observations), num_grid_states)

        flat_dim = int(ex_observations.shape[-1])
        num_tokens = _num_tokens_for_representation(flat_dim, token_representation, num_grid_states)
        use_goals = _get_bool(cfg, "transformer_use_goals", True)
        sequence_length = 1 + num_tokens + (num_tokens if use_goals else 0)

        action_dim = int(ex_actions.max() + 1)
        if cfg.get("target_entropy") is None:
            target_entropy_multiplier = _get_float(cfg, "target_entropy_multiplier", 0.5)
            cfg["target_entropy"] = -target_entropy_multiplier * action_dim / 2

        cfg["token_input_representation"] = token_representation
        cfg["num_grid_states"] = num_grid_states
        cfg["transformer_num_tokens"] = num_tokens

        d_model = _get_int(cfg, "transformer_d_model", 256)
        num_heads = _get_int(cfg, "transformer_num_heads", 8)
        critic_ensemble_size = _get_int(cfg, "critic_ensemble_size", 2)
        if critic_ensemble_size < 1:
            raise ValueError(f"critic_ensemble_size must be >= 1, got {critic_ensemble_size}.")
        if d_model % num_heads != 0:
            raise ValueError(
                f"transformer_d_model ({d_model}) must be divisible by transformer_num_heads ({num_heads})."
            )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        critic_def = TransformerDiscreteCritic(
            action_dim=action_dim,
            sequence_length=sequence_length,
            num_grid_states=num_grid_states,
            token_input_representation=token_representation,
            d_model=d_model,
            num_layers=_get_int(cfg, "transformer_num_layers", 4),
            num_heads=num_heads,
            mlp_hidden_dim=_get_int(cfg, "transformer_mlp_dim", 4 * d_model),
            dropout_rate=_get_float(cfg, "transformer_dropout", 0.0),
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

        params = network_params
        params["modules_target_critic"] = params["modules_critic"]

        return cls(rng, network=network, config=core.freeze(dict(cfg)))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="gcdqn_transformer",
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
            token_input_representation="auto",
            transformer_use_goals=True,
            transformer_d_model=128,
            transformer_num_layers=1,
            transformer_num_heads=4,
            transformer_mlp_dim=256,
            transformer_dropout=0.0,
            critic_ensemble_size=2,
            num_grid_states=NUM_GRID_STATES,
        )
    )
    return config
