import copy
from typing import Any

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from impls.utils.encoders import GCEncoder, encoder_modules
from impls.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from impls.utils.networks import LogParam, ensemblize


class TRMRecursiveCell(nn.Module):
    """Shared recursive update cell used by the tiny recursive critic."""

    d_model: int = 256
    num_layers: int = 1
    layer_norm: bool = True
    pre_layer_norm: bool = True

    def setup(self):
        self.pre_ln = nn.LayerNorm()
        self.hidden_layers = [
            nn.Dense(
                self.d_model,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=nn.initializers.zeros,
            )
            for _ in range(max(1, self.num_layers))
        ]
        self.out_proj = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros,
        )
        self.post_ln = nn.LayerNorm()

    def __call__(self, hidden, context, info=False):
        x = jnp.concatenate([hidden, context], axis=-1)
        if self.pre_layer_norm:
            x = self.pre_ln(x)

        layer_h_norms = [] if info else None
        for dense in self.hidden_layers:
            x = nn.gelu(dense(x))
            if info:
                layer_h_norms.append(jnp.linalg.norm(x, axis=-1).mean())

        delta = self.out_proj(x)
        new_hidden = hidden + delta
        if self.layer_norm:
            new_hidden = self.post_ln(new_hidden)

        if not info:
            return new_hidden

        return new_hidden, jnp.stack(layer_h_norms)


class TRMThinkingCritic(nn.Module):
    """Tiny recursive-model critic for goal-conditioned discrete actions."""

    d_model: int = 256
    thinking_steps: int = 3
    ensemble: bool = True
    gc_encoder: nn.Module = None
    layer_norm: bool = True
    pre_layer_norm: bool = True
    num_layers: int = 1

    def setup(self):
        self.input_embed_layer = nn.Dense(self.d_model)
        self.input_ln = nn.LayerNorm()
        self.start_token = self.param(
            'start_token',
            nn.initializers.normal(stddev=0.02),
            (1, self.d_model),
        )

        self.recursive_cell = TRMRecursiveCell(
            d_model=self.d_model,
            num_layers=self.num_layers,
            layer_norm=self.layer_norm,
            pre_layer_norm=self.pre_layer_norm,
        )

        self.value_head = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, observations, goals=None, actions=None, info=False):
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)

        if actions is not None:
            inputs = jnp.concatenate([inputs, actions], axis=-1)

        context = self.input_embed_layer(inputs)
        if self.pre_layer_norm:
            context = self.input_ln(context)

        hidden = context + jnp.broadcast_to(self.start_token, context.shape)

        h_seq = [] if info else None
        per_layer_norms = [] if info else None
        for _ in range(self.thinking_steps):
            if info:
                hidden, layer_norms = self.recursive_cell(hidden, context, info=True)
                h_seq.append(hidden)
                per_layer_norms.append(layer_norms)
            else:
                hidden = self.recursive_cell(hidden, context, info=False)

        q_value = self.value_head(hidden).squeeze(-1)

        if not info:
            return q_value

        h_stack = jnp.stack(h_seq, axis=1)
        q_seq = self.value_head(h_stack).squeeze(-1)
        h_norm_per_step = jnp.linalg.norm(h_stack, axis=-1).mean(axis=0)
        layer_norm_means = jnp.stack(per_layer_norms, axis=0).mean(axis=0)

        diag = {}
        for k in range(self.thinking_steps):
            q_k = q_seq[:, k]
            diag[f'q_step_{k}'] = q_k.mean()
            diag[f'q_step_{k}_std'] = q_k.std()
            diag[f'h_norm_step_{k}'] = h_norm_per_step[k]

        if self.thinking_steps > 1:
            q_deltas = q_seq[:, 1:] - q_seq[:, :-1]
            for k in range(1, self.thinking_steps):
                dq = q_deltas[:, k - 1]
                diag[f'q_delta_{k}'] = dq.mean()
                diag[f'q_delta_{k}_abs'] = jnp.abs(dq).mean()

        for layer_idx, h_norm in enumerate(layer_norm_means):
            diag[f'layer{layer_idx}_h_norm_mean'] = h_norm

        diag['final_h_norm'] = jnp.linalg.norm(hidden, axis=-1).mean()

        return q_value, diag


class GCTRMDiscreteCritic(nn.Module):
    """Goal-conditioned tiny recursive critic for discrete actions."""

    d_model: int = 256
    action_dim: int = 6
    thinking_steps: int = 3
    ensemble: bool = True
    gc_encoder: nn.Module = None
    layer_norm: bool = True
    pre_layer_norm: bool = True
    num_layers: int = 1

    def setup(self):
        critic_cls = TRMThinkingCritic
        if self.ensemble:
            critic_cls = ensemblize(critic_cls, 2)

        self.critic = critic_cls(
            d_model=self.d_model,
            thinking_steps=self.thinking_steps,
            ensemble=False,
            gc_encoder=self.gc_encoder,
            layer_norm=self.layer_norm,
            pre_layer_norm=self.pre_layer_norm,
            num_layers=self.num_layers,
        )

    def __call__(self, observations, goals=None, actions=None, info=False):
        actions_onehot = jnp.eye(self.action_dim)[actions]
        q = self.critic(observations, goals, actions_onehot, info)
        return q


class GCDQNTRMAgent(flax.struct.PyTreeNode):
    """Goal-conditioned DQN with tiny recursive model thinking steps."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _zero_diag_stats(self):
        diag = {}
        zero = jnp.array(0.0, dtype=jnp.float32)
        for k in range(self.config['thinking_steps']):
            diag[f'q_step_{k}'] = zero
            diag[f'q_step_{k}_std'] = zero
            diag[f'h_norm_step_{k}'] = zero

        if self.config['thinking_steps'] > 1:
            for k in range(1, self.config['thinking_steps']):
                diag[f'q_delta_{k}'] = zero
                diag[f'q_delta_{k}_abs'] = zero

        num_layers = self.config.get('num_layers', 1)
        for layer_idx in range(num_layers):
            diag[f'layer{layer_idx}_h_norm_mean'] = zero

        diag['final_h_norm'] = zero
        return diag

    def critic_loss(self, batch, grad_params):
        action_dim = 6

        q1_a, q2_a = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )
        q1_a = jnp.squeeze(q1_a, axis=-1) if q1_a.ndim > 1 else q1_a
        q2_a = jnp.squeeze(q2_a, axis=-1) if q2_a.ndim > 1 else q2_a

        all_actions = jnp.tile(jnp.arange(action_dim), (batch['next_observations'].shape[0], 1))
        qs = jax.lax.stop_gradient(
            jax.vmap(self.network.select('target_critic'), in_axes=(None, None, 1))(
                batch['next_observations'], batch['value_goals'], all_actions
            )
        )
        qs = qs.mean(axis=1)
        qs = qs.transpose(1, 0)
        max_next_q = jnp.max(qs, axis=-1)

        if self.config['use_discounted_mc_rewards']:
            target = batch['rewards']
        else:
            target = batch['rewards'] + self.config['discount'] * batch['masks'] * max_next_q

        critic_loss = ((q1_a - target) ** 2 + (q2_a - target) ** 2).mean()

        all_actions = jnp.tile(jnp.arange(action_dim), (batch['observations'].shape[0], 1))
        current_qs = jax.lax.stop_gradient(
            jax.vmap(self.network.select('critic'), in_axes=(None, None, 1))(
                batch['observations'],
                batch['value_goals'],
                all_actions,
            )
        )
        if len(current_qs.shape) == 2:
            current_qs = current_qs[:, None, ...]
        current_qs = current_qs.mean(axis=1)
        current_qs = current_qs.transpose(1, 0)

        alpha_temp = self.network.select('alpha_temp')(params=grad_params)
        dist = distrax.Categorical(logits=current_qs / jnp.maximum(1e-6, alpha_temp))
        entropy = dist.entropy()
        alpha_temp_loss = ((entropy + self.config['target_entropy']) ** 2).mean()

        total_loss = critic_loss + alpha_temp_loss

        diag_interval = self.config.get('log_diagnostics_every', 0)
        if diag_interval > 0:
            should_log_diag = (self.network.step % diag_interval) == 0

            def _compute_diag(_):
                diag_out = jax.lax.stop_gradient(
                    self.network.select('critic')(
                        batch['observations'],
                        batch['value_goals'],
                        batch['actions'],
                        info=True,
                        params=grad_params,
                    )
                )
                return diag_out[1]

            def _skip_diag(_):
                return self._zero_diag_stats()

            diag_dict = jax.lax.cond(should_log_diag, _compute_diag, _skip_diag, operand=None)
        else:
            diag_dict = self._zero_diag_stats()

        info_dict = {
            'critic_loss': critic_loss,
            'q_mean': target.mean(),
            'q_max': target.max(),
            'q_min': target.min(),
            'q_std': target.std(),
            'entropy': entropy.mean(),
            'alpha_temp': alpha_temp,
            'entropy_std': entropy.std(),
            'alpha_temp_loss': alpha_temp_loss,
        }

        for k, v in diag_dict.items():
            info_dict[k] = v.mean() if hasattr(v, 'mean') else v

        return total_loss, info_dict

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        loss = critic_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(
            loss_fn=loss_fn, compute_grad_stats=self.config.get('log_grad_stats', True)
        )
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        if not self.config['discrete']:
            raise NotImplementedError("GCDQNTRMAgent supports only discrete action spaces.")

        action_dim = 6

        all_actions = jnp.tile(jnp.arange(action_dim), (observations.shape[0], 1))
        qs = jax.lax.stop_gradient(
            jax.vmap(self.network.select('critic'), in_axes=(None, None, 1))(
                observations, goals, all_actions
            )
        )
        qs = qs.mean(axis=1)
        qs = qs.transpose(1, 0)

        if self.config['action_sampling'] == 'softmax':
            alpha_temp = jax.lax.stop_gradient(self.network.select('alpha_temp')())
            dist = distrax.Categorical(logits=qs / jnp.maximum(1e-6, alpha_temp))
            actions = dist.sample(seed=seed)
        elif self.config['action_sampling'] == 'epsilon_greedy':
            greedy_actions = jnp.argmax(qs, axis=-1)
            rng, rng_uniform = jax.random.split(seed)
            random_actions = jax.random.randint(rng, greedy_actions.shape, 0, action_dim)
            probs = jax.random.uniform(rng_uniform, greedy_actions.shape)
            actions = jnp.where(probs < self.config.get('epsilon', 0.1), random_actions, greedy_actions)
        elif self.config['action_sampling'] == 'greedy':
            actions = jnp.argmax(qs, axis=-1)
        else:
            raise ValueError(f"Unknown action sampling type: {self.config['action_sampling']}")

        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        if not config['discrete']:
            raise ValueError("GCDQNTRMAgent supports only discrete action spaces.")

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        action_dim = int(ex_actions.max() + 1)

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())

        critic_def = GCTRMDiscreteCritic(
            d_model=config['trm_hidden_size'],
            action_dim=action_dim,
            thinking_steps=config['thinking_steps'],
            ensemble=config['ensemble'],
            gc_encoder=encoders.get('critic'),
            layer_norm=config['layer_norm'],
            pre_layer_norm=config['pre_layer_norm'],
            num_layers=config['num_layers'],
        )

        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim / 2
        alpha_temp_def = LogParam()

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            alpha_temp=(alpha_temp_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='gcdqn_trm',
            lr=3e-4,
            batch_size=1024,
            trm_hidden_size=64,
            thinking_steps=3,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            pre_layer_norm=True,
            net_arch='mlp',
            discount=0.99,
            tau=0.005,
            target_entropy=None,
            target_entropy_multiplier=0.5,
            use_discounted_mc_rewards=False,
            action_sampling='softmax',
            epsilon=0.1,
            action_dim=6,
            log_diagnostics_every=100,
            log_grad_stats=False,
            expectile=0.9,
            actor_loss='ddpgbc',
            alpha=0.3,
            const_std=True,
            discrete=True,
            encoder=ml_collections.config_dict.placeholder(str),
            dataset_class='GCDataset',
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
            frame_stack=ml_collections.config_dict.placeholder(int),
            ensemble=True,
            num_layers=1,
        )
    )
    return config