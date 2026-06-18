import copy
from typing import Any

import distrax
import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from impls.utils.encoders import GCEncoder, encoder_modules
from impls.utils.action_utils import (
    all_actions,
    discrete_action_dim,
    factor_action_mask,
    mask_factor_logits,
    mask_joint_logits,
    mask_logits,
    multidiscrete_action_dims,
    multidiscrete_joint_q,
    sample_uniform_actions,
    unravel_multidiscrete,
)
from impls.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from impls.utils.networks import (
    GCActor,
    GCDiscreteActor,
    GCDiscreteCritic,
    GCDiscreteUniversalTransformerCritic,
    GCMultiDiscreteCritic,
    GCMultiDiscreteUniversalTransformerCritic,
    GCValue,
    LogParam,
)


class GCDQNAgent(flax.struct.PyTreeNode):
    """Goal-conditioned DQN (discrete actions only).

    Minimal changes from the GCIQL implementation: re-uses critic networks (two-head ensemble),
    trains critic with TD targets computed from the target critic and uses greedy action selection.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params):
        if self.config.get('action_mode', 'discrete') == 'multidiscrete':
            return self.multidiscrete_critic_loss(batch, grad_params)
        return self.scalar_critic_loss(batch, grad_params)

    def scalar_critic_loss(self, batch, grad_params):
        """Compute the DQN critic loss (discrete actions).

        Assumes:
         - batch['actions'] contains integer action indices.
         - critic when passed actions returns Q(s,a) scalar(s); when called without actions returns
           Q(s, a) vectors (this mirrors the original critic init/signature).
        """
        # Current Q for taken actions (may be scalars if critic uses actions input)
        q1_a, q2_a = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )
        # Ensure shapes are (batch,)
        q1_a = jnp.squeeze(q1_a, axis=-1) if q1_a.ndim > 1 else q1_a
        q2_a = jnp.squeeze(q2_a, axis=-1) if q2_a.ndim > 1 else q2_a

        # Target Q: use target critic to get Q-vector for next states, average ensemble, take max over actions
        action_dim = discrete_action_dim(self.config)
        target_actions = all_actions(batch['next_observations'].shape[0], action_dim)
        qs = jax.lax.stop_gradient(jax.vmap(self.network.select('target_critic'), in_axes=(None, None, 1))(batch['next_observations'], batch['value_goals'], target_actions)) # A x 2 x B
        qs = qs.mean(axis=1) # A x B
        qs = mask_logits(qs.transpose(1, 0), batch.get('next_action_masks')) # B x A
        # q1_next, q2_next expected shape: (batch, action_dim)
        max_next_q = jnp.max(qs, axis=-1)

        # TD or MC target
        if self.config['use_discounted_mc_rewards']:
            target = batch['rewards'] 
        else:
            target = batch['rewards'] + self.config['discount'] * batch['masks'] * max_next_q

        # MSE loss on both heads (keeps two-head training similar to your critic ensemble)
        critic_loss = ((q1_a - target) ** 2 + (q2_a - target) ** 2).mean()

        # Update target entropy
        current_actions = all_actions(batch['observations'].shape[0], action_dim)
        qs = jax.lax.stop_gradient(
            jax.vmap(self.network.select("critic"), in_axes=(None, None, 1))(batch['observations'], jnp.roll(batch['next_observations'], shift=1, axis=0), current_actions)
        )  # A x 2 x B
        if len(qs.shape) == 2:  # Non-ensemble.
            qs = qs[:, None, ...]
        qs = qs.mean(axis=1)  # A x B
        qs = mask_logits(qs.transpose(1, 0), batch.get('action_masks')) # B x A

        alpha_temp = self.network.select('alpha_temp')(params=grad_params)
        dist = distrax.Categorical(logits=qs / jnp.maximum(1e-6, alpha_temp))
        entropy = dist.entropy()
        alpha_temp_loss = ((entropy + self.config['target_entropy'])**2).mean()  # Target entropy is a negative constant like -log(6)

        total_loss = critic_loss +  alpha_temp_loss

        return total_loss, {
            'critic_loss': critic_loss,
            'q_mean': target.mean(),
            'q_max': target.max(),
            'q_min': target.min(),
            'q.std': target.std(),
            'entropy': entropy.mean(),
            'alpha_temp': alpha_temp,
            'entropy_std': dist.entropy().std(),
            'alpha_temp_loss': alpha_temp_loss,
        }

    def multidiscrete_critic_loss(self, batch, grad_params):
        action_dims = multidiscrete_action_dims(self.config)
        q_values = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )

        target_qs = jax.lax.stop_gradient(
            self.network.select('target_critic')(batch['next_observations'], batch['value_goals'])
        )
        target_qs = target_qs.mean(axis=0)
        max_next_q = self._multidiscrete_max_q(target_qs, batch.get('next_action_masks'), action_dims)

        if self.config['use_discounted_mc_rewards']:
            target = batch['rewards']
        else:
            target = batch['rewards'] + self.config['discount'] * batch['masks'] * max_next_q

        critic_loss = ((q_values - target[None, :, None]) ** 2).mean()

        current_qs = jax.lax.stop_gradient(
            self.network.select("critic")(
                batch['observations'],
                jnp.roll(batch['next_observations'], shift=1, axis=0),
            )
        )
        current_qs = current_qs.mean(axis=0)
        alpha_temp = self.network.select('alpha_temp')(params=grad_params)
        entropy = self._multidiscrete_entropy(current_qs, batch.get('action_masks'), action_dims, alpha_temp)
        alpha_temp_loss = ((entropy + self.config['target_entropy']) ** 2).mean()

        total_loss = critic_loss + alpha_temp_loss
        return total_loss, {
            'critic_loss': critic_loss,
            'q_mean': target.mean(),
            'q_max': target.max(),
            'q_min': target.min(),
            'q.std': target.std(),
            'entropy': entropy.mean(),
            'alpha_temp': alpha_temp,
            'entropy_std': entropy.std(),
            'alpha_temp_loss': alpha_temp_loss,
        }

    def _multidiscrete_max_q(self, qs, action_masks, action_dims):
        if self.config.get('action_mask_mode', 'factor') == 'joint':
            joint_q = multidiscrete_joint_q(qs, action_dims)
            joint_q = mask_joint_logits(joint_q, action_masks)
            return jnp.max(joint_q.reshape((joint_q.shape[0], -1)), axis=-1)

        qs = mask_factor_logits(qs, action_masks, action_dims)
        return jnp.max(qs, axis=-1).mean(axis=-1)

    def _multidiscrete_entropy(self, qs, action_masks, action_dims, alpha_temp):
        temperature = jnp.maximum(1e-6, alpha_temp)
        if self.config.get('action_mask_mode', 'factor') == 'joint':
            joint_q = multidiscrete_joint_q(qs, action_dims)
            joint_q = mask_joint_logits(joint_q, action_masks)
            dist = distrax.Categorical(logits=joint_q.reshape((joint_q.shape[0], -1)) / temperature)
            return dist.entropy()

        qs = mask_factor_logits(qs, action_masks, action_dims)
        dist = distrax.Categorical(logits=qs / temperature)
        return dist.entropy().mean(axis=-1)

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss (only critic loss for DQN)."""
        info = {}
        # Only critic is trained for DQN (no value / actor losses)
        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        loss = critic_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
        action_masks=None,
    ):
        """
        Returns integer action indices. Continuous actions are not supported here.
        """
        if not self.config['discrete']:
            raise NotImplementedError("ClearnSearchAgent.sample_actions supports only discrete action spaces.")

        if self.config.get('action_mode', 'discrete') == 'multidiscrete':
            return self.sample_multidiscrete_actions(observations, goals, seed, action_masks)
        
        action_dim = discrete_action_dim(self.config)
        candidate_actions = all_actions(observations.shape[0], action_dim)
        qs = jax.lax.stop_gradient(jax.vmap(self.network.select('critic'), in_axes=(None, None, 1))(observations, goals, candidate_actions)) # A x 2 x B
        qs = qs.mean(axis=1) # A x B
        qs = mask_logits(qs.transpose(1, 0), action_masks) # B x A

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
            random_actions = sample_uniform_actions(rng, action_masks, greedy_actions.shape, action_dim)

            # ε-greedy: pick random with prob ε, else greedy
            probs = jax.random.uniform(rng_uniform, greedy_actions.shape)
            actions = jnp.where(probs < 0.1, random_actions, greedy_actions)
        else:
            raise ValueError(f"Unknown action sampling type {self.config['action_sampling']}")

        return actions

    def sample_multidiscrete_actions(self, observations, goals, seed, action_masks):
        action_dims = multidiscrete_action_dims(self.config)
        qs = jax.lax.stop_gradient(self.network.select('critic')(observations, goals))
        qs = qs.mean(axis=0)

        if self.config.get('action_mask_mode', 'factor') == 'joint':
            return self._sample_joint_masked_multidiscrete(qs, action_masks, seed, action_dims)
        return self._sample_factor_masked_multidiscrete(qs, action_masks, seed, action_dims)

    def _sample_factor_masked_multidiscrete(self, qs, action_masks, seed, action_dims):
        qs = mask_factor_logits(qs, action_masks, action_dims)
        if self.config['action_sampling'] == 'softmax':
            alpha_temp = jax.lax.stop_gradient(self.network.select('alpha_temp')())
            dist = distrax.Categorical(logits=qs / jnp.maximum(1e-6, alpha_temp))
            return dist.sample(seed=seed).astype(jnp.int32)

        if self.config['action_sampling'] == 'epsilon_greedy':
            greedy_actions = jnp.argmax(qs, axis=-1).astype(jnp.int32)
            rng, rng_uniform = jax.random.split(seed)
            valid_masks = factor_action_mask(action_masks, qs.shape[0], action_dims)
            random_actions = jax.random.categorical(
                rng,
                mask_logits(jnp.zeros_like(valid_masks, dtype=jnp.float32), valid_masks),
            ).astype(jnp.int32)
            probs = jax.random.uniform(rng_uniform, greedy_actions.shape)
            return jnp.where(probs < 0.1, random_actions, greedy_actions)

        raise ValueError(f"Unknown action sampling type {self.config['action_sampling']}")

    def _sample_joint_masked_multidiscrete(self, qs, action_masks, seed, action_dims):
        joint_q = multidiscrete_joint_q(qs, action_dims)
        joint_q = mask_joint_logits(joint_q, action_masks)
        flat_q = joint_q.reshape((joint_q.shape[0], -1))

        if self.config['action_sampling'] == 'softmax':
            alpha_temp = jax.lax.stop_gradient(self.network.select('alpha_temp')())
            dist = distrax.Categorical(logits=flat_q / jnp.maximum(1e-6, alpha_temp))
            flat_actions = dist.sample(seed=seed)
            return unravel_multidiscrete(flat_actions, action_dims)

        if self.config['action_sampling'] == 'epsilon_greedy':
            greedy_actions = unravel_multidiscrete(jnp.argmax(flat_q, axis=-1), action_dims)
            rng, rng_uniform = jax.random.split(seed)
            random_logits = mask_joint_logits(jnp.zeros_like(joint_q), action_masks).reshape((joint_q.shape[0], -1))
            random_actions = unravel_multidiscrete(
                jax.random.categorical(rng, random_logits).astype(jnp.int32),
                action_dims,
            )
            probs = jax.random.uniform(rng_uniform, (greedy_actions.shape[0], 1))
            return jnp.where(probs < 0.1, random_actions, greedy_actions)

        raise ValueError(f"Unknown action sampling type {self.config['action_sampling']}")

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new DQN agent (discrete only)."""
        if not config['discrete']:
            raise ValueError("GCDQNAgent currently supports only discrete action spaces. Set config['discrete']=True.")

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        action_mode = config.get('action_mode', 'discrete')
        if action_mode == 'multidiscrete':
            action_dims = tuple(int(v) for v in config['action_dims'])
            action_dim = max(action_dims)
            num_action_factors = len(action_dims)
        else:
            action_dim = int(config.get('action_dim') or (ex_actions.max() + 1))
            num_action_factors = 1

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())

        critic_arch = config.get('critic_arch') or config['net_arch']
        if critic_arch in {"default", "none", "None"}:
            critic_arch = config['net_arch']
        if critic_arch == 'universal_transformer':
            if config['encoder'] is not None:
                raise ValueError("DQN universal transformer critic expects flat grid inputs and does not use encoders.")
            num_heads = int(config['transformer_num_heads'])
            d_model = int(config['transformer_d_model'])
            if d_model % num_heads != 0:
                raise ValueError(
                    f"transformer_d_model ({d_model}) must be divisible by transformer_num_heads ({num_heads})."
                )
            transformer_kwargs = dict(
                cell_dim=int(config['transformer_cell_dim']),
                d_model=d_model,
                num_heads=num_heads,
                thinking_steps=int(config['transformer_thinking_steps']),
                mlp_dim=int(config['transformer_mlp_dim']),
                pool=config['transformer_pool'],
                token_mode=config['transformer_token_mode'],
                token_subgrid=int(config['transformer_token_subgrid']),
            )
        else:
            transformer_kwargs = None

        # For DQN we only need a discrete critic (we keep other modules for compatibility/minimal changes).
        if critic_arch == 'universal_transformer' and action_mode == 'multidiscrete':
            critic_def = GCMultiDiscreteUniversalTransformerCritic(
                ensemble=True,
                num_action_factors=num_action_factors,
                action_dim=action_dim,
                **transformer_kwargs,
            )
        elif critic_arch == 'universal_transformer':
            critic_def = GCDiscreteUniversalTransformerCritic(
                ensemble=True,
                action_dim=action_dim,
                **transformer_kwargs,
            )
        elif action_mode == 'multidiscrete':
            critic_def = GCMultiDiscreteCritic(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
                num_action_factors=num_action_factors,
                action_dim=action_dim,
                net_arch=critic_arch,
            )
        else:
            critic_def = GCDiscreteCritic(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
                action_dim=action_dim,
                net_arch=critic_arch,
            )

        # Keep dummy value/actor defs to minimize code changes (they won't be used in training).
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=None,
            net_arch=config['net_arch'],
        )
        actor_def = GCDiscreteActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            gc_encoder=None,
            net_arch=config['net_arch'],
        )
        
        if config['target_entropy'] is None:
            if action_mode == 'multidiscrete':
                config['target_entropy'] = -config['target_entropy_multiplier'] * sum(config['action_dims']) / 2
            else:
                config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim/2
        alpha_temp_def = LogParam()

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
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
            # Agent hyperparameters.
            agent_name='gcdqn',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            net_arch='mlp',
            critic_arch='default',
            discount=0.99,
            tau=0.005,
            # legacy / unused fields from IQL left for compatibility:
            expectile=0.9,
            actor_loss='ddpgbc',
            alpha=0.3,
            const_std=True,
            discrete=True,  # DQN requires discrete actions
            encoder=ml_collections.config_dict.placeholder(str),
            action_sampling='softmax',
            action_dim=0,
            action_mode='discrete',
            action_dims=(),
            num_action_factors=1,
            action_mask_mode='categorical',
            transformer_cell_dim=12,
            transformer_d_model=128,
            transformer_num_heads=4,
            transformer_thinking_steps=1,
            transformer_mlp_dim=256,
            transformer_pool='cls',
            transformer_token_mode='paired',
            transformer_token_subgrid=1,
            # Dataset hyperparameters.
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
        )
    )
    return config
