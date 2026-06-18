import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from impls.utils.encoders import GCEncoder, encoder_modules
from impls.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from impls.utils.networks import (
    GCActor,
    GCDiscreteActor,
    GCDiscreteCritic,
    GCDiscreteUniversalTransformerActor,
    GCDiscreteUniversalTransformerCritic,
    GCMultiDiscreteActor,
    GCMultiDiscreteCritic,
    GCMultiDiscreteUniversalTransformerActor,
    GCMultiDiscreteUniversalTransformerCritic,
    GCUniversalTransformerValue,
    GCValue,
)


class GCIQLAgent(flax.struct.PyTreeNode):
    """Goal-conditioned implicit Q-learning (GCIQL) agent.

    This implementation supports both AWR (actor_loss='awr') and DDPG+BC (actor_loss='ddpgbc') for the actor loss.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def reduce_action_q(self, q):
        if self.config.get('action_mode', 'discrete') == 'multidiscrete' and q.ndim > 1:
            return q.mean(axis=-1)
        return q

    def value_loss(self, batch, grad_params):
        """Compute the IQL value loss."""
        q1, q2 = self.network.select('target_critic')(batch['observations'], batch['value_goals'], batch['actions'])
        q1 = self.reduce_action_q(q1)
        q2 = self.reduce_action_q(q2)
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """Compute the IQL critic loss."""
        next_v = self.network.select('value')(batch['next_observations'], batch['value_goals'])
        if self.config['use_discounted_mc_rewards']:
            q = batch['rewards'] 
        else:
            q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )
        if self.config.get('action_mode', 'discrete') == 'multidiscrete':
            critic_loss = ((q1 - q[:, None]) ** 2 + (q2 - q[:, None]) ** 2).mean()
        else:
            critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss (AWR or DDPG+BC)."""
        if self.config['actor_loss'] == 'awr':
            # AWR loss.
            v = self.network.select('value')(batch['observations'], batch['actor_goals'])
            q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], batch['actions'])
            q1 = self.reduce_action_q(q1)
            q2 = self.reduce_action_q(q2)
            q = jnp.minimum(q1, q2)
            adv = q - v

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(
                batch['observations'],
                batch['actor_goals'],
                action_masks=batch.get('action_masks'),
                params=grad_params,
            )
            log_prob = dist.log_prob(batch['actions'])

            actor_loss = -(exp_a * log_prob).mean()

            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
            }
            if not self.config['discrete']:
                actor_info.update(
                    {
                        'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                        'std': jnp.mean(dist.scale_diag),
                    }
                )

            return actor_loss, actor_info
        elif self.config['actor_loss'] == 'ddpgbc':
            # DDPG+BC loss.
            assert not self.config['discrete']

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], q_actions)
            q = jnp.minimum(q1, q2)

            # Normalize Q values by the absolute mean to make the loss scale invariant.
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch['actions'])

            bc_loss = -(self.config['alpha'] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
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
        """Sample actions from the actor."""
        actor_kwargs = dict(temperature=temperature)
        if self.config['discrete']:
            actor_kwargs['action_masks'] = action_masks
        dist = self.network.select('actor')(observations, goals, **actor_kwargs)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        action_mode = config.get('action_mode', 'discrete')
        if action_mode == 'multidiscrete':
            action_dims = tuple(int(v) for v in config['action_dims'])
            action_dim = max(action_dims)
            num_action_factors = len(action_dims)
        elif config['discrete']:
            action_dims = ()
            action_dim = int(config.get('action_dim') or (ex_actions.max() + 1))
            num_action_factors = 1
        else:
            action_dims = ()
            action_dim = ex_actions.shape[-1]
            num_action_factors = 1

        if config['discrete'] and config['actor_loss'] == 'ddpgbc':
            raise ValueError("GCIQL with discrete actions supports only actor_loss='awr'; ddpgbc is continuous-only.")

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(concat_encoder=encoder_module())
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        actor_arch = config.get('actor_arch', 'mlp')
        critic_arch = config.get('critic_arch') or config['net_arch']
        if critic_arch in {"default", "none", "None"}:
            critic_arch = config['net_arch']
        needs_transformer = actor_arch == 'universal_transformer' or critic_arch == 'universal_transformer'
        if needs_transformer:
            if config['encoder'] is not None:
                raise ValueError("GCIQL universal transformer networks expect flat grid inputs and do not use encoders.")
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

        # Define value and actor networks.
        if critic_arch == 'universal_transformer':
            if not config['discrete']:
                raise ValueError("GCIQL universal transformer critic currently supports only discrete action spaces.")
            value_def = GCUniversalTransformerValue(
                ensemble=False,
                **transformer_kwargs,
            )
        else:
            value_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=False,
                gc_encoder=encoders.get('value'),
                net_arch=critic_arch,
            )

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
        elif config['discrete']:
            critic_def = GCDiscreteCritic(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
                action_dim=action_dim,
                net_arch=critic_arch,
            )
        else:
            critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
                net_arch=critic_arch,
            )

        if config['discrete'] and actor_arch == 'universal_transformer' and action_mode == 'multidiscrete':
            actor_def = GCMultiDiscreteUniversalTransformerActor(
                num_action_factors=num_action_factors,
                action_dim=action_dim,
                action_dims=action_dims,
                action_mask_mode=config.get('action_mask_mode', 'factor'),
                **transformer_kwargs,
            )
        elif config['discrete'] and actor_arch == 'universal_transformer':
            actor_def = GCDiscreteUniversalTransformerActor(
                action_dim=action_dim,
                **transformer_kwargs,
            )
        elif action_mode == 'multidiscrete':
            actor_def = GCMultiDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                num_action_factors=num_action_factors,
                action_dim=action_dim,
                action_dims=action_dims,
                action_mask_mode=config.get('action_mask_mode', 'factor'),
                gc_encoder=encoders.get('actor'),
                net_arch=config['net_arch'],
            )
        elif config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
                net_arch=config['net_arch'],
            )
        else:
            if actor_arch == 'universal_transformer':
                raise ValueError("GCIQL universal transformer actor supports only discrete action spaces.")
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
                net_arch=config['net_arch'],
            )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
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
            agent_name='gciql',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            actor_arch='mlp',
            net_arch='mlp',
            critic_arch='default',
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            actor_loss='ddpgbc',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=0.3,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
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
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
