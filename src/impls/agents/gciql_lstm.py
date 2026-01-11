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
from impls.utils.networks import default_init, ensemblize


class LSTMThinkingValue(nn.Module):
    """LSTM-based value network with thinking steps for goal-conditioned RL.
    
    This module uses an LSTM to perform iterative "thinking" over the input,
    allowing the model to refine its value estimates through multiple processing steps.
    
    Attributes:
        d_model: Hidden dimension size for LSTM and embeddings.
        thinking_steps: Number of thinking iterations to perform.
        gc_encoder: Optional goal-conditioned encoder.
        layer_norm: Whether to apply layer normalization.
        num_layers: Number of LSTM layers.
    """
    
    d_model: int = 64
    thinking_steps: int = 3
    gc_encoder: nn.Module = None
    layer_norm: bool = True
    num_layers: int = 2
    
    def setup(self):
        # Input embedding layer
        self.input_embed_layer = nn.Dense(self.d_model)
        
        # Start token (learnable)
        self.start_token = self.param(
            'start_token',
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.d_model)
        )
        
        # Core LSTM layers
        self.lstm_layers = [
            nn.RNN(nn.LSTMCell(features=self.d_model)) 
            for _ in range(self.num_layers)
        ]
        
        # Output layers
        self.final_ln = nn.LayerNorm()
        self.value_head = nn.Dense(1, kernel_init=default_init())
    
    def __call__(self, observations, goals=None):
        """Forward pass with thinking steps.
        
        Args:
            observations: Observation tensor of shape (B, obs_dim)
            goals: Goal tensor of shape (B, goal_dim) (optional)
            
        Returns:
            Value: V(s) for the given observation, shape (B,)
        """
        # Encode inputs
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        
        input_shape = inputs.shape
        x_flat = inputs.reshape(-1, input_shape[-1])
        
        # Embed input: (B, d_model) -> (B, 1, d_model)
        x_emb = self.input_embed_layer(x_flat)[:, None, :]
        
        # Create sequence with start token for first step
        x_first = x_emb + self.start_token
        if self.thinking_steps > 1:
            x_rest = jnp.tile(x_emb, (1, self.thinking_steps - 1, 1))
            x_seq = jnp.concatenate([x_first, x_rest], axis=1)
        else:
            x_seq = x_first

        lstm_out = x_seq
        for lstm_layer in self.lstm_layers:
            lstm_out = lstm_layer(lstm_out)
        
        # Take last timestep output: (B, d_model)
        final_out = lstm_out[:, -1, :]
        
        # Apply layer norm and get value
        out = self.final_ln(final_out)
        value = self.value_head(out)
        value = value.reshape(input_shape[:-1] + (-1,))
        
        return jnp.squeeze(value, axis=-1)


class LSTMThinkingCritic(nn.Module):
    """LSTM-based critic with thinking steps for goal-conditioned discrete actions.
    
    This module uses an LSTM to perform iterative "thinking" over the input,
    allowing the model to refine its Q-value estimates through multiple processing steps.
    Takes action as input (concatenated with observations/goals) and outputs single Q-value.
    
    Attributes:
        d_model: Hidden dimension size for LSTM and embeddings.
        thinking_steps: Number of thinking iterations to perform.
        gc_encoder: Optional goal-conditioned encoder.
        layer_norm: Whether to apply layer normalization.
        num_layers: Number of LSTM layers.
    """
    
    d_model: int = 64
    thinking_steps: int = 3
    gc_encoder: nn.Module = None
    layer_norm: bool = True
    num_layers: int = 2
    
    def setup(self):
        # Input embedding layer
        self.input_embed_layer = nn.Dense(self.d_model)
        
        # Start token (learnable)
        self.start_token = self.param(
            'start_token',
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.d_model)
        )
        
        # Core LSTM layers
        self.lstm_layers = [
            nn.RNN(nn.LSTMCell(features=self.d_model)) 
            for _ in range(self.num_layers)
        ]
        
        # Output layers
        self.final_ln = nn.LayerNorm()
        self.q_head = nn.Dense(1, kernel_init=default_init())
    
    def __call__(self, observations, goals=None, actions=None):
        """Forward pass with thinking steps.
        
        Args:
            observations: Observation tensor of shape (B, obs_dim)
            goals: Goal tensor of shape (B, goal_dim) (optional)
            actions: One-hot encoded actions of shape (B, action_dim)
            
        Returns:
            Q-values: Q(s,a) for the given action, shape (B,)
        """
        # Encode inputs
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        
        # Concatenate action (one-hot encoded)
        if actions is not None:
            inputs = jnp.concatenate([inputs, actions], axis=-1)
        
        # Embed input: (B, d_model) -> (B, 1, d_model)
        x_emb = self.input_embed_layer(inputs)[:, None, :]
        
        # Create sequence with start token for first step
        x_first = x_emb + self.start_token
        if self.thinking_steps > 1:
            x_rest = jnp.tile(x_emb, (1, self.thinking_steps - 1, 1))
            x_seq = jnp.concatenate([x_first, x_rest], axis=1)
        else:
            x_seq = x_first

        lstm_out = x_seq
        for lstm_layer in self.lstm_layers:
            lstm_out = lstm_layer(lstm_out)
        
        # Take last timestep output: (B, d_model)
        final_out = lstm_out[:, -1, :]
        
        # Apply layer norm and get Q-value
        out = self.final_ln(final_out)
        q_value = self.q_head(out).squeeze(-1)  # (B,)
        
        return q_value


class LSTMThinkingActor(nn.Module):
    """LSTM-based actor with thinking steps for goal-conditioned discrete actions.
    
    This module uses an LSTM to perform iterative "thinking" over the input,
    allowing the model to refine its action distribution through multiple processing steps.
    
    Attributes:
        d_model: Hidden dimension size for LSTM and embeddings.
        thinking_steps: Number of thinking iterations to perform.
        action_dim: Number of discrete actions.
        gc_encoder: Optional goal-conditioned encoder.
        layer_norm: Whether to apply layer normalization.
        num_layers: Number of LSTM layers.
    """
    
    d_model: int = 64
    thinking_steps: int = 3
    action_dim: int = 6
    gc_encoder: nn.Module = None
    layer_norm: bool = True
    num_layers: int = 2
    
    def setup(self):
        # Input embedding layer
        self.input_embed_layer = nn.Dense(self.d_model)
        
        # Start token (learnable)
        self.start_token = self.param(
            'start_token',
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.d_model)
        )
        
        # Core LSTM layers
        self.lstm_layers = [
            nn.RNN(nn.LSTMCell(features=self.d_model)) 
            for _ in range(self.num_layers)
        ]
        
        # Output layers
        self.final_ln = nn.LayerNorm()
        self.actor_hwead = nn.Dense(self.action_dim, kernel_init=default_init())
    
    def __call__(self, observations, goals=None, temperature=1.0):
        """Forward pass with thinking steps.
        
        Args:
            observations: Observation tensor of shape (B, obs_dim)
            goals: Goal tensor of shape (B, goal_dim) (optional)
            temperature: Temperature for the categorical distribution.
            
        Returns:
            Distribution: Categorical distribution over actions.
        """
        # Encode inputs
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        
        # Embed input: (B, d_model) -> (B, 1, d_model)
        x_emb = self.input_embed_layer(inputs)[:, None, :]
        
        # Create sequence with start token for first step
        x_first = x_emb + self.start_token
        if self.thinking_steps > 1:
            x_rest = jnp.tile(x_emb, (1, self.thinking_steps - 1, 1))
            x_seq = jnp.concatenate([x_first, x_rest], axis=1)
        else:
            x_seq = x_first

        lstm_out = x_seq
        for lstm_layer in self.lstm_layers:
            lstm_out = lstm_layer(lstm_out)
        
        # Take last timestep output: (B, d_model)
        final_out = lstm_out[:, -1, :]
        
        # Apply layer norm and get logits
        out = self.final_ln(final_out)
        logits = self.actor_head(out)  # (B, action_dim)
        
        return distrax.Categorical(logits=logits / temperature)


class GCLSTMValue(nn.Module):
    """Goal-conditioned LSTM value network with optional ensemble support.
    
    Wraps LSTMThinkingValue with optional ensemble functionality.
    """
    
    d_model: int = 64
    thinking_steps: int = 3
    ensemble: bool = False
    gc_encoder: nn.Module = None
    layer_norm: bool = True
    num_layers: int = 2
    
    def setup(self):
        value_cls = LSTMThinkingValue
        if self.ensemble:
            value_cls = ensemblize(value_cls, 2)
        
        self.value = value_cls(
            d_model=self.d_model,
            thinking_steps=self.thinking_steps,
            gc_encoder=self.gc_encoder,
            layer_norm=self.layer_norm,
            num_layers=self.num_layers,
        )
    
    def __call__(self, observations, goals=None):
        """Forward pass.
        
        Args:
            observations: Observation tensor of shape (B, obs_dim)
            goals: Goal tensor of shape (B, goal_dim) (optional)
        
        Returns:
            If ensemble=True: Values with shape (2, B)
            If ensemble=False: Values with shape (B,)
        """
        return self.value(observations, goals)


class GCLSTMDiscreteCritic(nn.Module):
    """Goal-conditioned LSTM critic for discrete actions with ensemble support.
    
    Wraps LSTMThinkingCritic with optional ensemble functionality.
    Takes action indices as input and converts to one-hot encoding.
    """
    
    d_model: int = 64
    action_dim: int = 6
    thinking_steps: int = 3
    ensemble: bool = True
    gc_encoder: nn.Module = None
    layer_norm: bool = True
    num_layers: int = 2
    
    def setup(self):
        critic_cls = LSTMThinkingCritic
        if self.ensemble:
            critic_cls = ensemblize(critic_cls, 2)
        
        self.critic = critic_cls(
            d_model=self.d_model,
            thinking_steps=self.thinking_steps,
            gc_encoder=self.gc_encoder,
            layer_norm=self.layer_norm,
            num_layers=self.num_layers,
        )
    
    def __call__(self, observations, goals=None, actions=None):
        """Forward pass.
        
        Args:
            observations: Observation tensor of shape (B, obs_dim)
            goals: Goal tensor of shape (B, goal_dim) (optional)
            actions: Action indices of shape (B,) - will be converted to one-hot
        
        Returns:
            If ensemble=True: Q-values with shape (2, B)
            If ensemble=False: Q-values with shape (B,)
        """
        # Convert action indices to one-hot encoding
        actions_onehot = jnp.eye(self.action_dim)[actions]
        return self.critic(observations, goals, actions_onehot)


class GCLSTMDiscreteActor(nn.Module):
    """Goal-conditioned LSTM actor for discrete actions.
    
    Wraps LSTMThinkingActor for goal-conditioned discrete action selection.
    """
    
    d_model: int = 64
    action_dim: int = 6
    thinking_steps: int = 3
    gc_encoder: nn.Module = None
    layer_norm: bool = True
    num_layers: int = 2
    
    def setup(self):
        self.actor = LSTMThinkingActor(
            d_model=self.d_model,
            thinking_steps=self.thinking_steps,
            action_dim=self.action_dim,
            gc_encoder=self.gc_encoder,
            layer_norm=self.layer_norm,
            num_layers=self.num_layers,
        )
    
    def __call__(self, observations, goals=None, temperature=1.0):
        """Forward pass.
        
        Args:
            observations: Observation tensor of shape (B, obs_dim)
            goals: Goal tensor of shape (B, goal_dim) (optional)
            temperature: Temperature for the categorical distribution.
        
        Returns:
            Distribution: Categorical distribution over actions.
        """
        return self.actor(observations, goals, temperature=temperature)


class GCIQLLSTMAgent(flax.struct.PyTreeNode):
    """Goal-conditioned IQL with LSTM thinking steps (discrete actions only).
    
    Uses LSTM-based networks for value, critic, and actor that perform 
    multiple "thinking steps" before outputting their estimates.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IQL value loss."""
        q1, q2 = self.network.select('target_critic')(batch['observations'], batch['value_goals'], batch['actions'])
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
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss (AWR only for discrete LSTM)."""
        # AWR loss for discrete actions
        v = self.network.select('value')(batch['observations'], batch['actor_goals'])
        q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], batch['actions'])
        q = jnp.minimum(q1, q2)
        adv = q - v

        exp_a = jnp.exp(adv * self.config['alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }

        return actor_loss, actor_info

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
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new LSTM IQL agent (discrete only).

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions (should contain max action value for discrete).
            config: Configuration dictionary.
        """
        if not config['discrete']:
            raise ValueError("GCIQLLSTMAgent supports only discrete action spaces.")

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        action_dim = int(ex_actions.max() + 1)

        # Define encoders
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(concat_encoder=encoder_module())
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        # LSTM-based value network
        value_def = GCLSTMValue(
            d_model=config['lstm_hidden_size'],
            thinking_steps=config['thinking_steps'],
            ensemble=False,
            gc_encoder=encoders.get('value'),
            layer_norm=config['layer_norm'],
            num_layers=config['num_layers'],
        )

        # LSTM-based discrete critic
        critic_def = GCLSTMDiscreteCritic(
            d_model=config['lstm_hidden_size'],
            action_dim=action_dim,
            thinking_steps=config['thinking_steps'],
            ensemble=True,
            gc_encoder=encoders.get('critic'),
            layer_norm=config['layer_norm'],
            num_layers=config['num_layers'],
        )

        # LSTM-based discrete actor
        actor_def = GCLSTMDiscreteActor(
            d_model=config['lstm_hidden_size'],
            action_dim=action_dim,
            thinking_steps=config['thinking_steps'],
            gc_encoder=encoders.get('actor'),
            layer_norm=config['layer_norm'],
            num_layers=config['num_layers'],
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
            # Agent hyperparameters
            agent_name='gciql_lstm',
            lr=3e-4,
            batch_size=1024,
            
            # LSTM architecture
            lstm_hidden_size=64,
            thinking_steps=3,
            num_layers=2,
            layer_norm=True,
            
            # Training
            discount=0.99,
            tau=0.005,
            expectile=0.9,
            alpha=0.3,  # Temperature in AWR
            use_discounted_mc_rewards=False,
            discrete=True,
            encoder=ml_collections.config_dict.placeholder(str),
            
            # Dataset hyperparameters
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
