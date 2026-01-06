import copy
from typing import Any, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from impls.utils.encoders import GCEncoder, encoder_modules
from impls.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from impls.utils.networks import GCValue, GCDiscreteActor, default_init, ensemblize, LogParam


class LSTMThinkingCritic(nn.Module):
    """LSTM-based critic with thinking steps for goal-conditioned discrete actions.
    
    This module uses an LSTM to perform iterative "thinking" over the input,
    allowing the model to refine its Q-value estimates through multiple processing steps.
    
    Attributes:
        d_model: Hidden dimension size for LSTM and embeddings.
        action_dim: Number of discrete actions.
        thinking_steps: Number of thinking iterations to perform.
        ensemble: Whether to use an ensemble of critics.
        gc_encoder: Optional goal-conditioned encoder.
    """
    
    d_model: int = 256
    action_dim: int = 6
    thinking_steps: int = 3
    ensemble: bool = True
    gc_encoder: nn.Module = None
    
    def setup(self):
        # Input embedding layer
        self.input_embed_layer = nn.Dense(self.d_model, kernel_init=default_init())
        
        # Start token (learnable)
        self.start_token = self.param(
            'start_token',
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.d_model)
        )
        
        # Core LSTM - use LSTMCell with variance_scaling init to avoid cuSolver issues
        self.lstm = nn.RNN(
            nn.LSTMCell(
                features=self.d_model,
                kernel_init=nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform'),
                recurrent_kernel_init=nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform'),
            ),
            return_carry=True
        )
        
        # Output layers
        self.final_ln = nn.LayerNorm()
        self.q_head = nn.Dense(self.action_dim, kernel_init=default_init())
    
    def __call__(self, observations, goals=None, actions=None):
        """Forward pass with thinking steps.
        
        Args:
            observations: Observation tensor of shape (B, obs_dim)
            goals: Goal tensor of shape (B, goal_dim) (optional)
            actions: Action indices of shape (B,) (optional, for Q(s,a) queries)
            
        Returns:
            Q-values: If actions is None, returns Q(s,a) for all actions (B, action_dim).
                     If actions is provided, returns Q(s,a) for specified actions (B,).
        """
        # Encode inputs
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        
        B = inputs.shape[0]
        
        # Embed input: (B, d_model) -> (B, 1, d_model)
        x_emb = self.input_embed_layer(inputs)
        x_emb = jnp.expand_dims(x_emb, axis=1)
        
        # Create sequence with start token for first step
        x_first = x_emb + self.start_token
        if self.thinking_steps > 1:
            x_rest = jnp.broadcast_to(x_emb, (B, self.thinking_steps - 1, self.d_model))
            x_seq = jnp.concatenate([x_first, x_rest], axis=1)
        else:
            x_seq = x_first
        
        # Run LSTM: output shape (B, thinking_steps, d_model)
        carry, lstm_out = self.lstm(x_seq)
        
        # Take last timestep output: (B, d_model)
        final_out = lstm_out[:, -1, :]
        
        # Apply layer norm and get Q-values
        out = self.final_ln(final_out)
        q_values = self.q_head(out)  # (B, action_dim)
        
        # If actions provided, select Q-values for those actions
        if actions is not None:
            # actions shape: (B,) with integer indices
            q_values = jnp.take_along_axis(
                q_values, 
                jnp.expand_dims(actions, axis=-1), 
                axis=-1
            ).squeeze(-1)  # (B,)
        
        return q_values


class GCLSTMDiscreteCritic(nn.Module):
    """Goal-conditioned LSTM critic for discrete actions with ensemble support.
    
    Wraps LSTMThinkingCritic with optional ensemble functionality.
    """
    
    d_model: int = 256
    action_dim: int = 6
    thinking_steps: int = 3
    ensemble: bool = True
    gc_encoder: nn.Module = None
    
    def setup(self):
        critic_cls = LSTMThinkingCritic
        if self.ensemble:
            critic_cls = ensemblize(critic_cls, 2)
        
        self.critic = critic_cls(
            d_model=self.d_model,
            action_dim=self.action_dim,
            thinking_steps=self.thinking_steps,
            ensemble=False,  # Ensemble is handled by ensemblize wrapper
            gc_encoder=self.gc_encoder,
        )
    
    def __call__(self, observations, goals=None, actions=None):
        """Forward pass.
        
        Returns:
            If ensemble=True: array with shape (2, B) or (2, B, action_dim)
            If ensemble=False: single q tensor with shape (B,) or (B, action_dim)
        """
        q = self.critic(observations, goals, actions)
        # When ensembled, q already has shape (2, B) or (2, B, action_dim)
        # Return as-is to match GCValue/GCDiscreteCritic behavior
        return q


class GCDQNLSTMAgent(flax.struct.PyTreeNode):
    """Goal-conditioned DQN with LSTM thinking steps (discrete actions only).
    
    Uses an LSTM-based critic that performs multiple "thinking steps" before
    outputting Q-values, allowing for more sophisticated reasoning about
    state-goal relationships.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params):
        """Compute the DQN critic loss (discrete actions).

        Assumes:
         - batch['actions'] contains integer action indices.
         - critic returns Q(s,a) for specified actions or Q(s, :) for all actions.
        """
        # Current Q for taken actions - shape (2, B) when actions provided
        q_both = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )
        q1_a, q2_a = q_both[0], q_both[1]
        # Ensure shapes are (batch,)
        q1_a = jnp.squeeze(q1_a, axis=-1) if q1_a.ndim > 1 else q1_a
        q2_a = jnp.squeeze(q2_a, axis=-1) if q2_a.ndim > 1 else q2_a

        # Target Q: get Q-values for all actions at next states
        # Get Q-values for all actions (no action argument) - shape (2, B, action_dim)
        target_q = jax.lax.stop_gradient(
            self.network.select('target_critic')(
                batch['next_observations'], batch['value_goals'], actions=None
            )
        )
        # target_q shape: (2, B, action_dim)
        # Average ensemble and take max
        target_q_avg = (target_q[0] + target_q[1]) / 2  # (B, action_dim)
        max_next_q = jnp.max(target_q_avg, axis=-1)  # (B,)

        # TD or MC target
        if self.config['use_discounted_mc_rewards']:
            target = batch['rewards']
        else:
            target = batch['rewards'] + self.config['discount'] * batch['masks'] * max_next_q

        # MSE loss on both heads
        critic_loss = ((q1_a - target) ** 2 + (q2_a - target) ** 2).mean()

        # Entropy regularization for action selection
        current_q = jax.lax.stop_gradient(
            self.network.select('critic')(
                batch['observations'], 
                jnp.roll(batch['next_observations'], shift=1, axis=0), 
                actions=None,
                params=grad_params
            )
        )
        # current_q shape: (2, B, action_dim)
        current_q_avg = (current_q[0] + current_q[1]) / 2  # (B, action_dim)

        alpha_temp = self.network.select('alpha_temp')(params=grad_params)
        dist = distrax.Categorical(logits=current_q_avg / jnp.maximum(1e-6, alpha_temp))
        entropy = dist.entropy()
        alpha_temp_loss = ((entropy + self.config['target_entropy']) ** 2).mean()

        total_loss = critic_loss + alpha_temp_loss

        return total_loss, {
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

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss (only critic loss for DQN)."""
        info = {}
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
    ):
        """Returns integer action indices using the LSTM critic."""
        if not self.config['discrete']:
            raise NotImplementedError("GCDQNLSTMAgent supports only discrete action spaces.")
        
        # Get Q-values for all actions - shape (2, B, action_dim)
        q_both = self.network.select('critic')(observations, goals, actions=None)
        qs = (q_both[0] + q_both[1]) / 2  # (B, action_dim)

        if self.config['action_sampling'] == 'softmax':
            alpha_temp = jax.lax.stop_gradient(self.network.select('alpha_temp')())
            dist = distrax.Categorical(logits=qs / jnp.maximum(1e-6, alpha_temp))
            actions = dist.sample(seed=seed)
        elif self.config['action_sampling'] == 'epsilon_greedy':
            greedy_actions = jnp.argmax(qs, axis=-1)
            rng, rng_uniform = jax.random.split(seed)
            random_actions = jax.random.randint(rng, greedy_actions.shape, 0, self.config['action_dim'])
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
        """Create a new LSTM DQN agent (discrete only)."""
        if not config['discrete']:
            raise ValueError("GCDQNLSTMAgent supports only discrete action spaces.")

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        action_dim = int(ex_actions.max() + 1)

        # Define encoders
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())

        # LSTM-based discrete critic
        critic_def = GCLSTMDiscreteCritic(
            d_model=config['lstm_hidden_size'],
            action_dim=action_dim,
            thinking_steps=config['thinking_steps'],
            ensemble=True,
            gc_encoder=encoders.get('critic'),
        )

        # Keep dummy value/actor defs for compatibility
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=None,
            net_arch=config.get('net_arch', 'mlp'),
        )
        actor_def = GCDiscreteActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            gc_encoder=None,
            net_arch=config.get('net_arch', 'mlp'),
        )

        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim / 2
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
            # Agent hyperparameters
            agent_name='gcdqn_lstm',
            lr=3e-4,
            batch_size=1024,
            
            # LSTM architecture
            lstm_hidden_size=256,
            thinking_steps=3,
            
            # Legacy MLP dims (for compatibility with value/actor)
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            net_arch='mlp',
            
            # Training
            discount=0.99,
            tau=0.005,
            target_entropy=None,
            target_entropy_multiplier=0.5,
            use_discounted_mc_rewards=False,
            action_sampling='softmax',  # 'softmax', 'epsilon_greedy', or 'greedy'
            epsilon=0.1,  # For epsilon-greedy
            action_dim=6,
            
            # Legacy fields for compatibility
            expectile=0.9,
            actor_loss='ddpgbc',
            alpha=0.3,
            const_std=True,
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
