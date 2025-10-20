from typing import Any, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x

lecun_unfirom = nn.initializers.variance_scaling(1/3, "fan_in", "uniform")
bias_init = nn.initializers.zeros

class ResidualBlock(nn.Module):
    hidden_dim: int = 1024
    activation: Any = nn.swish
    
    @nn.compact
    def __call__(self, x):
        identity = x
        x = nn.Dense(self.hidden_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = nn.LayerNorm()(x)
        x = self.activation(x)
        x = nn.Dense(self.hidden_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = nn.LayerNorm()(x)
        x = self.activation(x)
        x = nn.Dense(self.hidden_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = nn.LayerNorm()(x)
        x = self.activation(x)
        x = nn.Dense(self.hidden_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = nn.LayerNorm()(x)
        x = self.activation(x)
        x = x + identity
        return x
    
class ResidualNetwork(nn.Module):
    blocks_dims: Sequence[int]
    activations: Any = nn.swish
    activate_final: bool = False
    kernel_init: Any = lecun_unfirom
    layer_norm: bool = True

    def setup(self):
        assert self.layer_norm, "ResidualNetwork requires layer_norm=True"
        return super().setup()

    @nn.compact
    def __call__(self, x):
        output_dim = self.blocks_dims[-1]
        blocks_dims = self.blocks_dims[:-1]
        x = nn.Dense(self.blocks_dims[0], kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = nn.LayerNorm()(x)
        x = self.activations(x)
        for block_dim in blocks_dims:
            x = ResidualBlock(block_dim, self.activations)(x)
        x = nn.Dense(output_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        if self.activate_final:
            x = self.activations(x)
        return x
    
def create_network(net_type: str) -> nn.Module:
    if net_type == 'mlp':
        return MLP
    elif net_type == 'res_block':
        return ResidualNetwork
    else:
        raise ValueError(f"Unknown network type {net_type}")

class LengthNormalize(nn.Module):
    """Length normalization layer.

    It normalizes the input along the last dimension to have a length of sqrt(dim).
    """

    @nn.compact
    def __call__(self, x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])


class Param(nn.Module):
    """Scalar parameter module."""

    init_value: float = 0.0

    @nn.compact
    def __call__(self):
        return self.param('value', init_fn=lambda key: jnp.full((), self.init_value))


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class RunningMeanStd(flax.struct.PyTreeNode):
    """Running mean and standard deviation.

    Attributes:
        eps: Epsilon value to avoid division by zero.
        mean: Running mean.
        var: Running variance.
        clip_max: Clip value after normalization.
        count: Number of samples.
    """

    eps: Any = 1e-6
    mean: Any = 1.0
    var: Any = 1.0
    clip_max: Any = 10.0
    count: int = 0

    def normalize(self, batch):
        batch = (batch - self.mean) / jnp.sqrt(self.var + self.eps)
        batch = jnp.clip(batch, -self.clip_max, self.clip_max)
        return batch

    def unnormalize(self, batch):
        return batch * jnp.sqrt(self.var + self.eps) + self.mean

    def update(self, batch):
        batch_mean, batch_var = jnp.mean(batch, axis=0), jnp.var(batch, axis=0)
        batch_count = len(batch)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        return self.replace(mean=new_mean, var=new_var, count=total_count)


class GridStateEmbedding(nn.Module):
    """Embedding layer for grid states.
    
    Converts discrete grid state values (0-11) into dense vector representations.
    
    Attributes:
        num_states: Number of possible grid states (12 for your environment)
        embed_dim: Dimension of embedding vectors
        grid_size: Size of the grid (for positional encoding)
        use_position: Whether to add positional embeddings
    """
    
    num_states: int = 12
    embed_dim: int = 64
    grid_size: int = 5
    use_position: bool = True
    
    def setup(self):
        # State embedding: maps each grid state (0-11) to a dense vector
        self.state_embedding = nn.Embed(
            num_embeddings=self.num_states, 
            features=self.embed_dim
        )
        
        # Optional positional embedding: adds spatial information
        if self.use_position:
            self.position_embedding = nn.Embed(
                num_embeddings=self.grid_size * self.grid_size,  # 25 positions for 5x5 grid
                features=self.embed_dim
            )
    
    def __call__(self, grid_obs):
        """
        Args:
            grid_obs: Grid observations, shape (batch_size, grid_size*grid_size) with values 0-11
        
        Returns:
            Embedded observations, shape (batch_size, grid_size*grid_size*embed_dim)
        """
        batch_size = grid_obs.shape[0]
        grid_flat = grid_obs.reshape(batch_size, -1)  
        
        state_embeds = self.state_embedding(grid_flat)  # (batch, 25, embed_dim)
        
        if self.use_position:
            # Create position indices: [0, 1, 2, ..., 24] for each batch item
            positions = jnp.tile(
                jnp.arange(self.grid_size * self.grid_size),
                (batch_size, 1)
            )
            pos_embeds = self.position_embedding(positions)  # (batch, 25, embed_dim)
            
            combined_embeds = state_embeds + pos_embeds
        else:
            combined_embeds = state_embeds
        
        # Flatten to (batch_size, grid_size*grid_size*embed_dim)
        return combined_embeds.reshape(batch_size, -1)

class GCActor(nn.Module):
    """Goal-conditioned actor.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None
    embedding_encoder: nn.Module = None
    net_arch: str = 'mlp'

    def setup(self):
        net = create_network(self.net_arch)
        self.actor_net = net(self.hidden_dims, activate_final=True)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Scaling factor for the standard deviation.
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            if hasattr(self, 'embedding_encoder') and self.embedding_encoder is not None:
                embedded_obs = self.embedding_encoder(observations)
                inputs = [embedded_obs]
                if goals is not None:
                    embedded_goals = self.embedding_encoder(goals)
                    inputs.append(embedded_goals)
                inputs = jnp.concatenate(inputs, axis=-1)
            else:
                inputs = [observations]
                if goals is not None:
                    inputs.append(goals)
                inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class GCDiscreteActor(nn.Module):
    """Goal-conditioned actor for discrete actions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None
    embedding_encoder: nn.Module = None
    net_arch: str = 'mlp'

    def setup(self):
        net = create_network(self.net_arch)
        self.actor_net = net(self.hidden_dims, activate_final=True)
        self.logit_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Inverse scaling factor for the logits (set to 0 to get the argmax).
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            if hasattr(self, 'embedding_encoder') and self.embedding_encoder is not None:
                embedded_obs = self.embedding_encoder(observations)
                inputs = [embedded_obs]
                if goals is not None:
                    embedded_goals = self.embedding_encoder(goals)
                    inputs.append(embedded_goals)
                inputs = jnp.concatenate(inputs, axis=-1)
            else:
                inputs = [observations]
                if goals is not None:
                    inputs.append(goals)
                inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        logits = self.logit_net(outputs)

        distribution = distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))

        return distribution


class GCValue(nn.Module):
    """Goal-conditioned value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    ensemble: bool = True
    gc_encoder: nn.Module = None
    embedding_encoder: nn.Module = None
    net_arch: str = 'mlp'

    def setup(self):
        mlp_module = create_network(self.net_arch)
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        value_net = mlp_module((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, goals=None, actions=None):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
        """
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            if hasattr(self, 'embedding_encoder') and self.embedding_encoder is not None:
                embedded_obs = self.embedding_encoder(observations)
                inputs = [embedded_obs]
                if goals is not None:
                    embedded_goals = self.embedding_encoder(goals)
                    inputs.append(embedded_goals)
            else:
                inputs = [observations]
                if goals is not None:
                    inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class GCDiscreteCritic(GCValue):
    """Goal-conditioned critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions)


class GCBilinearValue(nn.Module):
    """Goal-conditioned bilinear value/critic function.

    This module computes the value function as V(s, g) = phi(s)^T psi(g) / sqrt(d) or the critic function as
    Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d), where phi and psi output d-dimensional vectors.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        value_exp: Whether to exponentiate the value. Useful for contrastive learning.
        state_encoder: Optional state encoder.
        goal_encoder: Optional goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None
    embedding_encoder: nn.Module = None
    net_arch: str = 'mlp'

    def setup(self):
        mlp_module = create_network(self.net_arch)
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)

        self.phi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.psi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, actions=None, info=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions (optional).
            info: Whether to additionally return the representations phi and psi.
        """
        # Apply embedding encoder if available
        if hasattr(self, 'embedding_encoder') and self.embedding_encoder is not None:
            observations = self.embedding_encoder(observations)
            goals = self.embedding_encoder(goals)
        
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        phi = self.phi(phi_inputs)
        psi = self.psi(goals)

        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi, psi
        else:
            return v


class GCDiscreteBilinearCritic(GCBilinearValue):
    """Goal-conditioned bilinear critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None, info=False):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions, info)


class GCMRNValue(nn.Module):
    """Metric residual network (MRN) value function.

    This module computes the value function as the sum of a symmetric Euclidean distance and an asymmetric
    L^infinity-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    encoder: nn.Module = None
    embedding_encoder: nn.Module = None
    net_arch: str = 'mlp'

    def setup(self):
        net = create_network(self.net_arch)
        self.phi = net((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the MRN value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            # Apply embedding encoder if available
            if hasattr(self, 'embedding_encoder') and self.embedding_encoder is not None:
                observations = self.embedding_encoder(observations)
                goals = self.embedding_encoder(goals)
                
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        sym_s = phi_s[..., : self.latent_dim // 2]
        sym_g = phi_g[..., : self.latent_dim // 2]
        asym_s = phi_s[..., self.latent_dim // 2 :]
        asym_g = phi_g[..., self.latent_dim // 2 :]
        squared_dist = ((sym_s - sym_g) ** 2).sum(axis=-1)
        quasi = jax.nn.relu((asym_s - asym_g).max(axis=-1))
        v = jnp.sqrt(jnp.maximum(squared_dist, 1e-12)) + quasi

        if info:
            return v, phi_s, phi_g
        else:
            return v


class GCIQEValue(nn.Module):
    """Interval quasimetric embedding (IQE) value function.

    This module computes the value function as an IQE-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        dim_per_component: Dimension of each component in IQE (i.e., number of intervals in each group).
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    dim_per_component: int
    layer_norm: bool = True
    encoder: nn.Module = None
    embedding_encoder: nn.Module = None
    net_arch: str = 'mlp'

    def setup(self):
        net = create_network(self.net_arch)
        self.phi = net((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.alpha = Param()

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the IQE value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        alpha = jax.nn.sigmoid(self.alpha())
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            # Apply embedding encoder if available
            if hasattr(self, 'embedding_encoder') and self.embedding_encoder is not None:
                observations = self.embedding_encoder(observations)
                goals = self.embedding_encoder(goals)
                
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        x = jnp.reshape(phi_s, (*phi_s.shape[:-1], -1, self.dim_per_component))
        y = jnp.reshape(phi_g, (*phi_g.shape[:-1], -1, self.dim_per_component))
        valid = x < y
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        neg_inc_copies = jnp.take_along_axis(valid, ixy % self.dim_per_component, axis=-1) * jnp.where(
            ixy < self.dim_per_component, -1, 1
        )
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = -1.0 * (neg_inp_copies < 0)
        neg_incf = jnp.concatenate([neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1)
        components = (sxy * neg_incf).sum(axis=-1)
        v = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)

        if info:
            return v, phi_s, phi_g
        else:
            return v
