import functools
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from impls.utils.networks import MLP
from envs.block_moving.env_types import GridStatesEnum


class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out


class NormalizeEncoder(nn.Module):
    """Simple encoder that normalizes grid observations by dividing by normalize_value.
    
    This preserves semantic consistency: same value → same representation.
    Used for grid-based environments where cell values are in range [0, 11].
    
    Set normalize_value=1.0 for passthrough (no normalization, just converts to float32).
    Set normalize_value=11.0 for /11 normalization (values in [0.0, 1.0]).
    """
    
    normalize_value: float = 11.0
    
    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        """Normalize input by dividing by normalize_value.
        
        Args:
            x: Input tensor of shape (..., grid_size^2) with integer values [0, 11]
            train: Unused, kept for compatibility
            cond_var: Unused, kept for compatibility
            
        Returns:
            Tensor of shape (..., grid_size^2) with float values.
            If normalize_value=11.0: values in [0.0, 1.0]
            If normalize_value=1.0: values in [0.0, 11.0] (passthrough)
        """
        return x.astype(jnp.float32) / self.normalize_value


class OneHotEncoder(nn.Module):
    """Encoder that converts integer grid observations to one-hot vectors.
    
    This preserves semantic consistency: same value → same one-hot representation.
    Used for grid-based environments where cell values are in range [0, num_classes-1].
    
    Args:
        num_classes: Number of possible integer values (default 12 for grid states [0, 11]).
                     After remove_targets, only 6 values are used, but we keep 12 for safety.
    """
    
    num_classes: int = 12
    
    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        """Convert integer input to one-hot vectors.
        
        Args:
            x: Input tensor of shape (..., grid_size^2) with integer values [0, num_classes-1]
            train: Unused, kept for compatibility
            cond_var: Unused, kept for compatibility
            
        Returns:
            Tensor of shape (..., grid_size^2, num_classes) with one-hot vectors.
            Each spatial position gets a num_classes-dimensional one-hot vector.
        """
        # Convert to int32 for indexing
        x_int = x.astype(jnp.int32)
        # Clip values to valid range [0, num_classes-1]
        x_int = jnp.clip(x_int, 0, self.num_classes - 1)
        # Convert to one-hot: (..., grid_size^2) -> (..., grid_size^2, num_classes)
        one_hot = jnp.eye(self.num_classes)[x_int]
        return one_hot.astype(jnp.float32)


class OneHotSemanticEncoder(nn.Module):
    """Encoder that converts integer grid observations to one-hot + semantic channels.

    It appends 4 semantic features per cell:
      - has_agent
      - has_box
      - has_target
      - carrying_box
    """

    num_classes: int = 12

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        """Convert integer input to one-hot + semantic vectors.

        Args:
            x: Input tensor of shape (..., grid_size^2) with integer values [0, num_classes-1]
            train: Unused, kept for compatibility
            cond_var: Unused, kept for compatibility

        Returns:
            Tensor of shape (..., grid_size^2, num_classes + 4) with one-hot + semantic vectors.
        """
        x_int = x.astype(jnp.int32)
        x_int = jnp.clip(x_int, 0, self.num_classes - 1)
        one_hot = jnp.eye(self.num_classes)[x_int].astype(jnp.float32)

        cell = x_int

        has_agent = (
            (cell == int(GridStatesEnum.AGENT))
            | (cell == int(GridStatesEnum.AGENT_CARRYING_BOX))
            | (cell == int(GridStatesEnum.AGENT_ON_BOX))
            | (cell == int(GridStatesEnum.AGENT_ON_TARGET))
            | (cell == int(GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX))
            | (cell == int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX))
            | (cell == int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX))
            | (cell == int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX))
        )

        has_box = (
            (cell == int(GridStatesEnum.BOX))
            | (cell == int(GridStatesEnum.AGENT_ON_BOX))
            | (cell == int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX))
            | (cell == int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX))
            | (cell == int(GridStatesEnum.BOX_ON_TARGET))
            | (cell == int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX))
        )

        has_target = (
            (cell == int(GridStatesEnum.TARGET))
            | (cell == int(GridStatesEnum.AGENT_ON_TARGET))
            | (cell == int(GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX))
            | (cell == int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX))
            | (cell == int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX))
            | (cell == int(GridStatesEnum.BOX_ON_TARGET))
        )

        carrying_box = (
            (cell == int(GridStatesEnum.AGENT_CARRYING_BOX))
            | (cell == int(GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX))
            | (cell == int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX))
            | (cell == int(GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX))
        )

        semantic = jnp.stack(
            [
                has_agent.astype(jnp.float32),
                has_box.astype(jnp.float32),
                has_target.astype(jnp.float32),
                carrying_box.astype(jnp.float32),
            ],
            axis=-1,
        )

        return jnp.concatenate([one_hot, semantic], axis=-1)


class GCEncoder(nn.Module):
    """Helper module to handle inputs to goal-conditioned networks.

    It takes in observations (s) and goals (g) and returns the concatenation of `state_encoder(s)`, `goal_encoder(g)`,
    and `concat_encoder([s, g])`. It ignores the encoders that are not provided. This way, the module can handle both
    early and late fusion (or their variants) of state and goal information.
    """

    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None
    concat_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, goals=None, goal_encoded=False):
        """Returns the representations of observations and goals.

        If `goal_encoded` is True, `goals` is assumed to be already encoded representations. In this case, either
        `goal_encoder` or `concat_encoder` must be None.
        """
        reps = []
        if self.state_encoder is not None:
            reps.append(self.state_encoder(observations))
        if goals is not None:
            if goal_encoded:
                # Can't have both goal_encoder and concat_encoder in this case.
                assert self.goal_encoder is None or self.concat_encoder is None
                reps.append(goals)
            else:
                if self.goal_encoder is not None:
                    reps.append(self.goal_encoder(goals))
                if self.concat_encoder is not None:
                    reps.append(self.concat_encoder(jnp.concatenate([observations, goals], axis=-1)))
        reps = jnp.concatenate(reps, axis=-1)
        return reps


encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
    'normalize': NormalizeEncoder,
    'passthrough': functools.partial(NormalizeEncoder, normalize_value=1.0),
    'onehot': OneHotEncoder,
    'onehot_semantic': OneHotSemanticEncoder,
}
