from abc import ABC, abstractmethod

import jax
from brax import base
from jax import numpy as jnp


class ArmLevelGenerator(ABC):
    """Abstract base class for controlling the distribution of starting states and goals in arm environments.

    Subclass this to implement custom distributions for initial configurations and goals.
    """

    @abstractmethod
    def generate_initial_state(self, sys, rng: jax.Array) -> tuple:
        """Generate initial state (q, qd) for the environment.

        Args:
            sys: Brax system (from mjcf.load).
            rng: JAX random key.

        Returns:
            Tuple of (q, qd) for pipeline_init.
        """
        pass

    @abstractmethod
    def generate_goal(self, pipeline_state: base.State, rng: jax.Array) -> jax.Array:
        """Generate goal array for the environment.

        Args:
            pipeline_state: Current pipeline state (may be used for goal constraints).
            rng: JAX random key.

        Returns:
            Goal array (shape depends on environment).
        """
        pass


class DefaultPushLevelGenerator(ArmLevelGenerator):
    """Default level generator for push environments with uniform noise distributions.

    Samples starting state and goal from uniform distributions around default centers.
    """

    def __init__(
        self,
        arm_noise_scale: float,
        cube_noise_scale: float,
        goal_noise_scale: float,
        arm_q_default: jax.Array | None = None,
        goal_center: jax.Array | None = None,
    ):
        self.arm_noise_scale = arm_noise_scale
        self.cube_noise_scale = cube_noise_scale
        self.goal_noise_scale = goal_noise_scale
        self.arm_q_default = arm_q_default or jnp.array([1.571, 0.742, 0, -1.571, 0, 3.054, 1.449, 0.04, 0.04])
        self.goal_center = goal_center or jnp.array([0.1, 0.6, 0.03])

    def generate_initial_state(self, sys, rng: jax.Array) -> tuple:
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        cube_q_xy = sys.init_q[:2] + self.cube_noise_scale * jax.random.uniform(subkey1, [2], minval=-1)
        cube_q_remaining = sys.init_q[2:7]
        target_q = sys.init_q[7:14]
        arm_q = self.arm_q_default + self.arm_noise_scale * jax.random.uniform(subkey2, [sys.q_size() - 14], minval=-1)

        q = jnp.concatenate([cube_q_xy, cube_q_remaining, target_q, arm_q])
        qd = jnp.zeros([sys.qd_size()])
        return q, qd

    def generate_goal(self, pipeline_state: base.State, rng: jax.Array) -> jax.Array:
        del pipeline_state  # Unused for uniform sampling
        rng, subkey = jax.random.split(rng)
        cube_goal_pos = self.goal_center + jnp.array(
            [self.goal_noise_scale, self.goal_noise_scale, 0]
        ) * jax.random.uniform(subkey, [3], minval=-1)
        return cube_goal_pos
