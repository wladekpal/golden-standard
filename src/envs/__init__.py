from typing import Union, Annotated
from .block_moving_env import BoxPushingConfig, BoxPushingEnv, ProgressiveBoxEnv
from flax.struct import dataclass
import tyro
from dataclasses import asdict


#TODO: this is needed, because otherwise tyro doesn't treat legal_envs union as union, but a simple type
# Once we add any other environment this can be removed
@dataclass
class DummyEnv:
    x: int = 5

@dataclass
class ProgressiveBoxConfig:
    grid_size: int = 5
    number_of_boxes_min: int = 3
    number_of_boxes_max: int = 4
    number_of_moving_boxes_max: int = 2
    episode_length: int = 100
    truncate_when_success: bool = False
    dense_rewards: bool = False
    level_generator: str = 'default'
    generator_mirroring: bool = False

legal_envs = Union[
    Annotated[BoxPushingConfig, tyro.conf.subcommand(name="box_pushing")],
    Annotated[ProgressiveBoxConfig, tyro.conf.subcommand(name="progressive_box_pushing")],
    Annotated[DummyEnv, tyro.conf.subcommand(name="dummy")],
]


def create_env(env_config: legal_envs):
    if type(env_config) == BoxPushingConfig:
        return BoxPushingEnv(**asdict(env_config))
    elif type(env_config) == ProgressiveBoxConfig:
        return ProgressiveBoxEnv(**asdict(env_config))
    else:
        raise ValueError(f"Unknown environment type {type(env_config)}")