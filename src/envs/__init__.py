from typing import Union, Annotated
from envs.block_moving_env import BoxPushingConfig, BoxPushingEnv
from flax.struct import dataclass
import tyro
from dataclasses import asdict


#TODO: this is needed, because otherwise tyro doesn't treat legal_envs union as union, but a simple type
# Once we add any other environment this can be removed
@dataclass
class DummyEnv:
    x: int = 5

legal_envs = Union[
    Annotated[BoxPushingConfig, tyro.conf.subcommand(name="box_pushing")],
    Annotated[DummyEnv, tyro.conf.subcommand(name="dummy")],
]


def create_env(env_config: legal_envs):
    if type(env_config) == BoxPushingConfig:
        return BoxPushingEnv(**asdict(env_config))
    else:
        raise ValueError(f"Unknown environment type {type(env_config)}")