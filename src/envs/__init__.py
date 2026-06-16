from typing import Union, Annotated
from .block_moving.block_moving_env import BoxMovingEnv
from .block_moving.env_types import BoxMovingConfig
from .jumanji.env_types import JumanjiConfig
from .jumanji.jumanji_env import JumanjiDiscreteEnv
from flax.struct import dataclass
import tyro
from dataclasses import asdict


# TODO: this is needed, because otherwise tyro doesn't treat legal_envs union as union, but a simple type
# Once we add any other environment this can be removed
@dataclass
class DummyEnv:
    x: int = 5


legal_envs = Union[
    Annotated[BoxMovingConfig, tyro.conf.subcommand(name="box_moving")],
    Annotated[JumanjiConfig, tyro.conf.subcommand(name="jumanji")],
    Annotated[DummyEnv, tyro.conf.subcommand(name="dummy")],
]


def create_env(env_config: legal_envs):
    if isinstance(env_config, BoxMovingConfig):
        return BoxMovingEnv(**asdict(env_config))
    if isinstance(env_config, JumanjiConfig):
        return JumanjiDiscreteEnv.from_config(env_config)
    else:
        raise ValueError(f"Unknown environment type {type(env_config)}")
