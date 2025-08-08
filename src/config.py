import os
from dataclasses import dataclass
from typing import Optional

from git import List
from envs import legal_envs
from impls.agents import default_config
from ml_collections import FrozenConfigDict

SRC_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ExpConfig:
    name: str
    project: str = "crl_subgoal"
    mode: str = "online"
    entity: str = "cl-probing"
    num_envs: int = 1024
    batch_size: int = 1024
    seed: int = 0
    max_replay_size: int = 10000
    epochs: int = 10
    gif_every: int = 5
    use_targets: bool = False
    use_double_batch_trick: bool = False
    gamma: float = 0.99
    critic_temps: Optional[List[float]] = None  # <- added this line


@dataclass
class Config:
    exp: ExpConfig
    env: legal_envs
    agent: FrozenConfigDict = default_config

    
