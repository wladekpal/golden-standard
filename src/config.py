import os
from dataclasses import dataclass
from envs import legal_envs
from impls.agents.crl import get_config
from ml_collections import FrozenConfigDict

SRC_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ExpConfig:
    name: str
    project: str = "crl_subgoal"
    mode: str = "online"
    entity: str = None
    num_envs: int = 1024
    batch_size: int = 1024
    seed: int = 0
    max_replay_size = 10000
    epochs: int = 10



@dataclass
class Config:
    exp: ExpConfig
    env: legal_envs
    agent: FrozenConfigDict = get_config()

    
