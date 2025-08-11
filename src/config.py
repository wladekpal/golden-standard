import os
from dataclasses import dataclass
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
    entity: str | None = None
    num_envs: int = 1024
    batch_size: int = 1024
    seed: int = 0
    max_replay_size: int = 10000
    epochs: int = 10
    intervals_per_epoch: int = 100
    updates_per_rollout: int = 1000
    eval_interval: int = 10
    use_targets: bool = False
    use_double_batch_trick: bool = False
    gamma: float = 0.99
    eval_different_box_numbers: bool = False
    eval_mirrored: bool = False
    num_gifs: int = 1
    save_dir: str | None = None



@dataclass
class Config:
    exp: ExpConfig
    env: legal_envs
    agent: FrozenConfigDict = default_config

    
