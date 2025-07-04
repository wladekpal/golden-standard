from impls.agents.crl import CRLAgent, get_config
from impls.agents.gcbc import GCBCAgent
from impls.agents.gciql import GCIQLAgent
from impls.agents.gcivl import GCIVLAgent
from impls.agents.hiql import HIQLAgent
from impls.agents.qrl import QRLAgent
from impls.agents.sac import SACAgent
from ml_collections import FrozenConfigDict

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
)

 
def create_agent(config: FrozenConfigDict, example_batch: dict, seed: int):
    if config.agent_name == "crl":
        agent_class = CRLAgent
    else:
        raise ValueError(f"Unknown agent class {config.agent_name}")

    agent = agent_class.create(
        seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
        example_batch['value_goals'],
    )

    return agent
