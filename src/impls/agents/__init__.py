from impls.agents.crl import CRLAgent
from impls.agents.gcbc import GCBCAgent
from impls.agents.gciql import GCIQLAgent
from impls.agents.gcivl import GCIVLAgent
from impls.agents.hiql import HIQLAgent
from impls.agents.qrl import QRLAgent
from impls.agents.sac import SACAgent
import ml_collections

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
)


default_config = ml_collections.FrozenConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='crl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(256, 256),  # Actor network hidden dimensions.
            value_hidden_dims=(256, 256),  # Value network hidden dimensions.
            latent_dim=64, 
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            contrastive_loss = 'binary',
            energy_fn = 'dot',
            actor_loss='awr',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            tau=0.005,  # Target network update rate.
            expectile=0.9, # IQL expectile.
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=True,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.


            use_next_obs=False # If true, repaly buffer will return next state observation instead of future state
        )
    )

 
def create_agent(config: ml_collections.FrozenConfigDict, example_batch: dict, seed: int):
    if config.agent_name == "crl":
        agent = CRLAgent.create(
            seed,
            example_batch['observations'],
            example_batch['actions'],
            config,
            example_batch['value_goals'],
        )
    elif config.agent_name == "gciql":
        agent = GCIQLAgent.create(
            seed,
            example_batch['observations'],
            example_batch['actions'],
            config,
        )
    else:
        raise ValueError(f"Unknown agent class {config.agent_name}")

    return agent
