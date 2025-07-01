from impls.agents.crl import CRLAgent
from impls.agents.gcbc import GCBCAgent
from impls.agents.gciql import GCIQLAgent
from impls.agents.gcivl import GCIVLAgent
from impls.agents.hiql import HIQLAgent
from impls.agents.qrl import QRLAgent
from impls.agents.sac import SACAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
)
