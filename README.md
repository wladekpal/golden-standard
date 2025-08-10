# Installation

Using uv, there is installation happening while running the code.

```bash
uv run src/train.py 
```

To run different temperatures for critic action selection based on softmax(Q):

```bash
uv run train.py env:box-pushing \
    --agent.alpha 0 \
    --exp.name test \
    --agent.agent_name crl_search \
    --exp.mode disabled \
    --exp.critic_temps 0.01 0.2 0.5 1.0 2.0
```