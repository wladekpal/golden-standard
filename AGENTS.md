This is a research code that focues on Reinforcement Learning and stitching. The main idea is to test the stitching capabilities of different algorithms and architectures in controlled, easily configurable discrete environment. The environment is very simple, it is a grid world, with boxes and targets, the agent can take a box and move with it. It has to put all boxes on targets.

## Dev environment:
- Use uv when running anything
- For test training runs use `--exp.intervals_per_epoch 10 --exp.updates_per_rollout: 10`. The model will not learn anything in this setup, but it will progress through epochs pretty quickly, which is good for debugging.
- Doing full runs (i.e. with bigger parameters than those above) is not possible locally, it should be done on a GPU cluster, so try to avoid proposing such runs.

Look at CLAUDE.md for further guidence.