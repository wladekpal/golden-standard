
# Overview 📖

Reinforcement learning (RL) promises to solve long-horizon tasks even when training data contains only short fragments of the behaviors. This quality is called stitching, and is a crucial prerequisite for more general, foundational RL models. Conventional wisdom dictates, that only temporal difference (TD) methods are able stitch fragments of experiences gathered during the training and use them to solve more complex tasks. We show that, while on simple, low-dimensional settings TD methods can indeed stitch experiences, this does not transfer to more complex, high-dimensional tasks. Additionally we show that Monte Carlo (MC) methods, while they still fall behind TD methods, are able to exhibit some stitching behavior as well. Furthermore we determine that scaling the network sizes plays more of a critical role in closing the generalization gap than previously thought, and is a promising avenue of research, especially in the age of larger models in RL.


<p align="center">
    <img src="assets/stitching_1_success.gif" alt="Box Moving Task Illustration" width="600"/>
    <img src="assets/stitching_4_failure.gif" alt="Box Moving Task Illustration" width="600"/>
    <p style="text-align:center;font-size:2.0em;margin-top:8px;">
    TD methods can stitch in low-dimensional tasks (left), but fail in higher-dimensional setting (right).
    </p>
</p>


# Installation & Setup 🔧

This repo uses uv. To install uv, please follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
To install all dependencies and create the virual environment run:
```
uv sync
```
> [!NOTE]
> We are using `wandb` for experiment tracking by default, you may be prompted to login to wandb when running first experiment. If you don't want to use wandb, pass `--exp.mode disabled` flag to skip wandb logging.

# Running experiments 🔬
To run a simple training with CRL and default environment configuration use:
> [!WARNING]
> Our repository is optimized for GPU, running line below without decent GPU may take a very long time.
```bash
uv run src/train.py env:box-moving --exp.name test
```

Current version of the code only supports `box-moving` environment, thus in each experiment you should specify `env:box-moving` flag first.

## Wandb logging 📈
When using wandb logging, all experiment results, including detailed information about environment and algorithm data will be gathered and logged. Additionally, a simple gif with agent's behavior will be recorded and stored in wandb.

## Hyperparameters ⚙️
Hyperparameters and options can be seen by invoking: `uv run src/train.py --help`. All options can be divided into three general categories, indicated by prefixes used in the flag names:
* `exp.` - General experimental setup settings, including logging options, random seeds, experiment names etc (see [here](./src/config.py) for more details).
* `env.` - Environment related settings, including difficulty of the environment, goal and starting positions sampling distributions, number of boxes, size of the grid etc (see `BoxMovingConfig` class [here](./src/envs/block_moving/env_types.py) for more details).
* `actor.` - Algorithm related settings, including learning rates, batch sizes, network architectures and the choice of the algorithm itself (see [here](./src/impls/agents/__init__.py) for more details).

# Environment 🕹️
Our Box Moving environment is a grid-world environment where an agent needs to move boxes to target locations. The environment can be configured to have different grid sizes, number of boxes, and difficulty levels. While this may seem simple, the complexity of the task rises exponentially with the increase of grid size and number of boxes, which is ideal for testing of stitching capabilities. 

Environment supports 2 modes of sampling box and goal positions (`--env.level_generator` flag):
* `default` - Boxes and targets are spawned randomly on the grid.
* `variable` - Boxes and targets are spawned in corners of the grid. During normal operation the corners for boxes and goals are always adjacent. When `--exp.eval_special` flag is passed, the algorithm is additionaly evaluated on boxe and goal corners being diagonally opposite, the results from this mode is logged in wandb under `eval_special` tab.

# Supported algorithms 🧠
Due to the nature of investigated issue we primarly focused on goal-conditioned RL algorithms, and to further isolate the problems related to stitching we removed actors from all of the tested algorithms. The actions are instead sampled directly from the Q-function via softmax sampling. The algorithms we tested include:
* Contrastive RL (CRL) - MC algorithm running without rewards.
* C-Learning - TD algorithm running without rewards.
* GCDQN - Both TD and MC versions, with rewards.
* GCIQL - Both TD and MC versions, with rewards.


## Also see 👀
* [OGBench](https://github.com/seohongpark/ogbench) - Benchmark for offline goal-conditioned RL algorithms, on which we based our algorithms impelementation and code structure.
* [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL) - Online goal-conditioned RL benchmark, with impelementations of various goal-conditioned RL algorithms in JAX, primarly Contrastive RL. 
* [Jumanji](https://github.com/instadeepai/jumanji) - A collection of RL environments written in JAX, Sokoban environment included here inspired our box-moving environment.


## Citing 📄
If you find this work useful in your research, please cite us as follows:
```bibtex
@inproceedings{anonymous2026temporal,
  title={Is Temporal Difference Learning the Gold Standard for Stitching in RL?},
  author={Michał Bortkiewicz and Władysław Pałucki and Benjamin Eysenbach and Mateusz Ostaszewski},
  year={2026},
  url={https://michalbortkiewicz.github.io/golden-standard/}
}
```
## Questions or issues ❓
If you have any questions or issues feel free to open an issue on GitHub or contact us directly via email:
- Michał Bortkiewicz ([michalbortkiewicz8@gmail.com](michalbortkiewicz8@gmail.com)).
- Władysław Pałucki ([w.palucki@uw.edu.pl)](w.palucki@uw.edu.pl)).