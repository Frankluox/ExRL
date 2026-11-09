<div align="center">

# ExRL: Sample-Efficient Online Adaptation of Long-Chunk Policies via Execution Length Control

### [Website](https://exrl-project.github.io) &nbsp;&nbsp; [Paper](https://exrl-project.github.io/paper.pdf)

</div>

<p align="center">
  <a href="assets/teaser.png">
    <img src="assets/teaser.png" alt="ExRL maintains learning efficiency as action chunks grow longer and improves π₀.₅ on four real-world manipulation tasks." width="100%">
  </a>
</p>

## Overview

Adapting pretrained robot policies with reinforcement learning can require substantial interaction data, especially when long action chunks make the action space high-dimensional. ExRL addresses this challenge by learning how long to execute each predicted action chunk before replanning, enabling sample-efficient adaptation through a small discrete decision space. Across simulation and real robots, ExRL substantially improves success rates with limited interaction, including for π₀.₅ on contact-rich and long-horizon manipulation tasks.

## Getting started

```bash
git clone https://github.com/Frankluox/ExRL.git
cd ExRL
```

Choose a setup below. Each guide covers installation, data and base-policy preparation, ExRL training, evaluation, and the available baselines. For a first experiment, start with **DP on Robomimic Can**.

| Setup guide | Base policies |
| --- | --- |
| [Robomimic](docs/robomimic.md) | Diffusion Policy, ACT |
| [Adroit](docs/adroit.md) | Diffusion Policy |
| [LIBERO](docs/libero.md) | π₀, π₀.₅, π₀-FAST, Cosmos Policy |
| [Real robots](docs/real_robot.md) | Your action-chunked policy; includes a Piper/π₀.₅ example |

The [baseline guide](docs/baselines.md) describes the comparison methods and their supported configurations. See [code structure](docs/code.md) for implementation details and tests. Datasets and model weights are obtained separately.

## Tips for hyperparameter tuning

- The shared defaults work well across most tasks; start with them before tuning.
- On real robots, a larger UTD can accelerate early learning, but may reduce final performance.
- For more complex tasks, try hidden layers of width **1024 or 2048**.
- For long-horizon tasks, a discount factor of **0.999** often works better than the default 0.997 (`gamma` in Robomimic configs, `discount` in LIBERO and real-robot configs).
- After changing the reward function or scale, check that the critic target clipping range still matches the expected returns; the default range is `[-1000, 0]`.

## Acknowledgements

The codebase builds on [DSRL](https://github.com/ajwagen/dsrl), [DPPO](https://github.com/irom-princeton/dppo), Robomimic, LeRobot, Stable-Baselines3, jaxrl2, OpenPI, and Cosmos Policy. Many thanks to the authors and maintainers for their excellent work and for making it openly available!
