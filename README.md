# ExRL

ExRL adapts pretrained robot policies by learning **how much of each action chunk to execute before replanning**. RL chooses an execution length; the base policy produces the actions. Successful online trajectories can also be used to improve the base policy with behavior cloning.

[Project page](https://exrl-project.github.io)

This repository contains ExRL-Q and ExRL-SAC, online BC and demonstration augmentation, baseline RL implementations, and the core real-robot training code. Simulation and real-robot code share one repository, with separate installation instructions for their dependencies.

| Experiment | Base policies | Guide |
| --- | --- | --- |
| Robomimic | DP, ACT | [Training and evaluation](docs/experiments.md) · [Pretraining](docs/pretraining.md) |
| Adroit | DP | [Adroit](docs/adroit.md) |
| LIBERO | π₀, π₀.₅, π₀-FAST, Cosmos | [LIBERO](docs/libero.md) |
| Real robot | π₀.₅ on Piper | [Robot setup and training](docs/real_robot.md) |
| RL baselines | DSRL, Policy Decorator, PA-RL, ResFiT, OGPO, DICE-RL | [Baselines](docs/baselines.md) |

Datasets and pretrained weights are supplied separately. Analysis experiments, plotting code, and paper-figure data are not part of this repository.

## Quick start: ExRL on Robomimic

Use Python 3.10 on Linux with a CUDA GPU. Create a dedicated environment:

```bash
conda create -n exrl python=3.10
conda activate exrl
pip install -r requirements/robomimic-q.txt
pip install -e .
export MUJOCO_GL=egl
```

This environment runs the Can example below. Tool Hang and some SAC configurations use different simulator versions; see [experiment settings](docs/experiments.md).

Start with a pretrained DP checkpoint and its low-dimensional [Robomimic dataset](https://robomimic.github.io/docs/datasets/overview.html). Prepare the data once:

```bash
python prepare_data.py \
  --task can --dataset data/can_low_dim.hdf5 --output data/can
```

If you need to train the base policy, follow [pretraining](docs/pretraining.md). Keep the DP training `.hydra/config.yaml` beside its `checkpoint/` directory, and use the normalization statistics from that training dataset.

```bash
python train.py \
  --config configs/robomimic/can_dp_q.yaml \
  --data data/can \
  --base-policy models/can_dp/checkpoint/state_20000.pt \
  --output outputs/can_dp_q
```

Training saves checkpoints, TensorBoard logs, and the resolved experiment configuration. Run `tensorboard --logdir outputs` to inspect progress. Evaluate the saved execution-length policy with:

```bash
python evaluate.py \
  --checkpoint outputs/can_dp_q/run_DATE_TIME/checkpoints/latest_model.pt \
  --episodes 50
```

Replace `run_DATE_TIME` with the directory printed by training. For online BC, evaluation also restores the adapted base policy.

## Choose an experiment

Change the YAML while keeping the same `train.py` command:

| Configuration | Experiment |
| --- | --- |
| `configs/robomimic/can_dp_sac.yaml` | ExRL-SAC |
| `configs/robomimic/can_act_q.yaml` | ExRL-Q with ACT |
| `configs/robomimic/can_dp_online_bc.yaml` | ExRL-Q with online BC |
| `configs/robomimic/can_act_demo_bc.yaml` | ExRL-Q with online BC and demonstrations |
| `configs/robomimic/can_dp_demo.yaml` | Demonstrations in ExRL's RL replay |
| `configs/baselines/robomimic/can_dp_resfit.yaml` | ResFiT |

Shared settings live in `configs/defaults/`; task-specific files contain the differences. Robomimic uses 50-action chunks by default, with 32-action configurations in `configs/robomimic/h32/`. Override settings without editing a file:

```bash
python train.py --config configs/robomimic/can_dp_q.yaml --print-config

python train.py --config configs/robomimic/can_dp_q.yaml \
  --data data/can --base-policy models/can_dp/checkpoint/state_20000.pt \
  --seed 43 --set n_envs=4
```

See the benchmark guides for their checkpoint formats and evaluation commands. [Code structure](docs/code.md) points to the learner, replay, policy adapters, and robot interface.

## Acknowledgments

We build on DSRL, DPPO, Robomimic, LeRobot, Stable-Baselines3, OpenPI, and the baseline implementations cited in the paper. Third-party components retain their upstream notices; see [THIRD_PARTY.md](THIRD_PARTY.md).
