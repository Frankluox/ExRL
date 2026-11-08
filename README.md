# ExRL

ExRL improves pretrained action-chunked robot policies by learning **how long to execute each predicted chunk before replanning**. Reinforcement learning acts on the discrete execution length, while the base policy produces the robot actions. The base policy can also improve during RL by imitating successful online trajectories.

[Project page](https://exrl-project.github.io)

This repository contains the Robomimic experiments with Diffusion Policy (DP) and ACT: ExRL-Q, ExRL-SAC, online behavior cloning, and learning from offline demonstrations. It also includes DP and ACT pretraining.

## Installation

Use Python 3.10 on Linux with a CUDA GPU.

```bash
conda create -n exrl python=3.10
conda activate exrl
pip install -r requirements/robomimic-q.txt
pip install -e .
export MUJOCO_GL=egl
```

This installs the environment used by the Can example below. Tool Hang and some DP ExRL-SAC experiments use different simulator versions; see [experiment settings](docs/experiments.md). The bundled Robomimic, LeRobot, and vector-environment utilities are installed with ExRL.

## Run ExRL

The following example adapts a DP policy on Can. You need its checkpoint and the dataset used for pretraining. Obtain the low-dimensional dataset from [Robomimic](https://robomimic.github.io/docs/datasets/overview.html), then prepare the observations, normalization statistics, and environment metadata:

```bash
python prepare_data.py \
  --task can --dataset data/can_low_dim.hdf5 --output data/can
```

If you do not have a base policy, train one using the [pretraining instructions](docs/pretraining.md). Dataset files and pretrained weights are not included in this repository.

```bash
python train.py \
  --config configs/robomimic/can_dp_q.yaml \
  --data data/can \
  --base-policy models/can_dp/checkpoint/state_20000.pt \
  --output outputs/can_dp_q
```

Keep the DP checkpoint's `.hydra/config.yaml` alongside its `checkpoint/` directory. When using an existing policy, use its matching normalization statistics. ACT takes a checkpoint directory instead of a `.pt` file.

Training writes a run directory containing checkpoints, TensorBoard logs, and `experiment.yaml`. To inspect learning progress:

```bash
tensorboard --logdir outputs
```

Evaluate a saved checkpoint with the settings from its run:

```bash
python evaluate.py \
  --checkpoint outputs/can_dp_q/run_DATE_TIME/checkpoints/latest_model.pt \
  --episodes 50
```

Replace `run_DATE_TIME` with the directory printed during training. Evaluation restores both the execution-length policy and, for online BC, the updated base policy.

## Other experiments

Choose a configuration in [`configs/robomimic/`](configs/robomimic). For example:

| Configuration | Method |
| --- | --- |
| `can_dp_q.yaml` | ExRL-Q with DP |
| `can_dp_sac.yaml` | ExRL-SAC with DP |
| `can_act_q.yaml` | ExRL-Q with ACT |
| `can_dp_online_bc.yaml` | ExRL-Q with online BC |
| `can_act_demo_bc.yaml` | ExRL-Q with online BC and demonstrations |
| `can_dp_demo.yaml` | ExRL-Q with demonstrations in RL replay |

The main configurations use chunks of 50 actions; [`h32/`](configs/robomimic/h32) contains the 32-action experiments. The same training command works for each configuration.

Settings shared across tasks live in [`configs/defaults/`](configs/defaults). You can edit the YAML files or override individual options:

```bash
python train.py --config configs/robomimic/can_dp_q.yaml --print-config

python train.py --config configs/robomimic/can_dp_q.yaml \
  --data data/can --base-policy models/can_dp/checkpoint/state_20000.pt \
  --set n_envs=4 --seed 43
```

UTD is the number of gradient updates per collected execution-length transition. The launcher adjusts the number of updates when `n_envs` changes.

See [experiment settings](docs/experiments.md) for task coverage, BC variants, and resuming training, and [code structure](docs/code.md) for the learning algorithm and replay implementation.

## Acknowledgments

This code builds on [DSRL](https://github.com/ajwagen/dsrl), [DPPO](https://github.com/irom-princeton/dppo), [Robomimic](https://github.com/ARISE-Initiative/robomimic), [LeRobot](https://github.com/huggingface/lerobot), and [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3). See [third-party licenses](THIRD_PARTY.md).
