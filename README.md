# ExRL

**Sample-Efficient Online Adaptation of Long-Chunk Policies via Execution Length Control**

[Project page](https://exrl-project.github.io) · [Base-policy training](docs/pretraining.md) · [Simulation](docs/experiments.md) · [Real robots](docs/real_robot.md)

ExRL improves pretrained robot policies with reinforcement learning over **how long to execute each predicted action chunk before replanning**. The base policy supplies the actions; a lightweight execution-length policy learns when to query it again. Successful online trajectories can additionally improve the base policy through online behavior cloning.

This repository includes **ExRL-Q**, **ExRL-SAC**, online BC and demonstration augmentation, base-policy training, and six RL baselines. Simulation and real-robot implementations live together, with separate environments for their dependencies.

## Where to start

| Setting | Base policies | Training guide |
| --- | --- | --- |
| Robomimic | Diffusion Policy, ACT | [ExRL](docs/experiments.md) · [Base policies](docs/pretraining.md) |
| Adroit | Diffusion Policy | [Data, pretraining, and ExRL](docs/adroit.md) |
| LIBERO | π₀, π₀.₅, π₀-FAST, Cosmos Policy | [ExRL](docs/libero.md) · [Fine-tuning](docs/generalist_pretraining.md) |
| Your robot | A policy that predicts action chunks | [Connect your robot and policy](docs/real_robot.md) |
| RL baselines | DSRL, Policy Decorator, PA-RL, ResFiT, OGPO, DICE-RL | [Supported combinations and commands](docs/baselines.md) |

For a first run, use **ExRL-Q with DP on Robomimic Can**. The example below covers data preparation, base-policy training, online adaptation, and evaluation. Datasets and model weights are obtained separately; training code and configuration files are included here.

## Quick start

Use Linux, Python 3.10, and a CUDA GPU. From the repository root:

```bash
conda create -n exrl python=3.10
conda activate exrl
pip install -r requirements/robomimic.txt
pip install -e .
export MUJOCO_GL=egl
```

Lift, Can, Square, and Transport share this environment across ExRL variants and baselines. All Robomimic tasks use MuJoCo 3.3.5. **Tool Hang requires Robosuite 1.5.1** instead of 1.4.1: use `requirements/tool-hang.txt` in a separate environment and read the [version note](docs/experiments.md#simulator-versions). Adroit and LIBERO have their own installation guides.

### 1. Prepare demonstrations

Obtain the state-based Can **Multi-Human** dataset from [Robomimic](https://robomimic.github.io/docs/datasets/overview.html), then convert it:

```bash
python prepare_data.py --task can \
  --dataset data/can_low_dim.hdf5 --output data/can
```

The output contains training trajectories, normalization statistics, and environment metadata. Use the same statistics for base-policy training and RL.

### 2. Train the base policy

```bash
python pretrain.py --config configs/pretrain/dp_can_h50.yaml \
  --data data/can --output models/can_dp
```

You can also use a compatible existing DP checkpoint. Keep its `checkpoint/` directory and adjacent `.hydra/config.yaml` together. See [base-policy training](docs/pretraining.md) for ACT, other tasks, and checkpoint selection.

### 3. Adapt with ExRL

```bash
python train.py --config configs/robomimic/can_dp_q.yaml \
  --data data/can \
  --base-policy models/can_dp/checkpoint/state_20000.pt \
  --output outputs/can_dp_q
```

Training saves checkpoints, the resolved configuration, and TensorBoard logs. Inspect learning with `tensorboard --logdir outputs`.

### 4. Evaluate

```bash
python evaluate.py \
  --checkpoint outputs/can_dp_q/run_DATE_TIME/checkpoints/latest_model.pt \
  --episodes 50
```

Use the run directory printed by training. For online-BC experiments, evaluation also restores the adapted base policy.

## ExRL variants

Select a different configuration with the same training command:

| Configuration | What it trains |
| --- | --- |
| `configs/robomimic/can_dp_q.yaml` | Execution lengths with ExRL-Q |
| `configs/robomimic/can_dp_sac.yaml` | Execution lengths with ExRL-SAC |
| `configs/robomimic/can_act_q.yaml` | ExRL-Q around an ACT base policy |
| `configs/robomimic/can_dp_online_bc.yaml` | ExRL-Q and online BC of the base policy |
| `configs/robomimic/can_act_demo_bc.yaml` | ExRL-Q and BC with online successes and demonstrations |
| `configs/robomimic/can_dp_demo.yaml` | ExRL-Q with demonstrations in RL replay |

ACT runs take a LeRobot policy directory through `--base-policy`. Robomimic configurations use 50-action chunks by default; 32-action recipes are in `configs/robomimic/h32/`. Shared settings are in `configs/defaults/`, with task-specific changes in each recipe.

```bash
python train.py --config configs/robomimic/can_dp_q.yaml --print-config
```

Use `--seed` and `--set KEY=VALUE` to change settings without editing the YAML.

## Baselines

Baselines use the same `train.py` entry point. For DP on Can, select a file under `configs/baselines/robomimic/` with `--config`:

| Baseline | Configuration |
| --- | --- |
| DSRL | [can_dp_dsrl.yaml](configs/baselines/robomimic/can_dp_dsrl.yaml) |
| Policy Decorator | [can_dp_policy_decorator.yaml](configs/baselines/robomimic/can_dp_policy_decorator.yaml) |
| PA-RL | [can_dp_parl.yaml](configs/baselines/robomimic/can_dp_parl.yaml) |
| ResFiT | [can_dp_resfit.yaml](configs/baselines/robomimic/can_dp_resfit.yaml) |
| OGPO | [can_dp_ogpo.yaml](configs/baselines/robomimic/can_dp_ogpo.yaml) |
| DICE-RL | [can_dp_dice.yaml](configs/baselines/robomimic/can_dp_dice.yaml) |
| Online BC without RL | [can_dp_online_bc.yaml](configs/baselines/robomimic/can_dp_online_bc.yaml) |

The [baseline guide](docs/baselines.md) lists supported tasks and base policies, demonstration requirements, and evaluation commands.

## Real-robot adaptation

The real-robot core only needs an adapter that **reads observations and executes actions**, and a base-policy adapter that **predicts an action chunk**. It supports different state/action dimensions and configurable RGB cameras. The [integration guide](docs/real_robot.md) includes a software-only example and a concrete Piper/π₀.₅ adapter.

Real-robot rewards use **elapsed-time costs, including inference pauses**. Short execution lengths trigger more frequent replanning, which can make a rollout slow even when it uses few robot actions. Charging physical time makes the objective reflect actual task completion time; for multi-stage tasks, the cost decreases as stages are completed.

## Base-policy training

The repository provides DP/ACT training, Adroit DP training, and data converters plus task-specific fine-tuning configurations for OpenPI and Cosmos Policy. Generalist models use their upstream training environments, while ExRL supplies the task interfaces and recipes. Piper π₀.₅ fine-tuning is included as a robot-specific example; other robots can provide their own observation/action transforms.

See [base-policy training](docs/pretraining.md) for the full workflow. The paper uses released LIBERO weights for π₀ and task-finetuned checkpoints for the other generalist models.

## Code and credits

`exrl/` contains the learners, replay, policy training, and simulation interfaces; `baselines/` contains comparison methods; `real_robot/` contains robot-independent collection and training with optional hardware adapters. See [code structure and tests](docs/code.md).

We build on DSRL, DPPO, Robomimic, LeRobot, Stable-Baselines3, jaxrl2, OpenPI, Cosmos Policy, and the baseline implementations cited in the paper. Upstream notices are retained in [THIRD_PARTY.md](THIRD_PARTY.md).
