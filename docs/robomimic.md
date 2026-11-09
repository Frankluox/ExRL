# Robomimic

This guide covers demonstration preparation, DP/ACT pretraining, ExRL, and baseline RL on Lift, Can, Square, Transport, and Tool Hang. The experiments use state observations and 20 Hz control. Lift, Can, Square, and Transport use 300 Multi-Human demonstrations; Tool Hang uses 200 Proficient-Human demonstrations.

Run all commands from the repository root.

## Installation

Use Linux, Python 3.10, and a CUDA GPU:

```bash
conda create -n exrl python=3.10
conda activate exrl
pip install -r requirements/robomimic.txt
pip install -e .
export MUJOCO_GL=egl
```

### Simulator versions

All Robomimic experiments use **MuJoCo 3.3.5**. ExRL-Q, ExRL-SAC, online BC, and the baselines share the same environment; only Tool Hang needs a different Robosuite version.

| Tasks | Requirements file | Robosuite |
| --- | --- | --- |
| Lift, Can, Square, Transport | `requirements/robomimic.txt` | 1.4.1 |
| Tool Hang | `requirements/tool-hang.txt` | 1.5.1 |

**Tool Hang requires Robosuite 1.5.1.** In our experiments, base-policy training under 1.4.1 produced policies with 0% success on this task. For Tool Hang, create a separate environment using `requirements/tool-hang.txt` in the installation command above, and use it for pretraining, RL, and evaluation.

## Prepare demonstrations

Obtain the state-based dataset from [Robomimic](https://robomimic.github.io/docs/datasets/overview.html). For Can, use the Multi-Human dataset:

```bash
python prepare_data.py --task can \
  --dataset data/can_low_dim.hdf5 --output data/can
```

Conversion produces `train.npz`, `normalization.npz`, `can_lowdim_stats.npz`, and `can.json`, preserving episode boundaries. Use `--filter-key train` to select an HDF5 dataset mask. Keep the same data and statistics for pretraining and RL. Transport uses both arms: 59 state dimensions and 14 action dimensions.

## Train the base policy

If you already have a compatible checkpoint and its normalization statistics, skip to [Train ExRL](#train-exrl).

### Diffusion Policy

```bash
python pretrain.py --config configs/pretrain/dp_can_h50.yaml \
  --data data/can --output models/can_dp
```

DP uses an MLP noise predictor. Its architecture and training schedule are specified in the YAML file. Training writes `checkpoint/state_EPOCH.pt` and `.hydra/config.yaml`; keep both when copying a policy. The diffusion training schedule has 100 steps. During ExRL, DDIM uses 5 sampling steps for Lift and 20 for the other tasks.

### ACT

```bash
python pretrain.py --config configs/pretrain/act_can_h50.yaml \
  --data data/can --output models/can_act
```

ACT uses low-dimensional observations, no VAE, and zero dropout. Training writes checkpoint directories and a `final/` directory containing the policy, configuration, and observation/action processors. Pass the entire directory to `train.py --base-policy`, for example `models/can_act/final`.

`max_steps` sets the learning-rate schedule, while `stop_after_steps` selects when to stop training. Keeping these separate preserves the learning-rate schedule when using an intermediate checkpoint. ACT bundles contain inference state, not the optimizer state needed to resume pretraining exactly.

### Checkpoint selection

The configurations stop at the checkpoints used to initialize the corresponding adaptation experiments.

| Task | DP epoch, H=32 | DP epoch, H=50 | ACT step, H=32 | ACT step, H=50 |
| --- | ---: | ---: | ---: | ---: |
| Lift | 20,000 | 10,000 | 40 | 30 |
| Can | 20,000 | 20,000 | 300 | 400 |
| Square | 20,000 | 20,000 | 500,000 | 875,000 |
| Transport | 10,000 | 10,000 | 670,000 | 765,000 |
| Tool Hang | 4,000 | 28,000 | 100,000 | 150,000 |

## Train ExRL

Start with ExRL-Q around a DP base policy:

```bash
python train.py --config configs/robomimic/can_dp_q.yaml \
  --data data/can \
  --base-policy models/can_dp/checkpoint/state_20000.pt \
  --output outputs/can_dp_q
```

Select another configuration under `configs/robomimic/` with the same command. ACT runs take a LeRobot policy directory through `--base-policy`, such as `models/can_act/final`.

| Variant | Example configuration |
| --- | --- |
| ExRL-Q | `can_dp_q.yaml` or `can_act_q.yaml` |
| ExRL-SAC | `can_dp_sac.yaml` or `can_act_sac.yaml` |
| ExRL-Q with online BC | `can_dp_online_bc.yaml` |
| ExRL-Q with online BC and demonstrations | `can_act_demo_bc.yaml` |
| ExRL-Q with demonstrations in RL replay | `can_dp_demo.yaml` |

Main configurations use 50-action chunks; 32-action configurations are under `configs/robomimic/h32/`. ExRL-Q and ExRL-SAC support DP and ACT on all five tasks at both sizes. The same directories contain the available online BC and demonstration recipes; select the file matching your task, policy, and chunk size.

### Online BC and demonstrations

`*_online_bc.yaml` updates the base policy during RL using successful online trajectories. DP uses its diffusion training loss, and ACT uses its action prediction loss. An imitation constraint to the initial policy limits drift while the execution-length policy continues learning.

`*_act_demo_bc.yaml` also samples offline demonstration windows for the imitation objective. Half of each BC batch comes from demonstrations. This variant does not use the initial-policy constraint, and the BC update rate decreases from 5 to 1 per RL update over 600k environment steps after BC starts.

`can_dp_demo.yaml` uses demonstrations differently: it constructs execution-length replay transitions from trajectory segments and mixes them into critic training. The base policy stays fixed. This implementation expects successful demonstration episodes and reconstructs sparse terminal-success rewards from their boundaries.

Both demonstration variants read `train.npz` from `--data`.

### Configuration and parallelism

Task files inherit shared settings from `configs/defaults/`. Paths in `defaults` are relative to the YAML file; command-line paths are relative to your working directory. Use `--seed` and `--set KEY=VALUE` to override settings, and `--print-config` to inspect them:

```bash
python train.py --config configs/robomimic/can_dp_q.yaml \
  --set n_envs=8 --print-config
```

Add `--check` to a full training command to validate its data, checkpoint, and simulator installation. `eval_interval` counts primitive environment steps; `warmup_transitions` and `online_transitions` count execution-length decisions.

#### UTD and parallelism

In these YAML files, `training.utd` is the effective number of gradient updates per newly collected execution-length transition: 0.25 for ExRL-Q and 1 for ExRL-SAC. The launcher converts it to the lower-level runner's update count:

```text
updates per collection cycle = utd × n_envs × train_freq
```

For example, with `utd=0.25` and `train_freq=1`, four environments give one update per cycle; eight give two. Increasing `n_envs` therefore increases the update count automatically. Keep the YAML `utd` fixed to preserve the effective ratio. The product must be a positive integer.

When using the lower-level runners directly, their `utd` argument is the update count per cycle, so increase it proportionally with `n_envs`. This scaling keeps learning per collected transition constant; increasing the effective UTD is a separate tuning choice.

## Evaluate and resume

Training saves TensorBoard logs; use `tensorboard --logdir outputs` to inspect learning. For a saved ExRL checkpoint:

```bash
python evaluate.py \
  --checkpoint outputs/can_dp_q/run_DATE_TIME/checkpoints/latest_model.pt \
  --episodes 50
```

Replace `run_DATE_TIME` with the directory printed by training.

Training creates a unique `run_DATE_TIME` directory under `--output`. It contains the resolved `experiment.yaml`, detailed runner settings in `config.json`, metrics, TensorBoard events, and checkpoints. Full training checkpoints include replay and optimizer state; compact evaluation checkpoints do not support resuming training.

To resume a full checkpoint:

```bash
python train.py \
  --config outputs/can_dp_q/run_DATE_TIME/experiment.yaml \
  --resume outputs/can_dp_q/run_DATE_TIME
```

To evaluate after moving a checkpoint or its data:

```bash
python evaluate.py --checkpoint checkpoint.pt \
  --config experiment.yaml --data data/can \
  --base-policy models/can_dp/checkpoint/state_20000.pt \
  --episodes 50 --output results/can.json
```

Evaluate BC checkpoints with their original BC configuration so that the updated base-policy weights are restored. Full checkpoints can be large because they include replay buffers.

## Run a baseline

Reuse the same prepared data and compatible base checkpoint, selecting a configuration under `configs/baselines/robomimic/`:

```bash
python train.py --config configs/baselines/robomimic/can_dp_dsrl.yaml \
  --data data/can \
  --base-policy models/can_dp/checkpoint/state_20000.pt \
  --output outputs/can_dsrl
```

The [baseline guide](baselines.md) lists all methods, supported base policies, demonstration requirements, and method-specific evaluation commands. Baselines retain their own learning objectives and inference rules.
