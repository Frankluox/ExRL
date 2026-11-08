# Robomimic experiments

ExRL uses state observations on Lift, Can, Square, Transport, and Tool Hang. The control frequency is 20 Hz. Lift, Can, Square, and Transport use 300 Multi-Human demonstrations; Tool Hang uses 200 Proficient-Human demonstrations.

## Simulator versions

Create separate environments for experiments requiring different simulator versions. Install the corresponding requirements file, then run `pip install -e .`.

| Experiments | Requirements file | MuJoCo | Robosuite |
| --- | --- | --- | --- |
| ExRL-Q, BC variants, and ACT ExRL-SAC on Lift / Can / Square | `requirements/robomimic-q.txt` | 3.3.5 | 1.4.1 |
| DP ExRL-SAC on Lift / Can / Square | `requirements/robomimic-sac.txt` | 3.1.6 | 1.4.1 |
| Transport | `requirements/robomimic-q.txt` | 3.3.5 | 1.4.1 |
| Tool Hang | `requirements/tool-hang.txt` | 3.3.5 | 1.5.1 |

The simulator versions are also recorded in each YAML configuration. Add `--check` to the training command to check the data, checkpoint, and installation before training.

## Configuration layout

Each task file specifies the base policy, chunk size, simulator versions, and task-dependent parameters. It inherits shared training settings from `configs/defaults/`. Paths in `defaults` are relative to the YAML file; paths passed on the command line are relative to your working directory.

Main experiments are in `configs/robomimic/`; H=32 experiments are in `configs/robomimic/h32/`. ExRL-Q covers all five tasks at both chunk sizes. ExRL-SAC also covers both sizes except Tool Hang, which currently has H=50 configurations. DP online BC covers both sizes; ACT online BC has H=50 only for Tool Hang. Demonstration-augmented ACT BC uses H=50, and demonstration replay is provided for DP on Can at H=50.

ExRL-Q uses UTD=0.25 and ExRL-SAC uses UTD=1. A collected transition represents one execution-length decision. `eval_interval` counts primitive environment steps; `warmup_transitions` and `online_transitions` count execution-length transitions. For a fixed UTD, `utd * n_envs * train_freq` must be an integer.

## Online BC and demonstrations

`*_online_bc.yaml` updates the base policy during RL using successful online trajectories. DP uses its diffusion training loss, and ACT uses its action prediction loss. An imitation constraint to the initial policy limits drift while the execution-length policy continues learning.

`*_act_demo_bc.yaml` also samples offline demonstration windows for the imitation objective. Half of each BC batch comes from demonstrations. This variant does not use the initial-policy constraint, and the BC update rate decreases from 5 to 1 per RL update over 600k environment steps after BC starts.

`can_dp_demo.yaml` uses demonstrations differently: it constructs execution-length replay transitions from trajectory segments and mixes them into critic training. The base policy stays fixed. This implementation expects successful demonstration episodes and reconstructs sparse terminal-success rewards from their boundaries.

Both demonstration variants read `train.npz` from `--data`. No additional asset manifest is needed.

## Checkpoints and evaluation

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
