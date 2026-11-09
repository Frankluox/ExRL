# Baselines

The baseline adapters use the same pretrained policies, observation conventions, and task definitions as ExRL. They retain each method's own learning objective. `configs/baselines/` contains task recipes; `baselines/` contains the learners and their policy interfaces.

## Available implementations

| Method | Robomimic DP | Robomimic ACT | Adroit DP | LIBERO |
| --- | --- | --- | --- | --- |
| DSRL | ✓ | — | ✓ | — |
| Policy Decorator | ✓ | ✓ | — | — |
| PA-RL | ✓ | ✓ | — | — |
| ResFiT | ✓ | ✓ | ✓ | All four base policies |
| OGPO | ✓ | — | ✓ | — |
| DICE-RL | ✓ | ✓ | ✓ | — |
| Online BC without RL | ✓ | ✓ | — | — |

A dash means that this repository does not provide that combination. DSRL and OGPO rely on a stochastic generative-policy interface, whereas ACT inference here is deterministic.

## Run a baseline

Prepare the Robomimic assets as described in the main README, then select a recipe:

```bash
python train.py \
  --config configs/baselines/robomimic/can_dp_resfit.yaml \
  --data data/can \
  --base-policy models/can_dp/checkpoint/state_20000.pt \
  --output outputs/can_resfit
```

The same command selects `can_dp_dsrl.yaml`, `can_dp_policy_decorator.yaml`, `can_dp_parl.yaml`, `can_dp_ogpo.yaml`, or `can_dp_dice.yaml`. ACT configurations take a LeRobot checkpoint directory. Use the simulator versions in [experiment settings](experiments.md); Tool Hang requires its own environment.

ResFiT and DICE-RL need the demonstration dataset, and PA-RL uses demonstrations on Transport and Tool Hang. `prepare_data.py` records the original HDF5 location in `dataset_info.json`; keep that file available. Offline replay caches are built on the first run and reused subsequently.

For Adroit and LIBERO, use their [Adroit](adroit.md) and [LIBERO](libero.md) installation and data instructions, then select a file in the corresponding `configs/baselines/` subdirectory.

## What is adapted

- **DSRL:** learns the diffusion policy's latent noise with DSRL-NA or SAC. The noise spans the full native chunk; its range and learner settings are task-specific.
- **Policy Decorator:** learns a bounded chunk residual conditioned on the state and base chunk, with progressive exploration.
- **PA-RL:** uses a distributional critic ensemble, Cal-QL, action optimization, and periodic distillation into DP or ACT. Evaluation uses optimized actions.
- **ResFiT:** extends TD3 residual learning to chunks. Rewards accumulate within a chunk, and the target discount depends on its executed length. Online and demonstration transitions are mixed equally.
- **OGPO:** applies group-based policy optimization to stochastic DDIM trajectories. Training uses best-of-eight collection; evaluation samples one target-policy trajectory. Its imitation term uses DP's native denoising loss.
- **DICE-RL:** combines a noise-conditioned residual with candidate selection and selective behavior regularization. ACT keeps its deterministic base action; the residual still receives noise. Evaluation uses one candidate.
- **Online BC:** executes full chunks and learns from successful online trajectories, with the same frozen-teacher constraint as the online-BC ExRL variant. It does not train an execution-length policy.

## Configuration and evaluation

Training evaluates periodically and saves checkpoints in the output directory printed at startup. These evaluations use each method's intended inference rule; they are not interchangeable with ExRL's greedy execution-length evaluation.

For state-based ResFiT and DICE-RL, evaluate a saved checkpoint by reusing the training command with `--resume PATH --set eval_only=true`. To resume learning, omit `eval_only`. OGPO accepts `--resume PATH` and evaluates within training. For LIBERO ResFiT, use `--resume PATH --set mode=eval`. The Adroit DSRL entry currently starts fresh runs. Robomimic DSRL, Policy Decorator, and PA-RL retain their native checkpoint handling through `--resume`.

Most recipes expose their options under `training`. DSRL, Policy Decorator, and PA-RL use the resolved files in `configs/baselines/native/`, with dotted overrides such as `--set train.batch_size=256`. Other baseline recipes use `--set batch_size=256`. Use `--check` to validate the learner import and argument configuration before allocating a training GPU.

The public recipes use in-process evaluation and local output directories. They do not require the original cluster's launch scripts or external evaluation workers.
