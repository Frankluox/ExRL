# ExRL on your robot

ExRL learns execution lengths around an existing action-chunked policy. Its collector, replay, critic, reward, and training loop are independent of the robot SDK and policy architecture. Your integration supplies observations, primitive-action execution, and chunk inference. Piper with π₀.₅ is one included adapter.

## Installation

Use a separate Python environment for the image-based RL learner:

```bash
pip install -r requirements/real-robot.txt
pip install -e .
```

Install your controller SDK and model client separately. The base policy may run in the same process or on a model server. The core does not require ROS or OpenPI.

## Prepare a base policy

Use an existing policy that predicts action chunks, or train one on demonstrations from your setup. The RL collector only needs its inference interface; it does not depend on a particular policy-training framework.

If you use OpenPI, implement a `DataConfigFactory` following `exrl/pretraining/piper.py`, and set `data_factory: my_integration:MyRobotData` in a copied robot pretraining recipe. Additional constructor arguments go under `data_options`. The factory maps dataset columns to model observations and defines the action convention; use the same transforms during inference.

The [Piper example](#included-example-piper-with-π₀₅) below includes dataset conversion and π₀.₅ fine-tuning.

## Connect a robot and policy

Implement the interfaces in `real_robot/interfaces.py`:

| Method | Contract |
| --- | --- |
| `robot.observe(after=None)` | Return `state`, configured RGB images, and `time` from the collector host's monotonic clock. With `after`, wait for a fresh observation captured after that boundary. |
| `robot.command(action)` | Submit one primitive action in the coordinates expected by your controller. ExRL handles the control period. |
| `robot.close()` | Release connections and subscriptions. |
| `policy.predict(observation)` | Return a NumPy array shaped `(chunk_size, action_dim)` in the same coordinates accepted by `command`. |

State and action dimensions need not match. Images are `uint8` RGB arrays; `camera_keys` selects their names and order. Put hardware-specific unit conversions, camera synchronization, controller limits, and model input transforms in the adapters. If your cameras use a different clock, convert timestamps to the collector host's clock before returning observations.

Each adapter has a factory `make_robot(options, task)` or `make_policy(options, task)`. A small YAML selects it:

```yaml
factory: my_integration:make_robot
options:
  address: robot.local
```

The policy has its own YAML with the same format. `task` is the resolved experiment configuration, including the prompt and chunk size. For a stochastic policy, optional `rng_state()` and `restore_rng(state)` methods preserve its sampling stream across checkpoints.

`real_robot/example_adapter.py` is a minimal software-only implementation with four-dimensional actions and one camera. Check the complete interface without hardware:

```bash
python -m real_robot.train \
  --config configs/real_robot/task.example.yaml \
  --robot configs/real_robot/robot.example.yaml \
  --policy configs/real_robot/policy.example.yaml \
  --normalization configs/real_robot/normalization.example.json --check
```

Replace these two factories with your integration. `--check` reads observations and queries the policy but never calls `robot.command`.

## Task configuration and normalization

Start from `configs/real_robot/task.example.yaml`. Set the chunk size, control rate, episode time limit, camera names, and number of completion stages. The image critic infers state/action dimensions from the observations and predicted chunk.

For update scheduling, `utd` sets gradient updates per collected transition. An explicit `updates_per_episode` overrides that ratio; leave it unset to tune UTD. See the [tuning tips](../README.md#tips-for-hyperparameter-tuning) for learning speed, network width, and discount settings.

The critic uses mean/std statistics for state and actions. Supply a JSON with `state_mean`, `state_std`, `action_mean`, and `action_std`, or an OpenPI `norm_stats.json`. These statistics must match the action representation used by the critic. By default, the critic receives the policy's action coordinates unchanged. For state-relative actions, set `action_mode: state_delta` and list explicit `[action_index, state_index]` pairs under `action_state_indices`; this changes critic inputs, not the actions sent to the robot.

## Why elapsed-time costs?

During early training, the execution-length policy can favor very short prefixes. Each prefix requires another base-policy query, so frequent replanning can make physical rollouts slow. The reward therefore charges elapsed time from one query observation to the next, including inference and execution. Resetting the scene, waiting between episodes, and updating the critic are outside that interval.

For multi-stage tasks, each unfinished stage contributes to the time cost. Completing a stage reduces subsequent costs; completing the last stage ends the episode. Discounting uses the same elapsed-time interval. All provided robot recipes use this objective.

## Train and evaluate

```bash
python -m real_robot.train \
  --config task.yaml --robot robot.yaml --policy policy.yaml \
  --normalization normalization.json --output outputs/my_task
```

Before each episode, reset the scene and enter `start`. During execution, enter `s` for a completed stage, `f` for failure, or `q` to end the session. Labels are timestamped independently of policy inference. You retain control of physical resets and controller enable/disable.

The learner first collects random-length warmup episodes, then updates the execution-length critic between rollouts. It saves episode metrics and `latest.pkl`, including networks, optimizers, replay, random state, and counters. Resume with `--resume outputs/my_task/latest.pkl`.

Add `--evaluate 20 --resume outputs/my_task/latest.pkl` to evaluate the learned critic. To evaluate a fixed execution length, use `--evaluate 20 --fixed-k 50`. Results are written to `evaluation.json`.

## Included example: Piper with π₀.₅

The Piper adapter converts six joint coordinates and a gripper command, synchronizes two RGB views with feedback, and uses ROS. Install the Piper ROS driver and OpenPI client in addition to the core dependencies. Copy `configs/real_robot/piper.example.yaml`, then fill in your ROS topics, joint limits, offsets, and camera crops. The command topic in this example expects Piper raw units.

### Demonstrations and fine-tuning

Install [OpenPI](https://github.com/Physical-Intelligence/openpi) in its own model environment. Run the commands below from ExRL's repository root with OpenPI's Python:

```bash
export OPENPI_ROOT=/path/to/openpi
export PYTHONPATH="$PWD"
```

The included π₀.₅ example uses two RGB views and seven-dimensional state/action vectors: six joint angles in radians followed by gripper closed fraction in `[0, 1]`. Store absolute joint targets in demonstrations; the training transform converts the first six action coordinates to offsets from the query state. The gripper remains absolute.

The converter accepts an HDF5 file with episodes under `data/demo_0`, `data/demo_1`, etc. Each episode contains:

| Dataset | Shape and convention |
| --- | --- |
| `image`, `wrist_image` | `N × height × width × 3`, uint8 RGB |
| `state` | `N × 7`, observed joints and gripper |
| `actions` | `N × 7`, absolute commanded targets |

Streams must be synchronized at 20 Hz. Convert a dataset with:

```bash
"$OPENPI_ROOT/.venv/bin/python" -m exrl.pretraining.piper_data \
  --dataset data/piper_demos.hdf5 --output-root data/lerobot \
  --repo-id local/piper_task --prompt "Stack the cube on the other cube."
```

Initialize from the released π₀.₅ base weights. Compute statistics, then train:

```bash
"$OPENPI_ROOT/.venv/bin/python" pretrain.py \
  --config configs/pretrain/real_robot/pi05.yaml \
  --data data/lerobot --output models/pi05_piper \
  --upstream "$OPENPI_ROOT" --norm-stats

"$OPENPI_ROOT/.venv/bin/python" pretrain.py \
  --config configs/pretrain/real_robot/pi05.yaml \
  --data data/lerobot --output models/pi05_piper \
  --upstream "$OPENPI_ROOT" --base-policy /path/to/pi05_base_checkpoint
```

The stacking recipe uses 900 updates. For folding, convert to `local/clothing_folding` with its task instruction and `--crop-size 360`, then select `configs/pretrain/real_robot/clothing_folding.yaml`. It uses 10,000 updates of the action expert while keeping the vision/language backbone frozen. Match the image crop during deployment. Retain each checkpoint's `params/` and `assets/` directories, including normalization statistics.

### Serve the policy and run ExRL

In OpenPI's model environment, serve the task-finetuned checkpoint:

```bash
PYTHONPATH=/path/to/ExRL /path/to/openpi/.venv/bin/python \
  -m real_robot.serve_policy --checkpoint /path/to/pi05_checkpoint \
  --prompt "Stack the cube on the other cube." --port 8000
```

Then use `configs/real_robot/piper_policy.example.yaml` as the policy adapter and one of the four task recipes, for example:

```bash
python -m real_robot.train \
  --config configs/real_robot/stack_cube.yaml \
  --robot piper.local.yaml --policy configs/real_robot/piper_policy.example.yaml \
  --normalization /path/to/norm_stats.json --output outputs/stack_cube
```

The example tasks have one stage for stacking/insertion, three for folding, and five for organization. Set `episode_seconds` to a suitable task budget for your setup, accounting for both control and model-inference time.

Automated integration tests use software inputs and do not operate hardware.
