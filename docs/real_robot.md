# Real-robot ExRL

The real-robot implementation trains an image-conditioned execution-length critic around a frozen π₀.₅ policy. It contains the training and evaluation loops, reward calculation, replay, operator labels, and a Piper ROS interface. The robot driver and OpenPI model server run separately.

## Components

- `real_robot/collector.py` executes a selected prefix at 20 Hz and records the next query observation.
- `real_robot/rewards.py` implements primitive-step rewards and elapsed-time rewards, including policy-inference time.
- `real_robot/observations.py` applies camera crops and checkpoint normalization.
- `real_robot/piper.py` reads synchronized camera/robot observations and publishes joint commands.
- `real_robot/training.py` updates the shared image critic and saves training state.

Power insertion and parts organization use the original step-based rewards. Stack cube and clothing folding use elapsed-time costs: early RL can select short lengths frequently, so accounting for inference time encourages improvements in physical completion time as well as action efficiency.

## Installation and robot configuration

Use a dedicated Python environment with access to your ROS installation:

```bash
pip install -r requirements/real-robot.txt
pip install -e .
pip install -e /path/to/openpi/packages/openpi-client
```

Start the robot driver separately. In OpenPI's model environment, start the frozen policy server without installing ExRL's bundled dependencies:

```bash
PYTHONPATH=/path/to/ExRL /path/to/openpi/.venv/bin/python \
  -m real_robot.serve_policy --checkpoint /path/to/pi05_checkpoint \
  --prompt "Stack the cube on top of the other cube." --port 8000
```

The server uses the two RGB views and 7-dimensional Piper state from pretraining, predicts 50 actions, and applies the checkpoint's quantile normalization and state-relative joint transform. It accepts a per-request seed so training and evaluation can maintain independent sampling streams. Use the task's actual prompt and the complete checkpoint, including `assets/`.

Copy `configs/real_robot/robot.example.yaml` to a local YAML. Set the ROS image, wrist-image, state, and command topics; joint offsets; joint limits; and optional image crops. The example command topic expects Piper raw coordinates. Confirm that your controller uses the same units before enabling commands. The adapter rejects missing limits, stale observations, and camera/state timestamps that are too far apart.

Use the base checkpoint's `norm_stats.json`. The action representation is configured per task; in `state_delta` mode, the first six action coordinates are relative to the current state, while the gripper coordinate remains absolute.

Check the cameras, state, and policy connection without publishing actions:

```bash
python -m real_robot.train \
  --config configs/real_robot/stack_cube.yaml \
  --robot robot.local.yaml --normalization /path/to/norm_stats.json \
  --host localhost --port 8000 --check
```

## Train

```bash
python -m real_robot.train \
  --config configs/real_robot/stack_cube.yaml \
  --robot robot.local.yaml --normalization /path/to/norm_stats.json \
  --host localhost --port 8000 --output outputs/stack_cube
```

Reset the scene before each episode and enter `start`. During execution, enter `s` for a completed stage, `f` for failure, or `q` to end the session. Stage labels are timestamped independently of policy inference. In multi-stage tasks, the last stage marks success. Scene resets and robot enable/disable remain under the operator's control.

The task YAML files specify the prompt, episode budget, warmup, update schedule, and reward. Stack cube and power insertion have one stage, clothing folding has three, and parts organization has five. Checkpoints include the critic, target network, optimizers, replay, RNG, and counters. Continue with `--resume outputs/stack_cube/latest.pkl`.

## Evaluate

Add `--evaluate 20 --resume outputs/stack_cube/latest.pkl` to evaluate the learned critic without updates. To evaluate the base policy with a fixed execution length, use `--evaluate 20 --fixed-k 50` instead. Results are saved as episode records in `evaluation.json`.

The hardware adapter is specific to Piper and the stated message conventions. To use another robot, implement the same observation and command interface while retaining the collector and learner. Automated release checks use simulated robot inputs and do not operate hardware.
