# Code structure

The public entry points keep asset paths and task settings separate from learning code:

```text
train.py / evaluate.py   Robomimic ExRL and benchmark launchers
pretrain.py             DP / ACT pretraining, including Adroit DP
prepare_data.py         Robomimic data conversion
configs/                Shared defaults and task recipes
exrl/
  algorithms/           State-based execution-length Q / SAC learners
  runners/              Robomimic policy loading, replay, collection and BC
  adroit/               Adroit environments, DP adapter and learners
  pixel/                Image-based Q / SAC, encoders and shared replay
  libero/               LIBERO collection and frozen-policy servers
baselines/              Baseline learners and policy adapters
real_robot/             Piper collection, operator labels and training
vendor/                 Bundled dependency subsets
```

## Learning and collection

An ExRL transition records the query observation, predicted chunk, chosen execution length, accumulated reward, actual executed length, and next query observation. The critic's target uses the corresponding multi-step discount. The discrete CQL term regularizes the execution-length values. ExRL-Q selects lengths from the critic; ExRL-SAC learns a categorical actor.

Robomimic's `exrl/config.py` resolves YAML defaults and translates UTD into updates for the configured number of parallel environments. `exrl/experiment.py` routes the other benchmarks and baselines to their native learners without changing their objectives.

Online BC buffers successful trajectories and trains the base policy with its native loss. The teacher constraint anchors the policy to its initial behavior. Demonstration augmentation can feed the imitation buffer or, for the supported DP recipe, provide off-policy transitions to the execution-length critic.

Image-based ExRL shares its encoder, learner, replay, and checkpoint code across LIBERO and real-robot training. Collection is benchmark-specific: LIBERO advances simulation steps, while the robot collector also accounts for elapsed physical time and operator stage labels.

## Tests

In the state-based environment:

```bash
pip install pytest
pytest -q tests --ignore=tests/test_pixel_learner.py
```

In the image-learner environment:

```bash
JAX_PLATFORMS=cpu pytest -q tests/test_pixel_learner.py tests/test_real_rewards.py tests/test_real_collector.py
```

The tests cover learning targets, conservative losses, replay, BC, update schedules, checkpoint restoration, and simulated robot timing. They do not require model weights or a connected robot. Benchmark-level runs additionally require the assets described in each guide.
