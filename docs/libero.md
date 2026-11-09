# LIBERO

This guide covers base-policy preparation, serving, ExRL, and ResFiT for π₀, π₀.₅, π₀-FAST, and Cosmos Policy. The experiments use the LIBERO-90 task “pick up the cream cheese and put it in the tray” (task 57), 50-action chunks, and a 250-step episode limit.

Run commands from the ExRL repository root. If you already have a compatible checkpoint, skip the fine-tuning section and [start a policy server](#start-a-policy-server).

## Installation

Run the RL learner and the frozen base-policy model in separate environments. This avoids conflicts between the bundled LeRobot code and OpenPI's dependencies, and lets policy inference run on a separate GPU.

In the RL environment:

```bash
conda create -n exrl-libero python=3.10
conda activate exrl-libero
pip install -r requirements/libero.txt
pip install -e .
pip install -e /path/to/LIBERO
pip install -e /path/to/openpi/packages/openpi-client
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and [OpenPI](https://github.com/Physical-Intelligence/openpi) from their upstream repositories. Follow OpenPI's own installation instructions for its model environment. Do not install this repository's bundled packages into that model environment.

Cosmos Policy uses its own upstream model environment in place of OpenPI's. The same RL environment connects to either model server.

## Base-policy training

The paper uses released π₀ LIBERO weights and task-finetuned checkpoints for the other three models. Training code, data converters, and recipes are included; obtain model weights and demonstrations separately.

### OpenPI

Use OpenPI's model environment, including its pinned LeRobot version, and install LIBERO there for demonstration replay. Run the following commands from the ExRL directory with OpenPI's Python:

```bash
export OPENPI_ROOT=/path/to/openpi
export PYTHONPATH="$PWD"
export MUJOCO_GL=egl
```

#### Prepare demonstrations

For the paper's LIBERO-90 task 57, replay the original HDF5 demonstrations to obtain aligned external/wrist images, proprioception, and actions:

```bash
"$OPENPI_ROOT/.venv/bin/python" -m exrl.pretraining.libero_data \
  --raw-hdf5-root /path/to/libero_datasets \
  --output-root data/lerobot --repo-id local/libero_task57 \
  --suite libero_90 --task-id 57
```

The converter writes a local LeRobot dataset without uploading it. It stores RGB images and native LIBERO actions; the model-specific transforms handle action preprocessing during training.

#### Normalize and fine-tune

Obtain the initial model weights from OpenPI's checkpoint instructions and place them on the training machine. For π₀.₅, use the released **π₀.₅ LIBERO** checkpoint as initialization:

```bash
"$OPENPI_ROOT/.venv/bin/python" pretrain.py \
  --config configs/pretrain/libero/pi05.yaml \
  --data data/lerobot --output models/pi05_libero \
  --upstream "$OPENPI_ROOT" --norm-stats

"$OPENPI_ROOT/.venv/bin/python" pretrain.py \
  --config configs/pretrain/libero/pi05.yaml \
  --data data/lerobot --output models/pi05_libero \
  --upstream "$OPENPI_ROOT" --base-policy /path/to/pi05_libero_checkpoint
```

The first command computes statistics using the same action transforms and 50-step targets as training. The second runs OpenPI's trainer. Keep each saved checkpoint's `params/` and `assets/` together for ExRL's policy server.

| Recipe | Initialization | Task-training updates | Global batch |
| --- | --- | ---: | ---: |
| `configs/pretrain/libero/pi05.yaml` | π₀.₅ LIBERO | 50 | 128 |
| `configs/pretrain/libero/pi0fast.yaml` | π₀-FAST base | 10,000 | 32 |
| `configs/pretrain/libero/pi0.yaml` | π₀ base | 30,000 | 32 |

The π₀ recipe is provided for training on a new dataset. **The paper's π₀ experiments use OpenPI's released LIBERO checkpoint directly**, without this extra task-specific training. π₀.₅ and π₀-FAST use the task-finetuned checkpoints above. π₀-FAST also needs the official FAST tokenizer in its environment.

Use `--set repo_id=local/my_task` for another dataset. `--set training.fsdp_devices=4` shards the model across four GPUs; the batch size is global. `--resume` continues an OpenPI run. `--print-config` displays a recipe without loading a model. These integrations use the native checkpoint format and naming of the installed trainer.

### Cosmos Policy

Install [Cosmos Policy](https://github.com/NVlabs/cosmos-policy) with its LIBERO dependencies. Run this section in the Cosmos environment with ExRL on `PYTHONPATH`. Obtain the Cosmos-Predict2-2B-Video2World foundation checkpoint, its tokenizer, and the T5 model using the upstream instructions on the training machine.

Prepare the task dataset using the included Cosmos-compatible replay converter, then compute text embeddings with the native utility:

```bash
python -m exrl.pretraining.cosmos_data \
  --libero_task_suite libero_90 --task_id 57 \
  --libero_raw_data_dir /path/to/libero_datasets/libero_90 \
  --libero_target_dir data/cosmos_task57 --deterministic True

python -m cosmos_policy.datasets.save_libero_t5_text_embeddings \
  --data_dir data/cosmos_task57
```

The converter preserves Cosmos's rendering, image compression, and successful-demonstration filtering. Train the 50-action model with:

```bash
python pretrain.py --config configs/pretrain/libero/cosmos.yaml \
  --data data/cosmos_task57 --output models/cosmos_task57 \
  --base-policy /path/to/model-480p-16fps.pt \
  --set tokenizer_checkpoint=/path/to/tokenizer.pth
```

This launches the native distributed trainer with the task-specific configuration in `exrl/pretraining/cosmos_config.py`: 1,500 updates, a per-GPU batch of 4, and gradient accumulation of 8. Change `training.gpus` for your allocation. The original learning-rate schedule is retained when stopping at the selected checkpoint. The dataset loader computes the action/proprioception statistics saved beside the data.

## Start a policy server

### OpenPI

Use the released or task-finetuned OpenPI checkpoint with its `params/` and `assets/` directories. Start the server using OpenPI's Python, with ExRL on `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/ExRL /path/to/openpi/.venv/bin/python \
  -m exrl.libero.serve --family pi05 \
  --checkpoint /path/to/libero_checkpoint --port 8000
```

Use `pi0` or `pi0fast` for those backbones. The adapter applies the matching OpenPI observation, normalization, and action transforms. Noise for π₀/π₀.₅ is sampled by the RL client so evaluation can preserve the training RNG stream; π₀-FAST uses greedy decoding.

The RL process needs access to the checkpoint's small `assets/**/norm_stats.json` file, but does not load the model parameters. A matching local directory containing only those assets is sufficient on a separate learner machine.

### Cosmos Policy

In the Cosmos environment, set the saved recipe and launch the server with the training and inference configuration supplied here:

```bash
export PYTHONPATH=/path/to/ExRL
export EXRL_COSMOS_RECIPE=/path/to/models/cosmos_task57/pretraining.json

python -m exrl.libero.serve_cosmos \
  --ckpt_path /path/to/cosmos_checkpoint \
  --config exrl_libero_h50_inference \
  --config_file exrl/pretraining/cosmos_config.py \
  --dataset_stats_path /path/to/data/cosmos_task57/dataset_statistics.json \
  --t5_text_embeddings_path /path/to/data/cosmos_task57/t5_embeddings.pkl
```

The server uses port 19057 and binds to loopback; use an SSH tunnel if the learner is on another machine. For an independently trained checkpoint, supply its matching 50-action inference configuration.

## Train ExRL

In the RL environment:

```bash
python train.py --config configs/libero/pi05_q.yaml \
  --base-policy /path/to/libero_checkpoint \
  --output outputs/libero_pi05_q
```

The server defaults to `127.0.0.1:8000`; set `EXRL_POLICY_HOST` and `EXRL_POLICY_PORT` to change it. Use `pi05_sac.yaml` for SAC, or the corresponding π₀/π₀-FAST files. `--check` creates the simulation environment and checks policy inference and critic initialization without learning.

Training writes `evaluation.jsonl`, `experiment.yaml`, and `latest.pkl`. Resume with the same command and `--resume outputs/libero_pi05_q/latest.pkl`. Each resumed run starts a fresh simulator episode. Evaluate a saved learner with `python evaluate.py --checkpoint outputs/libero_pi05_q/latest.pkl --episodes 50` while the policy server is running.

For Cosmos, select `configs/libero/cosmos_q.yaml` or `cosmos_sac.yaml`. Its `--base-policy` argument supplies the π₀.₅ checkpoint statistics used to normalize the RL inputs; Cosmos loads its own weights and action statistics on the server. A local directory containing only the π₀.₅ normalization assets is sufficient for the learner.

## Run a baseline

The four `configs/baselines/libero/*_resfit.yaml` recipes connect the same policies to ResFiT. Set `assets.demo_hdf5` in the recipe to the original task demonstration HDF5, then run, for example:

```bash
python train.py --config configs/baselines/libero/pi05_resfit.yaml \
  --base-policy /path/to/libero_checkpoint \
  --output outputs/libero_pi05_resfit
```

ResFiT builds a demonstration replay cache on its first run and mixes it with online replay. See the [baseline guide](baselines.md) for resume and evaluation options.
