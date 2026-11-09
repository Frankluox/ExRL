# Fine-tuning generalist base policies

The base-policy stage learns from task demonstrations before ExRL starts. The code here supplies dataset conversion, model/data configurations, normalization, and training entry points. OpenPI and Cosmos provide their native model implementations and training loops in separate environments.

## OpenPI on LIBERO

Install [OpenPI](https://github.com/Physical-Intelligence/openpi) using its own environment and dependencies, including its pinned LeRobot version. Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) in the data-conversion environment. Do not install ExRL's bundled LeRobot into the OpenPI environment; use `PYTHONPATH=/path/to/ExRL` to expose the integration code.

Run the following commands from the ExRL directory with OpenPI's Python. Set the paths for your machine:

```bash
export OPENPI_ROOT=/path/to/openpi
export PYTHONPATH="$PWD"
export MUJOCO_GL=egl
```

### Prepare demonstrations

For the paper's LIBERO-90 task 57, replay the original HDF5 demonstrations to obtain aligned external/wrist images, proprioception, and actions:

```bash
"$OPENPI_ROOT/.venv/bin/python" -m exrl.pretraining.libero_data \
  --raw-hdf5-root /path/to/libero_datasets \
  --output-root data/lerobot --repo-id local/libero_task57 \
  --suite libero_90 --task-id 57
```

The converter writes a local LeRobot dataset without uploading it. It stores RGB images and native LIBERO actions; the model-specific transforms handle action preprocessing during training.

### Normalize and fine-tune

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

## OpenPI on a robot dataset

For another robot, implement an OpenPI `DataConfigFactory` following `exrl/pretraining/piper.py`, and set `data_factory: my_integration:MyRobotData` in a copied robot pretraining recipe. Additional constructor arguments go under `data_options`. The factory maps dataset columns to model observations and defines the action convention; use the same transforms during inference. ExRL's online collector does not require this particular model or robot.

### Piper example

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

Use `configs/pretrain/real_robot/pi05.yaml` with the same normalization and training commands as above, initializing from **π₀.₅ base**. Its stacking recipe uses 900 updates. The folding recipe, `configs/pretrain/real_robot/clothing_folding.yaml`, uses 10,000 updates and trains the action expert while keeping the vision/language backbone frozen. For folding, convert to `local/clothing_folding` with the corresponding instruction and `--crop-size 360`; match that crop during deployment.

## Cosmos Policy on LIBERO

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

For the policy server, point `EXRL_COSMOS_RECIPE` to `models/cosmos_task57/pretraining.json` and use `--config exrl_libero_h50_inference --config_file exrl/pretraining/cosmos_config.py` with the saved weights, dataset statistics, and text embeddings. See [LIBERO online adaptation](libero.md) for the server and learner commands.

Model weights and full training datasets are not bundled. Configuration/interface tests do not substitute for training a full generalist model on the target hardware.
