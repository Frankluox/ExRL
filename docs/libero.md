# LIBERO

ExRL adapts π₀, π₀.₅, π₀-FAST, and Cosmos on the LIBERO-90 task “pick up the cream cheese and put it in the tray” (task 57). All base policies predict 50 actions. The image critic receives an image, proprioception, and the predicted chunk; the episode limit is 250 primitive steps.

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

## Train or obtain the base policy

See [generalist pretraining](generalist_pretraining.md) for the LIBERO data converters, OpenPI/Cosmos training code, and model-specific recipes. The paper uses released π₀ LIBERO weights, while the other policies are task-finetuned.

## Start the base-policy server

Use a task-finetuned OpenPI checkpoint with its `params/` and `assets/` directories. Start the server using OpenPI's Python, with ExRL on `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/ExRL /path/to/openpi/.venv/bin/python \
  -m exrl.libero.serve --family pi05 \
  --checkpoint /path/to/libero_checkpoint --port 8000
```

Use `pi0` or `pi0fast` for those backbones. The adapter applies the matching OpenPI observation, normalization, and action transforms. Noise for π₀/π₀.₅ is sampled by the RL client so evaluation can preserve the training RNG stream; π₀-FAST uses greedy decoding.

The RL process needs access to the checkpoint's small `assets/**/norm_stats.json` file, but does not load the model parameters. A matching local directory containing only those assets is sufficient on a separate learner machine.

## Train ExRL

In the RL environment:

```bash
python train.py --config configs/libero/pi05_q.yaml \
  --base-policy /path/to/libero_checkpoint \
  --output outputs/libero_pi05_q
```

The server defaults to `127.0.0.1:8000`; set `EXRL_POLICY_HOST` and `EXRL_POLICY_PORT` to change it. Use `pi05_sac.yaml` for SAC, or the corresponding π₀/π₀-FAST files. `--check` creates the simulation environment and checks policy inference and critic initialization without learning.

Training writes `evaluation.jsonl`, `experiment.yaml`, and `latest.pkl`. Resume with the same command and `--resume outputs/libero_pi05_q/latest.pkl`. Each resumed run starts a fresh simulator episode. Evaluate a saved learner with `python evaluate.py --checkpoint outputs/libero_pi05_q/latest.pkl --episodes 50` while the policy server is running.

## Cosmos

Run Cosmos in its own model environment. `exrl.libero.serve_cosmos` provides the small RPC server used by the RL client:

```bash
PYTHONPATH=/path/to/ExRL python -m exrl.libero.serve_cosmos \
  --ckpt_path /path/to/cosmos_checkpoint \
  --config YOUR_COSMOS_INFERENCE_CONFIG \
  --config_file /path/to/cosmos_policy/config/config.py \
  --dataset_stats_path /path/to/dataset_statistics.json \
  --t5_text_embeddings_path /path/to/t5_embeddings.pkl
```

Supply a Cosmos model/configuration trained with a 50-action horizon. The model environment must provide `cosmos_policy` and its LIBERO inference utilities. The RPC server binds to loopback; use an SSH tunnel if the policy and learner are on different machines.

Select `configs/libero/cosmos_q.yaml` or `cosmos_sac.yaml`. For Cosmos, `--base-policy` supplies the π₀.₅ checkpoint statistics used to normalize the RL observations; Cosmos loads its own weights and action statistics on the server. The default Cosmos port is 19057.

## ResFiT

The four `configs/baselines/libero/*_resfit.yaml` recipes connect the same frozen policies to a chunk-residual learner. Supply the original LIBERO task demonstration HDF5 via `assets.demo_hdf5` in the recipe. ResFiT builds a demonstration replay cache on the first run and mixes it with online replay.

Policy weights, fine-tuning datasets, and GPU inference runtimes are not bundled. The release checks exercise the image learner and collection interfaces; a full LIBERO run also requires the matching model assets and simulator installation.
