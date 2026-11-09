# Base-policy training

| Setting | Included training path |
| --- | --- |
| Robomimic | DP and ACT from demonstrations, described below |
| Adroit | U-Net DP; [data and training](adroit.md) |
| LIBERO | OpenPI and Cosmos task fine-tuning; [guide](generalist_pretraining.md) |
| Real robot | π₀.₅ data transforms and a Piper fine-tuning example; [guide](generalist_pretraining.md) |

## Robomimic demonstrations

Prepare the low-dimensional demonstrations first:

```bash
python prepare_data.py --task can \
  --dataset data/can_low_dim.hdf5 --output data/can
```

The output directory contains normalized trajectories (`train.npz`), min–max normalization bounds (`normalization.npz`), observation and action statistics (`can_lowdim_stats.npz`), and environment metadata (`can.json`). Episode boundaries are preserved. Use `--filter-key train` if you want to select an HDF5 dataset mask.

The normalization must match the base-policy training data. For Transport, the observations and actions include both arms: 59 state dimensions and 14 action dimensions.

## Diffusion Policy

```bash
python pretrain.py --config configs/pretrain/dp_can_h50.yaml \
  --data data/can --output models/can_dp
```

DP uses an MLP noise predictor. Its architecture and training schedule are specified in the YAML file. Training writes `checkpoint/state_EPOCH.pt` and `.hydra/config.yaml`; keep both when copying a policy. The diffusion training schedule has 100 steps. During ExRL, DDIM uses 5 sampling steps for Lift and 20 for the other tasks.

## ACT

```bash
python pretrain.py --config configs/pretrain/act_can_h50.yaml \
  --data data/can --output models/can_act
```

ACT uses low-dimensional observations, no VAE, and zero dropout. Training writes checkpoint directories and a `final/` directory containing the policy, configuration, and observation/action processors. Pass the entire directory to `train.py --base-policy`, for example `models/can_act/final`.

`max_steps` sets the learning-rate schedule, while `stop_after_steps` selects when to stop training. Keeping these separate preserves the learning-rate schedule when using an intermediate checkpoint. ACT bundles contain inference state, not the optimizer state needed to resume pretraining exactly.

## Checkpoint selection

The configurations stop at the checkpoints used to initialize the corresponding adaptation experiments.

| Task | DP epoch, H=32 | DP epoch, H=50 | ACT step, H=32 | ACT step, H=50 |
| --- | ---: | ---: | ---: | ---: |
| Lift | 20,000 | 10,000 | 40 | 30 |
| Can | 20,000 | 20,000 | 300 | 400 |
| Square | 20,000 | 20,000 | 500,000 | 875,000 |
| Transport | 10,000 | 10,000 | 670,000 | 765,000 |
| Tool Hang | 4,000 | 28,000 | 100,000 | 150,000 |

To inspect settings or try a smaller model:

```bash
python pretrain.py --config configs/pretrain/dp_can_h50.yaml \
  --data data/can --output models/can_dp_small \
  --set train.n_epochs=100 --set 'model.network.mlp_dims=[256,256,256]' \
  --print-config
```

Remove `--print-config` to train. Use a fresh output directory for each pretraining run.

## Adroit DP

Adroit uses a temporal U-Net instead of the Robomimic MLP. See [Adroit](adroit.md) for Minari conversion and the `configs/pretrain/adroit_dp.yaml` entry.
