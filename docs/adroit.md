# Adroit

The Adroit experiments adapt a state-conditioned diffusion policy on Door, Hammer, Pen, and Relocate. The base policy predicts 32 actions; ExRL chooses an execution length from 1 to 32. Episodes use a 200-step limit and sparse success rewards.

## Installation

Use a separate Python 3.10 environment because Adroit and Robomimic require different Gymnasium and MuJoCo versions:

```bash
conda create -n exrl-adroit python=3.10
conda activate exrl-adroit
pip install -r requirements/adroit.txt
pip install -e .
export MUJOCO_GL=egl
```

## Demonstrations and base policy

The base models use the D4RL expert-v2 datasets distributed through Minari. Convert a local dataset to the Robomimic format used by the DP trainer:

```bash
python -m exrl.adroit.prepare_data \
  --task door --quality expert --no-download --output-root data/adroit
```

If the dataset is not installed, obtain it with Minari first, or omit `--no-download` to let Minari fetch it. The conversion writes `data/adroit/door/expert_v2_robomimic.hdf5`.

```bash
python pretrain.py --config configs/pretrain/adroit_dp.yaml \
  --data data/adroit/door --output models/adroit_door
```

Set `--set task=hammer` or `--set task=pen` and use the corresponding data directory for the other two short-training recipes. Relocate uses `--set task=relocate --set training.num_epochs=10`; the other tasks use two epochs. Each epoch contains 1,000 updates. Checkpoints include the U-Net architecture and normalization statistics, so retain the complete `.pth` file.

## Online adaptation

```bash
python train.py --config configs/adroit/door_q.yaml \
  --data data/adroit/door --base-policy models/adroit_door/PATH_TO_MODEL.pth \
  --output outputs/adroit_door_q
```

Choose `door_sac.yaml`, `door_online_bc.yaml`, or `door_demo_bc.yaml` for SAC, online BC, or online BC with demonstrations. Equivalent recipes exist for all four tasks. The demonstration-augmented BC recipe samples half of each imitation batch from offline demonstrations.

ExRL training performs periodic evaluation and saves full training checkpoints. `--resume PATH` continues from a saved checkpoint. The simulator starts a new episode when resuming; replay, learned networks, optimizer state, and counters are restored.

## RL baselines

DSRL, ResFiT, OGPO, and DICE-RL configurations are in `configs/baselines/adroit/`. They use the same DP `.pth` checkpoint.

ResFiT and DICE-RL additionally use the original Minari HDF5 file for explicit success and termination labels. Create `data/adroit/door/minari.hdf5` as a symlink to that dataset's `main_data.hdf5`, or set `assets.minari_hdf5` in a copied YAML. These labels are necessary: Adroit's dense dataset rewards do not uniquely identify task success. The converted demonstrations must come from the same Minari dataset and preserve its episode order.

The native trainers evaluate during learning. ResFiT and DICE-RL also support `--resume PATH --set eval_only=true` with the original experiment configuration.
