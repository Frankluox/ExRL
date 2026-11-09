#!/usr/bin/env python3
"""Generate robomimic Diffusion Policy configs for Adroit low-dim datasets."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from robomimic.config import config_factory


TASK_TO_ENV = {
    "door": "AdroitHandDoor-v1",
    "hammer": "AdroitHandHammer-v1",
    "pen": "AdroitHandPen-v1",
    "relocate": "AdroitHandRelocate-v1",
}


def build_config(args):
    cfg = config_factory("diffusion_policy")
    exp_name = args.name or (
        f"adroit_{args.task}_{args.quality}_dp_"
        f"o{args.observation_horizon}_a{args.action_horizon}_p{args.prediction_horizon}_s{args.seed}"
    )
    dataset_path = args.dataset or str(
        Path(args.dataset_root) / args.task / f"{args.quality}_v2_robomimic.hdf5"
    )
    output_dir = args.output_dir or str(Path(args.ckpt_root) / args.task / args.quality)

    with cfg.values_unlocked():
        cfg.experiment.name = exp_name
        cfg.experiment.validate = args.validate
        cfg.experiment.logging.terminal_output_to_txt = True
        cfg.experiment.logging.log_tb = True
        cfg.experiment.save.every_n_epochs = args.save_every_n_epochs
        cfg.experiment.save.on_best_rollout_success_rate = args.rollout
        cfg.experiment.epoch_every_n_steps = args.epoch_steps
        cfg.experiment.validation_epoch_every_n_steps = args.validation_epoch_steps
        cfg.experiment.env = None
        cfg.experiment.render = False
        cfg.experiment.render_video = False
        cfg.experiment.rollout.enabled = args.rollout
        cfg.experiment.rollout.n = args.rollout_n
        cfg.experiment.rollout.horizon = args.rollout_horizon
        cfg.experiment.rollout.rate = args.rollout_rate
        cfg.experiment.rollout.terminate_on_success = False

        cfg.train.data = [{"path": dataset_path}]
        cfg.train.output_dir = output_dir
        cfg.train.num_data_workers = args.num_data_workers
        cfg.train.hdf5_cache_mode = args.hdf5_cache_mode
        cfg.train.hdf5_use_swmr = True
        cfg.train.hdf5_load_next_obs = False
        cfg.train.hdf5_normalize_obs = args.normalize_obs
        cfg.train.hdf5_filter_key = "train" if args.validate else None
        cfg.train.hdf5_validation_filter_key = "valid" if args.validate else None
        cfg.train.seq_length = args.prediction_horizon
        cfg.train.frame_stack = args.observation_horizon
        cfg.train.dataset_keys = ["actions", "rewards", "dones"]
        cfg.train.action_keys = ["actions"]
        cfg.train.action_config = {"actions": {"normalization": args.action_normalization}}
        cfg.train.cuda = True
        cfg.train.batch_size = args.batch_size
        cfg.train.num_epochs = args.num_epochs
        cfg.train.seed = args.seed

        cfg.algo.horizon.observation_horizon = args.observation_horizon
        cfg.algo.horizon.action_horizon = args.action_horizon
        cfg.algo.horizon.prediction_horizon = args.prediction_horizon
        cfg.algo.unet.diffusion_step_embed_dim = args.diffusion_step_embed_dim
        cfg.algo.unet.down_dims = args.down_dims
        cfg.algo.optim_params.policy.learning_rate.warmup_steps = args.warmup_steps
        cfg.algo.ddpm.enabled = not args.ddim
        cfg.algo.ddim.enabled = args.ddim
        cfg.algo.ddim.num_inference_timesteps = args.ddim_steps

        cfg.observation.modalities.obs.low_dim = ["flat"]
        cfg.observation.modalities.obs.rgb = []
        cfg.observation.modalities.obs.depth = []
        cfg.observation.modalities.obs.scan = []
        cfg.observation.modalities.goal.low_dim = []
        cfg.observation.modalities.goal.rgb = []
        cfg.observation.modalities.goal.depth = []
        cfg.observation.modalities.goal.scan = []

    return cfg


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASK_TO_ENV, required=True)
    parser.add_argument("--quality", choices=("human", "cloned", "expert"), required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("ADROIT_ROBOMIMIC_DATASETS", "datasets/adroit_dp/robomimic"),
    )
    parser.add_argument(
        "--ckpt-root", default=os.environ.get("ADROIT_CKPT_DIR", "checkpoints/adroit_dp")
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config-out", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--epoch-steps", type=int, default=1000)
    parser.add_argument("--validation-epoch-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hdf5-cache-mode", choices=("all", "low_dim", "none"), default="all")
    parser.add_argument("--num-data-workers", type=int, default=0)
    parser.add_argument("--observation-horizon", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--prediction-horizon", type=int, default=32)
    parser.add_argument("--diffusion-step-embed-dim", type=int, default=128)
    parser.add_argument("--down-dims", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--save-every-n-epochs", type=int, default=5)
    parser.add_argument("--normalize-obs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-normalization", choices=("min_max", "none"), default="none")
    parser.add_argument("--ddim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ddim-steps", type=int, default=5)
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rollout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rollout-n", type=int, default=20)
    parser.add_argument("--rollout-horizon", type=int, default=200)
    parser.add_argument("--rollout-rate", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.hdf5_cache_mode == "none":
        args.hdf5_cache_mode = None
    if args.action_normalization == "none":
        args.action_normalization = None

    cfg = build_config(args)
    out = Path(args.config_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
