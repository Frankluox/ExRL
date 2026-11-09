#!/usr/bin/env python3
"""Pretrain DP or ACT on the configured benchmark demonstrations."""

import argparse
import os
from pathlib import Path
import sys

import yaml

from exrl.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML in configs/pretrain/")
    parser.add_argument(
        "--data", required=True, help="Prepared benchmark data directory or Adroit HDF5"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("ROBOMIMIC_ENABLE_LANG", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    config = load_config(args.config)
    for item in args.set:
        if "=" not in item:
            raise ValueError("Overrides use KEY=VALUE")
        key, value = item.split("=", 1)
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown setting: {key}")
        target[parts[-1]] = yaml.safe_load(value)

    if config.get("benchmark") == "adroit":
        from exrl.adroit.pretrain_config import build_parser, build_config
        from exrl.adroit.pretrain import train
        from robomimic.utils.torch_utils import get_torch_device

        data = Path(args.data).expanduser().resolve()
        dataset = data if data.is_file() else data / "expert_v2_robomimic.hdf5"
        output = Path(args.output).expanduser().resolve()
        options = build_parser().parse_args(
            [
                "--task",
                config["task"],
                "--quality",
                "expert",
                "--dataset",
                str(dataset),
                "--output-dir",
                str(output),
                "--config-out",
                str(output / "config.json"),
            ]
        )
        for key, value in config.get("training", {}).items():
            if not hasattr(options, key):
                raise ValueError(f"Unknown Adroit pretraining setting: {key}")
            setattr(options, key, value)
        options.action_normalization = None
        native = build_config(options)
        native.train.cuda = args.device.startswith("cuda")
        if args.print_config:
            print(native.dump())
            return
        if not dataset.is_file():
            raise FileNotFoundError(dataset)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Choose an empty output directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        native.dump(str(output / "config.json"))
        native.lock()
        train(native, get_torch_device(try_to_use_cuda=native.train.cuda))
        return

    data = Path(args.data).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    is_dp = "_target_" in config
    task = config["env_name"] if is_dp else config["task"]
    if is_dp:
        config.update(
            logdir=str(output),
            train_dataset_path=str(data / "train.npz"),
            device=args.device,
            wandb=None,
        )
        config["model"]["device"] = args.device
        config["train_dataset"]["device"] = args.device
    else:
        config.update(
            dataset=str(data / "train.npz"),
            stats_npz=str(data / f"{task}_lowdim_stats.npz"),
            dsrl_normalization_npz=str(data / "normalization.npz"),
            env_meta_json=str(data / f"{task}.json"),
            output_dir=str(output),
            device=args.device,
            disable_wandb=True,
        )
    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    if not (data / "train.npz").is_file():
        raise FileNotFoundError(data / "train.npz")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if is_dp:
        import hydra
        from omegaconf import OmegaConf

        config = OmegaConf.create(config)
        OmegaConf.resolve(config)
        (output / ".hydra").mkdir(exist_ok=True)
        OmegaConf.save(config, output / ".hydra/config.yaml")
        trainer = hydra.utils.get_class(config._target_)(config)
        trainer.run()
    else:
        from exrl.runners.act_pretrain import train

        options = {a.dest for a in train.build_parser()._actions}
        cli = []
        for key, value in config.items():
            if key not in options:
                raise ValueError(f"Unknown ACT pretraining setting: {key}")
            if isinstance(value, bool):
                if key == "use_vae":
                    cli.append("--use_vae" if value else "--no_use_vae")
                elif value:
                    cli.append("--" + key)
            elif value is not None:
                cli.extend(["--" + key, str(value)])
        sys.argv = ["pretrain.py", *cli]
        train.main()


if __name__ == "__main__":
    main()
