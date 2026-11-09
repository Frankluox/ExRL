#!/usr/bin/env python3
"""Train ExRL or an RL baseline on a configured simulation benchmark."""

import argparse
import os

import yaml

from exrl.config import check_assets, load_config, load_trainer, resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment YAML under configs/")
    parser.add_argument("--data", help="Directory created by prepare_data.py")
    parser.add_argument(
        "--base-policy", help="Base-policy checkpoint; see the benchmark guide for its format"
    )
    parser.add_argument("--output", help="Training output directory")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--resume", help="Resume checkpoint or run directory, depending on the learner"
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Show resolved settings without starting training",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate setup; the checks depend on the selected benchmark",
    )
    cli = parser.parse_args()
    os.environ.setdefault("ROBOMIMIC_ENABLE_LANG", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    config = load_config(cli.config)
    from exrl.experiment import uses_benchmark_runner

    if uses_benchmark_runner(config):
        from exrl.experiment import run

        run(config, cli)
        return
    if cli.resume:
        config.setdefault("training", {})["resume"] = cli.resume
    args, config = resolve(config, cli.data, cli.base_policy, cli.output, cli.seed, cli.set)
    if cli.print_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    check_assets(args, config)
    trainer = load_trainer(config, args)
    if cli.check:
        print("Data, base policy, simulator and trainer imports are ready.")
        return
    trainer.train(args)


if __name__ == "__main__":
    main()
