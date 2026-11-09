#!/usr/bin/env python3
"""Evaluate an ExRL checkpoint using the settings saved during training."""

import argparse
import json
import os
from pathlib import Path

from exrl.config import check_assets, load_config, load_trainer, resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Optional experiment YAML if the checkpoint was moved")
    parser.add_argument("--data", help="Override the saved data directory")
    parser.add_argument("--base-policy", help="Override the saved base-policy checkpoint")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", help="Output JSON; otherwise print results")
    cli = parser.parse_args()
    checkpoint = Path(cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config_path = cli.config
    if not config_path:
        config_path = next(
            (
                p / "experiment.yaml"
                for p in checkpoint.parents
                if (p / "experiment.yaml").is_file()
            ),
            None,
        )
    if not config_path:
        parser.error(
            "No experiment.yaml beside the run. Supply --config, --data and --base-policy."
        )
    recipe = load_config(config_path)
    if recipe.get("benchmark") == "libero" and recipe.get("method", "").startswith("exrl_"):
        from exrl.libero.evaluation import evaluate_checkpoint

        if cli.base_policy:
            recipe["checkpoint"] = cli.base_policy
        if cli.seed is not None:
            recipe["seed"] = cli.seed
        result = evaluate_checkpoint(checkpoint, recipe, cli.episodes)
        text = json.dumps(result, indent=2) + "\n"
        if cli.output:
            output = Path(cli.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text)
        print(text, end="")
        return
    if recipe.get("benchmark", "robomimic") != "robomimic" or recipe.get("method"):
        parser.error(
            "Use this learner's native evaluation entry; see its benchmark guide in docs/."
        )
    args, config = resolve(
        load_config(config_path),
        data=cli.data,
        base_policy=cli.base_policy,
        seed=cli.seed,
    )
    os.environ.setdefault("ROBOMIMIC_ENABLE_LANG", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    check_assets(args, config)
    trainer = load_trainer(config, args)
    from exrl.evaluation import evaluate

    result = evaluate(trainer, args, config, checkpoint, cli.episodes)
    text = json.dumps(result, indent=2) + "\n"
    if cli.output:
        output = Path(cli.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
