"""Run the benchmark-specific learners with the same public command line."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import yaml

from exrl.config import merge
from exrl.paths import VENDOR


ARGPARSE_RUNNERS = {
    ("online_bc", "robomimic", "act"): "baselines.online_bc.act",
    ("resfit", "robomimic", "dp"): "baselines.resfit.dp.train",
    ("resfit", "robomimic", "act"): "baselines.resfit.act.train",
    ("resfit", "adroit", "dp"): "baselines.resfit.adroit.train",
    ("dice", "robomimic", "dp"): "baselines.dice.dp.train",
    ("dice", "robomimic", "act"): "baselines.dice.act.train",
    ("dice", "adroit", "dp"): "baselines.dice.adroit.train",
    ("ogpo", "robomimic", "dp"): "baselines.ogpo.robomimic.train",
    ("ogpo", "adroit", "dp"): "baselines.ogpo.adroit.train",
    **{
        ("resfit", "libero", family): f"baselines.resfit.libero.{family}.train"
        for family in ("pi0", "pi05", "pi0fast", "cosmos")
    },
    ("exrl_q", "adroit", "dp"): "exrl.adroit.q.train",
    ("exrl_sac", "adroit", "dp"): "exrl.adroit.sac.train",
}


def uses_benchmark_runner(config):
    return config.get("method") is not None or config.get("benchmark", "robomimic") != "robomimic"


def prepare_config(config, cli):
    config = merge({}, config)
    for key, value in (
        ("data", cli.data),
        ("checkpoint", cli.base_policy),
        ("output", cli.output),
        ("seed", cli.seed),
    ):
        if value is not None:
            config[key] = value
    for item in cli.set:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError("Overrides use KEY=VALUE")
        key = key.removeprefix("training.")
        target = config.setdefault("training", {})
        # Native Hydra learners consume dotted paths; LIBERO consumes nested mappings.
        if config.get("benchmark") == "libero" and config["method"].startswith("exrl_"):
            parts = key.split(".")
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            key = parts[-1]
        target[key] = yaml.safe_load(value)
    config.setdefault("seed", 42)
    config.setdefault("output", f"outputs/{config['task']}_{config['method']}")
    for key in ("data", "checkpoint", "output"):
        if config.get(key):
            config[key] = str(Path(config[key]).expanduser().resolve())
    if cli.resume:
        config["resume"] = cli.resume
    return config


def build_arguments(config, parser):
    """Translate assets once; leave each learner's numerical options intact."""
    options = {a.dest: a for a in parser._actions if a.dest != "help"}
    values = dict(config.get("training", {}))
    task = config["task"]
    data = Path(config.get("data", "data/" + task)).expanduser().resolve()
    metadata = data / "dataset_info.json"
    dataset = json.loads(metadata.read_text()).get("source") if metadata.exists() else None
    dataset = dataset or str(
        data
        / ("expert_v2_robomimic.hdf5" if config["benchmark"] == "adroit" else "demonstrations.hdf5")
    )
    defaults = {
        "agent": config.get("checkpoint"),
        "checkpoint": config.get("checkpoint"),
        "checkpoint_dir": config.get("checkpoint"),
        "env_name": f"AdroitHand{task.title()}-v1"
        if config["method"].startswith("exrl_") and config["benchmark"] == "adroit"
        else task,
        "task": task,
        "seed": config["seed"],
        "max_chunk_len": config["chunk_size"],
        "base_horizon_steps": config["chunk_size"],
        "expected_horizon_steps": config["chunk_size"],
        "dsrl_official_root": str(VENDOR),
        "vendor_root": str(VENDOR),
        "env_meta_json": str(data / f"{task}.json"),
        "dsrl_normalization_npz": str(data / "normalization.npz"),
        "normalization_path": str(data / "normalization.npz"),
        "stats_npz": str(data / f"{task}_lowdim_stats.npz"),
        "bc_offline_demo_hdf5": dataset,
        "offline_cache_dir": str(Path(config["output"]) / "offline_cache"),
        "stats_npz_override": str(data / f"{task}_lowdim_stats.npz"),
        "official_demo_npz": str(data / "train.npz"),
        "reward_hdf5": dataset,
        "demo_hdf5": dataset,
        "minari_hdf5": str(data / "minari.hdf5"),
        "norm_checkpoint_dir": config.get("checkpoint"),
        "output": config["output"],
        "output_dir": config["output"],
        "log_dir": config["output"],
    }
    for key, value in defaults.items():
        if key in options and value is not None:
            values.setdefault(key, value)
    for key, value in config.get("assets", {}).items():
        if key not in options:
            raise ValueError(f"Unknown asset option: {key}")
        values[key] = str(Path(value).expanduser())
    if config.get("resume"):
        key = "resume_checkpoint" if "resume_checkpoint" in options else "resume"
        values[key] = config["resume"]
    if config["method"].startswith("exrl_") and config["benchmark"] == "adroit":
        count = (
            float(values["utd"]) * int(values.get("n_envs", 4)) * int(values.get("train_freq", 1))
        )
        if count <= 0 or count != int(count):
            raise ValueError("UTD * n_envs * train_freq must be a positive integer")
        values["utd"] = int(count)
    unknown = values.keys() - options.keys()
    if unknown:
        raise ValueError(f"Unknown training options: {sorted(unknown)}")
    argv = []
    for key, value in values.items():
        if value is None:
            continue
        action = options[key]
        flag = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(flag)
            elif action.default:
                raise ValueError(f"{key} cannot be disabled by this learner")
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                argv.append(flag)
        else:
            if isinstance(value, (list, tuple)):
                argv.extend([flag, *map(str, value)])
            else:
                argv.extend((flag, str(value)))
    # Catch misspelled options, invalid choices, and missing assets before launch.
    parser.parse_args(argv)
    return argv


def run(config, cli):
    config = prepare_config(config, cli)
    if cli.print_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    if not cli.check:
        if not config.get("checkpoint"):
            raise ValueError("Supply --base-policy with a pretrained checkpoint")
        if not Path(config["checkpoint"]).expanduser().exists():
            raise FileNotFoundError(config["checkpoint"])
        output = Path(config["output"]).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        (output / "experiment.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    if config["method"] == "dsrl" and config["benchmark"] == "adroit":
        from baselines.dsrl.adroit.launch import run as launch

        return launch(config, cli)
    if config.get("native_config"):
        from baselines.launch import run as launch

        return launch(config, cli)
    if config["benchmark"] == "libero" and config["method"].startswith("exrl_"):
        from exrl.libero.train import run as launch

        return launch(config, cli)
    key = (config["method"], config["benchmark"], config["base_policy"])
    if key not in ARGPARSE_RUNNERS:
        raise ValueError(f"Unsupported method/benchmark/base policy: {key}")
    module = importlib.import_module(ARGPARSE_RUNNERS[key])
    argv = build_arguments(config, module.build_parser())
    if cli.check:
        print("Learner imports and configuration are valid.")
        return
    original_argv = sys.argv
    try:
        sys.argv = [module.__file__, *argv]
        module.main()
    finally:
        sys.argv = original_argv
