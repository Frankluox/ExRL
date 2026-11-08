"""Load experiment settings without importing PyTorch or a simulator."""

from __future__ import annotations

import argparse
import copy
import importlib
import math
from pathlib import Path

import yaml

from exrl.paths import VENDOR


def merge(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path, _parents=()):
    """Resolve YAML defaults relative to the file that declares them."""
    path = Path(path).expanduser().resolve()
    if path in _parents:
        raise ValueError(f"Circular config defaults: {path}")
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    result = {}
    for parent in config.pop("defaults", []):
        result = merge(result, load_config(path.parent / parent, (*_parents, path)))
    return merge(result, config)


def save_config(config, path):
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False))


def runner_name(config):
    base, algorithm = config["base_policy"], config["algorithm"]
    if base not in ("dp", "act") or algorithm not in ("q", "sac"):
        raise ValueError("Choose base_policy: dp / act and algorithm: q / sac")
    if config.get("demonstration_replay"):
        if (base, algorithm) != ("dp", "q"):
            raise ValueError("Demonstration replay is configured for DP ExRL-Q")
        return "dp_demo"
    return f"{base}_{algorithm}"


def parser_for(config):
    module = importlib.import_module(f"exrl.runners.{runner_name(config)}.options")
    return module.build_parser()


def apply_overrides(config, overrides):
    config = copy.deepcopy(config)
    options = {a.dest for a in parser_for(config)._actions}
    for item in overrides:
        if "=" not in item:
            raise ValueError("Overrides use KEY=VALUE, for example --set n_envs=4")
        key, value = item.split("=", 1)
        value = yaml.safe_load(value)
        if key.startswith("training."):
            key = key[len("training.") :]
        if key in options:
            config.setdefault("training", {})[key] = value
        elif key in ("seed", "chunk_size"):
            config[key] = value
        else:
            raise ValueError(f"Unknown setting: {key}")
    return config


def resolve(config, data=None, base_policy=None, output=None, seed=None, overrides=()):
    """Translate user settings into the runner's validated arguments."""
    config = apply_overrides(config, overrides)
    parser = parser_for(config)
    actions = {a.dest: a for a in parser._actions if a.dest != "help"}
    params = {key: action.default for key, action in actions.items()}
    training = config.get("training", {})
    unknown = set(training) - set(params)
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    params.update(training)
    task = config["task"]
    params.update(env_name=task, max_chunk_len=config["chunk_size"])
    if "task" in params:
        params["task"] = task

    # Each collection iteration returns n_envs replay transitions.
    utd = float(training.get("utd", 0.25 if config["algorithm"] == "q" else 1.0))
    count = utd * params["n_envs"] * params["train_freq"]
    if not math.isfinite(count) or count <= 0 or not math.isclose(count, round(count)):
        raise ValueError(
            "utd * n_envs * train_freq must be a positive integer; use n_envs=4 for UTD=0.25"
        )
    params["utd"] = int(round(count))

    data = data or config.get("data")
    base_policy = base_policy or config.get("checkpoint")
    output = (
        output
        or config.get("output")
        or f"outputs/{task}_{config['base_policy']}_{config['algorithm']}"
    )
    chosen_seed = config.get("seed", 42) if seed is None else seed
    config.update(seed=chosen_seed, output=str(Path(output).expanduser().resolve()))
    if data:
        data = Path(data).expanduser().resolve()
        config["data"] = str(data)
        params.update(
            env_meta_json=str(data / f"{task}.json"),
            dsrl_normalization_npz=str(data / "normalization.npz"),
            stats_npz_override=str(data / f"{task}_lowdim_stats.npz"),
        )
        for key in (
            "bc_offline_demo_npz",
            "bc_base_dataset_path",
            "demo_prefill_npz",
            "offline_demo_npz",
        ):
            if params.get(key) == "@demonstrations":
                params[key] = str(data / "train.npz")
    if base_policy:
        config["checkpoint"] = str(Path(base_policy).expanduser().resolve())
        params["agent"] = config["checkpoint"]
    params.update(
        vendor_root=str(VENDOR),
        output=config["output"],
        log_dir=config["output"],
        seed=chosen_seed,
    )
    for key, value in params.items():
        action = actions[key]
        if action.choices is not None and value not in action.choices:
            raise ValueError(f"{key}={value!r}; expected one of {action.choices}")
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
        if value is not None and action.type in (int, float, str):
            params[key] = action.type(value)
    params["public_config"] = config
    return argparse.Namespace(**params), config


def check_assets(args, config):
    required = [
        "agent",
        "env_meta_json",
        "dsrl_normalization_npz",
        "stats_npz_override",
    ]
    required += [
        key
        for key in (
            "bc_offline_demo_npz",
            "bc_base_dataset_path",
            "demo_prefill_npz",
            "offline_demo_npz",
        )
        if getattr(args, key, None)
    ]
    for key in required:
        value = getattr(args, key, None)
        if not value or not Path(value).exists():
            raise FileNotFoundError(
                f"Missing {key}: {value}. Supply --data and --base-policy."
            )
    if config["base_policy"] == "dp":
        metadata = Path(args.agent).parent.parent / ".hydra/config.yaml"
        if not metadata.is_file():
            raise FileNotFoundError(
                f"Keep the DP training config next to checkpoint/: {metadata}"
            )
        cfg = yaml.safe_load(metadata.read_text())
        model = cfg.get("model", {})
        schedule = (
            model.get("scheduler_num_train_timesteps")
            if model.get("use_robomimic_scheduler")
            else None
        )
        schedule = (
            schedule or cfg.get("denoising_steps") or model.get("denoising_steps")
        )
        if isinstance(schedule, str) and schedule.startswith("${"):
            schedule = cfg.get(schedule[2:-1])
        if not schedule or int(schedule) != config.get("diffusion_training_steps", 100):
            raise ValueError(
                f"Expected the DP training diffusion schedule to have 100 steps; found {schedule}"
            )
    import numpy as np

    with np.load(args.stats_npz_override) as stats:
        state_dim = sum(
            np.asarray(stats[k + "_mean"]).size for k in args.lowdim_keys.split(",")
        )
        action_dim = np.asarray(stats["action_mean"]).size
    if config["task"] == "transport" and (state_dim, action_dim) != (59, 14):
        raise ValueError(
            "Transport requires 59 state dimensions and 14 action dimensions"
        )
    import mujoco
    import robosuite

    expected = config["runtime"]
    if (
        str(mujoco.__version__) != expected["mujoco"]
        or str(mujoco.mj_versionString()) != expected["mujoco"]
    ):
        raise RuntimeError(
            f"This config uses MuJoCo {expected['mujoco']}; see docs/experiments.md"
        )
    if str(robosuite.__version__) != expected["robosuite"]:
        raise RuntimeError(
            f"This config uses Robosuite {expected['robosuite']}; see docs/experiments.md"
        )


def load_trainer(config, args):
    trainer = importlib.import_module(f"exrl.runners.{runner_name(config)}.train")
    if hasattr(trainer, "_resolve_task_defaults"):
        trainer._resolve_task_defaults(args)
    trainer._validate_task_alignment(args)
    if hasattr(trainer, "_validate_bc_args"):
        trainer._validate_bc_args(args)
    return trainer
