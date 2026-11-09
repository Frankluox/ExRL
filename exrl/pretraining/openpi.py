"""Fine-tune generalist base policies with OpenPI's native JAX trainer."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def validate_recipe(recipe):
    if recipe["benchmark"] not in ("libero", "real_robot"):
        raise ValueError("OpenPI recipes support LIBERO and robot demonstrations")
    if recipe["base_policy"] not in ("pi0", "pi05", "pi0fast"):
        raise ValueError("Unknown OpenPI model family")
    if recipe["benchmark"] == "real_robot" and recipe["base_policy"] != "pi05":
        raise ValueError("The robot fine-tuning recipes currently use pi05")
    repo_id = Path(recipe["repo_id"])
    if repo_id.is_absolute() or len(repo_id.parts) != 2 or ".." in repo_id.parts:
        raise ValueError("repo_id must be namespace/dataset")
    if recipe["training"]["trainable"] not in ("all", "action_expert"):
        raise ValueError("trainable must be all or action_expert")
    for key in ("num_train_steps", "batch_size", "save_interval", "fsdp_devices"):
        if int(recipe["training"][key]) <= 0:
            raise ValueError(f"{key} must be positive")


def build_config(recipe, output, base_policy=None, resume=False):
    """Construct a native config without loading model parameters."""
    from openpi.models import pi0_config, pi0_fast
    from openpi.training import config as c, optimizer, weight_loaders

    validate_recipe(recipe)
    family = recipe["base_policy"]
    horizon = int(recipe["chunk_size"])
    settings = recipe["training"]
    if family == "pi0fast":
        model = pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=horizon, max_token_len=320)
    else:
        model = pi0_config.Pi0Config(
            pi05=family == "pi05",
            action_horizon=horizon,
            discrete_state_input=recipe["benchmark"] == "real_robot",
        )
    if recipe["benchmark"] == "libero":
        data = c.LeRobotLiberoDataConfig(
            repo_id=recipe["repo_id"],
            base_config=c.DataConfig(prompt_from_task=True),
            extra_delta_transform=family in ("pi0", "pi0fast"),
        )
    else:
        import importlib

        module, name = recipe["data_factory"].split(":", 1)
        factory = getattr(importlib.import_module(module), name)
        data = factory(
            repo_id=recipe["repo_id"],
            base_config=c.DataConfig(prompt_from_task=True),
            **recipe.get("data_options", {}),
        )
    extra = {}
    if settings["trainable"] == "action_expert":
        from flax import nnx
        from openpi.shared.nnx_utils import PathRegex

        if family != "pi05":
            raise ValueError("The action-expert filter is defined for pi05")
        extra["freeze_filter"] = nnx.Not(
            PathRegex(
                r"(?:PaliGemma/llm/.*_1(?:/.*)?|(?:action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/.*)"
            )
        )
    if base_policy is not None:
        params = Path(base_policy).expanduser().resolve()
        if (params / "params").is_dir():
            params = params / "params"
        extra["weight_loader"] = weight_loaders.CheckpointWeightLoader(str(params))
    output = Path(output).expanduser().resolve()
    return c.TrainConfig(
        name=recipe["name"],
        project_name="exrl",
        exp_name="run",
        model=model,
        data=data,
        assets_base_dir=str(output / "assets"),
        checkpoint_base_dir=str(output / "checkpoints"),
        seed=int(recipe["seed"]),
        batch_size=int(settings["batch_size"]),
        num_train_steps=int(settings["num_train_steps"]),
        num_workers=int(settings["num_workers"]),
        save_interval=int(settings["save_interval"]),
        keep_period=int(settings["save_interval"]),
        log_interval=int(settings["log_interval"]),
        fsdp_devices=int(settings["fsdp_devices"]),
        ema_decay=settings["ema_decay"],
        resume=resume,
        overwrite=False,
        wandb_enabled=False,
        lr_schedule=optimizer.CosineDecaySchedule(**recipe["learning_rate"]),
        optimizer=optimizer.AdamW(**recipe["optimizer"]),
        policy_metadata=dict(family=family, horizon=horizon, benchmark=recipe["benchmark"]),
        **extra,
    )


def load_script(upstream, name):
    path = Path(upstream).expanduser().resolve() / "scripts" / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"Set --upstream to the OpenPI checkout: {path}")
    spec = importlib.util.spec_from_file_location(f"exrl_openpi_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_statistics(native, upstream):
    from openpi.shared import normalize
    import numpy as np

    script = load_script(upstream, "compute_norm_stats")
    data = native.data.create(native.assets_dirs, native.model)
    loader, _ = script.create_torch_dataloader(
        data, native.model.action_horizon, native.batch_size, native.model, native.num_workers
    )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in loader:
        for key in stats:
            stats[key].update(np.asarray(batch[key]))
    target = native.assets_dirs / data.asset_id
    normalize.save(target, {key: value.get_statistics() for key, value in stats.items()})
    print(f"Saved normalization statistics to {target}")


def run(recipe, args):
    import yaml

    validate_recipe(recipe)
    if args.print_config:
        print(yaml.safe_dump(recipe, sort_keys=False))
        return
    if not args.upstream:
        raise ValueError("Supply --upstream /path/to/openpi and use its Python environment")
    data = Path(args.data).expanduser().resolve()
    if not (data / recipe["repo_id"] / "meta/info.json").is_file():
        raise FileNotFoundError(data / recipe["repo_id"] / "meta/info.json")
    os.environ["HF_LEROBOT_HOME"] = str(data)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")
    native = build_config(recipe, args.output, args.base_policy, args.resume)
    if args.norm_stats:
        compute_statistics(native, args.upstream)
        return
    if not args.resume and (
        args.base_policy is None or not Path(args.base_policy).expanduser().is_dir()
    ):
        raise ValueError("Supply a local initial OpenPI checkpoint with --base-policy")
    if not (native.assets_dirs / recipe["repo_id"] / "norm_stats.json").is_file():
        raise FileNotFoundError("Run the same command with --norm-stats before fine-tuning")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "pretraining.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    train = load_script(args.upstream, "train")
    train.main(native)
