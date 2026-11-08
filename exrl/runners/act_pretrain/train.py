from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from exrl.paths import ROOT as REPO_ROOT

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from exrl.runners.act_pretrain.policy_loader import (
    ACT_TASK_PRESETS,
    infer_lowdim_dims_from_stats,
    resolve_act_task_setup,
)
from exrl.runners.act_pretrain.npz_dataset import (
    DSRLNPZACTDataset,
    compute_dsrl_dataset_stats,
    infer_dsrl_feature_shapes,
    inspect_dsrl_npz,
)
from exrl.runners.act_pretrain.hdf5_dataset import (
    ACTION,
    OBS_ENV_STATE,
    OBS_STATE,
    RobomimicACTDataset,
    compute_dataset_stats,
    ensure_dataset_has_obs,
    infer_feature_shapes,
    inspect_robomimic_hdf5,
)


def parse_norm_mode(mode: str) -> NormalizationMode:
    mode = str(mode).upper()
    if mode == "MEAN_STD":
        return NormalizationMode.MEAN_STD
    if mode == "IDENTITY":
        return NormalizationMode.IDENTITY
    raise ValueError(f"unsupported normalization mode: {mode}")


def select_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_output_dir(path_like: str) -> Path:
    raw = Path(path_like).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (REPO_ROOT / raw).resolve()


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    total_steps: int,
    warmup_steps: int,
    min_lr: float,
):
    name = str(scheduler_name).strip().lower()
    if name in ("", "none") or total_steps <= 0:
        return None
    if name != "cosine":
        raise ValueError(f"unsupported lr scheduler: {scheduler_name}")

    warmup_steps = max(int(warmup_steps), 0)
    min_lr = float(min_lr)
    base_lr = float(optimizer.param_groups[0]["lr"])
    if warmup_steps >= total_steps:
        start_factor = 1.0 if base_lr <= 0 else min(1.0, max(min_lr / base_lr, 1e-4))
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=max(total_steps, 1),
        )

    if warmup_steps > 0:
        start_factor = 1.0 if base_lr <= 0 else min(1.0, max(min_lr / base_lr, 1e-4))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=min_lr,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps, 1),
        eta_min=min_lr,
    )


def save_training_bundle(
    save_dir: Path, policy: ACTPolicy, preprocessor, postprocessor, run_summary: dict
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)
    with open(save_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train low-dim ACT on robomimic datasets."
    )
    parser.add_argument(
        "--task", type=str, default="square", choices=sorted(ACT_TASK_PRESETS.keys())
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Optional dataset override."
    )
    parser.add_argument("--stats_npz", type=str, default=None)
    parser.add_argument("--dsrl_normalization_npz", type=str, default=None)
    parser.add_argument("--env_meta_json", type=str, default=None)
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Optional output dir override."
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Optional ACT checkpoint to initialize from.",
    )
    parser.add_argument(
        "--state_obs_keys", type=str, default="", help="Comma-separated override."
    )
    parser.add_argument(
        "--env_obs_keys", type=str, default="", help="Comma-separated override."
    )
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--n_action_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--auto_extract_obs", action="store_true")
    parser.add_argument("--extract_output_name", type=str, default=None)
    parser.add_argument("--extract_done_mode", type=int, default=2)
    parser.add_argument("--extract_copy_rewards", action="store_true")
    parser.add_argument("--extract_copy_dones", action="store_true")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument(
        "--stop_after_steps",
        type=int,
        default=0,
        help="Stop early while retaining the full learning-rate schedule",
    )
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--save_every_epochs", type=int, default=None)
    parser.add_argument("--grad_clip_norm", type=float, default=10.0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--compile_model", action="store_true")

    parser.add_argument("--use_vae", action="store_true", default=True)
    parser.add_argument("--no_use_vae", action="store_false", dest="use_vae")
    parser.add_argument("--kl_weight", type=float, default=10.0)
    parser.add_argument("--dim_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--n_encoder_layers", type=int, default=4)
    parser.add_argument("--n_decoder_layers", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--n_vae_encoder_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler", type=str, default="none", choices=["none", "cosine"]
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_min", type=float, default=1e-5)

    parser.add_argument(
        "--state_norm", type=str, default="mean_std", choices=["mean_std", "identity"]
    )
    parser.add_argument(
        "--env_norm", type=str, default="mean_std", choices=["mean_std", "identity"]
    )
    parser.add_argument(
        "--action_norm", type=str, default="mean_std", choices=["mean_std", "identity"]
    )

    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="act-robomimic")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_steps is not None and args.max_epochs is not None:
        raise ValueError("Please pass at most one of --max_steps and --max_epochs.")
    set_seed(args.seed)
    device = select_device(args.device)

    setup = resolve_act_task_setup(
        task=args.task,
        dataset=args.dataset,
        state_obs_keys=args.state_obs_keys,
        env_obs_keys=args.env_obs_keys,
        chunk_size=args.chunk_size,
        output_dir=args.output_dir,
    )
    for key in ("stats_npz", "dsrl_normalization_npz", "env_meta_json"):
        if getattr(args, key):
            setup[key] = getattr(args, key)
    if not setup["dataset"] or not Path(setup["dataset"]).exists():
        raise FileNotFoundError(
            f"dataset path not found for task={args.task}. "
            f"Resolved dataset={setup['dataset']}. Please pass --dataset explicitly."
        )

    output_dir = resolve_output_dir(setup["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    state_obs_keys = list(setup["state_obs_keys"])
    env_obs_keys = list(setup["env_obs_keys"])
    if not state_obs_keys:
        raise ValueError("state_obs_keys cannot be empty for low-dim ACT training")

    use_state_as_env_when_missing = len(env_obs_keys) == 0
    dataset_path = Path(setup["dataset"]).expanduser().resolve()
    dataset_kind = "robomimic_hdf5"
    observation_space_kind = "raw_lowdim"
    action_space_kind = "raw_action"
    lowdim_keys = list(dict.fromkeys(state_obs_keys + env_obs_keys))
    lowdim_dims = None

    if dataset_path.suffix.lower() == ".npz":
        if not setup["stats_npz"]:
            raise FileNotFoundError(
                f"stats_npz is required for DSRL npz training on task={args.task}. "
                "Please pass --stats_npz-compatible task preset or override."
            )
        lowdim_dims = infer_lowdim_dims_from_stats(setup["stats_npz"], lowdim_keys)
        dataset_info = inspect_dsrl_npz(dataset_path)
        feature_shapes = infer_dsrl_feature_shapes(
            dataset_path,
            state_obs_keys=state_obs_keys,
            env_obs_keys=env_obs_keys,
            lowdim_dims=lowdim_dims,
            use_state_as_env_when_missing=use_state_as_env_when_missing,
        )
        dataset_stats = compute_dsrl_dataset_stats(
            dataset_path,
            state_obs_keys=state_obs_keys,
            env_obs_keys=env_obs_keys,
            lowdim_dims=lowdim_dims,
            use_state_as_env_when_missing=use_state_as_env_when_missing,
        )
        action_delta_indices = list(range(int(setup["chunk_size"])))
        dataset = DSRLNPZACTDataset(
            dataset_path,
            state_obs_keys=state_obs_keys,
            env_obs_keys=env_obs_keys,
            lowdim_dims=lowdim_dims,
            action_delta_indices=action_delta_indices,
            use_state_as_env_when_missing=use_state_as_env_when_missing,
        )
        resolved_dataset = dataset_path
        dataset_kind = "dsrl_npz"
        observation_space_kind = "dsrl_normalized"
        action_space_kind = "dsrl_normalized"
    else:
        dataset_with_obs = ensure_dataset_has_obs(
            dataset_path=dataset_path,
            auto_extract_obs=args.auto_extract_obs,
            extract_output_name=args.extract_output_name,
            camera_names=[],
            camera_height=84,
            camera_width=84,
            done_mode=args.extract_done_mode,
            copy_rewards=args.extract_copy_rewards,
            copy_dones=args.extract_copy_dones,
        )
        dataset_info = inspect_robomimic_hdf5(dataset_with_obs)
        if not dataset_info.has_obs:
            raise ValueError(f"{dataset_with_obs} still has no obs group")

        available_keys = set(dataset_info.obs_keys)
        for key in state_obs_keys + env_obs_keys:
            if key not in available_keys:
                raise KeyError(
                    f"obs key '{key}' not found in dataset. Available: {dataset_info.obs_keys}"
                )
        feature_shapes = infer_feature_shapes(
            hdf5_path=dataset_with_obs,
            state_obs_keys=state_obs_keys,
            env_obs_keys=env_obs_keys,
            image_obs_keys=[],
        )
        dataset_stats = compute_dataset_stats(
            hdf5_path=dataset_with_obs,
            state_obs_keys=state_obs_keys,
            env_obs_keys=env_obs_keys,
            image_obs_keys=[],
            action_key="actions",
            include_image_stats=False,
            use_state_as_env_when_missing=use_state_as_env_when_missing,
        )
        action_delta_indices = list(range(int(setup["chunk_size"])))
        dataset = RobomimicACTDataset(
            hdf5_path=dataset_with_obs,
            state_obs_keys=state_obs_keys,
            env_obs_keys=env_obs_keys,
            image_obs_keys=[],
            action_delta_indices=action_delta_indices,
            action_key="actions",
            use_state_as_env_when_missing=use_state_as_env_when_missing,
        )
        resolved_dataset = dataset_with_obs

    if use_state_as_env_when_missing:
        feature_shapes[OBS_ENV_STATE] = feature_shapes[OBS_STATE]

    input_features = {
        OBS_STATE: PolicyFeature(
            type=FeatureType.STATE, shape=feature_shapes[OBS_STATE]
        ),
    }
    if env_obs_keys or use_state_as_env_when_missing:
        input_features[OBS_ENV_STATE] = PolicyFeature(
            type=FeatureType.ENV, shape=feature_shapes[OBS_ENV_STATE]
        )
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=feature_shapes[ACTION]),
    }

    n_action_steps = (
        args.n_action_steps
        if args.n_action_steps is not None
        else int(setup["chunk_size"])
    )
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        device=device.type,
        n_obs_steps=1,
        chunk_size=int(setup["chunk_size"]),
        n_action_steps=int(n_action_steps),
        use_vae=bool(args.use_vae),
        kl_weight=float(args.kl_weight),
        dim_model=int(args.dim_model),
        n_heads=int(args.n_heads),
        dim_feedforward=int(args.dim_feedforward),
        n_encoder_layers=int(args.n_encoder_layers),
        n_decoder_layers=int(args.n_decoder_layers),
        latent_dim=int(args.latent_dim),
        n_vae_encoder_layers=int(args.n_vae_encoder_layers),
        dropout=float(args.dropout),
        optimizer_lr=float(args.lr),
        optimizer_lr_backbone=float(args.lr_backbone),
        optimizer_weight_decay=float(args.weight_decay),
        pretrained_backbone_weights=None,
    )
    cfg.normalization_mapping["STATE"] = parse_norm_mode(args.state_norm)
    cfg.normalization_mapping["ENV"] = parse_norm_mode(args.env_norm)
    cfg.normalization_mapping["ACTION"] = parse_norm_mode(args.action_norm)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type != "cpu",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    steps_per_epoch = len(dataloader)
    if steps_per_epoch <= 0:
        raise ValueError(
            "resolved dataloader is empty after batching; please lower batch_size or inspect dataset."
        )
    resolved_max_steps = int(args.max_steps) if args.max_steps is not None else None
    if args.max_epochs is not None:
        resolved_max_steps = int(args.max_epochs) * steps_per_epoch
    if resolved_max_steps is None:
        resolved_max_steps = 5000
    resolved_save_interval = int(args.save_interval)
    if args.save_every_epochs is not None:
        resolved_save_interval = int(args.save_every_epochs) * steps_per_epoch

    if args.resume_ckpt:
        policy = ACTPolicy.from_pretrained(
            args.resume_ckpt, config=cfg, strict=True
        ).to(device)
    else:
        policy = ACTPolicy(cfg).to(device)
    policy.train()
    train_policy = (
        torch.compile(policy)
        if args.compile_model and hasattr(torch, "compile")
        else policy
    )
    preprocessor, postprocessor = make_act_pre_post_processors(
        cfg, dataset_stats=dataset_stats
    )
    optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params())
    scheduler = build_scheduler(
        optimizer,
        scheduler_name=args.lr_scheduler,
        total_steps=resolved_max_steps,
        warmup_steps=args.lr_warmup_steps,
        min_lr=args.lr_min,
    )

    wandb_run = None
    if not args.disable_wandb and args.wandb_mode != "disabled":
        try:
            import wandb
        except ModuleNotFoundError:
            print(
                "wandb is not installed in this environment; continuing without wandb logging."
            )
            wandb = None

        if wandb is not None:
            run_config = vars(args).copy()
            run_config.update(
                {
                    "task": args.task,
                    "resolved_dataset": str(resolved_dataset),
                    "resume_ckpt": str(args.resume_ckpt) if args.resume_ckpt else None,
                    "state_obs_keys": state_obs_keys,
                    "env_obs_keys": env_obs_keys,
                    "device": device.type,
                    "env_meta_json": setup["env_meta_json"],
                    "stats_npz": setup["stats_npz"],
                    "dsrl_normalization_npz": setup["dsrl_normalization_npz"],
                    "dataset_kind": dataset_kind,
                    "observation_space_kind": observation_space_kind,
                    "action_space_kind": action_space_kind,
                    "lowdim_keys": lowdim_keys,
                    "lowdim_dims": lowdim_dims,
                    "steps_per_epoch": steps_per_epoch,
                    "resolved_max_steps": resolved_max_steps,
                    "resolved_save_interval": resolved_save_interval,
                }
            )
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                config=run_config,
                dir=str(output_dir),
            )

    print("============================================")
    print(f"Task       : {args.task}")
    print(f"Dataset    : {resolved_dataset}")
    print(f"Data Kind  : {dataset_kind}")
    print(f"State keys : {state_obs_keys}")
    print(f"Env keys   : {env_obs_keys if env_obs_keys else '(using state as env)'}")
    print(f"Action dim : {feature_shapes[ACTION][0]}")
    print(f"Chunk size : {cfg.chunk_size}")
    print(f"Epoch step : {steps_per_epoch}")
    print(f"Train steps: {resolved_max_steps}")
    print(f"Device     : {device}")
    print("============================================")

    step = 0
    train_start = time.time()
    last_log_t = train_start
    stop_step = (
        min(resolved_max_steps, args.stop_after_steps)
        if args.stop_after_steps > 0
        else resolved_max_steps
    )
    while step < stop_step:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, loss_dict = train_policy.forward(batch)
            loss.backward()

            grad_norm = 0.0
            if args.grad_clip_norm > 0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), args.grad_clip_norm
                    )
                )

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_interval == 0 or step == 1:
                now = time.time()
                dt = max(now - last_log_t, 1e-6)
                steps_per_sec = args.log_interval / dt if step > 1 else 0.0
                epoch_float = step / steps_per_epoch
                metrics = {
                    "train/loss": float(loss.item()),
                    "train/l1_loss": float(loss_dict.get("l1_loss", 0.0)),
                    "train/kld_loss": float(loss_dict.get("kld_loss", 0.0)),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                    "train/grad_norm": grad_norm,
                    "train/steps_per_sec": steps_per_sec,
                    "train/epoch": epoch_float,
                }
                print(
                    f"step={step:07d} "
                    f"epoch={epoch_float:9.2f} "
                    f"loss={metrics['train/loss']:.6f} "
                    f"l1={metrics['train/l1_loss']:.6f} "
                    f"kld={metrics['train/kld_loss']:.6f} "
                    f"lr={metrics['train/lr']:.2e}"
                )
                if wandb_run is not None:
                    wandb_run.log(metrics, step=step)
                last_log_t = now

            if resolved_save_interval > 0 and step % resolved_save_interval == 0:
                ckpt_dir = output_dir / "checkpoints" / f"step_{step:07d}"
                save_training_bundle(
                    ckpt_dir,
                    policy,
                    preprocessor,
                    postprocessor,
                    {
                        "task": args.task,
                        "step": step,
                        "dataset": str(resolved_dataset),
                        "resume_ckpt": (
                            str(args.resume_ckpt) if args.resume_ckpt else None
                        ),
                        "state_obs_keys": state_obs_keys,
                        "env_obs_keys": env_obs_keys,
                        "chunk_size": int(cfg.chunk_size),
                        "n_action_steps": int(cfg.n_action_steps),
                        "dataset_kind": dataset_kind,
                        "observation_space_kind": observation_space_kind,
                        "action_space_kind": action_space_kind,
                        "env_meta_json": setup["env_meta_json"],
                        "stats_npz": setup["stats_npz"],
                        "dsrl_normalization_npz": setup["dsrl_normalization_npz"],
                        "lowdim_keys": lowdim_keys,
                        "lowdim_dims": lowdim_dims,
                        "steps_per_epoch": steps_per_epoch,
                        "epoch": step / steps_per_epoch,
                    },
                )
                if wandb_run is not None:
                    wandb_run.log({"checkpoint/step": step}, step=step)

            if step >= stop_step:
                break

    elapsed = time.time() - train_start
    final_dir = output_dir / "final"
    final_summary = {
        "task": args.task,
        "total_steps": step,
        "total_epochs": step / steps_per_epoch,
        "elapsed_sec": elapsed,
        "dataset": str(resolved_dataset),
        "resume_ckpt": str(args.resume_ckpt) if args.resume_ckpt else None,
        "state_obs_keys": state_obs_keys,
        "env_obs_keys": env_obs_keys,
        "chunk_size": int(cfg.chunk_size),
        "n_action_steps": int(cfg.n_action_steps),
        "device": device.type,
        "env_meta_json": setup["env_meta_json"],
        "stats_npz": setup["stats_npz"],
        "dsrl_normalization_npz": setup["dsrl_normalization_npz"],
        "dataset_kind": dataset_kind,
        "observation_space_kind": observation_space_kind,
        "action_space_kind": action_space_kind,
        "lowdim_keys": lowdim_keys,
        "lowdim_dims": lowdim_dims,
        "steps_per_epoch": steps_per_epoch,
        "resolved_max_steps": resolved_max_steps,
        "resolved_save_interval": resolved_save_interval,
    }
    save_training_bundle(final_dir, policy, preprocessor, postprocessor, final_summary)

    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        config_dump = vars(args).copy()
        config_dump.update(
            {
                "task": args.task,
                "resolved_dataset": str(resolved_dataset),
                "resume_ckpt": str(args.resume_ckpt) if args.resume_ckpt else None,
                "state_obs_keys": state_obs_keys,
                "env_obs_keys": env_obs_keys,
                "resolved_output_dir": str(output_dir),
                "env_meta_json": setup["env_meta_json"],
                "stats_npz": setup["stats_npz"],
                "dsrl_normalization_npz": setup["dsrl_normalization_npz"],
                "chunk_size": int(cfg.chunk_size),
                "n_action_steps": int(cfg.n_action_steps),
                "dataset_kind": dataset_kind,
                "observation_space_kind": observation_space_kind,
                "action_space_kind": action_space_kind,
                "lowdim_keys": lowdim_keys,
                "lowdim_dims": lowdim_dims,
                "steps_per_epoch": steps_per_epoch,
                "resolved_max_steps": resolved_max_steps,
                "resolved_save_interval": resolved_save_interval,
            }
        )
        json.dump(config_dump, f, indent=2, ensure_ascii=False)

    if wandb_run is not None:
        wandb_run.log(
            {"train/elapsed_sec": elapsed, "train/final_step": step}, step=step
        )
        wandb_run.finish()

    print(f"Training finished. Final checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
