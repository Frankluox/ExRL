"""DICE-local DSRL checkpoint loader using the explicit-noise adapter."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from exrl.runners.dp_q.diffusion import DiffusionPolicy
from baselines.dice.dp.dsrl_policy_adapter import (
    DSRLCkptPolicyAdapter,
    DSRLCkptPolicyBundle,
    DSRLCkptPolicyWrapper,
)


DEFAULT_LOWDIM_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]


def parse_lowdim_keys(spec: str) -> List[str]:
    text = str(spec).strip()
    if not text:
        return list(DEFAULT_LOWDIM_KEYS)
    out = [x.strip() for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("lowdim keys cannot be empty")
    return out


def infer_lowdim_dims_from_stats(stats_npz: str, lowdim_keys: List[str]) -> Dict[str, int]:
    if not os.path.exists(stats_npz):
        raise FileNotFoundError(f"stats npz not found: {stats_npz}")
    stats = np.load(stats_npz)
    out: Dict[str, int] = {}
    for k in lowdim_keys:
        mk = f"{k}_mean"
        if mk not in stats:
            raise ValueError(f"stats missing key: {mk}")
        out[k] = int(np.asarray(stats[mk]).reshape(-1).shape[0])
    return out


def infer_action_dim_from_stats(stats_npz: str) -> int:
    if not os.path.exists(stats_npz):
        raise FileNotFoundError(f"stats npz not found: {stats_npz}")
    stats = np.load(stats_npz)
    if "action_mean" not in stats:
        raise ValueError("stats missing action_mean")
    return int(np.asarray(stats["action_mean"]).reshape(-1).shape[0])


def _select_state_dict(raw_ckpt: dict, weight_source: str) -> Dict[str, torch.Tensor]:
    src = str(weight_source)
    if src in raw_ckpt and isinstance(raw_ckpt[src], dict):
        return raw_ckpt[src]
    if src == "ema" and "model" in raw_ckpt and isinstance(raw_ckpt["model"], dict):
        # Fallback to model only when ema missing.
        return raw_ckpt["model"]
    if src == "model" and "ema" in raw_ckpt and isinstance(raw_ckpt["ema"], dict):
        return raw_ckpt["ema"]

    # Last resort: maybe ckpt itself is state dict.
    tensor_items = [k for k, v in raw_ckpt.items() if torch.is_tensor(v)]
    if len(tensor_items) > 0:
        return raw_ckpt

    raise ValueError(
        f"cannot find state_dict in ckpt using weight_source={src}; "
        f"available keys={list(raw_ckpt.keys())[:20]}"
    )


def _load_neighbor_hydra_config(ckpt_path: str) -> Dict:
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    run_dir = os.path.dirname(ckpt_dir)
    hydra_cfg = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(hydra_cfg):
        return {}
    try:
        with open(hydra_cfg, "r") as f:
            data = yaml.safe_load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _infer_schedule_defaults(ckpt_path: str) -> Tuple[int, int]:
    cfg = _load_neighbor_hydra_config(ckpt_path)

    train_steps = cfg.get("denoising_steps", None)
    if train_steps is None:
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, dict):
            train_steps = model_cfg.get("denoising_steps", None)
    train_steps = int(train_steps) if train_steps is not None else 5

    default_infer_steps = None
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        default_infer_steps = model_cfg.get("ddim_steps", None)
    if default_infer_steps is None:
        default_infer_steps = min(train_steps, 5)
    return max(1, train_steps), max(1, int(default_infer_steps))


def load_dsrl_policy_wrapper(
    ckpt_path: str,
    device: torch.device,
    stats_npz: str,
    lowdim_keys: List[str],
    lowdim_dims: Dict[str, int],
    action_dim: int,
    cond_steps: int = 1,
    weight_source: str = "ema",
) -> Tuple[DSRLCkptPolicyWrapper, Dict]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"DSRL ckpt not found: {ckpt_path}")

    # The audited project checkpoint is trusted and may include Hydra metadata
    # in addition to tensor weights. Be explicit for PyTorch >= 2.6.
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected ckpt type: {type(raw)}")
    state_dict = _select_state_dict(raw, weight_source=weight_source)

    backbone, meta = DiffusionPolicy.from_state_dict(
        state_dict=state_dict,
        action_dim=int(action_dim),
        cond_steps=int(cond_steps),
    )

    obs_sum = int(sum(int(lowdim_dims[k]) for k in lowdim_keys))
    expected_obs = int(meta.obs_dim * meta.cond_steps)
    if obs_sum != expected_obs:
        raise ValueError(
            "[DSRL-CKPT] lowdim obs mismatch: "
            f"sum(lowdim_dims)={obs_sum}, expected(meta.obs_dim*cond_steps)={expected_obs}. "
            f"lowdim_keys={lowdim_keys}, dims={lowdim_dims}, meta={asdict(meta)}"
        )

    obs_shapes = {k: (int(lowdim_dims[k]),) for k in lowdim_keys}
    train_denoising_steps, default_inference_steps = _infer_schedule_defaults(ckpt_path)

    adapter = DSRLCkptPolicyAdapter(
        backbone=backbone,
        backbone_meta=meta,
        obs_shapes=obs_shapes,
        obs_key_order=lowdim_keys,
        device=device,
        default_weight_source=weight_source,
        train_denoising_steps=train_denoising_steps,
        default_inference_steps=default_inference_steps,
    )

    bundle = DSRLCkptPolicyBundle(
        policy=adapter,
        source_path=ckpt_path,
        weight_source=weight_source,
        meta=meta,
    )
    wrapper = DSRLCkptPolicyWrapper(bundle)

    ckpt_meta = {
        "ckpt_path": ckpt_path,
        "weight_source": weight_source,
        "raw_top_keys": list(raw.keys())[:50],
        "backbone_meta": asdict(meta),
        "lowdim_keys": list(lowdim_keys),
        "lowdim_dims": {k: int(v) for k, v in lowdim_dims.items()},
        "stats_npz": stats_npz,
        "action_dim": int(action_dim),
        "cond_steps": int(cond_steps),
        "schedule_defaults": {
            "train_denoising_steps": int(train_denoising_steps),
            "default_inference_steps": int(default_inference_steps),
        },
    }
    return wrapper, ckpt_meta


def load_env_meta_json(env_meta_json: str) -> Dict:
    if not os.path.exists(env_meta_json):
        raise FileNotFoundError(f"env_meta_json not found: {env_meta_json}")
    with open(env_meta_json, "r") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ValueError("env_meta_json must parse to dict")
    return meta
