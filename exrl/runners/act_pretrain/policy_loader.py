"""
ACT checkpoint loading utilities and shared task presets.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from exrl.paths import ROOT as REPO_ROOT

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from exrl.runners.act_pretrain.policy_adapter import (
    ACTCkptPolicyAdapter,
    ACTCkptPolicyBundle,
    ACTCkptPolicyWrapper,
)


DEFAULT_LOWDIM_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
]
DEFAULT_STATE_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
DEFAULT_ENV_KEYS = ["object"]
TRANSPORT_STATE_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot1_eef_pos",
    "robot1_eef_quat",
    "robot1_gripper_qpos",
]


@dataclass(frozen=True)
class ACTTaskPreset:
    task: str
    state_obs_keys: Tuple[str, ...]
    env_obs_keys: Tuple[str, ...]
    dataset_candidates: Tuple[str, ...]
    stats_npz_candidates: Tuple[str, ...]
    dsrl_normalization_candidates: Tuple[str, ...]
    env_meta_json_candidates: Tuple[str, ...]


ACT_TASK_PRESETS: Dict[str, ACTTaskPreset] = {
    task: ACTTaskPreset(
        task=task,
        state_obs_keys=tuple(
            TRANSPORT_STATE_KEYS if task == "transport" else DEFAULT_STATE_KEYS
        ),
        env_obs_keys=tuple(DEFAULT_ENV_KEYS),
        dataset_candidates=(f"data/{task}/train.npz",),
        stats_npz_candidates=(f"data/{task}/{task}_lowdim_stats.npz",),
        dsrl_normalization_candidates=(f"data/{task}/normalization.npz",),
        env_meta_json_candidates=(f"data/{task}/{task}.json",),
    )
    for task in ("lift", "can", "square", "transport", "tool_hang")
}


def parse_csv_list(spec: str | Sequence[str] | None) -> List[str]:
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        return [str(item).strip() for item in spec if str(item).strip()]
    text = str(spec).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _feature_to_jsonable(feature) -> Dict:
    return {
        "type": getattr(feature.type, "value", str(feature.type)),
        "shape": list(feature.shape),
    }


def parse_lowdim_keys(spec: str) -> List[str]:
    keys = parse_csv_list(spec)
    return keys if keys else list(DEFAULT_LOWDIM_KEYS)


def infer_lowdim_dims_from_stats(
    stats_npz: str, lowdim_keys: List[str]
) -> Dict[str, int]:
    if not os.path.exists(stats_npz):
        raise FileNotFoundError(f"stats npz not found: {stats_npz}")
    stats = np.load(stats_npz)
    return {
        key: int(np.asarray(stats[f"{key}_mean"]).reshape(-1).shape[0])
        for key in lowdim_keys
    }


def infer_action_dim_from_stats(stats_npz: str) -> int:
    if not os.path.exists(stats_npz):
        raise FileNotFoundError(f"stats npz not found: {stats_npz}")
    stats = np.load(stats_npz)
    return int(np.asarray(stats["action_mean"]).reshape(-1).shape[0])


def _resolve_existing_path(
    candidates: Sequence[str], required: bool = False
) -> Optional[str]:
    cleaned = [
        str(candidate).strip() for candidate in candidates if str(candidate).strip()
    ]
    for candidate in cleaned:
        if os.path.exists(candidate):
            return candidate
    if required and cleaned:
        raise FileNotFoundError(f"none of the candidate paths exist: {cleaned}")
    return cleaned[0] if cleaned else None


def get_act_task_preset(task: str) -> ACTTaskPreset:
    name = str(task).strip().lower()
    if name not in ACT_TASK_PRESETS:
        raise ValueError(f"unknown ACT task preset: {task}")
    return ACT_TASK_PRESETS[name]


def resolve_act_task_setup(
    *,
    task: str,
    dataset: str | None = None,
    state_obs_keys: str | Sequence[str] | None = None,
    env_obs_keys: str | Sequence[str] | None = None,
    chunk_size: int | None = None,
    output_dir: str | None = None,
) -> Dict:
    preset = get_act_task_preset(task)
    resolved_state = parse_csv_list(state_obs_keys) or list(preset.state_obs_keys)
    resolved_env = parse_csv_list(env_obs_keys) or list(preset.env_obs_keys)
    resolved_dataset = dataset or _resolve_existing_path(
        preset.dataset_candidates, required=False
    )
    resolved_stats = _resolve_existing_path(preset.stats_npz_candidates, required=False)
    resolved_dsrl_normalization = _resolve_existing_path(
        preset.dsrl_normalization_candidates, required=False
    )
    resolved_env_meta = _resolve_existing_path(
        preset.env_meta_json_candidates, required=False
    )
    resolved_output_dir = output_dir or str(REPO_ROOT / "models" / f"{preset.task}_act")
    resolved_chunk_size = int(chunk_size) if chunk_size is not None else 32
    return {
        "task": preset.task,
        "dataset": resolved_dataset,
        "state_obs_keys": resolved_state,
        "env_obs_keys": resolved_env,
        "stats_npz": resolved_stats,
        "dsrl_normalization_npz": resolved_dsrl_normalization,
        "env_meta_json": resolved_env_meta,
        "output_dir": resolved_output_dir,
        "chunk_size": resolved_chunk_size,
    }


def resolve_act_checkpoint_dir(path_or_dir: str | Path) -> Path:
    raw = Path(path_or_dir).expanduser()
    base = raw.resolve() if raw.is_absolute() else (Path.cwd() / raw).resolve()
    if (base / "model.safetensors").exists() and (base / "config.json").exists():
        return base
    final_dir = base / "final"
    if (final_dir / "model.safetensors").exists() and (
        final_dir / "config.json"
    ).exists():
        return final_dir
    checkpoints_dir = base / "checkpoints"
    if checkpoints_dir.exists():
        candidates = [
            item
            for item in checkpoints_dir.iterdir()
            if item.is_dir()
            and (item / "model.safetensors").exists()
            and (item / "config.json").exists()
        ]
        if candidates:
            return sorted(candidates)[-1]
    raise FileNotFoundError(f"unable to resolve ACT checkpoint dir from: {path_or_dir}")


def infer_train_output_dir(ckpt_dir: Path) -> Path:
    if ckpt_dir.parent.name == "final":
        return ckpt_dir.parent.parent
    if ckpt_dir.parent.parent.name == "checkpoints":
        return ckpt_dir.parent.parent.parent
    return ckpt_dir.parent


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def load_act_run_metadata(ckpt_dir: Path) -> Dict:
    output_dir = infer_train_output_dir(ckpt_dir)
    metadata: Dict = {}
    for candidate in (
        output_dir / "train_config.json",
        output_dir / "final" / "run_summary.json",
        ckpt_dir / "run_summary.json",
    ):
        metadata.update(_read_json(candidate))
    return metadata


def load_env_meta_json(env_meta_json: str) -> Dict:
    if not os.path.exists(env_meta_json):
        raise FileNotFoundError(f"env_meta_json not found: {env_meta_json}")
    with open(env_meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ValueError("env_meta_json must parse to dict")
    return meta


def _resolve_obs_keys(
    metadata: Dict,
    task: str | None,
    state_obs_keys: Sequence[str] | None,
    env_obs_keys: Sequence[str] | None,
) -> Tuple[List[str], List[str]]:
    resolved_task = str(task or metadata.get("task") or "square").lower()
    preset = get_act_task_preset(resolved_task)
    resolved_state = (
        parse_csv_list(state_obs_keys)
        or parse_csv_list(metadata.get("state_obs_keys"))
        or list(preset.state_obs_keys)
    )
    resolved_env = (
        parse_csv_list(env_obs_keys)
        or parse_csv_list(metadata.get("env_obs_keys"))
        or list(preset.env_obs_keys)
    )
    return resolved_state, resolved_env


def load_act_policy_wrapper(
    *,
    ckpt_path: str,
    device: torch.device,
    task: str | None = None,
    state_obs_keys: Sequence[str] | None = None,
    env_obs_keys: Sequence[str] | None = None,
) -> Tuple[ACTCkptPolicyWrapper, Dict]:
    ckpt_dir = resolve_act_checkpoint_dir(ckpt_path)
    metadata = load_act_run_metadata(ckpt_dir)
    resolved_task = str(task or metadata.get("task") or "square").lower()
    resolved_state, resolved_env = _resolve_obs_keys(
        metadata, resolved_task, state_obs_keys, env_obs_keys
    )

    cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    if not isinstance(cfg, ACTConfig):
        raise TypeError(f"checkpoint is not an ACT model: {type(cfg).__name__}")
    cfg.device = str(device)

    policy = (
        ACTPolicy.from_pretrained(ckpt_dir, config=cfg, strict=True).to(device).eval()
    )
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        ckpt_dir,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides={"device_processor": {"device": str(device)}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        ckpt_dir,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    use_state_as_env_when_missing = bool(
        cfg.env_state_feature is not None and len(resolved_env) == 0
    )
    adapter = ACTCkptPolicyAdapter(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        state_obs_keys=resolved_state,
        env_obs_keys=resolved_env,
        use_state_as_env_when_missing=use_state_as_env_when_missing,
        device=device,
    )

    dataset_hint = str(
        metadata.get("resolved_dataset") or metadata.get("dataset") or ""
    )
    observation_space_kind = str(
        metadata.get("observation_space_kind")
        or ("dsrl_normalized" if dataset_hint.endswith(".npz") else "raw_lowdim")
    )
    action_space_kind = str(
        metadata.get("action_space_kind")
        or ("dsrl_normalized" if dataset_hint.endswith(".npz") else "raw_action")
    )

    ckpt_meta = {
        "ckpt_dir": str(ckpt_dir),
        "task": resolved_task,
        "config_type": cfg.type,
        "chunk_size": int(cfg.chunk_size),
        "n_action_steps": int(cfg.n_action_steps),
        "action_dim": int(adapter.ac_dim),
        "state_obs_keys": list(resolved_state),
        "env_obs_keys": list(resolved_env),
        "use_state_as_env_when_missing": use_state_as_env_when_missing,
        "dataset_kind": metadata.get("dataset_kind"),
        "observation_space_kind": observation_space_kind,
        "action_space_kind": action_space_kind,
        "stats_npz": metadata.get("stats_npz"),
        "dsrl_normalization_npz": metadata.get("dsrl_normalization_npz"),
        "lowdim_keys": parse_csv_list(metadata.get("lowdim_keys"))
        or list(dict.fromkeys(list(resolved_state) + list(resolved_env))),
        "lowdim_dims": metadata.get("lowdim_dims"),
        "metadata": metadata,
        "input_features": {
            key: _feature_to_jsonable(feature)
            for key, feature in (cfg.input_features or {}).items()
        },
        "output_features": {
            key: _feature_to_jsonable(feature)
            for key, feature in (cfg.output_features or {}).items()
        },
    }

    bundle = ACTCkptPolicyBundle(
        policy=adapter,
        source_path=str(ckpt_dir),
        state_obs_keys=list(resolved_state),
        env_obs_keys=list(resolved_env),
        meta=ckpt_meta,
    )
    return ACTCkptPolicyWrapper(bundle), ckpt_meta
