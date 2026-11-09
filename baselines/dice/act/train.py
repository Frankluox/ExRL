"""Train DICE-RL with frozen action chunks and a noise-conditioned residual."""

from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
import random
import re
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np
import torch

REPO_ROOT = __import__("exrl.paths", fromlist=["ROOT"]).ROOT
from baselines.dice.act.act_task_specs import (
    expected_offline_transitions,
    get_task_contract,
    validate_horizon,
)
from baselines.dice.act.core import (
    CompleteEpisodeAccumulator,
    DICERLLearner,
    DICERLLearnerConfig,
    DICEReplayBuffer,
    LinearOfflineRatioSchedule,
    TransitionUpdateScheduler,
    sample_rlpd_batch,
)
from baselines.dice.act.act_chunk import (
    DICEACTPriorSampler,
    DICEChunkVecEnv,
    action_audit_metrics,
    add_chunk_step_to_replay,
)
from baselines.dice.act.act_ckpt_loader import (
    infer_action_dim_from_stats,
    infer_lowdim_dims_from_stats,
    load_act_policy_wrapper,
    parse_lowdim_keys,
    resolve_act_checkpoint_dir,
)
from baselines.dice.act.offline_chunk_dataset import build_offline_chunk_replay
from baselines.dice.act.offline_dataset_utils import (
    load_offline_cache,
    save_offline_cache,
    sha256_file,
)
from exrl.runners.dp_q.vector_env import RobomimicExecutor

DICE_PAPER_URL = "https://arxiv.org/abs/2603.10263v2"
DICE_REPOSITORY_URL = "https://github.com/real-stanford/dice-rl"


class MetricWriter:
    """TensorBoard when installed, plus an always-present JSONL training log."""

    def __init__(self, run_dir: str):
        self.jsonl_path = os.path.join(run_dir, "training_metrics.jsonl")
        self.tensorboard = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tensorboard = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))
        except Exception as exc:
            print("[WARN] TensorBoard unavailable: %s" % exc, flush=True)

    def add(self, metrics: Dict[str, float], step: int) -> None:
        record: Dict[str, object] = {
            "step": int(step),
            "time_utc": datetime.utcnow().isoformat() + "Z",
        }
        for key, value in metrics.items():
            scalar = float(value)
            if np.isfinite(scalar):
                record[str(key)] = scalar
                if self.tensorboard is not None:
                    self.tensorboard.add_scalar(str(key), scalar, int(step))
        with open(self.jsonl_path, "a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if self.tensorboard is not None:
            self.tensorboard.flush()

    def close(self) -> None:
        if self.tensorboard is not None:
            self.tensorboard.close()


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def capture_rng_state() -> Dict[str, object]:
    state: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextlib.contextmanager
def preserved_rng_state():
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


def optimizer_manifest(optimizer: torch.optim.Optimizer) -> Dict[str, object]:
    group = optimizer.param_groups[0]
    return {
        "class": type(optimizer).__name__,
        "lr": float(group["lr"]),
        "betas": [float(value) for value in group["betas"]],
        "eps": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "amsgrad": bool(group["amsgrad"]),
    }


def atomic_torch_save(payload: Dict[str, object], path: str) -> None:
    temporary_path = "%s.tmp.%d" % (path, os.getpid())
    torch.save(payload, temporary_path, pickle_protocol=4)
    with open(temporary_path, "rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)
    directory_fd = None
    try:
        directory_fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def atomic_json_write(path: str, payload: Dict[str, object]) -> None:
    temporary_path = "%s.tmp.%d" % (path, os.getpid())
    with open(temporary_path, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)
    directory_fd = None
    try:
        directory_fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def cpu_module_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def update_array_trace(digest, label: str, array: np.ndarray) -> None:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    digest.update(label.encode("utf-8"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())


def validate_args(args: argparse.Namespace) -> None:
    contract = get_task_contract(args.env_name)
    validate_horizon(contract, args.base_horizon_steps)
    if args.run_tag and (not re.fullmatch("[A-Za-z0-9_.-]+", str(args.run_tag))):
        raise ValueError("run_tag may contain only letters, digits, dot, underscore, and dash")
    if args.eval_n_envs <= 0 or args.eval_episodes <= 0:
        raise ValueError("evaluation env and episode counts must be positive")
    if args.eval_interval is None:
        args.eval_interval = int(contract.eval_interval_primitive_steps)
    if args.eval_interval <= 0 or args.log_interval <= 0:
        raise ValueError("evaluation and logging intervals must be positive")
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative")
    if args.eval_snapshot_interval < 0:
        raise ValueError("eval_snapshot_interval must be non-negative")
    if args.eval_snapshot_slots <= 0:
        raise ValueError("eval_snapshot_slots must be positive")
    if args.online_replay_capacity < args.min_online_replay_before_updates:
        raise ValueError("online replay capacity is smaller than its update gate")
    expected = expected_offline_transitions(contract, args.base_horizon_steps)
    if args.min_online_replay_before_updates <= 0:
        raise ValueError("min_online_replay_before_updates must be positive")
    if args.eval_only and args.offline_cache_only:
        raise ValueError("eval_only and offline_cache_only are mutually exclusive")
    if args.eval_only and (not args.resume):
        raise ValueError("eval_only requires --resume")
    if args.eval_only and args.disable_all_eval:
        raise ValueError("eval_only and disable_all_eval are mutually exclusive")
    if not args.eval_only:
        if not args.official_demo_npz or not args.reward_hdf5:
            raise ValueError("training/cache mode requires official demo NPZ and reward HDF5")
    if not args.eval_only and (not args.offline_cache_only):
        if args.total_primitive_steps is None or args.total_primitive_steps <= 0:
            raise ValueError("training requires a positive --total_primitive_steps")
    if args.total_primitive_steps is not None and args.total_primitive_steps <= 0:
        raise ValueError("total_primitive_steps must be positive")
    if args.disable_all_eval:
        args.disable_periodic_eval = True
    if not args.eval_only:
        pass


def act_checkpoint_file_hashes(path: str) -> Tuple[str, Dict[str, str]]:
    """Hash exactly the ACT files consumed by the standard loader."""
    checkpoint_dir = resolve_act_checkpoint_dir(path)
    required_names = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "run_summary.json",
    )
    files = [checkpoint_dir / name for name in required_names]
    preprocessor_state = sorted(checkpoint_dir.glob("policy_preprocessor_*.safetensors"))
    postprocessor_state = sorted(checkpoint_dir.glob("policy_postprocessor_*.safetensors"))
    if not preprocessor_state or not postprocessor_state:
        raise FileNotFoundError(
            "ACT checkpoint is missing serialized pre/postprocessor state: %s" % checkpoint_dir
        )
    files.extend(preprocessor_state)
    files.extend(postprocessor_state)
    missing = [str(item) for item in files if not item.is_file()]
    if missing:
        raise FileNotFoundError("ACT checkpoint files are missing: %s" % missing)
    hashes = {
        item.relative_to(checkpoint_dir).as_posix(): sha256_file(str(item))
        for item in sorted(files)
    }
    return (str(checkpoint_dir), hashes)


def act_checkpoint_composite_sha256(file_hashes: Dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative_path, file_sha256 in sorted(file_hashes.items()):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(file_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_assets(args: argparse.Namespace) -> Dict[str, object]:
    checkpoint = args.agent
    if os.path.isdir(checkpoint):
        checkpoint = os.path.join(checkpoint, "model.safetensors")
    paths = {"checkpoint": checkpoint}
    for name, key in (
        ("normalization", "dsrl_normalization_npz"),
        ("stats", "stats_npz_override"),
        ("env_meta", "env_meta_json"),
    ):
        value = getattr(args, key, None)
        if value:
            paths[name] = value
    if not getattr(args, "eval_only", False) and getattr(args, "data_mode", "mixed") != "online":
        for name, key in (
            ("official_demo", "official_demo_npz"),
            ("reward_hdf5", "reward_hdf5"),
            ("demo_hdf5", "demo_hdf5"),
            ("minari_hdf5", "minari_hdf5"),
        ):
            value = getattr(args, key, None)
            if value:
                paths[name] = value
    return {name: sha256_file(path) for name, path in paths.items()}


def configure_act_policy(policy) -> None:
    policy.set_eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)


def load_policy_and_dimensions(
    args: argparse.Namespace, device: torch.device
) -> Tuple[object, Dict[str, object], Tuple[str, ...], Dict[str, int], int, int]:
    contract = get_task_contract(args.env_name)
    lowdim_keys = tuple(parse_lowdim_keys(args.lowdim_keys))
    if lowdim_keys != tuple(contract.lowdim_keys):
        raise RuntimeError("%s low-dimensional observation key order mismatch" % contract.env_name)
    lowdim_dims = infer_lowdim_dims_from_stats(args.stats_npz_override, list(lowdim_keys))
    state_dim = int(sum(lowdim_dims.values()))
    action_dim = infer_action_dim_from_stats(args.stats_npz_override)
    if state_dim != contract.state_dim or action_dim != contract.action_dim:
        raise RuntimeError(
            "%s dimension mismatch: inferred=%d/%d required=%d/%d"
            % (contract.env_name, state_dim, action_dim, contract.state_dim, contract.action_dim)
        )
    wrapper, metadata = load_act_policy_wrapper(
        ckpt_path=args.agent, device=device, task=args.env_name
    )
    configure_act_policy(wrapper.policy)
    if int(metadata["chunk_size"]) != int(args.base_horizon_steps):
        raise RuntimeError(
            "base horizon mismatch: checkpoint=%d config=%d"
            % (int(metadata["chunk_size"]), int(args.base_horizon_steps))
        )
    if int(metadata["n_action_steps"]) != int(args.base_horizon_steps):
        raise RuntimeError(
            "ACT action horizon mismatch: checkpoint=%d config=%d"
            % (int(metadata["n_action_steps"]), int(args.base_horizon_steps))
        )
    if int(metadata["action_dim"]) != action_dim:
        raise RuntimeError("ACT checkpoint/action-stat dimension mismatch")
    if int(wrapper.policy.obs_dim) != state_dim:
        raise RuntimeError("ACT checkpoint/observation-stat dimension mismatch")
    if int(wrapper.policy.cond_steps) != 1:
        raise RuntimeError("DICE-ACT requires one-step low-dimensional observations")
    required_spaces = {
        "observation_space_kind": "dsrl_normalized",
        "action_space_kind": "dsrl_normalized",
        "dataset_kind": "dsrl_npz",
    }
    mismatches = {
        key: {"actual": metadata.get(key), "required": value}
        for key, value in required_spaces.items()
        if metadata.get(key) != value
    }
    actual_keys = list(metadata["state_obs_keys"]) + list(metadata["env_obs_keys"])
    if actual_keys != list(lowdim_keys):
        mismatches["lowdim_keys"] = {"actual": actual_keys, "required": list(lowdim_keys)}
    declared_task = str(metadata.get("metadata", {}).get("task", "")).lower()
    if declared_task != str(args.env_name).lower():
        mismatches["declared_task"] = {
            "actual": declared_task,
            "required": str(args.env_name).lower(),
        }
    metadata_dims = metadata.get("lowdim_dims")
    if metadata_dims:
        actual_dims = {key: int(value) for key, value in metadata_dims.items()}
        expected_dims = {key: int(value) for key, value in lowdim_dims.items()}
        if actual_dims != expected_dims:
            mismatches["lowdim_dims"] = {"actual": actual_dims, "required": expected_dims}
    if mismatches:
        raise RuntimeError(
            "strict ACT checkpoint metadata mismatch: %s" % json.dumps(mismatches, sort_keys=True)
        )
    return (wrapper, metadata, lowdim_keys, lowdim_dims, state_dim, action_dim)


def make_env(
    args: argparse.Namespace,
    lowdim_keys: Tuple[str, ...],
    action_dim: int,
    n_envs: int,
    seed: int,
    bootstrap_timeouts: bool,
) -> DICEChunkVecEnv:
    contract = get_task_contract(args.env_name)
    executor = RobomimicExecutor(
        vendor_root=args.dsrl_official_root,
        env_name=contract.env_name,
        env_meta_json=args.env_meta_json,
        normalization_path=args.dsrl_normalization_npz,
        lowdim_keys=list(lowdim_keys),
        n_envs=int(n_envs),
        max_episode_steps=contract.env_horizon,
        seed=int(seed),
        asynchronous=True,
        terminate_on_success=True,
        timeout_on_horizon=True,
    )
    return DICEChunkVecEnv(
        executor=executor,
        horizon_steps=args.base_horizon_steps,
        action_dim=action_dim,
        reward_offset=contract.reward_offset,
        bootstrap_timeouts=bootstrap_timeouts,
    )


def offline_cache_provenance(
    args: argparse.Namespace, asset_hashes: Dict[str, object]
) -> Dict[str, object]:
    contract = get_task_contract(args.env_name)
    return {
        "version": 2,
        "env_name": contract.env_name,
        "all_demo_episodes": contract.offline_episodes,
        "official_npz_sha256": asset_hashes["official_demo"],
        "reward_hdf5_sha256": asset_hashes["reward_hdf5"],
        "normalization_sha256": asset_hashes["normalization"],
        "checkpoint_sha256": asset_hashes["checkpoint"],
        "suffix_noise_seed": int(args.seed) + 20000,
        "base_policy_family": "act",
        "act_checkpoint_format": "lerobot_pretrained",
        "act_inference_latent": "deterministic_zero_latent",
        "act_action_postprocess": "clip_dsrl_normalized_to_unit_box",
        "dice_z_application": "residual_actor_only_base_act_ignores_z",
        "horizon_steps": int(args.base_horizon_steps),
        "state_dim": contract.state_dim,
        "action_dim": contract.action_dim,
        "n_step": contract.n_step,
        "gamma_per_macro_decision": contract.gamma,
        "within_chunk_discount": False,
        "per_primitive_reward": "raw_sparse_reward_minus_1",
        "stop_rule": "last_frame_inclusive"
        if contract.offline_success_mode == "last_frame"
        else "first_positive_reward_inclusive",
        "hdf5_state_used": False,
        "transition_placement": "one_nonoverlapping_transition_per_issued_fixed_ta_chunk",
    }


def validate_offline_summary(args: argparse.Namespace, summary) -> None:
    if summary.transitions <= 0:
        raise ValueError("The demonstrations produced no replay transitions")
    if getattr(summary, "action_alignment_max_abs", 0.0) > 1e-06:
        raise ValueError("Demonstration actions do not match the normalization statistics")


def maybe_build_offline_replay(
    args: argparse.Namespace,
    prior_sampler: DICEACTPriorSampler,
    state_dim: int,
    action_dim: int,
    asset_hashes: Dict[str, object],
):
    contract = get_task_contract(args.env_name)
    if args.eval_only:
        return (None, None, None, None)
    replay = DICEReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim * int(args.base_horizon_steps),
        capacity=args.offline_replay_capacity,
        seed=args.seed + 2000,
    )
    provenance = offline_cache_provenance(args, asset_hashes)
    key = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    cache_path = os.path.join(
        args.offline_cache_dir,
        "%s_ta%d_%s.pt" % (contract.env_name, int(args.base_horizon_steps), key),
    )
    if os.path.exists(cache_path):
        summary = load_offline_cache(cache_path, replay, provenance)
        print("[DICE OFFLINE CACHE] loaded %s" % cache_path, flush=True)
    else:
        summary = build_offline_chunk_replay(
            official_npz_path=args.official_demo_npz,
            reward_hdf5_path=args.reward_hdf5,
            normalization_path=args.dsrl_normalization_npz,
            expected_npz_sha256=None,
            prior_sampler=prior_sampler,
            replay=replay,
            horizon_steps=args.base_horizon_steps,
            action_dim=action_dim,
            state_dim=state_dim,
            n_step=contract.n_step,
            gamma=contract.gamma,
            reward_offset=contract.reward_offset,
            success_mode=contract.offline_success_mode,
            success_horizon=contract.env_horizon,
            expected_transition_count=None,
            precomputed_npz_sha256=asset_hashes["official_demo"],
            precomputed_hdf5_sha256=asset_hashes["reward_hdf5"],
            suffix_noise_seed=args.seed + 20000,
        )
        validate_offline_summary(args, summary)
        save_offline_cache(cache_path, replay, summary, provenance)
        print("[DICE OFFLINE CACHE] built %s" % cache_path, flush=True)
    validate_offline_summary(args, summary)
    return (replay, summary, cache_path, provenance)


def success_from_info(info: Dict[str, object]) -> bool:
    value = info.get("is_success", False)
    if isinstance(value, (list, tuple, np.ndarray)):
        return bool(value[-1]) if len(value) else False
    if isinstance(value, dict):
        return bool(value.get("task", False))
    return bool(value)


class RolloutTracker:
    """Telemetry only; its window never affects action selection or learning."""

    def __init__(self, n_envs: int, window: int = 200):
        self.returns = np.zeros((int(n_envs),), dtype=np.float64)
        self.lengths = np.zeros((int(n_envs),), dtype=np.int64)
        self.recent_returns = deque(maxlen=int(window))
        self.recent_lengths = deque(maxlen=int(window))
        self.recent_successes = deque(maxlen=int(window))
        self.total_episodes = 0

    def update(self, step) -> None:
        self.returns += np.asarray(step.rewards, dtype=np.float64)
        self.lengths += np.asarray(step.exec_lens, dtype=np.int64)
        for env_index in np.flatnonzero(step.dones):
            index = int(env_index)
            self.recent_returns.append(float(self.returns[index]))
            self.recent_lengths.append(int(self.lengths[index]))
            self.recent_successes.append(float(success_from_info(step.infos[index])))
            self.total_episodes += 1
            self.returns[index] = 0.0
            self.lengths[index] = 0

    def metrics(self) -> Dict[str, float]:
        if not self.recent_successes:
            return {"rollout/episodes": float(self.total_episodes)}
        return {
            "rollout/episodes": float(self.total_episodes),
            "rollout/success_rate": float(np.mean(self.recent_successes)),
            "rollout/ep_rew_mean": float(np.mean(self.recent_returns)),
            "rollout/ep_len_mean": float(np.mean(self.recent_lengths)),
            "rollout/window_episodes": float(len(self.recent_successes)),
        }


def evaluate(
    env: DICEChunkVecEnv,
    learner: DICERLLearner,
    episodes: int,
    base_seed: int,
    eval_step: int,
    candidate_count: int,
    trace_out: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    completed = 0
    successes = 0
    lengths = np.zeros((env.n_envs,), dtype=np.int64)
    completed_lengths = []
    trace_digest = hashlib.sha256() if trace_out is not None else None
    trace_primitives = 0
    max_residual_abs = 0.0
    max_action_identity_error = 0.0
    actor_was_training = learner.actor.training
    critic_was_training = learner.critic.training
    try:
        with preserved_rng_state():
            set_global_seed(int(base_seed) + int(eval_step))
            learner.actor.eval()
            learner.critic.eval()
            state = env.reset(base_seed=int(base_seed) + int(eval_step))
            while completed < int(episodes):
                selected = learner.select_action(
                    state,
                    primitive_transitions=int(eval_step),
                    eval_mode=True,
                    candidate_count_override=int(candidate_count),
                )
                if trace_digest is not None:
                    update_array_trace(trace_digest, "state", state)
                    update_array_trace(trace_digest, "base_action", selected["base_action"])
                    update_array_trace(trace_digest, "residual", selected["residual"])
                    update_array_trace(trace_digest, "action", selected["action"])
                    max_residual_abs = max(
                        max_residual_abs, float(np.max(np.abs(selected["residual"])))
                    )
                    max_action_identity_error = max(
                        max_action_identity_error,
                        float(
                            np.max(
                                np.abs(
                                    selected["action"]
                                    - selected["base_action"]
                                    - selected["residual"]
                                )
                            )
                        ),
                    )
                step = env.step(selected["action"], selected["base_action"], selected["residual"])
                primitive_count = int(np.asarray(step.exec_lens, dtype=np.int64).sum())
                trace_primitives += primitive_count
                lengths += np.asarray(step.exec_lens, dtype=np.int64)
                for env_index in np.flatnonzero(step.dones):
                    if completed >= int(episodes):
                        break
                    index = int(env_index)
                    successes += int(success_from_info(step.infos[index]))
                    completed_lengths.append(int(lengths[index]))
                    lengths[index] = 0
                    completed += 1
                state = step.next_state
    finally:
        learner.actor.train(actor_was_training)
        learner.critic.train(critic_was_training)
    if trace_out is not None:
        trace_out.update(
            {
                "sha256": trace_digest.hexdigest(),
                "primitive_transitions_hashed": int(trace_primitives),
                "candidate_count": int(candidate_count),
                "max_residual_abs": float(max_residual_abs),
                "max_action_minus_base_minus_residual_abs": float(max_action_identity_error),
                "fields": ["state", "base_action", "residual", "action"],
            }
        )
    return {
        "env_step_eval/success_rate": float(successes) / float(max(completed, 1)),
        "env_step_eval/successes": float(successes),
        "env_step_eval/episodes": float(completed),
        "env_step_eval/episode_length_mean": float(np.mean(completed_lengths)),
        "env_step_eval/candidate_count": float(candidate_count),
    }


def next_boundary(current: int, interval: int) -> int:
    return (int(current) // int(interval) + 1) * int(interval)


def periodic_eval_steps_skipped_during_warmup(
    warmup_end_primitive_steps: int, interval: int
) -> Tuple[int, ...]:
    """Absolute periodic boundaries suppressed through the DICE warmup."""
    return tuple(range(int(interval), int(warmup_end_primitive_steps) + 1, int(interval)))


def next_periodic_eval_step(
    current_primitive_steps: int, warmup_end_primitive_steps: int, interval: int
) -> int:
    """Next absolute boundary after both current progress and warmup."""
    return next_boundary(
        max(int(current_primitive_steps), int(warmup_end_primitive_steps)), int(interval)
    )


def resume_signature(args: argparse.Namespace) -> Dict[str, object]:
    contract = get_task_contract(args.env_name)
    return {
        "env_name": contract.env_name,
        "seed": int(args.seed),
        "n_envs": int(args.n_envs),
        "base_horizon_steps": int(args.base_horizon_steps),
        "online_replay_capacity": int(args.online_replay_capacity),
        "offline_replay_capacity": int(args.offline_replay_capacity),
        "min_online_replay_before_updates": int(args.min_online_replay_before_updates),
        "contract": asdict(contract),
        "critic_updates_per_macro_transition": 2.5,
        "actor_target_updates_per_macro_transition": 1.25,
    }


def checkpoint_payload(
    args: argparse.Namespace,
    learner: DICERLLearner,
    online_replay: DICEReplayBuffer,
    offline_replay: DICEReplayBuffer,
    scheduler: TransitionUpdateScheduler,
    counters: Dict[str, object],
    asset_hashes: Dict[str, object],
) -> Dict[str, object]:
    return {
        "version": 2,
        "resume_signature": resume_signature(args),
        "asset_hashes": dict(asset_hashes),
        "learner": learner.state_dict(),
        "online_replay": online_replay.state_dict(),
        "offline_replay_rng_state": offline_replay.rng.get_state(),
        "scheduler": scheduler.state_dict(),
        "counters": dict(counters),
        "rng": capture_rng_state(),
        "resume_rollout_policy": "reset_vector_env_and_drop_only_unfinished_complete_episode_accumulators",
    }


def save_checkpoint(
    path: str,
    args: argparse.Namespace,
    learner: DICERLLearner,
    online_replay: DICEReplayBuffer,
    offline_replay: DICEReplayBuffer,
    scheduler: TransitionUpdateScheduler,
    counters: Dict[str, object],
    asset_hashes: Dict[str, object],
) -> None:
    atomic_torch_save(
        checkpoint_payload(
            args, learner, online_replay, offline_replay, scheduler, counters, asset_hashes
        ),
        path,
    )


def eval_snapshot_payload(
    args: argparse.Namespace,
    learner: DICERLLearner,
    snapshot_step: int,
    actual_primitive_steps: int,
    asset_hashes: Dict[str, object],
) -> Dict[str, object]:
    return {
        "snapshot_format": "dice_rl_act_actor_critic_eval_v1",
        "snapshot_env_step": int(snapshot_step),
        "actual_primitive_steps_at_snapshot": int(actual_primitive_steps),
        "resume_signature": resume_signature(args),
        "learner_config": asdict(learner.config),
        "actor": cpu_module_state_dict(learner.actor),
        "critic": cpu_module_state_dict(learner.critic),
        "asset_hashes": dict(asset_hashes),
    }


def save_eval_snapshot(
    run_dir: str,
    args: argparse.Namespace,
    learner: DICERLLearner,
    snapshot_step: int,
    actual_primitive_steps: int,
    asset_hashes: Dict[str, object],
) -> str:
    snapshot_dir = os.path.join(run_dir, "eval_snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    interval = int(args.eval_snapshot_interval)
    if interval <= 0:
        raise ValueError("eval snapshots require a positive interval")
    slot_index = int(snapshot_step) // interval % int(args.eval_snapshot_slots)
    path = os.path.join(snapshot_dir, "eval_slot_%02d.pt" % slot_index)
    sidecar_path = os.path.join(snapshot_dir, "eval_slot_%02d.json" % slot_index)
    if os.path.isfile(sidecar_path):
        with open(sidecar_path) as stream:
            previous = json.load(stream)
        previous_step = int(previous["snapshot_env_step"])
        if previous_step == int(snapshot_step):
            raise FileExistsError(
                "refusing to duplicate DICE-ACT eval snapshot step %d" % int(snapshot_step)
            )
        acknowledgement = os.path.join(snapshot_dir, "ack_step_%012d.json" % previous_step)
        if not os.path.isfile(acknowledgement):
            raise RuntimeError(
                "external eval queue exhausted: slot %d step %d has no watcher ack"
                % (slot_index, previous_step)
            )
    atomic_torch_save(
        eval_snapshot_payload(args, learner, snapshot_step, actual_primitive_steps, asset_hashes),
        path,
    )
    atomic_json_write(
        sidecar_path,
        {
            "snapshot_format": "dice_rl_act_actor_critic_eval_v1",
            "snapshot_env_step": int(snapshot_step),
            "actual_primitive_steps_at_snapshot": int(actual_primitive_steps),
            "slot_index": int(slot_index),
            "snapshot_file": os.path.basename(path),
        },
    )
    return path


def load_checkpoint(
    path: str,
    args: argparse.Namespace,
    learner: DICERLLearner,
    online_replay: DICEReplayBuffer,
    offline_replay: Optional[DICEReplayBuffer],
    scheduler: TransitionUpdateScheduler,
    asset_hashes: Dict[str, object],
) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("version", -1)) != 2:
        raise RuntimeError("unsupported DICE checkpoint version")
    if payload.get("resume_signature") != resume_signature(args):
        raise RuntimeError("DICE resume configuration mismatch")
    saved_asset_hashes = dict(payload.get("asset_hashes", {}))
    mismatches = {
        key: {"checkpoint": saved_asset_hashes.get(key), "current": value}
        for key, value in asset_hashes.items()
        if saved_asset_hashes.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "DICE resume asset hashes do not match: %s" % json.dumps(mismatches, sort_keys=True)
        )
    learner.load_state_dict(payload["learner"])
    online_replay.load_state_dict(payload["online_replay"])
    if offline_replay is not None:
        offline_replay.rng.set_state(payload["offline_replay_rng_state"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload["rng"])
    return dict(payload["counters"])


def learner_config(args: argparse.Namespace) -> DICERLLearnerConfig:
    contract = get_task_contract(args.env_name)
    return DICERLLearnerConfig(
        state_dim=contract.state_dim,
        action_dim=contract.action_dim,
        horizon_steps=int(args.base_horizon_steps),
        actor_hidden_dims=contract.actor_hidden_dims,
        critic_hidden_dims=contract.critic_hidden_dims,
        critic_ensemble_size=contract.critic_ensemble_size,
        critic_use_layernorm=contract.critic_use_layernorm,
        actor_lr=contract.learning_rate,
        critic_lr=contract.learning_rate,
        actor_weight_decay=1e-05,
        critic_weight_decay=1e-05,
        tau=contract.tau,
        beta=contract.beta,
        actor_num_noise_samples=contract.actor_num_noise_samples,
        target_num_noise_samples=contract.target_num_noise_samples,
        rollout_num_noise_samples=contract.rollout_num_noise_samples,
        q_filter_enabled=True,
        q_filter_epsilon=contract.q_filter_epsilon,
        q_filter_warmup_primitive_steps=contract.q_filter_warmup_primitive_steps,
        best_of_n_warmup_primitive_steps=contract.best_of_n_warmup_primitive_steps,
        q_normalization=True,
        disable_actor_q_loss_for_offline=True,
        always_retain_bc_for_offline=False,
        actor_grad_clip=1.0,
        critic_grad_clip=1.0,
        squash_residual=False,
        residual_scale=1.0,
        prior_microbatch_size=args.prior_microbatch_size,
    )


def algorithm_manifest(
    args: argparse.Namespace, config: DICERLLearnerConfig, learner: DICERLLearner
) -> Dict[str, object]:
    return {"learner": asdict(config), "task": args.env_name, "chunk_size": args.base_horizon_steps}


def run_training(args: argparse.Namespace) -> str:
    validate_args(args)
    contract = get_task_contract(args.env_name)
    device = torch.device(
        args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
    )
    set_global_seed(args.seed)
    asset_hashes = verify_assets(args)
    policy_wrapper, policy_metadata, lowdim_keys, lowdim_dims, state_dim, action_dim = (
        load_policy_and_dimensions(args, device)
    )
    config = learner_config(args)
    prior_sampler = DICEACTPriorSampler(
        policy=policy_wrapper.policy,
        lowdim_keys=lowdim_keys,
        lowdim_dims=lowdim_dims,
        horizon_steps=args.base_horizon_steps,
        action_dim=action_dim,
    )
    learner = DICERLLearner(config, prior_sampler, device)
    zero_state = torch.zeros((2, state_dim), dtype=torch.float32, device=device)
    zero_noise = torch.zeros(
        (2, action_dim * int(args.base_horizon_steps)), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        initial_residual = learner.actor(zero_state, zero_noise)
    if not torch.equal(initial_residual, torch.zeros_like(initial_residual)):
        raise RuntimeError("DICE residual actor is not exactly zero at initialization")
    run_name = "%s_%s_ta%d_dice_act_s%d%s" % (
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        contract.env_name,
        int(args.base_horizon_steps),
        int(args.seed),
        "" if not args.run_tag else "_" + str(args.run_tag),
    )
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=False)
    writer = MetricWriter(run_dir)
    training_env = None
    evaluation_env = None
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    offline_replay = None
    offline_summary = None
    offline_cache_path = None
    offline_provenance = None
    try:
        offline_replay, offline_summary, offline_cache_path, offline_provenance = (
            maybe_build_offline_replay(args, prior_sampler, state_dim, action_dim, asset_hashes)
        )
        base_manifest: Dict[str, object] = {
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "device": str(device),
            "args": vars(args),
            "assets": asset_hashes,
            "policy_metadata": policy_metadata,
            "algorithm": algorithm_manifest(args, config, learner),
            "optimizers": {
                "actor": optimizer_manifest(learner.actor_optimizer),
                "critic": optimizer_manifest(learner.critic_optimizer),
            },
            "offline_summary": None if offline_summary is None else asdict(offline_summary),
            "offline_cache_path": offline_cache_path,
            "offline_cache_provenance": offline_provenance,
            "resume_rollout_policy": "reset_vector_env_and_drop_only_unfinished_complete_episode_accumulators",
        }
        atomic_json_write(manifest_path, base_manifest)
        if args.offline_cache_only:
            atomic_json_write(
                os.path.join(run_dir, "final_status.json"),
                {
                    "status": "offline_cache_ready",
                    "offline_cache_path": offline_cache_path,
                    "finished_utc": datetime.utcnow().isoformat() + "Z",
                },
            )
            print("[CACHE READY] %s" % offline_cache_path, flush=True)
            return run_dir
        online_replay = DICEReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim * int(args.base_horizon_steps),
            capacity=args.online_replay_capacity,
            seed=args.seed + 1000,
        )
        scheduler = TransitionUpdateScheduler(2.5, 1.25)
        counters: Dict[str, object] = {
            "actual_primitive_transitions": 0,
            "macro_transitions_collected": 0,
            "vector_steps": 0,
            "critic_updates": 0,
            "actor_updates": 0,
            "target_updates": 0,
            "update_eligible_macro_transitions": 0,
            "online_transitions_materialized": 0,
            "updates_started": False,
        }
        if args.resume:
            counters = load_checkpoint(
                args.resume, args, learner, online_replay, offline_replay, scheduler, asset_hashes
            )
        base_manifest["resumed_from"] = args.resume
        base_manifest["external_eval"] = {
            "enabled": bool(args.eval_snapshot_interval > 0),
            "watcher_required": bool(args.eval_snapshot_interval > 0),
            "snapshot_format": "dice_rl_act_actor_critic_eval_v1",
            "snapshot_interval": int(args.eval_snapshot_interval),
            "snapshot_slots": int(args.eval_snapshot_slots),
            "bounded_storage": True,
            "slot_reuse_requires_watcher_ack": True,
            "training_process_constructs_eval_env": not bool(args.disable_all_eval),
            "snapshot_weights": ["actor", "critic"],
            "optimizer_and_replay_excluded": True,
        }
        atomic_json_write(manifest_path, base_manifest)
        if args.eval_snapshot_interval > 0 and (not args.resume):
            snapshot_path = save_eval_snapshot(
                run_dir,
                args,
                learner,
                snapshot_step=0,
                actual_primitive_steps=0,
                asset_hashes=asset_hashes,
            )
            print("[ASYNC EVAL SNAPSHOT] primitive=0 path=%s" % snapshot_path, flush=True)
        if not args.eval_only:
            with preserved_rng_state():
                training_env = make_env(
                    args, lowdim_keys, action_dim, args.n_envs, args.seed, bootstrap_timeouts=True
                )
        if not args.disable_all_eval:
            with preserved_rng_state():
                evaluation_env = make_env(
                    args,
                    lowdim_keys,
                    action_dim,
                    args.eval_n_envs,
                    args.seed + args.eval_seed_offset,
                    bootstrap_timeouts=False,
                )
        if evaluation_env is not None:
            trace: Dict[str, object] = {}
            if args.resume:
                initial_eval = evaluate(
                    evaluation_env,
                    learner,
                    episodes=args.eval_episodes,
                    base_seed=args.seed + args.eval_seed_offset,
                    eval_step=int(counters["actual_primitive_transitions"]),
                    candidate_count=1,
                )
                base_manifest["resume_start_eval"] = dict(initial_eval)
            else:
                initial_eval = evaluate(
                    evaluation_env,
                    learner,
                    episodes=args.eval_episodes,
                    base_seed=args.seed + args.eval_seed_offset,
                    eval_step=0,
                    candidate_count=1,
                    trace_out=trace,
                )
                if float(trace["max_residual_abs"]) != 0.0:
                    raise RuntimeError("initial DICE policy does not equal the frozen ACT")
                if float(trace["max_action_minus_base_minus_residual_abs"]) > 2e-06:
                    raise RuntimeError("DICE action identity audit failed")
                base_manifest["initial_frozen_act_trace"] = trace
                base_manifest["base_policy_eval"] = dict(initial_eval)
            atomic_json_write(manifest_path, base_manifest)
            writer.add(initial_eval, int(counters["actual_primitive_transitions"]))
            print(
                "[EVAL] primitive=%d success_rate=%.4f mean_length=%.1f candidates=%d"
                % (
                    int(counters["actual_primitive_transitions"]),
                    initial_eval["env_step_eval/success_rate"],
                    initial_eval["env_step_eval/episode_length_mean"],
                    int(initial_eval["env_step_eval/candidate_count"]),
                ),
                flush=True,
            )
            if args.eval_only:
                atomic_json_write(
                    os.path.join(run_dir, "final_status.json"),
                    {
                        "status": "complete",
                        "mode": "checkpoint_eval_only",
                        "evaluation": initial_eval,
                        "finished_utc": datetime.utcnow().isoformat() + "Z",
                    },
                )
                return run_dir
        if offline_replay is None:
            raise RuntimeError("training requires the audited offline replay")
        accumulator = CompleteEpisodeAccumulator(
            n_step=contract.n_step, gamma=contract.gamma, n_envs=args.n_envs
        )
        ratio_schedule = LinearOfflineRatioSchedule(
            contract.offline_ratio_start,
            contract.offline_ratio_end,
            contract.offline_ratio_decay_primitive_steps,
        )
        rollout_tracker = RolloutTracker(args.n_envs)
        if training_env is None:
            raise RuntimeError("training environment was not constructed")
        with preserved_rng_state():
            state = training_env.reset(base_seed=args.seed)
        progress_started_at = time.monotonic()
        next_log = next_boundary(int(counters["actual_primitive_transitions"]), args.log_interval)
        next_eval = (
            None
            if args.disable_periodic_eval
            else next_periodic_eval_step(
                int(counters["actual_primitive_transitions"]),
                contract.periodic_eval_warmup_primitive_steps,
                args.eval_interval,
            )
        )
        next_eval_snapshot = (
            None
            if args.eval_snapshot_interval <= 0
            else next_periodic_eval_step(
                int(counters["actual_primitive_transitions"]),
                contract.periodic_eval_warmup_primitive_steps,
                args.eval_snapshot_interval,
            )
        )
        base_manifest["effective_periodic_eval_schedule"] = {
            "current_primitive_steps": int(counters["actual_primitive_transitions"]),
            "next_periodic_eval_step": next_eval,
            "warmup_end_inclusive": int(contract.periodic_eval_warmup_primitive_steps),
            "skip_warmup_boundaries_without_backfill": True,
            "execution": "external_async_watcher"
            if args.eval_snapshot_interval > 0
            else "in_process_or_disabled",
            "next_external_eval_snapshot_step": next_eval_snapshot,
        }
        atomic_json_write(manifest_path, base_manifest)
        next_checkpoint = (
            None
            if args.checkpoint_interval <= 0
            else next_boundary(
                int(counters["actual_primitive_transitions"]), args.checkpoint_interval
            )
        )
        last_metrics: Dict[str, float] = {}
        print(
            "[TRAIN] %s TA%d n_envs=4 actual critic UTD=2.5 actor/target UTD=1.25"
            % (contract.env_name, int(args.base_horizon_steps)),
            flush=True,
        )
        while int(counters["actual_primitive_transitions"]) < int(args.total_primitive_steps):
            previous_state = state
            selected = learner.select_action(
                state,
                primitive_transitions=int(counters["actual_primitive_transitions"]),
                eval_mode=False,
            )
            step = training_env.step(
                selected["action"], selected["base_action"], selected["residual"]
            )
            rollout_tracker.update(step)
            emitted = add_chunk_step_to_replay(previous_state, step, accumulator, online_replay)
            state = step.next_state
            primitive_count = int(np.asarray(step.exec_lens, dtype=np.int64).sum())
            counters["actual_primitive_transitions"] = (
                int(counters["actual_primitive_transitions"]) + primitive_count
            )
            counters["macro_transitions_collected"] = (
                int(counters["macro_transitions_collected"]) + args.n_envs
            )
            counters["vector_steps"] = int(counters["vector_steps"]) + 1
            counters["online_transitions_materialized"] = (
                int(counters["online_transitions_materialized"]) + emitted
            )
            audit = action_audit_metrics(step)
            if len(online_replay) >= int(args.min_online_replay_before_updates):
                counters["updates_started"] = True
                counters["update_eligible_macro_transitions"] = int(
                    counters["update_eligible_macro_transitions"]
                ) + int(args.n_envs)
                for update_actor_and_target in scheduler.plan(args.n_envs):
                    ratio = ratio_schedule.value(int(counters["actual_primitive_transitions"]))
                    batch = sample_rlpd_batch(
                        online_replay, offline_replay, contract.batch_size, ratio, device
                    )
                    last_metrics = learner.update(
                        batch,
                        update_actor=update_actor_and_target,
                        update_target=update_actor_and_target,
                        primitive_transitions=int(counters["actual_primitive_transitions"]),
                    )
                    counters["critic_updates"] = int(counters["critic_updates"]) + 1
                    if update_actor_and_target:
                        counters["actor_updates"] = int(counters["actor_updates"]) + 1
                        counters["target_updates"] = int(counters["target_updates"]) + 1
                    last_metrics["train/offline_ratio"] = ratio
            primitive_total = int(counters["actual_primitive_transitions"])
            if primitive_total >= next_log:
                elapsed = max(time.monotonic() - progress_started_at, 1e-09)
                metrics = dict(last_metrics)
                metrics.update(
                    {
                        "phase/train": 1.0,
                        "buffer/online_size": float(len(online_replay)),
                        "buffer/offline_size": float(len(offline_replay)),
                        "buffer/pending_episode_macros": float(sum(accumulator.pending_lengths())),
                        "collector/actual_primitive_transitions": float(primitive_total),
                        "collector/macro_transitions": float(
                            counters["macro_transitions_collected"]
                        ),
                        "collector/vector_steps": float(counters["vector_steps"]),
                        "collector/mean_duration_this_vector_step": float(np.mean(step.exec_lens)),
                        "collector/candidates_this_vector_step": float(
                            np.mean(selected["candidate_count"])
                        ),
                        "updates/critic": float(counters["critic_updates"]),
                        "updates/actor": float(counters["actor_updates"]),
                        "updates/target": float(counters["target_updates"]),
                        "updates/critic_per_collected_macro": float(counters["critic_updates"])
                        / float(max(int(counters["macro_transitions_collected"]), 1)),
                        "updates/actor_per_collected_macro": float(counters["actor_updates"])
                        / float(max(int(counters["macro_transitions_collected"]), 1)),
                        "updates/critic_per_update_eligible_macro": float(
                            counters["critic_updates"]
                        )
                        / float(max(int(counters["update_eligible_macro_transitions"]), 1)),
                        "updates/actor_per_update_eligible_macro": float(counters["actor_updates"])
                        / float(max(int(counters["update_eligible_macro_transitions"]), 1)),
                        "throughput/primitive_steps_per_sec": float(primitive_total) / elapsed,
                        **rollout_tracker.metrics(),
                        **audit,
                    }
                )
                writer.add(metrics, primitive_total)
                print(
                    "[TRAIN] primitive=%d macro=%d replay=%d success200=%.4f critic=%d actor=%d critic_loss=%.6f actor_loss=%.6f"
                    % (
                        primitive_total,
                        int(counters["macro_transitions_collected"]),
                        len(online_replay),
                        rollout_tracker.metrics().get("rollout/success_rate", float("nan")),
                        int(counters["critic_updates"]),
                        int(counters["actor_updates"]),
                        last_metrics.get("train/critic_loss", float("nan")),
                        last_metrics.get("train/actor_loss", float("nan")),
                    ),
                    flush=True,
                )
                next_log = next_boundary(primitive_total, args.log_interval)
            if next_eval is not None and primitive_total >= next_eval:
                eval_metrics = evaluate(
                    evaluation_env,
                    learner,
                    episodes=args.eval_episodes,
                    base_seed=args.seed + args.eval_seed_offset,
                    eval_step=next_eval,
                    candidate_count=1,
                )
                writer.add(eval_metrics, next_eval)
                print(
                    "[EVAL] primitive=%d success_rate=%.4f mean_length=%.1f"
                    % (
                        next_eval,
                        eval_metrics["env_step_eval/success_rate"],
                        eval_metrics["env_step_eval/episode_length_mean"],
                    ),
                    flush=True,
                )
                next_eval += args.eval_interval
            while next_eval_snapshot is not None and primitive_total >= next_eval_snapshot:
                snapshot_path = save_eval_snapshot(
                    run_dir,
                    args,
                    learner,
                    snapshot_step=next_eval_snapshot,
                    actual_primitive_steps=primitive_total,
                    asset_hashes=asset_hashes,
                )
                print(
                    "[ASYNC EVAL SNAPSHOT] boundary=%d actual=%d path=%s"
                    % (next_eval_snapshot, primitive_total, snapshot_path),
                    flush=True,
                )
                next_eval_snapshot += args.eval_snapshot_interval
            if next_checkpoint is not None and primitive_total >= next_checkpoint:
                save_checkpoint(
                    os.path.join(run_dir, "latest_model.pt"),
                    args,
                    learner,
                    online_replay,
                    offline_replay,
                    scheduler,
                    counters,
                    asset_hashes,
                )
                next_checkpoint = next_boundary(primitive_total, args.checkpoint_interval)
        counters["pending_episode_macros_at_finish"] = accumulator.pending_lengths()
        save_checkpoint(
            os.path.join(run_dir, "final_model.pt"),
            args,
            learner,
            online_replay,
            offline_replay,
            scheduler,
            counters,
            asset_hashes,
        )
        atomic_json_write(
            os.path.join(run_dir, "final_status.json"),
            {
                "status": "complete",
                "counters": counters,
                "finished_utc": datetime.utcnow().isoformat() + "Z",
            },
        )
    except Exception as exc:
        atomic_json_write(
            os.path.join(run_dir, "final_status.json"),
            {
                "status": "failed",
                "error": repr(exc),
                "finished_utc": datetime.utcnow().isoformat() + "Z",
            },
        )
        raise
    finally:
        if training_env is not None:
            training_env.close()
        if evaluation_env is not None:
            evaluation_env.close()
        writer.close()
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env_name", choices=["lift", "can", "square", "transport", "tool_hang"], required=True
    )
    parser.add_argument("--agent", required=True)
    parser.add_argument("--dsrl_official_root", required=True)
    parser.add_argument("--env_meta_json", required=True)
    parser.add_argument("--dsrl_normalization_npz", required=True)
    parser.add_argument("--stats_npz_override", required=True)
    parser.add_argument("--official_demo_npz", default=None)
    parser.add_argument("--reward_hdf5", default=None)
    parser.add_argument("--offline_cache_dir", default="./dice_rl_offline_cache")
    parser.add_argument("--output_dir", default="./dice_rl_results")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--run_tag", default="")
    parser.add_argument("--offline_cache_only", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--disable_all_eval", action="store_true")
    parser.add_argument("--disable_periodic_eval", action="store_true")
    parser.add_argument(
        "--lowdim_keys", default="robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,object"
    )
    parser.add_argument("--base_horizon_steps", type=int, choices=[32, 50], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--eval_n_envs", type=int, default=4)
    parser.add_argument("--eval_seed_offset", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=None,
        help="strict task default: Lift/Can 30000, Square 100000, Transport 200000",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--prior_microbatch_size", type=int, default=512)
    parser.add_argument("--online_replay_capacity", type=int, default=60000)
    parser.add_argument("--offline_replay_capacity", type=int, default=3000)
    parser.add_argument("--min_online_replay_before_updates", type=int, default=256)
    parser.add_argument("--total_primitive_steps", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--checkpoint_interval", type=int, default=100000)
    parser.add_argument(
        "--eval_snapshot_interval",
        type=int,
        default=0,
        help="write actor+critic snapshots at absolute task eval boundaries for an external watcher; 0 disables snapshot production",
    )
    parser.add_argument(
        "--eval_snapshot_slots",
        type=int,
        default=4,
        help="bounded async-eval queue; a slot is reusable only after watcher ack",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = run_training(args)
    print("[DONE] %s" % run_dir, flush=True)


if __name__ == "__main__":
    main()
