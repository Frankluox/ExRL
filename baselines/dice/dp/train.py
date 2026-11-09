"""Train DICE-RL with frozen action chunks and a noise-conditioned residual."""

from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import random
import re
import signal
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

_CHECKPOINT_STOP_SIGNAL: Optional[int] = None


def _request_checkpoint_stop(signum, _frame) -> None:
    """Request a checkpointed stop at the next safe training boundary."""
    global _CHECKPOINT_STOP_SIGNAL
    _CHECKPOINT_STOP_SIGNAL = int(signum)


def install_checkpoint_stop_handler():
    """Install the SIGUSR1 handler and reset stale in-process state."""
    global _CHECKPOINT_STOP_SIGNAL
    _CHECKPOINT_STOP_SIGNAL = None
    previous = signal.getsignal(signal.SIGUSR1)
    signal.signal(signal.SIGUSR1, _request_checkpoint_stop)
    return previous


def checkpoint_stop_signal() -> Optional[int]:
    return _CHECKPOINT_STOP_SIGNAL


@contextlib.contextmanager
def serialized_gpu_updates(lock_path: Optional[str]):
    """Serialize update bursts sharing one nearly-full physical GPU."""
    if not lock_path:
        yield
        return
    parent = os.path.dirname(os.path.abspath(lock_path))
    os.makedirs(parent, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 436)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


REPO_ROOT = __import__("exrl.paths", fromlist=["ROOT"]).ROOT
from baselines.dice.dp.can_task_specs import (
    expected_offline_transitions,
    get_task_contract,
    validate_horizon,
)
from baselines.dice.dp.core import (
    CompleteEpisodeAccumulator,
    DICERLLearner,
    DICERLLearnerConfig,
    DICEReplayBuffer,
    LinearOfflineRatioSchedule,
    TransitionUpdateScheduler,
    sample_rlpd_batch,
)
from baselines.dice.dp.dp_chunk import (
    DICEChunkVecEnv,
    action_audit_metrics,
    add_chunk_step_to_replay,
)
from baselines.dice.dp.dsrl_ckpt_loader import (
    infer_action_dim_from_stats,
    infer_lowdim_dims_from_stats,
    load_dsrl_policy_wrapper,
    parse_lowdim_keys,
)
from baselines.dice.dp.offline_chunk_dataset import build_offline_chunk_replay
from baselines.dice.dp.offline_dataset_utils import (
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
    if getattr(args, "disable_resume_start_eval", False) and (not args.resume):
        raise ValueError("disable_resume_start_eval requires --resume")
    if args.allow_optimization_resume_migration and (not args.resume):
        raise ValueError("optimization resume migration requires --resume")
    if args.allow_optimization_resume_migration and args.optimization_level == "baseline":
        raise ValueError("optimization resume migration requires phase1 or phase2")
    if args.eval_only and args.disable_all_eval:
        raise ValueError("eval_only and disable_all_eval are mutually exclusive")
    if args.optimization_level not in ("baseline", "phase1", "phase2"):
        raise ValueError("invalid optimization_level")
    if int(args.optimized_prior_microbatch_size) <= 0:
        raise ValueError("optimized_prior_microbatch_size must be positive")
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
    if args.eval_snapshot_interval > 0:
        args.disable_periodic_eval = True
    if not args.eval_only:
        pass


def verify_assets(args: argparse.Namespace) -> Dict[str, str]:
    paths = {"checkpoint": args.agent}
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


def configure_dp_policy(policy, infer_steps: int) -> None:
    policy.algo_config.ddpm.enabled = False
    policy.algo_config.ddim.enabled = True
    policy.algo_config.ddim.num_inference_timesteps = int(infer_steps)
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
    wrapper, metadata = load_dsrl_policy_wrapper(
        ckpt_path=args.agent,
        device=device,
        stats_npz=args.stats_npz_override,
        lowdim_keys=list(lowdim_keys),
        lowdim_dims=lowdim_dims,
        action_dim=action_dim,
        cond_steps=1,
        weight_source="ema",
    )
    configure_dp_policy(wrapper.policy, contract.dp_infer_steps)
    if int(wrapper.meta.horizon_steps) != int(args.base_horizon_steps):
        raise RuntimeError(
            "base horizon mismatch: checkpoint=%d config=%d"
            % (int(wrapper.meta.horizon_steps), int(args.base_horizon_steps))
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
    args: argparse.Namespace, asset_hashes: Dict[str, str]
) -> Dict[str, object]:
    contract = get_task_contract(args.env_name)
    return {
        "version": 1,
        "env_name": contract.env_name,
        "all_demo_episodes": contract.offline_episodes,
        "offline_primitive_transitions": contract.offline_primitive_transitions,
        "official_npz_sha256": asset_hashes["official_demo"],
        "reward_hdf5_sha256": asset_hashes["reward_hdf5"],
        "normalization_sha256": asset_hashes["normalization"],
        "checkpoint_sha256": asset_hashes["checkpoint"],
        "suffix_noise_seed": int(args.seed) + 20000,
        "dp_sampler": "ddim_eta0_explicit_initial_noise",
        "dp_infer_steps": contract.dp_infer_steps,
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
    args: argparse.Namespace, policy, state_dim: int, action_dim: int, asset_hashes: Dict[str, str]
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
            prior_sampler=policy,
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
            expected_primitive_transition_count=None,
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
    asset_hashes: Dict[str, str],
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
    asset_hashes: Dict[str, str],
) -> None:
    atomic_torch_save(
        checkpoint_payload(
            args, learner, online_replay, offline_replay, scheduler, counters, asset_hashes
        ),
        path,
    )


def cpu_module_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Copy one module to CPU without serializing optimizer or replay state."""
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def eval_snapshot_payload(
    args: argparse.Namespace,
    learner: DICERLLearner,
    snapshot_step: int,
    actual_primitive_steps: int,
    asset_hashes: Dict[str, str],
) -> Dict[str, object]:
    return {
        "snapshot_format": "dice_rl_dp_actor_critic_eval_v1",
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
    asset_hashes: Dict[str, str],
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
                "refusing to duplicate DICE eval snapshot step %d" % int(snapshot_step)
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
            "snapshot_format": "dice_rl_dp_actor_critic_eval_v1",
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
    asset_hashes: Dict[str, str],
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
    learner.load_state_dict(
        payload["learner"], allow_runtime_migration=args.allow_optimization_resume_migration
    )
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
        prior_microbatch_size=args.prior_microbatch_size
        if args.optimization_level == "baseline"
        else args.optimized_prior_microbatch_size,
        optimization_level=args.optimization_level,
        phase2_amp=getattr(args, "phase2_amp_enabled", not args.phase2_disable_amp),
        phase2_compile=getattr(args, "phase2_compile_enabled", not args.phase2_disable_compile),
    )


def resolve_phase2_runtime_options(
    args: argparse.Namespace, contract, device: torch.device
) -> Tuple[bool, bool]:
    """Choose the measured fast path while retaining explicit ablations."""
    if args.optimization_level != "phase2" or device.type != "cuda":
        return (False, False)
    major, _minor = torch.cuda.get_device_capability(device)
    wide_network = int(contract.actor_hidden_dims[0]) >= 2048
    amp_enabled = (wide_network or major < 8) and (not args.phase2_disable_amp)
    compile_enabled = wide_network and (not args.phase2_disable_compile)
    return (amp_enabled, compile_enabled)


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
    if args.optimization_level == "phase2" and device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args.phase2_amp_enabled, args.phase2_compile_enabled = resolve_phase2_runtime_options(
        args, contract, device
    )
    set_global_seed(args.seed)
    asset_hashes = verify_assets(args)
    policy_wrapper, policy_metadata, lowdim_keys, _lowdim_dims, state_dim, action_dim = (
        load_policy_and_dimensions(args, device)
    )
    policy_wrapper.policy.configure_runtime_optimization(
        args.optimization_level,
        phase2_amp=args.phase2_amp_enabled,
        phase2_compile=args.phase2_compile_enabled,
    )
    config = learner_config(args)
    learner = DICERLLearner(config, policy_wrapper.policy, device)
    zero_state = torch.zeros((2, state_dim), dtype=torch.float32, device=device)
    zero_noise = torch.zeros(
        (2, action_dim * int(args.base_horizon_steps)), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        initial_residual = learner.actor(zero_state, zero_noise)
    if not torch.equal(initial_residual, torch.zeros_like(initial_residual)):
        raise RuntimeError("DICE residual actor is not exactly zero at initialization")
    run_name = "%s_%s_ta%d_dice_dp_s%d%s" % (
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        contract.env_name,
        int(args.base_horizon_steps),
        int(args.seed),
        "" if not args.run_tag else "_" + str(args.run_tag),
    )
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=False)
    writer = MetricWriter(run_dir)
    previous_checkpoint_stop_handler = install_checkpoint_stop_handler()
    training_env = None
    evaluation_env = None
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    offline_replay = None
    offline_summary = None
    offline_cache_path = None
    offline_provenance = None
    try:
        offline_replay, offline_summary, offline_cache_path, offline_provenance = (
            maybe_build_offline_replay(
                args, policy_wrapper.policy, state_dim, action_dim, asset_hashes
            )
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
        if args.optimization_level != "baseline" and device.type == "cuda":
            offline_replay.enable_device_cache(device)
            online_replay.enable_device_cache(device)
        base_manifest["resumed_from"] = args.resume
        base_manifest["external_eval"] = {
            "enabled": bool(args.eval_snapshot_interval > 0),
            "watcher_required": bool(args.eval_snapshot_interval > 0),
            "snapshot_format": "dice_rl_dp_actor_critic_eval_v1",
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
            initial_eval = None
            if args.resume and args.disable_resume_start_eval:
                base_manifest["resume_start_eval"] = {
                    "status": "disabled_for_scheduled_day_night_resume",
                    "resume_step": int(counters["actual_primitive_transitions"]),
                }
            elif args.resume:
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
                    raise RuntimeError("initial DICE policy does not equal the frozen DP")
                if float(trace["max_action_minus_base_minus_residual_abs"]) > 2e-06:
                    raise RuntimeError("DICE action identity audit failed")
                base_manifest["initial_frozen_dp_trace"] = trace
                base_manifest["base_policy_eval"] = dict(initial_eval)
            atomic_json_write(manifest_path, base_manifest)
            if initial_eval is not None:
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
                if initial_eval is None:
                    raise RuntimeError("eval_only cannot disable its resume-start evaluation")
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
                update_plan = scheduler.plan(args.n_envs)
                with serialized_gpu_updates(args.gpu_update_lock):
                    for update_index, update_actor_and_target in enumerate(update_plan):
                        ratio = ratio_schedule.value(int(counters["actual_primitive_transitions"]))
                        batch = sample_rlpd_batch(
                            online_replay,
                            offline_replay,
                            contract.batch_size,
                            ratio,
                            device,
                            use_device_cache=args.optimization_level != "baseline",
                        )
                        collect_metrics = args.optimization_level == "baseline" or (
                            int(counters["actual_primitive_transitions"]) >= next_log
                            and update_index == len(update_plan) - 1
                        )
                        update_metrics = learner.update(
                            batch,
                            update_actor=update_actor_and_target,
                            update_target=update_actor_and_target,
                            primitive_transitions=int(counters["actual_primitive_transitions"]),
                            collect_metrics=collect_metrics,
                        )
                        counters["critic_updates"] = int(counters["critic_updates"]) + 1
                        if update_actor_and_target:
                            counters["actor_updates"] = int(counters["actor_updates"]) + 1
                            counters["target_updates"] = int(counters["target_updates"]) + 1
                        if update_metrics:
                            last_metrics = update_metrics
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
            if checkpoint_stop_signal() is not None:
                break
        stop_signum = checkpoint_stop_signal()
        if stop_signum is not None and int(counters["actual_primitive_transitions"]) < int(
            args.total_primitive_steps
        ):
            counters["pending_episode_macros_at_pause"] = accumulator.pending_lengths()
            pause_checkpoint = os.path.join(run_dir, "latest_model.pt")
            print(
                "[CHECKPOINT STOP] primitive=%d signal=%s saving=%s"
                % (
                    int(counters["actual_primitive_transitions"]),
                    signal.Signals(stop_signum).name,
                    pause_checkpoint,
                ),
                flush=True,
            )
            save_checkpoint(
                pause_checkpoint,
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
                    "status": "paused_after_checkpoint",
                    "signal": signal.Signals(stop_signum).name,
                    "checkpoint": pause_checkpoint,
                    "counters": counters,
                    "finished_utc": datetime.utcnow().isoformat() + "Z",
                },
            )
            print(
                "[PAUSED] primitive=%d checkpoint=%s"
                % (int(counters["actual_primitive_transitions"]), pause_checkpoint),
                flush=True,
            )
            return run_dir
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
        signal.signal(signal.SIGUSR1, previous_checkpoint_stop_handler)
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
    parser.add_argument(
        "--allow_optimization_resume_migration",
        action="store_true",
        help="explicitly migrate a legacy baseline critic and Adam state into the phase1/phase2 batched representation",
    )
    parser.add_argument(
        "--disable_resume_start_eval",
        action="store_true",
        help="skip the otherwise automatic audit eval at an exact scheduled day/night resume; absolute periodic eval boundaries are unchanged",
    )
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
        help="strict task default: Lift/Can 30000, Square 100000, Transport/ToolHang 200000",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--gpu_update_lock",
        default=None,
        help="optional per-physical-GPU flock path; serializes complete update bursts without changing batch size, UTD, or learner equations",
    )
    parser.add_argument("--prior_microbatch_size", type=int, default=512)
    parser.add_argument(
        "--optimization_level",
        choices=["baseline", "phase1", "phase2"],
        default="baseline",
        help="baseline preserves the original path; phase1 enables exact-FP32 runtime optimizations; phase2 adds compiled mixed precision",
    )
    parser.add_argument(
        "--optimized_prior_microbatch_size",
        type=int,
        default=4096,
        help="phase1/phase2 frozen-prior microbatch size",
    )
    parser.add_argument(
        "--phase2_disable_amp",
        action="store_true",
        help="ablation switch: keep phase2 TF32/compile but use FP32 tensors",
    )
    parser.add_argument(
        "--phase2_disable_compile",
        action="store_true",
        help="ablation switch: keep phase2 TF32/AMP in eager mode",
    )
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
