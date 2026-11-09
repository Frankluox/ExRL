"""Train primitive- or chunk-level ResFiT on a frozen state-based DSRL DP."""

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
from exrl.runners.dp_q.policy_loader import (
    infer_action_dim_from_stats,
    infer_lowdim_dims_from_stats,
    load_dsrl_policy_wrapper,
    parse_lowdim_keys,
)
from exrl.runners.dp_q.vector_env import RobomimicExecutor
from baselines.resfit.dp.dp_queue import (
    DPChunkQueue,
    PrimitiveResidualVecEnv,
    ResidualObservation,
    ResidualStepBatch,
)
from baselines.resfit.dp.dp_chunk import (
    ChunkResidualVecEnv,
    DPChunkPlanner,
    SMDPNStepAccumulator,
    add_chunk_step_to_replay,
)
from baselines.resfit.dp.offline_dataset import (
    build_offline_replay,
    load_offline_cache,
    save_offline_cache,
    sha256_file,
)
from baselines.resfit.dp.offline_chunk_dataset import build_offline_chunk_replay
from baselines.resfit.dp.dp_task_specs import (
    expected_offline_transitions as task_expected_offline_transitions,
    get_task_contract,
)
from baselines.resfit.dp.training_window import CheckpointControl, CheckpointPause
from baselines.resfit.dp.td3_core import (
    PrimitiveNStepAccumulator,
    PrimitiveReplayBuffer,
    PrimitiveTransition,
    ResFiTActionScaler,
    ResFiTLearner,
    ResFiTLearnerConfig,
    StatePreprocessor,
    TransitionUpdateScheduler,
    sample_training_batch,
)


class MetricWriter:
    """TensorBoard when available, plus an always-present JSONL training log."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.jsonl_path = os.path.join(run_dir, "training_metrics.jsonl")
        self.tensorboard = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tensorboard = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))
        except Exception as exc:
            print("[WARN] TensorBoard unavailable: %s" % exc, flush=True)

    def add(self, metrics: Dict[str, float], step: int) -> None:
        record = {"step": int(step), "time_utc": datetime.utcnow().isoformat() + "Z"}
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
    state = {
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


def parse_int_tuple(spec: str) -> Tuple[int, ...]:
    values = tuple((int(item.strip()) for item in str(spec).split(",") if item.strip()))
    if not values or any((value <= 0 for value in values)):
        raise ValueError("hidden dims must be a comma-separated list of positive integers")
    return values


def parse_cpu_ids(spec: str) -> Tuple[int, ...]:
    values = set()
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = (int(start_text), int(end_text))
            if start < 0 or end < start:
                raise ValueError("invalid CPU affinity range %r" % token)
            values.update(range(start, end + 1))
        else:
            value = int(token)
            if value < 0:
                raise ValueError("CPU ids must be non-negative")
            values.add(value)
    return tuple(sorted(values))


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


def update_array_trace(digest, label: str, array: np.ndarray) -> None:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    digest.update(label.encode("utf-8"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())


def visible_chunk_steps(args: argparse.Namespace) -> int:
    """Return the action horizon visible to ResFiT and the executor.

    ``base_horizon_steps`` always describes the frozen DP checkpoint. A
    configured prefix is sliced immediately after DP inference, so downstream
    tensors are physically smaller and never contain a masked policy tail.
    """
    prefix = getattr(args, "base_action_prefix_steps", None)
    return int(args.base_horizon_steps if prefix is None else prefix)


def offline_contract_horizon(args: argparse.Namespace) -> int:
    if str(args.residual_granularity).lower() == "chunk":
        return visible_chunk_steps(args)
    return int(args.base_horizon_steps)


def expected_offline_transition_count(args: argparse.Namespace) -> Optional[int]:
    return None


def validate_args(args: argparse.Namespace) -> None:
    for key, default in (
        ("dp_cuda_graph", False),
        ("env_step_runtime", "parent_loop"),
        ("env_compute_mode", "legacy"),
        ("env_worker_cpu_ids", ""),
        ("cpu_affinity", ""),
        ("learner_metric_mode", "legacy"),
        ("learner_compute_mode", "legacy"),
        ("base_action_prefix_steps", None),
    ):
        if not hasattr(args, key):
            setattr(args, key, default)
    args.data_mode = str(args.data_mode).lower()
    args.residual_granularity = str(args.residual_granularity).lower()
    if args.data_mode not in {"mixed", "online"}:
        raise ValueError("data_mode must be mixed or online")
    if args.residual_granularity not in {"chunk", "primitive"}:
        raise ValueError("residual_granularity must be chunk or primitive")
    prefix_steps = args.base_action_prefix_steps
    if prefix_steps is not None:
        if args.residual_granularity != "chunk":
            raise ValueError("base_action_prefix_steps is valid only for chunk residuals")
        if int(prefix_steps) <= 0 or int(prefix_steps) > int(args.base_horizon_steps):
            raise ValueError("base_action_prefix_steps must be in [1, base_horizon_steps]")
    if args.env_compute_mode not in {"legacy", "pruned_cached"}:
        raise ValueError("env_compute_mode must be legacy or pruned_cached")
    if args.learner_compute_mode not in {"legacy", "redundant_free", "fused"}:
        raise ValueError("learner_compute_mode must be legacy, redundant_free, or fused")
    if args.env_compute_mode != "legacy" and args.env_step_runtime != "worker_chunk":
        raise ValueError("pruned_cached env compute requires worker_chunk runtime")
    worker_cpu_ids = parse_cpu_ids(args.env_worker_cpu_ids)
    process_cpu_ids = parse_cpu_ids(args.cpu_affinity)
    if args.env_worker_cpu_ids and args.env_step_runtime != "worker_chunk":
        raise ValueError("env_worker_cpu_ids requires worker_chunk runtime")
    if args.env_worker_cpu_ids and len(worker_cpu_ids) < int(args.n_envs):
        raise ValueError("env_worker_cpu_ids must provide at least n_envs CPUs")
    if args.cpu_affinity and (not process_cpu_ids):
        raise ValueError("cpu_affinity did not select any CPUs")
    if process_cpu_ids and worker_cpu_ids and (not set(worker_cpu_ids).issubset(process_cpu_ids)):
        raise ValueError("env_worker_cpu_ids must be contained in cpu_affinity")
    if args.offline_success_mode not in {"reward", "last_frame"}:
        raise ValueError("offline_success_mode must be reward or last_frame")
    if args.data_mode == "mixed":
        if not args.official_demo_npz or not args.reward_hdf5:
            raise ValueError("mixed mode requires --official_demo_npz and --reward_hdf5")
    if args.offline_cache_only and args.data_mode != "mixed":
        raise ValueError("offline_cache_only is valid only in mixed mode")
    if args.eval_only and args.data_mode != "online":
        raise ValueError("eval_only must use online mode so no demo/cache path is touched")
    if args.eval_only and args.disable_all_eval:
        raise ValueError("eval_only and disable_all_eval are mutually exclusive")
    if args.eval_only and args.offline_cache_only:
        raise ValueError("eval_only and offline_cache_only are mutually exclusive")
    if args.disable_all_eval:
        args.disable_periodic_eval = True
    if args.run_tag and (not re.fullmatch("[A-Za-z0-9_.-]+", str(args.run_tag))):
        raise ValueError("run_tag may contain only letters, digits, dot, underscore, and dash")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.data_mode == "mixed" and args.batch_size % 2:
        raise ValueError("mixed batch size must be even")
    if args.n_step <= 0 or args.n_envs <= 0 or args.eval_n_envs <= 0:
        raise ValueError("n_step and env counts must be positive")
    if args.eval_episodes <= 0 or args.eval_interval <= 0:
        raise ValueError("evaluation counts and interval must be positive")
    if args.log_interval <= 0 or args.log_update_interval <= 0 or args.checkpoint_interval < 0:
        raise ValueError("logging intervals must be positive and checkpoint interval non-negative")
    expected_offline = expected_offline_transition_count(args)
    if args.action_scale < 0.0 or args.min_action_range <= 0.0 or args.min_state_std <= 0.0:
        raise ValueError("invalid ActionScaler configuration")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if args.warmup_residual_scale < 0.0:
        raise ValueError("warmup_residual_scale must be non-negative")
    if args.exploration_std < 0.0 or args.target_noise_std < 0.0:
        raise ValueError("exploration and target noise std must be non-negative")
    if args.exploration_clip < 0.0 or args.target_noise_clip < 0.0:
        raise ValueError("exploration and target noise clip must be non-negative")
    if args.disable_target_q_clip:
        args.target_q_clip_max = None
        args.target_q_clip_min = None
    elif args.target_q_clip_min is None:
        if args.gamma >= 1.0:
            raise ValueError(
                "automatic target_q_clip_min requires gamma < 1 or --disable_target_q_clip"
            )
        if args.reward_offset < 0.0:
            raise ValueError("automatic target_q_clip_min requires non-negative reward_offset")
        args.target_q_clip_min = -float(args.reward_offset) / (1.0 - float(args.gamma))
    if args.target_q_clip_max is not None and (not np.isfinite(args.target_q_clip_max)):
        raise ValueError("target_q_clip_max must be finite or omitted")
    if args.target_q_clip_min is not None and (not np.isfinite(args.target_q_clip_min)):
        raise ValueError("target_q_clip_min must be finite or omitted")
    if (
        args.target_q_clip_min is not None
        and args.target_q_clip_max is not None
        and (args.target_q_clip_min > args.target_q_clip_max)
    ):
        raise ValueError("target_q_clip_min must not exceed target_q_clip_max")
    if args.actor_lr <= 0.0 or args.critic_lr <= 0.0:
        raise ValueError("actor and critic learning rates must be positive")
    if not 0.0 < args.tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")
    if args.actor_grad_clip <= 0.0 or args.critic_grad_clip <= 0.0:
        raise ValueError("gradient clips must be positive")
    if not 0 < args.target_critic_subsample_size <= args.critic_ensemble_size:
        raise ValueError("target critic subset must fit inside the ensemble")
    if args.critic_updates_per_transition <= 0.0:
        raise ValueError("critic_updates_per_transition must be positive")
    if not 0.0 <= args.actor_updates_per_transition <= args.critic_updates_per_transition:
        raise ValueError("actor update rate must be in [0, critic update rate]")
    online_batch_size = args.batch_size if args.data_mode == "online" else args.batch_size // 2
    if args.learning_starts < online_batch_size + args.n_envs * (args.n_step - 1):
        raise ValueError("learning_starts is too small to guarantee the first replay batch")
    if args.online_replay_capacity < online_batch_size:
        raise ValueError("online replay capacity must fit its sampled batch")
    if args.learning_starts % args.n_envs or args.total_train_steps % args.n_envs:
        raise ValueError("warmup and training transition counts must be divisible by n_envs")
    if args.critic_warmup_updates < 0 or args.total_train_steps <= 0:
        raise ValueError("update and training counts are invalid")
    if args.total_train_env_steps is not None and args.total_train_env_steps <= 0:
        raise ValueError("total_train_env_steps must be positive when provided")


def verify_formal_assets(args: argparse.Namespace) -> Dict[str, str]:
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


def load_policy_and_meta(
    args: argparse.Namespace, device: torch.device, lowdim_keys, lowdim_dims, action_dim: int
):
    wrapper, metadata = load_dsrl_policy_wrapper(
        ckpt_path=args.agent,
        device=device,
        stats_npz=args.stats_npz_override,
        lowdim_keys=lowdim_keys,
        lowdim_dims=lowdim_dims,
        action_dim=action_dim,
        cond_steps=args.cond_steps,
        weight_source=args.weight_source,
    )
    configure_dp_policy(wrapper.policy, args.dp_infer_steps)
    dp_cuda_graph = bool(getattr(args, "dp_cuda_graph", False))
    if dp_cuda_graph and device.type != "cuda":
        raise ValueError("dp_cuda_graph requires a CUDA training device")
    wrapper.policy.use_cuda_graph_trajectory = dp_cuda_graph
    if int(wrapper.meta.horizon_steps) != int(args.base_horizon_steps):
        raise RuntimeError(
            "base horizon mismatch: checkpoint=%d config=%d"
            % (int(wrapper.meta.horizon_steps), int(args.base_horizon_steps))
        )
    return (wrapper, metadata)


def make_residual_env(
    args: argparse.Namespace,
    policy,
    lowdim_keys,
    lowdim_dims,
    action_dim: int,
    n_envs: int,
    seed: int,
    scaler: ResFiTActionScaler,
    state_preprocessor: StatePreprocessor,
):
    optimized_env_compute = getattr(args, "env_compute_mode", "legacy") == "pruned_cached"
    executor = RobomimicExecutor(
        vendor_root=args.dsrl_official_root,
        env_name=args.env_name,
        env_meta_json=args.env_meta_json,
        normalization_path=args.dsrl_normalization_npz,
        lowdim_keys=lowdim_keys,
        n_envs=int(n_envs),
        max_episode_steps=int(args.env_horizon),
        seed=int(seed),
        asynchronous=True,
        terminate_on_success=True,
        timeout_on_horizon=True,
        variable_k_runtime=getattr(args, "env_step_runtime", "parent_loop"),
        prune_unused_observables=optimized_env_compute,
        worker_cpu_ids=parse_cpu_ids(getattr(args, "env_worker_cpu_ids", "")),
        cache_static_robot_metadata=optimized_env_compute,
    )
    if args.residual_granularity == "chunk":
        planner = DPChunkPlanner(
            policy=policy,
            lowdim_keys=lowdim_keys,
            lowdim_dims=lowdim_dims,
            horizon_steps=visible_chunk_steps(args),
            action_dim=int(action_dim),
            policy_horizon_steps=int(args.base_horizon_steps),
        )
        return ChunkResidualVecEnv(
            executor=executor,
            planner=planner,
            action_scaler=scaler,
            state_preprocessor=state_preprocessor,
            gamma=args.gamma,
            reward_offset=args.reward_offset,
            replay_action_mode=args.replay_action_mode,
        )
    queue = DPChunkQueue(
        policy=policy,
        lowdim_keys=lowdim_keys,
        lowdim_dims=lowdim_dims,
        n_envs=int(n_envs),
        horizon_steps=int(args.base_horizon_steps),
        action_dim=int(action_dim),
    )
    return PrimitiveResidualVecEnv(
        executor=executor,
        queue=queue,
        action_scaler=scaler,
        state_preprocessor=state_preprocessor,
        reward_offset=args.reward_offset,
        replay_action_mode=args.replay_action_mode,
    )


def add_vector_step_to_replay(
    previous: ResidualObservation,
    step: ResidualStepBatch,
    accumulator: PrimitiveNStepAccumulator,
    replay: PrimitiveReplayBuffer,
) -> int:
    if not np.allclose(
        np.asarray(step.base_action_resfit),
        np.asarray(previous.base_action_resfit),
        rtol=0.0,
        atol=2e-07,
    ):
        raise RuntimeError("replay base action drifted from the action used for execution")
    if not np.all(np.isfinite(step.replay_action_resfit)):
        raise RuntimeError("replay action contains NaN/Inf")
    emitted = 0
    for env_index in range(previous.state_actor.shape[0]):
        transition = PrimitiveTransition(
            state=previous.state_actor[env_index],
            base_action=step.base_action_resfit[env_index],
            action=step.replay_action_resfit[env_index],
            reward=float(step.rewards[env_index]),
            next_state=step.replay_next_state[env_index],
            next_base_action=step.replay_next_base_action[env_index],
            terminated=bool(step.terminated[env_index]),
            truncated=bool(step.truncated[env_index]),
        )
        emitted += replay.extend(accumulator.append(env_index, transition))
    return emitted


def flatten_policy_action(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError("batched policy action must have at least two dimensions")
    return array.reshape(array.shape[0], -1)


def add_collector_step_to_replay(
    args: argparse.Namespace, previous, step, accumulator, replay: PrimitiveReplayBuffer
) -> int:
    if args.residual_granularity == "chunk":
        return add_chunk_step_to_replay(previous, step, accumulator, replay, gamma=args.gamma)
    return add_vector_step_to_replay(previous, step, accumulator, replay)


def collector_step_primitive_count(step) -> int:
    if hasattr(step, "exec_lens"):
        return int(np.asarray(step.exec_lens, dtype=np.int64).sum())
    return int(np.asarray(step.dones).shape[0])


def collector_step_durations(step) -> np.ndarray:
    if hasattr(step, "exec_lens"):
        return np.asarray(step.exec_lens, dtype=np.int64).reshape(-1)
    return np.ones((np.asarray(step.dones).shape[0],), dtype=np.int64)


def env_plan_metrics(env) -> Dict[str, float]:
    if hasattr(env.queue, "resample_count"):
        return {
            "queue/resamples": float(env.queue.resample_count),
            "queue/bootstrap_resamples": float(env.queue.bootstrap_resample_count),
        }
    return {
        "queue/resamples": float(env.queue.sample_count),
        "queue/bootstrap_resamples": float(env.queue.bootstrap_sample_count),
    }


def offline_cache_provenance(
    args: argparse.Namespace,
    asset_hashes: Dict[str, str],
    scaler: ResFiTActionScaler,
    state_preprocessor: StatePreprocessor,
) -> Dict[str, object]:
    official_sha = sha256_file(args.official_demo_npz)
    hdf5_sha = sha256_file(args.reward_hdf5)
    return {
        "version": 4,
        "seed": int(args.seed),
        "env_name": str(args.env_name).lower(),
        "residual_granularity": args.residual_granularity,
        "official_npz_sha256": official_sha,
        "reward_hdf5_sha256": hdf5_sha,
        "checkpoint_sha256": asset_hashes["checkpoint"],
        "normalization_sha256": asset_hashes["normalization"],
        "stats_sha256": asset_hashes["stats"],
        "env_meta_sha256": asset_hashes["env_meta"],
        "dp_sampler": "ddim",
        "dp_infer_steps": int(args.dp_infer_steps),
        "base_policy_horizon_steps": int(args.base_horizon_steps),
        "visible_chunk_steps": visible_chunk_steps(args),
        "discarded_tail_steps": int(args.base_horizon_steps) - visible_chunk_steps(args),
        "n_step": int(args.n_step),
        "gamma": float(args.gamma),
        "reward_offset": float(args.reward_offset),
        "reward_aggregation": "discounted_within_chunk"
        if args.residual_granularity == "chunk"
        else "primitive",
        "stop_rule": "first_success_inclusive"
        if str(args.offline_success_mode).lower() == "reward"
        else "last_frame_inclusive",
        "offline_success_mode": str(args.offline_success_mode).lower(),
        "offline_success_horizon": int(args.env_horizon),
        "state_mode": state_preprocessor.mode,
        "state_preprocessor": state_preprocessor.manifest(),
        "action_scaler": scaler.manifest(),
    }


def validate_offline_summary(args: argparse.Namespace, summary) -> None:
    if summary.transitions <= 0:
        raise ValueError("The demonstrations produced no replay transitions")
    if getattr(summary, "action_alignment_max_abs", 0.0) > 1e-06:
        raise ValueError("Demonstration actions do not match the normalization statistics")


def maybe_build_offline_replay(
    args: argparse.Namespace,
    policy,
    lowdim_keys,
    lowdim_dims,
    state_dim: int,
    action_dim: int,
    scaler: ResFiTActionScaler,
    state_preprocessor: StatePreprocessor,
    asset_hashes: Dict[str, str],
):
    """Mixed-only boundary.  Online mode returns before touching demo paths."""
    if args.data_mode == "online":
        return (None, None, None)
    replay_action_dim = (
        int(action_dim) * visible_chunk_steps(args)
        if args.residual_granularity == "chunk"
        else int(action_dim)
    )
    replay = PrimitiveReplayBuffer(
        state_dim=state_dim,
        action_dim=replay_action_dim,
        capacity=int(args.offline_replay_capacity),
        seed=int(args.seed) + 2000,
    )
    provenance = offline_cache_provenance(args, asset_hashes, scaler, state_preprocessor)
    cache_key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode("utf-8")).hexdigest()[
        :16
    ]
    cache_path = os.path.join(args.offline_cache_dir, "seed%d_%s.pt" % (args.seed, cache_key))
    if os.path.exists(cache_path):
        print("[OFFLINE CACHE HIT] %s" % cache_path, flush=True)
        with preserved_rng_state():
            summary = load_offline_cache(cache_path, replay, provenance)
        validate_offline_summary(args, summary)
        return (replay, summary, cache_path)
    print("[OFFLINE CACHE BUILD] %s" % cache_path, flush=True)
    with preserved_rng_state():
        set_global_seed(int(args.seed) + 300000)
        if args.residual_granularity == "chunk":
            planner = DPChunkPlanner(
                policy=policy,
                lowdim_keys=lowdim_keys,
                lowdim_dims=lowdim_dims,
                horizon_steps=visible_chunk_steps(args),
                action_dim=int(action_dim),
                policy_horizon_steps=int(args.base_horizon_steps),
            )
            summary = build_offline_chunk_replay(
                official_npz_path=args.official_demo_npz,
                reward_hdf5_path=args.reward_hdf5,
                normalization_path=args.dsrl_normalization_npz,
                expected_npz_sha256=None,
                planner=planner,
                action_scaler=scaler,
                state_preprocessor=state_preprocessor,
                replay=replay,
                n_step=args.n_step,
                gamma=args.gamma,
                reward_offset=args.reward_offset,
                success_mode=args.offline_success_mode,
                success_horizon=args.env_horizon,
                expected_transition_count=None,
                precomputed_npz_sha256=provenance["official_npz_sha256"],
                precomputed_hdf5_sha256=provenance["reward_hdf5_sha256"],
            )
        else:
            queue = DPChunkQueue(
                policy=policy,
                lowdim_keys=lowdim_keys,
                lowdim_dims=lowdim_dims,
                n_envs=1,
                horizon_steps=int(args.base_horizon_steps),
                action_dim=action_dim,
            )
            summary = build_offline_replay(
                official_npz_path=args.official_demo_npz,
                reward_hdf5_path=args.reward_hdf5,
                normalization_path=args.dsrl_normalization_npz,
                expected_npz_sha256=None,
                queue=queue,
                action_scaler=scaler,
                state_preprocessor=state_preprocessor,
                replay=replay,
                n_step=args.n_step,
                gamma=args.gamma,
                reward_offset=args.reward_offset,
                success_mode=args.offline_success_mode,
                success_horizon=args.env_horizon,
                expected_transition_count=None,
                precomputed_npz_sha256=provenance["official_npz_sha256"],
                precomputed_hdf5_sha256=provenance["reward_hdf5_sha256"],
            )
    validate_offline_summary(args, summary)
    with preserved_rng_state():
        save_offline_cache(cache_path, replay, summary, provenance)
    return (replay, summary, cache_path)


def success_from_info(info: Dict[str, object]) -> bool:
    value = info.get("is_success", False)
    if isinstance(value, (list, tuple, np.ndarray)):
        return bool(value[-1]) if len(value) else False
    if isinstance(value, dict):
        return bool(value.get("task", False))
    return bool(value)


class RolloutTracker:
    """Telemetry only; never feeds values back into action selection or updates."""

    def __init__(self, n_envs: int, window: int = 200):
        self.returns = np.zeros((int(n_envs),), dtype=np.float64)
        self.lengths = np.zeros((int(n_envs),), dtype=np.int64)
        self.recent_returns = deque(maxlen=int(window))
        self.recent_lengths = deque(maxlen=int(window))
        self.recent_successes = deque(maxlen=int(window))
        self.total_episodes = 0

    def update(self, step) -> None:
        self.returns += np.asarray(step.rewards, dtype=np.float64)
        self.lengths += collector_step_durations(step)
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
    env,
    learner: ResFiTLearner,
    episodes: int,
    base_seed: int,
    eval_step: int,
    trace_out: Optional[Dict[str, object]] = None,
    check_pause=None,
) -> Dict[str, float]:
    completed = 0
    successes = 0
    lengths = np.zeros((env.n_envs,), dtype=np.int64)
    completed_lengths = []
    trace_digest = hashlib.sha256() if trace_out is not None else None
    trace_transitions = 0
    trace_decisions = 0
    max_residual_abs = 0.0
    max_zero_residual_action_error = 0.0
    with preserved_rng_state():
        set_global_seed(int(base_seed) + int(eval_step))
        observation = env.reset(base_seed=int(base_seed) + int(eval_step))
        while completed < int(episodes):
            if check_pause is not None:
                check_pause()
            residual = learner.act(
                observation.state_actor,
                flatten_policy_action(observation.base_action_resfit),
                eval_mode=True,
            )
            if trace_digest is not None:
                update_array_trace(trace_digest, "base_action_dp", observation.base_action_dp)
                update_array_trace(
                    trace_digest, "base_action_resfit", observation.base_action_resfit
                )
                update_array_trace(trace_digest, "residual_resfit", residual)
                max_residual_abs = max(max_residual_abs, float(np.abs(residual).max()))
            step = env.step(residual)
            if trace_digest is not None:
                max_zero_residual_action_error = max(
                    max_zero_residual_action_error,
                    float(np.max(np.abs(step.action_dp - observation.base_action_dp))),
                )
                update_array_trace(trace_digest, "action_dp", step.action_dp)
                update_array_trace(trace_digest, "full_action_resfit", step.full_action_resfit)
                trace_transitions += collector_step_primitive_count(step)
                trace_decisions += env.n_envs
            lengths += collector_step_durations(step)
            for env_index in np.flatnonzero(step.dones):
                if completed >= int(episodes):
                    break
                successes += int(success_from_info(step.infos[int(env_index)]))
                completed_lengths.append(int(lengths[int(env_index)]))
                lengths[int(env_index)] = 0
                completed += 1
            observation = step.observation
    if trace_out is not None:
        trace_out.update(
            {
                "sha256": trace_digest.hexdigest(),
                "primitive_transitions_hashed": int(trace_transitions),
                "decision_transitions_hashed": int(trace_decisions),
                "max_residual_abs": float(max_residual_abs),
                "max_zero_residual_action_error": float(max_zero_residual_action_error),
                "fields": [
                    "base_action_dp",
                    "base_action_resfit",
                    "residual_resfit",
                    "action_dp",
                    "full_action_resfit",
                ],
            }
        )
    return {
        "env_step_eval/success_rate": float(successes) / float(max(completed, 1)),
        "env_step_eval/successes": float(successes),
        "env_step_eval/episodes": float(completed),
        "env_step_eval/episode_length_mean": float(np.mean(completed_lengths)),
    }


def next_periodic_eval_step(
    total_env_steps: int, eval_interval: int, disable_periodic_eval: bool
) -> Optional[int]:
    """Return the next total-env-step multiple, or ``None`` when disabled.

    ``total_env_steps`` deliberately includes the online replay warmup.  The
    training loop first calls this helper after warmup, so interval multiples
    reached while warmup was active are skipped rather than evaluated or
    shifted onto a post-warmup-only axis.
    """
    if bool(disable_periodic_eval):
        return None
    return (int(total_env_steps) // int(eval_interval) + 1) * int(eval_interval)


def periodic_eval_steps_skipped_during_warmup(
    warmup_end_total_env_steps: int, eval_interval: int, disable_periodic_eval: bool
) -> list:
    """List scheduled total-env-step evals intentionally skipped in warmup."""
    if bool(disable_periodic_eval):
        return []
    return list(range(int(eval_interval), int(warmup_end_total_env_steps) + 1, int(eval_interval)))


def training_budget_remaining(args: argparse.Namespace, counters: Dict[str, object]) -> bool:
    """Stop at the first configured post-warmup decision or env-step budget."""
    if int(counters["train_decision_transitions"]) >= int(args.total_train_steps):
        return False
    if args.total_train_env_steps is not None and int(counters["train_env_steps"]) >= int(
        args.total_train_env_steps
    ):
        return False
    return True


def checkpoint_payload(
    args: argparse.Namespace,
    learner: ResFiTLearner,
    online_replay: PrimitiveReplayBuffer,
    scheduler: TransitionUpdateScheduler,
    counters: Dict[str, object],
    scaler: ResFiTActionScaler,
) -> Dict[str, object]:
    return {
        "version": 2,
        "args": vars(args),
        "learner": learner.state_dict(),
        "online_replay": online_replay.state_dict(),
        "scheduler": scheduler.state_dict(),
        "counters": dict(counters),
        "rng": capture_rng_state(),
        "action_scaler": scaler.manifest(),
        "resume_rollout_policy": "reset_env_and_clear_pending_nstep_and_dp_state",
    }


def save_checkpoint(
    path: str,
    args: argparse.Namespace,
    learner: ResFiTLearner,
    online_replay: PrimitiveReplayBuffer,
    scheduler: TransitionUpdateScheduler,
    counters: Dict[str, object],
    scaler: ResFiTActionScaler,
) -> None:
    atomic_torch_save(
        checkpoint_payload(args, learner, online_replay, scheduler, counters, scaler), path
    )


def load_checkpoint(
    path: str,
    args: argparse.Namespace,
    learner: ResFiTLearner,
    online_replay: PrimitiveReplayBuffer,
    scheduler: TransitionUpdateScheduler,
    scaler: ResFiTActionScaler,
) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    version = int(payload.get("version", -1))
    if version not in {1, 2}:
        raise RuntimeError("unsupported ResFiT checkpoint version")
    if payload.get("action_scaler") != scaler.manifest():
        raise RuntimeError("resume ActionScaler does not match the current fixed limits")
    saved_args = payload.get("args", {})
    saved_granularity = saved_args.get("residual_granularity", "primitive")
    if saved_granularity != args.residual_granularity:
        raise RuntimeError(
            "resume argument mismatch for residual_granularity: checkpoint=%r current=%r"
            % (saved_granularity, args.residual_granularity)
        )
    resume_defaults = {
        "strict_task_contract": False,
        "offline_success_mode": "reward",
        "target_q_clip_max": None,
        "target_q_clip_min": None,
        "dp_cuda_graph": False,
        "env_step_runtime": "parent_loop",
        "env_compute_mode": "legacy",
        "env_worker_cpu_ids": "",
        "learner_metric_mode": "legacy",
        "learner_compute_mode": "legacy",
        "base_action_prefix_steps": None,
    }
    for key in (
        "data_mode",
        "env_name",
        "lowdim_keys",
        "seed",
        "weight_source",
        "cond_steps",
        "reward_offset",
        "n_envs",
        "base_horizon_steps",
        "base_action_prefix_steps",
        "dp_infer_steps",
        "env_horizon",
        "n_step",
        "gamma",
        "residual_state_mode",
        "offline_success_mode",
        "replay_action_mode",
        "td_loss_mode",
        "action_scale",
        "min_action_range",
        "min_state_std",
        "warmup_residual_scale",
        "exploration_std",
        "exploration_clip",
        "target_noise_std",
        "target_noise_clip",
        "target_q_clip_max",
        "target_q_clip_min",
        "dp_cuda_graph",
        "env_step_runtime",
        "env_compute_mode",
        "env_worker_cpu_ids",
        "learner_metric_mode",
        "learner_compute_mode",
        "batch_size",
        "critic_updates_per_transition",
        "actor_updates_per_transition",
        "actor_hidden_dims",
        "critic_hidden_dims",
        "critic_ensemble_size",
        "target_critic_subsample_size",
        "critic_use_layernorm",
        "actor_lr",
        "critic_lr",
        "tau",
        "actor_grad_clip",
        "critic_grad_clip",
        "learning_starts",
        "critic_warmup_updates",
        "online_replay_capacity",
    ):
        saved_value = saved_args.get(key, resume_defaults.get(key))
        if saved_value != getattr(args, key):
            raise RuntimeError(
                "resume argument mismatch for %s: checkpoint=%r current=%r"
                % (key, saved_value, getattr(args, key))
            )
    learner.load_state_dict(payload["learner"])
    online_replay.load_state_dict(payload["online_replay"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload["rng"])
    return dict(payload["counters"])


def write_manifest(path: str, manifest: Dict[str, object]) -> None:
    with open(path, "w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)


def scaler_metrics(scaler: ResFiTActionScaler) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for index in range(scaler.action_dim):
        prefix = "scaler/dim_%d" % index
        metrics[prefix + "/raw_min"] = float(scaler.action_min[index])
        metrics[prefix + "/raw_max"] = float(scaler.action_max[index])
        metrics[prefix + "/mid"] = float(scaler.mid[index])
        metrics[prefix + "/expanded_half_range"] = float(scaler.expanded_half_range[index])
        metrics[prefix + "/actor_mean_raw_delta_max"] = float(
            scaler.action_scale * scaler.expanded_half_range[index]
        )
    return metrics


def action_audit_metrics(scaler: ResFiTActionScaler, previous, step) -> Dict[str, float]:
    base_resfit = np.asarray(step.base_action_resfit, dtype=np.float32)
    executed_residual_resfit = np.asarray(step.full_action_resfit - base_resfit, dtype=np.float32)
    base_raw = np.asarray(scaler.dp_to_raw(previous.base_action_dp), dtype=np.float32)
    raw_residual = np.asarray(step.action_raw - base_raw, dtype=np.float32)
    metrics = {
        "action/command_residual_abs_mean": float(np.abs(step.residual_resfit).mean()),
        "action/command_residual_l2_mean": float(
            np.linalg.norm(step.residual_resfit, axis=-1).mean()
        ),
        "action/executed_residual_abs_mean": float(np.abs(executed_residual_resfit).mean()),
        "action/executed_residual_l2_mean": float(
            np.linalg.norm(executed_residual_resfit, axis=-1).mean()
        ),
        "action/raw_residual_abs_mean": float(np.abs(raw_residual).mean()),
        "action/raw_residual_l2_mean": float(np.linalg.norm(raw_residual, axis=-1).mean()),
        "action/full_saturation_rate": float(np.mean(np.abs(step.full_action_resfit) >= 0.999999)),
        "action/outer_clip_rate": float(np.mean(np.abs(step.preclamp_action_resfit) > 1.0)),
        "action/raw_base_abs_mean": float(np.abs(base_raw).mean()),
    }
    if np.asarray(step.residual_resfit).ndim == 3:
        residual_flat = np.asarray(step.residual_resfit).reshape(
            np.asarray(step.residual_resfit).shape[0], -1
        )
        executed_flat = executed_residual_resfit.reshape(executed_residual_resfit.shape[0], -1)
        raw_flat = raw_residual.reshape(raw_residual.shape[0], -1)
        metrics.update(
            {
                "action/command_chunk_l2_mean": float(
                    np.linalg.norm(residual_flat, axis=-1).mean()
                ),
                "action/executed_chunk_l2_mean": float(
                    np.linalg.norm(executed_flat, axis=-1).mean()
                ),
                "action/raw_chunk_l2_mean": float(np.linalg.norm(raw_flat, axis=-1).mean()),
            }
        )
    for index in range(scaler.action_dim):
        metrics["action/raw_residual_abs_dim_%d" % index] = float(
            np.abs(raw_residual[..., index]).mean()
        )
        metrics["action/executed_residual_abs_dim_%d" % index] = float(
            np.abs(executed_residual_resfit[..., index]).mean()
        )
    return metrics


def run_training(args: argparse.Namespace) -> str:
    validate_args(args)
    if args.cpu_affinity:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("cpu_affinity requires Linux sched_setaffinity")
        os.sched_setaffinity(0, set(parse_cpu_ids(args.cpu_affinity)))
    device = torch.device(
        args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
    )
    set_global_seed(args.seed)
    asset_hashes = verify_formal_assets(args)
    lowdim_keys = parse_lowdim_keys(args.lowdim_keys)
    lowdim_dims = infer_lowdim_dims_from_stats(args.stats_npz_override, lowdim_keys)
    state_dim = int(sum(lowdim_dims.values()))
    action_dim = infer_action_dim_from_stats(args.stats_npz_override)
    task_contract = get_task_contract(args.env_name) if None else None
    policy_action_dim = (
        action_dim * visible_chunk_steps(args)
        if args.residual_granularity == "chunk"
        else action_dim
    )
    scaler = ResFiTActionScaler.from_npz(
        args.dsrl_normalization_npz,
        action_scale=args.action_scale,
        min_range_per_dim=args.min_action_range,
    )
    state_preprocessor = StatePreprocessor.from_npz(
        normalization_path=args.dsrl_normalization_npz,
        stats_path=args.stats_npz_override,
        lowdim_keys=lowdim_keys,
        mode=args.residual_state_mode,
        min_state_std=args.min_state_std,
    )
    if scaler.action_dim != action_dim or state_preprocessor.state_dim != state_dim:
        raise RuntimeError("normalization/stats dimensions disagree with the inferred task")
    policy_wrapper, policy_metadata = load_policy_and_meta(
        args, device, lowdim_keys, lowdim_dims, action_dim
    )
    optimized_learner_compute = args.learner_compute_mode != "legacy"
    learner_config = ResFiTLearnerConfig(
        state_dim=state_dim,
        action_dim=policy_action_dim,
        actor_hidden_dims=parse_int_tuple(args.actor_hidden_dims),
        critic_hidden_dims=parse_int_tuple(args.critic_hidden_dims),
        critic_ensemble_size=args.critic_ensemble_size,
        target_critic_subsample_size=args.target_critic_subsample_size,
        critic_use_layernorm=bool(args.critic_use_layernorm),
        action_scale=args.action_scale,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        actor_grad_clip=args.actor_grad_clip,
        critic_grad_clip=args.critic_grad_clip,
        target_noise_std=args.target_noise_std,
        target_noise_clip=args.target_noise_clip,
        td_loss_mode=args.td_loss_mode,
        target_q_clip_max=args.target_q_clip_max,
        target_q_clip_min=args.target_q_clip_min,
        consolidate_metric_syncs=args.learner_metric_mode != "legacy",
        target_critic_subset_compute=optimized_learner_compute,
        foreach_soft_update=optimized_learner_compute,
        freeze_critic_during_actor_update=optimized_learner_compute,
        fused_adamw=args.learner_compute_mode == "fused",
    )
    learner = ResFiTLearner(learner_config, device)
    chunk_label = "ta%d" % int(args.base_horizon_steps)
    if args.residual_granularity == "chunk" and visible_chunk_steps(args) != int(
        args.base_horizon_steps
    ):
        chunk_label += "_prefix%d" % visible_chunk_steps(args)
    run_name = "%s_%s_%s_resfit_%s_%s_s%d%s" % (
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        str(args.env_name).lower(),
        chunk_label,
        args.residual_granularity,
        args.data_mode,
        args.seed,
        "" if not args.run_tag else "_" + str(args.run_tag),
    )
    run_dir = args.persistent_run_dir or os.path.join(args.output_dir, run_name)
    if args.persistent_run_dir and os.path.isdir(run_dir):
        if not args.resume:
            raise RuntimeError("existing persistent run requires an explicit resume checkpoint")
        if os.path.realpath(args.resume) != os.path.realpath(
            os.path.join(run_dir, "latest_model.pt")
        ):
            raise RuntimeError("persistent resume must use its own latest_model.pt")
    else:
        os.makedirs(run_dir, exist_ok=False)
    writer = MetricWriter(run_dir)
    offline_replay, offline_summary, offline_cache_path = maybe_build_offline_replay(
        args=args,
        policy=policy_wrapper.policy,
        lowdim_keys=lowdim_keys,
        lowdim_dims=lowdim_dims,
        state_dim=state_dim,
        action_dim=action_dim,
        scaler=scaler,
        state_preprocessor=state_preprocessor,
        asset_hashes=asset_hashes,
    )
    if args.offline_cache_only:
        cache_manifest = {
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "status": "offline_cache_ready",
            "args": vars(args),
            "strict_task_contract": None if task_contract is None else asdict(task_contract),
            "assets": asset_hashes,
            "policy_metadata": policy_metadata,
            "action_scaler": scaler.manifest(),
            "state_preprocessor": state_preprocessor.manifest(),
            "learner": asdict(learner_config),
            "optimizers": {
                "actor": optimizer_manifest(learner.actor_optimizer),
                "critic": optimizer_manifest(learner.critic_optimizer),
            },
            "offline_summary": asdict(offline_summary),
            "offline_cache_path": offline_cache_path,
        }
        write_manifest(os.path.join(run_dir, "run_manifest.json"), cache_manifest)
        write_manifest(
            os.path.join(run_dir, "final_status.json"),
            {
                "status": "offline_cache_ready",
                "offline_cache_path": offline_cache_path,
                "finished_utc": datetime.utcnow().isoformat() + "Z",
            },
        )
        writer.close()
        print("[CACHE READY] %s" % offline_cache_path, flush=True)
        return run_dir
    training_env = make_residual_env(
        args,
        policy_wrapper.policy,
        lowdim_keys,
        lowdim_dims,
        action_dim,
        args.n_envs,
        args.seed,
        scaler,
        state_preprocessor,
    )
    evaluation_env = None
    if not args.disable_all_eval:
        evaluation_env = make_residual_env(
            args,
            policy_wrapper.policy,
            lowdim_keys,
            lowdim_dims,
            action_dim,
            args.eval_n_envs,
            args.seed + args.eval_seed_offset,
            scaler,
            state_preprocessor,
        )
        evaluation_env.bootstrap_timeouts = False
    online_replay = PrimitiveReplayBuffer(
        state_dim=state_dim,
        action_dim=policy_action_dim,
        capacity=args.online_replay_capacity,
        seed=args.seed + 1000,
    )
    accumulator = (
        SMDPNStepAccumulator(args.n_step, args.n_envs)
        if args.residual_granularity == "chunk"
        else PrimitiveNStepAccumulator(args.n_step, args.gamma, args.n_envs)
    )
    scheduler = TransitionUpdateScheduler(
        args.critic_updates_per_transition, args.actor_updates_per_transition
    )
    counters: Dict[str, object] = {
        "total_env_steps": 0,
        "train_env_steps": 0,
        "total_decision_transitions": 0,
        "train_decision_transitions": 0,
        "critic_updates": 0,
        "actor_updates": 0,
        "critic_warmup_updates": 0,
        "warmup_complete": False,
        "critic_warmup_complete": False,
    }
    if args.resume:
        counters = load_checkpoint(args.resume, args, learner, online_replay, scheduler, scaler)
        if offline_replay is not None and "offline_replay_rng" in counters:
            offline_replay.rng.set_state(counters["offline_replay_rng"])
        accumulator.reset()
        if args.residual_granularity == "primitive":
            counters.setdefault("total_decision_transitions", counters["total_env_steps"])
            counters.setdefault("train_decision_transitions", counters["train_env_steps"])
    manifest = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "device": str(device),
        "args": vars(args),
        "strict_task_contract": None if task_contract is None else asdict(task_contract),
        "assets": asset_hashes,
        "policy_metadata": policy_metadata,
        "eval_policy_metadata": None if args.disable_all_eval else policy_metadata,
        "action_scaler": scaler.manifest(),
        "state_preprocessor": state_preprocessor.manifest(),
        "learner": asdict(learner_config),
        "optimizers": {
            "actor": optimizer_manifest(learner.actor_optimizer),
            "critic": optimizer_manifest(learner.critic_optimizer),
        },
        "offline_summary": None if offline_summary is None else asdict(offline_summary),
        "offline_cache_path": offline_cache_path,
        "online_demo_data_touched": False if args.data_mode == "online" else None,
        "warmup_trace": counters.get("warmup_trace"),
        "data_contract": {
            "mode": args.data_mode,
            "offline_fraction": 0.0 if args.data_mode == "online" else 0.5,
            "online_batch_size": args.batch_size
            if args.data_mode == "online"
            else args.batch_size // 2,
            "offline_batch_size": 0 if args.data_mode == "online" else args.batch_size // 2,
            "replay_action": args.replay_action_mode,
            "offline_hdf5_state_used": False,
            "transition_granularity": args.residual_granularity,
            "base_policy_horizon_steps": int(args.base_horizon_steps),
            "visible_chunk_steps": visible_chunk_steps(args)
            if args.residual_granularity == "chunk"
            else 1,
            "policy_action_dim": int(policy_action_dim),
            "physical_action_dim": int(action_dim),
        },
        "algorithm_exclusions": [
            "variable_k",
            "cql",
            "aac",
            "entropy_alpha",
            "bc_actor_loss",
            "prioritized_replay",
        ]
        + (
            ["q_target_clip"]
            if args.target_q_clip_max is None and args.target_q_clip_min is None
            else []
        )
        + (
            ["chunk_level_residual", "smdp_backup"]
            if args.residual_granularity == "primitive"
            else ["primitive_step_residual"]
        ),
        "resume_rollout_policy": "reset_env_and_clear_pending_nstep_and_dp_state",
    }
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if args.persistent_run_dir and args.resume and os.path.isfile(manifest_path):
        write_manifest(
            os.path.join(
                run_dir, "resume_manifest_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S")
            ),
            manifest,
        )
        with open(manifest_path) as stream:
            manifest = json.load(stream)
    else:
        write_manifest(manifest_path, manifest)
    rollout_tracker = RolloutTracker(args.n_envs)
    saved_rollout = counters.get("completed_rollout_history", {})
    for field in ("recent_returns", "recent_lengths", "recent_successes"):
        getattr(rollout_tracker, field).extend(saved_rollout.get(field, []))
    rollout_tracker.total_episodes = int(saved_rollout.get("total_episodes", 0))
    control = CheckpointControl(args.stop_at_unix, args.checkpoint_on_signal)
    control.install()

    def persist(filename="latest_model.pt"):
        if offline_replay is not None:
            counters["offline_replay_rng"] = offline_replay.rng.get_state()
        counters["completed_rollout_history"] = {
            field: list(getattr(rollout_tracker, field))
            for field in ("recent_returns", "recent_lengths", "recent_successes")
        }
        counters["completed_rollout_history"]["total_episodes"] = rollout_tracker.total_episodes
        save_checkpoint(
            os.path.join(run_dir, filename),
            args,
            learner,
            online_replay,
            scheduler,
            counters,
            scaler,
        )
        write_manifest(
            os.path.join(run_dir, "latest_checkpoint_status.json"),
            {
                "saved_utc": datetime.utcnow().isoformat() + "Z",
                "total_env_steps": int(counters["total_env_steps"]),
                "train_env_steps": int(counters["train_env_steps"]),
                "online_replay_size": len(online_replay),
                "critic_updates": int(counters["critic_updates"]),
                "actor_updates": int(counters["actor_updates"]),
                "filename": filename,
                "optimizer_and_replay_saved": True,
            },
        )

    def json_counters():
        return {key: value for key, value in counters.items() if key != "offline_replay_rng"}

    progress_started_at = time.monotonic()
    session_start_train_env_steps = int(counters["train_env_steps"])
    next_checkpoint_step = (
        None
        if args.checkpoint_interval <= 0
        else next_periodic_eval_step(
            int(counters["total_env_steps"]), args.checkpoint_interval, False
        )
    )
    try:
        control.check()
        writer.add(scaler_metrics(scaler), int(counters["total_env_steps"]))
        if evaluation_env is not None and (
            not args.persistent_run_dir or not counters.get("initial_eval_complete", False)
        ):
            initial_zero_residual_trace: Dict[str, object] = {}
            eval_metrics = evaluate(
                evaluation_env,
                learner,
                episodes=args.eval_episodes,
                base_seed=args.seed + args.eval_seed_offset,
                eval_step=int(counters["total_env_steps"]),
                trace_out=initial_zero_residual_trace
                if int(counters["total_env_steps"]) == 0
                else None,
                check_pause=control.check,
            )
            counters["initial_eval_complete"] = True
            if int(counters["total_env_steps"]) == 0:
                manifest["initial_zero_residual_trace"] = initial_zero_residual_trace
                manifest["base_policy_eval"] = dict(eval_metrics)
                write_manifest(os.path.join(run_dir, "run_manifest.json"), manifest)
                if float(initial_zero_residual_trace["max_residual_abs"]) != 0.0:
                    raise RuntimeError("initial residual actor is not exactly zero")
                if float(initial_zero_residual_trace["max_zero_residual_action_error"]) != 0.0:
                    raise RuntimeError(
                        "zero residual failed to reproduce the frozen DP action: max_abs_error=%.9g"
                        % float(initial_zero_residual_trace["max_zero_residual_action_error"])
                    )
            writer.add(eval_metrics, int(counters["total_env_steps"]))
            print(
                "[EVAL] step=%d success_rate=%.4f mean_length=%.1f"
                % (
                    int(counters["total_env_steps"]),
                    eval_metrics["env_step_eval/success_rate"],
                    eval_metrics["env_step_eval/episode_length_mean"],
                ),
                flush=True,
            )
            if args.eval_only:
                write_manifest(
                    os.path.join(run_dir, "final_status.json"),
                    {
                        "status": "complete",
                        "mode": "base_policy_eval_only",
                        "base_policy_eval": dict(eval_metrics),
                        "initial_zero_residual_trace": dict(initial_zero_residual_trace),
                        "finished_utc": datetime.utcnow().isoformat() + "Z",
                    },
                )
                return run_dir
        observation = training_env.reset(base_seed=args.seed)
        if not bool(counters["warmup_complete"]):
            print(
                "[PHASE] online warmup target_%s_decisions=%d"
                % (args.residual_granularity, int(args.learning_starts)),
                flush=True,
            )
            warmup_trace_digest = hashlib.sha256()
            warmup_trace_primitive_transitions = 0
            warmup_trace_decision_transitions = 0
            next_warmup_log_step = next_periodic_eval_step(
                int(counters["total_env_steps"]), args.log_interval, False
            )
            while int(counters["total_decision_transitions"]) < int(args.learning_starts):
                control.check()
                previous = observation
                residual = (
                    (
                        (torch.rand((args.n_envs, policy_action_dim), device=device) * 2.0 - 1.0)
                        * args.warmup_residual_scale
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                step = training_env.step(residual)
                rollout_tracker.update(step)
                update_array_trace(warmup_trace_digest, "base_action_dp", previous.base_action_dp)
                update_array_trace(warmup_trace_digest, "residual_resfit", residual)
                update_array_trace(warmup_trace_digest, "action_dp", step.action_dp)
                update_array_trace(
                    warmup_trace_digest, "full_action_resfit", step.full_action_resfit
                )
                primitive_count = collector_step_primitive_count(step)
                warmup_trace_primitive_transitions += primitive_count
                warmup_trace_decision_transitions += args.n_envs
                add_collector_step_to_replay(args, previous, step, accumulator, online_replay)
                observation = step.observation
                counters["total_env_steps"] = int(counters["total_env_steps"]) + primitive_count
                counters["total_decision_transitions"] = (
                    int(counters["total_decision_transitions"]) + args.n_envs
                )
                audit_metrics = action_audit_metrics(scaler, previous, step)
                counters["last_action_audit"] = audit_metrics
                if int(counters["total_env_steps"]) >= int(next_warmup_log_step):
                    writer.add(
                        {
                            "phase/warmup": 1.0,
                            "buffer/online_size": float(len(online_replay)),
                            "collector/decision_transitions": float(
                                counters["total_decision_transitions"]
                            ),
                            **rollout_tracker.metrics(),
                            **audit_metrics,
                            **env_plan_metrics(training_env),
                        },
                        int(counters["total_env_steps"]),
                    )
                    next_warmup_log_step = next_periodic_eval_step(
                        int(counters["total_env_steps"]), args.log_interval, False
                    )
                if (
                    next_checkpoint_step is not None
                    and int(counters["total_env_steps"]) >= next_checkpoint_step
                ):
                    persist()
                    next_checkpoint_step = next_periodic_eval_step(
                        int(counters["total_env_steps"]), args.checkpoint_interval, False
                    )
            counters["warmup_trace"] = {
                "sha256": warmup_trace_digest.hexdigest(),
                "primitive_transitions_hashed": int(warmup_trace_primitive_transitions),
                "decision_transitions_hashed": int(warmup_trace_decision_transitions),
                "fields": ["base_action_dp", "residual_resfit", "action_dp", "full_action_resfit"],
            }
            counters["warmup_complete"] = True
            manifest["warmup_trace"] = counters["warmup_trace"]
            if evaluation_env is not None and (not args.disable_periodic_eval):
                warmup_end_total_env_steps = int(counters["total_env_steps"])
                manifest["evaluation_schedule"] = {
                    "axis": "total_primitive_env_steps_including_online_warmup",
                    "initial_eval_step": 0,
                    "interval": int(args.eval_interval),
                    "warmup_end_total_env_steps": warmup_end_total_env_steps,
                    "skipped_periodic_steps_during_warmup": periodic_eval_steps_skipped_during_warmup(
                        warmup_end_total_env_steps, args.eval_interval, False
                    ),
                    "first_post_warmup_periodic_step": next_periodic_eval_step(
                        warmup_end_total_env_steps, args.eval_interval, False
                    ),
                }
            write_manifest(os.path.join(run_dir, "run_manifest.json"), manifest)
            print(
                "[PHASE COMPLETE] online warmup decisions=%d env_steps=%d replay=%d"
                % (
                    int(counters["total_decision_transitions"]),
                    int(counters["total_env_steps"]),
                    len(online_replay),
                ),
                flush=True,
            )
        if not bool(counters["critic_warmup_complete"]):
            print(
                "[PHASE] critic-only warmup target_updates=%d" % int(args.critic_warmup_updates),
                flush=True,
            )
            for update_index in range(
                int(counters["critic_warmup_updates"]), int(args.critic_warmup_updates)
            ):
                control.check()
                batch = sample_training_batch(
                    online_replay, offline_replay, args.data_mode, args.batch_size, device
                )
                collect_metrics = (
                    args.learner_metric_mode != "sparse"
                    or (update_index + 1) % args.log_update_interval == 0
                )
                metrics = learner.update(batch, update_actor=False, collect_metrics=collect_metrics)
                counters["critic_warmup_updates"] = update_index + 1
                counters["critic_updates"] = int(counters["critic_updates"]) + 1
                if (update_index + 1) % args.log_update_interval == 0:
                    metrics.update({"phase/critic_warmup": 1.0})
                    writer.add(metrics, int(counters["total_env_steps"]))
                    print(
                        "[CRITIC WARMUP] update=%d/%d loss=%.6f q=%.4f target_q=%.4f"
                        % (
                            update_index + 1,
                            int(args.critic_warmup_updates),
                            metrics["train/critic_loss"],
                            metrics["train/q_mean"],
                            metrics["train/target_q_mean"],
                        ),
                        flush=True,
                    )
            counters["critic_warmup_complete"] = True
            print("[PHASE COMPLETE] critic-only warmup", flush=True)
        next_eval_step = next_periodic_eval_step(
            int(counters["total_env_steps"]), args.eval_interval, args.disable_periodic_eval
        )
        if counters.get("pending_eval_step") is not None:
            next_eval_step = int(counters["pending_eval_step"])
        next_log_step = next_periodic_eval_step(
            int(counters["total_env_steps"]), args.log_interval, False
        )
        last_metrics: Dict[str, float] = {}
        while training_budget_remaining(args, counters):
            control.check()
            previous = observation
            residual = learner.act(
                observation.state_actor,
                flatten_policy_action(observation.base_action_resfit),
                eval_mode=False,
                exploration_std=args.exploration_std,
                exploration_clip=args.exploration_clip,
            )
            step = training_env.step(residual)
            rollout_tracker.update(step)
            add_collector_step_to_replay(args, previous, step, accumulator, online_replay)
            observation = step.observation
            primitive_count = collector_step_primitive_count(step)
            counters["total_env_steps"] = int(counters["total_env_steps"]) + primitive_count
            counters["train_env_steps"] = int(counters["train_env_steps"]) + primitive_count
            counters["total_decision_transitions"] = (
                int(counters["total_decision_transitions"]) + args.n_envs
            )
            counters["train_decision_transitions"] = (
                int(counters["train_decision_transitions"]) + args.n_envs
            )
            audit_metrics = action_audit_metrics(scaler, previous, step)
            counters["last_action_audit"] = audit_metrics
            update_plan = scheduler.plan(args.n_envs)
            will_log = int(counters["total_env_steps"]) >= int(next_log_step)
            for update_index, update_actor in enumerate(update_plan):
                batch = sample_training_batch(
                    online_replay, offline_replay, args.data_mode, args.batch_size, device
                )
                collect_metrics = args.learner_metric_mode != "sparse" or (
                    will_log and update_index == len(update_plan) - 1
                )
                update_metrics = learner.update(
                    batch, update_actor=update_actor, collect_metrics=collect_metrics
                )
                if update_metrics:
                    last_metrics = update_metrics
                counters["critic_updates"] = int(counters["critic_updates"]) + 1
                if update_actor:
                    counters["actor_updates"] = int(counters["actor_updates"]) + 1
            total_env_steps = int(counters["total_env_steps"])
            if total_env_steps >= int(next_log_step):
                last_metrics.update(
                    {
                        "phase/train": 1.0,
                        "buffer/online_size": float(len(online_replay)),
                        "buffer/offline_size": float(
                            0 if offline_replay is None else len(offline_replay)
                        ),
                        "collector/decision_transitions": float(
                            counters["total_decision_transitions"]
                        ),
                        "collector/train_decision_transitions": float(
                            counters["train_decision_transitions"]
                        ),
                        "collector/train_env_steps": float(counters["train_env_steps"]),
                        "collector/mean_duration_this_vector_step": float(
                            np.mean(collector_step_durations(step))
                        ),
                        **rollout_tracker.metrics(),
                        **audit_metrics,
                        **env_plan_metrics(training_env),
                        "updates/critic": float(counters["critic_updates"]),
                        "updates/actor": float(counters["actor_updates"]),
                    }
                )
                writer.add(last_metrics, total_env_steps)
                next_log_step = next_periodic_eval_step(total_env_steps, args.log_interval, False)
                elapsed = max(time.monotonic() - progress_started_at, 1e-09)
                print(
                    "[TRAIN] total_env_step=%d train_env_step=%d train_decisions=%d success200=%.4f critic_loss=%.6f actor_loss=%.6f env_steps_per_sec=%.2f"
                    % (
                        total_env_steps,
                        int(counters["train_env_steps"]),
                        int(counters["train_decision_transitions"]),
                        rollout_tracker.metrics().get("rollout/success_rate", float("nan")),
                        last_metrics.get("train/critic_loss", float("nan")),
                        last_metrics.get("train/actor_loss", float("nan")),
                        (int(counters["train_env_steps"]) - session_start_train_env_steps)
                        / elapsed,
                    ),
                    flush=True,
                )
            if next_eval_step is not None and total_env_steps >= next_eval_step:
                counters["pending_eval_step"] = int(next_eval_step)
                eval_metrics = evaluate(
                    evaluation_env,
                    learner,
                    episodes=args.eval_episodes,
                    base_seed=args.seed + args.eval_seed_offset,
                    eval_step=next_eval_step,
                    check_pause=control.check,
                )
                writer.add(eval_metrics, next_eval_step)
                print(
                    "[EVAL] step=%d success_rate=%.4f mean_length=%.1f"
                    % (
                        next_eval_step,
                        eval_metrics["env_step_eval/success_rate"],
                        eval_metrics["env_step_eval/episode_length_mean"],
                    ),
                    flush=True,
                )
                next_eval_step += args.eval_interval
                counters["pending_eval_step"] = None
            if next_checkpoint_step is not None and total_env_steps >= next_checkpoint_step:
                persist()
                next_checkpoint_step = next_periodic_eval_step(
                    total_env_steps, args.checkpoint_interval, False
                )
        persist("latest_model.pt" if args.persistent_run_dir else "final_model.pt")
        write_manifest(
            os.path.join(run_dir, "final_status.json"),
            {
                "status": "complete",
                "counters": json_counters(),
                "finished_utc": datetime.utcnow().isoformat() + "Z",
            },
        )
    except CheckpointPause as exc:
        print("[PAUSE] saving complete latest checkpoint: %s" % exc, flush=True)
        persist()
        write_manifest(
            os.path.join(run_dir, "final_status.json"),
            {
                "status": "paused",
                "reason": str(exc),
                "counters": json_counters(),
                "finished_utc": datetime.utcnow().isoformat() + "Z",
            },
        )
        print(
            "[PAUSE SAVED] step=%d replay=%d" % (counters["total_env_steps"], len(online_replay)),
            flush=True,
        )
    except Exception as exc:
        write_manifest(
            os.path.join(run_dir, "final_status.json"),
            {
                "status": "failed",
                "error": repr(exc),
                "finished_utc": datetime.utcnow().isoformat() + "Z",
            },
        )
        raise
    finally:
        training_env.close()
        if evaluation_env is not None:
            evaluation_env.close()
        writer.close()
        control.close()
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_mode", choices=["mixed", "online"], required=True)
    parser.add_argument(
        "--residual_granularity",
        choices=["chunk", "primitive"],
        default="chunk",
        help="chunk uses one complete fixed-TA macro action; primitive preserves the one-step adapter",
    )
    parser.add_argument("--agent", required=True)
    parser.add_argument("--dsrl_official_root", required=True)
    parser.add_argument("--env_meta_json", required=True)
    parser.add_argument("--dsrl_normalization_npz", required=True)
    parser.add_argument("--stats_npz_override", required=True)
    parser.add_argument("--official_demo_npz", default=None)
    parser.add_argument("--reward_hdf5", default=None)
    parser.add_argument("--offline_cache_dir", default="./resfit_offline_cache")
    parser.add_argument("--output_dir", default="./resfit_results")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--persistent_run_dir", default=None)
    parser.add_argument("--checkpoint_on_signal", action="store_true")
    parser.add_argument("--stop_at_unix", type=float, default=None)
    parser.add_argument("--run_tag", default="")
    parser.add_argument("--offline_cache_only", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--disable_all_eval",
        action="store_true",
        help="do not construct an evaluation env or run initial/periodic eval",
    )
    parser.add_argument(
        "--disable_periodic_eval",
        action="store_true",
        help="keep the initial safety eval but use rollout metrics during training",
    )
    parser.add_argument("--env_name", default="square")
    parser.add_argument(
        "--lowdim_keys", default="robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,object"
    )
    parser.add_argument("--cond_steps", type=int, default=1)
    parser.add_argument("--weight_source", default="ema")
    parser.add_argument("--base_horizon_steps", type=int, default=32)
    parser.add_argument(
        "--base_action_prefix_steps",
        type=int,
        default=None,
        help="expose and execute only this many leading actions from the frozen DP chunk; discard the remaining actions before actor, critic, replay, or executor can observe them",
    )
    parser.add_argument("--dp_infer_steps", type=int, default=20)
    parser.add_argument(
        "--dp_cuda_graph",
        action="store_true",
        help="opt into exact deterministic-DDIM CUDA Graph replay for A/B validation",
    )
    parser.add_argument("--env_horizon", type=int, default=400)
    parser.add_argument("--reward_offset", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument(
        "--env_step_runtime",
        choices=["parent_loop", "worker_chunk"],
        default="parent_loop",
        help="opt into one chunk RPC per env while preserving primitive MultiStep.step calls",
    )
    parser.add_argument(
        "--env_compute_mode",
        choices=["legacy", "pruned_cached"],
        default="legacy",
        help="prune unused robosuite observables and cache immutable robot metadata",
    )
    parser.add_argument(
        "--env_worker_cpu_ids",
        default="",
        help="optional comma/range CPU list assigned one per async env worker",
    )
    parser.add_argument(
        "--cpu_affinity",
        default="",
        help="optional comma/range CPU pool inherited by the trainer and env workers",
    )
    parser.add_argument("--eval_n_envs", type=int, default=4)
    parser.add_argument("--eval_seed_offset", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=10000)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--residual_state_mode",
        choices=["stats_no_clip", "dsrl_raw", "resfit_raw_zscore"],
        default="stats_no_clip",
    )
    parser.add_argument("--action_scale", type=float, default=0.1)
    parser.add_argument("--min_action_range", type=float, default=0.1)
    parser.add_argument("--min_state_std", type=float, default=0.1)
    parser.add_argument(
        "--replay_action_mode",
        choices=["executed_clamped", "release_preclamp"],
        default="executed_clamped",
    )
    parser.add_argument(
        "--td_loss_mode",
        choices=["per_head_mse", "release_mean_abs_square"],
        default="per_head_mse",
    )
    parser.add_argument("--actor_hidden_dims", default="2048,2048,2048")
    parser.add_argument("--critic_hidden_dims", default="512,512,512")
    parser.add_argument("--critic_ensemble_size", type=int, default=10)
    parser.add_argument("--target_critic_subsample_size", type=int, default=2)
    parser.add_argument("--critic_use_layernorm", type=int, choices=[0, 1], default=1)
    parser.add_argument("--actor_lr", type=float, default=1e-06)
    parser.add_argument("--critic_lr", type=float, default=0.0001)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor_grad_clip", type=float, default=1.0)
    parser.add_argument("--critic_grad_clip", type=float, default=1.0)
    parser.add_argument("--target_noise_std", type=float, default=0.05)
    parser.add_argument("--target_noise_clip", type=float, default=0.3)
    parser.add_argument(
        "--target_q_clip_max",
        type=float,
        default=0.0,
        help="upper bound on the complete Bellman target (default: 0)",
    )
    parser.add_argument(
        "--target_q_clip_min",
        type=float,
        default=None,
        help="lower bound on the complete Bellman target (default: -reward_offset / (1 - gamma))",
    )
    parser.add_argument(
        "--disable_target_q_clip",
        action="store_true",
        help="disable both target-Q bounds, including the default bounds",
    )
    parser.add_argument(
        "--learner_metric_mode",
        choices=["legacy", "consolidated", "sparse"],
        default="legacy",
        help="A/B mode for learner metric GPU synchronization; sparse logs only when consumed",
    )
    parser.add_argument(
        "--learner_compute_mode",
        choices=["legacy", "redundant_free", "fused"],
        default="legacy",
        help="redundant_free selects target critics before evaluation, uses foreach Polyak, and freezes critic weights during actor backward; fused additionally uses fused AdamW",
    )
    parser.add_argument("--exploration_std", type=float, default=0.05)
    parser.add_argument("--exploration_clip", type=float, default=0.3)
    parser.add_argument("--warmup_residual_scale", type=float, default=0.2)
    parser.add_argument(
        "--offline_success_mode",
        choices=["reward", "last_frame"],
        default="reward",
        help="reward uses first positive sparse reward; last_frame matches ResFiT's no-dones converter fallback",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n_step", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--online_replay_capacity", type=int, default=200000)
    parser.add_argument("--offline_replay_capacity", type=int, default=100000)
    parser.add_argument("--learning_starts", type=int, default=10000)
    parser.add_argument("--critic_warmup_updates", type=int, default=10000)
    parser.add_argument("--critic_updates_per_transition", type=float, default=4.0)
    parser.add_argument("--actor_updates_per_transition", type=float, default=1.0)
    parser.add_argument(
        "--total_train_steps",
        type=int,
        default=300000,
        help="post-warmup policy transitions at the selected residual granularity",
    )
    parser.add_argument(
        "--total_train_env_steps",
        type=int,
        default=None,
        help="optional post-warmup primitive env-step budget; the first active budget reached stops training",
    )
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--log_update_interval", type=int, default=100)
    parser.add_argument("--checkpoint_interval", type=int, default=50000)
    parser.add_argument("--expected_offline_episodes", type=int, default=300)
    parser.add_argument("--expected_offline_transitions", type=int, default=None)
    parser.add_argument("--expected_late_success_episodes", type=int, default=None)
    parser.add_argument("--expected_offline_partial_chunks", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = run_training(args)
    print("[DONE] %s" % run_dir, flush=True)


if __name__ == "__main__":
    main()
