"""Train ExRL with a DP base policy and Q updates."""

import argparse
import atexit
import fcntl
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecMonitor
from torch.utils.tensorboard import SummaryWriter

from exrl.paths import ROOT as REPO_ROOT

import robomimic.utils.torch_utils as TorchUtils
from exrl.runners.dp_demo.environment import ExecutionLengthEnv
from exrl.runners.dp_demo.policy_loader import (
    infer_action_dim_from_stats,
    infer_lowdim_dims_from_stats,
    load_dsrl_policy_wrapper,
    load_env_meta_json,
    parse_lowdim_keys,
)
from exrl.runners.dp_demo.demo_replay import populate_replay_from_demo_npz
from exrl.algorithms.dp_demo import (
    concat_replay_batches,
    extract_bootstrap_next_obs_from_venv,
    ingest_smdp_env_step_to_replay,
    PendingSMDPTransition,
    ExecutionLengthAgent,
    SMDPReplayBuffer,
    drain_smdp_nstep_queue_to_replay,
)


def _configure_stdio():
    """
    When tmux/bash redirect stdout to a file, Python may fully buffer prints.
    Force line-buffered / write-through mode so console progress keeps landing.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True, write_through=True)
            except Exception:
                pass


def _append_status_line(path: str, line: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(line.rstrip() + "\n")
        f.flush()
        os.fsync(f.fileno())


def _atomic_json_dump(data, path: str, backup_path: str = None) -> None:
    out_dir = os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".metrics.", suffix=".tmp", dir=out_dir)
    os.close(fd)
    backup_tmp = None
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        if backup_path:
            backup_dir = os.path.dirname(backup_path)
            os.makedirs(backup_dir, exist_ok=True)
            bfd, backup_tmp = tempfile.mkstemp(
                prefix=".metrics_bak.", suffix=".tmp", dir=backup_dir
            )
            os.close(bfd)
            with open(path, "rb") as src, open(backup_tmp, "wb") as dst:
                dst.write(src.read())
                dst.flush()
                os.fsync(dst.fileno())
            os.replace(backup_tmp, backup_path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        if backup_tmp is not None:
            try:
                if os.path.exists(backup_tmp):
                    os.remove(backup_tmp)
            except Exception:
                pass


def _load_json_with_backup(primary_path: str, backup_path: str = None):
    for p in [primary_path, backup_path]:
        if not p:
            continue
        if (not os.path.exists(p)) or os.path.getsize(p) <= 0:
            continue
        try:
            with open(p, "r") as f:
                return json.load(f), p
        except Exception:
            continue
    return None, None


class _RunDirLock:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.lock_path = os.path.join(run_dir, ".train_run.lock")
        self._fh = None

    def acquire(self, mode: str) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        self._fh = open(self.lock_path, "a+")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._fh.seek(0)
            holder = self._fh.read().strip()
            raise RuntimeError(
                f"[LOCK] run dir already in use: {self.run_dir}\n"
                f"[LOCK] holder={holder if holder else 'unknown'}"
            )
        self._fh.seek(0)
        self._fh.truncate()
        info = {
            "pid": os.getpid(),
            "mode": mode,
            "time": datetime.now().isoformat(timespec="seconds"),
        }
        self._fh.write(json.dumps(info))
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
        self._fh = None


def _create_new_run_dir(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    base = f"run_{datetime.now().strftime('%m%d_%H%M%S')}"
    for i in range(0, 100):
        name = base if i == 0 else f"{base}_{i:02d}"
        full = os.path.join(log_dir, name)
        try:
            os.makedirs(full, exist_ok=False)
            return full
        except FileExistsError:
            continue
    raise RuntimeError(f"failed to allocate unique run dir under {log_dir}")


def _parse_hidden_dims(s: str) -> List[int]:
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError(f"invalid --hidden_dims: {s}")
    return out


def _parse_min_k_schedule(
    spec: str, max_chunk_len: int, min_chunk_len: int = 1
) -> List[tuple]:
    """
    Parse min-k schedule string like:
    "6000:6,30000:5,50000:4,70000:3,90000:2,110000:1"
    Each pair means: when policy_steps >= step, set min_k = k.
    """
    text = str(spec).strip()
    if not text:
        return []
    items = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid --post_warmup_min_k_schedule item: {part}")
        s_step, s_k = part.split(":", 1)
        step = int(s_step.strip())
        k = int(s_k.strip())
        if step < 0:
            raise ValueError(f"Schedule step must be >= 0, got {step}")
        if not (int(min_chunk_len) <= k <= int(max_chunk_len)):
            raise ValueError(
                f"Schedule k must be in [{int(min_chunk_len)}, {max_chunk_len}], got {k}"
            )
        items.append((step, k))
    items.sort(key=lambda x: x[0])
    return items


def _parse_float_schedule(
    spec: str, min_value: float = 0.0, max_value: float = 1.0
) -> List[tuple]:
    """
    Parse schedule string like:
    "4000:0.7,16000:0.5,32000:0.3,60000:0.1,100000:0.0"
    Each pair means: when policy_steps >= step, set coeff = value.
    """
    text = str(spec).strip()
    if not text:
        return []
    items = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid float schedule item: {part}")
        s_step, s_val = part.split(":", 1)
        step = int(s_step.strip())
        val = float(s_val.strip())
        if step < 0:
            raise ValueError(f"Schedule step must be >= 0, got {step}")
        if val < float(min_value):
            raise ValueError(f"Schedule coeff must be >= {min_value}, got {val}")
        if max_value is not None and val > float(max_value):
            raise ValueError(f"Schedule coeff must be <= {max_value}, got {val}")
        items.append((step, val))
    items.sort(key=lambda x: x[0])
    return items


def _parse_int_schedule(
    spec: str, min_value: int = 0, max_value: Optional[int] = None
) -> List[tuple]:
    text = str(spec).strip()
    if not text:
        return []
    items = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid int schedule item: {part}")
        s_step, s_val = part.split(":", 1)
        step = int(s_step.strip())
        val = int(s_val.strip())
        if step < 0:
            raise ValueError(f"Schedule step must be >= 0, got {step}")
        if val < int(min_value):
            raise ValueError(f"Schedule value must be >= {min_value}, got {val}")
        if max_value is not None and val > int(max_value):
            raise ValueError(f"Schedule value must be <= {max_value}, got {val}")
        items.append((step, val))
    items.sort(key=lambda x: x[0])
    return items


def _resolve_min_k_floor(counter_value: int, args, schedule_items: List[tuple]) -> int:
    # Priority 1: explicit schedule
    if schedule_items:
        k_floor = 1
        for step, k in schedule_items:
            if counter_value >= int(step):
                k_floor = int(k)
            else:
                break
        return k_floor
    # Fixed minimum length before the schedule begins
    if int(args.post_warmup_min_k_until_policy_steps) > 0 and counter_value < int(
        args.post_warmup_min_k_until_policy_steps
    ):
        return int(args.post_warmup_min_k)
    return 1


def _resolve_float_schedule(
    policy_steps: int, items: List[tuple], default: float = 0.0
) -> float:
    if not items:
        return float(default)
    value = float(default)
    for step, coeff in items:
        if policy_steps >= int(step):
            value = float(coeff)
        else:
            break
    return float(value)


def _resolve_int_schedule(counter_value: int, items: List[tuple], default: int) -> int:
    if not items:
        return int(default)
    value = int(default)
    for step, scheduled in items:
        if counter_value >= int(step):
            value = int(scheduled)
        else:
            break
    return int(value)


def _resolve_dynamic_floor_signal(
    metric_name: str, latest_train_metrics: Dict[str, float], success_buffer: deque
) -> Optional[float]:
    metric_name = str(metric_name)
    if metric_name == "none":
        return None
    if metric_name == "success_rate_recent":
        if len(success_buffer) == 0:
            return None
        return float(np.mean(success_buffer))
    tag_map = {
        "critic_loss": "train/critic_loss",
        "actor_loss": "train/actor_loss",
        "q_argmax_k_mean": "train/q_argmax_k_mean",
    }
    tag = tag_map.get(metric_name)
    if tag is None:
        return None
    value = latest_train_metrics.get(tag)
    if value is None:
        return None
    return float(value)


def _parse_cpu_affinity(spec: str):
    cpus = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError(f"Invalid cpu affinity spec: {spec}")
    return cpus


def _record(history: Dict, key: str, value: float):
    if key not in history["metrics"]:
        history["metrics"][key] = []
    history["metrics"][key].append(float(value))


def _extract_success(info: Dict) -> bool:
    raw = info.get("is_success", None)
    if raw is None:
        episode = info.get("episode", {})
        if isinstance(episode, dict):
            raw = episode.get("is_success", None)
    if raw is None:
        raw = info.get("success", False)
    if isinstance(raw, dict):
        return bool(raw.get("task", False))
    if isinstance(raw, (list, tuple, np.ndarray)):
        if len(raw) == 0:
            return False
        raw = raw[-1]
        if isinstance(raw, dict):
            return bool(raw.get("task", False))
    return bool(raw)


def _extract_bootstrap_next_obs(
    venv, env_idx: int, info: Dict, fallback_next_obs: np.ndarray
) -> np.ndarray:
    return extract_bootstrap_next_obs_from_venv(
        venv=venv,
        env_idx=int(env_idx),
        info=info,
        fallback_next_obs=fallback_next_obs,
    )


def _mean_meter(meter: Dict[str, List[float]]) -> Dict[str, float]:
    out = {}
    for key, vals in meter.items():
        if vals:
            out[key] = float(np.mean(vals))
    return out


def _validate_offline_actorless_args(args, floor_support_mask_mode: str) -> None:
    if int(args.min_chunk_len) != 1:
        raise ValueError(
            "q_learning currently requires --min_chunk_len 1 so k=0 is not exposed"
        )
    if str(args.offline_phase0_source) not in ("online_random", "demo_npz"):
        raise ValueError("--offline_phase0_source must be online_random or demo_npz")
    if str(args.offline_phase0_source) == "demo_npz":
        if not str(args.offline_demo_npz).strip():
            raise ValueError(
                "--offline_demo_npz is required when --offline_phase0_source demo_npz"
            )
        if str(args.offline_demo_window_mode) != "full_only":
            raise ValueError(
                "offline demo phase0 currently supports only --offline_demo_window_mode full_only"
            )
        if int(args.offline_demo_max_episodes) < 0:
            raise ValueError("--offline_demo_max_episodes must be >= 0")
        if int(args.offline_demo_max_transitions) < 0:
            raise ValueError("--offline_demo_max_transitions must be >= 0")
        if int(args.offline_demo_target_batch_size) <= 0:
            raise ValueError("--offline_demo_target_batch_size must be > 0")
        if int(args.action_history_len) != 0:
            raise ValueError(
                "demo_npz phase0 currently requires --action_history_len 0"
            )
    if float(args.demo_prefill_ratio) < 0.0 or float(args.demo_prefill_ratio) > 1.0:
        raise ValueError("--demo_prefill_ratio must be in [0, 1]")
    demo_prefill_schedule_items = _parse_float_schedule(
        str(args.demo_prefill_ratio_schedule), min_value=0.0, max_value=1.0
    )
    if str(args.demo_prefill_ratio_schedule_axis) not in (
        "env",
        "policy",
        "phase2_env",
        "critic_updates",
    ):
        raise ValueError(
            "--demo_prefill_ratio_schedule_axis must be env, policy, phase2_env, or critic_updates"
        )
    demo_prefill_enabled = float(args.demo_prefill_ratio) > 0.0 or bool(
        demo_prefill_schedule_items
    )
    if demo_prefill_enabled:
        if not str(args.demo_prefill_npz).strip():
            raise ValueError(
                "--demo_prefill_npz is required when demo prefill mixing is enabled"
            )
        if int(args.demo_prefill_target_batch_size) <= 0:
            raise ValueError("--demo_prefill_target_batch_size must be > 0")
        if int(args.demo_prefill_replay_size) < 0:
            raise ValueError("--demo_prefill_replay_size must be >= 0")
        if int(args.demo_prefill_max_episodes) < 0:
            raise ValueError("--demo_prefill_max_episodes must be >= 0")
        if int(args.demo_prefill_max_transitions) < 0:
            raise ValueError("--demo_prefill_max_transitions must be >= 0")
        if int(args.action_history_len) != 0:
            raise ValueError("demo prefill currently requires --action_history_len 0")
    if bool(args.use_mc_lower_bound) and int(args.n_step) != 1:
        raise ValueError(
            "q_learning with --use_mc_lower_bound requires --n_step 1; "
            "overlapping n-step replay transitions would make MC return-to-go double-count rewards"
        )
    if bool(args.use_mc_lower_bound) and float(args.cql_alpha) <= 0.0:
        raise ValueError(
            "--use_mc_lower_bound is applied inside CQL, so it requires --cql_alpha > 0"
        )
    if str(args.warmup_action_mode) != "uniform":
        raise ValueError(
            "q_learning phase-0 must use pure random k: set --warmup_action_mode uniform"
        )
    if (
        bool(args.mask_illegal_actions_with_floor)
        or str(floor_support_mask_mode) != "none"
    ):
        raise ValueError(
            "q_learning removes floor/legal-mask regularization; use floor_support_mask_mode=none"
        )
    if str(args.post_warmup_min_k_schedule).strip():
        raise ValueError("q_learning does not allow --post_warmup_min_k_schedule")
    if int(args.post_warmup_min_k_until_policy_steps) > 0:
        raise ValueError(
            "q_learning does not allow --post_warmup_min_k_until_policy_steps"
        )
    if (
        str(args.dynamic_floor_release_metric) != "none"
        or str(args.dynamic_floor_release_target_schedule).strip()
    ):
        raise ValueError("q_learning does not allow dynamic floor release")
    if str(args.post_warmup_actor_prior_schedule).strip():
        raise ValueError("q_learning has no actor; actor prior schedule must be empty")
    if str(args.target_policy_mode) not in ("q_soft", "q_greedy"):
        raise ValueError("q_learning requires --target_policy_mode q_soft or q_greedy")
    if int(args.initial_critic_updates) <= 0:
        raise ValueError("--initial_critic_updates must be > 0")
    if int(args.online_transitions) < 0:
        raise ValueError("--online_transitions must be >= 0")
    if int(args.eval_interval) < 0:
        raise ValueError("--eval_interval must be >= 0")
    if int(args.offline_resume_save_interval_steps) < 0:
        raise ValueError("--offline_resume_save_interval_steps must be >= 0")
    if not (0.0 <= float(args.collection_epsilon) <= 1.0):
        raise ValueError("--collection_epsilon must be in [0, 1]")
    if float(args.collection_ucb_beta) < 0.0:
        raise ValueError("--collection_ucb_beta must be >= 0")
    if not (0.0 < float(args.critic_bootstrap_mask_prob) <= 1.0):
        raise ValueError("--critic_bootstrap_mask_prob must be in (0, 1]")
    if float(args.td_huber_delta) <= 0.0:
        raise ValueError("--td_huber_delta must be > 0")
    if (
        args.target_q_clip_min is not None
        and args.target_q_clip_max is not None
        and float(args.target_q_clip_min) > float(args.target_q_clip_max)
    ):
        raise ValueError("--target_q_clip_min cannot exceed --target_q_clip_max")
    if float(args.critic_q_upper_penalty) < 0.0:
        raise ValueError("--critic_q_upper_penalty must be >= 0")
    if int(args.td_target_next_obs_samples) < 1:
        raise ValueError("--td_target_next_obs_samples must be >= 1")


def _build_task_venv(
    *,
    args,
    policy,
    device,
    lowdim_keys,
    lowdim_dims,
    env_meta,
    n_envs: int,
    seed: int,
):
    venv = ExecutionLengthEnv(
        venv=None,
        policy=policy,
        max_chunk_len=args.max_chunk_len,
        device=device,
        reward_offset=args.reward_offset,
        resample_penalty=args.resample_penalty,
        action_history_len=args.action_history_len,
        dp_infer_steps=args.dp_infer_steps,
        dp_sampler=args.dp_sampler,
        dp_amp=args.dp_amp,
        dp_compile=args.dp_compile,
        ipc_chunk_dtype=args.ipc_chunk_dtype,
        dp_infer_gpus=args.dp_infer_gpus,
        dp_grouping=args.dp_grouping,
        dp_infer_parallel=args.dp_infer_parallel,
        ipc_send_mode=args.ipc_send_mode,
        perf_profile=False,
        vendor_root=args.vendor_root,
        env_name=args.env_name,
        env_meta_json=args.env_meta_json,
        normalization_path=args.dsrl_normalization_npz,
        stats_npz_path=args.stats_npz_override,
        lowdim_keys=lowdim_keys,
        lowdim_dims=lowdim_dims,
        n_envs=int(n_envs),
        max_episode_steps=int(
            env_meta.get("env_kwargs", {}).get("horizon", args.env_horizon_override)
        ),
        gamma=args.gamma,
        seed=int(seed),
        min_chunk_len=int(args.min_chunk_len),
        zero_action_fail_streak=int(args.zero_action_fail_streak),
        zero_action_fail_reward=float(args.zero_action_fail_reward),
        terminate_on_success=bool(args.terminate_on_success),
        timeout_on_horizon=bool(args.timeout_on_horizon),
        td_target_next_obs_samples=int(args.td_target_next_obs_samples),
        critic_obs_unnormalize_dp_chunk=bool(args.critic_obs_unnormalize_dp_chunk),
        critic_obs_action_mode=str(args.critic_obs_action_mode),
        critic_obs_state_mode=str(args.critic_obs_state_mode),
    )
    return VecMonitor(venv)


def _run_critic_policy_eval(
    *,
    venv,
    sac: ExecutionLengthAgent,
    action_shift: int,
    episodes: int,
    max_decisions: int,
    policy_mode: str,
    critic_reduce: str,
):
    eval_obs = np.asarray(venv.reset(), dtype=np.float32)
    successes = []
    returns = np.zeros((venv.num_envs,), dtype=np.float32)
    lengths = np.zeros((venv.num_envs,), dtype=np.int64)
    exec_lens = []
    eval_env_steps = 0
    decisions = 0
    while len(successes) < int(episodes) and decisions < int(max_decisions):
        actions = sac.select_critic_action(
            eval_obs,
            deterministic=(str(policy_mode) == "q_greedy"),
            min_action_id=0,
            mode=str(policy_mode),
            reduce=str(critic_reduce),
        )
        next_obs, rewards, dones, infos = venv.step(actions + int(action_shift))
        next_obs = np.asarray(next_obs, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        dones = np.asarray(dones, dtype=bool).reshape(-1)
        for i in range(venv.num_envs):
            info = infos[i] if i < len(infos) else {}
            k = float(info.get("exec_len", 0.0))
            exec_lens.append(k)
            eval_env_steps += int(k)
            returns[i] += float(rewards[i])
            lengths[i] += 1
            if bool(dones[i]):
                successes.append(1 if _extract_success(info) else 0)
                returns[i] = 0.0
                lengths[i] = 0
                if len(successes) >= int(episodes):
                    break
        eval_obs = next_obs
        decisions += int(venv.num_envs)
    if len(successes) == 0:
        success_rate = 0.0
        success_rate_recent20 = 0.0
    else:
        success_rate = float(np.mean(successes[: int(episodes)]))
        success_rate_recent20 = float(np.mean(successes[: int(episodes)][-20:]))
    return {
        "offline_eval/critic_success_rate": success_rate,
        "offline_eval/critic_success_rate_recent20": success_rate_recent20,
        "offline_eval/critic_avg_exec_len": (
            float(np.mean(exec_lens)) if exec_lens else 0.0
        ),
        "offline_eval/critic_episodes": float(min(len(successes), int(episodes))),
        "offline_eval/critic_env_steps": float(eval_env_steps),
        "offline_eval/critic_decisions": float(decisions),
        "offline_eval/critic_reduce_id": float(
            {"min": 0, "mean": 1}[str(critic_reduce)]
        ),
    }


from exrl.runners.dp_demo.options import build_parser


def _validate_task_alignment(args):
    env_name = str(args.env_name).strip().lower()
    if not env_name:
        raise ValueError("--env_name must be non-empty")

    meta_name = os.path.basename(str(args.env_meta_json)).lower()
    if env_name not in meta_name:
        raise ValueError(
            f"env/task mismatch: env_name={args.env_name} but env_meta_json={args.env_meta_json}"
        )

    norm_path = os.path.normpath(str(args.dsrl_normalization_npz)).lower()
    if f"/{env_name}/" not in norm_path and f"\\{env_name}\\" not in norm_path:
        raise ValueError(
            f"env/task mismatch: env_name={args.env_name} but dsrl_normalization_npz={args.dsrl_normalization_npz}"
        )

    stats_name = os.path.basename(str(args.stats_npz_override)).lower()
    if env_name not in stats_name:
        raise ValueError(
            f"env/task mismatch: env_name={args.env_name} but stats_npz_override={args.stats_npz_override}"
        )

    agent_path = str(args.agent).lower()
    if env_name not in agent_path:
        print(
            f"[ExRL][WARN] agent path does not contain env_name={args.env_name}: {args.agent}",
            flush=True,
        )


def _set_runtime_env(args):
    if bool(args.disable_checkpoints):
        os.environ["EXRL_DISABLE_CHECKPOINTS"] = "1"
    else:
        os.environ.pop("EXRL_DISABLE_CHECKPOINTS", None)
    os.environ["OPT_MIN_CHUNK_LEN"] = str(int(args.min_chunk_len))
    os.environ["OPT_TASK_STATS_NPZ"] = str(args.stats_npz_override)
    os.environ.pop("OPT_CAN_STATS_NPZ", None)
    # Base wrapper still loads OPT_TOOLHANG_STATS_NPZ first; point it to the task stats.
    os.environ["OPT_TOOLHANG_STATS_NPZ"] = str(args.stats_npz_override)

    if args.force_thread_limits:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    if args.decision_runtime == "asyncqueue":
        os.environ["OPT_ASYNCQUEUE_INFER_MICROBATCH_MAX"] = str(
            args.infer_microbatch_max
        )
        os.environ["OPT_ASYNCQUEUE_INFER_MICROBATCH_TIMEOUT_MS"] = str(
            args.infer_microbatch_timeout_ms
        )
        os.environ["OPT_ASYNCQUEUE_INFER_QUEUE_SIZE"] = str(args.infer_queue_size)
        os.environ["OPT_ASYNCQUEUE_INFER_RESULT_QUEUE_SIZE"] = str(
            args.infer_result_queue_size
        )
        os.environ["OPT_ASYNCQUEUE_ENV_POLL_TIMEOUT_MS"] = str(args.env_poll_timeout_ms)
        os.environ["OPT_ASYNCQUEUE_READY_QUEUE_SIZE"] = str(args.ready_queue_size)
        os.environ["OPT_ASYNCQUEUE_MAX_INFLIGHT_PER_ENV"] = str(
            args.max_inflight_per_env
        )
        os.environ["OPT_ASYNCQUEUE_ROLLOUT_MODE"] = str(args.async_rollout_mode)


def _save_ckpt(path: str, payload: Dict):
    if os.environ.get("EXRL_DISABLE_CHECKPOINTS") == "1":
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".sac.", suffix=".tmp", dir=os.path.dirname(path))
    os.close(fd)
    try:
        torch.save(payload, tmp, pickle_protocol=4)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _ckpt_payload_includes_replay(args) -> bool:
    return str(args.pipeline) == "q_learning" and not bool(
        getattr(args, "no_ckpt_replay_state", False)
    )


def _load_ckpt(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _next_multiple_after(current: int, freq: int) -> int:
    freq = max(int(freq), 1)
    return ((int(current) // freq) + 1) * freq


def _pack_rng_state() -> Dict:
    out = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        out["torch_cuda"] = torch.cuda.get_rng_state_all()
    return out


def _restore_rng_state(state: Optional[Dict]) -> None:
    if not state:
        return
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"].cpu())
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            cuda_states = state["torch_cuda"]
            if len(cuda_states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_states)
    except Exception as exc:
        print(
            f"[ExRL][RESUME][WARN] failed to restore RNG state: {type(exc).__name__}: {exc}",
            flush=True,
        )


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.num_workers_cpu_affinity:
        cpu_set = _parse_cpu_affinity(args.num_workers_cpu_affinity)
        os.sched_setaffinity(0, cpu_set)
        print(f"[ExRL] set CPU affinity: total={len(cpu_set)}")

    _set_runtime_env(args)

    if not os.path.exists(args.stats_npz_override):
        raise FileNotFoundError(f"stats npz not found: {args.stats_npz_override}")

    if args.resume:
        run_dir = args.resume
        if not os.path.exists(run_dir):
            raise FileNotFoundError(f"resume dir not found: {run_dir}")
    else:
        run_dir = _create_new_run_dir(args.log_dir)
    os.makedirs(run_dir, exist_ok=True)
    if hasattr(args, "public_config"):
        from exrl.config import save_config

        save_config(args.public_config, os.path.join(run_dir, "experiment.yaml"))
    print(f"Run directory: {run_dir}", flush=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    video_dir = os.path.join(run_dir, "videos")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    lock = _RunDirLock(run_dir)
    lock.acquire(mode="resume" if args.resume else "new")
    atexit.register(lock.release)

    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)

    metadata = {
        "created_utc": datetime.utcnow().isoformat(),
        "python": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "agent_path": args.agent,
        "stats_path": args.stats_npz_override,
        "nozero": int(args.min_chunk_len) > 0,
        "args": vars(args),
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print(f"[ExRL] device={device}")
    lowdim_keys = parse_lowdim_keys(args.lowdim_keys)
    lowdim_dims = infer_lowdim_dims_from_stats(
        args.stats_npz_override, lowdim_keys=lowdim_keys
    )
    action_dim_from_stats = infer_action_dim_from_stats(args.stats_npz_override)
    policy, ckpt_meta = load_dsrl_policy_wrapper(
        ckpt_path=args.agent,
        device=device,
        stats_npz=args.stats_npz_override,
        lowdim_keys=lowdim_keys,
        lowdim_dims=lowdim_dims,
        action_dim=action_dim_from_stats,
        cond_steps=args.cond_steps,
        weight_source=args.weight_source,
    )
    env_meta = load_env_meta_json(args.env_meta_json)
    env_meta.setdefault("env_kwargs", {})
    env_meta["env_kwargs"]["horizon"] = int(args.env_horizon_override)
    metadata["dsrl_ckpt_meta"] = ckpt_meta
    metadata["env_meta_json"] = args.env_meta_json
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(
        f"[ExRL][ExRL] env={args.env_name} horizon={env_meta['env_kwargs']['horizon']} "
        f"weight_source={args.weight_source}"
    )

    if args.decision_runtime != "sync":
        print("[ExRL][WARN] This runner uses synchronous chunk execution.")

    eval_venv = None
    venv = _build_task_venv(
        args=args,
        policy=policy,
        device=device,
        lowdim_keys=lowdim_keys,
        lowdim_dims=lowdim_dims,
        env_meta=env_meta,
        n_envs=int(args.n_envs),
        seed=int(args.seed),
    )

    obs = venv.reset()
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim != 2:
        raise RuntimeError(f"expect obs shape [n_envs, obs_dim], got {obs.shape}")
    obs_dim = int(obs.shape[1])
    action_dim_raw = int(venv.action_space.n)
    expected_action_dim = int(args.max_chunk_len) - int(args.min_chunk_len) + 1
    if action_dim_raw == expected_action_dim:
        action_shift = 0
        action_dim = expected_action_dim
    elif action_dim_raw == int(args.max_chunk_len) + 1:
        # Fallback: underlying wrapper still exposes 0..K. Shift policy action by min_chunk_len.
        action_shift = int(args.min_chunk_len)
        action_dim = expected_action_dim
    else:
        raise RuntimeError(
            f"Unexpected env action dim: raw={action_dim_raw}, expected {expected_action_dim} or {args.max_chunk_len + 1}"
        )
    print(
        f"[ExRL] obs_dim={obs_dim} action_dim_effective={action_dim} "
        f"action_dim_raw={action_dim_raw} action_shift={action_shift}"
    )

    def ensure_eval_venv(*, allow_collector_fallback: bool = False):
        nonlocal eval_venv
        if eval_venv is not None:
            return eval_venv
        eval_requested = (
            int(args.offline_eval_every_updates) > 0
            or int(args.eval_interval) > 0
            or bool(args.eval_at_start)
            or bool(args.offline_eval_after_training)
        )
        if str(args.pipeline) != "q_learning" or not eval_requested:
            raise RuntimeError("offline eval requested while offline eval is disabled")
        eval_n_envs = int(args.eval_n_envs)
        if eval_n_envs <= 0:
            eval_n_envs = max(1, min(int(args.n_envs), int(args.eval_episodes)))
        candidate = None
        try:
            candidate = _build_task_venv(
                args=args,
                policy=policy,
                device=device,
                lowdim_keys=lowdim_keys,
                lowdim_dims=lowdim_dims,
                env_meta=env_meta,
                n_envs=eval_n_envs,
                seed=int(args.seed) + int(args.offline_eval_seed_offset),
            )
            eval_action_dim_raw = int(candidate.action_space.n)
            if eval_action_dim_raw != action_dim_raw:
                raise RuntimeError(
                    f"Eval env action dim mismatch: eval_raw={eval_action_dim_raw}, train_raw={action_dim_raw}"
                )
            eval_venv = candidate
            print(
                f"[ExRL][OFFLINE] built separate eval env n_envs={eval_n_envs}",
                flush=True,
            )
            return eval_venv
        except Exception as exc:
            if candidate is not None:
                try:
                    candidate.close()
                except Exception:
                    pass
            if allow_collector_fallback and venv is not None:
                line = (
                    "[ExRL][OFFLINE][WARN] separate eval env creation failed; "
                    f"falling back to phase0 collector env: {type(exc).__name__}: {exc}"
                )
                print(line, flush=True)
                _append_status_line(status_log_path, line)
                eval_venv = venv
                return eval_venv
            raise

    if not (0 <= int(args.min_chunk_len) <= int(args.max_chunk_len)):
        raise ValueError(f"--min_chunk_len must be in [0, {args.max_chunk_len}]")
    if int(args.zero_action_fail_streak) < 0:
        raise ValueError("--zero_action_fail_streak must be >= 0")
    if not (1 <= int(args.warmup_fixed_k) <= int(args.max_chunk_len)):
        raise ValueError(f"--warmup_fixed_k must be in [1, {args.max_chunk_len}]")
    if not (0.0 <= float(args.warmup_mixed_uniform_prob) <= 1.0):
        raise ValueError("--warmup_mixed_uniform_prob must be in [0, 1]")
    if not (0.0 <= float(args.epsilon_greedy_prob) <= 1.0):
        raise ValueError("--epsilon_greedy_prob must be in [0, 1]")
    if int(args.n_step) <= 0:
        raise ValueError("--n_step must be >= 1")
    if not (1 <= int(args.post_warmup_min_k) <= int(args.max_chunk_len)):
        raise ValueError(f"--post_warmup_min_k must be in [1, {args.max_chunk_len}]")
    if int(args.learning_starts_policy_steps) > 0 and int(
        args.learning_starts_policy_steps
    ) < int(args.warmup_policy_steps):
        raise ValueError(
            "--learning_starts_policy_steps must be >= warmup_policy_steps when set"
        )
    if int(args.train_freq_transitions) < 0:
        raise ValueError("--train_freq_transitions must be >= 0")
    floor_support_mask_mode = str(args.floor_support_mask_mode)
    if bool(args.mask_illegal_actions_with_floor) and floor_support_mask_mode == "none":
        floor_support_mask_mode = "all"
    min_k_schedule_axis = str(args.post_warmup_min_k_schedule_axis)
    min_k_schedule_items = _parse_min_k_schedule(
        args.post_warmup_min_k_schedule, args.max_chunk_len, args.min_chunk_len
    )
    dynamic_release_target_items = _parse_min_k_schedule(
        args.dynamic_floor_release_target_schedule,
        args.max_chunk_len,
        args.min_chunk_len,
    )
    behavior_mix_schedule_items = _parse_float_schedule(
        args.post_warmup_behavior_mix_schedule, min_value=0.0, max_value=1.0
    )
    actor_prior_schedule_items = _parse_float_schedule(
        args.post_warmup_actor_prior_schedule, min_value=0.0, max_value=None
    )
    phase2_ucb_beta_schedule_items = _parse_float_schedule(
        args.collection_ucb_schedule, min_value=0.0, max_value=None
    )
    utd_schedule_axis = str(args.utd_schedule_axis)
    utd_schedule_items = _parse_int_schedule(
        args.utd_schedule, min_value=1, max_value=None
    )

    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    target_entropy = float(args.target_entropy_scale) * float(np.log(action_dim))
    sac = ExecutionLengthAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        init_alpha=args.init_alpha,
        fixed_alpha=args.fixed_alpha,
        max_alpha=args.max_alpha,
        target_entropy=target_entropy,
        activation=args.network_activation,
        actor_use_layernorm=args.actor_use_layernorm,
        critic_use_layernorm=args.critic_use_layernorm,
        actor_enabled=(str(args.pipeline) != "q_learning"),
        target_policy_mode=(
            str(args.target_policy_mode)
            if str(args.pipeline) == "q_learning"
            else "actor"
        ),
        critic_backup_temp=float(args.critic_backup_temp),
        cql_alpha=float(args.cql_alpha),
        cql_temp=float(args.cql_temp),
        use_mc_lower_bound=bool(args.use_mc_lower_bound),
        mc_lower_bound_scope=str(args.mc_lower_bound_scope),
        mc_lower_bound_topk=int(args.mc_lower_bound_topk),
        critic_ensemble_size=int(args.critic_ensemble_size),
        target_critic_subsample_size=int(args.target_critic_subsample_size),
        critic_bootstrap_mask_prob=float(args.critic_bootstrap_mask_prob),
        distributional_critic=bool(args.distributional_critic),
        distributional_q_min=float(args.distributional_q_min),
        distributional_q_max=float(args.distributional_q_max),
        distributional_num_bins=int(args.distributional_num_bins),
        td_loss_type=str(args.td_loss_type),
        td_huber_delta=float(args.td_huber_delta),
        target_q_clip_min=args.target_q_clip_min,
        target_q_clip_max=args.target_q_clip_max,
        critic_q_upper_bound=args.critic_q_upper_bound,
        critic_q_upper_penalty=float(args.critic_q_upper_penalty),
        device=device,
    )
    replay = SMDPReplayBuffer(
        obs_dim=obs_dim,
        capacity=args.replay_size,
        device=device,
        target_next_obs_samples=int(args.td_target_next_obs_samples),
    )
    demo_replay = None
    nstep_queues = [deque() for _ in range(args.n_envs)]

    policy_steps = 0
    env_steps = 0
    step_calls = 0
    pending_transitions_for_update = 0
    num_emitted_transitions_total = 0
    num_updates_total = 0
    num_done_for_backup_true = 0
    num_timeout_truncations = 0
    duration_sum_total = 0.0
    best_success = -1.0
    history = {
        "timesteps": [],
        "timesteps_env": [],
        "timesteps_policy": [],
        "metrics": {},
    }
    metrics_path = os.path.join(run_dir, "training_metrics.json")
    metrics_backup = os.path.join(run_dir, "training_metrics.backup.json")
    status_log_path = os.path.join(run_dir, "status.log")
    offline_phase_state = {}

    if args.resume:
        latest_path = os.path.join(ckpt_dir, args.resume_ckpt_name)
        if not os.path.exists(latest_path):
            raise FileNotFoundError(f"resume checkpoint not found: {latest_path}")
        state = _load_ckpt(latest_path, map_location=device)
        sac.load_state_dict(state["sac_state"])
        if state.get("replay_state") is not None:
            replay.load_state_dict(state["replay_state"])
        elif str(args.pipeline) == "q_learning":
            resume_msg = "[ExRL][RESUME][WARN] checkpoint has no replay_state; offline resume may not be valid"
            print(resume_msg, flush=True)
            _append_status_line(status_log_path, resume_msg)
        policy_steps = int(state.get("policy_steps", 0))
        env_steps = int(state.get("env_steps", 0))
        step_calls = int(state.get("step_calls", 0))
        pending_transitions_for_update = int(
            state.get("pending_transitions_for_update", 0)
        )
        num_updates_total = int(
            state.get("num_updates_total", state.get("num_updates", 0))
        )
        num_emitted_transitions_total = int(
            state.get("num_emitted_transitions_total", 0)
        )
        num_done_for_backup_true = int(state.get("num_done_for_backup_true", 0))
        num_timeout_truncations = int(state.get("num_timeout_truncations", 0))
        duration_sum_total = float(state.get("duration_sum_total", 0.0))
        best_success = float(state.get("best_success", -1.0))
        dynamic_floor_release_latched = bool(
            state.get("dynamic_floor_release_latched", False)
        )
        dynamic_floor_release_policy_step = state.get(
            "dynamic_floor_release_policy_step"
        )
        dynamic_floor_release_env_step = state.get("dynamic_floor_release_env_step")
        offline_phase_state = dict(state.get("offline_phase_state") or {})
        if "phase_id" in state and "phase_id" not in offline_phase_state:
            offline_phase_state["phase_id"] = state.get("phase_id")
        _restore_rng_state(state.get("rng_state"))
        loaded_history, loaded_from = _load_json_with_backup(
            metrics_path, metrics_backup
        )
        if loaded_history is not None:
            history = loaded_history
            if "timesteps" not in history:
                history["timesteps"] = []
            if "timesteps_env" not in history:
                history["timesteps_env"] = []
            if "timesteps_policy" not in history:
                history["timesteps_policy"] = []
            if "metrics" not in history:
                history["metrics"] = {}
        resume_msg = f"[ExRL][RESUME] loaded metrics from {loaded_from}"
        print(resume_msg, flush=True)
        _append_status_line(status_log_path, resume_msg)
        resume_msg = (
            f"[ExRL][RESUME] model/replay resumed; policy_steps={policy_steps} "
            f"env_steps={env_steps} replay={len(replay)} "
            f"phase_id={offline_phase_state.get('phase_id', 'unknown')}"
        )
        print(resume_msg, flush=True)
        _append_status_line(status_log_path, resume_msg)
    else:
        dynamic_floor_release_latched = False
        dynamic_floor_release_policy_step = None
        dynamic_floor_release_env_step = None

    writer = SummaryWriter(log_dir=os.path.join(run_dir, "scalars_1"))
    success_buffer = deque(maxlen=args.window_size)
    success_buffer_20 = deque(maxlen=20)
    episode_return_buffer = deque(maxlen=args.window_size)
    episode_len_buffer = deque(maxlen=args.window_size)
    exec_buffer = deque(maxlen=2000)
    raw_exec_buffer = deque(maxlen=2000)
    floor_apply_buffer = deque(maxlen=2000)
    behavior_mix_buffer = deque(maxlen=2000)
    actor_prior_buffer = deque(maxlen=2000)
    epsilon_apply_buffer = deque(maxlen=2000)
    train_meter = defaultdict(list)
    latest_train_metrics: Dict[str, float] = {}
    episode_returns = np.zeros((args.n_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.n_envs,), dtype=np.int64)
    start_time = time.time()
    log_idx = 0

    def dump_metrics():
        _atomic_json_dump(history, metrics_path, metrics_backup)

    def pack_ckpt() -> Dict:
        return {
            "sac_state": sac.state_dict(),
            "replay_state": (
                replay.state_dict() if _ckpt_payload_includes_replay(args) else None
            ),
            "ckpt_includes_replay_state": bool(_ckpt_payload_includes_replay(args)),
            "rng_state": _pack_rng_state(),
            "policy_steps": int(policy_steps),
            "env_steps": int(env_steps),
            "step_calls": int(step_calls),
            "pending_transitions_for_update": int(pending_transitions_for_update),
            "num_updates_total": int(num_updates_total),
            "num_emitted_transitions_total": int(num_emitted_transitions_total),
            "num_done_for_backup_true": int(num_done_for_backup_true),
            "num_timeout_truncations": int(num_timeout_truncations),
            "duration_sum_total": float(duration_sum_total),
            "best_success": float(best_success),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dims": hidden_dims,
            "target_entropy": target_entropy,
            "network_activation": args.network_activation,
            "actor_use_layernorm": bool(args.actor_use_layernorm),
            "critic_use_layernorm": bool(args.critic_use_layernorm),
            "max_chunk_len": int(args.max_chunk_len),
            "min_chunk_len": int(args.min_chunk_len),
            "pipeline": str(args.pipeline),
            "target_policy_mode": str(args.target_policy_mode),
            "cql_alpha": float(args.cql_alpha),
            "use_mc_lower_bound": bool(args.use_mc_lower_bound),
            "mc_lower_bound_scope": str(args.mc_lower_bound_scope),
            "mc_lower_bound_topk": int(args.mc_lower_bound_topk),
            "critic_bootstrap_mask_prob": float(args.critic_bootstrap_mask_prob),
            "td_loss_type": str(args.td_loss_type),
            "td_huber_delta": float(args.td_huber_delta),
            "target_q_clip_min": args.target_q_clip_min,
            "target_q_clip_max": args.target_q_clip_max,
            "critic_q_upper_bound": args.critic_q_upper_bound,
            "critic_q_upper_penalty": float(args.critic_q_upper_penalty),
            "td_target_next_obs_samples": int(args.td_target_next_obs_samples),
            "critic_obs_unnormalize_dp_chunk": bool(
                args.critic_obs_unnormalize_dp_chunk
            ),
            "critic_obs_action_mode": str(args.critic_obs_action_mode),
            "critic_obs_state_mode": str(args.critic_obs_state_mode),
            "demo_prefill_ratio": float(args.demo_prefill_ratio),
            "demo_prefill_ratio_schedule": str(args.demo_prefill_ratio_schedule),
            "demo_prefill_ratio_schedule_axis": str(
                args.demo_prefill_ratio_schedule_axis
            ),
            "demo_prefill_npz": str(args.demo_prefill_npz),
            "demo_prefill_replay_capacity": int(
                int(args.demo_prefill_replay_size)
                if int(args.demo_prefill_replay_size) > 0
                else int(args.replay_size)
            ),
            "demo_prefill_replay_size": int(
                0 if demo_replay is None else len(demo_replay)
            ),
            "demo_prefill_replay_state_saved": False,
            "dynamic_floor_release_latched": bool(dynamic_floor_release_latched),
            "dynamic_floor_release_policy_step": dynamic_floor_release_policy_step,
            "dynamic_floor_release_env_step": dynamic_floor_release_env_step,
            "phase_id": offline_phase_state.get("phase_id"),
            "offline_phase_state": dict(offline_phase_state),
            "run_dir": run_dir,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    next_periodic_ckpt = _next_multiple_after(policy_steps, int(args.save_freq))
    next_latest_ckpt = _next_multiple_after(
        policy_steps, int(args.latest_model_save_freq)
    )
    next_log = policy_steps + max(int(args.log_every_policy_steps), 1)
    if int(args.learning_starts_policy_steps) > 0:
        learning_starts = int(args.learning_starts_policy_steps)
    else:
        learning_starts = int(args.warmup_policy_steps)
    if int(args.actor_learning_starts_policy_steps) > 0:
        actor_learning_starts = int(args.actor_learning_starts_policy_steps)
    else:
        actor_learning_starts = int(learning_starts)
    if int(args.alpha_learning_starts_policy_steps) > 0:
        alpha_learning_starts = int(args.alpha_learning_starts_policy_steps)
    else:
        alpha_learning_starts = int(actor_learning_starts)
    print("[ExRL] start training loop", flush=True)
    _append_status_line(status_log_path, "[ExRL] start training loop")
    try:
        if str(args.pipeline) == "q_learning":
            _validate_offline_actorless_args(args, floor_support_mask_mode)
            phase0_target = (
                int(args.warmup_transitions)
                if int(args.warmup_transitions) > 0
                else int(args.warmup_policy_steps)
            )
            try:
                resume_phase_id = int(float(offline_phase_state.get("phase_id", 0)))
            except Exception:
                resume_phase_id = 0
            offline_eval_env_steps_total = int(
                offline_phase_state.get("offline_eval_env_steps_total", 0)
            )
            phase0_source = str(args.offline_phase0_source)
            demo_prefill_ratio_schedule_items = _parse_float_schedule(
                str(args.demo_prefill_ratio_schedule), min_value=0.0, max_value=1.0
            )
            offline_msg = (
                f"[ExRL][OFFLINE] phase0_source={phase0_source} target_policy_steps={phase0_target} "
                f"phase1_critic_updates={int(args.initial_critic_updates)} "
                f"phase2_policy_steps={int(args.online_transitions)} "
                f"target_policy_mode={args.target_policy_mode} cql_alpha={float(args.cql_alpha)} "
                f"demo_prefill_ratio={float(args.demo_prefill_ratio)} "
                f"demo_prefill_ratio_schedule={str(args.demo_prefill_ratio_schedule) or 'none'} "
                f"demo_prefill_ratio_schedule_axis={str(args.demo_prefill_ratio_schedule_axis)}"
            )
            print(offline_msg, flush=True)
            _append_status_line(status_log_path, offline_msg)

            def demo_prefill_ratio_counter() -> int:
                axis = str(args.demo_prefill_ratio_schedule_axis)
                if axis == "policy":
                    return int(policy_steps)
                if axis == "critic_updates":
                    return int(num_updates_total)
                if axis == "phase2_env":
                    phase2_start = offline_phase_state.get("phase2_start_env_steps")
                    if phase2_start is None:
                        return 0
                    return max(0, int(env_steps) - int(phase2_start))
                return int(env_steps)

            def current_demo_prefill_ratio() -> float:
                return float(
                    _resolve_float_schedule(
                        demo_prefill_ratio_counter(),
                        demo_prefill_ratio_schedule_items,
                        default=float(args.demo_prefill_ratio),
                    )
                )

            def current_demo_prefill_batch_sizes() -> tuple:
                ratio = current_demo_prefill_ratio()
                demo_batch = int(round(int(args.batch_size) * float(ratio)))
                demo_batch = max(0, min(int(args.batch_size), int(demo_batch)))
                if ratio > 0.0 and demo_batch <= 0:
                    demo_batch = 1
                if ratio < 1.0 and demo_batch >= int(args.batch_size):
                    demo_batch = int(args.batch_size) - 1
                online_batch = int(args.batch_size) - int(demo_batch)
                return float(ratio), int(demo_batch), int(online_batch)

            def demo_prefill_metrics() -> Dict[str, float]:
                ratio, demo_batch, online_batch = current_demo_prefill_batch_sizes()
                return {
                    "demo_prefill/ratio": float(ratio),
                    "demo_prefill/ratio_schedule_counter": float(
                        demo_prefill_ratio_counter()
                    ),
                    "demo_prefill/batch_size": float(demo_batch),
                    "demo_prefill/online_batch_size": float(online_batch),
                    "demo_prefill/capacity": float(
                        int(args.demo_prefill_replay_size)
                        if int(args.demo_prefill_replay_size) > 0
                        else int(args.replay_size)
                    ),
                    "demo_prefill/size": float(
                        0 if demo_replay is None else len(demo_replay)
                    ),
                    "replay/online_size": float(len(replay)),
                    "replay/demo_size": float(
                        0 if demo_replay is None else len(demo_replay)
                    ),
                }

            def has_offline_update_batch() -> bool:
                _, demo_batch, online_batch = current_demo_prefill_batch_sizes()
                if int(online_batch) > 0 and len(replay) < int(online_batch):
                    return False
                if int(demo_batch) > 0:
                    if demo_replay is None or len(demo_replay) < int(demo_batch):
                        return False
                return True

            def sample_offline_update_batch():
                _, demo_batch, online_batch = current_demo_prefill_batch_sizes()
                if int(demo_batch) <= 0:
                    return replay.sample(int(args.batch_size)), demo_prefill_metrics()
                if demo_replay is None:
                    raise RuntimeError(
                        "demo prefill batch requested but demo_replay is not built"
                    )
                if int(online_batch) <= 0:
                    return (
                        demo_replay.sample(int(args.batch_size)),
                        demo_prefill_metrics(),
                    )
                online_sample = replay.sample(int(online_batch))
                demo_sample = demo_replay.sample(int(demo_batch))
                mixed_batch, _ = concat_replay_batches(
                    online_sample, demo_sample, shuffle=True
                )
                return mixed_batch, demo_prefill_metrics()

            def record_offline_event(
                metrics: Dict[str, float], *, env_axis: int, critic_updates: int
            ):
                history["timesteps"].append(int(env_axis))
                history["timesteps_env"].append(int(env_axis))
                history["timesteps_policy"].append(int(policy_steps))
                if "timesteps_critic_updates" not in history:
                    history["timesteps_critic_updates"] = []
                history["timesteps_critic_updates"].append(int(critic_updates))
                for mk, mv in metrics.items():
                    _record(history, mk, float(mv))

            def record_env_step_eval(
                *, eval_axis: int, phase_id: float, critic_updates: int
            ):
                eval_metrics = _run_critic_policy_eval(
                    venv=ensure_eval_venv(allow_collector_fallback=False),
                    sac=sac,
                    action_shift=action_shift,
                    episodes=int(args.eval_episodes),
                    max_decisions=int(args.offline_eval_max_decisions),
                    policy_mode=str(args.offline_eval_policy_mode),
                    critic_reduce=str(args.offline_eval_critic_reduce),
                )
                metrics = {
                    "env_step_eval/success_rate": float(
                        eval_metrics["offline_eval/critic_success_rate"]
                    ),
                    "env_step_eval/success_rate_recent20": float(
                        eval_metrics["offline_eval/critic_success_rate_recent20"]
                    ),
                    "env_step_eval/avg_exec_len": float(
                        eval_metrics["offline_eval/critic_avg_exec_len"]
                    ),
                    "env_step_eval/episodes": float(
                        eval_metrics["offline_eval/critic_episodes"]
                    ),
                    "env_step_eval/eval_env_steps": float(
                        eval_metrics["offline_eval/critic_env_steps"]
                    ),
                    "env_step_eval/decisions": float(
                        eval_metrics["offline_eval/critic_decisions"]
                    ),
                    "env_step_eval/phase": float(phase_id),
                    "env_step_eval/train_env_steps_actual": float(env_steps),
                    "env_step_eval/train_policy_steps": float(policy_steps),
                    "env_step_eval/critic_updates": float(critic_updates),
                }
                record_offline_event(
                    metrics, env_axis=int(eval_axis), critic_updates=int(critic_updates)
                )
                for mk, mv in metrics.items():
                    writer.add_scalar(mk, float(mv), int(eval_axis))
                line = (
                    f"[ExRL][OFFLINE][ENV-EVAL] eval_axis={int(eval_axis)} "
                    f"actual_env_steps={int(env_steps)} phase={float(phase_id):.0f} "
                    f"critic_updates={int(critic_updates)} "
                    f"sr={metrics['env_step_eval/success_rate']:.4f} "
                    f"sr20={metrics['env_step_eval/success_rate_recent20']:.4f} "
                    f"avg_k={metrics['env_step_eval/avg_exec_len']:.2f}"
                )
                print(line, flush=True)
                _append_status_line(status_log_path, line)
                dump_metrics()
                writer.flush()

            env_eval_interval = int(args.eval_interval)
            offline_resume_env_ckpt_freq = int(args.offline_resume_save_interval_steps)
            if offline_resume_env_ckpt_freq <= 0:
                offline_resume_env_ckpt_freq = (
                    int(env_eval_interval) if int(env_eval_interval) > 0 else 0
                )
            next_env_eval_axis = (
                int(
                    offline_phase_state.get(
                        "next_env_eval_axis",
                        _next_multiple_after(env_steps, env_eval_interval),
                    )
                )
                if env_eval_interval > 0
                else None
            )
            next_offline_latest_env = (
                int(
                    offline_phase_state.get(
                        "next_offline_latest_env",
                        _next_multiple_after(env_steps, offline_resume_env_ckpt_freq),
                    )
                )
                if offline_resume_env_ckpt_freq > 0
                else None
            )

            def refresh_common_phase_state(phase_id: int, **extra):
                offline_phase_state.update(
                    {
                        "phase_id": int(phase_id),
                        "policy_steps": int(policy_steps),
                        "env_steps": int(env_steps),
                        "step_calls": int(step_calls),
                        "replay_size": int(len(replay)),
                        "offline_eval_env_steps_total": int(
                            offline_eval_env_steps_total
                        ),
                        "next_env_eval_axis": (
                            None
                            if next_env_eval_axis is None
                            else int(next_env_eval_axis)
                        ),
                        "offline_resume_env_ckpt_freq": int(
                            offline_resume_env_ckpt_freq
                        ),
                        "next_offline_latest_env": (
                            None
                            if next_offline_latest_env is None
                            else int(next_offline_latest_env)
                        ),
                    }
                )
                offline_phase_state.update(extra)

            if args.resume:
                resume_msg = (
                    f"[ExRL][OFFLINE][RESUME] entering phase_id={resume_phase_id} "
                    f"policy_steps={policy_steps} env_steps={env_steps} replay={len(replay)}"
                )
                print(resume_msg, flush=True)
                _append_status_line(status_log_path, resume_msg)

            if (not args.resume) and bool(args.eval_at_start):
                record_env_step_eval(eval_axis=0, phase_id=0.0, critic_updates=0)

            if float(args.demo_prefill_ratio) > 0.0 or bool(
                demo_prefill_ratio_schedule_items
            ):
                demo_prefill_replay_capacity = (
                    int(args.demo_prefill_replay_size)
                    if int(args.demo_prefill_replay_size) > 0
                    else int(args.replay_size)
                )
                demo_replay = SMDPReplayBuffer(
                    obs_dim=obs_dim,
                    capacity=int(demo_prefill_replay_capacity),
                    device=device,
                    target_next_obs_samples=int(args.td_target_next_obs_samples),
                )
                demo_prefill_summary = populate_replay_from_demo_npz(
                    replay=demo_replay,
                    policy=policy,
                    demo_npz=str(args.demo_prefill_npz),
                    stats_npz=str(args.stats_npz_override),
                    normalization_npz=str(args.dsrl_normalization_npz),
                    lowdim_keys=lowdim_keys,
                    lowdim_dims=lowdim_dims,
                    action_dim=action_dim_from_stats,
                    max_chunk_len=int(args.max_chunk_len),
                    min_chunk_len=int(args.min_chunk_len),
                    gamma=float(args.gamma),
                    reward_offset=float(args.reward_offset),
                    reward_duration_mode=str(args.reward_duration_mode),
                    target_next_obs_samples=int(args.td_target_next_obs_samples),
                    policy_batch_size=int(args.demo_prefill_target_batch_size),
                    dp_sampler=str(args.dp_sampler),
                    dp_infer_steps=int(args.dp_infer_steps),
                    action_history_len=int(args.action_history_len),
                    critic_obs_state_mode=str(args.critic_obs_state_mode),
                    critic_obs_action_mode=str(args.critic_obs_action_mode),
                    critic_obs_unnormalize_dp_chunk=bool(
                        args.critic_obs_unnormalize_dp_chunk
                    ),
                    max_episodes=int(args.demo_prefill_max_episodes),
                    max_transitions=int(args.demo_prefill_max_transitions),
                    window_mode=str(args.offline_demo_window_mode),
                )
                metrics = demo_prefill_summary.as_metrics(prefix="demo_prefill")
                metrics.update(
                    {
                        "offline/phase": 0.0,
                        "time/policy_steps": float(policy_steps),
                        "time/env_steps": float(env_steps),
                        "replay/size": float(len(replay)),
                        **demo_prefill_metrics(),
                    }
                )
                record_offline_event(metrics, env_axis=int(env_steps), critic_updates=0)
                for mk, mv in metrics.items():
                    writer.add_scalar(mk, float(mv), int(env_steps))
                metadata["demo_prefill_summary"] = dict(demo_prefill_summary.__dict__)
                metadata["demo_prefill_replay_state_saved"] = False
                with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
                current_mix_ratio, current_demo_batch, current_online_batch = (
                    current_demo_prefill_batch_sizes()
                )
                line = (
                    f"[ExRL][OFFLINE][DEMO-PREFILL] transitions={demo_prefill_summary.transitions_added} "
                    f"starts={demo_prefill_summary.full_window_starts} "
                    f"terminals={demo_prefill_summary.terminal_count} "
                    f"target_states={demo_prefill_summary.cached_target_states} "
                    f"demo_replay={len(demo_replay)} "
                    f"demo_capacity={demo_prefill_replay_capacity} "
                    f"mix_ratio={current_mix_ratio:.4f} "
                    f"mix_demo_batch={current_demo_batch} "
                    f"mix_online_batch={current_online_batch}"
                )
                print(line, flush=True)
                _append_status_line(status_log_path, line)
                dump_metrics()
                writer.flush()

            # Phase 0: pure random k collection, no critic/actor/alpha updates and
            # no rollout-success logging. This is intentionally just replay fill.
            next_phase0_log = int(
                offline_phase_state.get(
                    "next_phase0_log",
                    _next_multiple_after(
                        policy_steps, int(args.log_every_policy_steps)
                    ),
                )
            )
            next_offline_latest_policy = int(
                offline_phase_state.get(
                    "next_offline_latest_policy",
                    _next_multiple_after(
                        policy_steps, int(args.latest_model_save_freq)
                    ),
                )
            )
            next_offline_periodic_policy = int(
                offline_phase_state.get(
                    "next_offline_periodic_policy",
                    _next_multiple_after(policy_steps, int(args.save_freq)),
                )
            )
            phase0_saved_latest_model = False
            if resume_phase_id <= 0:
                refresh_common_phase_state(
                    0,
                    phase0_source=phase0_source,
                    phase0_target=int(phase0_target),
                    next_phase0_log=int(next_phase0_log),
                    next_offline_latest_policy=int(next_offline_latest_policy),
                    next_offline_periodic_policy=int(next_offline_periodic_policy),
                )
            if resume_phase_id <= 0 and phase0_source == "demo_npz":
                demo_summary = populate_replay_from_demo_npz(
                    replay=replay,
                    policy=policy,
                    demo_npz=str(args.offline_demo_npz),
                    stats_npz=str(args.stats_npz_override),
                    normalization_npz=str(args.dsrl_normalization_npz),
                    lowdim_keys=lowdim_keys,
                    lowdim_dims=lowdim_dims,
                    action_dim=action_dim_from_stats,
                    max_chunk_len=int(args.max_chunk_len),
                    min_chunk_len=int(args.min_chunk_len),
                    gamma=float(args.gamma),
                    reward_offset=float(args.reward_offset),
                    reward_duration_mode=str(args.reward_duration_mode),
                    target_next_obs_samples=int(args.td_target_next_obs_samples),
                    policy_batch_size=int(args.offline_demo_target_batch_size),
                    dp_sampler=str(args.dp_sampler),
                    dp_infer_steps=int(args.dp_infer_steps),
                    action_history_len=int(args.action_history_len),
                    critic_obs_state_mode=str(args.critic_obs_state_mode),
                    critic_obs_action_mode=str(args.critic_obs_action_mode),
                    critic_obs_unnormalize_dp_chunk=bool(
                        args.critic_obs_unnormalize_dp_chunk
                    ),
                    max_episodes=int(args.offline_demo_max_episodes),
                    max_transitions=int(args.offline_demo_max_transitions),
                    window_mode=str(args.offline_demo_window_mode),
                )
                num_emitted_transitions_total += int(demo_summary.transitions_added)
                num_done_for_backup_true += int(demo_summary.terminal_count)
                duration_sum_total += float(demo_summary.duration_sum)
                metrics = demo_summary.as_metrics()
                metrics.update(
                    {
                        "offline/phase": 0.0,
                        "time/policy_steps": float(policy_steps),
                        "time/env_steps": float(env_steps),
                        "replay/size": float(len(replay)),
                    }
                )
                record_offline_event(metrics, env_axis=int(env_steps), critic_updates=0)
                for mk, mv in metrics.items():
                    writer.add_scalar(mk, float(mv), int(env_steps))
                metadata["offline_demo_summary"] = dict(demo_summary.__dict__)
                with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
                line = (
                    f"[ExRL][OFFLINE][DEMO] transitions={demo_summary.transitions_added} "
                    f"starts={demo_summary.full_window_starts} terminals={demo_summary.terminal_count} "
                    f"target_states={demo_summary.cached_target_states} replay={len(replay)}"
                )
                print(line, flush=True)
                _append_status_line(status_log_path, line)
                dump_metrics()
                writer.flush()
                refresh_common_phase_state(
                    1,
                    phase0_source=phase0_source,
                    phase0_target=int(phase0_target),
                    offline_demo_summary=dict(demo_summary.__dict__),
                    next_phase0_log=int(next_phase0_log),
                    next_offline_latest_policy=int(next_offline_latest_policy),
                    next_offline_periodic_policy=int(next_offline_periodic_policy),
                )
                _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
                phase0_saved_latest_model = True
                resume_phase_id = 1
            while (
                resume_phase_id <= 0
                and phase0_source == "online_random"
                and policy_steps < int(phase0_target)
            ):
                actions = np.random.randint(
                    0, action_dim, size=args.n_envs, dtype=np.int64
                )
                env_actions = actions + action_shift
                next_obs, rewards, dones, infos = venv.step(env_actions)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
                dones = np.asarray(dones, dtype=bool).reshape(-1)
                emitted_transitions = 0
                for i in range(args.n_envs):
                    info = infos[i] if i < len(infos) else {}
                    ingest_result = ingest_smdp_env_step_to_replay(
                        venv=venv,
                        env_idx=i,
                        obs_i=obs[i],
                        next_obs_i=next_obs[i],
                        reward_i=float(rewards[i]),
                        done_i=bool(dones[i]),
                        info=info,
                        action_id=int(actions[i]),
                        pending_queue=nstep_queues[i],
                        replay=replay,
                        gamma=float(args.gamma),
                        n_step=int(args.n_step),
                        reward_duration_mode=args.reward_duration_mode,
                        max_chunk_len=int(args.max_chunk_len),
                        action_dim=action_dim,
                        target_next_obs_samples=int(args.td_target_next_obs_samples),
                    )
                    k = float(ingest_result.duration_k)
                    emitted_transitions += int(ingest_result.emitted_transitions)
                    num_emitted_transitions_total += int(
                        ingest_result.emitted_transitions
                    )
                    num_done_for_backup_true += int(bool(ingest_result.done_for_backup))
                    num_timeout_truncations += int(bool(ingest_result.truncated))
                    duration_sum_total += float(k)
                    env_steps += int(k)
                    exec_buffer.append(k)
                obs = next_obs
                policy_steps += int(args.n_envs)
                step_calls += 1
                if env_eval_interval > 0:
                    while next_env_eval_axis is not None and int(env_steps) >= int(
                        next_env_eval_axis
                    ):
                        record_env_step_eval(
                            eval_axis=int(next_env_eval_axis),
                            phase_id=0.0,
                            critic_updates=0,
                        )
                        next_env_eval_axis += env_eval_interval
                if policy_steps >= next_phase0_log:
                    elapsed = max(time.time() - start_time, 1e-6)
                    avg_exec = (
                        float(np.mean(exec_buffer)) if len(exec_buffer) > 0 else 0.0
                    )
                    metrics = {
                        "offline/phase": 0.0,
                        "time/policy_steps": float(policy_steps),
                        "time/env_steps": float(env_steps),
                        "time/policy_steps_per_sec": float(policy_steps / elapsed),
                        "time/env_steps_per_sec": float(env_steps / elapsed),
                        "adaptive/avg_exec_len": avg_exec,
                        "replay/size": float(len(replay)),
                        **demo_prefill_metrics(),
                    }
                    record_offline_event(
                        metrics, env_axis=int(env_steps), critic_updates=0
                    )
                    for mk, mv in metrics.items():
                        writer.add_scalar(mk, float(mv), int(env_steps))
                    line = (
                        f"[ExRL][Warmup] policy_steps={policy_steps}/{phase0_target} "
                        f"env_steps={env_steps} avg_k={avg_exec:.2f} replay={len(replay)} "
                        f"demo={0 if demo_replay is None else len(demo_replay)}"
                    )
                    print(line, flush=True)
                    _append_status_line(status_log_path, line)
                    dump_metrics()
                    writer.flush()
                    next_phase0_log += max(int(args.log_every_policy_steps), 1)

                refresh_common_phase_state(
                    0,
                    phase0_target=int(phase0_target),
                    next_phase0_log=int(next_phase0_log),
                    next_offline_latest_policy=int(next_offline_latest_policy),
                    next_offline_periodic_policy=int(next_offline_periodic_policy),
                )
                if (
                    bool(args.save_periodic_models)
                    and policy_steps >= next_offline_periodic_policy
                ):
                    next_offline_periodic_policy += max(int(args.save_freq), 1)
                    refresh_common_phase_state(
                        0,
                        phase0_target=int(phase0_target),
                        next_phase0_log=int(next_phase0_log),
                        next_offline_latest_policy=int(next_offline_latest_policy),
                        next_offline_periodic_policy=int(next_offline_periodic_policy),
                    )
                    _save_ckpt(
                        os.path.join(
                            ckpt_dir, f"adaptive_agent_{policy_steps}_steps.pt"
                        ),
                        pack_ckpt(),
                    )
                if policy_steps >= next_offline_latest_policy:
                    next_offline_latest_policy += max(
                        int(args.latest_model_save_freq), 1
                    )
                    refresh_common_phase_state(
                        0,
                        phase0_target=int(phase0_target),
                        next_phase0_log=int(next_phase0_log),
                        next_offline_latest_policy=int(next_offline_latest_policy),
                        next_offline_periodic_policy=int(next_offline_periodic_policy),
                    )
                    _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
                if next_offline_latest_env is not None and int(env_steps) >= int(
                    next_offline_latest_env
                ):
                    while next_offline_latest_env is not None and int(env_steps) >= int(
                        next_offline_latest_env
                    ):
                        next_offline_latest_env += max(
                            int(offline_resume_env_ckpt_freq), 1
                        )
                    refresh_common_phase_state(
                        0,
                        phase0_target=int(phase0_target),
                        next_phase0_log=int(next_phase0_log),
                        next_offline_latest_policy=int(next_offline_latest_policy),
                        next_offline_periodic_policy=int(next_offline_periodic_policy),
                    )
                    _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())

            if not has_offline_update_batch():
                _, required_demo_batch, required_online_batch = (
                    current_demo_prefill_batch_sizes()
                )
                raise RuntimeError(
                    "offline replay sources too small for mixed batch: "
                    f"online_replay={len(replay)} required_online={required_online_batch} "
                    f"demo_replay={0 if demo_replay is None else len(demo_replay)} "
                    f"required_demo={required_demo_batch} batch={args.batch_size}"
                )
            phase1_target = int(args.initial_critic_updates)
            phase1_updates = int(
                offline_phase_state.get(
                    "phase1_updates",
                    phase1_target if resume_phase_id >= 2 else 0,
                )
            )
            if resume_phase_id <= 0:
                phase1_updates = 0
                resume_phase_id = 1
            if (
                bool(args.use_mc_lower_bound)
                and phase1_updates <= 0
                and resume_phase_id <= 1
            ):
                valid_mc = replay.compute_mc_returns_in_place(gamma=float(args.gamma))
                line = f"[ExRL][OFFLINE] computed MC returns for {valid_mc}/{len(replay)} replay transitions"
                print(line, flush=True)
                _append_status_line(status_log_path, line)
                if int(valid_mc) <= 0:
                    raise RuntimeError(
                        "MC lower bound requested but no replay transition has a complete return-to-go; "
                        "increase phase-0 collection length or enable terminal/horizon episode boundaries"
                    )
            refresh_common_phase_state(
                1 if resume_phase_id <= 1 else int(resume_phase_id),
                phase0_target=int(phase0_target),
                phase1_updates=int(phase1_updates),
            )
            if (
                resume_phase_id <= 1
                and phase1_updates == 0
                and (not phase0_saved_latest_model)
            ):
                _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())

            # Phase 1: critic-only replay training. Evaluation periodically rolls
            # out critic-as-actor and logs success, so pure critic improvement is visible.
            next_phase1_log = int(
                offline_phase_state.get(
                    "next_phase1_log",
                    _next_multiple_after(
                        phase1_updates, int(args.log_every_policy_steps)
                    ),
                )
            )
            eval_every = int(args.offline_eval_every_updates)
            offline_resume_critic_ckpt_freq = int(
                args.offline_resume_save_interval_steps
            )
            if offline_resume_critic_ckpt_freq <= 0:
                offline_resume_critic_ckpt_freq = (
                    int(eval_every) if int(eval_every) > 0 else 0
                )
            if eval_every <= 0 and resume_phase_id <= 1:
                try:
                    venv.close()
                except Exception:
                    pass
                venv = None
            next_eval = int(
                offline_phase_state.get(
                    "next_eval",
                    (
                        _next_multiple_after(phase1_updates, eval_every)
                        if eval_every > 0
                        else phase1_target + 1
                    ),
                )
            )
            next_offline_latest_critic = int(
                offline_phase_state.get(
                    "next_offline_latest_critic",
                    _next_multiple_after(
                        phase1_updates, int(args.latest_model_save_freq)
                    ),
                )
            )
            next_offline_periodic_critic = int(
                offline_phase_state.get(
                    "next_offline_periodic_critic",
                    _next_multiple_after(phase1_updates, int(args.save_freq)),
                )
            )
            next_offline_latest_critic_resume = (
                int(
                    offline_phase_state.get(
                        "next_offline_latest_critic_resume",
                        _next_multiple_after(
                            phase1_updates, offline_resume_critic_ckpt_freq
                        ),
                    )
                )
                if offline_resume_critic_ckpt_freq > 0
                else None
            )
            last_phase1_eval_update = int(
                offline_phase_state.get("last_phase1_eval_update", -1)
            )
            train_meter.clear()

            def refresh_phase1_state(phase_id: int = 1):
                refresh_common_phase_state(
                    int(phase_id),
                    phase0_target=int(phase0_target),
                    phase1_updates=int(phase1_updates),
                    next_phase1_log=int(next_phase1_log),
                    next_eval=int(next_eval),
                    next_offline_latest_critic=int(next_offline_latest_critic),
                    next_offline_periodic_critic=int(next_offline_periodic_critic),
                    offline_resume_critic_ckpt_freq=int(
                        offline_resume_critic_ckpt_freq
                    ),
                    next_offline_latest_critic_resume=(
                        None
                        if next_offline_latest_critic_resume is None
                        else int(next_offline_latest_critic_resume)
                    ),
                    last_phase1_eval_update=int(last_phase1_eval_update),
                )

            refresh_phase1_state(1 if resume_phase_id <= 1 else int(resume_phase_id))
            while resume_phase_id <= 1 and phase1_updates < int(
                args.initial_critic_updates
            ):
                update_batch, mix_metrics = sample_offline_update_batch()
                m = sac.update(
                    update_batch,
                    update_actor=False,
                    update_alpha=False,
                    transition_weight_mode=str(args.transition_weight_mode),
                    actor_prior_coeff=0.0,
                    actor_prior_min_action_id=0,
                    legal_min_action_id=0,
                )
                m.update(mix_metrics)
                for k, v in m.items():
                    train_meter[k].append(float(v))
                    latest_train_metrics[k] = float(v)
                phase1_updates += 1
                num_updates_total += 1

                do_eval = eval_every > 0 and phase1_updates >= next_eval
                do_log = phase1_updates >= next_phase1_log or do_eval
                if do_log:
                    elapsed = max(time.time() - start_time, 1e-6)
                    metrics = {
                        "offline/phase": 1.0,
                        "offline/critic_updates": float(phase1_updates),
                        "time/policy_steps": float(policy_steps),
                        "time/env_steps": float(env_steps),
                        "time/critic_updates_per_sec": float(phase1_updates / elapsed),
                        "replay/size": float(len(replay)),
                        **demo_prefill_metrics(),
                    }
                    metrics.update(_mean_meter(train_meter))
                    train_meter.clear()
                    # Keep the primary x-axis on training rollout env steps.
                    # Phase-1 eval probes critic quality but does not consume
                    # training data, so eval env steps are diagnostics only.
                    env_axis = int(env_steps)
                    if do_eval:
                        eval_metrics = _run_critic_policy_eval(
                            venv=ensure_eval_venv(allow_collector_fallback=True),
                            sac=sac,
                            action_shift=action_shift,
                            episodes=int(args.eval_episodes),
                            max_decisions=int(args.offline_eval_max_decisions),
                            policy_mode=str(args.offline_eval_policy_mode),
                            critic_reduce=str(args.offline_eval_critic_reduce),
                        )
                        offline_eval_env_steps_total += int(
                            eval_metrics["offline_eval/critic_env_steps"]
                        )
                        env_axis = int(env_steps)
                        metrics.update(eval_metrics)
                        metrics["time/offline_eval_env_steps_total"] = float(
                            offline_eval_env_steps_total
                        )
                        metrics["rollout/success_rate"] = float(
                            eval_metrics["offline_eval/critic_success_rate"]
                        )
                        metrics["rollout/success_rate_recent20"] = float(
                            eval_metrics["offline_eval/critic_success_rate_recent20"]
                        )
                        metrics["adaptive/avg_exec_len"] = float(
                            eval_metrics["offline_eval/critic_avg_exec_len"]
                        )
                        if bool(args.save_best_model) and float(
                            metrics["rollout/success_rate"]
                        ) > float(best_success):
                            best_success = float(metrics["rollout/success_rate"])
                            refresh_phase1_state(1)
                            _save_ckpt(
                                os.path.join(ckpt_dir, "best_model.pt"), pack_ckpt()
                            )
                        last_phase1_eval_update = int(phase1_updates)
                        next_eval += max(eval_every, 1)
                    record_offline_event(
                        metrics, env_axis=env_axis, critic_updates=phase1_updates
                    )
                    for mk, mv in metrics.items():
                        writer_step = (
                            int(phase1_updates)
                            if mk.startswith("offline") or mk.startswith("train/")
                            else env_axis
                        )
                        writer.add_scalar(mk, float(mv), writer_step)
                    line = (
                        f"[ExRL][Critic initialization] updates={phase1_updates}/{args.initial_critic_updates} "
                        f"env_axis={env_axis} replay={len(replay)} demo={0 if demo_replay is None else len(demo_replay)} "
                        f"critic_loss={metrics.get('train/critic_loss', 0.0):.4f}"
                    )
                    if "offline_eval/critic_success_rate" in metrics:
                        line += (
                            f" eval_sr={metrics['offline_eval/critic_success_rate']:.4f} "
                            f"eval_avg_k={metrics['offline_eval/critic_avg_exec_len']:.2f}"
                        )
                    print(line, flush=True)
                    _append_status_line(status_log_path, line)
                    dump_metrics()
                    writer.flush()
                    if phase1_updates >= next_phase1_log:
                        next_phase1_log += max(int(args.log_every_policy_steps), 1)

                refresh_phase1_state(1)
                if (
                    bool(args.save_periodic_models)
                    and phase1_updates >= next_offline_periodic_critic
                ):
                    next_offline_periodic_critic += max(int(args.save_freq), 1)
                    refresh_phase1_state(1)
                    _save_ckpt(
                        os.path.join(
                            ckpt_dir,
                            f"adaptive_agent_critic_update_{phase1_updates}.pt",
                        ),
                        pack_ckpt(),
                    )
                if phase1_updates >= next_offline_latest_critic:
                    next_offline_latest_critic += max(
                        int(args.latest_model_save_freq), 1
                    )
                    refresh_phase1_state(1)
                    _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
                if (
                    next_offline_latest_critic_resume is not None
                    and phase1_updates >= int(next_offline_latest_critic_resume)
                ):
                    while (
                        next_offline_latest_critic_resume is not None
                        and phase1_updates >= int(next_offline_latest_critic_resume)
                    ):
                        next_offline_latest_critic_resume += max(
                            int(offline_resume_critic_ckpt_freq), 1
                        )
                    refresh_phase1_state(1)
                    _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())

            if (
                resume_phase_id <= 1
                and bool(args.offline_eval_after_training)
                and int(last_phase1_eval_update) != int(phase1_updates)
            ):
                eval_metrics = _run_critic_policy_eval(
                    venv=ensure_eval_venv(allow_collector_fallback=True),
                    sac=sac,
                    action_shift=action_shift,
                    episodes=int(args.eval_episodes),
                    max_decisions=int(args.offline_eval_max_decisions),
                    policy_mode=str(args.offline_eval_policy_mode),
                    critic_reduce=str(args.offline_eval_critic_reduce),
                )
                offline_eval_env_steps_total += int(
                    eval_metrics["offline_eval/critic_env_steps"]
                )
                metrics = {
                    "offline/phase": 1.0,
                    "offline/critic_updates": float(phase1_updates),
                    "time/policy_steps": float(policy_steps),
                    "time/env_steps": float(env_steps),
                    "time/offline_eval_env_steps_total": float(
                        offline_eval_env_steps_total
                    ),
                    "replay/size": float(len(replay)),
                    "rollout/success_rate": float(
                        eval_metrics["offline_eval/critic_success_rate"]
                    ),
                    "rollout/success_rate_recent20": float(
                        eval_metrics["offline_eval/critic_success_rate_recent20"]
                    ),
                    "adaptive/avg_exec_len": float(
                        eval_metrics["offline_eval/critic_avg_exec_len"]
                    ),
                    **demo_prefill_metrics(),
                }
                metrics.update(eval_metrics)
                metrics.update(_mean_meter(train_meter))
                train_meter.clear()
                last_phase1_eval_update = int(phase1_updates)
                if bool(args.save_best_model) and float(
                    metrics["rollout/success_rate"]
                ) > float(best_success):
                    best_success = float(metrics["rollout/success_rate"])
                    refresh_phase1_state(1)
                    _save_ckpt(os.path.join(ckpt_dir, "best_model.pt"), pack_ckpt())
                refresh_phase1_state(1)
                record_offline_event(
                    metrics, env_axis=int(env_steps), critic_updates=int(phase1_updates)
                )
                for mk, mv in metrics.items():
                    writer_step = (
                        int(phase1_updates)
                        if mk.startswith("offline") or mk.startswith("train/")
                        else int(env_steps)
                    )
                    writer.add_scalar(mk, float(mv), writer_step)
                line = (
                    f"[ExRL][OFFLINE][FINAL-EVAL] updates={phase1_updates}/{args.initial_critic_updates} "
                    f"sr={metrics['offline_eval/critic_success_rate']:.4f} "
                    f"sr20={metrics['offline_eval/critic_success_rate_recent20']:.4f} "
                    f"avg_k={metrics['offline_eval/critic_avg_exec_len']:.2f}"
                )
                print(line, flush=True)
                _append_status_line(status_log_path, line)
                dump_metrics()
                writer.flush()

            if resume_phase_id <= 1:
                resume_phase_id = 2
            phase2_target = int(args.online_transitions)
            if phase2_target <= 0:
                refresh_common_phase_state(
                    3,
                    phase0_target=int(phase0_target),
                    phase1_updates=int(phase1_updates),
                    phase2_target=int(phase2_target),
                )
                _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
                dump_metrics()
                print("[ExRL][OFFLINE] phase01 finished", flush=True)
                _append_status_line(status_log_path, "[ExRL][OFFLINE] phase01 finished")
                return

            # Phase 2: use the critic itself as the rollout policy, append real
            # environment data, and keep training the critic only. The x-axis is
            # still real training env steps: phase-1 updates did not advance it.
            if venv is None:
                venv = _build_task_venv(
                    args=args,
                    policy=policy,
                    device=device,
                    lowdim_keys=lowdim_keys,
                    lowdim_dims=lowdim_dims,
                    env_meta=env_meta,
                    n_envs=int(args.n_envs),
                    seed=int(args.seed),
                )
            obs = np.asarray(venv.reset(), dtype=np.float32)
            for q in nstep_queues:
                q.clear()
            phase2_start_env_steps = int(
                offline_phase_state.get("phase2_start_env_steps", int(env_steps))
            )
            phase2_policy_steps = int(offline_phase_state.get("phase2_policy_steps", 0))
            phase2_critic_updates = int(
                offline_phase_state.get("phase2_critic_updates", 0)
            )
            phase2_step_calls = int(offline_phase_state.get("phase2_step_calls", 0))
            phase2_pending_transitions = int(
                offline_phase_state.get("phase2_pending_transitions", 0)
            )
            phase2_next_log = int(
                offline_phase_state.get(
                    "phase2_next_log",
                    _next_multiple_after(
                        phase2_policy_steps, int(args.log_every_policy_steps)
                    ),
                )
            )
            phase2_success_buffer = deque(
                offline_phase_state.get("phase2_success_buffer", []),
                maxlen=int(args.window_size),
            )
            phase2_success_buffer_20 = deque(
                offline_phase_state.get("phase2_success_buffer_20", []),
                maxlen=20,
            )
            phase2_episode_return_buffer = deque(
                offline_phase_state.get("phase2_episode_return_buffer", []),
                maxlen=int(args.window_size),
            )
            phase2_episode_len_buffer = deque(
                offline_phase_state.get("phase2_episode_len_buffer", []),
                maxlen=int(args.window_size),
            )
            phase2_exec_buffer = deque(
                offline_phase_state.get("phase2_exec_buffer", []),
                maxlen=int(args.window_size),
            )
            phase2_episode_returns = np.zeros((args.n_envs,), dtype=np.float32)
            phase2_episode_lengths = np.zeros((args.n_envs,), dtype=np.int64)
            saved_thompson_heads = np.asarray(
                offline_phase_state.get("phase2_thompson_heads", []),
                dtype=np.int64,
            ).reshape(-1)
            if saved_thompson_heads.shape[0] == int(args.n_envs):
                phase2_thompson_heads = saved_thompson_heads.copy()
            else:
                phase2_thompson_heads = np.random.randint(
                    0,
                    max(int(args.critic_ensemble_size), 1),
                    size=int(args.n_envs),
                    dtype=np.int64,
                )
            next_offline_latest_policy = int(
                offline_phase_state.get(
                    "next_offline_latest_policy",
                    _next_multiple_after(
                        policy_steps, int(args.latest_model_save_freq)
                    ),
                )
            )
            next_offline_periodic_policy = int(
                offline_phase_state.get(
                    "next_offline_periodic_policy",
                    _next_multiple_after(policy_steps, int(args.save_freq)),
                )
            )
            train_meter.clear()
            phase2_msg = (
                f"[ExRL][Training] start policy_steps={phase2_policy_steps}/{phase2_target} "
                f"mode={args.collection_policy} reduce={args.collection_critic_reduce} "
                f"epsilon={float(args.collection_epsilon)} "
                f"ucb_beta={float(args.collection_ucb_beta)} "
                f"start_env_steps={phase2_start_env_steps}"
            )
            print(phase2_msg, flush=True)
            _append_status_line(status_log_path, phase2_msg)

            def refresh_phase2_state(phase_id: int = 2):
                refresh_common_phase_state(
                    int(phase_id),
                    phase0_target=int(phase0_target),
                    phase1_updates=int(phase1_updates),
                    phase2_target=int(phase2_target),
                    phase2_start_env_steps=int(phase2_start_env_steps),
                    phase2_policy_steps=int(phase2_policy_steps),
                    phase2_critic_updates=int(phase2_critic_updates),
                    phase2_step_calls=int(phase2_step_calls),
                    phase2_pending_transitions=int(phase2_pending_transitions),
                    phase2_next_log=int(phase2_next_log),
                    phase2_success_buffer=list(phase2_success_buffer),
                    phase2_success_buffer_20=list(phase2_success_buffer_20),
                    phase2_episode_return_buffer=list(phase2_episode_return_buffer),
                    phase2_episode_len_buffer=list(phase2_episode_len_buffer),
                    phase2_exec_buffer=list(phase2_exec_buffer),
                    phase2_thompson_heads=phase2_thompson_heads.tolist(),
                    next_offline_latest_policy=int(next_offline_latest_policy),
                    next_offline_periodic_policy=int(next_offline_periodic_policy),
                )

            refresh_common_phase_state(
                2,
                phase0_target=int(phase0_target),
                phase1_updates=int(phase1_updates),
                phase2_target=int(phase2_target),
                phase2_start_env_steps=int(phase2_start_env_steps),
                phase2_policy_steps=int(phase2_policy_steps),
                phase2_critic_updates=int(phase2_critic_updates),
                phase2_step_calls=int(phase2_step_calls),
                phase2_pending_transitions=int(phase2_pending_transitions),
                phase2_next_log=int(phase2_next_log),
                phase2_success_buffer=list(phase2_success_buffer),
                phase2_success_buffer_20=list(phase2_success_buffer_20),
                phase2_episode_return_buffer=list(phase2_episode_return_buffer),
                phase2_episode_len_buffer=list(phase2_episode_len_buffer),
                phase2_exec_buffer=list(phase2_exec_buffer),
                phase2_thompson_heads=phase2_thompson_heads.tolist(),
                next_offline_latest_policy=int(next_offline_latest_policy),
                next_offline_periodic_policy=int(next_offline_periodic_policy),
            )
            _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())

            while phase2_policy_steps < phase2_target:
                current_ucb_beta = _resolve_float_schedule(
                    int(env_steps),
                    phase2_ucb_beta_schedule_items,
                    default=float(args.collection_ucb_beta),
                )
                select_out = sac.select_critic_action(
                    obs,
                    deterministic=(str(args.collection_policy) == "q_greedy"),
                    min_action_id=0,
                    mode=str(args.collection_policy),
                    reduce=str(args.collection_critic_reduce),
                    ucb_beta=float(current_ucb_beta),
                    thompson_heads=(
                        phase2_thompson_heads
                        if str(args.collection_policy) == "q_thompson"
                        else None
                    ),
                    return_info=True,
                )
                actions, action_info = select_out
                for k, v in action_info.items():
                    train_meter[f"diagnostic/{k}"].append(float(v))
                phase2_epsilon = float(args.collection_epsilon)
                if phase2_epsilon > 0.0:
                    random_actions = np.random.randint(
                        0, action_dim, size=args.n_envs, dtype=np.int64
                    )
                    use_random = np.random.rand(args.n_envs) < phase2_epsilon
                    actions = np.where(use_random, random_actions, actions).astype(
                        np.int64
                    )
                env_actions = actions + action_shift
                next_obs, rewards, dones, infos = venv.step(env_actions)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
                dones = np.asarray(dones, dtype=bool).reshape(-1)
                emitted_transitions = 0
                for i in range(args.n_envs):
                    info = infos[i] if i < len(infos) else {}
                    ingest_result = ingest_smdp_env_step_to_replay(
                        venv=venv,
                        env_idx=i,
                        obs_i=obs[i],
                        next_obs_i=next_obs[i],
                        reward_i=float(rewards[i]),
                        done_i=bool(dones[i]),
                        info=info,
                        action_id=int(actions[i]),
                        pending_queue=nstep_queues[i],
                        replay=replay,
                        gamma=float(args.gamma),
                        n_step=int(args.n_step),
                        reward_duration_mode=args.reward_duration_mode,
                        max_chunk_len=int(args.max_chunk_len),
                        action_dim=action_dim,
                        target_next_obs_samples=int(args.td_target_next_obs_samples),
                    )
                    k = float(ingest_result.duration_k)
                    emitted_transitions += int(ingest_result.emitted_transitions)
                    num_emitted_transitions_total += int(
                        ingest_result.emitted_transitions
                    )
                    num_done_for_backup_true += int(bool(ingest_result.done_for_backup))
                    num_timeout_truncations += int(bool(ingest_result.truncated))
                    duration_sum_total += float(k)
                    env_steps += int(k)
                    phase2_exec_buffer.append(k)
                    phase2_episode_returns[i] += float(rewards[i])
                    phase2_episode_lengths[i] += 1
                    if bool(dones[i]):
                        success_value = 1 if _extract_success(info) else 0
                        phase2_success_buffer.append(success_value)
                        phase2_success_buffer_20.append(success_value)
                        phase2_episode_return_buffer.append(
                            float(phase2_episode_returns[i])
                        )
                        phase2_episode_len_buffer.append(
                            float(phase2_episode_lengths[i])
                        )
                        phase2_episode_returns[i] = 0.0
                        phase2_episode_lengths[i] = 0
                        if str(args.collection_policy) == "q_thompson":
                            phase2_thompson_heads[i] = np.random.randint(
                                0, max(int(args.critic_ensemble_size), 1)
                            )

                obs = next_obs
                policy_steps += int(args.n_envs)
                phase2_policy_steps += int(args.n_envs)
                step_calls += 1
                phase2_step_calls += 1
                phase2_pending_transitions += int(emitted_transitions)

                should_update = False
                if has_offline_update_batch():
                    if int(args.train_freq_transitions) > 0:
                        should_update = phase2_pending_transitions >= int(
                            args.train_freq_transitions
                        )
                    else:
                        should_update = phase2_step_calls % int(args.train_freq) == 0
                current_utd = _resolve_int_schedule(
                    int(env_steps) if utd_schedule_axis == "env" else int(policy_steps),
                    utd_schedule_items,
                    default=int(args.utd),
                )
                if should_update:
                    if int(args.train_freq_transitions) > 0:
                        phase2_pending_transitions -= int(args.train_freq_transitions)
                    for _ in range(int(current_utd)):
                        update_batch, mix_metrics = sample_offline_update_batch()
                        m = sac.update(
                            update_batch,
                            update_actor=False,
                            update_alpha=False,
                            transition_weight_mode=str(args.transition_weight_mode),
                            actor_prior_coeff=0.0,
                            actor_prior_min_action_id=0,
                            legal_min_action_id=0,
                        )
                        m.update(mix_metrics)
                        for k, v in m.items():
                            train_meter[k].append(float(v))
                            latest_train_metrics[k] = float(v)
                    num_updates_total += int(current_utd)
                    phase2_critic_updates += int(current_utd)

                if env_eval_interval > 0:
                    while next_env_eval_axis is not None and int(env_steps) >= int(
                        next_env_eval_axis
                    ):
                        record_env_step_eval(
                            eval_axis=int(next_env_eval_axis),
                            phase_id=2.0,
                            critic_updates=int(phase1_updates + phase2_critic_updates),
                        )
                        next_env_eval_axis += env_eval_interval

                if phase2_policy_steps >= phase2_next_log:
                    elapsed = max(time.time() - start_time, 1e-6)
                    success_recent = (
                        float(np.mean(phase2_success_buffer))
                        if len(phase2_success_buffer) > 0
                        else 0.0
                    )
                    success_recent20 = (
                        float(np.mean(phase2_success_buffer_20))
                        if len(phase2_success_buffer_20) > 0
                        else 0.0
                    )
                    avg_exec = (
                        float(np.mean(phase2_exec_buffer))
                        if len(phase2_exec_buffer) > 0
                        else 0.0
                    )
                    ep_rew_mean = (
                        float(np.mean(phase2_episode_return_buffer))
                        if len(phase2_episode_return_buffer) > 0
                        else 0.0
                    )
                    ep_len_mean = (
                        float(np.mean(phase2_episode_len_buffer))
                        if len(phase2_episode_len_buffer) > 0
                        else 0.0
                    )
                    metrics = {
                        "offline/phase": 2.0,
                        "offline/critic_updates": float(
                            phase1_updates + phase2_critic_updates
                        ),
                        "offline/phase2_policy_steps": float(phase2_policy_steps),
                        "offline/phase2_env_steps": float(
                            int(env_steps) - phase2_start_env_steps
                        ),
                        "offline/phase2_critic_updates": float(phase2_critic_updates),
                        "time/policy_steps": float(policy_steps),
                        "time/env_steps": float(env_steps),
                        "time/policy_steps_per_sec": float(policy_steps / elapsed),
                        "time/env_steps_per_sec": float(env_steps / elapsed),
                        "adaptive/success_rate_recent": success_recent,
                        "adaptive/success_rate_recent20": success_recent20,
                        "rollout/success_rate": success_recent,
                        "rollout/success_rate_recent20": success_recent20,
                        "rollout/ep_rew_mean": ep_rew_mean,
                        "rollout/ep_len_mean": ep_len_mean,
                        "adaptive/avg_exec_len": avg_exec,
                        "diagnostic/current_utd": float(current_utd),
                        "diagnostic/current_ucb_beta": float(current_ucb_beta),
                        "replay/size": float(len(replay)),
                        **demo_prefill_metrics(),
                    }
                    metrics.update(_mean_meter(train_meter))
                    train_meter.clear()
                    record_offline_event(
                        metrics,
                        env_axis=int(env_steps),
                        critic_updates=int(phase1_updates + phase2_critic_updates),
                    )
                    for mk, mv in metrics.items():
                        writer_step = (
                            int(phase1_updates + phase2_critic_updates)
                            if mk.startswith("offline") or mk.startswith("train/")
                            else int(env_steps)
                        )
                        writer.add_scalar(mk, float(mv), writer_step)
                    line = (
                        f"[ExRL][Training] policy_steps={phase2_policy_steps}/{phase2_target} "
                        f"env_steps={env_steps} sr={success_recent:.4f} sr20={success_recent20:.4f} "
                        f"avg_k={avg_exec:.2f} replay={len(replay)} demo={0 if demo_replay is None else len(demo_replay)} "
                        f"updates={phase2_critic_updates}"
                    )
                    print(line, flush=True)
                    _append_status_line(status_log_path, line)
                    dump_metrics()
                    writer.flush()
                    refresh_phase2_state()
                    if (
                        bool(args.save_best_model)
                        and success_recent > best_success
                        and len(phase2_success_buffer) > 0
                    ):
                        best_success = success_recent
                        _save_ckpt(os.path.join(ckpt_dir, "best_model.pt"), pack_ckpt())
                    phase2_next_log += max(int(args.log_every_policy_steps), 1)
                    refresh_phase2_state()

                refresh_phase2_state()
                if (
                    bool(args.save_periodic_models)
                    and policy_steps >= next_offline_periodic_policy
                ):
                    next_offline_periodic_policy += max(int(args.save_freq), 1)
                    refresh_phase2_state()
                    _save_ckpt(
                        os.path.join(
                            ckpt_dir, f"adaptive_agent_{policy_steps}_steps.pt"
                        ),
                        pack_ckpt(),
                    )
                if policy_steps >= next_offline_latest_policy:
                    next_offline_latest_policy += max(
                        int(args.latest_model_save_freq), 1
                    )
                    refresh_phase2_state()
                    _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
                if next_offline_latest_env is not None and int(env_steps) >= int(
                    next_offline_latest_env
                ):
                    while next_offline_latest_env is not None and int(env_steps) >= int(
                        next_offline_latest_env
                    ):
                        next_offline_latest_env += max(
                            int(offline_resume_env_ckpt_freq), 1
                        )
                    refresh_phase2_state()
                    _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())

            refresh_phase2_state(phase_id=3)
            _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
            dump_metrics()
            print("[ExRL][OFFLINE] phase012 finished", flush=True)
            _append_status_line(status_log_path, "[ExRL][OFFLINE] phase012 finished")
            return

        while policy_steps < int(args.total_policy_steps):
            model_probe_k = None
            current_min_k_floor = 1
            current_min_action_id = max(
                int(current_min_k_floor) - int(args.min_chunk_len), 0
            )
            dynamic_floor_signal = None
            if policy_steps < int(args.warmup_policy_steps) and bool(
                args.probe_model_k_during_warmup
            ):
                probe_actions = sac.select_action(
                    obs, deterministic=bool(args.warmup_probe_deterministic)
                )
                model_probe_k = float(
                    np.mean(
                        probe_actions.astype(np.float32) + float(args.min_chunk_len)
                    )
                )
            if policy_steps < int(args.warmup_policy_steps):
                mode = str(args.warmup_action_mode)
                if mode == "uniform":
                    actions = np.random.randint(
                        0, action_dim, size=args.n_envs, dtype=np.int64
                    )
                elif mode == "fixed":
                    actions = np.full(
                        (args.n_envs,),
                        int(args.warmup_fixed_k) - int(args.min_chunk_len),
                        dtype=np.int64,
                    )
                elif mode == "mixed":
                    fixed_id = int(args.warmup_fixed_k) - int(args.min_chunk_len)
                    rand_actions = np.random.randint(
                        0, action_dim, size=args.n_envs, dtype=np.int64
                    )
                    use_rand = np.random.rand(args.n_envs) < float(
                        args.warmup_mixed_uniform_prob
                    )
                    actions = np.where(use_rand, rand_actions, fixed_id).astype(
                        np.int64
                    )
                else:
                    raise RuntimeError(f"unknown warmup action mode: {mode}")
            else:
                schedule_counter = (
                    int(env_steps)
                    if min_k_schedule_axis == "env"
                    else int(policy_steps)
                )
                scheduled_min_k_floor = _resolve_min_k_floor(
                    schedule_counter, args, min_k_schedule_items
                )
                dynamic_floor_signal = _resolve_dynamic_floor_signal(
                    args.dynamic_floor_release_metric,
                    latest_train_metrics,
                    success_buffer,
                )
                if (
                    not dynamic_floor_release_latched
                    and str(args.dynamic_floor_release_metric) != "none"
                    and int(policy_steps)
                    >= int(args.dynamic_floor_release_after_policy_steps)
                    and dynamic_floor_signal is not None
                    and float(dynamic_floor_signal)
                    >= float(args.dynamic_floor_release_threshold)
                ):
                    dynamic_floor_release_latched = True
                    dynamic_floor_release_policy_step = int(policy_steps)
                    dynamic_floor_release_env_step = int(env_steps)
                    release_msg = (
                        f"[ExRL][FLOOR-RELEASE] metric={args.dynamic_floor_release_metric} "
                        f"value={float(dynamic_floor_signal):.6f} threshold={float(args.dynamic_floor_release_threshold):.6f} "
                        f"policy_steps={policy_steps} env_steps={env_steps}"
                    )
                    print(release_msg, flush=True)
                    _append_status_line(status_log_path, release_msg)
                if dynamic_floor_release_latched:
                    if dynamic_release_target_items:
                        current_min_k_floor = _resolve_min_k_floor(
                            schedule_counter, args, dynamic_release_target_items
                        )
                    else:
                        current_min_k_floor = 1
                else:
                    current_min_k_floor = int(scheduled_min_k_floor)
                min_action_id = max(
                    int(current_min_k_floor) - int(args.min_chunk_len), 0
                )
                current_min_action_id = int(min_action_id)
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=sac.device, dtype=torch.float32)
                    logits_t, probs_t, _ = sac.actor.distribution(obs_t)
                    raw_dist = torch.distributions.Categorical(logits=logits_t)
                    actions_raw = (
                        raw_dist.sample().detach().cpu().numpy().astype(np.int64)
                    )
                    # When min_chunk_len=0, floor=1 should already exclude k=0.
                    use_floor_mask_sample = (
                        floor_support_mask_mode in ("sample_only", "all")
                        and min_action_id > 0
                    )
                    if use_floor_mask_sample:
                        _, sampling_probs_t, _ = sac.actor.distribution(
                            obs_t, min_action_id=min_action_id
                        )
                    else:
                        sampling_probs_t = probs_t
                    behavior_mix_beta = _resolve_float_schedule(
                        int(policy_steps), behavior_mix_schedule_items, default=0.0
                    )
                    if behavior_mix_beta > 0.0:
                        probs_np = (
                            sampling_probs_t.detach().cpu().numpy().astype(np.float64)
                        )
                        prior = np.zeros_like(probs_np, dtype=np.float64)
                        prior[:, min_action_id:] = 1.0 / float(
                            action_dim - min_action_id
                        )
                        mixed = (1.0 - float(behavior_mix_beta)) * probs_np + float(
                            behavior_mix_beta
                        ) * prior
                        mixed /= np.clip(
                            mixed.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None
                        )
                        actions = np.asarray(
                            [np.random.choice(action_dim, p=row) for row in mixed],
                            dtype=np.int64,
                        )
                    elif use_floor_mask_sample:
                        masked_dist = torch.distributions.Categorical(
                            probs=sampling_probs_t
                        )
                        actions = (
                            masked_dist.sample().detach().cpu().numpy().astype(np.int64)
                        )
                    else:
                        actions = actions_raw.copy()

                    epsilon_greedy_prob = float(args.epsilon_greedy_prob)
                    if epsilon_greedy_prob > 0.0:
                        rand_low = min_action_id if min_action_id > 0 else 0
                        random_actions = np.random.randint(
                            rand_low,
                            action_dim,
                            size=args.n_envs,
                            dtype=np.int64,
                        )
                        use_random = np.random.rand(args.n_envs) < epsilon_greedy_prob
                        actions = np.where(use_random, random_actions, actions).astype(
                            np.int64
                        )
                        epsilon_apply_buffer.append(float(np.mean(use_random)))
                    else:
                        epsilon_apply_buffer.append(0.0)

                if min_action_id > 0:
                    actions = np.maximum(actions, min_action_id).astype(np.int64)
                floor_apply_buffer.append(float(np.mean(actions > actions_raw)))
                raw_exec_buffer.append(
                    float(
                        np.mean(
                            actions_raw.astype(np.float32) + float(args.min_chunk_len)
                        )
                    )
                )
                behavior_mix_buffer.append(float(behavior_mix_beta))
            env_actions = actions + action_shift

            next_obs, rewards, dones, infos = venv.step(env_actions)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
            dones = np.asarray(dones, dtype=bool).reshape(-1)
            if next_obs.shape != obs.shape:
                raise RuntimeError(
                    f"obs shape changed unexpectedly: {obs.shape} -> {next_obs.shape}"
                )

            durations = np.zeros((args.n_envs,), dtype=np.float32)
            emitted_transitions = 0
            for i in range(args.n_envs):
                info = infos[i] if i < len(infos) else {}
                ingest_result = ingest_smdp_env_step_to_replay(
                    venv=venv,
                    env_idx=i,
                    obs_i=obs[i],
                    next_obs_i=next_obs[i],
                    reward_i=float(rewards[i]),
                    done_i=bool(dones[i]),
                    info=info,
                    action_id=int(actions[i]),
                    pending_queue=nstep_queues[i],
                    replay=replay,
                    gamma=float(args.gamma),
                    n_step=int(args.n_step),
                    reward_duration_mode=args.reward_duration_mode,
                    max_chunk_len=int(args.max_chunk_len),
                    action_dim=action_dim,
                    target_next_obs_samples=int(args.td_target_next_obs_samples),
                )
                k = float(ingest_result.duration_k)
                durations[i] = k
                emitted_transitions += int(ingest_result.emitted_transitions)
                num_emitted_transitions_total += int(ingest_result.emitted_transitions)
                num_done_for_backup_true += int(bool(ingest_result.done_for_backup))
                num_timeout_truncations += int(bool(ingest_result.truncated))
                duration_sum_total += float(k)

                env_steps += int(k)
                exec_buffer.append(k)
                episode_returns[i] += float(rewards[i])
                episode_lengths[i] += 1

                if bool(dones[i]):
                    success_value = 1 if _extract_success(info) else 0
                    success_buffer.append(success_value)
                    success_buffer_20.append(success_value)
                    episode_return_buffer.append(float(episode_returns[i]))
                    episode_len_buffer.append(float(episode_lengths[i]))
                    episode_returns[i] = 0.0
                    episode_lengths[i] = 0

            obs = next_obs
            policy_steps += int(args.n_envs)
            step_calls += 1
            pending_transitions_for_update += int(emitted_transitions)

            should_update = False
            if policy_steps >= int(learning_starts) and len(replay) >= int(
                args.batch_size
            ):
                if int(args.train_freq_transitions) > 0:
                    should_update = pending_transitions_for_update >= int(
                        args.train_freq_transitions
                    )
                else:
                    should_update = step_calls % int(args.train_freq) == 0

            if should_update:
                if int(args.train_freq_transitions) > 0:
                    pending_transitions_for_update -= int(args.train_freq_transitions)
                update_actor = policy_steps >= int(actor_learning_starts)
                update_alpha = policy_steps >= int(alpha_learning_starts)
                utd_counter = (
                    int(env_steps) if utd_schedule_axis == "env" else int(policy_steps)
                )
                current_utd = _resolve_int_schedule(
                    utd_counter, utd_schedule_items, default=int(args.utd)
                )
                actor_prior_coeff = _resolve_float_schedule(
                    int(policy_steps), actor_prior_schedule_items, default=0.0
                )
                actor_prior_buffer.append(float(actor_prior_coeff))
                for _ in range(int(current_utd)):
                    m = sac.update(
                        replay.sample(int(args.batch_size)),
                        update_actor=bool(update_actor),
                        update_alpha=bool(update_alpha),
                        transition_weight_mode=str(args.transition_weight_mode),
                        actor_prior_coeff=float(actor_prior_coeff),
                        actor_prior_min_action_id=int(current_min_action_id),
                        legal_min_action_id=(
                            int(current_min_action_id)
                            if floor_support_mask_mode in ("update_only", "all")
                            else 0
                        ),
                    )
                    for k, v in m.items():
                        train_meter[k].append(float(v))
                        latest_train_metrics[k] = float(v)
                num_updates_total += int(current_utd)

            if policy_steps >= next_log:
                elapsed = max(time.time() - start_time, 1e-6)
                success_recent = (
                    float(np.mean(success_buffer)) if len(success_buffer) > 0 else 0.0
                )
                success_recent20 = (
                    float(np.mean(success_buffer_20))
                    if len(success_buffer_20) > 0
                    else 0.0
                )
                avg_exec = float(np.mean(exec_buffer)) if len(exec_buffer) > 0 else 0.0
                ep_rew_mean = (
                    float(np.mean(episode_return_buffer))
                    if len(episode_return_buffer) > 0
                    else 0.0
                )
                ep_len_mean = (
                    float(np.mean(episode_len_buffer))
                    if len(episode_len_buffer) > 0
                    else 0.0
                )
                avg_duration_k_cum = float(
                    duration_sum_total / max(int(policy_steps), 1)
                )
                emitted_per_env_transition = float(
                    num_emitted_transitions_total / max(int(policy_steps), 1)
                )
                emitted_per_collector_step = float(
                    num_emitted_transitions_total / max(int(step_calls), 1)
                )
                updates_per_collector_step = float(
                    num_updates_total / max(int(step_calls), 1)
                )

                _record(history, "time/policy_steps", float(policy_steps))
                _record(history, "time/env_steps", float(env_steps))
                _record(
                    history, "time/policy_steps_per_sec", float(policy_steps / elapsed)
                )
                _record(history, "time/env_steps_per_sec", float(env_steps / elapsed))
                _record(history, "adaptive/success_rate_recent", success_recent)
                _record(history, "adaptive/success_rate_recent20", success_recent20)
                _record(history, "rollout/success_rate", success_recent)
                _record(history, "rollout/success_rate_recent20", success_recent20)
                _record(history, "rollout/ep_rew_mean", ep_rew_mean)
                _record(history, "rollout/ep_len_mean", ep_len_mean)
                _record(history, "adaptive/avg_exec_len", avg_exec)
                _record(
                    history, "adaptive/success_rate_recent_env_step", success_recent
                )
                _record(
                    history, "adaptive/success_rate_recent20_env_step", success_recent20
                )
                _record(history, "rollout/success_rate_env_step", success_recent)
                _record(
                    history, "rollout/success_rate_recent20_env_step", success_recent20
                )
                _record(history, "adaptive/avg_exec_len_env_step", avg_exec)
                if len(raw_exec_buffer) > 0:
                    _record(
                        history,
                        "diagnostic/policy_exec_len_raw_recent",
                        float(np.mean(raw_exec_buffer)),
                    )
                if len(floor_apply_buffer) > 0:
                    _record(
                        history,
                        "diagnostic/floor_applied_ratio_recent",
                        float(np.mean(floor_apply_buffer)),
                    )
                if len(behavior_mix_buffer) > 0:
                    _record(
                        history,
                        "diagnostic/behavior_mix_beta_recent",
                        float(np.mean(behavior_mix_buffer)),
                    )
                if len(epsilon_apply_buffer) > 0:
                    _record(
                        history,
                        "diagnostic/epsilon_greedy_applied_ratio_recent",
                        float(np.mean(epsilon_apply_buffer)),
                    )
                _record(
                    history,
                    "diagnostic/epsilon_greedy_prob",
                    float(args.epsilon_greedy_prob),
                )
                if len(actor_prior_buffer) > 0:
                    _record(
                        history,
                        "diagnostic/actor_prior_coeff_recent",
                        float(np.mean(actor_prior_buffer)),
                    )
                _record(
                    history,
                    "diagnostic/current_utd",
                    float(
                        current_utd
                        if should_update
                        else _resolve_int_schedule(
                            (
                                int(env_steps)
                                if utd_schedule_axis == "env"
                                else int(policy_steps)
                            ),
                            utd_schedule_items,
                            default=int(args.utd),
                        )
                    ),
                )
                _record(history, "diagnostic/min_k_floor", float(current_min_k_floor))
                _record(
                    history,
                    "diagnostic/dynamic_floor_release_latched",
                    float(dynamic_floor_release_latched),
                )
                if dynamic_floor_signal is not None:
                    _record(
                        history,
                        "diagnostic/dynamic_floor_signal_recent",
                        float(dynamic_floor_signal),
                    )
                _record(history, "replay/size", float(len(replay)))
                if bool(args.debug_cadence_metrics):
                    _record(history, "debug/collector_steps", float(step_calls))
                    _record(
                        history,
                        "debug/num_emitted_transitions",
                        float(num_emitted_transitions_total),
                    )
                    _record(history, "debug/num_updates", float(num_updates_total))
                    _record(
                        history,
                        "debug/num_done_for_backup_true",
                        float(num_done_for_backup_true),
                    )
                    _record(
                        history,
                        "debug/num_timeout_truncations",
                        float(num_timeout_truncations),
                    )
                    _record(history, "debug/avg_duration_k_cum", avg_duration_k_cum)
                    _record(
                        history,
                        "debug/emitted_per_env_transition",
                        emitted_per_env_transition,
                    )
                    _record(
                        history,
                        "debug/emitted_per_collector_step",
                        emitted_per_collector_step,
                    )
                    _record(
                        history,
                        "debug/updates_per_collector_step",
                        updates_per_collector_step,
                    )
                # Keep primary history x-axis on env-step to match TensorBoard.
                history["timesteps"].append(int(env_steps))
                history["timesteps_env"].append(int(env_steps))
                history["timesteps_policy"].append(int(policy_steps))

                writer.add_scalar(
                    "adaptive/success_rate_recent", success_recent, env_steps
                )
                writer.add_scalar(
                    "adaptive/success_rate_recent20", success_recent20, env_steps
                )
                writer.add_scalar("rollout/success_rate", success_recent, env_steps)
                writer.add_scalar(
                    "rollout/success_rate_recent20", success_recent20, env_steps
                )
                writer.add_scalar("rollout/ep_rew_mean", ep_rew_mean, env_steps)
                writer.add_scalar("rollout/ep_len_mean", ep_len_mean, env_steps)
                writer.add_scalar("adaptive/avg_exec_len", avg_exec, env_steps)
                if len(raw_exec_buffer) > 0:
                    writer.add_scalar(
                        "diagnostic/policy_exec_len_raw_recent",
                        float(np.mean(raw_exec_buffer)),
                        env_steps,
                    )
                if len(floor_apply_buffer) > 0:
                    writer.add_scalar(
                        "diagnostic/floor_applied_ratio_recent",
                        float(np.mean(floor_apply_buffer)),
                        env_steps,
                    )
                if len(behavior_mix_buffer) > 0:
                    writer.add_scalar(
                        "diagnostic/behavior_mix_beta_recent",
                        float(np.mean(behavior_mix_buffer)),
                        env_steps,
                    )
                if len(epsilon_apply_buffer) > 0:
                    writer.add_scalar(
                        "diagnostic/epsilon_greedy_applied_ratio_recent",
                        float(np.mean(epsilon_apply_buffer)),
                        env_steps,
                    )
                writer.add_scalar(
                    "diagnostic/epsilon_greedy_prob",
                    float(args.epsilon_greedy_prob),
                    env_steps,
                )
                if len(actor_prior_buffer) > 0:
                    writer.add_scalar(
                        "diagnostic/actor_prior_coeff_recent",
                        float(np.mean(actor_prior_buffer)),
                        env_steps,
                    )
                writer.add_scalar(
                    "diagnostic/current_utd",
                    float(
                        current_utd
                        if should_update
                        else _resolve_int_schedule(
                            (
                                int(env_steps)
                                if utd_schedule_axis == "env"
                                else int(policy_steps)
                            ),
                            utd_schedule_items,
                            default=int(args.utd),
                        )
                    ),
                    env_steps,
                )
                writer.add_scalar(
                    "diagnostic/min_k_floor", float(current_min_k_floor), env_steps
                )
                writer.add_scalar(
                    "diagnostic/dynamic_floor_release_latched",
                    float(dynamic_floor_release_latched),
                    env_steps,
                )
                if dynamic_floor_signal is not None:
                    writer.add_scalar(
                        "diagnostic/dynamic_floor_signal_recent",
                        float(dynamic_floor_signal),
                        env_steps,
                    )
                writer.add_scalar(
                    "time/policy_steps_per_sec",
                    float(policy_steps / elapsed),
                    env_steps,
                )
                writer.add_scalar(
                    "time/env_steps_per_sec", float(env_steps / elapsed), env_steps
                )
                writer.add_scalar("replay/size", float(len(replay)), env_steps)
                if bool(args.debug_cadence_metrics):
                    writer.add_scalar(
                        "debug/collector_steps", float(step_calls), env_steps
                    )
                    writer.add_scalar(
                        "debug/num_emitted_transitions",
                        float(num_emitted_transitions_total),
                        env_steps,
                    )
                    writer.add_scalar(
                        "debug/num_updates", float(num_updates_total), env_steps
                    )
                    writer.add_scalar(
                        "debug/num_done_for_backup_true",
                        float(num_done_for_backup_true),
                        env_steps,
                    )
                    writer.add_scalar(
                        "debug/num_timeout_truncations",
                        float(num_timeout_truncations),
                        env_steps,
                    )
                    writer.add_scalar(
                        "debug/avg_duration_k_cum", avg_duration_k_cum, env_steps
                    )
                    writer.add_scalar(
                        "debug/emitted_per_env_transition",
                        emitted_per_env_transition,
                        env_steps,
                    )
                    writer.add_scalar(
                        "debug/emitted_per_collector_step",
                        emitted_per_collector_step,
                        env_steps,
                    )
                    writer.add_scalar(
                        "debug/updates_per_collector_step",
                        updates_per_collector_step,
                        env_steps,
                    )
                if model_probe_k is not None:
                    _record(
                        history, "diagnostic/model_exec_len_probe_warmup", model_probe_k
                    )
                    writer.add_scalar(
                        "diagnostic/model_exec_len_probe_warmup",
                        model_probe_k,
                        env_steps,
                    )

                for k, vals in list(train_meter.items()):
                    if len(vals) > 0:
                        mean_val = float(np.mean(vals))
                        _record(history, k, mean_val)
                        writer.add_scalar(k, mean_val, env_steps)
                    train_meter[k].clear()

                log_idx += 1
                if (log_idx % max(int(args.metrics_flush_every_logs), 1)) == 0:
                    dump_metrics()

                line = (
                    f"[ExRL] steps={policy_steps}/{args.total_policy_steps} "
                    f"env_steps={env_steps} sr={success_recent:.4f} "
                    f"avg_k={avg_exec:.2f} replay={len(replay)} "
                    f"floor={current_min_k_floor} rel={int(dynamic_floor_release_latched)}"
                )
                if bool(args.debug_cadence_metrics):
                    line += (
                        f" collector_steps={step_calls}"
                        f" emitted={num_emitted_transitions_total}"
                        f" updates={num_updates_total}"
                        f" done_backup={num_done_for_backup_true}"
                        f" trunc={num_timeout_truncations}"
                        f" avg_dur_cum={avg_duration_k_cum:.2f}"
                        f" emit_per_env={emitted_per_env_transition:.3f}"
                        f" emit_per_collect={emitted_per_collector_step:.3f}"
                        f" upd_per_collect={updates_per_collector_step:.3f}"
                    )
                print(line, flush=True)
                _append_status_line(status_log_path, line)
                try:
                    writer.flush()
                except Exception:
                    pass
                next_log += max(int(args.log_every_policy_steps), 1)

                if (
                    bool(args.save_best_model)
                    and success_recent > best_success
                    and len(success_buffer) > 0
                ):
                    best_success = success_recent
                    _save_ckpt(os.path.join(ckpt_dir, "best_model.pt"), pack_ckpt())

            if bool(args.save_periodic_models) and policy_steps >= next_periodic_ckpt:
                _save_ckpt(
                    os.path.join(ckpt_dir, f"adaptive_agent_{policy_steps}_steps.pt"),
                    pack_ckpt(),
                )
                next_periodic_ckpt += max(int(args.save_freq), 1)

            if policy_steps >= next_latest_ckpt:
                _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
                next_latest_ckpt += max(int(args.latest_model_save_freq), 1)

        _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
        dump_metrics()
        print("[ExRL] training finished", flush=True)
        _append_status_line(status_log_path, "[ExRL] training finished")
    finally:
        try:
            writer.flush()
            writer.close()
        except Exception:
            pass
        try:
            if eval_venv is not None and eval_venv is not venv:
                eval_venv.close()
        except Exception:
            pass
        try:
            if venv is not None:
                venv.close()
        except Exception:
            pass


def main():
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args()
    _validate_task_alignment(args)
    os.makedirs(args.output, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
