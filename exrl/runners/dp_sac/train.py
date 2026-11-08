"""Train ExRL with a DP base policy and SAC updates."""

import argparse
import atexit
import fcntl
import gc
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

# Keep Ampere/Ada behavior numerically aligned with the V100 reference runs.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

from exrl.paths import ROOT as REPO_ROOT

import robomimic.utils.torch_utils as TorchUtils
from exrl.runners.dp_sac.environment import ExecutionLengthEnv
from exrl.runners.dp_sac.policy_loader import (
    infer_action_dim_from_stats,
    infer_lowdim_dims_from_stats,
    load_dsrl_policy_wrapper,
    load_env_meta_json,
    parse_lowdim_keys,
)
from exrl.algorithms.dp_sac import (
    extract_bootstrap_next_obs_from_venv,
    ingest_smdp_env_step_to_replay,
    PendingSMDPTransition,
    SMDPCudaGraphUpdate,
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
    if int(args.post_warmup_min_k_until_policy_steps) > 0 and int(counter_value) < int(
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


def _find_wrapper_perf_acc(venv):
    """Find cumulative timing counters through VecEnv wrappers."""
    seen = set()
    stack = [venv]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        perf = getattr(obj, "_perf_acc", None)
        if isinstance(perf, dict):
            return perf
        for attr in ("venv", "env"):
            child = getattr(obj, attr, None)
            if child is not None:
                stack.append(child)
    return None


def _history_has_env_step_eval_at_zero(history: Dict) -> bool:
    steps = history.get("env_step_eval", [])
    values = history.get("metrics", {}).get("env_step_eval/success_rate", [])
    for step, _value in zip(steps, values):
        if int(step) == 0:
            return True
    return False


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


def _make_online_eval_venv(
    args, policy, device, lowdim_keys, lowdim_dims, max_episode_steps: int
):
    eval_episodes = int(args.online_eval_episodes)
    if eval_episodes <= 0:
        raise ValueError(
            "--online_eval_episodes must be > 0 when online eval is enabled"
        )
    eval_n_envs = int(args.online_eval_n_envs)
    if eval_n_envs <= 0:
        eval_n_envs = min(int(args.n_envs), eval_episodes)
    eval_n_envs = max(1, min(eval_n_envs, eval_episodes))

    eval_venv = ExecutionLengthEnv(
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
        dp_cuda_graph=args.dp_cuda_graph,
        ipc_chunk_dtype=args.ipc_chunk_dtype,
        dp_infer_gpus=args.dp_infer_gpus,
        dp_grouping=args.dp_grouping,
        dp_infer_parallel=args.dp_infer_parallel,
        ipc_send_mode=args.ipc_send_mode,
        perf_profile=bool(args.perf_profile),
        vendor_root=args.vendor_root,
        env_name=args.env_name,
        env_meta_json=args.env_meta_json,
        normalization_path=args.dsrl_normalization_npz,
        stats_npz_path=args.stats_npz_override,
        lowdim_keys=lowdim_keys,
        lowdim_dims=lowdim_dims,
        n_envs=eval_n_envs,
        max_episode_steps=int(max_episode_steps),
        gamma=args.gamma,
        seed=int(args.seed) + 1_000_003,
        min_chunk_len=int(args.min_chunk_len),
        zero_action_fail_streak=int(args.zero_action_fail_streak),
        zero_action_fail_reward=float(args.zero_action_fail_reward),
        terminate_on_success=bool(args.terminate_on_success),
        timeout_on_horizon=bool(args.timeout_on_horizon),
        dsrl_variable_k_runtime=str(args.dsrl_variable_k_runtime),
        dsrl_prune_unused_observables=bool(args.dsrl_prune_unused_observables),
        dsrl_worker_cpu_ids=(
            sorted(_parse_cpu_affinity(args.dsrl_worker_cpu_ids))
            if args.dsrl_worker_cpu_ids
            else ()
        ),
        dsrl_cache_static_robot_metadata=bool(args.dsrl_cache_static_robot_metadata),
        physical_obs_clip=float(args.physical_obs_clip),
        physical_obs_normalization=str(args.physical_obs_normalization),
        action_obs_normalization=str(args.action_obs_normalization),
    )
    return VecMonitor(eval_venv)


def _run_online_greedy_eval(
    *,
    eval_venv,
    sac: ExecutionLengthAgent,
    action_shift: int,
    episodes: int,
    max_decisions_per_episode: int,
) -> float:
    episodes = int(episodes)
    max_decisions_per_episode = int(max_decisions_per_episode)
    if episodes <= 0:
        raise ValueError("online eval episodes must be > 0")
    if max_decisions_per_episode <= 0:
        raise ValueError("online eval max decisions must be > 0")

    was_training = bool(sac.actor.training)
    sac.actor.eval()
    successes: List[int] = []
    obs = np.asarray(eval_venv.reset(), dtype=np.float32)
    decisions = np.zeros((eval_venv.num_envs,), dtype=np.int64)
    try:
        while len(successes) < episodes:
            actions = sac.select_action(obs, deterministic=True, min_action_id=0)
            next_obs, _, dones, infos = eval_venv.step(actions + int(action_shift))
            obs = np.asarray(next_obs, dtype=np.float32)
            dones = np.asarray(dones, dtype=bool).reshape(-1)
            decisions += 1

            force_reset_all = False
            for i, done_i in enumerate(dones):
                if len(successes) >= episodes:
                    break
                if bool(done_i):
                    info = infos[i] if i < len(infos) else {}
                    successes.append(1 if _extract_success(info) else 0)
                    decisions[i] = 0
                elif decisions[i] >= max_decisions_per_episode:
                    successes.append(0)
                    force_reset_all = True
                    if len(successes) >= episodes:
                        break

            if force_reset_all and len(successes) < episodes:
                obs = np.asarray(eval_venv.reset(), dtype=np.float32)
                decisions[:] = 0
    finally:
        if was_training:
            sac.actor.train()
    return float(np.mean(successes[:episodes])) if successes else 0.0


def _run_critic_policy_eval(
    *,
    eval_venv,
    sac: ExecutionLengthAgent,
    action_shift: int,
    episodes: int,
    max_decisions_per_episode: int,
    policy_mode: str,
    critic_reduce: str,
) -> float:
    episodes = int(episodes)
    max_decisions_per_episode = int(max_decisions_per_episode)
    if episodes <= 0:
        raise ValueError("offline eval episodes must be > 0")
    if max_decisions_per_episode <= 0:
        raise ValueError("offline eval max decisions must be > 0")

    successes: List[int] = []
    obs = np.asarray(eval_venv.reset(), dtype=np.float32)
    decisions = np.zeros((eval_venv.num_envs,), dtype=np.int64)
    while len(successes) < episodes:
        actions = sac.select_critic_action(
            obs,
            deterministic=(str(policy_mode).lower() == "q_greedy"),
            min_action_id=0,
            mode=str(policy_mode),
            reduce=str(critic_reduce),
        )
        next_obs, _, dones, infos = eval_venv.step(actions + int(action_shift))
        obs = np.asarray(next_obs, dtype=np.float32)
        dones = np.asarray(dones, dtype=bool).reshape(-1)
        decisions += 1

        force_reset_all = False
        for i, done_i in enumerate(dones):
            if len(successes) >= episodes:
                break
            if bool(done_i):
                info = infos[i] if i < len(infos) else {}
                successes.append(1 if _extract_success(info) else 0)
                decisions[i] = 0
            elif decisions[i] >= max_decisions_per_episode:
                successes.append(0)
                force_reset_all = True
                if len(successes) >= episodes:
                    break

        if force_reset_all and len(successes) < episodes:
            obs = np.asarray(eval_venv.reset(), dtype=np.float32)
            decisions[:] = 0
    return float(np.mean(successes[:episodes])) if successes else 0.0


from exrl.runners.dp_sac.options import build_parser


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".sac.", suffix=".tmp", dir=os.path.dirname(path))
    os.close(fd)
    policy_steps = int(payload.get("policy_steps", -1))
    replay_state = payload.get("replay_state")
    replay_size = (
        int(replay_state.get("size", 0)) if isinstance(replay_state, dict) else 0
    )
    started_at = time.time()
    print(
        f"[ExRL][CHECKPOINT] saving path={path} policy_steps={policy_steps} "
        f"replay_size={replay_size}",
        flush=True,
    )
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
        print(
            f"[ExRL][CHECKPOINT] saved path={path} bytes={os.path.getsize(path)} "
            f"elapsed_sec={time.time() - started_at:.1f}",
            flush=True,
        )
    finally:
        payload = None
        replay_state = None
        gc.collect()
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _next_interval_boundary(counter: int, interval: int) -> int:
    interval = max(int(interval), 1)
    return (max(int(counter), 0) // interval + 1) * interval


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
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
        if isinstance(config_data, dict):
            config_data["dsrl_schedule_defaults"] = ckpt_meta.get(
                "schedule_defaults", {}
            )
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)
    except Exception as exc:
        print(
            f"[ExRL][WARN] failed to annotate config.json with DSRL schedule defaults: {exc}"
        )
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(
        f"[ExRL][ExRL] env={args.env_name} horizon={env_meta['env_kwargs']['horizon']} "
        f"weight_source={args.weight_source}"
    )
    schedule_defaults = ckpt_meta.get("schedule_defaults", {})
    print(
        "[ExRL][ExRL] diffusion_schedule "
        f"train_denoising_steps={schedule_defaults.get('train_denoising_steps')} "
        f"default_inference_steps={schedule_defaults.get('default_inference_steps')} "
        f"source={schedule_defaults.get('source')} runtime_dp_infer_steps={args.dp_infer_steps}"
    )

    if args.decision_runtime != "sync":
        print("[ExRL][WARN] This runner uses synchronous chunk execution.")

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
        dp_cuda_graph=args.dp_cuda_graph,
        ipc_chunk_dtype=args.ipc_chunk_dtype,
        dp_infer_gpus=args.dp_infer_gpus,
        dp_grouping=args.dp_grouping,
        dp_infer_parallel=args.dp_infer_parallel,
        ipc_send_mode=args.ipc_send_mode,
        perf_profile=bool(args.perf_profile),
        vendor_root=args.vendor_root,
        env_name=args.env_name,
        env_meta_json=args.env_meta_json,
        normalization_path=args.dsrl_normalization_npz,
        stats_npz_path=args.stats_npz_override,
        lowdim_keys=lowdim_keys,
        lowdim_dims=lowdim_dims,
        n_envs=args.n_envs,
        max_episode_steps=int(
            env_meta.get("env_kwargs", {}).get("horizon", args.env_horizon_override)
        ),
        gamma=args.gamma,
        seed=args.seed,
        min_chunk_len=int(args.min_chunk_len),
        zero_action_fail_streak=int(args.zero_action_fail_streak),
        zero_action_fail_reward=float(args.zero_action_fail_reward),
        terminate_on_success=bool(args.terminate_on_success),
        timeout_on_horizon=bool(args.timeout_on_horizon),
        dsrl_variable_k_runtime=str(args.dsrl_variable_k_runtime),
        dsrl_prune_unused_observables=bool(args.dsrl_prune_unused_observables),
        dsrl_worker_cpu_ids=(
            sorted(_parse_cpu_affinity(args.dsrl_worker_cpu_ids))
            if args.dsrl_worker_cpu_ids
            else ()
        ),
        dsrl_cache_static_robot_metadata=bool(args.dsrl_cache_static_robot_metadata),
        physical_obs_clip=float(args.physical_obs_clip),
        physical_obs_normalization=str(args.physical_obs_normalization),
        action_obs_normalization=str(args.action_obs_normalization),
    )
    venv = VecMonitor(venv)

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
    if not (
        int(args.min_chunk_len)
        <= int(args.post_warmup_min_k)
        <= int(args.max_chunk_len)
    ):
        raise ValueError(
            f"--post_warmup_min_k must be in [{int(args.min_chunk_len)}, {args.max_chunk_len}]"
        )
    if int(args.learning_starts_policy_steps) > 0 and int(
        args.learning_starts_policy_steps
    ) < int(args.warmup_policy_steps):
        raise ValueError(
            "--learning_starts_policy_steps must be >= warmup_policy_steps when set"
        )
    if int(args.train_freq_transitions) < 0:
        raise ValueError("--train_freq_transitions must be >= 0")
    if int(args.online_eval_every_env_steps) < 0:
        raise ValueError("--online_eval_every_env_steps must be >= 0")
    if int(args.eval_snapshot_env_freq) < 0:
        raise ValueError("--eval_snapshot_env_freq must be >= 0")
    if (
        not np.isfinite(float(args.physical_obs_clip))
        or float(args.physical_obs_clip) < 0.0
    ):
        raise ValueError(
            "--physical_obs_clip must be finite and >= 0; use 0 to disable"
        )
    if int(args.online_eval_every_env_steps) > 0:
        if int(args.online_eval_episodes) <= 0:
            raise ValueError(
                "--online_eval_episodes must be > 0 when online eval is enabled"
            )
        if int(args.online_eval_max_decisions_per_episode) <= 0:
            raise ValueError("--online_eval_max_decisions_per_episode must be > 0")
    if (
        args.target_q_clip_min is not None
        and args.target_q_clip_max is not None
        and float(args.target_q_clip_min) > float(args.target_q_clip_max)
    ):
        raise ValueError("--target_q_clip_min cannot exceed --target_q_clip_max")
    if str(args.pipeline) == "sac_warmstart":
        offline_phase0_steps = (
            int(args.warmup_transitions)
            if int(args.warmup_transitions) >= 0
            else int(args.warmup_policy_steps)
        )
        if offline_phase0_steps <= 0:
            raise ValueError("offline warmstart requires phase0 steps > 0")
        if int(args.initial_critic_updates) <= 0:
            raise ValueError("--initial_critic_updates must be > 0")
        if int(args.initial_actor_updates) < 0:
            raise ValueError("--initial_actor_updates must be >= 0")
        if int(args.offline_eval_every_updates) < 0:
            raise ValueError("--offline_eval_every_updates must be >= 0")
        if int(args.eval_episodes) <= 0:
            raise ValueError("--eval_episodes must be > 0")
        if int(args.offline_eval_max_decisions) <= 0:
            raise ValueError("--offline_eval_max_decisions must be > 0")
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
    utd_schedule_axis = str(args.utd_schedule_axis)
    utd_schedule_items = _parse_int_schedule(
        args.utd_schedule, min_value=1, max_value=None
    )

    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    if bool(args.sac_cuda_graph) and bool(args.selective_critic_forward):
        raise ValueError(
            "--sac_cuda_graph and --selective_critic_forward are mutually exclusive"
        )
    effective_foreach_target_update = bool(
        args.foreach_target_update or args.sac_cuda_graph
    )
    effective_async_batch_validation = bool(
        args.async_batch_validation or args.sac_cuda_graph
    )
    if bool(args.sac_cuda_graph):
        print(
            "[ExRL][EXPERIMENTAL] full CUDA Graph uses capturable Adam and changes "
            "the numerical optimization trajectory versus the standard run",
            flush=True,
        )
    if bool(args.fused_optimizer):
        print(
            "[ExRL][EXPERIMENTAL] fused Adam changes the numerical optimization "
            "trajectory versus the standard run",
            flush=True,
        )
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
        critic_backup_temp=float(args.critic_backup_temp),
        cql_alpha=float(args.cql_alpha),
        cql_temp=float(args.cql_temp),
        critic_ensemble_size=int(args.critic_ensemble_size),
        target_critic_subsample_size=int(args.target_critic_subsample_size),
        critic_bootstrap_mask_prob=float(args.critic_bootstrap_mask_prob),
        actor_target_critic_subsample=bool(args.actor_target_critic_subsample),
        actor_loss_critic_subsample=bool(args.actor_loss_critic_subsample),
        selective_critic_forward=bool(args.selective_critic_forward),
        foreach_target_update=effective_foreach_target_update,
        async_batch_validation=effective_async_batch_validation,
        capturable_optimizer=bool(args.sac_cuda_graph),
        fused_optimizer=bool(args.fused_optimizer),
        target_q_clip_min=args.target_q_clip_min,
        target_q_clip_max=args.target_q_clip_max,
        device=device,
    )
    replay = SMDPReplayBuffer(obs_dim=obs_dim, capacity=args.replay_size, device=device)
    nstep_queues = [deque() for _ in range(args.n_envs)]

    policy_steps = 0
    env_steps = 0
    step_calls = 0
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
    pending_transitions_for_update = 0
    num_emitted_transitions_total = 0
    num_updates_total = 0
    num_done_for_backup_true = 0
    num_timeout_truncations = 0
    duration_sum_total = 0.0
    offline_phase_state = {
        "phase": 0,
        "phase0_policy_steps": 0,
        "phase1_critic_updates": 0,
        "phase2_actor_updates": 0,
        "pre_warmstart_eval_done": False,
        "complete": False,
    }

    if args.resume:
        latest_path = os.path.join(ckpt_dir, args.resume_ckpt_name)
        if not os.path.exists(latest_path):
            raise FileNotFoundError(f"resume checkpoint not found: {latest_path}")
        state = torch.load(latest_path, map_location="cpu")
        sac.load_state_dict(state["sac_state"])
        policy_steps = int(state.get("policy_steps", 0))
        env_steps = int(state.get("env_steps", 0))
        best_success = float(state.get("best_success", -1.0))
        dynamic_floor_release_latched = bool(
            state.get("dynamic_floor_release_latched", False)
        )
        dynamic_floor_release_policy_step = state.get(
            "dynamic_floor_release_policy_step"
        )
        dynamic_floor_release_env_step = state.get("dynamic_floor_release_env_step")
        replay_state = state.get("replay_state")
        replay_was_restored = replay_state is not None
        if replay_state is not None:
            replay.load_state_dict(replay_state)
        loaded_offline_phase_state = dict(state.get("offline_phase_state", {}))
        offline_phase_state.update(loaded_offline_phase_state)
        step_calls = int(state.get("step_calls", 0))
        pending_transitions_for_update = int(
            state.get("pending_transitions_for_update", 0)
        )
        num_emitted_transitions_total = int(
            state.get("num_emitted_transitions_total", 0)
        )
        num_updates_total = int(state.get("num_updates_total", 0))
        num_done_for_backup_true = int(state.get("num_done_for_backup_true", 0))
        num_timeout_truncations = int(state.get("num_timeout_truncations", 0))
        duration_sum_total = float(state.get("duration_sum_total", 0.0))
        replay_state = None
        state = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
            if _history_has_env_step_eval_at_zero(history):
                offline_phase_state["pre_warmstart_eval_done"] = True
        resume_msg = f"[ExRL][RESUME] loaded metrics from {loaded_from}"
        print(resume_msg, flush=True)
        _append_status_line(status_log_path, resume_msg)
        if replay_was_restored:
            resume_msg = f"[ExRL][RESUME] replay restored size={len(replay)}"
        else:
            resume_msg = (
                "[ExRL][RESUME] model resumed; replay starts empty by design, "
                "warmup will refill before updates."
            )
            if str(args.pipeline) == "sac_warmstart":
                resume_msg = "[ExRL][RESUME][WARN] offline checkpoint had no replay_state; restarting offline replay collection"
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
    perf_profile = bool(args.perf_profile)
    perf_totals = defaultdict(float)
    sac_cuda_graph_updaters = {}

    def dump_metrics():
        _atomic_json_dump(history, metrics_path, metrics_backup)

    def pack_ckpt() -> Dict:
        return {
            "sac_state": sac.state_dict(),
            "policy_steps": int(policy_steps),
            "env_steps": int(env_steps),
            "best_success": float(best_success),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dims": hidden_dims,
            "target_entropy": target_entropy,
            "network_activation": args.network_activation,
            "actor_use_layernorm": bool(args.actor_use_layernorm),
            "critic_use_layernorm": bool(args.critic_use_layernorm),
            "critic_backup_temp": float(args.critic_backup_temp),
            "cql_alpha": float(args.cql_alpha),
            "cql_temp": float(args.cql_temp),
            "critic_ensemble_size": int(args.critic_ensemble_size),
            "target_critic_subsample_size": int(args.target_critic_subsample_size),
            "critic_bootstrap_mask_prob": float(args.critic_bootstrap_mask_prob),
            "actor_target_critic_subsample": bool(args.actor_target_critic_subsample),
            "actor_loss_critic_subsample": bool(args.actor_loss_critic_subsample),
            "selective_critic_forward": bool(args.selective_critic_forward),
            "foreach_target_update": bool(args.foreach_target_update),
            "prefetch_utd_batches": bool(args.prefetch_utd_batches),
            "async_batch_validation": bool(args.async_batch_validation),
            "defer_utd_metric_sync": bool(args.defer_utd_metric_sync),
            "sparse_utd_diagnostics": bool(args.sparse_utd_diagnostics),
            "sac_cuda_graph": bool(args.sac_cuda_graph),
            "fused_optimizer": bool(args.fused_optimizer),
            "effective_foreach_target_update": effective_foreach_target_update,
            "effective_async_batch_validation": effective_async_batch_validation,
            "max_chunk_len": int(args.max_chunk_len),
            "min_chunk_len": int(args.min_chunk_len),
            "dynamic_floor_release_latched": bool(dynamic_floor_release_latched),
            "dynamic_floor_release_policy_step": dynamic_floor_release_policy_step,
            "dynamic_floor_release_env_step": dynamic_floor_release_env_step,
            "replay_state": (
                replay.state_dict(copy_arrays=False, as_tensors=True)
                if str(args.pipeline) == "sac_warmstart"
                else None
            ),
            "offline_phase_state": dict(offline_phase_state),
            "step_calls": int(step_calls),
            "pending_transitions_for_update": int(pending_transitions_for_update),
            "num_emitted_transitions_total": int(num_emitted_transitions_total),
            "num_updates_total": int(num_updates_total),
            "num_done_for_backup_true": int(num_done_for_backup_true),
            "num_timeout_truncations": int(num_timeout_truncations),
            "duration_sum_total": float(duration_sum_total),
            "warmstart_actor": bool(args.warmstart_actor),
            "run_dir": run_dir,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    eval_snapshot_interval = int(args.eval_snapshot_env_freq)
    next_eval_snapshot_env_step: Optional[int] = None
    if eval_snapshot_interval > 0:
        next_eval_snapshot_env_step = _next_interval_boundary(
            env_steps, eval_snapshot_interval
        )

    def pack_eval_actor_snapshot(snapshot_env_step: int) -> Dict:
        actor_state = {
            key: value.detach().cpu().clone()
            for key, value in sac.actor.state_dict().items()
        }
        return {
            "snapshot_format": "exrl_sac_actor_eval_v1",
            "actor_state": actor_state,
            "snapshot_env_step": int(snapshot_env_step),
            "env_steps": int(env_steps),
            "policy_steps": int(policy_steps),
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "action_shift": int(action_shift),
            "hidden_dims": list(hidden_dims),
            "network_activation": str(args.network_activation),
            "actor_use_layernorm": bool(args.actor_use_layernorm),
            "max_chunk_len": int(args.max_chunk_len),
            "min_chunk_len": int(args.min_chunk_len),
            "run_dir": str(run_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def save_eval_actor_snapshot(snapshot_env_step: int):
        if eval_snapshot_interval <= 0:
            return
        path = os.path.join(
            ckpt_dir,
            f"eval_snapshot_env_{int(snapshot_env_step):010d}_steps.pt",
        )
        if os.path.exists(path):
            return
        _save_ckpt(path, pack_eval_actor_snapshot(snapshot_env_step))
        line = (
            f"[ExRL][EXTERNAL-EVAL-SNAPSHOT] boundary_env_steps={int(snapshot_env_step)} "
            f"actual_env_steps={int(env_steps)} path={path}"
        )
        print(line, flush=True)
        _append_status_line(status_log_path, line)

    def maybe_save_due_eval_actor_snapshots():
        nonlocal next_eval_snapshot_env_step
        if eval_snapshot_interval <= 0 or next_eval_snapshot_env_step is None:
            return
        while int(env_steps) >= int(next_eval_snapshot_env_step):
            save_eval_actor_snapshot(int(next_eval_snapshot_env_step))
            next_eval_snapshot_env_step += eval_snapshot_interval

    # Resume strictly after the restored counter instead of replaying every old boundary.
    next_periodic_ckpt = _next_interval_boundary(policy_steps, args.save_freq)
    next_latest_ckpt = _next_interval_boundary(
        policy_steps, args.latest_model_save_freq
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
    online_eval_venv = None
    online_eval_interval = int(args.online_eval_every_env_steps)
    next_online_eval_env_step: Optional[int] = None

    if online_eval_interval > 0:
        eval_max_episode_steps = int(
            env_meta.get("env_kwargs", {}).get("horizon", args.env_horizon_override)
        )
        online_eval_venv = _make_online_eval_venv(
            args=args,
            policy=policy,
            device=device,
            lowdim_keys=lowdim_keys,
            lowdim_dims=lowdim_dims,
            max_episode_steps=eval_max_episode_steps,
        )
        history.setdefault("env_step_eval", [])
        next_online_eval_env_step = (
            0
            if int(env_steps) == 0
            else ((int(env_steps) // online_eval_interval + 1) * online_eval_interval)
        )

    def maybe_run_online_eval(force: bool = False):
        nonlocal next_online_eval_env_step
        if online_eval_venv is None or online_eval_interval <= 0:
            return None
        if (
            (not force)
            and next_online_eval_env_step is not None
            and int(env_steps) < int(next_online_eval_env_step)
        ):
            return None
        eval_t0 = time.time()
        success_rate = _run_online_greedy_eval(
            eval_venv=online_eval_venv,
            sac=sac,
            action_shift=int(action_shift),
            episodes=int(args.online_eval_episodes),
            max_decisions_per_episode=int(args.online_eval_max_decisions_per_episode),
        )
        history.setdefault("env_step_eval", []).append(int(env_steps))
        _record(history, "env_step_eval/success_rate", float(success_rate))
        writer.add_scalar(
            "env_step_eval/success_rate", float(success_rate), int(env_steps)
        )
        try:
            writer.flush()
        except Exception:
            pass
        dump_metrics()
        eval_line = (
            f"[ExRL][ONLINE-EVAL] env_steps={env_steps} "
            f"episodes={int(args.online_eval_episodes)} "
            f"success_rate={float(success_rate):.4f} "
            f"elapsed_sec={time.time() - eval_t0:.1f}"
        )
        print(eval_line, flush=True)
        _append_status_line(status_log_path, eval_line)
        if next_online_eval_env_step is None:
            next_online_eval_env_step = 0
        while int(next_online_eval_env_step) <= int(env_steps):
            next_online_eval_env_step += online_eval_interval
        return float(success_rate)

    if eval_snapshot_interval > 0 and not args.resume:
        # Preserve the exact initial actor for deterministic and stochastic evaluation.
        save_eval_actor_snapshot(0)

    if bool(args.resume_eval_before_training):
        if not args.resume:
            raise ValueError("--resume_eval_before_training requires --resume")
        resume_eval_success = maybe_run_online_eval(force=True)
        if resume_eval_success is None:
            raise RuntimeError(
                "resume pre-training evaluation was requested but online evaluation is disabled"
            )
        gate_line = (
            "[ExRL][RESUME-EVAL-GATE] "
            f"env_steps={int(env_steps)} success_rate={float(resume_eval_success):.4f} "
            f"min_success={float(args.resume_eval_min_success):.4f}"
        )
        print(gate_line, flush=True)
        _append_status_line(status_log_path, gate_line)
        if float(args.resume_eval_min_success) >= 0.0 and float(
            resume_eval_success
        ) < float(args.resume_eval_min_success):
            raise RuntimeError(
                "resume pre-training evaluation failed: "
                f"success_rate={float(resume_eval_success):.4f} < "
                f"min_success={float(args.resume_eval_min_success):.4f}"
            )

    print("[ExRL] start training loop", flush=True)
    _append_status_line(status_log_path, "[ExRL] start training loop")
    try:
        offline_warmstart_complete = str(args.pipeline) == "sac_warmstart" and bool(
            offline_phase_state.get("complete", False)
        )
        if str(args.pipeline) == "online_sac":
            if online_eval_venv is not None and int(env_steps) == 0:
                maybe_run_online_eval(force=True)
        else:
            offline_phase0_target = (
                int(args.warmup_transitions)
                if int(args.warmup_transitions) >= 0
                else int(args.warmup_policy_steps)
            )
            offline_phase1_target = int(args.initial_critic_updates)
            offline_phase2_target = int(args.initial_actor_updates)
            offline_phase0_done = int(offline_phase_state.get("phase0_policy_steps", 0))
            offline_phase1_done = int(
                offline_phase_state.get("phase1_critic_updates", 0)
            )
            offline_phase2_done = int(
                offline_phase_state.get("phase2_actor_updates", 0)
            )
            offline_eval_interval = int(args.offline_eval_every_updates)
            phase1_actor_critic = bool(args.warmstart_actor)
            phase1_target_policy_mode = "actor" if phase1_actor_critic else "q_greedy"
            phase1_update_actor = bool(phase1_actor_critic)
            phase1_update_alpha = bool(phase1_actor_critic)

            if (
                not offline_warmstart_complete
                and online_eval_venv is not None
                and not bool(offline_phase_state.get("pre_warmstart_eval_done", False))
                and int(env_steps) == 0
                and int(offline_phase0_done) == 0
                and int(offline_phase1_done) == 0
                and int(offline_phase2_done) == 0
            ):
                line = "[ExRL][OFFLINE-PRE-EVAL] running actor eval before phase0 warmstart at env_steps=0"
                print(line, flush=True)
                _append_status_line(status_log_path, line)
                maybe_run_online_eval(force=True)
                offline_phase_state["pre_warmstart_eval_done"] = True
                _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())

            def log_offline_scalars(
                phase_id: int,
                critic_updates: int,
                actor_updates: int,
                success_rate: Optional[float] = None,
            ):
                history.setdefault("offline_steps", []).append(int(env_steps))
                _record(history, "offline/phase", float(phase_id))
                _record(history, "offline/critic_updates", float(critic_updates))
                _record(history, "offline/actor_updates", float(actor_updates))
                _record(
                    history, "offline/phase0_policy_steps", float(offline_phase0_done)
                )
                _record(
                    history, "offline/phase1_actor_critic", float(phase1_actor_critic)
                )
                _record(history, "replay/size", float(len(replay)))
                writer.add_scalar("offline/phase", float(phase_id), int(env_steps))
                writer.add_scalar(
                    "offline/critic_updates", float(critic_updates), int(env_steps)
                )
                writer.add_scalar(
                    "offline/actor_updates", float(actor_updates), int(env_steps)
                )
                writer.add_scalar(
                    "offline/phase0_policy_steps",
                    float(offline_phase0_done),
                    int(env_steps),
                )
                writer.add_scalar(
                    "offline/phase1_actor_critic",
                    float(phase1_actor_critic),
                    int(env_steps),
                )
                writer.add_scalar("replay/size", float(len(replay)), int(env_steps))
                writer.add_scalar("diagnostic/min_k_floor", 0.0, int(env_steps))
                writer.add_scalar(
                    "diagnostic/actor_prior_coeff_recent", 0.0, int(env_steps)
                )
                if success_rate is not None:
                    _record(history, "offline/eval_success_rate", float(success_rate))
                    _record(history, "env_step_eval/success_rate", float(success_rate))
                    writer.add_scalar(
                        "offline/eval_success_rate", float(success_rate), int(env_steps)
                    )
                    writer.add_scalar(
                        "env_step_eval/success_rate",
                        float(success_rate),
                        int(env_steps),
                    )
                dump_metrics()
                try:
                    writer.flush()
                except Exception:
                    pass

            if (
                not offline_warmstart_complete
                and offline_phase0_done < offline_phase0_target
            ):
                offline_phase_state["phase"] = 0
                line = (
                    f"[ExRL][OFFLINE-PHASE0] collect uniform random chunks "
                    f"{offline_phase0_done}/{offline_phase0_target}"
                )
                print(line, flush=True)
                _append_status_line(status_log_path, line)
                phase0_next_log = offline_phase0_done + max(
                    int(args.log_every_policy_steps), 1
                )
                while offline_phase0_done < offline_phase0_target:
                    actions = np.random.randint(
                        0, action_dim, size=args.n_envs, dtype=np.int64
                    )
                    env_actions = actions + action_shift
                    next_obs, rewards, dones, infos = venv.step(env_actions)
                    next_obs = np.asarray(next_obs, dtype=np.float32)
                    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
                    dones = np.asarray(dones, dtype=bool).reshape(-1)
                    if next_obs.shape != obs.shape:
                        raise RuntimeError(
                            f"obs shape changed unexpectedly: {obs.shape} -> {next_obs.shape}"
                        )

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
                        )
                        k = float(ingest_result.duration_k)
                        num_emitted_transitions_total += int(
                            ingest_result.emitted_transitions
                        )
                        num_done_for_backup_true += int(
                            bool(ingest_result.done_for_backup)
                        )
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
                    offline_phase0_done += int(args.n_envs)
                    offline_phase_state["phase0_policy_steps"] = int(
                        offline_phase0_done
                    )
                    offline_phase_state["phase"] = 0
                    if (
                        offline_phase0_done >= phase0_next_log
                        or offline_phase0_done >= offline_phase0_target
                    ):
                        log_offline_scalars(
                            phase_id=0,
                            critic_updates=offline_phase1_done,
                            actor_updates=offline_phase2_done,
                        )
                        _save_ckpt(
                            os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt()
                        )
                        phase0_next_log += max(int(args.log_every_policy_steps), 1)

            if (
                not offline_warmstart_complete
                and offline_phase0_done >= offline_phase0_target
            ):
                flushed = 0
                for q in nstep_queues:
                    flushed += drain_smdp_nstep_queue_to_replay(
                        q,
                        replay=replay,
                        gamma=float(args.gamma),
                        n_step=int(args.n_step),
                        action_dim=action_dim,
                        flush_all=True,
                    )
                if flushed > 0:
                    num_emitted_transitions_total += int(flushed)
                    pending_transitions_for_update += int(flushed)
                nstep_queues = [deque() for _ in range(args.n_envs)]
                if len(replay) < int(args.batch_size):
                    raise RuntimeError(
                        f"offline phase0 replay too small for batch_size: replay={len(replay)} batch={args.batch_size}"
                    )

            if (
                not offline_warmstart_complete
                and offline_phase1_done < offline_phase1_target
            ):
                offline_phase_state["phase"] = 1
                phase1_label = (
                    "actor-critic actor target"
                    if phase1_actor_critic
                    else "critic-only q_greedy target"
                )
                line = (
                    f"[ExRL][OFFLINE-PHASE1] {phase1_label} updates "
                    f"{offline_phase1_done}/{offline_phase1_target} "
                    f"update_actor={phase1_update_actor} "
                    f"update_alpha={phase1_update_alpha} "
                    f"target_policy_mode={phase1_target_policy_mode} "
                    f"cql_alpha={float(args.cql_alpha)}"
                )
                print(line, flush=True)
                _append_status_line(status_log_path, line)
                phase1_next_log = offline_phase1_done + max(
                    int(args.log_every_policy_steps), 1
                )
                while offline_phase1_done < offline_phase1_target:
                    m = sac.update(
                        replay.sample(int(args.batch_size)),
                        update_actor=phase1_update_actor,
                        update_alpha=phase1_update_alpha,
                        transition_weight_mode=str(args.transition_weight_mode),
                        actor_prior_coeff=0.0,
                        actor_prior_min_action_id=0,
                        legal_min_action_id=0,
                        target_policy_mode=phase1_target_policy_mode,
                    )
                    for k, v in m.items():
                        train_meter[k].append(float(v))
                        latest_train_metrics[k] = float(v)
                    offline_phase1_done += 1
                    num_updates_total += 1
                    offline_phase_state["phase1_critic_updates"] = int(
                        offline_phase1_done
                    )

                    should_eval = (
                        offline_eval_interval > 0
                        and (offline_phase1_done % offline_eval_interval) == 0
                    )
                    should_log = (
                        offline_phase1_done >= phase1_next_log
                        or offline_phase1_done >= offline_phase1_target
                    )
                    if should_eval or should_log:
                        for k, vals in list(train_meter.items()):
                            if len(vals) > 0:
                                mean_val = float(np.mean(vals))
                                _record(history, k, mean_val)
                                writer.add_scalar(k, mean_val, int(env_steps))
                            train_meter[k].clear()
                        success_rate = None
                        if should_eval or offline_phase1_done >= offline_phase1_target:
                            if online_eval_venv is not None:
                                eval_t0 = time.time()
                                if phase1_actor_critic:
                                    success_rate = _run_online_greedy_eval(
                                        eval_venv=online_eval_venv,
                                        sac=sac,
                                        action_shift=int(action_shift),
                                        episodes=int(args.eval_episodes),
                                        max_decisions_per_episode=int(
                                            args.offline_eval_max_decisions
                                        ),
                                    )
                                    eval_policy_label = "actor_deterministic"
                                else:
                                    success_rate = _run_critic_policy_eval(
                                        eval_venv=online_eval_venv,
                                        sac=sac,
                                        action_shift=int(action_shift),
                                        episodes=int(args.eval_episodes),
                                        max_decisions_per_episode=int(
                                            args.offline_eval_max_decisions
                                        ),
                                        policy_mode=str(args.offline_eval_policy_mode),
                                        critic_reduce=str(
                                            args.offline_eval_critic_reduce
                                        ),
                                    )
                                    eval_policy_label = str(
                                        args.offline_eval_policy_mode
                                    )
                                eval_line = (
                                    f"[ExRL][OFFLINE-EVAL] env_steps={env_steps} "
                                    f"critic_updates={offline_phase1_done} "
                                    f"policy={eval_policy_label} "
                                    f"success_rate={float(success_rate):.4f} "
                                    f"elapsed_sec={time.time() - eval_t0:.1f}"
                                )
                                print(eval_line, flush=True)
                                _append_status_line(status_log_path, eval_line)
                        log_offline_scalars(
                            phase_id=1,
                            critic_updates=offline_phase1_done,
                            actor_updates=offline_phase2_done,
                            success_rate=success_rate,
                        )
                        _save_ckpt(
                            os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt()
                        )
                        if should_log:
                            line = (
                                f"[ExRL][OFFLINE-PHASE1] critic_updates={offline_phase1_done}/{offline_phase1_target} "
                                f"replay={len(replay)}"
                            )
                            print(line, flush=True)
                            _append_status_line(status_log_path, line)
                            phase1_next_log += max(int(args.log_every_policy_steps), 1)

            if (
                not offline_warmstart_complete
                and offline_phase1_done >= offline_phase1_target
                and offline_phase2_done < offline_phase2_target
            ):
                offline_phase_state["phase"] = 2
                line = (
                    f"[ExRL][OFFLINE-PHASE2] actor-only updates "
                    f"{offline_phase2_done}/{offline_phase2_target} "
                    f"update_alpha={bool(args.warmstart_temperature)}"
                )
                print(line, flush=True)
                _append_status_line(status_log_path, line)
                phase2_next_log = offline_phase2_done + max(
                    int(args.log_every_policy_steps), 1
                )
                while offline_phase2_done < offline_phase2_target:
                    m = sac.update(
                        replay.sample(int(args.batch_size)),
                        update_actor=True,
                        update_alpha=bool(args.warmstart_temperature),
                        update_critic=False,
                        transition_weight_mode=str(args.transition_weight_mode),
                        actor_prior_coeff=0.0,
                        actor_prior_min_action_id=0,
                        legal_min_action_id=0,
                        target_policy_mode="actor",
                    )
                    for k, v in m.items():
                        train_meter[k].append(float(v))
                        latest_train_metrics[k] = float(v)
                    offline_phase2_done += 1
                    num_updates_total += 1
                    offline_phase_state["phase2_actor_updates"] = int(
                        offline_phase2_done
                    )

                    should_eval = (
                        offline_eval_interval > 0
                        and (offline_phase2_done % offline_eval_interval) == 0
                    )
                    should_log = (
                        offline_phase2_done >= phase2_next_log
                        or offline_phase2_done >= offline_phase2_target
                    )
                    if should_eval or should_log:
                        for k, vals in list(train_meter.items()):
                            if len(vals) > 0:
                                mean_val = float(np.mean(vals))
                                _record(history, k, mean_val)
                                writer.add_scalar(k, mean_val, int(env_steps))
                            train_meter[k].clear()
                        success_rate = None
                        if should_eval or offline_phase2_done >= offline_phase2_target:
                            if online_eval_venv is not None:
                                eval_t0 = time.time()
                                success_rate = _run_online_greedy_eval(
                                    eval_venv=online_eval_venv,
                                    sac=sac,
                                    action_shift=int(action_shift),
                                    episodes=int(args.eval_episodes),
                                    max_decisions_per_episode=int(
                                        args.offline_eval_max_decisions
                                    ),
                                )
                                eval_line = (
                                    f"[ExRL][OFFLINE-ACTOR-EVAL] env_steps={env_steps} "
                                    f"actor_updates={offline_phase2_done} "
                                    f"success_rate={float(success_rate):.4f} "
                                    f"elapsed_sec={time.time() - eval_t0:.1f}"
                                )
                                print(eval_line, flush=True)
                                _append_status_line(status_log_path, eval_line)
                        log_offline_scalars(
                            phase_id=2,
                            critic_updates=offline_phase1_done,
                            actor_updates=offline_phase2_done,
                            success_rate=success_rate,
                        )
                        _save_ckpt(
                            os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt()
                        )
                        if should_log:
                            line = (
                                f"[ExRL][OFFLINE-PHASE2] actor_updates={offline_phase2_done}/{offline_phase2_target} "
                                f"replay={len(replay)}"
                            )
                            print(line, flush=True)
                            _append_status_line(status_log_path, line)
                            phase2_next_log += max(int(args.log_every_policy_steps), 1)

            if (
                not offline_warmstart_complete
                and offline_phase1_done >= offline_phase1_target
                and offline_phase2_done >= offline_phase2_target
            ):
                offline_phase_state["phase"] = 3 if offline_phase2_target > 0 else 2
                offline_phase_state["complete"] = True
                _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
                obs = np.asarray(venv.reset(), dtype=np.float32)
                nstep_queues = [deque() for _ in range(args.n_envs)]
                pending_transitions_for_update = 0
                episode_returns[:] = 0.0
                episode_lengths[:] = 0
                learning_starts = min(int(learning_starts), int(policy_steps))
                actor_learning_starts = min(
                    int(actor_learning_starts), int(policy_steps)
                )
                alpha_learning_starts = min(
                    int(alpha_learning_starts), int(policy_steps)
                )
                next_log = int(policy_steps) + max(int(args.log_every_policy_steps), 1)
                offline_warmstart_complete = True
                line = (
                    f"[ExRL][OFFLINE-HANDOFF] env reset; entering online SAC at "
                    f"policy_steps={policy_steps} env_steps={env_steps} replay={len(replay)} "
                    f"critic_updates={offline_phase1_done} actor_updates={offline_phase2_done}"
                )
                print(line, flush=True)
                _append_status_line(status_log_path, line)
                if online_eval_venv is not None:
                    maybe_run_online_eval(force=True)

            if offline_warmstart_complete:
                learning_starts = min(int(learning_starts), int(policy_steps))
                actor_learning_starts = min(
                    int(actor_learning_starts), int(policy_steps)
                )
                alpha_learning_starts = min(
                    int(alpha_learning_starts), int(policy_steps)
                )

        perf_last_totals = dict(perf_totals)
        perf_last_step_calls = int(step_calls)
        perf_last_updates = int(num_updates_total)
        wrapper_perf = _find_wrapper_perf_acc(venv) if perf_profile else None
        perf_last_wrapper = dict(wrapper_perf) if wrapper_perf is not None else {}

        while policy_steps < int(args.total_policy_steps) and (
            int(args.stop_after_env_steps) <= 0
            or int(env_steps) < int(args.stop_after_env_steps)
        ):
            perf_t0 = time.perf_counter() if perf_profile else 0.0
            model_probe_k = None
            current_min_k_floor = 1
            current_min_action_id = max(
                int(current_min_k_floor) - int(args.min_chunk_len), 0
            )
            dynamic_floor_signal = None
            in_online_warmup = not bool(
                offline_warmstart_complete
            ) and policy_steps < int(args.warmup_policy_steps)
            if in_online_warmup and bool(args.probe_model_k_during_warmup):
                probe_actions = sac.select_action(
                    obs, deterministic=bool(args.warmup_probe_deterministic)
                )
                model_probe_k = float(
                    np.mean(
                        probe_actions.astype(np.float32) + float(args.min_chunk_len)
                    )
                )
            if in_online_warmup:
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
            if perf_profile:
                perf_totals["select_sec"] += time.perf_counter() - perf_t0
            env_actions = actions + action_shift

            perf_t0 = time.perf_counter() if perf_profile else 0.0
            next_obs, rewards, dones, infos = venv.step(env_actions)
            if perf_profile:
                perf_totals["venv_step_sec"] += time.perf_counter() - perf_t0
            next_obs = np.asarray(next_obs, dtype=np.float32)
            rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
            dones = np.asarray(dones, dtype=bool).reshape(-1)
            if next_obs.shape != obs.shape:
                raise RuntimeError(
                    f"obs shape changed unexpectedly: {obs.shape} -> {next_obs.shape}"
                )

            perf_t0 = time.perf_counter() if perf_profile else 0.0
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

            if perf_profile:
                perf_totals["ingest_sec"] += time.perf_counter() - perf_t0
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
                if bool(args.prefetch_utd_batches):
                    update_batches = replay.sample_many(
                        int(args.batch_size), int(current_utd)
                    )
                else:
                    update_batches = (
                        replay.sample(int(args.batch_size))
                        for _ in range(int(current_utd))
                    )
                deferred_metrics = []
                for update_index, update_batch in enumerate(update_batches):
                    perf_t0 = time.perf_counter() if perf_profile else 0.0
                    legal_min_action_id = (
                        int(current_min_action_id)
                        if floor_support_mask_mode in ("update_only", "all")
                        else 0
                    )
                    if bool(args.sac_cuda_graph):
                        graph_key = (
                            bool(update_actor),
                            bool(update_alpha),
                            str(args.transition_weight_mode),
                            float(actor_prior_coeff),
                            int(current_min_action_id),
                            int(legal_min_action_id),
                        )
                        graph_updater = sac_cuda_graph_updaters.get(graph_key)
                        if graph_updater is None:
                            graph_updater = SMDPCudaGraphUpdate(
                                sac,
                                update_batch,
                                update_actor=bool(update_actor),
                                update_alpha=bool(update_alpha),
                                transition_weight_mode=str(args.transition_weight_mode),
                                actor_prior_coeff=float(actor_prior_coeff),
                                actor_prior_min_action_id=int(current_min_action_id),
                                legal_min_action_id=int(legal_min_action_id),
                                target_policy_mode="actor",
                                compute_diagnostics=True,
                            )
                            sac_cuda_graph_updaters[graph_key] = graph_updater
                            graph_line = (
                                "[ExRL][CUDA-GRAPH] captured full update "
                                f"key={graph_key} capture_sec={graph_updater.capture_sec:.3f}"
                            )
                            print(graph_line, flush=True)
                            _append_status_line(status_log_path, graph_line)
                        m = graph_updater.replay(update_batch)
                    else:
                        m = sac.update(
                            update_batch,
                            update_actor=bool(update_actor),
                            update_alpha=bool(update_alpha),
                            transition_weight_mode=str(args.transition_weight_mode),
                            actor_prior_coeff=float(actor_prior_coeff),
                            actor_prior_min_action_id=int(current_min_action_id),
                            legal_min_action_id=int(legal_min_action_id),
                            target_policy_mode="actor",
                            materialize_metrics=not bool(args.defer_utd_metric_sync),
                            compute_diagnostics=(
                                not bool(args.sparse_utd_diagnostics)
                                or int(update_index) == int(current_utd) - 1
                            ),
                        )
                    if perf_profile:
                        perf_totals["update_sec"] += time.perf_counter() - perf_t0
                    deferred_metrics.append(m)
                if bool(args.sac_cuda_graph):
                    perf_t0 = time.perf_counter() if perf_profile else 0.0
                    deferred_metrics = [sac._materialize_metrics(deferred_metrics[-1])]
                    if perf_profile:
                        perf_totals["update_sec"] += time.perf_counter() - perf_t0
                elif bool(args.defer_utd_metric_sync):
                    perf_t0 = time.perf_counter() if perf_profile else 0.0
                    deferred_metrics = sac.materialize_metric_sequence(deferred_metrics)
                    if perf_profile:
                        perf_totals["update_sec"] += time.perf_counter() - perf_t0
                for m in deferred_metrics:
                    for k, v in m.items():
                        train_meter[k].append(float(v))
                        latest_train_metrics[k] = float(v)
                num_updates_total += int(current_utd)

            maybe_run_online_eval()
            maybe_save_due_eval_actor_snapshots()

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

                perf_metrics: Dict[str, float] = {}
                if perf_profile:
                    decision_delta = max(int(step_calls) - perf_last_step_calls, 1)
                    update_delta = max(int(num_updates_total) - perf_last_updates, 1)
                    for source_key, metric_key, denom in (
                        ("select_sec", "perf/select_sec_per_decision", decision_delta),
                        (
                            "venv_step_sec",
                            "perf/venv_step_sec_per_decision",
                            decision_delta,
                        ),
                        ("ingest_sec", "perf/ingest_sec_per_decision", decision_delta),
                        ("update_sec", "perf/update_sec_per_update", update_delta),
                    ):
                        delta = float(perf_totals[source_key]) - float(
                            perf_last_totals.get(source_key, 0.0)
                        )
                        perf_metrics[metric_key] = delta / float(denom)

                    wrapper_perf = _find_wrapper_perf_acc(venv)
                    if wrapper_perf is not None:
                        dp_call_delta = max(
                            float(wrapper_perf.get("dp_calls", 0.0))
                            - float(perf_last_wrapper.get("dp_calls", 0.0)),
                            1.0,
                        )
                        wait_call_delta = max(
                            float(wrapper_perf.get("step_wait_calls", 0.0))
                            - float(perf_last_wrapper.get("step_wait_calls", 0.0)),
                            1.0,
                        )
                        for source_key, metric_key, denom in (
                            (
                                "dp_infer_sec",
                                "perf/dp_infer_sec_per_call",
                                dp_call_delta,
                            ),
                            (
                                "env_wait_sec",
                                "perf/env_wait_sec_per_decision",
                                wait_call_delta,
                            ),
                            (
                                "step_wait_sec",
                                "perf/step_wait_sec_per_decision",
                                wait_call_delta,
                            ),
                        ):
                            delta = float(wrapper_perf.get(source_key, 0.0)) - float(
                                perf_last_wrapper.get(source_key, 0.0)
                            )
                            perf_metrics[metric_key] = delta / float(denom)

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
                for key, value in perf_metrics.items():
                    _record(history, key, value)
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
                for key, value in perf_metrics.items():
                    writer.add_scalar(key, value, env_steps)
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
                if perf_metrics:
                    line += (
                        f" env_s={perf_metrics.get('perf/env_wait_sec_per_decision', 0.0):.4f}"
                        f" dp_s={perf_metrics.get('perf/dp_infer_sec_per_call', 0.0):.4f}"
                        f" upd_s={perf_metrics.get('perf/update_sec_per_update', 0.0):.4f}"
                    )
                print(line, flush=True)
                _append_status_line(status_log_path, line)
                try:
                    writer.flush()
                except Exception:
                    pass
                if perf_profile:
                    perf_last_totals = dict(perf_totals)
                    perf_last_step_calls = int(step_calls)
                    perf_last_updates = int(num_updates_total)
                    wrapper_perf = _find_wrapper_perf_acc(venv)
                    perf_last_wrapper = (
                        dict(wrapper_perf) if wrapper_perf is not None else {}
                    )
                next_log += max(int(args.log_every_policy_steps), 1)

                if success_recent > best_success and len(success_buffer) > 0:
                    best_success = success_recent
                    _save_ckpt(os.path.join(ckpt_dir, "best_model.pt"), pack_ckpt())

            if policy_steps >= next_periodic_ckpt:
                _save_ckpt(
                    os.path.join(ckpt_dir, f"adaptive_agent_{policy_steps}_steps.pt"),
                    pack_ckpt(),
                )
                next_periodic_ckpt += max(int(args.save_freq), 1)

            if policy_steps >= next_latest_ckpt:
                _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
                next_latest_ckpt += max(int(args.latest_model_save_freq), 1)

        if int(args.stop_after_env_steps) > 0 and int(env_steps) >= int(
            args.stop_after_env_steps
        ):
            line = (
                f"[ExRL] reached stop_after_env_steps={int(args.stop_after_env_steps)} "
                f"env_steps={int(env_steps)}"
            )
            print(line, flush=True)
            _append_status_line(status_log_path, line)

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
            venv.close()
        except Exception:
            pass
        if online_eval_venv is not None:
            try:
                online_eval_venv.close()
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
