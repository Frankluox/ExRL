"""Train ExRL with an ACT base policy and SAC updates."""

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
from typing import Any, Dict, List, Optional, Sequence

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
from exrl.runners.act_sac.environment import ExecutionLengthEnv
from exrl.runners.act_sac.policy_loader import (
    infer_lowdim_dims_from_stats,
    load_act_policy_wrapper,
    load_env_meta_json,
    parse_lowdim_keys,
    resolve_act_task_setup,
)
from exrl.algorithms.act_sac import (
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


def _next_checkpoint_step(current_step: int, interval: int) -> int:
    interval = max(int(interval), 1)
    current_step = max(int(current_step), 0)
    return ((current_step // interval) + 1) * interval


def _copy_deque_values(values) -> List[Any]:
    return list(values)


def _restore_deque_values(target: deque, values) -> None:
    target.clear()
    if values is None:
        return
    target.extend(
        list(values)[-target.maxlen :] if target.maxlen is not None else list(values)
    )


def _pack_replay_state(replay: SMDPReplayBuffer) -> Dict[str, Any]:
    # Torch writes tensor storages outside data.pkl. Keeping replay as zero-copy
    # CPU tensor views avoids both a replay-sized RAM spike and pickle's 4 GiB
    # bytes-object limit for large Transport checkpoints.
    return replay.state_dict(copy_arrays=False, as_torch=True)


def _restore_replay_state(replay: SMDPReplayBuffer, state: Dict[str, Any]) -> None:
    if not isinstance(state, dict):
        raise ValueError("invalid replay state")
    obs_dim = int(state.get("obs_dim", -1))
    capacity = int(state.get("capacity", -1))
    size = int(state.get("size", -1))
    pos = int(state.get("pos", -1))
    if obs_dim != int(replay.obs_dim):
        raise ValueError(
            f"replay obs_dim mismatch: ckpt={obs_dim} current={int(replay.obs_dim)}"
        )
    if capacity != int(replay.capacity):
        raise ValueError(
            f"replay capacity mismatch: ckpt={capacity} current={int(replay.capacity)}"
        )
    if size < 0 or size > int(replay.capacity):
        raise ValueError(f"invalid replay size in checkpoint: {size}")
    if pos < 0 or pos >= int(replay.capacity):
        raise ValueError(f"invalid replay pos in checkpoint: {pos}")

    replay.pos = pos
    replay.size = size
    if size == 0:
        return
    replay.obs[:size] = np.asarray(state["obs"], dtype=np.float32).reshape(
        size, replay.obs_dim
    )
    replay.actions[:size] = np.asarray(state["actions"], dtype=np.int64).reshape(size)
    replay.rewards[:size] = np.asarray(state["rewards"], dtype=np.float32).reshape(size)
    replay.durations[:size] = np.asarray(state["durations"], dtype=np.float32).reshape(
        size
    )
    replay.next_obs[:size] = np.asarray(state["next_obs"], dtype=np.float32).reshape(
        size, replay.obs_dim
    )
    replay.dones[:size] = np.asarray(state["dones"], dtype=np.float32).reshape(size)


def _pending_transition_to_state(tr: PendingSMDPTransition) -> Dict[str, Any]:
    return {
        "obs": np.asarray(tr.obs, dtype=np.float32).reshape(-1).copy(),
        "action_id": int(tr.action_id),
        "reward": float(tr.reward),
        "duration_k": float(tr.duration_k),
        "next_obs": np.asarray(tr.next_obs, dtype=np.float32).reshape(-1).copy(),
        "done": bool(tr.done),
        "episode_done": bool(tr.episode_done),
    }


def _pending_transition_from_state(state: Dict[str, Any]) -> PendingSMDPTransition:
    return PendingSMDPTransition(
        obs=np.asarray(state["obs"], dtype=np.float32).reshape(-1),
        action_id=int(state["action_id"]),
        reward=float(state["reward"]),
        duration_k=float(state["duration_k"]),
        next_obs=np.asarray(state["next_obs"], dtype=np.float32).reshape(-1),
        done=bool(state["done"]),
        episode_done=bool(state.get("episode_done", state["done"])),
    )


def _pack_nstep_queues(nstep_queues: Sequence[deque]) -> List[List[Dict[str, Any]]]:
    return [[_pending_transition_to_state(tr) for tr in q] for q in nstep_queues]


def _restore_nstep_queues(nstep_queues: Sequence[deque], state) -> None:
    if state is None:
        return
    if len(state) != len(nstep_queues):
        raise ValueError(
            f"n-step queue count mismatch: ckpt={len(state)} current={len(nstep_queues)}"
        )
    for q, q_state in zip(nstep_queues, state):
        q.clear()
        for tr_state in q_state:
            q.append(_pending_transition_from_state(tr_state))


def _pack_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["torch_cuda_all"] = None
    return state


def _restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not isinstance(state, dict):
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch_cpu_state = state["torch_cpu"]
        if hasattr(torch_cpu_state, "cpu"):
            torch_cpu_state = torch_cpu_state.cpu()
        torch.set_rng_state(torch_cpu_state)
    cuda_state = state.get("torch_cuda_all")
    if cuda_state is not None and torch.cuda.is_available():
        try:
            if isinstance(cuda_state, (list, tuple)):
                cuda_state = [s.cpu() if hasattr(s, "cpu") else s for s in cuda_state]
            torch.cuda.set_rng_state_all(cuda_state)
        except Exception as exc:
            print(
                f"[ExRL][RESUME][WARN] failed to restore CUDA RNG state: {exc}",
                flush=True,
            )


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
    spec: str, max_chunk_len: int, min_allowed_k: int = 1
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
        if not (int(min_allowed_k) <= k <= int(max_chunk_len)):
            raise ValueError(
                f"Schedule k must be in [{int(min_allowed_k)}, {max_chunk_len}], got {k}"
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


def _history_has_env_step_eval_at_zero(history: Dict) -> bool:
    steps = history.get("env_step_eval", [])
    values = history.get("metrics", {}).get(
        "env_step_eval/success_rate_deterministic", []
    )
    if not values:
        values = history.get("metrics", {}).get("env_step_eval/success_rate", [])
    for step, _value in zip(steps, values):
        if int(step) == 0:
            return True
    return False


def _trim_history_to_resume_step(
    history: Dict[str, Any], policy_steps: int, env_steps: int
) -> Dict[str, Any]:
    """Drop JSON metric points that are ahead of the checkpoint being resumed."""
    history.setdefault("timesteps", [])
    history.setdefault("timesteps_env", [])
    history.setdefault("timesteps_policy", [])
    history.setdefault("metrics", {})

    axis_keys = ("timesteps", "timesteps_env", "timesteps_policy")
    axes = {
        key: history.get(key, []) if isinstance(history.get(key, []), list) else []
        for key in axis_keys
    }
    original_axis_len = max((len(v) for v in axes.values()), default=0)
    keep_len = original_axis_len

    for idx in range(original_axis_len):
        too_far = False
        if idx < len(axes["timesteps_env"]):
            try:
                too_far = too_far or int(axes["timesteps_env"][idx]) > int(env_steps)
            except Exception:
                pass
        if idx < len(axes["timesteps_policy"]):
            try:
                too_far = too_far or int(axes["timesteps_policy"][idx]) > int(
                    policy_steps
                )
            except Exception:
                pass
        if idx < len(axes["timesteps"]) and len(axes["timesteps_env"]) == 0:
            try:
                too_far = too_far or int(axes["timesteps"][idx]) > int(env_steps)
            except Exception:
                pass
        if too_far:
            keep_len = idx
            break

    trimmed_axis_points = 0
    for key in axis_keys:
        vals = history.get(key, [])
        if not isinstance(vals, list):
            history[key] = []
            continue
        if len(vals) > keep_len:
            trimmed_axis_points = max(trimmed_axis_points, len(vals) - keep_len)
            history[key] = vals[:keep_len]

    trimmed_eval_points = 0
    eval_axis = history.get("env_step_eval", [])
    eval_keep_len = None
    if isinstance(eval_axis, list):
        eval_keep_len = len(eval_axis)
        for idx, step in enumerate(eval_axis):
            try:
                if int(step) > int(env_steps):
                    eval_keep_len = idx
                    break
            except Exception:
                pass
        if len(eval_axis) > eval_keep_len:
            trimmed_eval_points = len(eval_axis) - eval_keep_len
            history["env_step_eval"] = eval_axis[:eval_keep_len]

    trimmed_metric_keys = []
    metrics = history.get("metrics", {})
    if not isinstance(metrics, dict):
        history["metrics"] = {}
        metrics = history["metrics"]
    for key, vals in list(metrics.items()):
        if not isinstance(vals, list):
            continue
        if str(key).startswith("env_step_eval/"):
            if eval_keep_len is not None and len(vals) > eval_keep_len:
                metrics[key] = vals[:eval_keep_len]
                trimmed_metric_keys.append(str(key))
            continue
        if len(vals) > keep_len:
            metrics[key] = vals[:keep_len]
            trimmed_metric_keys.append(str(key))

    return {
        "original_axis_len": int(original_axis_len),
        "kept_axis_len": int(keep_len),
        "trimmed_axis_points": int(trimmed_axis_points),
        "trimmed_eval_points": int(trimmed_eval_points),
        "trimmed_metric_keys": trimmed_metric_keys,
    }


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


from exrl.runners.act_sac.options import build_parser


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


def _resolve_task_defaults(args):
    task_name = str(args.task or args.env_name).strip().lower()
    if not task_name:
        raise ValueError("either --task or --env_name must be provided")

    setup = resolve_act_task_setup(task=task_name)
    args.task = task_name
    if not args.env_name:
        args.env_name = task_name
    if not args.env_meta_json:
        if not setup["env_meta_json"]:
            raise FileNotFoundError(
                f"no env_meta_json preset found for task={task_name}"
            )
        args.env_meta_json = setup["env_meta_json"]
    if not args.stats_npz_override:
        if not setup["stats_npz"]:
            raise FileNotFoundError(f"no stats_npz preset found for task={task_name}")
        args.stats_npz_override = setup["stats_npz"]
    if not args.dsrl_normalization_npz:
        if not setup["dsrl_normalization_npz"]:
            raise FileNotFoundError(
                f"no dsrl_normalization_npz preset found for task={task_name}"
            )
        args.dsrl_normalization_npz = setup["dsrl_normalization_npz"]
    if not args.lowdim_keys:
        ordered_keys = list(
            dict.fromkeys(list(setup["state_obs_keys"]) + list(setup["env_obs_keys"]))
        )
        args.lowdim_keys = ",".join(ordered_keys)
    return args


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
    try:
        torch.save(payload, tmp, pickle_protocol=4)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


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
        physical_obs_normalization=args.physical_obs_normalization,
        action_obs_normalization=args.action_obs_normalization,
        physical_obs_clip=args.physical_obs_clip,
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
        n_envs=eval_n_envs,
        max_episode_steps=int(max_episode_steps),
        gamma=args.gamma,
        seed=int(args.seed) + 1_000_003,
        min_chunk_len=int(args.min_chunk_len),
        terminate_on_success=bool(args.terminate_on_success),
        timeout_on_horizon=bool(args.timeout_on_horizon),
    )
    return VecMonitor(eval_venv)


def _run_online_greedy_eval(
    *,
    eval_venv,
    sac: ExecutionLengthAgent,
    action_shift: int,
    episodes: int,
    max_decisions_per_episode: int,
    deterministic: bool = True,
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
            actions = sac.select_action(
                obs, deterministic=bool(deterministic), min_action_id=0
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

    if args.resume and args.resume_from:
        raise ValueError("--resume and --resume_from are mutually exclusive")
    resume_source_dir = args.resume or args.resume_from
    resume_active = resume_source_dir is not None
    if resume_source_dir is not None and not os.path.exists(resume_source_dir):
        raise FileNotFoundError(f"resume source dir not found: {resume_source_dir}")
    if args.resume:
        run_dir = args.resume
    else:
        run_dir = _create_new_run_dir(args.log_dir)
    os.makedirs(run_dir, exist_ok=True)
    if hasattr(args, "public_config"):
        from exrl.config import save_config

        save_config(args.public_config, os.path.join(run_dir, "experiment.yaml"))
    print(f"Run directory: {run_dir}", flush=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    resume_ckpt_dir = (
        os.path.join(resume_source_dir, "checkpoints") if resume_active else None
    )
    video_dir = os.path.join(run_dir, "videos")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    lock = _RunDirLock(run_dir)
    lock.acquire(
        mode="resume" if args.resume else ("resume_from" if args.resume_from else "new")
    )
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
        "resume_source_dir": resume_source_dir,
        "resume_checkpoint_path": (
            os.path.join(resume_ckpt_dir, args.resume_ckpt_name)
            if resume_ckpt_dir is not None
            else None
        ),
        "resume_writes_source_dir": bool(args.resume),
        "args": vars(args),
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    act_inference_device = device
    if device.type == "cuda":
        requested_act_gpus = [
            item.strip() for item in str(args.dp_infer_gpus).split(",") if item.strip()
        ]
        if len(requested_act_gpus) != 1:
            raise ValueError(
                "ACT direct inference requires exactly one --dp_infer_gpus index; "
                f"got {args.dp_infer_gpus!r}"
            )
        try:
            act_gpu_index = int(requested_act_gpus[0])
        except ValueError as exc:
            raise ValueError(
                f"invalid ACT --dp_infer_gpus index: {requested_act_gpus[0]!r}"
            ) from exc
        visible_gpu_count = int(torch.cuda.device_count())
        if not 0 <= act_gpu_index < visible_gpu_count:
            raise ValueError(
                f"ACT inference cuda:{act_gpu_index} is not visible "
                f"(visible GPU count={visible_gpu_count})"
            )
        act_inference_device = torch.device(f"cuda:{act_gpu_index}")
    print(f"[ExRL] device={device} ACT inference device={act_inference_device}")
    lowdim_keys = parse_lowdim_keys(args.lowdim_keys)
    lowdim_dims = infer_lowdim_dims_from_stats(
        args.stats_npz_override, lowdim_keys=lowdim_keys
    )
    policy, ckpt_meta = load_act_policy_wrapper(
        ckpt_path=args.agent,
        device=act_inference_device,
        task=args.task or args.env_name,
    )
    env_meta = load_env_meta_json(args.env_meta_json)
    env_meta.setdefault("env_kwargs", {})
    env_meta["env_kwargs"]["horizon"] = int(args.env_horizon_override)
    metadata["act_ckpt_meta"] = ckpt_meta
    metadata["env_meta_json"] = args.env_meta_json
    metadata["sac_device"] = str(device)
    metadata["act_inference_device"] = str(act_inference_device)
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(
        f"[ExRL][ACT-COMPAT] env={args.env_name} horizon={env_meta['env_kwargs']['horizon']} "
        f"chunk_size={ckpt_meta['chunk_size']}"
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
        physical_obs_normalization=args.physical_obs_normalization,
        action_obs_normalization=args.action_obs_normalization,
        physical_obs_clip=args.physical_obs_clip,
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
        n_envs=args.n_envs,
        max_episode_steps=int(
            env_meta.get("env_kwargs", {}).get("horizon", args.env_horizon_override)
        ),
        gamma=args.gamma,
        seed=args.seed,
        min_chunk_len=int(args.min_chunk_len),
        terminate_on_success=bool(args.terminate_on_success),
        timeout_on_horizon=bool(args.timeout_on_horizon),
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
    warmup_fixed_k_min = max(1, int(args.min_chunk_len))
    if not (warmup_fixed_k_min <= int(args.warmup_fixed_k) <= int(args.max_chunk_len)):
        raise ValueError(
            f"--warmup_fixed_k must be in [{warmup_fixed_k_min}, {args.max_chunk_len}] "
            f"when --min_chunk_len={int(args.min_chunk_len)}"
        )
    if not (0.0 <= float(args.warmup_mixed_uniform_prob) <= 1.0):
        raise ValueError("--warmup_mixed_uniform_prob must be in [0, 1]")
    if not (0.0 <= float(args.epsilon_greedy_prob) <= 1.0):
        raise ValueError("--epsilon_greedy_prob must be in [0, 1]")
    if int(args.rollout_deterministic_after_env_steps) < -1:
        raise ValueError("--rollout_deterministic_after_env_steps must be >= -1")
    if int(args.rollout_deterministic_after_policy_steps) < -1:
        raise ValueError("--rollout_deterministic_after_policy_steps must be >= -1")
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
    if int(args.online_eval_every_env_steps) < 0:
        raise ValueError("--online_eval_every_env_steps must be >= 0")
    if int(args.eval_snapshot_env_freq) < 0:
        raise ValueError("--eval_snapshot_env_freq must be >= 0")
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
        args.post_warmup_min_k_schedule,
        args.max_chunk_len,
        min_allowed_k=int(args.min_chunk_len),
    )
    dynamic_release_target_items = _parse_min_k_schedule(
        args.dynamic_floor_release_target_schedule,
        args.max_chunk_len,
        min_allowed_k=int(args.min_chunk_len),
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
    pending_transitions_for_update = 0
    num_emitted_transitions_total = 0
    num_updates_total = 0
    num_done_for_backup_true = 0
    num_timeout_truncations = 0
    duration_sum_total = 0.0
    history = {
        "timesteps": [],
        "timesteps_env": [],
        "timesteps_policy": [],
        "metrics": {},
    }
    metrics_path = os.path.join(run_dir, "training_metrics.json")
    metrics_backup = os.path.join(run_dir, "training_metrics.backup.json")
    status_log_path = os.path.join(run_dir, "status.log")
    resume_state = None
    offline_phase_state = {
        "phase": 0,
        "phase0_policy_steps": 0,
        "phase1_critic_updates": 0,
        "phase2_actor_updates": 0,
        "pre_warmstart_eval_done": False,
        "complete": False,
    }

    if resume_active:
        latest_path = os.path.join(resume_ckpt_dir, args.resume_ckpt_name)
        if not os.path.exists(latest_path):
            raise FileNotFoundError(f"resume checkpoint not found: {latest_path}")
        # Keep replay tensor storages on the host; module and optimizer loaders
        # move their own state to the SAC device without staging replay on CUDA.
        resume_state = torch.load(latest_path, map_location="cpu", weights_only=False)
        sac.load_state_dict(resume_state["sac_state"])
        policy_steps = int(resume_state.get("policy_steps", 0))
        env_steps = int(resume_state.get("env_steps", 0))
        best_success = float(resume_state.get("best_success", -1.0))
        dynamic_floor_release_latched = bool(
            resume_state.get("dynamic_floor_release_latched", False)
        )
        dynamic_floor_release_policy_step = resume_state.get(
            "dynamic_floor_release_policy_step"
        )
        dynamic_floor_release_env_step = resume_state.get(
            "dynamic_floor_release_env_step"
        )
        replay_state = resume_state.get("replay_state")
        if replay_state is not None:
            replay.load_state_dict(replay_state)
        offline_phase_state.update(dict(resume_state.get("offline_phase_state", {})))
        step_calls = int(resume_state.get("step_calls", step_calls))
        pending_transitions_for_update = int(
            resume_state.get(
                "pending_transitions_for_update", pending_transitions_for_update
            )
        )
        num_emitted_transitions_total = int(
            resume_state.get(
                "num_emitted_transitions_total", num_emitted_transitions_total
            )
        )
        num_updates_total = int(
            resume_state.get("num_updates_total", num_updates_total)
        )
        num_done_for_backup_true = int(
            resume_state.get("num_done_for_backup_true", num_done_for_backup_true)
        )
        num_timeout_truncations = int(
            resume_state.get("num_timeout_truncations", num_timeout_truncations)
        )
        duration_sum_total = float(
            resume_state.get("duration_sum_total", duration_sum_total)
        )
        source_metrics_path = (
            metrics_path
            if args.resume
            else os.path.join(resume_source_dir, "training_metrics.json")
        )
        source_metrics_backup = (
            metrics_backup
            if args.resume
            else os.path.join(resume_source_dir, "training_metrics.backup.json")
        )
        loaded_history, loaded_from = _load_json_with_backup(
            source_metrics_path, source_metrics_backup
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
            trim_info = _trim_history_to_resume_step(
                history, policy_steps=policy_steps, env_steps=env_steps
            )
            if (
                trim_info["trimmed_axis_points"] > 0
                or trim_info["trimmed_eval_points"] > 0
                or len(trim_info["trimmed_metric_keys"]) > 0
            ):
                trim_msg = (
                    "[ExRL][RESUME] trimmed loaded metrics to checkpoint: "
                    f"policy_steps={policy_steps} env_steps={env_steps} "
                    f"kept_logs={trim_info['kept_axis_len']}/{trim_info['original_axis_len']} "
                    f"trimmed_eval_points={trim_info['trimmed_eval_points']} "
                    f"trimmed_metric_keys={len(trim_info['trimmed_metric_keys'])}"
                )
                print(trim_msg, flush=True)
                _append_status_line(status_log_path, trim_msg)
            if _history_has_env_step_eval_at_zero(history):
                offline_phase_state["pre_warmstart_eval_done"] = True
            if args.resume_from:
                _atomic_json_dump(history, metrics_path, metrics_backup)
        resume_msg = f"[ExRL][RESUME] loaded metrics from {loaded_from}"
        print(resume_msg, flush=True)
        _append_status_line(status_log_path, resume_msg)
        if replay_state is not None:
            resume_msg = f"[ExRL][RESUME] replay restored size={len(replay)}"
            print(resume_msg, flush=True)
            _append_status_line(status_log_path, resume_msg)
    else:
        dynamic_floor_release_latched = False
        dynamic_floor_release_policy_step = None
        dynamic_floor_release_env_step = None

    writer = SummaryWriter(log_dir=os.path.join(run_dir, "scalars_1"))
    success_buffer = deque(maxlen=args.window_size)
    episode_return_buffer = deque(maxlen=args.window_size)
    episode_len_buffer = deque(maxlen=args.window_size)
    exec_buffer = deque(maxlen=2000)
    raw_exec_buffer = deque(maxlen=2000)
    floor_apply_buffer = deque(maxlen=2000)
    behavior_mix_buffer = deque(maxlen=2000)
    rollout_deterministic_buffer = deque(maxlen=2000)
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

    def _include_full_resume_state(kind: str) -> bool:
        mode = str(args.full_resume_save_mode)
        if mode == "none":
            return False
        if kind == "latest":
            return mode in ("latest", "periodic", "all")
        if kind == "periodic":
            return mode in ("periodic", "all")
        if kind == "best":
            return mode == "all"
        return False

    def _pack_full_resume_state() -> Dict[str, Any]:
        elapsed = max(time.time() - start_time, 0.0)
        return {
            "schema_version": 1,
            "replay": _pack_replay_state(replay),
            "rng": _pack_rng_state(),
            "buffers": {
                "success_buffer": _copy_deque_values(success_buffer),
                "episode_return_buffer": _copy_deque_values(episode_return_buffer),
                "episode_len_buffer": _copy_deque_values(episode_len_buffer),
                "exec_buffer": _copy_deque_values(exec_buffer),
                "raw_exec_buffer": _copy_deque_values(raw_exec_buffer),
                "floor_apply_buffer": _copy_deque_values(floor_apply_buffer),
                "behavior_mix_buffer": _copy_deque_values(behavior_mix_buffer),
                "rollout_deterministic_buffer": _copy_deque_values(
                    rollout_deterministic_buffer
                ),
                "actor_prior_buffer": _copy_deque_values(actor_prior_buffer),
                "epsilon_apply_buffer": _copy_deque_values(epsilon_apply_buffer),
            },
            "latest_train_metrics": dict(latest_train_metrics),
            "train_meter": {str(k): list(v) for k, v in train_meter.items()},
            "counters": {
                "step_calls": int(step_calls),
                "pending_transitions_for_update": int(pending_transitions_for_update),
                "num_emitted_transitions_total": int(num_emitted_transitions_total),
                "num_updates_total": int(num_updates_total),
                "num_done_for_backup_true": int(num_done_for_backup_true),
                "num_timeout_truncations": int(num_timeout_truncations),
                "duration_sum_total": float(duration_sum_total),
                "log_idx": int(log_idx),
                "elapsed_wall_sec": float(elapsed),
            },
            "collector_env_state": {
                "obs": np.asarray(obs, dtype=np.float32).copy(),
                "episode_returns": np.asarray(episode_returns, dtype=np.float32).copy(),
                "episode_lengths": np.asarray(episode_lengths, dtype=np.int64).copy(),
                "nstep_queues": _pack_nstep_queues(nstep_queues),
            },
            "collector_env_state_restored": False,
            "collector_env_state_note": (
                "Stored for diagnostics only. DSRL async env simulator state is not restored yet; "
                "resume resets envs and intentionally does not reattach these partial episode queues."
            ),
        }

    def pack_ckpt(include_full_resume_state: bool = False) -> Dict:
        payload = {
            "checkpoint_schema_version": 2,
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
            "target_q_clip_min": args.target_q_clip_min,
            "target_q_clip_max": args.target_q_clip_max,
            "max_chunk_len": int(args.max_chunk_len),
            "min_chunk_len": int(args.min_chunk_len),
            "dynamic_floor_release_latched": bool(dynamic_floor_release_latched),
            "dynamic_floor_release_policy_step": dynamic_floor_release_policy_step,
            "dynamic_floor_release_env_step": dynamic_floor_release_env_step,
            "replay_state": (
                _pack_replay_state(replay)
                if str(args.pipeline) == "sac_warmstart"
                and not bool(include_full_resume_state)
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
            "full_resume_state_included": bool(include_full_resume_state),
        }
        if bool(include_full_resume_state):
            payload["full_resume_state"] = _pack_full_resume_state()
        return payload

    eval_snapshot_interval = int(args.eval_snapshot_env_freq)
    next_eval_snapshot_env_step: Optional[int] = (
        _next_checkpoint_step(env_steps, eval_snapshot_interval)
        if eval_snapshot_interval > 0
        else None
    )

    def pack_eval_actor_snapshot(snapshot_env_step: int) -> Dict[str, Any]:
        actor_state = {
            key: value.detach().cpu().clone()
            for key, value in sac.actor.state_dict().items()
        }
        return {
            "snapshot_format": "act_exrl_sac_actor_eval_v1",
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

    next_periodic_ckpt = _next_checkpoint_step(policy_steps, int(args.save_freq))
    next_latest_ckpt = _next_checkpoint_step(
        policy_steps, int(args.latest_model_save_freq)
    )
    save_env_freq = int(args.save_env_freq)
    latest_model_save_env_freq = int(args.latest_model_save_env_freq)
    next_periodic_env_ckpt = (
        _next_checkpoint_step(env_steps, save_env_freq) if save_env_freq > 0 else None
    )
    next_latest_env_ckpt = (
        _next_checkpoint_step(env_steps, latest_model_save_env_freq)
        if latest_model_save_env_freq > 0
        else None
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
    if resume_active:
        full_state = (
            None if resume_state is None else resume_state.get("full_resume_state")
        )
        if full_state is None:
            msg = (
                "[ExRL][RESUME][PARTIAL] checkpoint has no full_resume_state. "
                "This would resume model/optimizers with fresh RNG/window state."
            )
            print(msg, flush=True)
            _append_status_line(status_log_path, msg)
            has_top_level_replay = replay_state is not None
            if (not has_top_level_replay) and not bool(args.resume_allow_partial_state):
                raise RuntimeError(
                    msg
                    + " Pass --resume_allow_partial_state only if you intentionally want legacy partial resume."
                )
            if has_top_level_replay:
                msg = (
                    "[ExRL][RESUME][PARTIAL] continuing with top-level replay_state; "
                    "RNG/window state is fresh."
                )
            else:
                msg = "[ExRL][RESUME][PARTIAL] allowed by --resume_allow_partial_state; replay remains empty."
            print(msg, flush=True)
            _append_status_line(status_log_path, msg)
        else:
            _restore_replay_state(replay, full_state.get("replay", {}))
            _restore_rng_state(full_state.get("rng"))

            buffers = full_state.get("buffers", {})
            _restore_deque_values(success_buffer, buffers.get("success_buffer"))
            _restore_deque_values(
                episode_return_buffer, buffers.get("episode_return_buffer")
            )
            _restore_deque_values(episode_len_buffer, buffers.get("episode_len_buffer"))
            _restore_deque_values(exec_buffer, buffers.get("exec_buffer"))
            _restore_deque_values(raw_exec_buffer, buffers.get("raw_exec_buffer"))
            _restore_deque_values(floor_apply_buffer, buffers.get("floor_apply_buffer"))
            _restore_deque_values(
                behavior_mix_buffer, buffers.get("behavior_mix_buffer")
            )
            _restore_deque_values(
                rollout_deterministic_buffer,
                buffers.get("rollout_deterministic_buffer"),
            )
            _restore_deque_values(actor_prior_buffer, buffers.get("actor_prior_buffer"))
            _restore_deque_values(
                epsilon_apply_buffer, buffers.get("epsilon_apply_buffer")
            )

            latest_train_metrics.update(
                {
                    str(k): float(v)
                    for k, v in dict(full_state.get("latest_train_metrics", {})).items()
                }
            )
            for k, vals in dict(full_state.get("train_meter", {})).items():
                train_meter[str(k)].extend(float(v) for v in vals)

            counters = full_state.get("counters", {})
            step_calls = int(counters.get("step_calls", step_calls))
            pending_transitions_for_update = int(
                counters.get(
                    "pending_transitions_for_update", pending_transitions_for_update
                )
            )
            num_emitted_transitions_total = int(
                counters.get(
                    "num_emitted_transitions_total", num_emitted_transitions_total
                )
            )
            num_updates_total = int(
                counters.get("num_updates_total", num_updates_total)
            )
            num_done_for_backup_true = int(
                counters.get("num_done_for_backup_true", num_done_for_backup_true)
            )
            num_timeout_truncations = int(
                counters.get("num_timeout_truncations", num_timeout_truncations)
            )
            duration_sum_total = float(
                counters.get("duration_sum_total", duration_sum_total)
            )
            log_idx = int(counters.get("log_idx", log_idx))
            elapsed_wall_sec = float(counters.get("elapsed_wall_sec", 0.0))
            if elapsed_wall_sec > 0:
                start_time = time.time() - elapsed_wall_sec

            # Current simulator/vector-env state is not serializable in the DSRL async stack yet.
            # Do not restore pending n-step queues or per-env episode accumulators against a freshly
            # reset env, because mixing old partial episodes with new env states would corrupt replay.
            skipped_env_state = bool(full_state.get("collector_env_state"))
            msg = (
                "[ExRL][RESUME] loaded full training state: "
                f"replay={len(replay)} success_window={len(success_buffer)} "
                f"step_calls={step_calls} updates={num_updates_total}"
            )
            print(msg, flush=True)
            _append_status_line(status_log_path, msg)
            if skipped_env_state:
                msg = (
                    "[ExRL][RESUME][WARN] env simulator state is not restored; collector resumes from fresh reset. "
                    "Replay/RNG/windows/counters are restored, but this is not bit-exact physical continuation."
                )
                print(msg, flush=True)
                _append_status_line(status_log_path, msg)

    online_eval_venv = None
    online_eval_interval = int(args.online_eval_every_env_steps)
    next_online_eval_env_step: Optional[int] = None
    online_eval_modes = (
        ("deterministic", "stochastic")
        if str(args.online_eval_action_mode) == "both"
        else (str(args.online_eval_action_mode),)
    )
    if online_eval_interval > 0:
        eval_max_episode_steps = int(
            env_meta.get("env_kwargs", {}).get("horizon", args.env_horizon_override)
        )
        # Eval env construction should not perturb training RNG, especially after full resume.
        rng_before_eval_env = _pack_rng_state()
        online_eval_venv = _make_online_eval_venv(
            args=args,
            policy=policy,
            device=device,
            lowdim_keys=lowdim_keys,
            lowdim_dims=lowdim_dims,
            max_episode_steps=eval_max_episode_steps,
        )
        _restore_rng_state(rng_before_eval_env)
        history.setdefault("env_step_eval", [])
        next_online_eval_env_step = (
            0
            if int(env_steps) == 0
            else ((int(env_steps) // online_eval_interval + 1) * online_eval_interval)
        )

    def maybe_run_online_eval(force: bool = False):
        nonlocal next_online_eval_env_step
        if online_eval_venv is None or online_eval_interval <= 0:
            return
        if (
            (not force)
            and next_online_eval_env_step is not None
            and int(env_steps) < int(next_online_eval_env_step)
        ):
            return
        eval_rng_state = _pack_rng_state()
        results: Dict[str, float] = {}
        try:
            for mode in online_eval_modes:
                mode_t0 = time.time()
                success_rate = _run_online_greedy_eval(
                    eval_venv=online_eval_venv,
                    sac=sac,
                    action_shift=int(action_shift),
                    episodes=int(args.online_eval_episodes),
                    max_decisions_per_episode=int(
                        args.online_eval_max_decisions_per_episode
                    ),
                    deterministic=(mode == "deterministic"),
                )
                results[mode] = float(success_rate)
                key = f"env_step_eval/success_rate_{mode}"
                _record(history, key, float(success_rate))
                writer.add_scalar(key, float(success_rate), int(env_steps))
                eval_line = (
                    f"[ExRL][ONLINE-EVAL] env_steps={env_steps} "
                    f"mode={mode} episodes={int(args.online_eval_episodes)} "
                    f"success_rate={float(success_rate):.4f} "
                    f"elapsed_sec={time.time() - mode_t0:.1f}"
                )
                print(eval_line, flush=True)
                _append_status_line(status_log_path, eval_line)
        finally:
            _restore_rng_state(eval_rng_state)

        history.setdefault("env_step_eval", []).append(int(env_steps))
        primary_mode = (
            "deterministic" if "deterministic" in results else online_eval_modes[0]
        )
        primary_success_rate = float(results[primary_mode])
        _record(history, "env_step_eval/success_rate", primary_success_rate)
        writer.add_scalar(
            "env_step_eval/success_rate", primary_success_rate, int(env_steps)
        )
        try:
            writer.flush()
        except Exception:
            pass
        dump_metrics()
        if next_online_eval_env_step is None:
            next_online_eval_env_step = 0
        while int(next_online_eval_env_step) <= int(env_steps):
            next_online_eval_env_step += online_eval_interval

    if eval_snapshot_interval > 0 and not resume_active:
        # Preserve the true initial actor before offline collection or warmstart updates.
        save_eval_actor_snapshot(0)

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
                _save_ckpt(
                    os.path.join(ckpt_dir, "latest_model.pt"),
                    pack_ckpt(
                        include_full_resume_state=_include_full_resume_state("latest")
                    ),
                )

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
                            success_buffer.append(1 if _extract_success(info) else 0)
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
                            os.path.join(ckpt_dir, "latest_model.pt"),
                            pack_ckpt(
                                include_full_resume_state=_include_full_resume_state(
                                    "latest"
                                )
                            ),
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
                    f"cql_alpha={float(args.cql_alpha)} "
                    f"critic_ensemble_size={int(args.critic_ensemble_size)} "
                    f"target_q_clip=[{args.target_q_clip_min},{args.target_q_clip_max}]"
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
                        if should_eval:
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
                            os.path.join(ckpt_dir, "latest_model.pt"),
                            pack_ckpt(
                                include_full_resume_state=_include_full_resume_state(
                                    "latest"
                                )
                            ),
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
                        if should_eval:
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
                            os.path.join(ckpt_dir, "latest_model.pt"),
                            pack_ckpt(
                                include_full_resume_state=_include_full_resume_state(
                                    "latest"
                                )
                            ),
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
                _save_ckpt(
                    os.path.join(ckpt_dir, "latest_model.pt"),
                    pack_ckpt(
                        include_full_resume_state=_include_full_resume_state("latest")
                    ),
                )
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

        while policy_steps < int(args.total_policy_steps) and (
            int(args.stop_after_env_steps) <= 0
            or int(env_steps) < int(args.stop_after_env_steps)
        ):
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
                    use_floor_mask_sample = (
                        floor_support_mask_mode in ("sample_only", "all")
                        and current_min_k_floor > 1
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
                    rollout_deterministic = (
                        int(args.rollout_deterministic_after_env_steps) >= 0
                        and int(env_steps)
                        >= int(args.rollout_deterministic_after_env_steps)
                    ) or (
                        int(args.rollout_deterministic_after_policy_steps) >= 0
                        and int(policy_steps)
                        >= int(args.rollout_deterministic_after_policy_steps)
                    )
                    if rollout_deterministic:
                        if str(args.rollout_deterministic_source) == "q":
                            actions = sac.select_critic_action(
                                obs,
                                deterministic=True,
                                min_action_id=int(min_action_id),
                                mode="q_greedy",
                                reduce="min",
                            )
                        else:
                            actions = (
                                torch.argmax(sampling_probs_t, dim=1)
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.int64)
                            )
                    elif behavior_mix_beta > 0.0:
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
                        rand_low = min_action_id if current_min_k_floor > 1 else 0
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

                if current_min_k_floor > 1:
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
                rollout_deterministic_buffer.append(float(rollout_deterministic))
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
                    success_buffer.append(1 if _extract_success(info) else 0)
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
                        target_policy_mode="actor",
                    )
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
                    float(np.mean(list(success_buffer)[-20:]))
                    if len(success_buffer) > 0
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
                if len(rollout_deterministic_buffer) > 0:
                    _record(
                        history,
                        "diagnostic/rollout_deterministic_recent",
                        float(np.mean(rollout_deterministic_buffer)),
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
                if len(rollout_deterministic_buffer) > 0:
                    writer.add_scalar(
                        "diagnostic/rollout_deterministic_recent",
                        float(np.mean(rollout_deterministic_buffer)),
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
                    f"sr20={success_recent20:.4f} "
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

                if success_recent > best_success and len(success_buffer) > 0:
                    best_success = success_recent
                    _save_ckpt(
                        os.path.join(ckpt_dir, "best_model.pt"),
                        pack_ckpt(
                            include_full_resume_state=_include_full_resume_state("best")
                        ),
                    )

            if policy_steps >= next_periodic_ckpt:
                _save_ckpt(
                    os.path.join(ckpt_dir, f"adaptive_agent_{policy_steps}_steps.pt"),
                    pack_ckpt(
                        include_full_resume_state=_include_full_resume_state("periodic")
                    ),
                )
                next_periodic_ckpt += max(int(args.save_freq), 1)

            if policy_steps >= next_latest_ckpt:
                _save_ckpt(
                    os.path.join(ckpt_dir, "latest_model.pt"),
                    pack_ckpt(
                        include_full_resume_state=_include_full_resume_state("latest")
                    ),
                )
                next_latest_ckpt += max(int(args.latest_model_save_freq), 1)

            if (
                next_periodic_env_ckpt is not None
                and env_steps >= next_periodic_env_ckpt
            ):
                _save_ckpt(
                    os.path.join(ckpt_dir, f"adaptive_agent_{env_steps}_env_steps.pt"),
                    pack_ckpt(
                        include_full_resume_state=_include_full_resume_state("periodic")
                    ),
                )
                while env_steps >= next_periodic_env_ckpt:
                    next_periodic_env_ckpt += save_env_freq

            if next_latest_env_ckpt is not None and env_steps >= next_latest_env_ckpt:
                _save_ckpt(
                    os.path.join(ckpt_dir, "latest_model.pt"),
                    pack_ckpt(
                        include_full_resume_state=_include_full_resume_state("latest")
                    ),
                )
                while env_steps >= next_latest_env_ckpt:
                    next_latest_env_ckpt += latest_model_save_env_freq

        if int(args.stop_after_env_steps) > 0 and int(env_steps) >= int(
            args.stop_after_env_steps
        ):
            line = f"[ExRL] reached stop_after_env_steps={int(args.stop_after_env_steps)} env_steps={int(env_steps)}"
            print(line, flush=True)
            _append_status_line(status_log_path, line)

        _save_ckpt(
            os.path.join(ckpt_dir, "latest_model.pt"),
            pack_ckpt(include_full_resume_state=_include_full_resume_state("latest")),
        )
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
    _resolve_task_defaults(args)
    _validate_task_alignment(args)
    os.makedirs(args.output, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
