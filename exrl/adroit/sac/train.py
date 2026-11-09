"""
Adroit SAC + AAC (Discrete k, NoZero) training entry.

This entry keeps the ExRL-SAC actor / critic / alpha training loop from
train_adaptive_sac_task_dsrl_ckpt_compat.py and swaps only the base-policy and
environment adapter to the Adroit robomimic DP setup.
"""

import argparse
import atexit
import copy
import fcntl
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecMonitor
from torch.utils.tensorboard import SummaryWriter

CURRENT_DIR = __import__("exrl.paths", fromlist=["ROOT"]).ROOT
REPO_ROOT = __import__("exrl.paths", fromlist=["ROOT"]).ROOT

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from exrl.adroit.sac.environment_adroit_sac import (
    BatchedAdaptiveWrapperAdroitSAC,
    BatchedAdaptiveWrapperAdroitSACAsyncQueue,
)
from exrl.adroit.sac.learning import (
    extract_bootstrap_next_obs_from_venv,
    ingest_smdp_env_step_to_replay,
    PendingSMDPTransition,
    SMDPDiscreteSAC,
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


@contextmanager
def _preserve_training_rng():
    """Keep evaluation from advancing any RNG stream used by training."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


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


def _trim_history_to_checkpoint(history: Dict, policy_steps: int, env_steps: int):
    trimmed = copy.deepcopy(history)
    policy_axis = list(history.get("timesteps_policy", []))
    env_axis = list(history.get("timesteps_env", []))
    time_axis = list(history.get("timesteps", []))
    if not (len(policy_axis) == len(env_axis) == len(time_axis)):
        raise ValueError("resume metrics axes have inconsistent lengths")

    keep_online = 0
    for policy_step, env_step in zip(policy_axis, env_axis):
        if int(policy_step) <= int(policy_steps) and int(env_step) <= int(env_steps):
            keep_online += 1
        else:
            break
    if any(
        int(policy_step) <= int(policy_steps) and int(env_step) <= int(env_steps)
        for policy_step, env_step in zip(policy_axis[keep_online:], env_axis[keep_online:])
    ):
        raise ValueError("resume metrics axes are not prefix-monotonic")

    eval_axis = list(history.get("env_step_eval", []))
    keep_eval = 0
    for eval_step in eval_axis:
        if int(eval_step) <= int(env_steps):
            keep_eval += 1
        else:
            break
    if any(int(eval_step) <= int(env_steps) for eval_step in eval_axis[keep_eval:]):
        raise ValueError("resume eval axis is not prefix-monotonic")

    trimmed["timesteps"] = time_axis[:keep_online]
    trimmed["timesteps_env"] = env_axis[:keep_online]
    trimmed["timesteps_policy"] = policy_axis[:keep_online]
    if "env_step_eval" in trimmed:
        trimmed["env_step_eval"] = eval_axis[:keep_eval]

    metrics = trimmed.setdefault("metrics", {})
    for key, values in list(metrics.items()):
        if not isinstance(values, list):
            continue
        if key.startswith("env_step_eval/"):
            if len(values) != len(eval_axis):
                raise ValueError(f"resume eval metric length mismatch for {key}")
            metrics[key] = values[:keep_eval]
        elif key.startswith("offline/"):
            continue
        else:
            if len(values) < len(policy_axis):
                raise ValueError(f"resume online metric length mismatch for {key}")
            prefix_len = len(values) - len(policy_axis)
            metrics[key] = values[: prefix_len + keep_online]

    return trimmed, {
        "online_before": len(policy_axis),
        "online_after": keep_online,
        "eval_before": len(eval_axis),
        "eval_after": keep_eval,
        "trimmed": keep_online < len(policy_axis) or keep_eval < len(eval_axis),
    }


def _archive_metrics_before_resume(
    paths: Sequence[str], policy_steps: int, env_steps: int
) -> List[str]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archived = []
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        base, ext = os.path.splitext(path)
        archive_path = f"{base}.pre_resume_p{int(policy_steps)}_e{int(env_steps)}_{stamp}{ext}"
        shutil.copy2(path, archive_path)
        if os.path.getsize(archive_path) != os.path.getsize(path):
            raise RuntimeError(f"resume metrics archive size mismatch: {archive_path}")
        archived.append(archive_path)
    return archived


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
            "host": socket.gethostname(),
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
    base = f"SAC_{datetime.now().strftime('%m%d_%H%M%S')}"
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


def _next_periodic_step(current_step: int, interval: int) -> int:
    interval = max(int(interval), 1)
    current_step = max(int(current_step), 0)
    return ((current_step // interval) + 1) * interval


def _parse_min_k_schedule(spec: str, max_chunk_len: int, min_chunk_len: int = 1) -> List[tuple]:
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


def _parse_float_schedule(spec: str, min_value: float = 0.0, max_value: float = 1.0) -> List[tuple]:
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
    # Priority 2: legacy fixed floor window
    if int(args.post_warmup_min_k_until_policy_steps) > 0 and int(counter_value) < int(
        args.post_warmup_min_k_until_policy_steps
    ):
        return int(args.post_warmup_min_k)
    return 1


def _resolve_float_schedule(policy_steps: int, items: List[tuple], default: float = 0.0) -> float:
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


def _sha256(path: str):
    if (not path) or (not os.path.exists(path)):
        return None
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info(repo_dir):
    commit = None
    dirty = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_dir, text=True
            ).strip()
        )
    except Exception:
        pass
    return commit, dirty


def _record(history: Dict, key: str, value: float):
    if key not in history["metrics"]:
        history["metrics"][key] = []
    history["metrics"][key].append(float(value))


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


def _extract_ckpt_stat(stats: Dict, key: str) -> np.ndarray:
    if key in stats:
        return np.asarray(stats[key], dtype=np.float32).reshape(-1)
    raise KeyError(f"normalization stat missing key={key}; available={list(stats.keys())}")


def _write_adroit_stats_npz_from_ckpt(ckpt_dict: Dict, path: str, obs_key: str = "flat") -> str:
    obs_stats = ckpt_dict.get("obs_normalization_stats", {}).get(obs_key)
    if obs_stats is None:
        raise KeyError(f"ckpt missing obs_normalization_stats[{obs_key!r}]")
    action_stats = ckpt_dict.get("action_normalization_stats", {}).get("actions")
    if action_stats is None:
        raise KeyError("ckpt missing action_normalization_stats['actions']")
    flat_mean = _extract_ckpt_stat(obs_stats, "offset")
    flat_std = _extract_ckpt_stat(obs_stats, "scale")
    action_mean = _extract_ckpt_stat(action_stats, "offset")
    action_std = _extract_ckpt_stat(action_stats, "scale")
    flat_std = np.where(flat_std < 1e-8, 1.0, flat_std).astype(np.float32)
    action_std = np.where(action_std < 1e-8, 1.0, action_std).astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        flat_mean=flat_mean.astype(np.float32),
        flat_std=flat_std.astype(np.float32),
        action_mean=action_mean.astype(np.float32),
        action_std=action_std.astype(np.float32),
    )
    return path


def _configure_adroit_policy(policy, dp_sampler: str, dp_infer_steps: int) -> Dict:
    algo = policy.policy if hasattr(policy, "policy") else policy
    pred_h = int(algo.algo_config.horizon.prediction_horizon)
    with algo.algo_config.values_unlocked():
        algo.algo_config.horizon.action_horizon = pred_h
        algo.algo_config.horizon.prediction_horizon = pred_h
        if str(dp_sampler).lower() == "ddim":
            algo.algo_config.ddpm.enabled = False
            algo.algo_config.ddim.enabled = True
            algo.algo_config.ddim.num_inference_timesteps = int(dp_infer_steps)
        else:
            algo.algo_config.ddpm.enabled = True
            algo.algo_config.ddim.enabled = False
            algo.algo_config.ddpm.num_inference_timesteps = int(dp_infer_steps)
    if hasattr(algo, "ac_horizon"):
        algo.ac_horizon = pred_h
    if hasattr(algo, "use_action_scheduler"):
        algo.use_action_scheduler = False
    if hasattr(algo, "use_ood_monitor"):
        algo.use_ood_monitor = False
    try:
        algo.nets.eval()
        for p in algo.nets.parameters():
            p.requires_grad = False
    except Exception:
        pass
    try:
        policy.start_episode()
    except Exception:
        pass
    return {
        "prediction_horizon": int(pred_h),
        "action_horizon": int(getattr(algo, "ac_horizon", pred_h)),
        "dp_sampler": str(dp_sampler),
        "dp_infer_steps": int(dp_infer_steps),
        "use_action_scheduler": bool(getattr(algo, "use_action_scheduler", False)),
        "use_ood_monitor": bool(getattr(algo, "use_ood_monitor", False)),
    }


def _build_adroit_venv(
    *,
    args,
    policy,
    ckpt_dict,
    device,
    env_meta: Dict,
    stats_npz_path: str,
    n_envs: int,
    seed: int,
    runtime: str,
):
    wrapper_cls = (
        BatchedAdaptiveWrapperAdroitSACAsyncQueue
        if str(runtime) == "asyncqueue"
        else BatchedAdaptiveWrapperAdroitSAC
    )
    args_for_env = copy.copy(args)
    args_for_env.seed = int(seed)
    venv = wrapper_cls(
        policy=policy,
        ckpt_dict=ckpt_dict,
        env_meta=copy.deepcopy(env_meta),
        stats_npz_path=stats_npz_path,
        max_chunk_len=int(args.max_chunk_len),
        device=device,
        n_envs=int(n_envs),
        vecenv_start_method=str(args.vecenv_start_method),
        resample_penalty=float(args.resample_penalty),
        action_history_len=int(args.action_history_len),
        dp_infer_steps=int(args.dp_infer_steps),
        dp_sampler=str(args.dp_sampler),
        dp_amp=bool(args.dp_amp),
        dp_compile=bool(args.dp_compile),
        ipc_chunk_dtype=str(args.ipc_chunk_dtype),
        dp_infer_gpus=str(args.dp_infer_gpus),
        dp_grouping=str(args.dp_grouping),
        dp_infer_parallel=bool(args.dp_infer_parallel),
        ipc_send_mode=str(args.ipc_send_mode),
        perf_profile=False,
        min_chunk_len=int(args.min_chunk_len),
        td_target_next_obs_samples=int(args.td_target_next_obs_samples),
        critic_obs_state_mode=str(args.critic_obs_state_mode),
        args=args_for_env,
    )
    if str(runtime) == "asyncqueue":
        required = ("get_policy_obs", "dispatch_policy_actions", "poll_step_results")
        missing = [name for name in required if not callable(getattr(venv, name, None))]
        if missing:
            raise RuntimeError(f"Adroit asyncqueue wrapper missing required primitives: {missing}")
    return VecMonitor(venv)


def _make_online_eval_venv(args, policy, ckpt_dict, device, env_meta: Dict, stats_npz_path: str):
    eval_episodes = int(args.online_eval_episodes)
    if eval_episodes <= 0:
        raise ValueError("--online_eval_episodes must be > 0 when online eval is enabled")
    eval_n_envs = int(args.online_eval_n_envs)
    if eval_n_envs <= 0:
        eval_n_envs = min(int(args.n_envs), eval_episodes)
    eval_n_envs = max(1, min(eval_n_envs, eval_episodes))

    return _build_adroit_venv(
        args=args,
        policy=policy,
        ckpt_dict=ckpt_dict,
        device=device,
        env_meta=env_meta,
        stats_npz_path=stats_npz_path,
        n_envs=eval_n_envs,
        seed=int(args.seed) + 1_000_003,
        runtime="sync",
    )


def _unwrap_asyncqueue_env(venv):
    cur = venv
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if (
            callable(getattr(cur, "get_policy_obs", None))
            and callable(getattr(cur, "dispatch_policy_actions", None))
            and callable(getattr(cur, "poll_step_results", None))
        ):
            return cur
        cur = getattr(cur, "venv", None)
    raise RuntimeError(
        "decision_runtime=asyncqueue requested, but asyncqueue wrapper was not found in env chain"
    )


def _step_train_venv(venv, args, actions: np.ndarray, action_shift: int, obs_template: np.ndarray):
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    if str(args.decision_runtime) != "asyncqueue":
        return venv.step(actions + int(action_shift))
    if int(action_shift) != 0:
        raise RuntimeError(
            "Adroit asyncqueue path expects unshifted action ids; got nonzero action_shift"
        )

    real_env = _unwrap_asyncqueue_env(venv)
    n_envs = int(real_env.num_envs)
    if actions.shape[0] != n_envs:
        raise RuntimeError(
            f"asyncqueue action shape mismatch: actions={actions.shape[0]} n_envs={n_envs}"
        )

    env_ids = np.arange(n_envs, dtype=np.int64)
    real_env.dispatch_policy_actions(env_ids, actions)

    next_obs = np.zeros_like(np.asarray(obs_template, dtype=np.float32))
    rewards = np.zeros((n_envs,), dtype=np.float32)
    dones = np.zeros((n_envs,), dtype=bool)
    infos = [{} for _ in range(n_envs)]
    pending = set(int(i) for i in env_ids.tolist())
    timeout_sec = float(os.environ.get("OPT_ASYNCQUEUE_BATCH_TIMEOUT_SEC", "300"))
    t0 = time.time()
    while pending:
        ready = real_env.poll_step_results(
            min_ready=1,
            timeout_ms=int(args.env_poll_timeout_ms),
            max_events=n_envs,
        )
        if not ready:
            if time.time() - t0 > timeout_sec:
                raise RuntimeError(
                    f"asyncqueue batch step timed out; pending_envs={sorted(pending)}"
                )
            continue
        for item in ready:
            env_id = int(item["env_id"])
            if env_id not in pending:
                continue
            next_obs[env_id] = np.asarray(item["next_obs"], dtype=np.float32)
            rewards[env_id] = float(item["reward"])
            dones[env_id] = bool(item["done"])
            infos[env_id] = dict(item.get("info", {}))
            pending.remove(env_id)
    return next_obs, rewards, dones, infos


def _run_online_greedy_eval(
    *,
    eval_venv,
    sac: SMDPDiscreteSAC,
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
    sac: SMDPDiscreteSAC,
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


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", type=str, required=True, help="robomimic Adroit Diffusion Policy ckpt (.pth)"
    )
    parser.add_argument("--env_name", type=str, default="AdroitHandDoor-v1")
    parser.add_argument("--obs_key", type=str, default="flat")
    parser.add_argument("--output", type=str, default="./adaptive_rl_results")
    parser.add_argument(
        "--log_dir", type=str, default="./adaptive_rl_results/tb_logs/adroit_exrl_sac"
    )
    parser.add_argument("--checkpoint_root", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Existing SAC run dir to resume from latest_model.pt",
    )
    parser.add_argument("--resume_ckpt_name", type=str, default="latest_model.pt")

    parser.add_argument(
        "--stats_npz_override",
        type=str,
        default=None,
        help="Optional precomputed Adroit AAC stats npz",
    )
    parser.add_argument("--max_chunk_len", type=int, default=32)
    parser.add_argument(
        "--min_chunk_len",
        type=int,
        default=1,
        help="Minimum legal chunk length in the action space. Use 0 to expose a zero-length resample action.",
    )
    parser.add_argument("--env_horizon_override", type=int, default=200)
    parser.add_argument("--action_history_len", type=int, default=0)
    parser.add_argument("--resample_penalty", type=float, default=-5.0)
    parser.add_argument("--sparse_step_reward", type=float, default=-1.0)
    parser.add_argument("--sparse_success_reward", type=float, default=0.0)
    parser.add_argument(
        "--zero_action_fail_streak",
        type=int,
        default=0,
        help="If > 0, terminate the episode as failure after this many consecutive k=0 actions.",
    )
    parser.add_argument(
        "--zero_action_fail_reward",
        type=float,
        default=-1000.0,
        help="Terminal reward used when --zero_action_fail_streak triggers. Should be strongly negative to avoid becoming a shortcut.",
    )
    parser.add_argument("--reward_offset", type=float, default=1.0)
    parser.add_argument(
        "--terminate_on_success",
        dest="terminate_on_success",
        action="store_true",
        default=True,
        help="End episode immediately when info['is_success']=True. Default on for Adroit sparse-success semantics.",
    )
    parser.add_argument(
        "--no_terminate_on_success",
        dest="terminate_on_success",
        action="store_false",
        help="Disable success-terminal semantics and keep rolling after success.",
    )
    parser.add_argument(
        "--timeout_on_horizon",
        action="store_true",
        help="Mark horizon-end as timeout/truncation so off-policy backup bootstraps through horizon.",
    )
    parser.add_argument(
        "--reward_duration_mode",
        choices=["sum", "mean"],
        default="sum",
        help="SAC backup reward mode: sum=original chunk return, mean=chunk return divided by exec_len.",
    )
    parser.add_argument(
        "--transition_weight_mode",
        choices=["none", "duration", "sqrt_duration"],
        default="none",
        help="Optional loss weighting for replay transitions; use to counter decision-count bias without changing reward target.",
    )

    parser.add_argument("--total_policy_steps", type=int, default=100_000_000)
    parser.add_argument("--warmup_policy_steps", type=int, default=512)
    parser.add_argument(
        "--warmup_action_mode", choices=["uniform", "fixed", "mixed"], default="uniform"
    )
    parser.add_argument("--warmup_fixed_k", type=int, default=12)
    parser.add_argument("--warmup_mixed_uniform_prob", type=float, default=0.3)
    parser.add_argument(
        "--epsilon_greedy_prob",
        type=float,
        default=0.0,
        help="After warmup, replace the policy-selected chunk size with a uniformly random legal chunk size with probability epsilon.",
    )
    parser.add_argument(
        "--mask_illegal_actions_with_floor",
        action="store_true",
        help="While min-k floor is active, sample and update only over legal actions >= floor. Default off preserves prior behavior.",
    )
    parser.add_argument(
        "--floor_support_mask_mode",
        choices=["none", "sample_only", "update_only", "all"],
        default="none",
        help="More precise floor-support masking. 'all' matches --mask_illegal_actions_with_floor; 'sample_only' is a lighter early-phase intervention.",
    )
    parser.add_argument(
        "--learning_starts_policy_steps",
        type=int,
        default=-1,
        help="If > 0, delay SAC updates until this policy-step count (can be > warmup_policy_steps).",
    )
    parser.add_argument(
        "--probe_model_k_during_warmup",
        action="store_true",
        help="Log model-proposed k (diagnostic) during warmup without affecting executed actions.",
    )
    parser.add_argument(
        "--warmup_probe_deterministic",
        action="store_true",
        help="Use deterministic actor for warmup probe logging (default stochastic).",
    )
    parser.add_argument("--post_warmup_min_k", type=int, default=1)
    parser.add_argument("--post_warmup_min_k_until_policy_steps", type=int, default=0)
    parser.add_argument(
        "--post_warmup_min_k_schedule",
        type=str,
        default="",
        help="Optional piecewise min-k schedule, e.g. '6000:6,30000:5,50000:4,70000:3,90000:2,110000:1'.",
    )
    parser.add_argument(
        "--post_warmup_min_k_schedule_axis",
        choices=["policy", "env"],
        default="policy",
        help="Axis for explicit min-k schedule. Default keeps old behavior on policy-steps; env aligns release timing to env-step AUC.",
    )
    parser.add_argument(
        "--post_warmup_behavior_mix_schedule",
        type=str,
        default="",
        help="Optional policy/uniform behavior-mix schedule, e.g. '4000:0.7,16000:0.5,32000:0.3,60000:0.1,100000:0.0'. "
        "Only affects early data collection; empty means disabled.",
    )
    parser.add_argument(
        "--post_warmup_actor_prior_schedule",
        type=str,
        default="",
        help="Optional early actor-prior schedule, e.g. '4000:4.0,16000:3.0,30000:2.0,45000:1.0,60000:0.5,80000:0.0'. "
        "Adds a decaying KL-style prior toward larger-k support only in the early phase; empty means disabled.",
    )
    parser.add_argument(
        "--dynamic_floor_release_metric",
        choices=["none", "critic_loss", "actor_loss", "q_argmax_k_mean", "success_rate_recent"],
        default="none",
        help="Optional one-way trigger that permanently releases min-k floor to 1 once the chosen metric crosses threshold.",
    )
    parser.add_argument(
        "--dynamic_floor_release_threshold",
        type=float,
        default=0.0,
        help="Threshold for --dynamic_floor_release_metric; trigger fires when metric >= threshold.",
    )
    parser.add_argument(
        "--dynamic_floor_release_after_policy_steps",
        type=int,
        default=0,
        help="Do not allow dynamic floor release before this policy-step count.",
    )
    parser.add_argument(
        "--dynamic_floor_release_target_schedule",
        type=str,
        default="",
        help="Optional schedule to use after dynamic release is triggered. Empty means force floor to 1 permanently.",
    )
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument(
        "--train_freq_transitions",
        type=int,
        default=0,
        help="If > 0, trigger SAC updates every N decision-transitions collected, making update intensity less sensitive to n_env.",
    )
    parser.add_argument("--utd", type=int, default=4)
    parser.add_argument(
        "--utd_schedule",
        type=str,
        default="",
        help="Optional piecewise UTD schedule, e.g. '0:2,400000:6,600000:8'. Empty keeps fixed --utd.",
    )
    parser.add_argument(
        "--utd_schedule_axis",
        choices=["policy", "env"],
        default="env",
        help="Axis for --utd_schedule. Default env is better for fly-up phase alignment.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--replay_size", type=int, default=1_000_000)
    parser.add_argument(
        "--pipeline",
        choices=["online_sac", "offline_critic_warmstart_sac"],
        default="offline_critic_warmstart_sac",
        help="Training pipeline. Default uses Adroit door offline warmstart.",
    )
    parser.add_argument(
        "--offline_phase0_policy_steps",
        type=int,
        default=512,
        help="Uniform-random chunk collection steps before critic-only warmstart. Negative uses --warmup_policy_steps.",
    )
    parser.add_argument(
        "--offline_phase1_critic_updates",
        type=int,
        default=5000,
        help="Offline update count after phase0 replay collection.",
    )
    parser.add_argument(
        "--offline_phase1_actor_critic",
        dest="offline_phase1_actor_critic",
        action="store_true",
        default=True,
        help="Update actor and alpha during phase1, using normal SAC actor soft-value targets. Default on for Adroit ExRL-SAC.",
    )
    parser.add_argument(
        "--offline_phase1_critic_only",
        dest="offline_phase1_actor_critic",
        action="store_false",
        help="Disable actor/alpha updates during phase1 and use critic-only q_greedy warmstart.",
    )
    parser.add_argument(
        "--offline_phase2_actor_updates",
        type=int,
        default=0,
        help="Optional actor-only updates after offline critic warmstart and before online SAC handoff.",
    )
    parser.add_argument(
        "--offline_phase2_update_alpha",
        action="store_true",
        help="Also update SAC alpha during optional phase2 actor-only offline updates.",
    )
    parser.add_argument(
        "--offline_eval_every_updates",
        type=int,
        default=0,
        help="Run offline eval every N phase1/phase2 updates. Set <=0 to evaluate only at phase end.",
    )
    parser.add_argument("--offline_eval_episodes", type=int, default=50)
    parser.add_argument("--offline_eval_max_decisions", type=int, default=20000)
    parser.add_argument(
        "--offline_eval_policy_mode", choices=["q_greedy", "q_soft"], default="q_greedy"
    )
    parser.add_argument(
        "--offline_eval_critic_reduce", choices=["min", "mean", "q1", "q2"], default="mean"
    )
    parser.add_argument(
        "--n_step",
        type=int,
        default=1,
        help="Duration-aware SMDP n-step target. Default 1 preserves existing one-step behavior.",
    )
    parser.add_argument(
        "--td_target_next_obs_samples",
        type=int,
        default=1,
        help="Number of terminal RL obs samples for timeout target compatibility; 1 preserves existing SAC behavior.",
    )
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--actor_learning_starts_policy_steps",
        type=int,
        default=-1,
        help="If > 0, delay actor updates until this policy-step count; critic can still update earlier.",
    )
    parser.add_argument(
        "--alpha_learning_starts_policy_steps",
        type=int,
        default=-1,
        help="If > 0, delay alpha updates until this policy-step count; useful to prevent early entropy collapse.",
    )

    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument(
        "--vecenv_start_method", choices=["spawn", "forkserver", "fork"], default="forkserver"
    )
    parser.add_argument("--num_workers_cpu_affinity", type=str, default="")
    parser.add_argument(
        "--critic_obs_state_mode",
        choices=["stats_clip", "dp_raw"],
        default="dp_raw",
        help="Shared SAC actor/critic state representation. dp_raw matches checkpoint-normalized Adroit DP input without clipping.",
    )

    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor_lr", type=float, default=5e-5)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=1e-5)
    parser.add_argument(
        "--init_alpha",
        type=float,
        default=0.3,
        help="Initial entropy coefficient before alpha updates; default preserves prior behavior.",
    )
    parser.add_argument(
        "--fixed_alpha",
        type=float,
        default=None,
        help="If set, freeze alpha to this value and skip alpha optimization entirely.",
    )
    parser.add_argument(
        "--max_alpha",
        type=float,
        default=None,
        help="Optional hard cap on the learned alpha. When unset, alpha remains uncapped.",
    )
    parser.add_argument("--target_entropy_scale", type=float, default=0.98)
    parser.add_argument("--hidden_dims", type=str, default="512,512,512")
    parser.add_argument("--network_activation", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--actor_use_layernorm", action="store_true")
    parser.add_argument("--critic_use_layernorm", action="store_true")
    parser.add_argument("--critic_backup_temp", type=float, default=1.0)
    parser.add_argument("--cql_alpha", type=float, default=0.01)
    parser.add_argument("--cql_temp", type=float, default=1.0)
    parser.add_argument("--critic_ensemble_size", type=int, default=10)
    parser.add_argument("--target_critic_subsample_size", type=int, default=2)
    parser.add_argument("--critic_bootstrap_mask_prob", type=float, default=1.0)
    parser.add_argument(
        "--target_q_clip_min",
        type=float,
        default=-1000.0,
        help="Optional lower clamp for TD target_q. Unset disables lower clamp.",
    )
    parser.add_argument(
        "--target_q_clip_max",
        type=float,
        default=0.0,
        help="Optional upper clamp for TD target_q. Unset disables upper clamp.",
    )
    parser.add_argument(
        "--actor_target_critic_subsample",
        action="store_true",
        help=(
            "When SAC online target_policy_mode=actor, subsample target critic heads "
            "before the ensemble-min target, using --target_critic_subsample_size."
        ),
    )
    parser.add_argument(
        "--actor_loss_critic_subsample",
        action="store_true",
        help=(
            "When updating the SAC actor, subsample current critic heads before "
            "the ensemble-min actor objective, using --target_critic_subsample_size."
        ),
    )

    parser.add_argument("--decision_runtime", choices=["sync", "asyncqueue"], default="sync")
    parser.add_argument("--dp_infer_steps", type=int, default=5)
    parser.add_argument("--dp_sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--dp_amp", action="store_true")
    parser.add_argument("--dp_compile", action="store_true")
    parser.add_argument("--dp_infer_parallel", action="store_true")
    parser.add_argument("--dp_infer_gpus", type=str, default="0")
    parser.add_argument(
        "--dp_grouping", choices=["round_robin", "contiguous"], default="round_robin"
    )
    parser.add_argument("--ipc_chunk_dtype", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument(
        "--ipc_send_mode", choices=["full", "executed_only"], default="executed_only"
    )

    parser.add_argument("--infer_microbatch_max", type=int, default=64)
    parser.add_argument("--infer_microbatch_timeout_ms", type=int, default=3)
    parser.add_argument("--infer_queue_size", type=int, default=4096)
    parser.add_argument("--infer_result_queue_size", type=int, default=4096)
    parser.add_argument("--env_poll_timeout_ms", type=int, default=5)
    parser.add_argument("--ready_queue_size", type=int, default=8192)
    parser.add_argument("--max_inflight_per_env", type=int, default=1)
    parser.add_argument("--async_rollout_mode", type=str, default="strict_on_policy")

    parser.add_argument("--log_every_policy_steps", type=int, default=2048)
    parser.add_argument(
        "--online_eval_every_env_steps",
        type=int,
        default=0,
        help="Run greedy actor evaluation every N env steps. Set <=0 to disable.",
    )
    parser.add_argument(
        "--online_eval_episodes",
        type=int,
        default=50,
        help="Number of episodes for each online greedy actor evaluation.",
    )
    parser.add_argument(
        "--online_eval_n_envs",
        type=int,
        default=10,
        help="Parallel eval env count for online evaluation. Values <=0 use min(train n_envs, episodes).",
    )
    parser.add_argument(
        "--online_eval_max_decisions_per_episode",
        type=int,
        default=1000,
        help="Safety cap for greedy eval decisions per episode, mainly to avoid infinite k=0 loops.",
    )
    parser.add_argument(
        "--online_eval_at_handoff",
        dest="online_eval_at_handoff",
        action="store_true",
        default=True,
        help="Run an additional online eval when offline warmstart hands off to online SAC.",
    )
    parser.add_argument(
        "--no_online_eval_at_handoff",
        dest="online_eval_at_handoff",
        action="store_false",
        help="Keep online evals on the 0,N,2N,... env-step schedule without a handoff eval.",
    )
    parser.add_argument(
        "--offline_eval_at_phase_end",
        dest="offline_eval_at_phase_end",
        action="store_true",
        default=True,
        help="Run an extra evaluation when each offline warmstart phase finishes.",
    )
    parser.add_argument(
        "--no_offline_eval_at_phase_end",
        dest="offline_eval_at_phase_end",
        action="store_false",
        help="Do not add offline phase-end evaluations outside the online env-step schedule.",
    )
    parser.add_argument(
        "--debug_cadence_metrics",
        action="store_true",
        help="Print and log collector/replay/update cadence counters for short smoke runs.",
    )
    parser.add_argument("--metrics_flush_every_logs", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=200_000)
    parser.add_argument("--latest_model_save_freq", type=int, default=200_000)

    parser.add_argument("--force_thread_limits", action="store_true")
    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument(
        "--smoke_only",
        action="store_true",
        help="Build ckpt/env/wrapper, print dimensions, then exit.",
    )
    parser.add_argument(
        "--disable_checkpoints",
        action="store_true",
        help="Skip torch checkpoint writes for tiny tests.",
    )
    return parser


def _validate_task_alignment(args):
    env_name = str(args.env_name).strip()
    if not env_name:
        raise ValueError("--env_name must be non-empty")
    if not env_name.startswith("AdroitHand"):
        raise ValueError(
            f"Adroit ExRL-SAC currently supports AdroitHand* envs only, got {args.env_name}"
        )
    if str(args.obs_key) != "flat":
        raise ValueError("Adroit ExRL-SAC currently expects --obs_key flat")
    if not os.path.exists(str(args.agent)):
        raise FileNotFoundError(f"agent checkpoint not found: {args.agent}")


def _set_runtime_env(args):
    if bool(args.disable_checkpoints):
        os.environ["AAC_DISABLE_CHECKPOINTS"] = "1"
    else:
        os.environ.pop("AAC_DISABLE_CHECKPOINTS", None)
    os.environ["OPT_MIN_CHUNK_LEN"] = str(int(args.min_chunk_len))
    os.environ["ROBOMIMIC_ENABLE_LANG"] = os.environ.get("ROBOMIMIC_ENABLE_LANG", "0")
    os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")
    os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")

    if args.force_thread_limits:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    if args.decision_runtime == "asyncqueue":
        os.environ["OPT_ASYNCQUEUE_INFER_MICROBATCH_MAX"] = str(args.infer_microbatch_max)
        os.environ["OPT_ASYNCQUEUE_INFER_MICROBATCH_TIMEOUT_MS"] = str(
            args.infer_microbatch_timeout_ms
        )
        os.environ["OPT_ASYNCQUEUE_INFER_QUEUE_SIZE"] = str(args.infer_queue_size)
        os.environ["OPT_ASYNCQUEUE_INFER_RESULT_QUEUE_SIZE"] = str(args.infer_result_queue_size)
        os.environ["OPT_ASYNCQUEUE_ENV_POLL_TIMEOUT_MS"] = str(args.env_poll_timeout_ms)
        os.environ["OPT_ASYNCQUEUE_READY_QUEUE_SIZE"] = str(args.ready_queue_size)
        os.environ["OPT_ASYNCQUEUE_MAX_INFLIGHT_PER_ENV"] = str(args.max_inflight_per_env)
        os.environ["OPT_ASYNCQUEUE_ROLLOUT_MODE"] = str(args.async_rollout_mode)


def _save_ckpt(path: str, payload: Dict):
    if os.environ.get("AAC_DISABLE_CHECKPOINTS") == "1":
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


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.num_workers_cpu_affinity:
        cpu_set = _parse_cpu_affinity(args.num_workers_cpu_affinity)
        os.sched_setaffinity(0, cpu_set)
        print(f"[SAC] set CPU affinity: total={len(cpu_set)}")

    _set_runtime_env(args)
    _validate_task_alignment(args)

    if args.resume:
        run_dir = args.resume
        if not os.path.exists(run_dir):
            raise FileNotFoundError(f"resume dir not found: {run_dir}")
    else:
        run_dir = _create_new_run_dir(args.log_dir)
    os.makedirs(run_dir, exist_ok=True)
    if args.checkpoint_root:
        ckpt_dir = os.path.join(
            args.checkpoint_root,
            os.path.basename(os.path.normpath(args.log_dir)),
            os.path.basename(os.path.normpath(run_dir)),
        )
    else:
        ckpt_dir = os.path.join(run_dir, "checkpoints")
    video_dir = os.path.join(run_dir, "videos")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    lock = _RunDirLock(run_dir)
    lock.acquire(mode="resume" if args.resume else "new")
    atexit.register(lock.release)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print(f"[SAC] device={device}")

    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=args.agent)
    shape_meta = ckpt_dict.get("shape_metadata", {})
    flat_shape = shape_meta.get("all_shapes", {}).get(str(args.obs_key))
    if flat_shape is None:
        raise KeyError(f"ckpt shape_metadata missing all_shapes[{args.obs_key!r}]")
    obs_phys_dim = int(np.prod(flat_shape))
    action_phys_dim = int(shape_meta.get("ac_dim", 0))
    if obs_phys_dim != 39:
        print(f"[SAC][WARN] expected Adroit flat obs dim 39, got {obs_phys_dim}")
    if action_phys_dim != 28:
        print(f"[SAC][WARN] expected Adroit action dim 28, got {action_phys_dim}")

    policy, _ = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=False)
    policy_meta = _configure_adroit_policy(
        policy,
        dp_sampler=args.dp_sampler,
        dp_infer_steps=args.dp_infer_steps,
    )

    env_meta = copy.deepcopy(ckpt_dict.get("env_metadata", {}))
    env_meta["env_name"] = str(args.env_name or env_meta.get("env_name", "AdroitHandDoor-v1"))
    env_meta.setdefault("env_kwargs", {})
    env_meta["env_kwargs"]["max_episode_steps"] = int(args.env_horizon_override)
    env_meta["env_kwargs"].pop("horizon", None)

    if args.stats_npz_override:
        stats_npz_path = str(args.stats_npz_override)
        if not os.path.exists(stats_npz_path):
            raise FileNotFoundError(f"stats npz not found: {stats_npz_path}")
    else:
        stats_npz_path = os.path.join(run_dir, "adroit_aac_stats_from_ckpt.npz")
        _write_adroit_stats_npz_from_ckpt(ckpt_dict, stats_npz_path, obs_key=args.obs_key)
    args.stats_npz_override = stats_npz_path
    os.environ["OPT_TOOLHANG_STATS_NPZ"] = str(stats_npz_path)

    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)

    git_commit, git_dirty = _git_info(REPO_ROOT)
    metadata = {
        "created_utc": datetime.utcnow().isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "agent_path": args.agent,
        "agent_sha256": _sha256(args.agent),
        "stats_path": stats_npz_path,
        "stats_sha256": _sha256(stats_npz_path),
        "obs_key": str(args.obs_key),
        "obs_phys_dim": int(obs_phys_dim),
        "action_phys_dim": int(action_phys_dim),
        "adroit_policy_meta": policy_meta,
        "env_meta": env_meta,
        "reward_contract": {
            "mode": "sparse_adroit",
            "step_reward": float(args.sparse_step_reward),
            "success_reward": float(args.sparse_success_reward),
            "resample_penalty": float(args.resample_penalty),
            "terminate_on_success": bool(args.terminate_on_success),
            "timeout_on_horizon": bool(args.timeout_on_horizon),
            "success_sources": ["env.is_success", "_is_success", "_check_success", "info success"],
            "reward_threshold_success": False,
        },
        "nozero": int(args.min_chunk_len) > 0,
        "args": vars(args),
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(
        f"[SAC][ADROIT] env={env_meta['env_name']} "
        f"horizon={env_meta['env_kwargs'].get('max_episode_steps')} "
        f"obs_dim={obs_phys_dim} action_dim={action_phys_dim} "
        f"stats={stats_npz_path} policy_meta={policy_meta}"
    )

    venv = _build_adroit_venv(
        args=args,
        policy=policy,
        ckpt_dict=ckpt_dict,
        device=device,
        env_meta=env_meta,
        stats_npz_path=stats_npz_path,
        n_envs=int(args.n_envs),
        seed=args.seed,
        runtime=str(args.decision_runtime),
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
        f"[SAC] obs_dim={obs_dim} action_dim_effective={action_dim} "
        f"action_dim_raw={action_dim_raw} action_shift={action_shift}"
    )
    if bool(args.smoke_only):
        print(
            f"[SAC][SMOKE] built Adroit env/runtime={args.decision_runtime} "
            f"obs_dim={obs_dim} action_dim={action_dim} "
            f"k_range=[{int(args.min_chunk_len)},{int(args.max_chunk_len)}]"
        )
        try:
            venv.close()
        except Exception:
            pass
        return
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
    if int(args.td_target_next_obs_samples) < 1:
        raise ValueError("--td_target_next_obs_samples must be >= 1")
    if not (int(args.min_chunk_len) <= int(args.post_warmup_min_k) <= int(args.max_chunk_len)):
        raise ValueError(
            f"--post_warmup_min_k must be in [{int(args.min_chunk_len)}, {args.max_chunk_len}]"
        )
    if int(args.learning_starts_policy_steps) > 0 and int(args.learning_starts_policy_steps) < int(
        args.warmup_policy_steps
    ):
        raise ValueError("--learning_starts_policy_steps must be >= warmup_policy_steps when set")
    if int(args.train_freq_transitions) < 0:
        raise ValueError("--train_freq_transitions must be >= 0")
    if int(args.online_eval_every_env_steps) < 0:
        raise ValueError("--online_eval_every_env_steps must be >= 0")
    if int(args.online_eval_every_env_steps) > 0:
        if int(args.online_eval_episodes) <= 0:
            raise ValueError("--online_eval_episodes must be > 0 when online eval is enabled")
        if int(args.online_eval_max_decisions_per_episode) <= 0:
            raise ValueError("--online_eval_max_decisions_per_episode must be > 0")
    if (
        args.target_q_clip_min is not None
        and args.target_q_clip_max is not None
        and float(args.target_q_clip_min) > float(args.target_q_clip_max)
    ):
        raise ValueError("--target_q_clip_min cannot exceed --target_q_clip_max")
    if str(args.pipeline) == "offline_critic_warmstart_sac":
        offline_phase0_steps = (
            int(args.offline_phase0_policy_steps)
            if int(args.offline_phase0_policy_steps) >= 0
            else int(args.warmup_policy_steps)
        )
        if offline_phase0_steps <= 0:
            raise ValueError("offline warmstart requires phase0 steps > 0")
        if int(args.offline_phase1_critic_updates) <= 0:
            raise ValueError("--offline_phase1_critic_updates must be > 0")
        if int(args.offline_phase2_actor_updates) < 0:
            raise ValueError("--offline_phase2_actor_updates must be >= 0")
        if int(args.offline_eval_every_updates) < 0:
            raise ValueError("--offline_eval_every_updates must be >= 0")
        if int(args.offline_eval_episodes) <= 0:
            raise ValueError("--offline_eval_episodes must be > 0")
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
        args.dynamic_floor_release_target_schedule, args.max_chunk_len, args.min_chunk_len
    )
    behavior_mix_schedule_items = _parse_float_schedule(
        args.post_warmup_behavior_mix_schedule, min_value=0.0, max_value=1.0
    )
    actor_prior_schedule_items = _parse_float_schedule(
        args.post_warmup_actor_prior_schedule, min_value=0.0, max_value=None
    )
    utd_schedule_axis = str(args.utd_schedule_axis)
    utd_schedule_items = _parse_int_schedule(args.utd_schedule, min_value=1, max_value=None)

    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    target_entropy = float(args.target_entropy_scale) * float(np.log(action_dim))
    sac = SMDPDiscreteSAC(
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
    history = {"timesteps": [], "timesteps_env": [], "timesteps_policy": [], "metrics": {}}
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
        state = torch.load(latest_path, map_location=device, weights_only=False)
        loaded_obs_state_mode = state.get("critic_obs_state_mode")
        if loaded_obs_state_mode is None:
            raise ValueError(
                "Resume checkpoint predates critic_obs_state_mode metadata; "
                "start a new run to avoid mixing clipped and dp_raw replay observations"
            )
        if str(loaded_obs_state_mode) != str(args.critic_obs_state_mode):
            raise ValueError(
                "Resume critic_obs_state_mode mismatch: "
                f"checkpoint={loaded_obs_state_mode}, current={args.critic_obs_state_mode}"
            )
        loaded_fixed_alpha = state["sac_state"].get("fixed_alpha")
        if loaded_fixed_alpha != args.fixed_alpha:
            raise ValueError(
                "Resume fixed_alpha mismatch: "
                f"checkpoint={loaded_fixed_alpha}, current={args.fixed_alpha}"
            )
        sac.load_state_dict(state["sac_state"])
        policy_steps = int(state.get("policy_steps", 0))
        env_steps = int(state.get("env_steps", 0))
        best_success = float(state.get("best_success", -1.0))
        dynamic_floor_release_latched = bool(state.get("dynamic_floor_release_latched", False))
        dynamic_floor_release_policy_step = state.get("dynamic_floor_release_policy_step")
        dynamic_floor_release_env_step = state.get("dynamic_floor_release_env_step")
        replay_state = state.get("replay_state")
        if replay_state is not None:
            replay.load_state_dict(replay_state)
        loaded_offline_phase_state = dict(state.get("offline_phase_state", {}))
        offline_phase_state.update(loaded_offline_phase_state)
        step_calls = int(state.get("step_calls", 0))
        pending_transitions_for_update = int(state.get("pending_transitions_for_update", 0))
        num_emitted_transitions_total = int(state.get("num_emitted_transitions_total", 0))
        num_updates_total = int(state.get("num_updates_total", 0))
        num_done_for_backup_true = int(state.get("num_done_for_backup_true", 0))
        num_timeout_truncations = int(state.get("num_timeout_truncations", 0))
        duration_sum_total = float(state.get("duration_sum_total", 0.0))
        loaded_history, loaded_from = _load_json_with_backup(metrics_path, metrics_backup)
        if loaded_history is not None:
            history, trim_info = _trim_history_to_checkpoint(
                loaded_history,
                policy_steps=policy_steps,
                env_steps=env_steps,
            )
            if trim_info["trimmed"]:
                archived_metrics = _archive_metrics_before_resume(
                    (metrics_path, metrics_backup),
                    policy_steps=policy_steps,
                    env_steps=env_steps,
                )
                trim_msg = (
                    "[SAC][RESUME] trimmed metrics to checkpoint lineage "
                    f"online={trim_info['online_before']}->{trim_info['online_after']} "
                    f"eval={trim_info['eval_before']}->{trim_info['eval_after']} "
                    f"archives={archived_metrics}"
                )
                print(trim_msg, flush=True)
                _append_status_line(status_log_path, trim_msg)
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
        resume_msg = f"[SAC][RESUME] loaded metrics from {loaded_from}"
        print(resume_msg, flush=True)
        _append_status_line(status_log_path, resume_msg)
        if replay_state is not None:
            resume_msg = f"[SAC][RESUME] replay restored size={len(replay)}"
        else:
            resume_msg = (
                "[SAC][RESUME] model resumed; replay starts empty by design, "
                "warmup will refill before updates."
            )
            if str(args.pipeline) == "offline_critic_warmstart_sac":
                resume_msg = "[SAC][RESUME][WARN] offline checkpoint had no replay_state; restarting offline replay collection"
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
            "critic_obs_state_mode": str(args.critic_obs_state_mode),
            "critic_backup_temp": float(args.critic_backup_temp),
            "cql_alpha": float(args.cql_alpha),
            "cql_temp": float(args.cql_temp),
            "critic_ensemble_size": int(args.critic_ensemble_size),
            "target_critic_subsample_size": int(args.target_critic_subsample_size),
            "critic_bootstrap_mask_prob": float(args.critic_bootstrap_mask_prob),
            "actor_target_critic_subsample": bool(args.actor_target_critic_subsample),
            "actor_loss_critic_subsample": bool(args.actor_loss_critic_subsample),
            "max_chunk_len": int(args.max_chunk_len),
            "min_chunk_len": int(args.min_chunk_len),
            "dynamic_floor_release_latched": bool(dynamic_floor_release_latched),
            "dynamic_floor_release_policy_step": dynamic_floor_release_policy_step,
            "dynamic_floor_release_env_step": dynamic_floor_release_env_step,
            "replay_state": replay.state_dict()
            if str(args.pipeline) == "offline_critic_warmstart_sac"
            else None,
            "offline_phase_state": dict(offline_phase_state),
            "step_calls": int(step_calls),
            "pending_transitions_for_update": int(pending_transitions_for_update),
            "num_emitted_transitions_total": int(num_emitted_transitions_total),
            "num_updates_total": int(num_updates_total),
            "num_done_for_backup_true": int(num_done_for_backup_true),
            "num_timeout_truncations": int(num_timeout_truncations),
            "duration_sum_total": float(duration_sum_total),
            "offline_phase1_actor_critic": bool(args.offline_phase1_actor_critic),
            "run_dir": run_dir,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    next_periodic_ckpt = _next_periodic_step(policy_steps, args.save_freq)
    next_latest_ckpt = _next_periodic_step(policy_steps, args.latest_model_save_freq)
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
        with _preserve_training_rng():
            online_eval_venv = _make_online_eval_venv(
                args=args,
                policy=policy,
                ckpt_dict=ckpt_dict,
                device=device,
                env_meta=env_meta,
                stats_npz_path=stats_npz_path,
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
            return
        if (
            (not force)
            and next_online_eval_env_step is not None
            and int(env_steps) < int(next_online_eval_env_step)
        ):
            return
        eval_t0 = time.time()
        with _preserve_training_rng():
            success_rate = _run_online_greedy_eval(
                eval_venv=online_eval_venv,
                sac=sac,
                action_shift=int(action_shift),
                episodes=int(args.online_eval_episodes),
                max_decisions_per_episode=int(args.online_eval_max_decisions_per_episode),
            )
        history.setdefault("env_step_eval", []).append(int(env_steps))
        _record(history, "env_step_eval/success_rate", float(success_rate))
        writer.add_scalar("env_step_eval/success_rate", float(success_rate), int(env_steps))
        try:
            writer.flush()
        except Exception:
            pass
        dump_metrics()
        eval_line = (
            f"[SAC][ONLINE-EVAL] env_steps={env_steps} "
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

    print("[SAC] start training loop", flush=True)
    _append_status_line(status_log_path, "[SAC] start training loop")
    try:
        offline_warmstart_complete = str(args.pipeline) == "offline_critic_warmstart_sac" and bool(
            offline_phase_state.get("complete", False)
        )
        if str(args.pipeline) == "online_sac":
            if online_eval_venv is not None and int(env_steps) == 0:
                maybe_run_online_eval(force=True)
        else:
            offline_phase0_target = (
                int(args.offline_phase0_policy_steps)
                if int(args.offline_phase0_policy_steps) >= 0
                else int(args.warmup_policy_steps)
            )
            offline_phase1_target = int(args.offline_phase1_critic_updates)
            offline_phase2_target = int(args.offline_phase2_actor_updates)
            offline_phase0_done = int(offline_phase_state.get("phase0_policy_steps", 0))
            offline_phase1_done = int(offline_phase_state.get("phase1_critic_updates", 0))
            offline_phase2_done = int(offline_phase_state.get("phase2_actor_updates", 0))
            offline_eval_interval = int(args.offline_eval_every_updates)
            phase1_actor_critic = bool(args.offline_phase1_actor_critic)
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
                line = "[SAC][OFFLINE-PRE-EVAL] running actor eval before phase0 warmstart at env_steps=0"
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
                _record(history, "offline/phase0_policy_steps", float(offline_phase0_done))
                _record(history, "offline/phase1_actor_critic", float(phase1_actor_critic))
                _record(history, "replay/size", float(len(replay)))
                writer.add_scalar("offline/phase", float(phase_id), int(env_steps))
                writer.add_scalar("offline/critic_updates", float(critic_updates), int(env_steps))
                writer.add_scalar("offline/actor_updates", float(actor_updates), int(env_steps))
                writer.add_scalar(
                    "offline/phase0_policy_steps", float(offline_phase0_done), int(env_steps)
                )
                writer.add_scalar(
                    "offline/phase1_actor_critic", float(phase1_actor_critic), int(env_steps)
                )
                writer.add_scalar("replay/size", float(len(replay)), int(env_steps))
                writer.add_scalar("diagnostic/min_k_floor", 0.0, int(env_steps))
                writer.add_scalar("diagnostic/actor_prior_coeff_recent", 0.0, int(env_steps))
                if success_rate is not None:
                    _record(history, "offline/eval_success_rate", float(success_rate))
                    _record(history, "env_step_eval/success_rate", float(success_rate))
                    writer.add_scalar(
                        "offline/eval_success_rate", float(success_rate), int(env_steps)
                    )
                    writer.add_scalar(
                        "env_step_eval/success_rate", float(success_rate), int(env_steps)
                    )
                dump_metrics()
                try:
                    writer.flush()
                except Exception:
                    pass

            if not offline_warmstart_complete and offline_phase0_done < offline_phase0_target:
                offline_phase_state["phase"] = 0
                line = (
                    f"[SAC][OFFLINE-PHASE0] collect uniform random chunks "
                    f"{offline_phase0_done}/{offline_phase0_target}"
                )
                print(line, flush=True)
                _append_status_line(status_log_path, line)
                phase0_next_log = offline_phase0_done + max(int(args.log_every_policy_steps), 1)
                while offline_phase0_done < offline_phase0_target:
                    actions = np.random.randint(0, action_dim, size=args.n_envs, dtype=np.int64)
                    next_obs, rewards, dones, infos = _step_train_venv(
                        venv=venv,
                        args=args,
                        actions=actions,
                        action_shift=int(action_shift),
                        obs_template=obs,
                    )
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
                    offline_phase0_done += int(args.n_envs)
                    offline_phase_state["phase0_policy_steps"] = int(offline_phase0_done)
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
                        _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
                        phase0_next_log += max(int(args.log_every_policy_steps), 1)

            if not offline_warmstart_complete and offline_phase0_done >= offline_phase0_target:
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

            if not offline_warmstart_complete and offline_phase1_done < offline_phase1_target:
                offline_phase_state["phase"] = 1
                phase1_label = (
                    "actor-critic actor target"
                    if phase1_actor_critic
                    else "critic-only q_greedy target"
                )
                line = (
                    f"[SAC][OFFLINE-PHASE1] {phase1_label} updates "
                    f"{offline_phase1_done}/{offline_phase1_target} "
                    f"update_actor={phase1_update_actor} "
                    f"update_alpha={phase1_update_alpha} "
                    f"target_policy_mode={phase1_target_policy_mode} "
                    f"cql_alpha={float(args.cql_alpha)}"
                )
                print(line, flush=True)
                _append_status_line(status_log_path, line)
                phase1_next_log = offline_phase1_done + max(int(args.log_every_policy_steps), 1)
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
                    offline_phase_state["phase1_critic_updates"] = int(offline_phase1_done)

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
                        if should_eval or (
                            offline_phase1_done >= offline_phase1_target
                            and bool(args.offline_eval_at_phase_end)
                        ):
                            if online_eval_venv is not None:
                                eval_t0 = time.time()
                                with _preserve_training_rng():
                                    if phase1_actor_critic:
                                        success_rate = _run_online_greedy_eval(
                                            eval_venv=online_eval_venv,
                                            sac=sac,
                                            action_shift=int(action_shift),
                                            episodes=int(args.offline_eval_episodes),
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
                                            episodes=int(args.offline_eval_episodes),
                                            max_decisions_per_episode=int(
                                                args.offline_eval_max_decisions
                                            ),
                                            policy_mode=str(args.offline_eval_policy_mode),
                                            critic_reduce=str(args.offline_eval_critic_reduce),
                                        )
                                        eval_policy_label = str(args.offline_eval_policy_mode)
                                eval_line = (
                                    f"[SAC][OFFLINE-EVAL] env_steps={env_steps} "
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
                        _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
                        if should_log:
                            line = (
                                f"[SAC][OFFLINE-PHASE1] critic_updates={offline_phase1_done}/{offline_phase1_target} "
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
                    f"[SAC][OFFLINE-PHASE2] actor-only updates "
                    f"{offline_phase2_done}/{offline_phase2_target} "
                    f"update_alpha={bool(args.offline_phase2_update_alpha)}"
                )
                print(line, flush=True)
                _append_status_line(status_log_path, line)
                phase2_next_log = offline_phase2_done + max(int(args.log_every_policy_steps), 1)
                while offline_phase2_done < offline_phase2_target:
                    m = sac.update(
                        replay.sample(int(args.batch_size)),
                        update_actor=True,
                        update_alpha=bool(args.offline_phase2_update_alpha),
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
                    offline_phase_state["phase2_actor_updates"] = int(offline_phase2_done)

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
                        if should_eval or (
                            offline_phase2_done >= offline_phase2_target
                            and bool(args.offline_eval_at_phase_end)
                        ):
                            if online_eval_venv is not None:
                                eval_t0 = time.time()
                                with _preserve_training_rng():
                                    success_rate = _run_online_greedy_eval(
                                        eval_venv=online_eval_venv,
                                        sac=sac,
                                        action_shift=int(action_shift),
                                        episodes=int(args.offline_eval_episodes),
                                        max_decisions_per_episode=int(
                                            args.offline_eval_max_decisions
                                        ),
                                    )
                                eval_line = (
                                    f"[SAC][OFFLINE-ACTOR-EVAL] env_steps={env_steps} "
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
                        _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
                        if should_log:
                            line = (
                                f"[SAC][OFFLINE-PHASE2] actor_updates={offline_phase2_done}/{offline_phase2_target} "
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
                actor_learning_starts = min(int(actor_learning_starts), int(policy_steps))
                alpha_learning_starts = min(int(alpha_learning_starts), int(policy_steps))
                next_log = int(policy_steps) + max(int(args.log_every_policy_steps), 1)
                offline_warmstart_complete = True
                line = (
                    f"[SAC][OFFLINE-HANDOFF] env reset; entering online SAC at "
                    f"policy_steps={policy_steps} env_steps={env_steps} replay={len(replay)} "
                    f"critic_updates={offline_phase1_done} actor_updates={offline_phase2_done}"
                )
                print(line, flush=True)
                _append_status_line(status_log_path, line)
                if online_eval_venv is not None and bool(args.online_eval_at_handoff):
                    maybe_run_online_eval(force=True)

            if offline_warmstart_complete:
                learning_starts = min(int(learning_starts), int(policy_steps))
                actor_learning_starts = min(int(actor_learning_starts), int(policy_steps))
                alpha_learning_starts = min(int(alpha_learning_starts), int(policy_steps))

        while policy_steps < int(args.total_policy_steps):
            model_probe_k = None
            current_min_k_floor = 1
            current_min_action_id = max(int(current_min_k_floor) - int(args.min_chunk_len), 0)
            dynamic_floor_signal = None
            in_online_warmup = not bool(offline_warmstart_complete) and policy_steps < int(
                args.warmup_policy_steps
            )
            if in_online_warmup and bool(args.probe_model_k_during_warmup):
                probe_actions = sac.select_action(
                    obs, deterministic=bool(args.warmup_probe_deterministic)
                )
                model_probe_k = float(
                    np.mean(probe_actions.astype(np.float32) + float(args.min_chunk_len))
                )
            if in_online_warmup:
                mode = str(args.warmup_action_mode)
                if mode == "uniform":
                    actions = np.random.randint(0, action_dim, size=args.n_envs, dtype=np.int64)
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
                    use_rand = np.random.rand(args.n_envs) < float(args.warmup_mixed_uniform_prob)
                    actions = np.where(use_rand, rand_actions, fixed_id).astype(np.int64)
                else:
                    raise RuntimeError(f"unknown warmup action mode: {mode}")
            else:
                schedule_counter = (
                    int(env_steps) if min_k_schedule_axis == "env" else int(policy_steps)
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
                    and int(policy_steps) >= int(args.dynamic_floor_release_after_policy_steps)
                    and dynamic_floor_signal is not None
                    and float(dynamic_floor_signal) >= float(args.dynamic_floor_release_threshold)
                ):
                    dynamic_floor_release_latched = True
                    dynamic_floor_release_policy_step = int(policy_steps)
                    dynamic_floor_release_env_step = int(env_steps)
                    release_msg = (
                        f"[SAC][FLOOR-RELEASE] metric={args.dynamic_floor_release_metric} "
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
                min_action_id = max(int(current_min_k_floor) - int(args.min_chunk_len), 0)
                current_min_action_id = int(min_action_id)
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=sac.device, dtype=torch.float32)
                    logits_t, probs_t, _ = sac.actor.distribution(obs_t)
                    raw_dist = torch.distributions.Categorical(logits=logits_t)
                    actions_raw = raw_dist.sample().detach().cpu().numpy().astype(np.int64)
                    # When min_chunk_len=0, floor=1 should already exclude k=0.
                    use_floor_mask_sample = (
                        floor_support_mask_mode in ("sample_only", "all") and min_action_id > 0
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
                        probs_np = sampling_probs_t.detach().cpu().numpy().astype(np.float64)
                        prior = np.zeros_like(probs_np, dtype=np.float64)
                        prior[:, min_action_id:] = 1.0 / float(action_dim - min_action_id)
                        mixed = (1.0 - float(behavior_mix_beta)) * probs_np + float(
                            behavior_mix_beta
                        ) * prior
                        mixed /= np.clip(mixed.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)
                        actions = np.asarray(
                            [np.random.choice(action_dim, p=row) for row in mixed],
                            dtype=np.int64,
                        )
                    elif use_floor_mask_sample:
                        masked_dist = torch.distributions.Categorical(probs=sampling_probs_t)
                        actions = masked_dist.sample().detach().cpu().numpy().astype(np.int64)
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
                        actions = np.where(use_random, random_actions, actions).astype(np.int64)
                        epsilon_apply_buffer.append(float(np.mean(use_random)))
                    else:
                        epsilon_apply_buffer.append(0.0)

                if min_action_id > 0:
                    actions = np.maximum(actions, min_action_id).astype(np.int64)
                floor_apply_buffer.append(float(np.mean(actions > actions_raw)))
                raw_exec_buffer.append(
                    float(np.mean(actions_raw.astype(np.float32) + float(args.min_chunk_len)))
                )
                behavior_mix_buffer.append(float(behavior_mix_beta))
            next_obs, rewards, dones, infos = _step_train_venv(
                venv=venv,
                args=args,
                actions=actions,
                action_shift=int(action_shift),
                obs_template=obs,
            )
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
            if policy_steps >= int(learning_starts) and len(replay) >= int(args.batch_size):
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
                utd_counter = int(env_steps) if utd_schedule_axis == "env" else int(policy_steps)
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

            if policy_steps >= next_log:
                elapsed = max(time.time() - start_time, 1e-6)
                success_recent = float(np.mean(success_buffer)) if len(success_buffer) > 0 else 0.0
                success_recent20 = (
                    float(np.mean(success_buffer_20)) if len(success_buffer_20) > 0 else 0.0
                )
                avg_exec = float(np.mean(exec_buffer)) if len(exec_buffer) > 0 else 0.0
                ep_rew_mean = (
                    float(np.mean(episode_return_buffer)) if len(episode_return_buffer) > 0 else 0.0
                )
                ep_len_mean = (
                    float(np.mean(episode_len_buffer)) if len(episode_len_buffer) > 0 else 0.0
                )
                avg_duration_k_cum = float(duration_sum_total / max(int(policy_steps), 1))
                emitted_per_env_transition = float(
                    num_emitted_transitions_total / max(int(policy_steps), 1)
                )
                emitted_per_collector_step = float(
                    num_emitted_transitions_total / max(int(step_calls), 1)
                )
                updates_per_collector_step = float(num_updates_total / max(int(step_calls), 1))

                _record(history, "time/policy_steps", float(policy_steps))
                _record(history, "time/env_steps", float(env_steps))
                _record(history, "time/policy_steps_per_sec", float(policy_steps / elapsed))
                _record(history, "time/env_steps_per_sec", float(env_steps / elapsed))
                _record(history, "adaptive/success_rate_recent", success_recent)
                _record(history, "adaptive/success_rate_recent20", success_recent20)
                _record(history, "rollout/success_rate", success_recent)
                _record(history, "rollout/success_rate_recent20", success_recent20)
                _record(history, "rollout/ep_rew_mean", ep_rew_mean)
                _record(history, "rollout/ep_len_mean", ep_len_mean)
                _record(history, "adaptive/avg_exec_len", avg_exec)
                _record(history, "adaptive/success_rate_recent_env_step", success_recent)
                _record(history, "adaptive/success_rate_recent20_env_step", success_recent20)
                _record(history, "rollout/success_rate_env_step", success_recent)
                _record(history, "rollout/success_rate_recent20_env_step", success_recent20)
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
                _record(history, "diagnostic/epsilon_greedy_prob", float(args.epsilon_greedy_prob))
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
                            int(env_steps) if utd_schedule_axis == "env" else int(policy_steps),
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
                        history, "debug/num_done_for_backup_true", float(num_done_for_backup_true)
                    )
                    _record(
                        history, "debug/num_timeout_truncations", float(num_timeout_truncations)
                    )
                    _record(history, "debug/avg_duration_k_cum", avg_duration_k_cum)
                    _record(history, "debug/emitted_per_env_transition", emitted_per_env_transition)
                    _record(history, "debug/emitted_per_collector_step", emitted_per_collector_step)
                    _record(history, "debug/updates_per_collector_step", updates_per_collector_step)
                # Keep primary history x-axis on env-step to match TensorBoard.
                history["timesteps"].append(int(env_steps))
                history["timesteps_env"].append(int(env_steps))
                history["timesteps_policy"].append(int(policy_steps))

                writer.add_scalar("adaptive/success_rate_recent", success_recent, env_steps)
                writer.add_scalar("adaptive/success_rate_recent20", success_recent20, env_steps)
                writer.add_scalar("rollout/success_rate", success_recent, env_steps)
                writer.add_scalar("rollout/success_rate_recent20", success_recent20, env_steps)
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
                    "diagnostic/epsilon_greedy_prob", float(args.epsilon_greedy_prob), env_steps
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
                            int(env_steps) if utd_schedule_axis == "env" else int(policy_steps),
                            utd_schedule_items,
                            default=int(args.utd),
                        )
                    ),
                    env_steps,
                )
                writer.add_scalar("diagnostic/min_k_floor", float(current_min_k_floor), env_steps)
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
                    "time/policy_steps_per_sec", float(policy_steps / elapsed), env_steps
                )
                writer.add_scalar("time/env_steps_per_sec", float(env_steps / elapsed), env_steps)
                writer.add_scalar("replay/size", float(len(replay)), env_steps)
                if bool(args.debug_cadence_metrics):
                    writer.add_scalar("debug/collector_steps", float(step_calls), env_steps)
                    writer.add_scalar(
                        "debug/num_emitted_transitions",
                        float(num_emitted_transitions_total),
                        env_steps,
                    )
                    writer.add_scalar("debug/num_updates", float(num_updates_total), env_steps)
                    writer.add_scalar(
                        "debug/num_done_for_backup_true", float(num_done_for_backup_true), env_steps
                    )
                    writer.add_scalar(
                        "debug/num_timeout_truncations", float(num_timeout_truncations), env_steps
                    )
                    writer.add_scalar("debug/avg_duration_k_cum", avg_duration_k_cum, env_steps)
                    writer.add_scalar(
                        "debug/emitted_per_env_transition", emitted_per_env_transition, env_steps
                    )
                    writer.add_scalar(
                        "debug/emitted_per_collector_step", emitted_per_collector_step, env_steps
                    )
                    writer.add_scalar(
                        "debug/updates_per_collector_step", updates_per_collector_step, env_steps
                    )
                if model_probe_k is not None:
                    _record(history, "diagnostic/model_exec_len_probe_warmup", model_probe_k)
                    writer.add_scalar(
                        "diagnostic/model_exec_len_probe_warmup", model_probe_k, env_steps
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
                    f"[SAC] steps={policy_steps}/{args.total_policy_steps} "
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

                if success_recent > best_success and len(success_buffer) > 0:
                    best_success = success_recent
                    _save_ckpt(os.path.join(ckpt_dir, "best_model.pt"), pack_ckpt())

            if policy_steps >= next_periodic_ckpt:
                _save_ckpt(
                    os.path.join(ckpt_dir, f"adaptive_agent_{policy_steps}_steps.pt"), pack_ckpt()
                )
                next_periodic_ckpt += max(int(args.save_freq), 1)

            if policy_steps >= next_latest_ckpt:
                _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
                next_latest_ckpt += max(int(args.latest_model_save_freq), 1)

        _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
        dump_metrics()
        print("[SAC] training finished", flush=True)
        _append_status_line(status_log_path, "[SAC] training finished")
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
