"""
Adroit Door ExRL-Q training entry.

This keeps the current AAC_offline ExRL-Q algorithm path and swaps only the
base-policy / environment adapter to the Adroit Door DP setup.
"""

import argparse
import atexit
import copy
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

CURRENT_DIR = __import__("exrl.paths", fromlist=["ROOT"]).ROOT
REPO_ROOT = __import__("exrl.paths", fromlist=["ROOT"]).ROOT

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from exrl.adroit.q.policy_adapter import AdroitOnlineTeacherAdapter
from exrl.adroit.q.environment_adroit_q import BatchedAdaptiveWrapperAdroit
from exrl.adroit.q.online_bc import (
    OnlineSuccessBCReplay,
    RobomimicHDF5BCSampler,
    ema_update_module_,
)
from exrl.adroit.q.learning import (
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


def _truncate_metrics_history_to_checkpoint(
    history: Dict,
    *,
    checkpoint_env_steps: int,
    checkpoint_policy_steps: int,
    checkpoint_phase2_policy_steps: int = None,
) -> Dict[str, int]:
    """Keep the latest phase2 lineage up to the checkpoint being resumed."""
    env_axis = history.get("timesteps_env", [])
    policy_axis = history.get("timesteps_policy", [])
    if not isinstance(env_axis, list) or not isinstance(policy_axis, list):
        raise ValueError("Resume metrics history axes must be lists")
    if len(env_axis) != len(policy_axis):
        raise ValueError(
            "Resume metrics history axis length mismatch: "
            f"env={len(env_axis)} policy={len(policy_axis)}"
        )

    event_count = len(env_axis)
    metrics = history.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("Resume metrics history metrics must be a dict")
    phase2_axis = metrics.get("offline/phase2_policy_steps", [])
    if not isinstance(phase2_axis, list):
        raise ValueError("offline/phase2_policy_steps history must be a list")
    phase2_event_count = len(phase2_axis)
    if phase2_event_count <= 0:
        keep_event_count = event_count
        while keep_event_count > 0 and (
            int(env_axis[keep_event_count - 1]) > int(checkpoint_env_steps)
            or int(policy_axis[keep_event_count - 1]) > int(checkpoint_policy_steps)
        ):
            keep_event_count -= 1
        for key, values in list(history.items()):
            if key != "metrics" and key.startswith("timesteps") and isinstance(values, list):
                if len(values) == event_count:
                    history[key] = values[:keep_event_count]
        return {
            "event_count_before": int(event_count),
            "event_count_after": int(keep_event_count),
            "removed_events": int(event_count - keep_event_count),
            "phase2_event_count_before": 0,
            "removed_phase2_events": 0,
            "rollback_pruned_phase2_events": 0,
            "tail_pruned_phase2_events": 0,
            "truncated_metric_lists": 0,
        }

    if checkpoint_phase2_policy_steps is None:
        raise ValueError(
            "Checkpoint is missing phase2_policy_steps required for metrics truncation"
        )
    phase2_prefix_events = int(event_count - phase2_event_count)
    if phase2_prefix_events < 0:
        raise ValueError(
            "Phase2 metrics history is longer than the root event axis: "
            f"phase2={phase2_event_count} events={event_count}"
        )

    policy_offsets = {
        int(policy_axis[phase2_prefix_events + i]) - int(phase2_axis[i])
        for i in range(phase2_event_count)
    }
    if len(policy_offsets) != 1:
        raise ValueError(
            "Root metrics events are not one-to-one with phase2 events; "
            "cannot safely truncate sparse metrics"
        )

    # A prior resume can roll the x-axis back. Keep the latest branch by
    # removing older points that the new branch supersedes.
    canonical_phase2_indices = []
    for i, phase2_policy_step in enumerate(phase2_axis):
        while canonical_phase2_indices and float(
            phase2_axis[canonical_phase2_indices[-1]]
        ) >= float(phase2_policy_step):
            canonical_phase2_indices.pop()
        canonical_phase2_indices.append(i)
    rollback_pruned_phase2_events = int(phase2_event_count - len(canonical_phase2_indices))

    canonical_env_axis = [env_axis[phase2_prefix_events + i] for i in canonical_phase2_indices]
    canonical_policy_axis = [
        policy_axis[phase2_prefix_events + i] for i in canonical_phase2_indices
    ]
    for name, axis in (
        ("timesteps_env", canonical_env_axis),
        ("timesteps_policy", canonical_policy_axis),
    ):
        if any(float(axis[i]) < float(axis[i - 1]) for i in range(1, len(axis))):
            raise ValueError(f"Canonical resume metrics axis is not monotonic: {name}")

    kept_phase2_indices = []
    for i in canonical_phase2_indices:
        event_i = phase2_prefix_events + i
        if (
            int(env_axis[event_i]) <= int(checkpoint_env_steps)
            and int(policy_axis[event_i]) <= int(checkpoint_policy_steps)
            and int(phase2_axis[i]) <= int(checkpoint_phase2_policy_steps)
        ):
            kept_phase2_indices.append(i)
        else:
            break
    tail_pruned_phase2_events = int(len(canonical_phase2_indices) - len(kept_phase2_indices))
    removed_phase2_events = int(phase2_event_count - len(kept_phase2_indices))
    keep_event_count = int(phase2_prefix_events + len(kept_phase2_indices))
    removed_events = int(event_count - keep_event_count)

    def splice_phase2(values: list, prefix_len: int) -> list:
        return values[:prefix_len] + [values[prefix_len + i] for i in kept_phase2_indices]

    truncated_metric_lists = 0
    for key, values in list(metrics.items()):
        if not isinstance(values, list):
            continue
        has_phase2_tail = key.startswith(
            ("train/", "rollout/", "adaptive/", "offline/phase2_")
        ) or key in {
            "offline/phase",
            "offline/critic_updates",
            "time/env_steps",
            "time/policy_steps",
            "time/env_steps_per_sec",
            "time/policy_steps_per_sec",
            "diagnostic/current_utd",
            "diagnostic/current_ucb_beta",
            "replay/size",
        }
        if len(values) == event_count:
            metrics[key] = splice_phase2(values, phase2_prefix_events)
            new_len = len(metrics[key])
        elif has_phase2_tail:
            metric_prefix_len = int(len(values) - phase2_event_count)
            if metric_prefix_len < 0:
                raise ValueError(
                    f"Metric {key!r} is shorter than the phase2 history: "
                    f"values={len(values)} phase2={phase2_event_count}"
                )
            metrics[key] = splice_phase2(values, metric_prefix_len)
            new_len = len(metrics[key])
        else:
            continue
        if new_len < len(values):
            truncated_metric_lists += 1

    for key, values in list(history.items()):
        if key == "metrics" or not key.startswith("timesteps"):
            continue
        if isinstance(values, list) and len(values) == event_count:
            history[key] = splice_phase2(values, phase2_prefix_events)

    return {
        "event_count_before": int(event_count),
        "event_count_after": int(keep_event_count),
        "removed_events": int(removed_events),
        "phase2_event_count_before": int(phase2_event_count),
        "removed_phase2_events": int(removed_phase2_events),
        "rollback_pruned_phase2_events": int(rollback_pruned_phase2_events),
        "tail_pruned_phase2_events": int(tail_pruned_phase2_events),
        "truncated_metric_lists": int(truncated_metric_lists),
    }


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
    if int(args.post_warmup_min_k_until_policy_steps) > 0 and counter_value < int(
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


def _resolve_linear_float_schedule(
    counter_value: int,
    items: List[tuple],
    default: float = 0.0,
) -> float:
    """Piecewise-linear schedule with endpoint values held outside its range."""
    if not items:
        return float(default)
    counter = int(counter_value)
    if counter <= int(items[0][0]):
        return float(items[0][1])
    for (left_step, left_value), (right_step, right_value) in zip(items, items[1:]):
        left_step = int(left_step)
        right_step = int(right_step)
        if right_step <= left_step:
            raise ValueError("Linear schedule steps must be strictly increasing")
        if counter <= right_step:
            fraction = float(counter - left_step) / float(right_step - left_step)
            return float(left_value) + fraction * (float(right_value) - float(left_value))
    return float(items[-1][1])


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


def _bc_enabled(args) -> bool:
    return str(getattr(args, "bc_finetune_mode", "off")) != "off"


def _unwrap_vec_env(venv):
    return getattr(venv, "venv", venv)


def _infer_adroit_bc_padded(ckpt_dict: Dict) -> bool:
    config = ckpt_dict.get("config", {})
    if isinstance(config, str):
        config = json.loads(config)
    train_config = config.get("train", {}) if isinstance(config, dict) else {}
    return bool(train_config.get("pad_seq_length", True))


def _validate_bc_args(args) -> None:
    mode = str(getattr(args, "bc_finetune_mode", "off"))
    if mode == "off":
        return
    if mode != "online_success":
        raise ValueError(f"unsupported Adroit --bc_finetune_mode: {mode}")
    if str(args.pipeline) != "offline_actorless_phase01":
        raise ValueError("Adroit online-success BC currently requires offline_actorless_phase01")
    if bool(args.dp_compile):
        raise ValueError("Adroit online-success BC requires --dp_compile to be disabled")
    if int(args.bc_updates_per_rl_update) < 0:
        raise ValueError("--bc_updates_per_rl_update must be >= 0")
    bc_linear_schedule = _parse_float_schedule(
        args.bc_updates_per_rl_update_linear_schedule,
        min_value=0.0,
        max_value=None,
    )
    if any(
        int(right[0]) <= int(left[0])
        for left, right in zip(bc_linear_schedule, bc_linear_schedule[1:])
    ):
        raise ValueError(
            "--bc_updates_per_rl_update_linear_schedule steps must be strictly increasing"
        )
    if int(args.bc_update_every_rl_updates) <= 0:
        raise ValueError("--bc_update_every_rl_updates must be > 0")
    if int(args.bc_start_after_steps) < 0:
        raise ValueError("--bc_start_after_steps must be >= 0")
    if int(args.bc_batch_size) <= 0:
        raise ValueError("--bc_batch_size must be > 0")
    if float(args.bc_lr) <= 0.0:
        raise ValueError("--bc_lr must be > 0")
    if int(args.bc_success_capacity) <= 0:
        raise ValueError("--bc_success_capacity must be > 0")
    if str(args.bc_online_sampling) not in ("window", "episode"):
        raise ValueError("--bc_online_sampling must be 'window' or 'episode'")
    offline_fraction = float(args.bc_offline_fraction)
    if not 0.0 <= offline_fraction <= 1.0:
        raise ValueError("--bc_offline_fraction must be in [0, 1]")
    offline_samples_exact = int(args.bc_batch_size) * offline_fraction
    if not np.isclose(
        offline_samples_exact,
        round(offline_samples_exact),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("--bc_offline_fraction must yield an integer offline sample count")
    if offline_fraction > 0.0:
        if not str(args.bc_offline_demo_hdf5).strip():
            raise ValueError("--bc_offline_demo_hdf5 is required when --bc_offline_fraction > 0")
        if not os.path.isfile(args.bc_offline_demo_hdf5):
            raise FileNotFoundError(f"offline BC demo file not found: {args.bc_offline_demo_hdf5}")
    if str(args.bc_offline_sampling) not in ("window", "episode"):
        raise ValueError("--bc_offline_sampling must be 'window' or 'episode'")
    if str(args.bc_offline_start_mode) not in ("immediate", "match_online"):
        raise ValueError("--bc_offline_start_mode must be 'immediate' or 'match_online'")
    if float(args.bc_teacher_constraint_weight) < 0.0:
        raise ValueError("--bc_teacher_constraint_weight must be >= 0")
    if not 0.0 <= float(args.bc_teacher_ema_decay) <= 1.0:
        raise ValueError("--bc_teacher_ema_decay must be in [0, 1]")


def _collect_bc_success_infos(
    infos,
    bc_replay: Optional[OnlineSuccessBCReplay],
) -> tuple:
    if bc_replay is None:
        return 0, 0
    episodes = 0
    windows = 0
    for info in infos or []:
        if not isinstance(info, dict):
            continue
        states = info.get("bc_success_episode_states")
        actions = info.get("bc_success_episode_actions")
        if states is None or actions is None:
            continue
        added = bc_replay.add_success_episode(states, actions)
        if added > 0:
            episodes += 1
            windows += int(added)
    return int(episodes), int(windows)


def _validate_offline_actorless_args(args, floor_support_mask_mode: str) -> None:
    if int(args.min_chunk_len) != 1:
        raise ValueError(
            "offline_actorless_phase01 currently requires --min_chunk_len 1 so k=0 is not exposed"
        )
    if bool(args.use_mc_lower_bound) and int(args.n_step) != 1:
        raise ValueError(
            "offline_actorless_phase01 with --use_mc_lower_bound requires --n_step 1; "
            "overlapping n-step replay transitions would make MC return-to-go double-count rewards"
        )
    if bool(args.use_mc_lower_bound) and float(args.cql_alpha) <= 0.0:
        raise ValueError(
            "--use_mc_lower_bound is applied inside CQL, so it requires --cql_alpha > 0"
        )
    if str(args.warmup_action_mode) != "uniform":
        raise ValueError(
            "offline_actorless_phase01 phase-0 must use pure random k: set --warmup_action_mode uniform"
        )
    if bool(args.mask_illegal_actions_with_floor) or str(floor_support_mask_mode) != "none":
        raise ValueError(
            "offline_actorless_phase01 removes floor/legal-mask regularization; use floor_support_mask_mode=none"
        )
    if str(args.post_warmup_min_k_schedule).strip():
        raise ValueError("offline_actorless_phase01 does not allow --post_warmup_min_k_schedule")
    if int(args.post_warmup_min_k_until_policy_steps) > 0:
        raise ValueError(
            "offline_actorless_phase01 does not allow --post_warmup_min_k_until_policy_steps"
        )
    if (
        str(args.dynamic_floor_release_metric) != "none"
        or str(args.dynamic_floor_release_target_schedule).strip()
    ):
        raise ValueError("offline_actorless_phase01 does not allow dynamic floor release")
    if str(args.post_warmup_actor_prior_schedule).strip():
        raise ValueError(
            "offline_actorless_phase01 has no actor; actor prior schedule must be empty"
        )
    if str(args.target_policy_mode) not in ("q_soft", "q_greedy"):
        raise ValueError(
            "offline_actorless_phase01 requires --target_policy_mode q_soft or q_greedy"
        )
    if int(args.offline_phase1_critic_updates) <= 0:
        raise ValueError("--offline_phase1_critic_updates must be > 0")
    if int(args.offline_phase2_policy_steps) < 0:
        raise ValueError("--offline_phase2_policy_steps must be >= 0")
    if int(args.offline_phase2_max_env_steps) < 0:
        raise ValueError("--offline_phase2_max_env_steps must be >= 0")
    if int(args.offline_env_eval_interval_steps) < 0:
        raise ValueError("--offline_env_eval_interval_steps must be >= 0")
    if int(args.offline_resume_save_interval_steps) < 0:
        raise ValueError("--offline_resume_save_interval_steps must be >= 0")
    if (
        str(args.best_checkpoint_metric) == "env_eval"
        and int(args.offline_env_eval_interval_steps) <= 0
        and not bool(args.offline_env_eval_at_start)
    ):
        raise ValueError(
            "--best_checkpoint_metric env_eval requires env-step eval or eval-at-start"
        )
    if not (0.0 <= float(args.offline_phase2_epsilon_greedy_prob) <= 1.0):
        raise ValueError("--offline_phase2_epsilon_greedy_prob must be in [0, 1]")
    if float(args.offline_phase2_ucb_beta) < 0.0:
        raise ValueError("--offline_phase2_ucb_beta must be >= 0")
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
    ckpt_dict,
    device,
    env_meta,
    stats_npz_path: str,
    n_envs: int,
    seed: int,
    collect_success_bc: bool = False,
):
    args_for_env = copy.copy(args)
    args_for_env.seed = int(seed)
    venv = BatchedAdaptiveWrapperAdroit(
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
        critic_obs_action_mode=str(args.critic_obs_action_mode),
        critic_obs_state_mode=str(args.critic_obs_state_mode),
        collect_success_bc=bool(collect_success_bc),
        args=args_for_env,
    )
    return VecMonitor(venv)


def _run_critic_policy_eval(
    *,
    venv,
    sac: SMDPDiscreteSAC,
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
        "offline_eval/critic_avg_exec_len": float(np.mean(exec_lens)) if exec_lens else 0.0,
        "offline_eval/critic_episodes": float(min(len(successes), int(episodes))),
        "offline_eval/critic_env_steps": float(eval_env_steps),
        "offline_eval/critic_decisions": float(decisions),
        "offline_eval/critic_reduce_id": float({"min": 0, "mean": 1}[str(critic_reduce)]),
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", type=str, required=True, help="robomimic Adroit Diffusion Policy ckpt (.pth)"
    )
    parser.add_argument("--env_name", type=str, default="AdroitHandDoor-v1")
    parser.add_argument("--obs_key", type=str, default="flat")
    parser.add_argument("--output", type=str, default="./adaptive_rl_results")
    parser.add_argument(
        "--log_dir", type=str, default="./adaptive_rl_results/tb_logs/adroit_door_exrl_q"
    )
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default=None,
        help="Optional external checkpoint root. When unset, checkpoints live under the run dir.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Existing SAC run dir to resume from latest_model.pt",
    )
    parser.add_argument("--resume_ckpt_name", type=str, default="latest_model.pt")

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
    parser.add_argument(
        "--terminate_on_success",
        dest="terminate_on_success",
        action="store_true",
        default=True,
        help="End episode immediately when info['is_success']=True. Default on for Adroit sparse success semantics.",
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
        default=True,
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
    parser.add_argument("--utd", type=int, default=1)
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
        choices=["online_sac", "offline_actorless_phase01"],
        default="offline_actorless_phase01",
        help="online_sac preserves existing AAC; offline_actorless_phase01 runs random phase-0 collection then critic-only phase-1.",
    )
    parser.add_argument(
        "--offline_phase0_policy_steps",
        type=int,
        default=-1,
        help="Phase-0 random collection policy steps. If <=0, reuse --warmup_policy_steps.",
    )
    parser.add_argument(
        "--offline_phase1_critic_updates",
        type=int,
        default=5_000,
        help="Number of critic-only updates after phase-0 collection.",
    )
    parser.add_argument(
        "--offline_phase2_policy_steps",
        type=int,
        default=100_000_000,
        help=(
            "If >0, continue after phase-1 with critic-policy rollout plus "
            "critic-only updates for this many policy decisions."
        ),
    )
    parser.add_argument(
        "--offline_phase2_max_env_steps",
        type=int,
        default=0,
        help=(
            "Optional absolute training env-step cap for phase-2. 0 disables the cap; "
            "when enabled, phase-2 stops at whichever limit is reached first."
        ),
    )
    parser.add_argument(
        "--offline_phase2_policy_mode",
        choices=["q_greedy", "q_soft", "q_ucb", "q_thompson"],
        default="q_greedy",
        help="How the critic selects behavior actions during phase-2 rollout.",
    )
    parser.add_argument(
        "--offline_phase2_critic_reduce",
        choices=["min", "mean"],
        default="mean",
        help="How phase-2 behavior reduces the critic ensemble before selecting k.",
    )
    parser.add_argument(
        "--offline_phase2_epsilon_greedy_prob",
        type=float,
        default=0.0,
        help="During phase-2, replace critic-selected k with uniform random legal k with this probability.",
    )
    parser.add_argument(
        "--offline_phase2_ucb_beta",
        type=float,
        default=0.0,
        help="UCB exploration beta for --offline_phase2_policy_mode q_ucb.",
    )
    parser.add_argument(
        "--offline_phase2_ucb_beta_schedule",
        type=str,
        default="",
        help="Env-step schedule for phase-2 q_ucb beta, e.g. '0:1.0,1000000:0.3,3000000:0.0'.",
    )
    parser.add_argument(
        "--offline_env_eval_interval_steps",
        type=int,
        default=30_000,
        help=(
            "Offline pipeline env-step eval interval on the training env-step x-axis. "
            "0 disables this eval track. Phase-1 critic-only updates do not advance this schedule."
        ),
    )
    parser.add_argument(
        "--offline_resume_save_interval_steps",
        type=int,
        default=0,
        help=(
            "Extra offline latest checkpoint interval for robust resume. If 0, phase0/2 "
            "reuse --offline_env_eval_interval_steps when enabled, and phase1 reuses "
            "--offline_eval_every_updates when enabled."
        ),
    )
    parser.add_argument(
        "--offline_env_eval_at_start",
        dest="offline_env_eval_at_start",
        action="store_true",
        default=True,
        help="Run env-step eval once at training env_step=0 before phase-0 random collection.",
    )
    parser.add_argument(
        "--no_offline_env_eval_at_start",
        dest="offline_env_eval_at_start",
        action="store_false",
        help="Disable the initial env_step=0 env-step eval.",
    )
    parser.add_argument(
        "--offline_eval_every_updates",
        type=int,
        default=1_000,
        help="Run critic-as-actor eval every this many phase-1 updates. 0 disables eval.",
    )
    parser.add_argument("--offline_eval_episodes", type=int, default=50)
    parser.add_argument(
        "--offline_eval_max_decisions",
        type=int,
        default=20000,
        help="Safety cap on critic-policy eval decision steps.",
    )
    parser.add_argument(
        "--offline_eval_policy_mode",
        choices=["q_greedy", "q_soft"],
        default="q_greedy",
        help="How critic chooses actions during phase-1 eval.",
    )
    parser.add_argument(
        "--offline_eval_critic_reduce",
        choices=["min", "mean"],
        default="mean",
        help="How eval reduces the critic ensemble before selecting k. PA-RL optimized eval uses mean-like ranking.",
    )
    parser.add_argument(
        "--offline_eval_n_envs",
        type=int,
        default=-1,
        help="Number of envs for the separate offline eval env. <=0 uses min(n_envs, offline_eval_episodes).",
    )
    parser.add_argument(
        "--offline_eval_seed_offset",
        type=int,
        default=10000,
        help="Seed offset for the separate offline eval env.",
    )
    parser.add_argument(
        "--n_step",
        type=int,
        default=1,
        help="Duration-aware SMDP n-step target. Default 1 preserves existing one-step behavior.",
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
        "--vecenv_start_method", choices=["spawn", "forkserver", "fork"], default="fork"
    )
    parser.add_argument("--num_workers_cpu_affinity", type=str, default="")

    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument(
        "--init_alpha",
        type=float,
        default=0.1,
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
    parser.add_argument(
        "--target_policy_mode",
        choices=["actor", "q_soft", "q_greedy"],
        default="q_greedy",
        help="SMDP target policy. Existing AAC uses actor; actorless offline uses q_soft/q_greedy.",
    )
    parser.add_argument("--critic_backup_temp", type=float, default=1.0)
    parser.add_argument("--cql_alpha", type=float, default=0.30)
    parser.add_argument("--cql_temp", type=float, default=1.0)
    parser.add_argument("--use_mc_lower_bound", action="store_true")
    parser.add_argument(
        "--mc_lower_bound_scope",
        choices=["all", "non_data", "data", "topk_non_data"],
        default="topk_non_data",
        help="Where MC lower bound is applied inside CQL. topk_non_data is closest to PA-RL CalQL.",
    )
    parser.add_argument(
        "--mc_lower_bound_topk",
        type=int,
        default=10,
        help="Top-k action count for --mc_lower_bound_scope topk_non_data.",
    )
    parser.add_argument("--critic_ensemble_size", type=int, default=10)
    parser.add_argument("--target_critic_subsample_size", type=int, default=2)
    parser.add_argument("--critic_bootstrap_mask_prob", type=float, default=1.0)
    parser.add_argument("--distributional_critic", action="store_true")
    parser.add_argument("--distributional_q_min", type=float, default=-300.0)
    parser.add_argument("--distributional_q_max", type=float, default=0.0)
    parser.add_argument("--distributional_num_bins", type=int, default=128)
    parser.add_argument(
        "--td_loss_type",
        choices=["mse", "huber"],
        default="mse",
        help="Critic TD regression loss. Huber is a simple guard against large TD explosions.",
    )
    parser.add_argument(
        "--td_huber_delta",
        type=float,
        default=10.0,
        help="Beta/delta for --td_loss_type huber.",
    )
    parser.add_argument(
        "--target_q_clip_min",
        type=float,
        default=None,
        help="Optional lower clamp for TD target_q. Unset disables lower clamp.",
    )
    parser.add_argument(
        "--target_q_clip_max",
        type=float,
        default=None,
        help="Optional upper clamp for TD target_q. Unset disables upper clamp.",
    )
    parser.add_argument(
        "--critic_q_upper_bound",
        type=float,
        default=None,
        help="Optional upper bound used by a soft Q-cap penalty on all critic outputs.",
    )
    parser.add_argument(
        "--critic_q_upper_penalty",
        type=float,
        default=0.0,
        help="Weight for relu(Q - critic_q_upper_bound)^2. 0 disables the Q-cap penalty.",
    )
    parser.add_argument(
        "--td_target_next_obs_samples",
        type=int,
        default=1,
        help=(
            "Number of independently sampled base-policy chunks used for TD target next-state averaging. "
            "1 preserves the old single-sample behavior."
        ),
    )
    parser.add_argument(
        "--critic_obs_unnormalize_dp_chunk",
        action="store_true",
        help=(
            "Only for critic/RL observations: map DSRL-normalized DP action chunks through "
            "normalization.npz action_min/action_max before applying action_mean/action_std. "
            "Execution chunks are unchanged."
        ),
    )
    parser.add_argument(
        "--critic_obs_action_mode",
        choices=["stats", "dp_raw"],
        default="stats",
        help=(
            "Action chunk representation used only inside critic/RL observations. "
            "For Adroit, execution uses denormalized actions, so dp_raw maps the execution "
            "chunk back exactly once to the normalized DP sampler output."
        ),
    )
    parser.add_argument(
        "--critic_obs_state_mode",
        choices=["stats_clip", "dp_raw"],
        default="stats_clip",
        help=(
            "State representation used only inside critic/RL observations. "
            "stats_clip=checkpoint-normalized flat state followed by clip[-10,10]; "
            "dp_raw=the checkpoint-normalized state consumed by the DP policy, without clipping."
        ),
    )
    parser.add_argument(
        "--bc_finetune_mode",
        choices=["off", "online_success"],
        default="off",
        help="Online-success diffusion BC mode. off preserves the original frozen-DP path.",
    )
    parser.add_argument("--bc_updates_per_rl_update", type=int, default=1)
    parser.add_argument(
        "--bc_updates_per_rl_update_linear_schedule",
        type=str,
        default="",
        help=(
            "Optional piecewise-linear target BC updates per eligible RL update, "
            "formatted as step:value pairs such as 0:5,600000:1. Schedule progress "
            "starts at the first eligible BC update and uses the axis selected by "
            "--bc_schedule_axis. Fractional targets are realized with a saved accumulator."
        ),
    )
    parser.add_argument(
        "--bc_update_every_rl_updates",
        type=int,
        default=2,
        help="Run one BC update group every N critic updates.",
    )
    parser.add_argument(
        "--bc_schedule_axis",
        choices=["policy", "env"],
        default="env",
        help="Counter used by the one-stage BC start threshold.",
    )
    parser.add_argument(
        "--bc_start_after_steps",
        type=int,
        default=0,
        help="Start threshold on --bc_schedule_axis. Formal one-stage uses 0.",
    )
    parser.add_argument("--bc_batch_size", type=int, default=256)
    parser.add_argument("--bc_lr", type=float, default=1e-5)
    parser.add_argument(
        "--bc_success_capacity",
        type=int,
        default=25_000,
        help="Successful physical-step BC replay capacity counted in sampleable windows.",
    )
    parser.add_argument(
        "--bc_online_sampling",
        choices=["window", "episode"],
        default="episode",
    )
    parser.add_argument(
        "--bc_offline_demo_hdf5",
        type=str,
        default="",
        help="Optional robomimic HDF5 demonstration source for BC updates.",
    )
    parser.add_argument(
        "--bc_offline_fraction",
        type=float,
        default=0.0,
        help="Fraction of each BC batch sampled from --bc_offline_demo_hdf5.",
    )
    parser.add_argument(
        "--bc_offline_sampling",
        choices=["window", "episode"],
        default="episode",
    )
    parser.add_argument(
        "--bc_offline_start_mode",
        choices=["immediate", "match_online"],
        default="match_online",
        help=(
            "immediate permits demo-only BC immediately; match_online waits until "
            "the online portion of the mixed batch can be sampled."
        ),
    )
    parser.add_argument(
        "--bc_teacher_constraint_weight",
        type=float,
        default=1.0,
        help="Function-space consistency weight against the online teacher.",
    )
    parser.add_argument(
        "--bc_teacher_ema_decay",
        type=float,
        default=1.0,
        help="Teacher EMA decay. Formal one-stage uses 1.0 for a frozen teacher.",
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

    parser.add_argument("--log_every_policy_steps", type=int, default=512)
    parser.add_argument(
        "--debug_cadence_metrics",
        action="store_true",
        help="Print and log collector/replay/update cadence counters for short smoke runs.",
    )
    parser.add_argument("--metrics_flush_every_logs", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=200_000)
    parser.add_argument("--latest_model_save_freq", type=int, default=200_000)
    parser.add_argument(
        "--best_checkpoint_metric",
        choices=["rollout", "env_eval"],
        default="rollout",
        help=(
            "Metric controlling best_model.pt. env_eval uses only scheduled "
            "env_step_eval/success_rate and avoids rollout-window checkpoint churn."
        ),
    )
    parser.add_argument("--disable_checkpoints", action="store_true")

    parser.add_argument("--force_thread_limits", action="store_true")
    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument(
        "--smoke_only",
        action="store_true",
        help="Build ckpt/env/wrapper, print dimensions, then exit.",
    )
    return parser


def _validate_task_alignment(args):
    env_name = str(args.env_name).strip()
    if not env_name:
        raise ValueError("--env_name must be non-empty")
    if not env_name.startswith("AdroitHand"):
        raise ValueError(f"Adroit-Q currently supports AdroitHand* envs only, got {args.env_name}")
    agent_path = str(args.agent)
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"agent checkpoint not found: {agent_path}")
    if str(args.obs_key) != "flat":
        raise ValueError("Adroit-Q currently expects --obs_key flat")
    if bool(args.critic_obs_unnormalize_dp_chunk):
        raise ValueError("Adroit-Q does not use DSRL action unnormalization")
    if str(args.critic_obs_action_mode) not in ("stats", "dp_raw"):
        raise ValueError(f"unsupported Adroit critic action mode: {args.critic_obs_action_mode}")
    if str(args.critic_obs_state_mode) not in ("stats_clip", "dp_raw"):
        raise ValueError(f"unsupported Adroit critic state mode: {args.critic_obs_state_mode}")


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
            f"[SAC][RESUME][WARN] failed to restore RNG state: {type(exc).__name__}: {exc}",
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
        print(f"[SAC] set CPU affinity: total={len(cpu_set)}")

    _set_runtime_env(args)
    _validate_task_alignment(args)
    _validate_bc_args(args)
    bc_mode = str(args.bc_finetune_mode)
    bc_is_enabled = _bc_enabled(args)

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
        "stats_path": None,
        "stats_sha256": None,
        "nozero": int(args.min_chunk_len) > 0,
        "args": vars(args),
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

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
        policy, dp_sampler=args.dp_sampler, dp_infer_steps=args.dp_infer_steps
    )

    env_meta = copy.deepcopy(ckpt_dict.get("env_metadata", {}))
    env_meta["env_name"] = str(args.env_name or env_meta.get("env_name", "AdroitHandDoor-v1"))
    env_meta.setdefault("env_kwargs", {})
    env_meta["env_kwargs"]["max_episode_steps"] = int(args.env_horizon_override)
    env_meta["env_kwargs"].pop("horizon", None)

    stats_npz_path = os.path.join(run_dir, "adroit_aac_stats_from_ckpt.npz")
    _write_adroit_stats_npz_from_ckpt(ckpt_dict, stats_npz_path, obs_key=args.obs_key)
    os.environ["OPT_TOOLHANG_STATS_NPZ"] = str(stats_npz_path)

    metadata["adroit_policy_meta"] = policy_meta
    metadata["env_meta"] = env_meta
    metadata["stats_path"] = stats_npz_path
    metadata["stats_sha256"] = _sha256(stats_npz_path)
    metadata["reward_contract"] = {
        "mode": "sparse_robomimic_style",
        "step_reward": float(args.sparse_step_reward),
        "success_reward": float(args.sparse_success_reward),
        "resample_penalty": float(args.resample_penalty),
        "success_sources": ["env.is_success", "_is_success", "_check_success", "info success"],
        "reward_threshold_success": False,
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(
        f"[SAC][ADROIT-Q] env={env_meta['env_name']} "
        f"horizon={env_meta['env_kwargs'].get('max_episode_steps')} "
        f"obs_dim={obs_phys_dim} action_dim={action_phys_dim} policy_meta={policy_meta}"
    )

    if args.decision_runtime != "sync":
        print(
            "[SAC][WARN] Adroit-Q currently runs with sync runtime; asyncqueue args are accepted but not used."
        )

    eval_venv = None
    venv = _build_task_venv(
        args=args,
        policy=policy,
        ckpt_dict=ckpt_dict,
        device=device,
        env_meta=env_meta,
        stats_npz_path=stats_npz_path,
        n_envs=int(args.n_envs),
        seed=int(args.seed),
        collect_success_bc=bool(bc_is_enabled),
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
            f"[ADROIT-Q-SMOKE] obs_dim={obs_dim} action_dim={action_dim} "
            f"k_range=[{int(args.min_chunk_len)},{int(args.max_chunk_len)}]"
        )
        try:
            venv.close()
        except Exception:
            pass
        return

    def ensure_eval_venv(*, allow_collector_fallback: bool = False):
        nonlocal eval_venv
        if eval_venv is not None:
            return eval_venv
        eval_requested = (
            int(args.offline_eval_every_updates) > 0
            or int(args.offline_env_eval_interval_steps) > 0
            or bool(args.offline_env_eval_at_start)
        )
        if str(args.pipeline) != "offline_actorless_phase01" or not eval_requested:
            raise RuntimeError("offline eval requested while offline eval is disabled")
        eval_n_envs = int(args.offline_eval_n_envs)
        if eval_n_envs <= 0:
            eval_n_envs = max(1, min(int(args.n_envs), int(args.offline_eval_episodes)))
        candidate = None
        try:
            candidate = _build_task_venv(
                args=args,
                policy=policy,
                ckpt_dict=ckpt_dict,
                device=device,
                env_meta=env_meta,
                stats_npz_path=stats_npz_path,
                n_envs=eval_n_envs,
                seed=int(args.seed) + int(args.offline_eval_seed_offset),
                collect_success_bc=False,
            )
            eval_action_dim_raw = int(candidate.action_space.n)
            if eval_action_dim_raw != action_dim_raw:
                raise RuntimeError(
                    f"Eval env action dim mismatch: eval_raw={eval_action_dim_raw}, train_raw={action_dim_raw}"
                )
            if bc_is_enabled:
                eval_wrapper = _unwrap_vec_env(candidate)
                if hasattr(eval_wrapper, "sync_policy_replicas_from_primary"):
                    eval_wrapper.sync_policy_replicas_from_primary()
                train_wrapper = _unwrap_vec_env(venv)
                train_wrapper.set_primary_policy_trainable(True)
                train_wrapper.sync_policy_replicas_from_primary()
            eval_venv = candidate
            print(f"[SAC][OFFLINE] built separate eval env n_envs={eval_n_envs}", flush=True)
            return eval_venv
        except Exception as exc:
            if candidate is not None:
                try:
                    candidate.close()
                except Exception:
                    pass
            if allow_collector_fallback and venv is not None and not bc_is_enabled:
                line = (
                    "[SAC][OFFLINE][WARN] separate eval env creation failed; "
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
    if int(args.learning_starts_policy_steps) > 0 and int(args.learning_starts_policy_steps) < int(
        args.warmup_policy_steps
    ):
        raise ValueError("--learning_starts_policy_steps must be >= warmup_policy_steps when set")
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
        args.dynamic_floor_release_target_schedule, args.max_chunk_len, args.min_chunk_len
    )
    behavior_mix_schedule_items = _parse_float_schedule(
        args.post_warmup_behavior_mix_schedule, min_value=0.0, max_value=1.0
    )
    actor_prior_schedule_items = _parse_float_schedule(
        args.post_warmup_actor_prior_schedule, min_value=0.0, max_value=None
    )
    phase2_ucb_beta_schedule_items = _parse_float_schedule(
        args.offline_phase2_ucb_beta_schedule, min_value=0.0, max_value=None
    )
    utd_schedule_axis = str(args.utd_schedule_axis)
    utd_schedule_items = _parse_int_schedule(args.utd_schedule, min_value=1, max_value=None)
    bc_updates_linear_schedule_items = _parse_float_schedule(
        args.bc_updates_per_rl_update_linear_schedule,
        min_value=0.0,
        max_value=None,
    )

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
        actor_enabled=(str(args.pipeline) != "offline_actorless_phase01"),
        target_policy_mode=(
            str(args.target_policy_mode)
            if str(args.pipeline) == "offline_actorless_phase01"
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
    history = {"timesteps": [], "timesteps_env": [], "timesteps_policy": [], "metrics": {}}
    metrics_path = os.path.join(run_dir, "training_metrics.json")
    metrics_backup = os.path.join(run_dir, "training_metrics.backup.json")
    status_log_path = os.path.join(run_dir, "status.log")
    offline_phase_state = {}

    bc_adapter = None
    bc_replay = None
    bc_offline_sampler = None
    bc_optimizer = None
    bc_teacher_policy = None
    bc_padded = False
    bc_update_count = 0
    bc_rl_update_seen_count = 0
    bc_update_credit = 0.0
    bc_update_schedule_origin = None
    if bc_is_enabled:
        train_wrapper = _unwrap_vec_env(venv)
        algo = train_wrapper.set_primary_policy_trainable(True)
        bc_adapter = AdroitOnlineTeacherAdapter(algo)
        bc_padded = _infer_adroit_bc_padded(ckpt_dict)
        if int(bc_adapter.horizon_steps) != int(args.max_chunk_len):
            raise ValueError(
                "Adroit online BC requires DP prediction horizon to match max_chunk_len: "
                f"dp={bc_adapter.horizon_steps} max_chunk_len={args.max_chunk_len}"
            )
        if int(bc_adapter.action_dim) != int(action_phys_dim):
            raise ValueError(
                f"Adroit online BC action dim mismatch: dp={bc_adapter.action_dim} "
                f"checkpoint={action_phys_dim}"
            )
        if int(bc_adapter.obs_dim) != int(obs_phys_dim):
            raise ValueError(
                f"Adroit online BC obs dim mismatch: dp={bc_adapter.obs_dim} "
                f"checkpoint={obs_phys_dim}"
            )
        bc_replay = OnlineSuccessBCReplay(
            capacity=int(args.bc_success_capacity),
            horizon_steps=int(bc_adapter.horizon_steps),
            action_dim=int(bc_adapter.action_dim),
            cond_steps=int(bc_adapter.cond_steps),
            obs_dim=int(bc_adapter.obs_dim),
            device=device,
            padded=bool(bc_padded),
        )
        if float(args.bc_offline_fraction) > 0.0:
            bc_offline_sampler = RobomimicHDF5BCSampler(
                dataset_path=str(args.bc_offline_demo_hdf5),
                obs_key=str(args.obs_key),
                obs_normalization_stats=ckpt_dict.get("obs_normalization_stats", {}),
                action_normalization_stats=ckpt_dict.get("action_normalization_stats", {}),
                horizon_steps=int(bc_adapter.horizon_steps),
                cond_steps=int(bc_adapter.cond_steps),
                obs_dim=int(bc_adapter.obs_dim),
                action_dim=int(bc_adapter.action_dim),
                device=device,
                padded=bool(bc_padded),
            )
        bc_optimizer = torch.optim.AdamW(
            bc_adapter.parameters(),
            lr=float(args.bc_lr),
            weight_decay=0.0,
        )
        if float(args.bc_teacher_constraint_weight) > 0.0:
            bc_teacher_policy = bc_adapter.frozen_teacher_copy()
        bc_adapter.set_eval(keep_trainable=True)
        train_wrapper.sync_policy_replicas_from_primary()
        line = (
            "[BC][ADROIT] enabled one-stage online-success BC "
            f"horizon={bc_adapter.horizon_steps} cond={bc_adapter.cond_steps} "
            f"obs={bc_adapter.obs_dim} action={bc_adapter.action_dim} "
            f"capacity={int(args.bc_success_capacity)} padded={int(bc_padded)} "
            f"sampling={args.bc_online_sampling} lr={float(args.bc_lr):.3g} "
            f"teacher_weight={float(args.bc_teacher_constraint_weight):.3f} "
            f"teacher_ema_decay={float(args.bc_teacher_ema_decay):.6f} "
            f"offline_fraction={float(args.bc_offline_fraction):.3f} "
            f"offline_start={args.bc_offline_start_mode} "
            f"offline_sampling={args.bc_offline_sampling}"
        )
        if bc_offline_sampler is not None:
            line += (
                f" offline_demos={bc_offline_sampler.n_episodes}"
                f" offline_transitions={bc_offline_sampler.n_transitions}"
                f" offline_windows={len(bc_offline_sampler)}"
                f" offline_path={bc_offline_sampler.dataset_path}"
            )
        print(line, flush=True)
        _append_status_line(status_log_path, line)

    if args.resume:
        latest_path = os.path.join(ckpt_dir, args.resume_ckpt_name)
        if not os.path.exists(latest_path):
            raise FileNotFoundError(f"resume checkpoint not found: {latest_path}")
        state = _load_ckpt(latest_path, map_location=device)
        checkpoint_bc_mode = str(state.get("bc_finetune_mode", "off"))
        if checkpoint_bc_mode != bc_mode:
            raise RuntimeError(
                f"BC resume mode mismatch: checkpoint={checkpoint_bc_mode} current={bc_mode}"
            )
        if bc_is_enabled:
            required_bc_state = (
                "bc_live_policy_state",
                "bc_optimizer_state",
                "bc_replay_state",
                "bc_updates_per_rl_update",
                "bc_update_every_rl_updates",
                "bc_batch_size",
                "bc_lr",
                "bc_padded",
            )
            missing_bc_state = [key for key in required_bc_state if state.get(key) is None]
            if missing_bc_state:
                raise RuntimeError(f"Adroit online-BC resume is missing state: {missing_bc_state}")
            checkpoint_sampling = str(state.get("bc_online_sampling", "window"))
            if checkpoint_sampling != str(args.bc_online_sampling):
                raise RuntimeError(
                    "BC online sampling mismatch: "
                    f"checkpoint={checkpoint_sampling} current={args.bc_online_sampling}"
                )
            checkpoint_offline_fraction = float(state.get("bc_offline_fraction", 0.0))
            if not np.isclose(
                checkpoint_offline_fraction,
                float(args.bc_offline_fraction),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "BC offline fraction mismatch: "
                    f"checkpoint={checkpoint_offline_fraction} "
                    f"current={float(args.bc_offline_fraction)}"
                )
            if str(state.get("bc_offline_sampling", "episode")) != str(args.bc_offline_sampling):
                raise RuntimeError("BC offline sampling mismatch on resume")
            if str(state.get("bc_offline_start_mode", "match_online")) != str(
                args.bc_offline_start_mode
            ):
                raise RuntimeError("BC offline start mode mismatch on resume")
            checkpoint_teacher_weight = float(state.get("bc_teacher_constraint_weight", 0.0))
            if not np.isclose(
                checkpoint_teacher_weight,
                float(args.bc_teacher_constraint_weight),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "BC teacher weight mismatch: "
                    f"checkpoint={checkpoint_teacher_weight} "
                    f"current={float(args.bc_teacher_constraint_weight)}"
                )
            checkpoint_teacher_decay = float(state.get("bc_teacher_ema_decay", 1.0))
            if not np.isclose(
                checkpoint_teacher_decay,
                float(args.bc_teacher_ema_decay),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "BC teacher EMA decay mismatch: "
                    f"checkpoint={checkpoint_teacher_decay} "
                    f"current={float(args.bc_teacher_ema_decay)}"
                )
            if int(state.get("bc_success_capacity", args.bc_success_capacity)) != int(
                args.bc_success_capacity
            ):
                raise RuntimeError(
                    "BC success capacity mismatch: "
                    f"checkpoint={state.get('bc_success_capacity')} "
                    f"current={int(args.bc_success_capacity)}"
                )
            if str(state.get("bc_schedule_axis", args.bc_schedule_axis)) != str(
                args.bc_schedule_axis
            ):
                raise RuntimeError("BC schedule axis mismatch on resume")
            if str(state.get("bc_updates_per_rl_update_linear_schedule", "")) != str(
                args.bc_updates_per_rl_update_linear_schedule
            ):
                raise RuntimeError("BC linear update schedule mismatch on resume")
            if int(state.get("bc_start_after_steps", args.bc_start_after_steps)) != int(
                args.bc_start_after_steps
            ):
                raise RuntimeError("BC start threshold mismatch on resume")
            resume_int_fields = (
                ("bc_updates_per_rl_update", int(args.bc_updates_per_rl_update)),
                (
                    "bc_update_every_rl_updates",
                    int(args.bc_update_every_rl_updates),
                ),
                ("bc_batch_size", int(args.bc_batch_size)),
            )
            for key, current_value in resume_int_fields:
                if int(state[key]) != int(current_value):
                    raise RuntimeError(
                        f"{key} mismatch on resume: "
                        f"checkpoint={int(state[key])} current={int(current_value)}"
                    )
            if not np.isclose(
                float(state["bc_lr"]),
                float(args.bc_lr),
                rtol=0.0,
                atol=1e-15,
            ):
                raise RuntimeError(
                    "bc_lr mismatch on resume: "
                    f"checkpoint={float(state['bc_lr'])} current={float(args.bc_lr)}"
                )
            if bool(state["bc_padded"]) != bool(bc_padded):
                raise RuntimeError(
                    "bc_padded mismatch on resume: "
                    f"checkpoint={bool(state['bc_padded'])} current={bool(bc_padded)}"
                )
            if (
                float(args.bc_teacher_constraint_weight) > 0.0
                and state.get("bc_teacher_state") is None
            ):
                raise RuntimeError("Adroit online-BC resume is missing frozen teacher state")
            bc_adapter.load_state_dict(state["bc_live_policy_state"], strict=True)
            bc_optimizer.load_state_dict(state["bc_optimizer_state"])
            bc_replay.load_state_dict(state["bc_replay_state"])
            if bc_teacher_policy is not None:
                bc_teacher_policy.load_state_dict(state["bc_teacher_state"], strict=True)
            bc_update_count = int(state.get("bc_update_count", 0))
            bc_rl_update_seen_count = int(state.get("bc_rl_update_seen_count", 0))
            bc_update_credit = float(state.get("bc_update_credit", 0.0))
            bc_update_schedule_origin = state.get("bc_update_schedule_origin")
            if bc_update_schedule_origin is not None:
                bc_update_schedule_origin = int(bc_update_schedule_origin)
            bc_adapter.set_eval(keep_trainable=True)
            train_wrapper = _unwrap_vec_env(venv)
            train_wrapper.sync_policy_replicas_from_primary()
            refreshed_obs = train_wrapper.refresh_current_policy_plan()
            if refreshed_obs is not None:
                obs = np.asarray(refreshed_obs, dtype=np.float32)
        sac.load_state_dict(state["sac_state"])
        if state.get("replay_state") is not None:
            replay.load_state_dict(state["replay_state"])
        elif str(args.pipeline) == "offline_actorless_phase01":
            resume_msg = "[SAC][RESUME][WARN] checkpoint has no replay_state; offline resume may not be valid"
            print(resume_msg, flush=True)
            _append_status_line(status_log_path, resume_msg)
        policy_steps = int(state.get("policy_steps", 0))
        env_steps = int(state.get("env_steps", 0))
        step_calls = int(state.get("step_calls", 0))
        pending_transitions_for_update = int(state.get("pending_transitions_for_update", 0))
        num_updates_total = int(state.get("num_updates_total", state.get("num_updates", 0)))
        num_emitted_transitions_total = int(state.get("num_emitted_transitions_total", 0))
        num_done_for_backup_true = int(state.get("num_done_for_backup_true", 0))
        num_timeout_truncations = int(state.get("num_timeout_truncations", 0))
        duration_sum_total = float(state.get("duration_sum_total", 0.0))
        best_success = float(state.get("best_success", -1.0))
        dynamic_floor_release_latched = bool(state.get("dynamic_floor_release_latched", False))
        dynamic_floor_release_policy_step = state.get("dynamic_floor_release_policy_step")
        dynamic_floor_release_env_step = state.get("dynamic_floor_release_env_step")
        offline_phase_state = dict(state.get("offline_phase_state") or {})
        if "phase_id" in state and "phase_id" not in offline_phase_state:
            offline_phase_state["phase_id"] = state.get("phase_id")
        _restore_rng_state(state.get("rng_state"))
        loaded_history, loaded_from = _load_json_with_backup(metrics_path, metrics_backup)
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
            truncation = _truncate_metrics_history_to_checkpoint(
                history,
                checkpoint_env_steps=env_steps,
                checkpoint_policy_steps=policy_steps,
                checkpoint_phase2_policy_steps=offline_phase_state.get("phase2_policy_steps"),
            )
            if truncation["removed_events"] > 0 or truncation["removed_phase2_events"] > 0:
                _atomic_json_dump(history, metrics_path, metrics_backup)
                truncate_msg = (
                    "[SAC][RESUME] truncated metrics history to checkpoint; "
                    f"events={truncation['event_count_before']}->{truncation['event_count_after']} "
                    f"phase2_removed={truncation['removed_phase2_events']} "
                    f"rollback_pruned={truncation['rollback_pruned_phase2_events']} "
                    f"tail_pruned={truncation['tail_pruned_phase2_events']} "
                    f"metric_lists={truncation['truncated_metric_lists']}"
                )
                print(truncate_msg, flush=True)
                _append_status_line(status_log_path, truncate_msg)
        resume_msg = f"[SAC][RESUME] loaded metrics from {loaded_from}"
        print(resume_msg, flush=True)
        _append_status_line(status_log_path, resume_msg)
        resume_msg = (
            f"[SAC][RESUME] model/replay resumed; policy_steps={policy_steps} "
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
                replay.state_dict() if str(args.pipeline) == "offline_actorless_phase01" else None
            ),
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
            "best_checkpoint_metric": str(args.best_checkpoint_metric),
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
            "offline_phase2_max_env_steps": int(args.offline_phase2_max_env_steps),
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
            "critic_obs_unnormalize_dp_chunk": bool(args.critic_obs_unnormalize_dp_chunk),
            "critic_obs_action_mode": str(args.critic_obs_action_mode),
            "critic_obs_state_mode": str(args.critic_obs_state_mode),
            "dynamic_floor_release_latched": bool(dynamic_floor_release_latched),
            "dynamic_floor_release_policy_step": dynamic_floor_release_policy_step,
            "dynamic_floor_release_env_step": dynamic_floor_release_env_step,
            "phase_id": offline_phase_state.get("phase_id"),
            "offline_phase_state": dict(offline_phase_state),
            "bc_finetune_mode": str(bc_mode),
            "bc_live_policy_state": (
                bc_adapter.state_dict() if bc_is_enabled and bc_adapter is not None else None
            ),
            "bc_optimizer_state": (
                bc_optimizer.state_dict() if bc_is_enabled and bc_optimizer is not None else None
            ),
            "bc_teacher_state": (
                bc_teacher_policy.state_dict()
                if bc_is_enabled and bc_teacher_policy is not None
                else None
            ),
            "bc_replay_state": (
                bc_replay.state_dict(copy_arrays=False, as_torch=True)
                if bc_is_enabled and bc_replay is not None
                else None
            ),
            "bc_update_count": int(bc_update_count),
            "bc_rl_update_seen_count": int(bc_rl_update_seen_count),
            "bc_update_credit": float(bc_update_credit),
            "bc_update_schedule_origin": bc_update_schedule_origin,
            "bc_updates_per_rl_update": int(args.bc_updates_per_rl_update),
            "bc_updates_per_rl_update_linear_schedule": str(
                args.bc_updates_per_rl_update_linear_schedule
            ),
            "bc_update_every_rl_updates": int(args.bc_update_every_rl_updates),
            "bc_batch_size": int(args.bc_batch_size),
            "bc_lr": float(args.bc_lr),
            "bc_success_capacity": int(args.bc_success_capacity),
            "bc_success_windows": int(len(bc_replay)) if bc_replay is not None else 0,
            "bc_success_episodes_added": (
                int(bc_replay.success_episodes_added) if bc_replay is not None else 0
            ),
            "bc_total_windows_added": (
                int(bc_replay.total_windows_added) if bc_replay is not None else 0
            ),
            "bc_online_sampling": str(args.bc_online_sampling),
            "bc_offline_demo_hdf5": str(args.bc_offline_demo_hdf5),
            "bc_offline_fraction": float(args.bc_offline_fraction),
            "bc_offline_sampling": str(args.bc_offline_sampling),
            "bc_offline_start_mode": str(args.bc_offline_start_mode),
            "bc_offline_episodes": (
                int(bc_offline_sampler.n_episodes) if bc_offline_sampler is not None else 0
            ),
            "bc_offline_transitions": (
                int(bc_offline_sampler.n_transitions) if bc_offline_sampler is not None else 0
            ),
            "bc_offline_windows": (
                int(len(bc_offline_sampler)) if bc_offline_sampler is not None else 0
            ),
            "bc_teacher_constraint_weight": float(args.bc_teacher_constraint_weight),
            "bc_teacher_ema_decay": float(args.bc_teacher_ema_decay),
            "bc_schedule_axis": str(args.bc_schedule_axis),
            "bc_start_after_steps": int(args.bc_start_after_steps),
            "bc_padded": bool(bc_padded),
            "run_dir": run_dir,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def run_bc_updates() -> int:
        nonlocal bc_update_count, bc_rl_update_seen_count, bc_update_credit
        nonlocal bc_update_schedule_origin, obs
        if not bc_is_enabled or bc_adapter is None or bc_optimizer is None or bc_replay is None:
            return 0
        schedule_counter = (
            int(env_steps) if str(args.bc_schedule_axis) == "env" else int(policy_steps)
        )
        if schedule_counter < int(args.bc_start_after_steps):
            train_meter["bc/skipped_start_delay"].append(1.0)
            return 0
        schedule_progress = schedule_counter
        if bc_updates_linear_schedule_items:
            if bc_update_schedule_origin is None:
                bc_update_schedule_origin = int(schedule_counter)
            schedule_progress = max(
                int(schedule_counter) - int(bc_update_schedule_origin),
                0,
            )
        target_updates_per_rl = _resolve_linear_float_schedule(
            schedule_progress,
            bc_updates_linear_schedule_items,
            default=float(args.bc_updates_per_rl_update),
        )
        if target_updates_per_rl <= 0.0:
            return 0

        bc_rl_update_seen_count += 1
        update_every = max(int(args.bc_update_every_rl_updates), 1)
        if (bc_rl_update_seen_count % update_every) != 0:
            train_meter["bc/skipped_update_interval"].append(1.0)
            return 0
        batch_size = int(args.bc_batch_size)
        offline_batch_size = int(round(batch_size * float(args.bc_offline_fraction)))
        offline_batch_size = min(max(offline_batch_size, 0), batch_size)
        online_batch_size = batch_size - offline_batch_size
        online_gate_size = online_batch_size
        if str(args.bc_offline_start_mode) == "match_online":
            online_gate_size = max(online_gate_size, batch_size)
        if offline_batch_size > 0 and bc_offline_sampler is None:
            raise RuntimeError("offline BC fraction is positive but demo sampler is missing")
        if online_gate_size > 0 and not bc_replay.can_sample(online_gate_size):
            train_meter["bc/skipped_insufficient_online"].append(1.0)
            train_meter["bc/success_windows"].append(float(len(bc_replay)))
            return 0

        bc_update_credit += float(target_updates_per_rl)
        planned_updates = int(np.floor(bc_update_credit + 1e-12))
        bc_update_credit -= float(planned_updates)
        completed = 0
        teacher_weight = float(args.bc_teacher_constraint_weight)
        for _ in range(planned_updates):
            if online_gate_size > 0 and not bc_replay.can_sample(online_gate_size):
                break
            bc_adapter.set_train()
            bc_optimizer.zero_grad(set_to_none=True)
            online_sampling = (
                "uniform_episode" if str(args.bc_online_sampling) == "episode" else "uniform_window"
            )
            demo_sampling = (
                "uniform_episode"
                if str(args.bc_offline_sampling) == "episode"
                else "uniform_window"
            )
            batches = []
            if offline_batch_size > 0:
                batches.append(
                    bc_offline_sampler.sample(
                        offline_batch_size,
                        sampling=demo_sampling,
                    )
                )
            if online_batch_size > 0:
                batches.append(
                    bc_replay.sample(
                        online_batch_size,
                        sampling=online_sampling,
                    )
                )
            if len(batches) == 1:
                batch = batches[0]
            else:
                mixed_actions = torch.cat([item.actions for item in batches], dim=0)
                mixed_states = torch.cat(
                    [item.conditions["state"] for item in batches],
                    dim=0,
                )
                permutation = torch.randperm(batch_size, device=mixed_actions.device)
                batch_actions = mixed_actions[permutation]
                batch_conditions = {"state": mixed_states[permutation]}
                batch = type(batches[0])(
                    actions=batch_actions,
                    conditions=batch_conditions,
                )
            teacher_modules = []
            if teacher_weight > 0.0:
                if bc_teacher_policy is None:
                    raise RuntimeError("Adroit online teacher is enabled but missing")
                teacher_modules.append(bc_teacher_policy)
            data_loss, teacher_losses = bc_adapter.diffusion_bc_loss_multi_teacher_terms(
                batch.actions,
                batch.conditions,
                teacher_noise_pred_nets=teacher_modules,
            )
            teacher_loss = teacher_losses[0] if teacher_losses else data_loss.new_zeros(())
            loss = data_loss + teacher_weight * teacher_loss
            loss.backward()
            bc_optimizer.step()
            if bc_teacher_policy is not None and float(args.bc_teacher_ema_decay) < 1.0:
                ema_update_module_(
                    bc_teacher_policy,
                    bc_adapter.live_policy["policy"]["noise_pred_net"],
                    float(args.bc_teacher_ema_decay),
                )
            bc_update_count += 1
            completed += 1
            train_meter["bc/loss"].append(float(loss.detach().cpu()))
            train_meter["bc/loss_data"].append(float(data_loss.detach().cpu()))
            train_meter["bc/loss_online"].append(float(data_loss.detach().cpu()))
            train_meter["bc/loss_teacher"].append(float(teacher_loss.detach().cpu()))
            train_meter["bc/offline_fraction"].append(float(offline_batch_size) / float(batch_size))
            train_meter["bc/update_count"].append(float(bc_update_count))
            train_meter["bc/rl_update_seen_count"].append(float(bc_rl_update_seen_count))
            train_meter["bc/success_windows"].append(float(len(bc_replay)))
            train_meter["bc/success_episodes_added"].append(float(bc_replay.success_episodes_added))
            train_meter["bc/active_success_episodes"].append(float(bc_replay.active_episode_count))
            train_meter["bc/current_lr"].append(float(args.bc_lr))
            latest_train_metrics["bc/loss"] = float(loss.detach().cpu())
            latest_train_metrics["bc/loss_data"] = float(data_loss.detach().cpu())
            latest_train_metrics["bc/loss_online"] = float(data_loss.detach().cpu())
            latest_train_metrics["bc/loss_teacher"] = float(teacher_loss.detach().cpu())
            latest_train_metrics["bc/offline_fraction"] = float(offline_batch_size) / float(
                batch_size
            )

        if completed < planned_updates:
            bc_update_credit += float(planned_updates - completed)
        train_meter["bc/target_updates_per_rl"].append(float(target_updates_per_rl))
        train_meter["bc/update_credit"].append(float(bc_update_credit))
        latest_train_metrics["bc/target_updates_per_rl"] = float(target_updates_per_rl)
        latest_train_metrics["bc/update_credit"] = float(bc_update_credit)

        if completed > 0:
            bc_adapter.set_eval(keep_trainable=True)
            train_wrapper = _unwrap_vec_env(venv)
            train_wrapper.sync_policy_replicas_from_primary()
            if eval_venv is not None and eval_venv is not venv:
                eval_wrapper = _unwrap_vec_env(eval_venv)
                eval_wrapper.sync_policy_replicas_from_primary()
            refreshed_obs = train_wrapper.refresh_current_policy_plan()
            if refreshed_obs is not None:
                obs = np.asarray(refreshed_obs, dtype=np.float32)
        return int(completed)

    next_periodic_ckpt = _next_multiple_after(policy_steps, int(args.save_freq))
    next_latest_ckpt = _next_multiple_after(policy_steps, int(args.latest_model_save_freq))
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
    print("[SAC] start training loop", flush=True)
    _append_status_line(status_log_path, "[SAC] start training loop")
    try:
        if str(args.pipeline) == "offline_actorless_phase01":
            _validate_offline_actorless_args(args, floor_support_mask_mode)
            phase0_target = (
                int(args.offline_phase0_policy_steps)
                if int(args.offline_phase0_policy_steps) > 0
                else int(args.warmup_policy_steps)
            )
            try:
                resume_phase_id = int(float(offline_phase_state.get("phase_id", 0)))
            except Exception:
                resume_phase_id = 0
            offline_eval_env_steps_total = int(
                offline_phase_state.get("offline_eval_env_steps_total", 0)
            )
            offline_msg = (
                f"[SAC][OFFLINE] phase0 random collection target_policy_steps={phase0_target} "
                f"phase1_critic_updates={int(args.offline_phase1_critic_updates)} "
                f"phase2_policy_steps={int(args.offline_phase2_policy_steps)} "
                f"phase2_max_env_steps={int(args.offline_phase2_max_env_steps)} "
                f"target_policy_mode={args.target_policy_mode} cql_alpha={float(args.cql_alpha)}"
            )
            print(offline_msg, flush=True)
            _append_status_line(status_log_path, offline_msg)

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

            def record_env_step_eval(*, eval_axis: int, phase_id: float, critic_updates: int):
                nonlocal best_success
                eval_metrics = _run_critic_policy_eval(
                    venv=ensure_eval_venv(allow_collector_fallback=False),
                    sac=sac,
                    action_shift=action_shift,
                    episodes=int(args.offline_eval_episodes),
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
                    "env_step_eval/episodes": float(eval_metrics["offline_eval/critic_episodes"]),
                    "env_step_eval/eval_env_steps": float(
                        eval_metrics["offline_eval/critic_env_steps"]
                    ),
                    "env_step_eval/decisions": float(eval_metrics["offline_eval/critic_decisions"]),
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
                    f"[SAC][OFFLINE][ENV-EVAL] eval_axis={int(eval_axis)} "
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
                eval_success = float(metrics["env_step_eval/success_rate"])
                if str(args.best_checkpoint_metric) == "env_eval" and eval_success > float(
                    best_success
                ):
                    best_success = eval_success
                    _save_ckpt(os.path.join(ckpt_dir, "best_model.pt"), pack_ckpt())

            env_eval_interval = int(args.offline_env_eval_interval_steps)
            offline_resume_env_ckpt_freq = int(args.offline_resume_save_interval_steps)
            if offline_resume_env_ckpt_freq <= 0:
                offline_resume_env_ckpt_freq = (
                    int(env_eval_interval) if int(env_eval_interval) > 0 else 0
                )
            next_env_eval_axis = (
                int(
                    offline_phase_state.get(
                        "next_env_eval_axis", _next_multiple_after(env_steps, env_eval_interval)
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
                        "offline_eval_env_steps_total": int(offline_eval_env_steps_total),
                        "next_env_eval_axis": (
                            None if next_env_eval_axis is None else int(next_env_eval_axis)
                        ),
                        "offline_resume_env_ckpt_freq": int(offline_resume_env_ckpt_freq),
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
                    f"[SAC][OFFLINE][RESUME] entering phase_id={resume_phase_id} "
                    f"policy_steps={policy_steps} env_steps={env_steps} replay={len(replay)}"
                )
                print(resume_msg, flush=True)
                _append_status_line(status_log_path, resume_msg)

            if (not args.resume) and bool(args.offline_env_eval_at_start):
                record_env_step_eval(eval_axis=0, phase_id=0.0, critic_updates=0)

            # Phase 0: pure random k collection, no critic/actor/alpha updates and
            # no rollout-success logging. This is intentionally just replay fill.
            next_phase0_log = int(
                offline_phase_state.get(
                    "next_phase0_log",
                    _next_multiple_after(policy_steps, int(args.log_every_policy_steps)),
                )
            )
            next_offline_latest_policy = int(
                offline_phase_state.get(
                    "next_offline_latest_policy",
                    _next_multiple_after(policy_steps, int(args.latest_model_save_freq)),
                )
            )
            next_offline_periodic_policy = int(
                offline_phase_state.get(
                    "next_offline_periodic_policy",
                    _next_multiple_after(policy_steps, int(args.save_freq)),
                )
            )
            if resume_phase_id <= 0:
                refresh_common_phase_state(
                    0,
                    phase0_target=int(phase0_target),
                    next_phase0_log=int(next_phase0_log),
                    next_offline_latest_policy=int(next_offline_latest_policy),
                    next_offline_periodic_policy=int(next_offline_periodic_policy),
                )
            while resume_phase_id <= 0 and policy_steps < int(phase0_target):
                actions = np.random.randint(0, action_dim, size=args.n_envs, dtype=np.int64)
                env_actions = actions + action_shift
                next_obs, rewards, dones, infos = venv.step(env_actions)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
                dones = np.asarray(dones, dtype=bool).reshape(-1)
                if bc_is_enabled:
                    collected_episodes, collected_windows = _collect_bc_success_infos(
                        infos,
                        bc_replay,
                    )
                    if collected_episodes > 0:
                        train_meter["bc/episodes_collected"].append(float(collected_episodes))
                        train_meter["bc/windows_collected"].append(float(collected_windows))
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
                    num_emitted_transitions_total += int(ingest_result.emitted_transitions)
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
                    avg_exec = float(np.mean(exec_buffer)) if len(exec_buffer) > 0 else 0.0
                    metrics = {
                        "offline/phase": 0.0,
                        "time/policy_steps": float(policy_steps),
                        "time/env_steps": float(env_steps),
                        "time/policy_steps_per_sec": float(policy_steps / elapsed),
                        "time/env_steps_per_sec": float(env_steps / elapsed),
                        "adaptive/avg_exec_len": avg_exec,
                        "replay/size": float(len(replay)),
                    }
                    record_offline_event(metrics, env_axis=int(env_steps), critic_updates=0)
                    for mk, mv in metrics.items():
                        writer.add_scalar(mk, float(mv), int(env_steps))
                    line = (
                        f"[SAC][OFFLINE][PHASE0] policy_steps={policy_steps}/{phase0_target} "
                        f"env_steps={env_steps} avg_k={avg_exec:.2f} replay={len(replay)}"
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
                if policy_steps >= next_offline_periodic_policy:
                    next_offline_periodic_policy += max(int(args.save_freq), 1)
                    refresh_common_phase_state(
                        0,
                        phase0_target=int(phase0_target),
                        next_phase0_log=int(next_phase0_log),
                        next_offline_latest_policy=int(next_offline_latest_policy),
                        next_offline_periodic_policy=int(next_offline_periodic_policy),
                    )
                    _save_ckpt(
                        os.path.join(ckpt_dir, f"adaptive_agent_{policy_steps}_steps.pt"),
                        pack_ckpt(),
                    )
                if policy_steps >= next_offline_latest_policy:
                    next_offline_latest_policy += max(int(args.latest_model_save_freq), 1)
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
                        next_offline_latest_env += max(int(offline_resume_env_ckpt_freq), 1)
                    refresh_common_phase_state(
                        0,
                        phase0_target=int(phase0_target),
                        next_phase0_log=int(next_phase0_log),
                        next_offline_latest_policy=int(next_offline_latest_policy),
                        next_offline_periodic_policy=int(next_offline_periodic_policy),
                    )
                    _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())

            if len(replay) < int(args.batch_size):
                raise RuntimeError(
                    f"offline phase0 replay too small for batch_size: replay={len(replay)} batch={args.batch_size}"
                )
            phase1_target = int(args.offline_phase1_critic_updates)
            phase1_updates = int(
                offline_phase_state.get(
                    "phase1_updates",
                    phase1_target if resume_phase_id >= 2 else 0,
                )
            )
            if resume_phase_id <= 0:
                phase1_updates = 0
                resume_phase_id = 1
            if bool(args.use_mc_lower_bound) and phase1_updates <= 0 and resume_phase_id <= 1:
                valid_mc = replay.compute_mc_returns_in_place(gamma=float(args.gamma))
                line = f"[SAC][OFFLINE] computed MC returns for {valid_mc}/{len(replay)} replay transitions"
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
            if resume_phase_id <= 1 and phase1_updates == 0:
                _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())

            # Phase 1: critic-only replay training. Evaluation periodically rolls
            # out critic-as-actor and logs success, so pure critic improvement is visible.
            next_phase1_log = int(
                offline_phase_state.get(
                    "next_phase1_log",
                    _next_multiple_after(phase1_updates, int(args.log_every_policy_steps)),
                )
            )
            eval_every = int(args.offline_eval_every_updates)
            offline_resume_critic_ckpt_freq = int(args.offline_resume_save_interval_steps)
            if offline_resume_critic_ckpt_freq <= 0:
                offline_resume_critic_ckpt_freq = int(eval_every) if int(eval_every) > 0 else 0
            if eval_every <= 0 and resume_phase_id <= 1:
                try:
                    venv.close()
                except Exception:
                    pass
                venv = None
            next_eval = int(
                offline_phase_state.get(
                    "next_eval",
                    _next_multiple_after(phase1_updates, eval_every)
                    if eval_every > 0
                    else phase1_target + 1,
                )
            )
            next_offline_latest_critic = int(
                offline_phase_state.get(
                    "next_offline_latest_critic",
                    _next_multiple_after(phase1_updates, int(args.latest_model_save_freq)),
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
                        _next_multiple_after(phase1_updates, offline_resume_critic_ckpt_freq),
                    )
                )
                if offline_resume_critic_ckpt_freq > 0
                else None
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
                    offline_resume_critic_ckpt_freq=int(offline_resume_critic_ckpt_freq),
                    next_offline_latest_critic_resume=(
                        None
                        if next_offline_latest_critic_resume is None
                        else int(next_offline_latest_critic_resume)
                    ),
                )

            refresh_phase1_state(1 if resume_phase_id <= 1 else int(resume_phase_id))
            while resume_phase_id <= 1 and phase1_updates < int(args.offline_phase1_critic_updates):
                m = sac.update(
                    replay.sample(int(args.batch_size)),
                    update_actor=False,
                    update_alpha=False,
                    transition_weight_mode=str(args.transition_weight_mode),
                    actor_prior_coeff=0.0,
                    actor_prior_min_action_id=0,
                    legal_min_action_id=0,
                )
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
                    }
                    if bc_is_enabled:
                        metrics.update(
                            {
                                "bc/success_windows": float(len(bc_replay)),
                                "bc/success_episodes_added": float(
                                    bc_replay.success_episodes_added
                                ),
                                "bc/active_success_episodes": float(bc_replay.active_episode_count),
                                "bc/total_windows_added": float(bc_replay.total_windows_added),
                                "bc/update_count": float(bc_update_count),
                                "bc/rl_update_seen_count": float(bc_rl_update_seen_count),
                            }
                        )
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
                            episodes=int(args.offline_eval_episodes),
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
                        if str(args.best_checkpoint_metric) == "rollout" and float(
                            metrics["rollout/success_rate"]
                        ) > float(best_success):
                            best_success = float(metrics["rollout/success_rate"])
                            refresh_phase1_state(1)
                            _save_ckpt(os.path.join(ckpt_dir, "best_model.pt"), pack_ckpt())
                        next_eval += max(eval_every, 1)
                    record_offline_event(metrics, env_axis=env_axis, critic_updates=phase1_updates)
                    for mk, mv in metrics.items():
                        writer_step = (
                            int(phase1_updates)
                            if mk.startswith("offline") or mk.startswith("train/")
                            else env_axis
                        )
                        writer.add_scalar(mk, float(mv), writer_step)
                    line = (
                        f"[SAC][OFFLINE][PHASE1] updates={phase1_updates}/{args.offline_phase1_critic_updates} "
                        f"env_axis={env_axis} replay={len(replay)} "
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
                if phase1_updates >= next_offline_periodic_critic:
                    next_offline_periodic_critic += max(int(args.save_freq), 1)
                    refresh_phase1_state(1)
                    _save_ckpt(
                        os.path.join(ckpt_dir, f"adaptive_agent_critic_update_{phase1_updates}.pt"),
                        pack_ckpt(),
                    )
                if phase1_updates >= next_offline_latest_critic:
                    next_offline_latest_critic += max(int(args.latest_model_save_freq), 1)
                    refresh_phase1_state(1)
                    _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
                if next_offline_latest_critic_resume is not None and phase1_updates >= int(
                    next_offline_latest_critic_resume
                ):
                    while next_offline_latest_critic_resume is not None and phase1_updates >= int(
                        next_offline_latest_critic_resume
                    ):
                        next_offline_latest_critic_resume += max(
                            int(offline_resume_critic_ckpt_freq), 1
                        )
                    refresh_phase1_state(1)
                    _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())

            if resume_phase_id <= 1:
                resume_phase_id = 2
            phase2_target = int(args.offline_phase2_policy_steps)
            phase2_max_env_steps = int(args.offline_phase2_max_env_steps)
            if phase2_target <= 0:
                refresh_common_phase_state(
                    3,
                    phase0_target=int(phase0_target),
                    phase1_updates=int(phase1_updates),
                    phase2_target=int(phase2_target),
                )
                _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
                dump_metrics()
                print("[SAC][OFFLINE] phase01 finished", flush=True)
                _append_status_line(status_log_path, "[SAC][OFFLINE] phase01 finished")
                return

            # Phase 2: use the critic itself as the rollout policy, append real
            # environment data, and keep training the critic only. The x-axis is
            # still real training env steps: phase-1 updates did not advance it.
            if venv is None:
                venv = _build_task_venv(
                    args=args,
                    policy=policy,
                    ckpt_dict=ckpt_dict,
                    device=device,
                    env_meta=env_meta,
                    stats_npz_path=stats_npz_path,
                    n_envs=int(args.n_envs),
                    seed=int(args.seed),
                    collect_success_bc=bool(bc_is_enabled),
                )
                if bc_is_enabled:
                    train_wrapper = _unwrap_vec_env(venv)
                    train_wrapper.set_primary_policy_trainable(True)
                    train_wrapper.sync_policy_replicas_from_primary()
            obs = np.asarray(venv.reset(), dtype=np.float32)
            for q in nstep_queues:
                q.clear()
            phase2_start_env_steps = int(
                offline_phase_state.get("phase2_start_env_steps", int(env_steps))
            )
            phase2_policy_steps = int(offline_phase_state.get("phase2_policy_steps", 0))
            phase2_critic_updates = int(offline_phase_state.get("phase2_critic_updates", 0))
            phase2_step_calls = int(offline_phase_state.get("phase2_step_calls", 0))
            phase2_pending_transitions = int(
                offline_phase_state.get("phase2_pending_transitions", 0)
            )
            phase2_next_log = int(
                offline_phase_state.get(
                    "phase2_next_log",
                    _next_multiple_after(phase2_policy_steps, int(args.log_every_policy_steps)),
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
                    _next_multiple_after(policy_steps, int(args.latest_model_save_freq)),
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
                f"[SAC][OFFLINE][PHASE2] start policy_steps={phase2_policy_steps}/{phase2_target} "
                f"max_env_steps={phase2_max_env_steps or 'disabled'} "
                f"mode={args.offline_phase2_policy_mode} reduce={args.offline_phase2_critic_reduce} "
                f"epsilon={float(args.offline_phase2_epsilon_greedy_prob)} "
                f"ucb_beta={float(args.offline_phase2_ucb_beta)} "
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
                    phase2_max_env_steps=int(phase2_max_env_steps),
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
                phase2_max_env_steps=int(phase2_max_env_steps),
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

            while phase2_policy_steps < phase2_target and (
                phase2_max_env_steps <= 0 or int(env_steps) < phase2_max_env_steps
            ):
                current_ucb_beta = _resolve_float_schedule(
                    int(env_steps),
                    phase2_ucb_beta_schedule_items,
                    default=float(args.offline_phase2_ucb_beta),
                )
                select_out = sac.select_critic_action(
                    obs,
                    deterministic=(str(args.offline_phase2_policy_mode) == "q_greedy"),
                    min_action_id=0,
                    mode=str(args.offline_phase2_policy_mode),
                    reduce=str(args.offline_phase2_critic_reduce),
                    ucb_beta=float(current_ucb_beta),
                    thompson_heads=(
                        phase2_thompson_heads
                        if str(args.offline_phase2_policy_mode) == "q_thompson"
                        else None
                    ),
                    return_info=True,
                )
                actions, action_info = select_out
                for k, v in action_info.items():
                    train_meter[f"diagnostic/{k}"].append(float(v))
                phase2_epsilon = float(args.offline_phase2_epsilon_greedy_prob)
                if phase2_epsilon > 0.0:
                    random_actions = np.random.randint(
                        0, action_dim, size=args.n_envs, dtype=np.int64
                    )
                    use_random = np.random.rand(args.n_envs) < phase2_epsilon
                    actions = np.where(use_random, random_actions, actions).astype(np.int64)
                env_actions = actions + action_shift
                next_obs, rewards, dones, infos = venv.step(env_actions)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
                dones = np.asarray(dones, dtype=bool).reshape(-1)
                if bc_is_enabled:
                    collected_episodes, collected_windows = _collect_bc_success_infos(
                        infos,
                        bc_replay,
                    )
                    if collected_episodes > 0:
                        train_meter["bc/episodes_collected"].append(float(collected_episodes))
                        train_meter["bc/windows_collected"].append(float(collected_windows))
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
                    num_emitted_transitions_total += int(ingest_result.emitted_transitions)
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
                        phase2_episode_return_buffer.append(float(phase2_episode_returns[i]))
                        phase2_episode_len_buffer.append(float(phase2_episode_lengths[i]))
                        phase2_episode_returns[i] = 0.0
                        phase2_episode_lengths[i] = 0
                        if str(args.offline_phase2_policy_mode) == "q_thompson":
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
                if len(replay) >= int(args.batch_size):
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
                        m = sac.update(
                            replay.sample(int(args.batch_size)),
                            update_actor=False,
                            update_alpha=False,
                            transition_weight_mode=str(args.transition_weight_mode),
                            actor_prior_coeff=0.0,
                            actor_prior_min_action_id=0,
                            legal_min_action_id=0,
                        )
                        for k, v in m.items():
                            train_meter[k].append(float(v))
                            latest_train_metrics[k] = float(v)
                        run_bc_updates()
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
                        float(np.mean(phase2_exec_buffer)) if len(phase2_exec_buffer) > 0 else 0.0
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
                        "offline/critic_updates": float(phase1_updates + phase2_critic_updates),
                        "offline/phase2_policy_steps": float(phase2_policy_steps),
                        "offline/phase2_env_steps": float(int(env_steps) - phase2_start_env_steps),
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
                        f"[SAC][OFFLINE][PHASE2] policy_steps={phase2_policy_steps}/{phase2_target} "
                        f"env_steps={env_steps} sr={success_recent:.4f} sr20={success_recent20:.4f} "
                        f"avg_k={avg_exec:.2f} replay={len(replay)} updates={phase2_critic_updates}"
                        + (
                            f" bc_windows={len(bc_replay)} bc_updates={bc_update_count}"
                            f" bc_target_per_rl={latest_train_metrics.get('bc/target_updates_per_rl', 0.0):.3f}"
                            f" bc_credit={bc_update_credit:.3f}"
                            if bc_is_enabled
                            else ""
                        )
                    )
                    print(line, flush=True)
                    _append_status_line(status_log_path, line)
                    dump_metrics()
                    writer.flush()
                    refresh_phase2_state()
                    if (
                        str(args.best_checkpoint_metric) == "rollout"
                        and success_recent > best_success
                        and len(phase2_success_buffer) > 0
                    ):
                        best_success = success_recent
                        _save_ckpt(os.path.join(ckpt_dir, "best_model.pt"), pack_ckpt())
                    phase2_next_log += max(int(args.log_every_policy_steps), 1)
                    refresh_phase2_state()

                refresh_phase2_state()
                if policy_steps >= next_offline_periodic_policy:
                    next_offline_periodic_policy += max(int(args.save_freq), 1)
                    refresh_phase2_state()
                    _save_ckpt(
                        os.path.join(ckpt_dir, f"adaptive_agent_{policy_steps}_steps.pt"),
                        pack_ckpt(),
                    )
                if policy_steps >= next_offline_latest_policy:
                    next_offline_latest_policy += max(int(args.latest_model_save_freq), 1)
                    refresh_phase2_state()
                    _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
                if next_offline_latest_env is not None and int(env_steps) >= int(
                    next_offline_latest_env
                ):
                    while next_offline_latest_env is not None and int(env_steps) >= int(
                        next_offline_latest_env
                    ):
                        next_offline_latest_env += max(int(offline_resume_env_ckpt_freq), 1)
                    refresh_phase2_state()
                    _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())

            refresh_phase2_state(phase_id=3)
            _save_ckpt(os.path.join(ckpt_dir, "latest_model.pt"), pack_ckpt())
            dump_metrics()
            print("[SAC][OFFLINE] phase012 finished", flush=True)
            _append_status_line(status_log_path, "[SAC][OFFLINE] phase012 finished")
            return

        while policy_steps < int(args.total_policy_steps):
            model_probe_k = None
            current_min_k_floor = 1
            current_min_action_id = max(int(current_min_k_floor) - int(args.min_chunk_len), 0)
            dynamic_floor_signal = None
            if policy_steps < int(args.warmup_policy_steps) and bool(
                args.probe_model_k_during_warmup
            ):
                probe_actions = sac.select_action(
                    obs, deterministic=bool(args.warmup_probe_deterministic)
                )
                model_probe_k = float(
                    np.mean(probe_actions.astype(np.float32) + float(args.min_chunk_len))
                )
            if policy_steps < int(args.warmup_policy_steps):
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
                    )
                    for k, v in m.items():
                        train_meter[k].append(float(v))
                        latest_train_metrics[k] = float(v)
                num_updates_total += int(current_utd)

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

                if (
                    str(args.best_checkpoint_metric) == "rollout"
                    and success_recent > best_success
                    and len(success_buffer) > 0
                ):
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
