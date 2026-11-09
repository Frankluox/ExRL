"""Small audited helpers copied from the ResFiT offline-data transfer."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import torch

from baselines.dice.dp.core import DICEReplayBuffer


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass
class OfflineBuildSummary:
    episodes: int
    transitions: int
    success_episodes: int
    late_success_episodes: int
    success_horizon: int
    success_mode: str
    official_npz_sha256: str
    reward_hdf5_sha256: str
    states_sha256: str
    normalized_actions_sha256: str
    traj_lengths_sha256: str
    action_alignment_max_abs: float
    stop_rule: str = "last_frame_inclusive"
    hdf5_state_used: bool = False
    terminal_partial_chunks: int = 0
    terminal_full_chunks: int = 0
    min_macro_decisions_per_episode: int = 0
    max_macro_decisions_per_episode: int = 0
    primitive_transitions: int = 0


def numeric_demo_names(data_group) -> list:
    names = list(data_group.keys())
    try:
        return sorted(names, key=lambda value: int(str(value)[5:]))
    except Exception as exc:
        raise ValueError("HDF5 demo names must follow demo_<integer>") from exc


def read_episode_action_and_success(
    group,
    *,
    length: int,
    action_dim: int,
    success_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Read actions plus the exact ResFiT success fallback; never read HDF5 states."""

    mode = str(success_mode).lower()
    if mode not in {"reward", "last_frame"}:
        raise ValueError("offline success mode must be reward or last_frame")
    raw_actions = np.asarray(group["actions"][()])
    if raw_actions.shape != (int(length), int(action_dim)):
        raise ValueError("raw action shape mismatch")
    if not np.all(np.isfinite(raw_actions)):
        raise ValueError("raw actions contain NaN/Inf")
    if mode == "reward":
        if "rewards" not in group:
            raise KeyError("reward success mode requires HDF5 rewards")
        rewards = np.asarray(group["rewards"][()]).reshape(-1)
        if rewards.shape[0] != int(length):
            raise ValueError("reward length mismatch")
        if not np.all(np.isfinite(rewards)):
            raise ValueError("rewards contain NaN/Inf")
        success = np.flatnonzero(rewards > 0.0)
        if success.size == 0:
            raise RuntimeError("episode has no positive success reward")
        first_success = int(success[0])
    else:
        # Can MH v1.4.1 has no reward/done arrays. This is the exact fallback
        # used by the validated ResFiT converter and treats the last frame as
        # the successful terminal transition.
        rewards = np.zeros((int(length),), dtype=np.float32)
        rewards[-1] = 1.0
        first_success = int(length) - 1
    if "dones" in group:
        dones = np.asarray(group["dones"][()]).reshape(-1).astype(bool)
        if dones.shape[0] != int(length):
            raise ValueError("done length mismatch")
    else:
        dones = np.zeros((int(length),), dtype=bool)
        dones[-1] = True
    return raw_actions, np.asarray(rewards), dones, first_success


def save_offline_cache(
    cache_path: str,
    replay: DICEReplayBuffer,
    summary: OfflineBuildSummary,
    provenance: Dict[str, object],
) -> None:
    """Atomically save the small audited replay and a readable sidecar."""

    parent = os.path.dirname(os.path.abspath(cache_path))
    os.makedirs(parent, exist_ok=True)
    payload = {
        "version": 1,
        "replay": replay.state_dict(),
        "summary": asdict(summary),
        "provenance": dict(provenance),
    }
    temporary_cache_path = "%s.tmp.%d" % (cache_path, os.getpid())
    temporary_json_path = "%s.json.tmp.%d" % (cache_path, os.getpid())
    torch.save(payload, temporary_cache_path)
    with open(temporary_cache_path, "rb") as stream:
        os.fsync(stream.fileno())
    with open(temporary_json_path, "w") as stream:
        json.dump(
            {"summary": asdict(summary), "provenance": provenance},
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_cache_path, cache_path)
    os.replace(temporary_json_path, cache_path + ".json")
    directory_fd = None
    try:
        directory_fd = os.open(parent, os.O_RDONLY)
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def load_offline_cache(
    cache_path: str,
    replay: DICEReplayBuffer,
    expected_provenance: Dict[str, object],
) -> OfflineBuildSummary:
    # The provenance-keyed local cache contains RNG and NumPy replay state.
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if int(payload.get("version", -1)) != 1:
        raise RuntimeError("unsupported DICE offline cache version")
    if payload.get("provenance") != expected_provenance:
        raise RuntimeError("DICE offline cache provenance mismatch")
    replay.load_state_dict(payload["replay"], strict_capacity=False)
    return OfflineBuildSummary(**dict(payload["summary"]))
