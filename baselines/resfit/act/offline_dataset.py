"""Build ResFiT offline replay without ever using the mismatched HDF5 states."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from baselines.resfit.act.act_queue import ACTChunkQueue
from baselines.resfit.act.td3_core import (
    PrimitiveNStepAccumulator,
    PrimitiveReplayBuffer,
    PrimitiveTransition,
    ResFiTActionScaler,
    StatePreprocessor,
)


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
    stop_rule: str = "first_success_inclusive"
    hdf5_state_used: bool = False
    terminal_partial_chunks: int = 0
    terminal_full_chunks: int = 0
    min_macro_decisions_per_episode: int = 0
    max_macro_decisions_per_episode: int = 0


def _numeric_demo_names(data_group) -> list:
    names = list(data_group.keys())
    try:
        return sorted(names, key=lambda value: int(str(value)[5:]))
    except Exception as exc:
        raise ValueError("HDF5 demo names must follow demo_<integer>") from exc


def _read_episode_action_and_success(
    group,
    *,
    length: int,
    action_dim: int,
    success_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Read only action/reward/done data and resolve an explicit success rule."""
    mode = str(success_mode).lower()
    if mode not in {"reward", "last_frame"}:
        raise ValueError("offline success mode must be reward or last_frame")
    raw_actions = np.asarray(group["actions"][()])
    if raw_actions.shape != (int(length), int(action_dim)):
        raise ValueError(
            "raw action shape mismatch: expected (%d, %d), got %s"
            % (int(length), int(action_dim), raw_actions.shape)
        )

    if mode == "reward":
        if "rewards" not in group:
            raise KeyError("reward success mode requires an HDF5 rewards array")
        rewards = np.asarray(group["rewards"][()]).reshape(-1)
        if rewards.shape[0] != int(length):
            raise ValueError("reward length mismatch")
        success_indices = np.flatnonzero(rewards > 0.0)
        if success_indices.size == 0:
            raise RuntimeError("episode has no positive success reward")
        first_success = int(success_indices[0])
    else:
        # This is the fallback used by the upstream ResFiT robomimic
        # converter when the source HDF5 has no dones.  Lift and Can MH v1.4.1
        # contain actions/states but no reward/done arrays.
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


def build_offline_replay(
    official_npz_path: str,
    reward_hdf5_path: str,
    normalization_path: str,
    expected_npz_sha256: Optional[str],
    queue: ACTChunkQueue,
    action_scaler: ResFiTActionScaler,
    state_preprocessor: StatePreprocessor,
    replay: PrimitiveReplayBuffer,
    n_step: int = 3,
    gamma: float = 0.99,
    reward_offset: float = 1.0,
    success_mode: str = "reward",
    success_horizon: int = 400,
    expected_transition_count: Optional[int] = None,
    precomputed_npz_sha256: Optional[str] = None,
    precomputed_hdf5_sha256: Optional[str] = None,
    progress_interval_episodes: int = 25,
) -> OfflineBuildSummary:
    """Populate replay from official states and HDF5 action/reward/done arrays.

    ``h5py`` is imported only inside this mixed-mode function.  The HDF5
    ``obs``, ``next_obs``, and ``states`` datasets are intentionally never read.
    """
    import h5py  # Local import is a hard online-mode isolation boundary.

    npz_sha = precomputed_npz_sha256 or sha256_file(official_npz_path)
    if expected_npz_sha256 and npz_sha != str(expected_npz_sha256):
        raise RuntimeError(
            "official NPZ SHA mismatch: expected %s, got %s" % (expected_npz_sha256, npz_sha)
        )
    hdf5_sha = precomputed_hdf5_sha256 or sha256_file(reward_hdf5_path)
    official = np.load(official_npz_path)
    required = {"states", "actions", "traj_lengths"}
    if not required.issubset(set(official.files)):
        raise KeyError("official NPZ must contain %s" % sorted(required))
    states = np.asarray(official["states"])
    normalized_actions = np.asarray(official["actions"])
    traj_lengths = np.asarray(official["traj_lengths"], dtype=np.int64)
    if states.ndim != 2 or normalized_actions.ndim != 2:
        raise ValueError("official states/actions must be matrices")
    if states.shape[0] != normalized_actions.shape[0] or states.shape[0] != int(traj_lengths.sum()):
        raise ValueError("official state/action/trajectory lengths are inconsistent")

    norm = np.load(normalization_path)
    action_min64 = np.asarray(norm["action_min"], dtype=np.float64)
    action_max64 = np.asarray(norm["action_max"], dtype=np.float64)

    accumulator = PrimitiveNStepAccumulator(n_step=n_step, gamma=gamma, n_envs=1)
    episode_count = 0
    success_count = 0
    late_success_count = 0
    raw_action_parts = []
    hdf5_lengths = []
    state_offset = 0
    started_at = time.monotonic()

    with h5py.File(reward_hdf5_path, "r") as handle:
        demos = _numeric_demo_names(handle["data"])
        if len(demos) != len(traj_lengths):
            raise ValueError("HDF5/NPZ episode count mismatch")
        for episode_index, demo_name in enumerate(demos):
            group = handle["data"][demo_name]
            length = int(traj_lengths[episode_index])
            try:
                raw_actions, rewards, dones, first_success = _read_episode_action_and_success(
                    group,
                    length=length,
                    action_dim=action_scaler.action_dim,
                    success_mode=success_mode,
                )
            except Exception as exc:
                raise type(exc)("%s: %s" % (demo_name, exc)) from exc
            hdf5_lengths.append(length)
            raw_action_parts.append(raw_actions)

            late_success_count += int(first_success >= int(success_horizon))
            success_count += 1
            episode_count += 1

            episode_states = states[state_offset : state_offset + length]
            queue.reset()
            for step_index in range(first_success + 1):
                state_dp = np.asarray(episode_states[step_index], dtype=np.float32)
                base_dp = queue.current(state_dp[None])[0]
                base_resfit = np.asarray(action_scaler.dp_to_resfit(base_dp), dtype=np.float32)
                state_actor = state_preprocessor.transform(state_dp)
                full_action = np.asarray(
                    action_scaler.raw_to_resfit(
                        np.asarray(raw_actions[step_index], dtype=np.float32)
                    ),
                    dtype=np.float32,
                )
                is_terminal = step_index == first_success
                if is_terminal:
                    next_state_actor = np.zeros_like(state_actor)
                    next_base_resfit = np.zeros((action_scaler.action_dim,), dtype=np.float32)
                else:
                    next_state_dp = np.asarray(episode_states[step_index + 1], dtype=np.float32)
                    next_state_actor = state_preprocessor.transform(next_state_dp)
                    next_base_dp = queue.advance(next_state_dp[None], np.asarray([False]))[0]
                    next_base_resfit = np.asarray(
                        action_scaler.dp_to_resfit(next_base_dp), dtype=np.float32
                    )
                transition = PrimitiveTransition(
                    state=state_actor,
                    base_action=base_resfit,
                    action=full_action,
                    reward=float(rewards[step_index]) - float(reward_offset),
                    next_state=next_state_actor,
                    next_base_action=next_base_resfit,
                    terminated=is_terminal,
                    truncated=False,
                )
                replay.extend(accumulator.append(0, transition))
            if accumulator.queues[0]:
                raise RuntimeError("terminal transition failed to flush offline n-step queue")
            state_offset += length
            if int(progress_interval_episodes) > 0 and (
                episode_count % int(progress_interval_episodes) == 0 or episode_count == len(demos)
            ):
                print(
                    "[OFFLINE CACHE] episodes=%d/%d transitions=%d elapsed_sec=%.1f"
                    % (
                        episode_count,
                        len(demos),
                        len(replay),
                        time.monotonic() - started_at,
                    ),
                    flush=True,
                )

    raw_actions_all = np.concatenate(raw_action_parts, axis=0).astype(np.float64, copy=False)
    hdf5_lengths_array = np.asarray(hdf5_lengths, dtype=np.int64)
    derived_normalized = (
        2.0 * (raw_actions_all - action_min64) / (action_max64 - action_min64 + 1e-6) - 1.0
    )
    max_abs = float(np.max(np.abs(derived_normalized - normalized_actions)))
    if max_abs > 1e-7:
        raise RuntimeError("official/HDF5 action alignment failed: max_abs=%g" % max_abs)
    if not np.array_equal(hdf5_lengths_array, traj_lengths):
        raise RuntimeError("official/HDF5 trajectory lengths differ")
    if state_offset != states.shape[0]:
        raise RuntimeError("offline state offset did not consume the official NPZ")
    if expected_transition_count is not None and len(replay) != int(expected_transition_count):
        raise RuntimeError(
            "offline transition count mismatch: expected %d, got %d"
            % (int(expected_transition_count), len(replay))
        )

    return OfflineBuildSummary(
        episodes=episode_count,
        transitions=len(replay),
        success_episodes=success_count,
        late_success_episodes=late_success_count,
        success_horizon=int(success_horizon),
        success_mode=str(success_mode).lower(),
        official_npz_sha256=npz_sha,
        reward_hdf5_sha256=hdf5_sha,
        states_sha256=sha256_array(states),
        normalized_actions_sha256=sha256_array(normalized_actions),
        traj_lengths_sha256=sha256_array(traj_lengths),
        action_alignment_max_abs=max_abs,
        stop_rule=(
            "first_success_inclusive"
            if str(success_mode).lower() == "reward"
            else "last_frame_inclusive"
        ),
    )


def save_offline_cache(
    cache_path: str,
    replay: PrimitiveReplayBuffer,
    summary: OfflineBuildSummary,
    provenance: Dict[str, object],
) -> None:
    parent = os.path.dirname(os.path.abspath(cache_path))
    os.makedirs(parent, exist_ok=True)
    payload = {
        "replay": replay.state_dict(),
        "summary": asdict(summary),
        "provenance": dict(provenance),
    }
    temporary_cache_path = "%s.tmp.%d" % (cache_path, os.getpid())
    temporary_json_path = "%s.json.tmp.%d" % (cache_path, os.getpid())
    torch.save(payload, temporary_cache_path)
    with open(temporary_json_path, "w") as stream:
        json.dump(
            {"summary": asdict(summary), "provenance": provenance}, stream, indent=2, sort_keys=True
        )
    os.replace(temporary_cache_path, cache_path)
    os.replace(temporary_json_path, cache_path + ".json")


def load_offline_cache(
    cache_path: str,
    replay: PrimitiveReplayBuffer,
    expected_provenance: Dict[str, object],
) -> OfflineBuildSummary:
    payload = torch.load(cache_path, map_location="cpu")
    if payload.get("provenance") != expected_provenance:
        raise RuntimeError("offline cache provenance mismatch")
    # Cache payloads contain only the populated prefix, so their construction
    # capacity is not part of the data. Training may safely choose any replay
    # capacity that fits the actual cached transitions. Checkpoint resume keeps
    # the default strict capacity contract in PrimitiveReplayBuffer.
    replay.load_state_dict(payload["replay"], strict_capacity=False)
    summary_state = dict(payload["summary"])
    # Read old Square-only caches for diagnostics.  They cannot be selected by
    # a new provenance key, but this keeps the cache reader self-describing.
    if "late_success_episodes" not in summary_state:
        summary_state["late_success_episodes"] = summary_state.pop("first_success_at_or_after_400")
        summary_state.setdefault("success_horizon", 400)
        summary_state.setdefault("success_mode", "reward")
    return OfflineBuildSummary(**summary_state)
