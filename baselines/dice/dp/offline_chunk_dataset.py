"""Build an all-demo replay for fixed-TA DICE-RL.

This follows the audited ResFiT converter's episode/action alignment and
task-specific success rule. It stores one non-overlapping transition per
issued TA32/TA50 chunk, uses normalized official NPZ states/actions, and never
reads HDF5 states. Per-primitive rewards are ``raw_sparse - 1``; rewards are
summed within a chunk and discounted only across macro decisions.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from baselines.dice.dp.core import (
    CompleteEpisodeAccumulator,
    DICEReplayBuffer,
    MacroTransition,
)
from baselines.dice.dp.offline_dataset_utils import (
    OfflineBuildSummary,
    numeric_demo_names,
    read_episode_action_and_success,
    sha256_array,
    sha256_file,
)


def build_offline_chunk_replay(
    official_npz_path: str,
    reward_hdf5_path: str,
    normalization_path: str,
    expected_npz_sha256: Optional[str],
    prior_sampler,
    replay: DICEReplayBuffer,
    horizon_steps: int,
    action_dim: int,
    state_dim: int = 23,
    n_step: int = 3,
    gamma: float = 0.99,
    reward_offset: float = 1.0,
    success_mode: str = "last_frame",
    success_horizon: int = 300,
    expected_transition_count: Optional[int] = None,
    expected_primitive_transition_count: Optional[int] = None,
    precomputed_npz_sha256: Optional[str] = None,
    precomputed_hdf5_sha256: Optional[str] = None,
    suffix_noise_seed: int = 0,
    progress_interval_episodes: int = 25,
) -> OfflineBuildSummary:
    """Populate a CPU replay with deterministic, fully audited task chunks."""

    import h5py  # Local import preserves online-only import isolation.

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
    states = np.asarray(official["states"], dtype=np.float32)
    normalized_actions = np.asarray(official["actions"], dtype=np.float32)
    traj_lengths = np.asarray(official["traj_lengths"], dtype=np.int64)
    if traj_lengths.ndim != 1 or np.any(traj_lengths <= 0):
        raise ValueError("official trajectory lengths must be a positive vector")
    if states.shape != (int(traj_lengths.sum()), int(state_dim)):
        raise ValueError("official state shape/trajectory lengths mismatch")
    if normalized_actions.shape != (states.shape[0], int(action_dim)):
        raise ValueError("official action shape mismatch")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(normalized_actions)):
        raise ValueError("official NPZ contains NaN/Inf")
    if len(replay) != 0:
        raise ValueError("offline replay must be empty before conversion")

    norm = np.load(normalization_path)
    action_min64 = np.asarray(norm["action_min"], dtype=np.float64)
    action_max64 = np.asarray(norm["action_max"], dtype=np.float64)
    if action_min64.shape != (int(action_dim),) or action_max64.shape != (int(action_dim),):
        raise ValueError("normalization action bounds shape mismatch")
    if not np.all(np.isfinite(action_min64)) or not np.all(np.isfinite(action_max64)):
        raise ValueError("normalization action bounds contain NaN/Inf")
    if np.any(action_max64 <= action_min64):
        raise ValueError("normalization action bounds are not strictly increasing")
    horizon = int(horizon_steps)
    if int(prior_sampler.horizon_steps) != horizon or int(prior_sampler.ac_dim) != int(action_dim):
        raise ValueError("prior/offline chunk shape mismatch")
    accumulator = CompleteEpisodeAccumulator(n_step=n_step, gamma=gamma, n_envs=1)
    noise_rng = np.random.RandomState(int(suffix_noise_seed))
    episode_count = 0
    primitive_transition_count = 0
    success_count = 0
    late_success_count = 0
    raw_action_parts = []
    hdf5_lengths = []
    state_offset = 0
    terminal_partial_chunks = 0
    terminal_full_chunks = 0
    macro_decisions_per_episode = []
    started_at = time.monotonic()

    with h5py.File(reward_hdf5_path, "r") as handle:
        demos = numeric_demo_names(handle["data"])
        if len(demos) != len(traj_lengths):
            raise ValueError("HDF5/NPZ episode count mismatch")
        for episode_index, demo_name in enumerate(demos):
            group = handle["data"][demo_name]
            length = int(traj_lengths[episode_index])
            try:
                raw_actions, rewards, _dones, first_success = read_episode_action_and_success(
                    group,
                    length=length,
                    action_dim=action_dim,
                    success_mode=success_mode,
                )
            except Exception as exc:
                raise type(exc)("%s: %s" % (demo_name, exc)) from exc
            hdf5_lengths.append(length)
            raw_action_parts.append(raw_actions)
            late_success_count += int(first_success >= int(success_horizon))
            success_count += 1
            episode_count += 1
            primitive_transition_count += first_success + 1

            episode_states = states[state_offset : state_offset + length]
            episode_actions = normalized_actions[state_offset : state_offset + length]
            decision_starts = list(range(0, first_success + 1, horizon))
            macro_decisions_per_episode.append(len(decision_starts))
            terminal_duration = first_success + 1 - decision_starts[-1]
            terminal_partial_chunks += int(terminal_duration < horizon)
            terminal_full_chunks += int(terminal_duration == horizon)

            decision_states = torch.as_tensor(
                episode_states[decision_starts],
                dtype=torch.float32,
                device=prior_sampler.device,
            )
            suffix_noise = torch.as_tensor(
                noise_rng.randn(len(decision_starts), horizon, action_dim),
                dtype=torch.float32,
                device=prior_sampler.device,
            )
            with torch.no_grad():
                base_chunks = (
                    prior_sampler.sample_from_state_noise(decision_states, suffix_noise)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            expected_base_shape = (len(decision_starts), horizon, int(action_dim))
            if base_chunks.shape != expected_base_shape or not np.all(np.isfinite(base_chunks)):
                raise RuntimeError("frozen prior returned invalid offline suffix chunks")

            for decision_index, start in enumerate(decision_starts):
                duration = min(horizon, first_success + 1 - start)
                terminal = start + duration - 1 == first_success
                full_action = np.asarray(base_chunks[decision_index], dtype=np.float32).copy()
                full_action[:duration] = episode_actions[start : start + duration]
                chunk_reward = float(
                    np.sum(
                        np.asarray(rewards[start : start + duration], dtype=np.float64)
                        - float(reward_offset)
                    )
                )
                next_state = (
                    np.zeros((state_dim,), dtype=np.float32)
                    if terminal
                    else np.asarray(episode_states[start + duration], dtype=np.float32)
                )
                macro = MacroTransition(
                    state=np.asarray(episode_states[start], dtype=np.float32),
                    action=full_action.reshape(-1),
                    reward=chunk_reward,
                    next_state=next_state,
                    terminated=terminal,
                    truncated=False,
                )
                replay.extend(accumulator.append(0, macro))
            if accumulator.pending_lengths()[0] != 0:
                raise RuntimeError("terminal transition failed to flush offline episode")
            state_offset += length
            if int(progress_interval_episodes) > 0 and (
                episode_count % int(progress_interval_episodes) == 0 or episode_count == len(demos)
            ):
                print(
                    "[DICE OFFLINE CACHE] episodes=%d/%d transitions=%d elapsed_sec=%.1f"
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
        raise RuntimeError("offline state offset did not consume official NPZ")
    if expected_primitive_transition_count is not None and primitive_transition_count != int(
        expected_primitive_transition_count
    ):
        raise RuntimeError(
            "offline success-truncated primitive transition count mismatch: "
            "expected %d, got %d"
            % (int(expected_primitive_transition_count), primitive_transition_count)
        )
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
            "last_frame_inclusive"
            if str(success_mode).lower() == "last_frame"
            else "first_positive_reward_inclusive"
        ),
        terminal_partial_chunks=terminal_partial_chunks,
        terminal_full_chunks=terminal_full_chunks,
        min_macro_decisions_per_episode=min(macro_decisions_per_episode),
        max_macro_decisions_per_episode=max(macro_decisions_per_episode),
        primitive_transitions=primitive_transition_count,
    )
