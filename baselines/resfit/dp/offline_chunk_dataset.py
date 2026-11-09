"""Build fixed-TA chunk-level ResFiT replay from official robomimic demos."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from baselines.resfit.dp.dp_chunk import (
    DPChunkPlanner,
    SMDPNStepAccumulator,
    SMDPTransition,
)
from baselines.resfit.dp.offline_dataset import (
    OfflineBuildSummary,
    _numeric_demo_names,
    _read_episode_action_and_success,
    sha256_array,
    sha256_file,
)
from baselines.resfit.dp.td3_core import (
    PrimitiveReplayBuffer,
    ResFiTActionScaler,
    StatePreprocessor,
)


def build_offline_chunk_replay(
    official_npz_path: str,
    reward_hdf5_path: str,
    normalization_path: str,
    expected_npz_sha256: Optional[str],
    planner: DPChunkPlanner,
    action_scaler: ResFiTActionScaler,
    state_preprocessor: StatePreprocessor,
    replay: PrimitiveReplayBuffer,
    n_step: int = 1,
    gamma: float = 0.99,
    reward_offset: float = 1.0,
    success_mode: str = "reward",
    success_horizon: int = 400,
    expected_transition_count: Optional[int] = None,
    precomputed_npz_sha256: Optional[str] = None,
    precomputed_hdf5_sha256: Optional[str] = None,
    progress_interval_episodes: int = 25,
) -> OfflineBuildSummary:
    """Aggregate demos into fixed-TA SMDP decisions.

    Official NPZ supplies every state.  HDF5 supplies only raw action/reward/
    done arrays; its state datasets are never read.  For a terminal partial
    chunk, the unexecuted suffix is the frozen base-policy suffix (zero
    residual), which is the only commanded suffix available without inventing
    demonstration actions.
    """
    import h5py  # Mixed-only local import preserves online-mode isolation.

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
    horizon = int(planner.horizon_steps)
    accumulator = SMDPNStepAccumulator(n_step=n_step, n_envs=1)
    episode_count = 0
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
            decision_starts = list(range(0, first_success + 1, horizon))
            macro_decisions_per_episode.append(len(decision_starts))
            terminal_duration = first_success + 1 - decision_starts[-1]
            terminal_partial_chunks += int(terminal_duration < horizon)
            terminal_full_chunks += int(terminal_duration == horizon)
            base_chunks_dp = [
                planner.plan(np.asarray(episode_states[start], dtype=np.float32)[None])[0]
                for start in decision_starts
            ]
            for decision_index, start in enumerate(decision_starts):
                duration = min(horizon, first_success + 1 - start)
                terminal = start + duration - 1 == first_success
                state_actor = state_preprocessor.transform(
                    np.asarray(episode_states[start], dtype=np.float32)
                )
                base_dp = np.asarray(base_chunks_dp[decision_index], dtype=np.float32)
                base_resfit = np.asarray(action_scaler.dp_to_resfit(base_dp), dtype=np.float32)

                # The entire fixed-TA command is represented in replay. Only the
                # demonstrated prefix was executed; retain base policy actions
                # for the causally irrelevant terminal suffix.
                full_raw = np.asarray(action_scaler.dp_to_raw(base_dp), dtype=np.float32)
                full_raw[:duration] = np.asarray(
                    raw_actions[start : start + duration], dtype=np.float32
                )
                full_resfit = np.asarray(action_scaler.raw_to_resfit(full_raw), dtype=np.float32)
                within_rewards = np.asarray(
                    rewards[start : start + duration], dtype=np.float64
                ) - float(reward_offset)
                powers = np.power(float(gamma), np.arange(duration, dtype=np.float64))
                chunk_reward = float(np.sum(powers * within_rewards))

                if terminal:
                    next_state_actor = np.zeros_like(state_actor)
                    next_base_resfit = np.zeros_like(base_resfit)
                    discount = 0.0
                else:
                    next_state_actor = state_preprocessor.transform(
                        np.asarray(episode_states[start + duration], dtype=np.float32)
                    )
                    next_base_resfit = np.asarray(
                        action_scaler.dp_to_resfit(base_chunks_dp[decision_index + 1]),
                        dtype=np.float32,
                    )
                    discount = float(gamma) ** duration

                transition = SMDPTransition(
                    state=state_actor,
                    base_action=base_resfit.reshape(-1),
                    action=full_resfit.reshape(-1),
                    reward=chunk_reward,
                    discount=discount,
                    next_state=next_state_actor,
                    next_base_action=next_base_resfit.reshape(-1),
                    terminated=terminal,
                    truncated=False,
                )
                replay.extend(accumulator.append(0, transition))
            if accumulator.queues[0]:
                raise RuntimeError("terminal transition failed to flush offline SMDP queue")
            state_offset += length
            if int(progress_interval_episodes) > 0 and (
                episode_count % int(progress_interval_episodes) == 0 or episode_count == len(demos)
            ):
                print(
                    "[OFFLINE CHUNK CACHE] episodes=%d/%d transitions=%d elapsed_sec=%.1f"
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
        terminal_partial_chunks=terminal_partial_chunks,
        terminal_full_chunks=terminal_full_chunks,
        min_macro_decisions_per_episode=min(macro_decisions_per_episode),
        max_macro_decisions_per_episode=max(macro_decisions_per_episode),
    )
