"""Build Adroit DICE replay from paired robomimic and Minari datasets.

The alignment and boundary audit is copied from the validated Adroit ResFiT
transfer.  DICE stores one non-overlapping fixed-TA transition per decision,
uses undiscounted primitive ``-1/0`` reward sums, and computes returns with its
existing macro-decision n-step / Monte-Carlo accumulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from baselines.dice.adroit.core import (
    CompleteEpisodeAccumulator,
    DICEReplayBuffer,
    MacroTransition,
)


@dataclass
class AdroitDICEOfflineSummary:
    episodes: int
    source_episodes: int
    transitions: int
    success_episodes: int
    timeout_episodes: int
    failure_terminal_episodes: int
    primitive_transitions: int
    terminal_partial_chunks: int
    terminal_full_chunks: int
    action_alignment_max_abs: float
    state_alignment_max_abs: float
    reward_alignment_max_abs: float
    success_horizon: int
    success_mode: str = "minari_infos"
    stop_rule: str = "first_success_or_native_boundary_or_horizon_inclusive"
    hdf5_state_used: bool = True


def _ordered_names(group, prefix: str):
    names = sorted(group, key=lambda name: int(name[len(prefix) :]))
    if names != [prefix + str(index) for index in range(len(names))]:
        raise ValueError("episodes must have contiguous numeric IDs: %s" % prefix)
    return names


def read_aligned_adroit_episode(
    demo, source, *, state_dim: int, action_dim: int, env_horizon: int
) -> Dict[str, object]:
    """Validate every source row, then return the causally valid prefix."""

    actions = np.asarray(demo["actions"], dtype=np.float32)
    states = np.asarray(demo["obs/flat"], dtype=np.float32)
    next_states = np.asarray(demo["next_obs/flat"], dtype=np.float32)
    length = int(len(actions))
    if (
        length < 1
        or actions.shape != (length, int(action_dim))
        or states.shape != (length, int(state_dim))
        or next_states.shape != states.shape
    ):
        raise ValueError("invalid Adroit demonstration shape")
    source_states = np.asarray(source["observations"], dtype=np.float32)
    source_actions = np.asarray(source["actions"], dtype=np.float32)
    if source_states.shape != (length + 1, int(state_dim)):
        raise ValueError("Minari observations must include the final state")
    if source_actions.shape != actions.shape:
        raise ValueError("Minari action shape mismatch")

    state_error = max(
        float(np.max(np.abs(states - source_states[:-1]))),
        float(np.max(np.abs(next_states - source_states[1:]))),
    )
    action_error = float(np.max(np.abs(actions - source_actions)))
    demo_rewards = np.asarray(demo["rewards"], dtype=np.float32)
    source_rewards = np.asarray(source["rewards"], dtype=np.float32)
    if demo_rewards.shape != (length,) or source_rewards.shape != (length,):
        raise ValueError("Adroit reward alignment shape mismatch")
    reward_error = float(np.max(np.abs(demo_rewards - source_rewards)))
    if not all(np.isfinite(value) for value in (state_error, action_error, reward_error)):
        raise ValueError("Adroit paired dataset contains NaN or Inf")
    if state_error != 0.0 or action_error != 0.0 or reward_error != 0.0:
        raise ValueError(
            "BC/Minari alignment failed: state=%g action=%g reward=%g"
            % (state_error, action_error, reward_error)
        )

    if "infos/success" in source:
        success = np.asarray(source["infos/success"], dtype=bool)
    elif "infos/is_success" in source:
        success = np.asarray(source["infos/is_success"], dtype=bool)
    else:
        raise ValueError("Minari source is missing explicit success; dense reward is invalid")
    if success.shape != (length + 1,):
        raise ValueError("Minari success must include reset and post-action infos")
    success = success[1:]
    terminated = np.asarray(source["terminations"], dtype=bool)
    truncated = np.asarray(source["truncations"], dtype=bool)
    if terminated.shape != (length,) or truncated.shape != (length,):
        raise ValueError("Minari boundary arrays disagree with actions")
    expected_dones = terminated | truncated
    expected_dones[-1] = True
    if not np.array_equal(np.asarray(demo["dones"], dtype=bool), expected_dones):
        raise ValueError("BC/Minari done alignment failed")

    stop = min(length, int(env_horizon))
    boundaries = np.flatnonzero((success | terminated | truncated)[:stop])
    if boundaries.size:
        stop = int(boundaries[0]) + 1
    terminal_success = bool(success[stop - 1])
    terminal = terminal_success or bool(terminated[stop - 1] and not truncated[stop - 1])
    timeout = not terminal
    return {
        "states": states[:stop],
        "next_states": next_states[:stop],
        "actions": actions[:stop],
        "success": success[:stop],
        "terminated": terminal,
        "truncated": timeout,
        "length": stop,
        "success_episode": terminal_success,
        "state_error": state_error,
        "action_error": action_error,
        "reward_error": reward_error,
    }


def build_adroit_dice_offline_replay(
    *,
    demo_hdf5: str,
    minari_hdf5: str,
    prior_sampler,
    replay: DICEReplayBuffer,
    horizon_steps: int = 32,
    state_dim: int,
    action_dim: int,
    env_horizon: int = 200,
    gamma: float = 0.997,
    n_step: int = 3,
    suffix_noise_seed: int = 0,
    expected_episodes: Optional[int] = None,
    expected_transitions: Optional[int] = None,
    expected_primitive_transitions: Optional[int] = None,
    prior_batch_size: int = 256,
) -> AdroitDICEOfflineSummary:
    import h5py

    if len(replay) != 0:
        raise ValueError("Adroit offline replay must start empty")
    horizon = int(horizon_steps)
    if int(prior_sampler.horizon_steps) != horizon:
        raise ValueError("Adroit prior/offline horizon mismatch")
    if int(prior_sampler.ac_dim) != int(action_dim):
        raise ValueError("Adroit prior/offline action dimension mismatch")
    if int(prior_batch_size) <= 0:
        raise ValueError("Adroit offline prior_batch_size must be positive")

    accumulator = CompleteEpisodeAccumulator(n_step=int(n_step), gamma=float(gamma), n_envs=1)
    noise_rng = np.random.RandomState(int(suffix_noise_seed))
    successes = 0
    timeouts = 0
    failures = 0
    primitive_count = 0
    partial_chunks = 0
    full_chunks = 0
    state_error = 0.0
    action_error = 0.0
    reward_error = 0.0
    started = time.monotonic()
    processed_episodes = 0
    pending: List[Dict[str, object]] = []
    pending_decisions = 0

    def flush_pending() -> None:
        nonlocal pending, pending_decisions, partial_chunks, full_chunks
        nonlocal processed_episodes
        if not pending:
            return
        decision_states = torch.as_tensor(
            np.concatenate([item["decision_states"] for item in pending], axis=0),
            dtype=torch.float32,
            device=prior_sampler.device,
        )
        suffix_noise = torch.as_tensor(
            np.concatenate([item["suffix_noise"] for item in pending], axis=0),
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
        expected_shape = (
            int(pending_decisions),
            horizon,
            int(action_dim),
        )
        if base_chunks.shape != expected_shape or not np.isfinite(base_chunks).all():
            raise RuntimeError("Adroit prior returned invalid offline suffix chunks")

        chunk_cursor = 0
        for item in pending:
            episode = item["episode"]
            normalized_states = item["normalized_states"]
            normalized_actions = item["normalized_actions"]
            starts = item["starts"]
            count = len(starts)
            episode_base_chunks = base_chunks[chunk_cursor : chunk_cursor + count]
            chunk_cursor += count
            length = int(episode["length"])
            for decision_index, start in enumerate(starts):
                duration = min(horizon, length - start)
                boundary = start + duration == length
                terminated = bool(boundary and episode["terminated"])
                truncated = bool(boundary and episode["truncated"])
                full_action = episode_base_chunks[decision_index].copy()
                full_action[:duration] = normalized_actions[start : start + duration]
                primitive_rewards = np.where(
                    episode["success"][start : start + duration], 0.0, -1.0
                )
                reward = float(np.sum(primitive_rewards, dtype=np.float64))
                if terminated:
                    next_state = np.zeros((int(state_dim),), dtype=np.float32)
                else:
                    raw_next = episode["next_states"][start + duration - 1]
                    next_state = prior_sampler.normalize_raw_state(raw_next)
                transition = MacroTransition(
                    state=np.asarray(normalized_states[start], dtype=np.float32),
                    action=full_action.reshape(-1),
                    reward=reward,
                    next_state=np.asarray(next_state, dtype=np.float32),
                    terminated=terminated,
                    truncated=truncated,
                )
                emitted = accumulator.append(0, transition)
                if len(replay) + len(emitted) > replay.capacity:
                    raise ValueError("offline replay capacity would overwrite Adroit demos")
                replay.extend(emitted)
                if boundary:
                    partial_chunks += int(duration < horizon)
                    full_chunks += int(duration == horizon)

            if accumulator.pending_lengths()[0] != 0:
                raise RuntimeError("Adroit episode boundary did not flush DICE accumulator")
            processed_episodes += 1
            if processed_episodes % 100 == 0:
                print(
                    "[ADROIT DICE OFFLINE] episodes=%d transitions=%d elapsed=%.1fs"
                    % (
                        processed_episodes,
                        len(replay),
                        time.monotonic() - started,
                    ),
                    flush=True,
                )
        if chunk_cursor != int(pending_decisions):
            raise RuntimeError("Adroit offline prior batch accounting mismatch")
        pending = []
        pending_decisions = 0

    with h5py.File(demo_hdf5, "r") as demo_file, h5py.File(minari_hdf5, "r") as source_file:
        demos = demo_file["data"]
        demo_names = _ordered_names(demos, "demo_")
        source_names = _ordered_names(source_file, "episode_")
        if len(demo_names) != len(source_names):
            raise ValueError("BC/Minari episode counts differ")
        source_count = len(source_names)
        if expected_episodes is not None and len(demo_names) != int(expected_episodes):
            raise ValueError("Adroit episode count disagrees with task contract")

        for episode_index, demo_name in enumerate(demo_names):
            episode = read_aligned_adroit_episode(
                demos[demo_name],
                source_file[source_names[episode_index]],
                state_dim=int(state_dim),
                action_dim=int(action_dim),
                env_horizon=int(env_horizon),
            )
            length = int(episode["length"])
            starts = list(range(0, length, horizon))
            normalized_states = prior_sampler.normalize_raw_state(episode["states"])
            normalized_actions = prior_sampler.normalize_raw_action(episode["actions"])
            # Draw each episode's z independently in the original order, then
            # concatenate only for frozen-DP execution. This keeps the exact
            # seeded noise stream while avoiding thousands of tiny GPU calls.
            pending.append(
                {
                    "episode": episode,
                    "normalized_states": normalized_states,
                    "normalized_actions": normalized_actions,
                    "decision_states": np.asarray(normalized_states[starts], dtype=np.float32),
                    "suffix_noise": noise_rng.randn(len(starts), horizon, int(action_dim)).astype(
                        np.float32
                    ),
                    "starts": starts,
                }
            )
            pending_decisions += len(starts)
            if pending_decisions >= int(prior_batch_size):
                flush_pending()
            successes += int(episode["success_episode"])
            timeouts += int(episode["truncated"])
            failures += int(episode["terminated"] and not episode["success_episode"])
            primitive_count += length
            state_error = max(state_error, float(episode["state_error"]))
            action_error = max(action_error, float(episode["action_error"]))
            reward_error = max(reward_error, float(episode["reward_error"]))
        flush_pending()
        print(
            "[ADROIT DICE OFFLINE] episodes=%d/%d transitions=%d elapsed=%.1fs"
            % (
                processed_episodes,
                len(demo_names),
                len(replay),
                time.monotonic() - started,
            ),
            flush=True,
        )

    if expected_transitions is not None and len(replay) != int(expected_transitions):
        raise ValueError(
            "Adroit offline transition count: expected %d, got %d"
            % (int(expected_transitions), len(replay))
        )
    if expected_primitive_transitions is not None and primitive_count != int(
        expected_primitive_transitions
    ):
        raise ValueError(
            "Adroit offline primitive count: expected %d, got %d"
            % (int(expected_primitive_transitions), primitive_count)
        )
    return AdroitDICEOfflineSummary(
        episodes=len(demo_names),
        source_episodes=source_count,
        transitions=len(replay),
        success_episodes=successes,
        timeout_episodes=timeouts,
        failure_terminal_episodes=failures,
        primitive_transitions=primitive_count,
        terminal_partial_chunks=partial_chunks,
        terminal_full_chunks=full_chunks,
        action_alignment_max_abs=action_error,
        state_alignment_max_abs=state_error,
        reward_alignment_max_abs=reward_error,
        success_horizon=int(env_horizon),
    )


def load_adroit_offline_cache(
    path: str,
    replay: DICEReplayBuffer,
    expected_provenance: Dict[str, object],
) -> AdroitDICEOfflineSummary:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("version", -1)) != 1:
        raise RuntimeError("unsupported Adroit DICE offline cache version")
    if payload.get("provenance") != expected_provenance:
        raise RuntimeError("Adroit DICE offline cache provenance mismatch")
    replay.load_state_dict(payload["replay"], strict_capacity=False)
    return AdroitDICEOfflineSummary(**dict(payload["summary"]))
