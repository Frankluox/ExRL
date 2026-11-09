"""Adroit mixed replay adapted from ResFiT's offline chunk dataset builder.

Use the exact BC HDF5 for states/actions and the matching Minari source for
success/termination/truncation. Dense rewards and merged HDF5 dones cannot
identify success. Every transition is checked against the Minari source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import time

import numpy as np
import torch

from baselines.resfit.adroit.dp_chunk import SMDPNStepAccumulator, SMDPTransition


@dataclass
class AdroitOfflineSummary:
    episodes: int
    source_episodes: int
    transitions: int
    success_episodes: int
    timeout_episodes: int
    failure_terminal_episodes: int
    primitive_transitions: int
    terminal_partial_chunks: int
    terminal_full_chunks: int
    max_alignment_error: float
    success_horizon: int
    success_mode: str = "minari_infos"
    stop_rule: str = "first_success_or_native_boundary_or_horizon_inclusive"
    hdf5_state_used: bool = True


def _ordered_names(group, prefix):
    names = sorted(group, key=lambda name: int(name[len(prefix) :]))
    if names != [prefix + str(i) for i in range(len(names))]:
        raise ValueError("episodes must have contiguous numeric IDs: " + prefix)
    return names


def read_aligned_episode(demo, source, state_dim, action_dim, env_horizon):
    """Validate all source rows, then return the causally valid episode prefix."""
    actions = np.asarray(demo["actions"], dtype=np.float32)
    states = np.asarray(demo["obs/flat"], dtype=np.float32)
    next_states = np.asarray(demo["next_obs/flat"], dtype=np.float32)
    length = len(actions)
    if (
        length < 1
        or actions.shape != (length, action_dim)
        or states.shape != (length, state_dim)
        or next_states.shape != states.shape
    ):
        raise ValueError("invalid demonstration shape")
    source_states = np.asarray(source["observations"], dtype=np.float32)
    source_actions = np.asarray(source["actions"], dtype=np.float32)
    if source_states.shape != (length + 1, state_dim) or source_actions.shape != actions.shape:
        raise ValueError("Minari observations must include the final post-action state")
    error = 0.0
    for left, right in (
        (actions, source_actions),
        (states, source_states[:-1]),
        (next_states, source_states[1:]),
        (
            np.asarray(demo["rewards"], dtype=np.float32),
            np.asarray(source["rewards"], dtype=np.float32),
        ),
    ):
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("invalid source alignment shape or non-finite data")
        error = max(error, float(np.max(np.abs(left - right))))
    if error != 0.0:
        raise ValueError("BC/Minari state/action/reward alignment failed: %g" % error)
    if "infos/success" in source:
        success = np.asarray(source["infos/success"], dtype=bool)
    elif "infos/is_success" in source:
        success = np.asarray(source["infos/is_success"], dtype=bool)
    else:
        raise ValueError(
            "Minari source is missing explicit success; dense reward is not a success signal"
        )
    if success.shape != (length + 1,):
        raise ValueError("Minari success must include reset and all post-action infos")
    success = success[1:]  # info[t+1] belongs to action[t], not action[t+1].
    terminated = np.asarray(source["terminations"], dtype=bool)
    truncated = np.asarray(source["truncations"], dtype=bool)
    if terminated.shape != (length,) or truncated.shape != (length,):
        raise ValueError("Minari boundary arrays disagree with actions")
    expected_dones = terminated | truncated
    expected_dones[-1] = True  # Existing Adroit_DP converter's explicit rule.
    if not np.array_equal(np.asarray(demo["dones"], dtype=bool), expected_dones):
        raise ValueError("BC/Minari done alignment failed")
    stop = min(length, int(env_horizon))
    boundaries = np.flatnonzero((success | terminated | truncated)[:stop])
    if boundaries.size:
        stop = int(boundaries[0]) + 1
    terminal_success = bool(success[stop - 1])
    terminal = terminal_success or bool(terminated[stop - 1] and not truncated[stop - 1])
    timeout = not terminal
    return dict(
        states=states[:stop],
        next_states=next_states[:stop],
        actions=actions[:stop],
        success=success[:stop],
        terminated=terminal,
        truncated=timeout,
        length=stop,
        success_episode=terminal_success,
        alignment_error=error,
    )


def build_adroit_offline_replay(
    *,
    demo_hdf5,
    minari_hdf5,
    planner,
    action_scaler,
    state_preprocessor,
    replay,
    env_horizon=200,
    gamma=0.997,
    n_step=1,
    max_episodes=None,
    expected_episodes=None,
    expected_transitions=None,
):
    import h5py

    accumulator = SMDPNStepAccumulator(n_step=n_step, n_envs=1)
    horizon = int(planner.horizon_steps)
    successes = timeouts = failures = primitive = partial = full = 0
    max_error = 0.0
    started = time.monotonic()
    with h5py.File(demo_hdf5, "r") as demos_file, h5py.File(minari_hdf5, "r") as source_file:
        demos = demos_file["data"]
        names = _ordered_names(demos, "demo_")
        source_names = _ordered_names(source_file, "episode_")
        if len(names) != len(source_names):
            raise ValueError("BC/Minari episode counts differ")
        source_count = len(names)
        if max_episodes is not None:
            if max_episodes <= 0:
                raise ValueError("max_episodes must be positive")
            names = names[: int(max_episodes)]
        if expected_episodes is not None and len(names) != expected_episodes:
            raise ValueError("offline episode count disagrees with expected_episodes")
        for episode_index, name in enumerate(names):
            episode = read_aligned_episode(
                demos[name],
                source_file[source_names[episode_index]],
                state_preprocessor.state_dim,
                action_scaler.action_dim,
                env_horizon,
            )
            length = episode["length"]
            starts = list(range(0, length, horizon))
            # Preserve DP ResFiT's per-decision inference order and cached chunks.
            base_chunks = [planner.plan(episode["states"][start][None])[0] for start in starts]
            for decision_index, start in enumerate(starts):
                duration = min(horizon, length - start)
                boundary = start + duration == length
                terminal = boundary and episode["terminated"]
                timeout = boundary and episode["truncated"]
                base_dp = base_chunks[decision_index]
                base_resfit = action_scaler.dp_to_resfit(base_dp)
                state_actor = state_preprocessor.transform(episode["states"][start])
                # Same terminal suffix rule as ResFiT DP: unexecuted suffix is base.
                full_raw = np.asarray(action_scaler.dp_to_raw(base_dp), dtype=np.float32).copy()
                full_raw[:duration] = episode["actions"][start : start + duration]
                full_resfit = action_scaler.raw_to_resfit(full_raw)
                rewards = np.where(episode["success"][start : start + duration], 0.0, -1.0)
                reward = float(np.sum(np.power(float(gamma), np.arange(duration)) * rewards))
                if terminal:
                    next_state = np.zeros_like(state_actor)
                    next_base = np.zeros_like(base_resfit)
                    discount = 0.0
                else:
                    raw_next = episode["next_states"][start + duration - 1]
                    next_state = state_preprocessor.transform(raw_next)
                    next_dp = (
                        planner.plan(raw_next[None], bootstrap=True)[0]
                        if boundary
                        else base_chunks[decision_index + 1]
                    )
                    next_base = action_scaler.dp_to_resfit(next_dp)
                    discount = float(gamma) ** duration
                transition = SMDPTransition(
                    state=state_actor,
                    base_action=base_resfit.reshape(-1),
                    action=full_resfit.reshape(-1),
                    reward=reward,
                    discount=discount,
                    next_state=next_state,
                    next_base_action=next_base.reshape(-1),
                    terminated=terminal,
                    truncated=timeout,
                )
                emitted = accumulator.append(0, transition)
                if len(replay) + len(emitted) > replay.capacity:
                    raise ValueError("offline replay capacity would overwrite demonstrations")
                replay.extend(emitted)
                if boundary:
                    partial += int(duration < horizon)
                    full += int(duration == horizon)
            if accumulator.queues[0]:
                raise RuntimeError("offline boundary failed to flush SMDP queue")
            successes += int(episode["success_episode"])
            timeouts += int(episode["truncated"])
            failures += int(episode["terminated"] and not episode["success_episode"])
            primitive += length
            max_error = max(max_error, episode["alignment_error"])
            if (episode_index + 1) % 100 == 0 or episode_index + 1 == len(names):
                print(
                    "[ADROIT OFFLINE] episodes=%d/%d transitions=%d elapsed=%.1fs"
                    % (episode_index + 1, len(names), len(replay), time.monotonic() - started),
                    flush=True,
                )
    if expected_transitions is not None and len(replay) != expected_transitions:
        raise ValueError("offline transition count disagrees with expected_transitions")
    return AdroitOfflineSummary(
        len(names),
        source_count,
        len(replay),
        successes,
        timeouts,
        failures,
        primitive,
        partial,
        full,
        max_error,
        int(env_horizon),
    )


def save_adroit_cache(path, replay, summary, provenance):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {"provenance": provenance, "summary": asdict(summary), "replay": replay.state_dict()},
        path,
        pickle_protocol=4,
    )


def load_adroit_cache(path, replay, provenance):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["provenance"] != provenance:
        raise ValueError("Adroit cache provenance mismatch")
    replay.load_state_dict(payload["replay"])
    return AdroitOfflineSummary(**payload["summary"])
