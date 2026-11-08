"""Online successful-trajectory BC utilities for ExRL-Q runs.

This module intentionally lives next to the SAC code but does not reuse the
SMDP replay buffer. The BC data has low-level state/action windows, while SAC
stores decision-level SMDP transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class BCTensorBatch:
    actions: torch.Tensor
    conditions: Dict[str, torch.Tensor]
    action_is_pad: Optional[torch.Tensor] = None
    episode_ids: Optional[torch.Tensor] = None
    replan_boundaries: Optional[torch.Tensor] = None
    coherent_prefix_steps: Optional[torch.Tensor] = None


_ACT_BC_SCOPE_PREFIXES = {
    "action_head": ("model.action_head.",),
    "decoder_head": (
        "model.decoder.",
        "model.decoder_pos_embed.",
        "model.action_head.",
    ),
}


def select_act_bc_trainable_parameters(
    policy_model: nn.Module,
    scope: str,
) -> Tuple[List[str], List[nn.Parameter]]:
    """Select the ACT parameters updated by online BC without mutating the model."""
    scope = str(scope).strip().lower()
    if scope == "all":
        selected = list(policy_model.named_parameters())
    else:
        prefixes = _ACT_BC_SCOPE_PREFIXES.get(scope)
        if prefixes is None:
            valid = ["all", *_ACT_BC_SCOPE_PREFIXES]
            raise ValueError(
                f"invalid ACT BC trainable scope={scope!r}; expected one of {valid}"
            )
        selected = [
            (name, parameter)
            for name, parameter in policy_model.named_parameters()
            if name.startswith(prefixes)
        ]
    if not selected:
        raise RuntimeError(f"ACT BC trainable scope={scope!r} selected no parameters")
    names, parameters = zip(*selected)
    return list(names), list(parameters)


def ema_update_module_(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    """Update a teacher toward a student while keeping the teacher non-trainable."""
    decay = float(decay)
    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"EMA decay must be in [0, 1], got {decay}")

    teacher_parameters = dict(teacher.named_parameters())
    student_parameters = dict(student.named_parameters())
    if teacher_parameters.keys() != student_parameters.keys():
        raise ValueError("teacher/student parameter structures do not match")
    teacher_buffers = dict(teacher.named_buffers())
    student_buffers = dict(student.named_buffers())
    if teacher_buffers.keys() != student_buffers.keys():
        raise ValueError("teacher/student buffer structures do not match")

    with torch.no_grad():
        for name, teacher_parameter in teacher_parameters.items():
            teacher_parameter.mul_(decay).add_(
                student_parameters[name].detach(),
                alpha=1.0 - decay,
            )
        for name, teacher_buffer in teacher_buffers.items():
            teacher_buffer.copy_(student_buffers[name].detach())


def _as_2d_float(name: str, value: np.ndarray) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={arr.shape}")
    return arr


def _make_state_condition(
    states: np.ndarray,
    start: int,
    cond_steps: int,
    *,
    episode_start: int = 0,
) -> np.ndarray:
    """Match StitchedSequenceDataset history semantics.

    For early-episode starts, missing history is filled by repeating the first
    available state in the episode. The output is ordered from oldest to most
    recent, with the current state at the end.
    """
    out = []
    start_i = int(start)
    episode_start_i = int(episode_start)
    for offset in reversed(range(int(cond_steps))):
        idx = max(start_i - offset, episode_start_i)
        out.append(states[idx])
    return np.stack(out, axis=0).astype(np.float32)


def _make_action_window(
    actions: np.ndarray, start: int, horizon_steps: int, *, padded: bool
) -> np.ndarray:
    start_i = int(start)
    end = start_i + int(horizon_steps)
    window = actions[start_i:end]
    if window.shape[0] == int(horizon_steps):
        return window.astype(np.float32)
    if not padded:
        raise ValueError(
            f"unpadded action window too short: start={start_i} "
            f"got={window.shape[0]} horizon={int(horizon_steps)}"
        )
    if window.shape[0] <= 0:
        raise ValueError(f"empty action window at start={start_i}")
    pad = np.repeat(window[-1:], int(horizon_steps) - window.shape[0], axis=0)
    return np.concatenate([window, pad], axis=0).astype(np.float32)


def _make_action_padding_mask(
    action_count: int,
    start: int,
    horizon_steps: int,
) -> np.ndarray:
    target_indices = int(start) + np.arange(int(horizon_steps), dtype=np.int64)
    return (target_indices >= int(action_count)).astype(np.bool_)


def _decision_start_indices(decision_ids: np.ndarray) -> np.ndarray:
    decision_ids = np.asarray(decision_ids, dtype=np.int64).reshape(-1)
    if decision_ids.size <= 0:
        return np.zeros((0,), dtype=np.int64)
    is_start = np.ones((decision_ids.size,), dtype=np.bool_)
    is_start[1:] = decision_ids[1:] != decision_ids[:-1]
    return np.flatnonzero(is_start).astype(np.int64)


def _make_decision_state_condition(
    states: np.ndarray,
    decision_starts: np.ndarray,
    decision_position: int,
    cond_steps: int,
) -> np.ndarray:
    """Build the observation history seen by a policy called only at AAC decisions."""
    indices = []
    for offset in reversed(range(int(cond_steps))):
        history_position = max(int(decision_position) - offset, 0)
        indices.append(int(decision_starts[history_position]))
    return np.stack([states[index] for index in indices], axis=0).astype(np.float32)


def episode_to_bc_windows(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    horizon_steps: int,
    cond_steps: int,
    padded: bool,
    return_action_is_pad: bool = False,
    decision_ids: Optional[np.ndarray] = None,
    decision_starts_only: bool = False,
    return_replan_boundaries: bool = False,
    return_coherent_prefix_steps: bool = False,
):
    """Convert one successful low-level episode into DP BC windows.

    Args:
        states: normalized low-level state before each executed action, [T, obs_dim].
        actions: actually executed low-level action, [T, action_dim].
        horizon_steps: diffusion prediction/action horizon.
        cond_steps: diffusion observation horizon.
        padded: whether to repeat-pad tail action windows.

    Returns:
        conditions_state: [N, cond_steps, obs_dim]
        action_windows: [N, horizon_steps, action_dim]
        action_is_pad: optional [N, horizon_steps] mask when
            return_action_is_pad=True.
    """
    states = _as_2d_float("states", states)
    actions = _as_2d_float("actions", actions)
    if states.shape[0] != actions.shape[0]:
        raise ValueError(
            f"states/actions length mismatch: {states.shape[0]} vs {actions.shape[0]}"
        )
    decision_ids_array = None
    if decision_ids is not None:
        if torch.is_tensor(decision_ids):
            decision_ids = decision_ids.detach().cpu().numpy()
        decision_ids_array = np.asarray(decision_ids, dtype=np.int64).reshape(-1)
        if int(decision_ids_array.shape[0]) != int(actions.shape[0]):
            raise ValueError(
                "decision_ids/actions length mismatch: "
                f"{decision_ids_array.shape[0]} vs {actions.shape[0]}"
            )
    if int(horizon_steps) <= 0:
        raise ValueError("horizon_steps must be > 0")
    if int(cond_steps) <= 0:
        raise ValueError("cond_steps must be > 0")

    t = int(actions.shape[0])
    if t <= 0:
        empty = (
            np.zeros((0, int(cond_steps), int(states.shape[1])), dtype=np.float32),
            np.zeros((0, int(horizon_steps), int(actions.shape[1])), dtype=np.float32),
        )
        result = empty
        if return_action_is_pad:
            result = (*result, np.zeros((0, int(horizon_steps)), dtype=np.bool_))
        if return_replan_boundaries:
            result = (*result, np.zeros((0,), dtype=np.int16))
        if return_coherent_prefix_steps:
            result = (*result, np.zeros((0,), dtype=np.int16))
        return result

    if padded:
        starts = list(range(0, t))
    else:
        if t < int(horizon_steps):
            empty = (
                np.zeros((0, int(cond_steps), int(states.shape[1])), dtype=np.float32),
                np.zeros(
                    (0, int(horizon_steps), int(actions.shape[1])), dtype=np.float32
                ),
            )
            result = empty
            if return_action_is_pad:
                result = (*result, np.zeros((0, int(horizon_steps)), dtype=np.bool_))
            if return_replan_boundaries:
                result = (*result, np.zeros((0,), dtype=np.int16))
            if return_coherent_prefix_steps:
                result = (*result, np.zeros((0,), dtype=np.int16))
            return result
        starts = list(range(0, t - int(horizon_steps) + 1))

    all_decision_starts = None
    decision_positions = None
    if bool(decision_starts_only):
        if decision_ids_array is None:
            raise ValueError("decision_starts_only requires decision_ids")
        all_decision_starts = _decision_start_indices(decision_ids_array)
        allowed_starts = set(int(index) for index in starts)
        starts = [
            int(index) for index in all_decision_starts if int(index) in allowed_starts
        ]
        decision_positions = {
            int(index): position
            for position, index in enumerate(all_decision_starts.tolist())
        }

    conds = []
    chunks = []
    action_is_pad = []
    replan_boundaries = []
    coherent_prefix_steps = []
    for start in starts:
        if bool(decision_starts_only):
            conds.append(
                _make_decision_state_condition(
                    states,
                    all_decision_starts,
                    decision_positions[int(start)],
                    int(cond_steps),
                )
            )
        else:
            conds.append(
                _make_state_condition(states, start, int(cond_steps), episode_start=0)
            )
        chunks.append(
            _make_action_window(actions, start, int(horizon_steps), padded=bool(padded))
        )
        action_is_pad.append(_make_action_padding_mask(t, start, int(horizon_steps)))
        if decision_ids_array is None:
            replan_boundaries.append(-1)
            coherent_prefix_steps.append(-1)
        else:
            ids = decision_ids_array[start : start + int(horizon_steps)]
            if ids.shape[0] < int(horizon_steps):
                ids = np.pad(ids, (0, int(horizon_steps) - ids.shape[0]), mode="edge")
            change_positions = np.flatnonzero(ids[1:] != ids[:-1]) + 1
            replan_boundaries.append(int(change_positions.size))
            coherent_prefix_steps.append(
                int(change_positions[0])
                if change_positions.size > 0
                else int(horizon_steps)
            )
    result = (
        np.stack(conds, axis=0).astype(np.float32),
        np.stack(chunks, axis=0).astype(np.float32),
    )
    if return_action_is_pad:
        result = (*result, np.stack(action_is_pad, axis=0).astype(np.bool_))
    if return_replan_boundaries:
        result = (*result, np.asarray(replan_boundaries, dtype=np.int16))
    if return_coherent_prefix_steps:
        result = (*result, np.asarray(coherent_prefix_steps, dtype=np.int16))
    return result


class OnlineSuccessBCReplay:
    """Fixed-capacity ring buffer of online successful BC windows.

    Windows retain their source episode id so callers can either sample windows
    uniformly (legacy behavior) or sample an episode uniformly before choosing
    a window from that episode.
    """

    def __init__(
        self,
        *,
        capacity: int,
        horizon_steps: int,
        action_dim: int,
        cond_steps: int,
        obs_dim: int,
        device: torch.device,
        padded: bool,
        decision_starts_only: bool = False,
    ):
        self.capacity = int(capacity)
        self.horizon_steps = int(horizon_steps)
        self.action_dim = int(action_dim)
        self.cond_steps = int(cond_steps)
        self.obs_dim = int(obs_dim)
        self.device = torch.device(device)
        self.padded = bool(padded)
        self.decision_starts_only = bool(decision_starts_only)
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        if min(self.horizon_steps, self.action_dim, self.cond_steps, self.obs_dim) <= 0:
            raise ValueError(
                "horizon_steps/action_dim/cond_steps/obs_dim must all be positive, got "
                f"{self.horizon_steps}/{self.action_dim}/{self.cond_steps}/{self.obs_dim}"
            )
        self.conditions = np.zeros(
            (self.capacity, self.cond_steps, self.obs_dim), dtype=np.float32
        )
        self.actions = np.zeros(
            (self.capacity, self.horizon_steps, self.action_dim), dtype=np.float32
        )
        self.action_is_pad = np.zeros(
            (self.capacity, self.horizon_steps), dtype=np.bool_
        )
        self.episode_ids = np.full((self.capacity,), -1, dtype=np.int64)
        self.replan_boundaries = np.full((self.capacity,), -1, dtype=np.int16)
        self.coherent_prefix_steps = np.full((self.capacity,), -1, dtype=np.int16)
        self.pos = 0
        self.size = 0
        self.total_windows_added = 0
        self.success_episodes_added = 0
        self.next_episode_id = 0
        self._episode_slots: Dict[int, Set[int]] = {}
        self._episode_slot_arrays: Dict[int, np.ndarray] = {}
        self._active_episode_ids_cache: Optional[np.ndarray] = None
        self._filtered_window_indices_cache: Dict[int, np.ndarray] = {}
        self._filtered_episode_slots_cache: Dict[int, Dict[int, np.ndarray]] = {}
        self._filtered_active_episode_ids_cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return int(self.size)

    def can_sample(
        self,
        batch_size: int,
        *,
        max_replan_boundaries: Optional[int] = None,
    ) -> bool:
        batch_size = int(batch_size)
        if batch_size <= 0:
            return False
        return self.eligible_window_count(max_replan_boundaries) >= batch_size

    @property
    def active_episode_count(self) -> int:
        return int(len(self._episode_slots))

    def _invalidate_episode_sampling_cache(self) -> None:
        self._episode_slot_arrays.clear()
        self._active_episode_ids_cache = None
        self._filtered_window_indices_cache.clear()
        self._filtered_episode_slots_cache.clear()
        self._filtered_active_episode_ids_cache.clear()

    def _rebuild_episode_index(self) -> None:
        slots: Dict[int, Set[int]] = {}
        for slot in range(int(self.size)):
            episode_id = int(self.episode_ids[slot])
            if episode_id >= 0:
                slots.setdefault(episode_id, set()).add(slot)
        self._episode_slots = slots
        self._invalidate_episode_sampling_cache()

    def add_windows(
        self,
        conditions_state: np.ndarray,
        action_windows: np.ndarray,
        *,
        action_is_pad: Optional[np.ndarray] = None,
        replan_boundaries: Optional[np.ndarray] = None,
        coherent_prefix_steps: Optional[np.ndarray] = None,
        episode_id: Optional[int] = None,
    ) -> int:
        conditions_state = _as_2d_or_3d("conditions_state", conditions_state, ndim=3)
        action_windows = _as_2d_or_3d("action_windows", action_windows, ndim=3)
        if conditions_state.shape[0] != action_windows.shape[0]:
            raise ValueError(
                f"window count mismatch: cond={conditions_state.shape[0]} action={action_windows.shape[0]}"
            )
        if tuple(conditions_state.shape[1:]) != (self.cond_steps, self.obs_dim):
            raise ValueError(
                f"condition shape mismatch: got {conditions_state.shape[1:]}, "
                f"expected {(self.cond_steps, self.obs_dim)}"
            )
        if tuple(action_windows.shape[1:]) != (self.horizon_steps, self.action_dim):
            raise ValueError(
                f"action window shape mismatch: got {action_windows.shape[1:]}, "
                f"expected {(self.horizon_steps, self.action_dim)}"
            )
        if action_is_pad is None:
            action_is_pad = np.zeros(
                (int(action_windows.shape[0]), self.horizon_steps),
                dtype=np.bool_,
            )
        else:
            if torch.is_tensor(action_is_pad):
                action_is_pad = action_is_pad.detach().cpu().numpy()
            action_is_pad = np.asarray(action_is_pad, dtype=np.bool_)
            expected_pad_shape = (int(action_windows.shape[0]), self.horizon_steps)
            if tuple(action_is_pad.shape) != expected_pad_shape:
                raise ValueError(
                    f"action_is_pad shape mismatch: got {action_is_pad.shape}, "
                    f"expected {expected_pad_shape}"
                )
        if coherent_prefix_steps is None:
            coherent_prefix_steps = np.full(
                (int(action_windows.shape[0]),), -1, dtype=np.int16
            )
        else:
            if torch.is_tensor(coherent_prefix_steps):
                coherent_prefix_steps = coherent_prefix_steps.detach().cpu().numpy()
            coherent_prefix_steps = np.asarray(
                coherent_prefix_steps, dtype=np.int16
            ).reshape(-1)
            if int(coherent_prefix_steps.shape[0]) != int(action_windows.shape[0]):
                raise ValueError(
                    "coherent prefix count mismatch: "
                    f"got {coherent_prefix_steps.shape[0]}, expected {action_windows.shape[0]}"
                )
            known = coherent_prefix_steps >= 0
            if np.any(coherent_prefix_steps[known] < 1) or np.any(
                coherent_prefix_steps[known] > self.horizon_steps
            ):
                raise ValueError(
                    f"coherent prefix steps must be in [1, {self.horizon_steps}] or -1"
                )
        if replan_boundaries is None:
            replan_boundaries = np.full(
                (int(action_windows.shape[0]),), -1, dtype=np.int16
            )
        else:
            if torch.is_tensor(replan_boundaries):
                replan_boundaries = replan_boundaries.detach().cpu().numpy()
            replan_boundaries = np.asarray(replan_boundaries, dtype=np.int16).reshape(
                -1
            )
            if int(replan_boundaries.shape[0]) != int(action_windows.shape[0]):
                raise ValueError(
                    "replan boundary count mismatch: "
                    f"got {replan_boundaries.shape[0]}, expected {action_windows.shape[0]}"
                )

        n = int(action_windows.shape[0])
        if n <= 0:
            return 0
        if n >= self.capacity:
            conditions_state = conditions_state[-self.capacity :]
            action_windows = action_windows[-self.capacity :]
            action_is_pad = action_is_pad[-self.capacity :]
            replan_boundaries = replan_boundaries[-self.capacity :]
            coherent_prefix_steps = coherent_prefix_steps[-self.capacity :]
            n = self.capacity

        if episode_id is None:
            episode_id = int(self.next_episode_id)
            self.next_episode_id += 1
        else:
            episode_id = int(episode_id)
            if episode_id < 0:
                raise ValueError(f"episode_id must be >= 0, got {episode_id}")
            self.next_episode_id = max(int(self.next_episode_id), episode_id + 1)

        write_slots = (np.arange(n, dtype=np.int64) + int(self.pos)) % int(
            self.capacity
        )
        for slot_raw in write_slots:
            slot = int(slot_raw)
            old_episode_id = int(self.episode_ids[slot])
            if old_episode_id >= 0:
                old_slots = self._episode_slots.get(old_episode_id)
                if old_slots is not None:
                    old_slots.discard(slot)
                    if not old_slots:
                        self._episode_slots.pop(old_episode_id, None)

        self.conditions[write_slots] = conditions_state
        self.actions[write_slots] = action_windows
        self.action_is_pad[write_slots] = action_is_pad
        self.episode_ids[write_slots] = episode_id
        self.replan_boundaries[write_slots] = replan_boundaries
        self.coherent_prefix_steps[write_slots] = coherent_prefix_steps
        self._episode_slots.setdefault(episode_id, set()).update(
            int(slot) for slot in write_slots
        )
        self._invalidate_episode_sampling_cache()

        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.capacity, self.size + n)
        self.total_windows_added += n
        return n

    def add_success_episode(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        decision_ids: Optional[np.ndarray] = None,
    ) -> int:
        conds, chunks, action_is_pad, replan_boundaries, coherent_prefix_steps = (
            episode_to_bc_windows(
                states,
                actions,
                horizon_steps=self.horizon_steps,
                cond_steps=self.cond_steps,
                padded=self.padded,
                return_action_is_pad=True,
                decision_ids=decision_ids,
                decision_starts_only=self.decision_starts_only,
                return_replan_boundaries=True,
                return_coherent_prefix_steps=True,
            )
        )
        added = self.add_windows(
            conds,
            chunks,
            action_is_pad=action_is_pad,
            replan_boundaries=replan_boundaries,
            coherent_prefix_steps=coherent_prefix_steps,
        )
        if added > 0:
            self.success_episodes_added += 1
        return int(added)

    @staticmethod
    def _normalize_max_replan_boundaries(value: Optional[int]) -> Optional[int]:
        if value is None or int(value) < 0:
            return None
        return int(value)

    def _eligible_window_indices(
        self, max_replan_boundaries: Optional[int]
    ) -> np.ndarray:
        threshold = self._normalize_max_replan_boundaries(max_replan_boundaries)
        if threshold is None:
            return np.arange(int(self.size), dtype=np.int64)
        cached = self._filtered_window_indices_cache.get(threshold)
        if cached is None:
            values = self.replan_boundaries[: self.size]
            # Unknown legacy values (-1) cannot satisfy a coherence filter.
            cached = np.flatnonzero((values >= 0) & (values <= threshold)).astype(
                np.int64
            )
            self._filtered_window_indices_cache[threshold] = cached
        return cached

    def eligible_window_count(self, max_replan_boundaries: Optional[int] = None) -> int:
        return int(self._eligible_window_indices(max_replan_boundaries).size)

    def _filtered_episode_index(
        self, threshold: int
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        threshold = int(threshold)
        slots_by_episode = self._filtered_episode_slots_cache.get(threshold)
        active_episode_ids = self._filtered_active_episode_ids_cache.get(threshold)
        if slots_by_episode is None or active_episode_ids is None:
            grouped: Dict[int, List[int]] = {}
            for slot_raw in self._eligible_window_indices(threshold):
                slot = int(slot_raw)
                episode_id = int(self.episode_ids[slot])
                if episode_id >= 0:
                    grouped.setdefault(episode_id, []).append(slot)
            slots_by_episode = {
                episode_id: np.asarray(slots, dtype=np.int64)
                for episode_id, slots in grouped.items()
            }
            active_episode_ids = np.asarray(sorted(slots_by_episode), dtype=np.int64)
            self._filtered_episode_slots_cache[threshold] = slots_by_episode
            self._filtered_active_episode_ids_cache[threshold] = active_episode_ids
        return active_episode_ids, slots_by_episode

    def _sample_episode_uniform_indices(
        self,
        batch_size: int,
        *,
        max_replan_boundaries: Optional[int] = None,
    ) -> np.ndarray:
        threshold = self._normalize_max_replan_boundaries(max_replan_boundaries)
        if threshold is not None:
            active_episode_ids, episode_slot_arrays = self._filtered_episode_index(
                threshold
            )
            if active_episode_ids.size <= 0:
                raise RuntimeError(
                    "BC replay has no episodes satisfying max_replan_boundaries="
                    f"{threshold}"
                )
            picked_episode_ids = np.random.choice(
                active_episode_ids,
                size=int(batch_size),
                replace=True,
            )
            indices = np.empty((int(batch_size),), dtype=np.int64)
            for episode_id_raw in np.unique(picked_episode_ids):
                episode_id = int(episode_id_raw)
                slots = episode_slot_arrays[episode_id]
                batch_positions = np.flatnonzero(picked_episode_ids == episode_id)
                chosen = np.random.randint(0, len(slots), size=len(batch_positions))
                indices[batch_positions] = slots[chosen]
            return indices

        if self._active_episode_ids_cache is None:
            self._active_episode_ids_cache = np.asarray(
                sorted(self._episode_slots), dtype=np.int64
            )
        if self._active_episode_ids_cache.size <= 0:
            raise RuntimeError("BC replay has no active episode ids")

        picked_episode_ids = np.random.choice(
            self._active_episode_ids_cache,
            size=int(batch_size),
            replace=True,
        )
        indices = np.empty((int(batch_size),), dtype=np.int64)
        for episode_id_raw in np.unique(picked_episode_ids):
            episode_id = int(episode_id_raw)
            slots = self._episode_slot_arrays.get(episode_id)
            if slots is None:
                slots = np.asarray(
                    sorted(self._episode_slots[episode_id]), dtype=np.int64
                )
                self._episode_slot_arrays[episode_id] = slots
            batch_positions = np.flatnonzero(picked_episode_ids == episode_id)
            chosen = np.random.randint(0, len(slots), size=len(batch_positions))
            indices[batch_positions] = slots[chosen]
        return indices

    def sample(
        self,
        batch_size: int,
        *,
        sampling: str = "uniform_window",
        max_replan_boundaries: Optional[int] = None,
    ) -> BCTensorBatch:
        bs = int(batch_size)
        threshold = self._normalize_max_replan_boundaries(max_replan_boundaries)
        eligible_count = self.eligible_window_count(threshold)
        if eligible_count < bs or bs <= 0:
            raise RuntimeError(
                "not enough eligible BC replay windows: "
                f"size={self.size} eligible={eligible_count} batch={bs} "
                f"max_replan_boundaries={threshold}"
            )
        sampling_mode = str(sampling)
        if sampling_mode == "uniform_window":
            eligible = self._eligible_window_indices(threshold)
            idx = eligible[np.random.randint(0, eligible.size, size=bs)]
        elif sampling_mode == "uniform_episode":
            idx = self._sample_episode_uniform_indices(
                bs,
                max_replan_boundaries=threshold,
            )
        else:
            raise ValueError(
                f"unknown BC replay sampling mode {sampling_mode!r}; "
                "expected 'uniform_window' or 'uniform_episode'"
            )
        actions = torch.as_tensor(
            self.actions[idx], device=self.device, dtype=torch.float32
        )
        state = torch.as_tensor(
            self.conditions[idx], device=self.device, dtype=torch.float32
        )
        action_is_pad = torch.as_tensor(
            self.action_is_pad[idx], device=self.device, dtype=torch.bool
        )
        episode_ids = torch.as_tensor(
            self.episode_ids[idx], device=self.device, dtype=torch.long
        )
        replan_boundaries = torch.as_tensor(
            self.replan_boundaries[idx], device=self.device, dtype=torch.long
        )
        coherent_prefix_steps = torch.as_tensor(
            self.coherent_prefix_steps[idx], device=self.device, dtype=torch.long
        )
        return BCTensorBatch(
            actions=actions,
            conditions={"state": state},
            action_is_pad=action_is_pad,
            episode_ids=episode_ids,
            replan_boundaries=replan_boundaries,
            coherent_prefix_steps=coherent_prefix_steps,
        )

    def replan_boundary_stats(self) -> Dict[str, float]:
        values = self.replan_boundaries[: self.size]
        values = values[values >= 0]
        if values.size <= 0:
            return {
                "mean": float("nan"),
                "p50": float("nan"),
                "p90": float("nan"),
                "max": float("nan"),
            }
        return {
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "max": float(np.max(values)),
        }

    def state_dict(
        self,
        *,
        copy_arrays: bool = True,
        as_torch: bool = False,
    ) -> Dict:
        def stored(array):
            value = array[: self.size]
            if as_torch:
                return torch.from_numpy(value)
            return value.copy() if copy_arrays else value

        return {
            "capacity": int(self.capacity),
            "horizon_steps": int(self.horizon_steps),
            "action_dim": int(self.action_dim),
            "cond_steps": int(self.cond_steps),
            "obs_dim": int(self.obs_dim),
            "padded": bool(self.padded),
            "decision_starts_only": bool(self.decision_starts_only),
            "pos": int(self.pos),
            "size": int(self.size),
            "conditions": stored(self.conditions),
            "actions": stored(self.actions),
            "action_is_pad": stored(self.action_is_pad),
            "episode_ids": stored(self.episode_ids),
            "replan_boundaries": stored(self.replan_boundaries),
            "coherent_prefix_steps": stored(self.coherent_prefix_steps),
            "next_episode_id": int(self.next_episode_id),
            "total_windows_added": int(self.total_windows_added),
            "success_episodes_added": int(self.success_episodes_added),
        }

    def load_state_dict(self, state: Dict) -> None:
        if not state:
            return
        expected = {
            "horizon_steps": self.horizon_steps,
            "action_dim": self.action_dim,
            "cond_steps": self.cond_steps,
            "obs_dim": self.obs_dim,
        }
        for key, value in expected.items():
            if int(state.get(key, value)) != int(value):
                raise ValueError(
                    f"BC replay {key} mismatch: ckpt={state.get(key)} current={value}"
                )
        if bool(state.get("padded", self.padded)) != bool(self.padded):
            raise ValueError(
                f"BC replay padded mismatch: ckpt={state.get('padded')} current={self.padded}"
            )
        saved_decision_starts_only = bool(state.get("decision_starts_only", False))
        if saved_decision_starts_only != bool(self.decision_starts_only):
            raise ValueError(
                "BC replay decision_starts_only mismatch: "
                f"ckpt={saved_decision_starts_only} current={self.decision_starts_only}"
            )
        conditions_raw = state.get("conditions", [])
        actions_raw = state.get("actions", [])
        if torch.is_tensor(conditions_raw):
            conditions_raw = conditions_raw.detach().cpu().numpy()
        if torch.is_tensor(actions_raw):
            actions_raw = actions_raw.detach().cpu().numpy()
        conditions = np.asarray(conditions_raw, dtype=np.float32)
        actions = np.asarray(actions_raw, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.conditions.fill(0.0)
        self.actions.fill(0.0)
        self.action_is_pad.fill(False)
        self.episode_ids.fill(-1)
        self.replan_boundaries.fill(-1)
        self.coherent_prefix_steps.fill(-1)
        self._episode_slots.clear()
        self._invalidate_episode_sampling_cache()
        n = min(
            int(state.get("size", len(actions))), int(actions.shape[0]), self.capacity
        )
        if n > 0:
            if conditions.ndim != 3 or actions.ndim != 3:
                raise ValueError(
                    f"invalid BC replay checkpoint arrays: conditions={conditions.shape} actions={actions.shape}"
                )
            stored_episode_ids = state.get("episode_ids")
            if stored_episode_ids is None:
                episode_ids = np.arange(int(actions.shape[0]), dtype=np.int64)
            else:
                if torch.is_tensor(stored_episode_ids):
                    stored_episode_ids = stored_episode_ids.detach().cpu().numpy()
                episode_ids = np.asarray(stored_episode_ids, dtype=np.int64)
                if episode_ids.ndim != 1 or int(episode_ids.shape[0]) < n:
                    raise ValueError(
                        f"invalid BC replay episode_ids shape={episode_ids.shape} for n={n}"
                    )

            stored_action_is_pad = state.get("action_is_pad")
            if stored_action_is_pad is None:
                action_is_pad = np.zeros(
                    (int(actions.shape[0]), self.horizon_steps),
                    dtype=np.bool_,
                )
            else:
                if torch.is_tensor(stored_action_is_pad):
                    stored_action_is_pad = stored_action_is_pad.detach().cpu().numpy()
                action_is_pad = np.asarray(stored_action_is_pad, dtype=np.bool_)
                expected_shape = (int(actions.shape[0]), self.horizon_steps)
                if tuple(action_is_pad.shape) != expected_shape:
                    raise ValueError(
                        f"invalid BC replay action_is_pad shape={action_is_pad.shape}; "
                        f"expected {expected_shape}"
                    )

            stored_replan_boundaries = state.get("replan_boundaries")
            if stored_replan_boundaries is None:
                replan_boundaries = np.full(
                    (int(actions.shape[0]),), -1, dtype=np.int16
                )
            else:
                if torch.is_tensor(stored_replan_boundaries):
                    stored_replan_boundaries = (
                        stored_replan_boundaries.detach().cpu().numpy()
                    )
                replan_boundaries = np.asarray(
                    stored_replan_boundaries, dtype=np.int16
                ).reshape(-1)
                if int(replan_boundaries.shape[0]) < n:
                    raise ValueError(
                        "invalid BC replay replan_boundaries shape="
                        f"{replan_boundaries.shape} for n={n}"
                    )

            stored_coherent_prefix_steps = state.get("coherent_prefix_steps")
            if stored_coherent_prefix_steps is None:
                coherent_prefix_steps = np.full(
                    (int(actions.shape[0]),), -1, dtype=np.int16
                )
            else:
                if torch.is_tensor(stored_coherent_prefix_steps):
                    stored_coherent_prefix_steps = (
                        stored_coherent_prefix_steps.detach().cpu().numpy()
                    )
                coherent_prefix_steps = np.asarray(
                    stored_coherent_prefix_steps, dtype=np.int16
                ).reshape(-1)
                if int(coherent_prefix_steps.shape[0]) < n:
                    raise ValueError(
                        "invalid BC replay coherent_prefix_steps shape="
                        f"{coherent_prefix_steps.shape} for n={n}"
                    )

            stored_capacity = int(state.get("capacity", int(actions.shape[0])))
            stored_size = min(
                int(state.get("size", int(actions.shape[0]))),
                int(actions.shape[0]),
            )
            stored_pos = int(state.get("pos", stored_size)) % max(stored_capacity, 1)
            if stored_size == self.capacity and stored_capacity == self.capacity:
                source_idx = np.arange(n, dtype=np.int64)
                self.pos = stored_pos
            else:
                if stored_size >= stored_capacity and stored_size > 0:
                    logical_idx = np.concatenate(
                        [
                            np.arange(stored_pos, stored_size, dtype=np.int64),
                            np.arange(0, stored_pos, dtype=np.int64),
                        ]
                    )
                else:
                    logical_idx = np.arange(stored_size, dtype=np.int64)
                source_idx = logical_idx[-n:]
                self.pos = n % self.capacity
            self.conditions[:n] = conditions[source_idx]
            self.actions[:n] = actions[source_idx]
            self.action_is_pad[:n] = action_is_pad[source_idx]
            self.episode_ids[:n] = episode_ids[source_idx]
            self.replan_boundaries[:n] = replan_boundaries[source_idx]
            self.coherent_prefix_steps[:n] = coherent_prefix_steps[source_idx]
            self.size = n
            self._rebuild_episode_index()
        else:
            self.pos = 0
        self.next_episode_id = int(
            state.get(
                "next_episode_id",
                int(np.max(self.episode_ids[: self.size])) + 1 if self.size > 0 else 0,
            )
        )
        self.total_windows_added = int(state.get("total_windows_added", self.size))
        self.success_episodes_added = int(state.get("success_episodes_added", 0))


def _as_2d_or_3d(name: str, value: np.ndarray, *, ndim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != int(ndim):
        raise ValueError(f"{name} must be {int(ndim)}D, got shape={arr.shape}")
    return arr


class BaseBCDatasetSampler:
    """Sampler matching DSRL StitchedSequenceDataset/PaddedStitchedSequenceDataset."""

    def __init__(
        self,
        *,
        dataset_path: str,
        horizon_steps: int,
        cond_steps: int,
        device: torch.device,
        padded: bool,
        max_n_episodes: int = 10000,
    ):
        self.dataset_path = str(dataset_path)
        self.horizon_steps = int(horizon_steps)
        self.cond_steps = int(cond_steps)
        self.device = torch.device(device)
        self.padded = bool(padded)
        self.max_n_episodes = int(max_n_episodes)
        if self.horizon_steps <= 0 or self.cond_steps <= 0:
            raise ValueError("horizon_steps and cond_steps must be > 0")
        dataset = np.load(self.dataset_path, allow_pickle=False)
        traj_lengths = np.asarray(dataset["traj_lengths"], dtype=np.int64)[
            : self.max_n_episodes
        ]
        total_steps = int(np.sum(traj_lengths))
        self.traj_lengths = traj_lengths
        self.states = _as_2d_float("states", dataset["states"][:total_steps])
        self.actions = _as_2d_float("actions", dataset["actions"][:total_steps])
        self.indices = self._make_indices(traj_lengths)
        if not self.indices:
            raise ValueError(
                f"base BC dataset has no valid windows: {self.dataset_path} "
                f"padded={self.padded} horizon={self.horizon_steps}"
            )
        self.obs_dim = int(self.states.shape[1])
        self.action_dim = int(self.actions.shape[1])

    def __len__(self) -> int:
        return len(self.indices)

    def _make_indices(self, traj_lengths: Sequence[int]) -> List[Tuple[int, int]]:
        indices: List[Tuple[int, int]] = []
        cur = 0
        for traj_length_raw in traj_lengths:
            traj_length = int(traj_length_raw)
            if traj_length <= 0:
                continue
            if self.padded:
                for start in range(cur, cur + traj_length):
                    indices.append((start, cur))
            elif traj_length >= self.horizon_steps:
                max_start = cur + traj_length - self.horizon_steps
                for start in range(cur, max_start + 1):
                    indices.append((start, cur))
            cur += traj_length
        return indices

    def sample(self, batch_size: int) -> BCTensorBatch:
        bs = int(batch_size)
        if bs <= 0:
            raise ValueError("batch_size must be > 0")
        picked = np.random.randint(0, len(self.indices), size=bs)
        conds = []
        chunks = []
        for item_idx in picked:
            start, episode_start = self.indices[int(item_idx)]
            conds.append(
                _make_state_condition(
                    self.states,
                    int(start),
                    self.cond_steps,
                    episode_start=int(episode_start),
                )
            )
            chunks.append(
                _make_action_window(
                    self.actions, int(start), self.horizon_steps, padded=self.padded
                )
            )
        actions = torch.as_tensor(
            np.stack(chunks, axis=0), device=self.device, dtype=torch.float32
        )
        state = torch.as_tensor(
            np.stack(conds, axis=0), device=self.device, dtype=torch.float32
        )
        return BCTensorBatch(actions=actions, conditions={"state": state})
