"""Online successful-trajectory BC utilities for DSRL ExRL-Q runs.

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
    episode_ids: Optional[torch.Tensor] = None


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


def episode_to_bc_windows(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    horizon_steps: int,
    cond_steps: int,
    padded: bool,
) -> Tuple[np.ndarray, np.ndarray]:
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
    """
    states = _as_2d_float("states", states)
    actions = _as_2d_float("actions", actions)
    if states.shape[0] != actions.shape[0]:
        raise ValueError(f"states/actions length mismatch: {states.shape[0]} vs {actions.shape[0]}")
    if int(horizon_steps) <= 0:
        raise ValueError("horizon_steps must be > 0")
    if int(cond_steps) <= 0:
        raise ValueError("cond_steps must be > 0")

    t = int(actions.shape[0])
    if t <= 0:
        return (
            np.zeros((0, int(cond_steps), int(states.shape[1])), dtype=np.float32),
            np.zeros((0, int(horizon_steps), int(actions.shape[1])), dtype=np.float32),
        )

    if padded:
        starts = range(0, t)
    else:
        if t < int(horizon_steps):
            return (
                np.zeros((0, int(cond_steps), int(states.shape[1])), dtype=np.float32),
                np.zeros((0, int(horizon_steps), int(actions.shape[1])), dtype=np.float32),
            )
        starts = range(0, t - int(horizon_steps) + 1)

    conds = []
    chunks = []
    for start in starts:
        conds.append(_make_state_condition(states, start, int(cond_steps), episode_start=0))
        chunks.append(_make_action_window(actions, start, int(horizon_steps), padded=bool(padded)))
    return np.stack(conds, axis=0).astype(np.float32), np.stack(chunks, axis=0).astype(np.float32)


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
    ):
        self.capacity = int(capacity)
        self.horizon_steps = int(horizon_steps)
        self.action_dim = int(action_dim)
        self.cond_steps = int(cond_steps)
        self.obs_dim = int(obs_dim)
        self.device = torch.device(device)
        self.padded = bool(padded)
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        if min(self.horizon_steps, self.action_dim, self.cond_steps, self.obs_dim) <= 0:
            raise ValueError(
                "horizon_steps/action_dim/cond_steps/obs_dim must all be positive, got "
                f"{self.horizon_steps}/{self.action_dim}/{self.cond_steps}/{self.obs_dim}"
            )
        self.conditions = np.zeros((self.capacity, self.cond_steps, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(
            (self.capacity, self.horizon_steps, self.action_dim), dtype=np.float32
        )
        self.episode_ids = np.full((self.capacity,), -1, dtype=np.int64)
        self.pos = 0
        self.size = 0
        self.total_windows_added = 0
        self.success_episodes_added = 0
        self.next_episode_id = 0
        self._episode_slots: Dict[int, Set[int]] = {}
        self._episode_slot_arrays: Dict[int, np.ndarray] = {}
        self._active_episode_ids_cache: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.size)

    def can_sample(self, batch_size: int) -> bool:
        return int(self.size) >= int(batch_size) and int(batch_size) > 0

    @property
    def active_episode_count(self) -> int:
        return int(len(self._episode_slots))

    def _invalidate_episode_sampling_cache(self) -> None:
        self._episode_slot_arrays.clear()
        self._active_episode_ids_cache = None

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

        n = int(action_windows.shape[0])
        if n <= 0:
            return 0
        if n >= self.capacity:
            conditions_state = conditions_state[-self.capacity :]
            action_windows = action_windows[-self.capacity :]
            n = self.capacity

        if episode_id is None:
            episode_id = int(self.next_episode_id)
            self.next_episode_id += 1
        else:
            episode_id = int(episode_id)
            if episode_id < 0:
                raise ValueError(f"episode_id must be >= 0, got {episode_id}")
            self.next_episode_id = max(int(self.next_episode_id), episode_id + 1)

        write_slots = (np.arange(n, dtype=np.int64) + int(self.pos)) % int(self.capacity)
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
        self.episode_ids[write_slots] = episode_id
        self._episode_slots.setdefault(episode_id, set()).update(int(slot) for slot in write_slots)
        self._invalidate_episode_sampling_cache()

        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.capacity, self.size + n)
        self.total_windows_added += n
        return n

    def add_success_episode(self, states: np.ndarray, actions: np.ndarray) -> int:
        conds, chunks = episode_to_bc_windows(
            states,
            actions,
            horizon_steps=self.horizon_steps,
            cond_steps=self.cond_steps,
            padded=self.padded,
        )
        added = self.add_windows(conds, chunks)
        if added > 0:
            self.success_episodes_added += 1
        return int(added)

    def _sample_episode_uniform_indices(self, batch_size: int) -> np.ndarray:
        if self._active_episode_ids_cache is None:
            self._active_episode_ids_cache = np.asarray(sorted(self._episode_slots), dtype=np.int64)
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
                slots = np.asarray(sorted(self._episode_slots[episode_id]), dtype=np.int64)
                self._episode_slot_arrays[episode_id] = slots
            batch_positions = np.flatnonzero(picked_episode_ids == episode_id)
            chosen = np.random.randint(0, len(slots), size=len(batch_positions))
            indices[batch_positions] = slots[chosen]
        return indices

    def sample(self, batch_size: int, *, sampling: str = "uniform_window") -> BCTensorBatch:
        bs = int(batch_size)
        if not self.can_sample(bs):
            raise RuntimeError(f"not enough BC replay windows: size={self.size} batch={bs}")
        sampling_mode = str(sampling)
        if sampling_mode == "uniform_window":
            idx = np.random.randint(0, self.size, size=bs)
        elif sampling_mode == "uniform_episode":
            idx = self._sample_episode_uniform_indices(bs)
        else:
            raise ValueError(
                f"unknown BC replay sampling mode {sampling_mode!r}; "
                "expected 'uniform_window' or 'uniform_episode'"
            )
        actions = torch.as_tensor(self.actions[idx], device=self.device, dtype=torch.float32)
        state = torch.as_tensor(self.conditions[idx], device=self.device, dtype=torch.float32)
        episode_ids = torch.as_tensor(self.episode_ids[idx], device=self.device, dtype=torch.long)
        return BCTensorBatch(
            actions=actions,
            conditions={"state": state},
            episode_ids=episode_ids,
        )

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
            "pos": int(self.pos),
            "size": int(self.size),
            "conditions": stored(self.conditions),
            "actions": stored(self.actions),
            "episode_ids": stored(self.episode_ids),
            "next_episode_id": int(self.next_episode_id),
            "total_windows_added": int(self.total_windows_added),
            "success_episodes_added": int(self.success_episodes_added),
        }

    def load_state_dict(self, state: Dict) -> None:
        if not state:
            return

        def as_numpy(value, dtype):
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=dtype)

        expected = {
            "horizon_steps": self.horizon_steps,
            "action_dim": self.action_dim,
            "cond_steps": self.cond_steps,
            "obs_dim": self.obs_dim,
        }
        for key, value in expected.items():
            if int(state.get(key, value)) != int(value):
                raise ValueError(f"BC replay {key} mismatch: ckpt={state.get(key)} current={value}")
        if bool(state.get("padded", self.padded)) != bool(self.padded):
            raise ValueError(
                f"BC replay padded mismatch: ckpt={state.get('padded')} current={self.padded}"
            )
        conditions = as_numpy(state.get("conditions", []), np.float32)
        actions = as_numpy(state.get("actions", []), np.float32)
        self.pos = 0
        self.size = 0
        self.conditions.fill(0.0)
        self.actions.fill(0.0)
        self.episode_ids.fill(-1)
        self._episode_slots.clear()
        self._invalidate_episode_sampling_cache()
        n = min(int(state.get("size", len(actions))), int(actions.shape[0]), self.capacity)
        if n > 0:
            if conditions.ndim != 3 or actions.ndim != 3:
                raise ValueError(
                    f"invalid BC replay checkpoint arrays: conditions={conditions.shape} actions={actions.shape}"
                )
            stored_episode_ids = state.get("episode_ids")
            if stored_episode_ids is None:
                episode_ids = np.arange(int(actions.shape[0]), dtype=np.int64)
            else:
                episode_ids = as_numpy(stored_episode_ids, np.int64)
                if episode_ids.ndim != 1 or int(episode_ids.shape[0]) < n:
                    raise ValueError(
                        f"invalid BC replay episode_ids shape={episode_ids.shape} for n={n}"
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
            self.episode_ids[:n] = episode_ids[source_idx]
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
        traj_lengths = np.asarray(dataset["traj_lengths"], dtype=np.int64)[: self.max_n_episodes]
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
        actions = torch.as_tensor(np.stack(chunks, axis=0), device=self.device, dtype=torch.float32)
        state = torch.as_tensor(np.stack(conds, axis=0), device=self.device, dtype=torch.float32)
        return BCTensorBatch(actions=actions, conditions={"state": state})


class RobomimicHDF5BCSampler:
    """In-memory BC sampler for a low-dimensional robomimic demonstration file.

    The sampler deliberately uses normalization statistics from the policy
    checkpoint instead of recomputing them from the HDF5 file. This keeps
    demo observations in exactly the same coordinate system as online
    successful trajectories collected by the Adroit environment wrapper.
    """

    def __init__(
        self,
        *,
        dataset_path: str,
        obs_key: str,
        obs_normalization_stats: Dict,
        action_normalization_stats: Optional[Dict],
        horizon_steps: int,
        cond_steps: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        padded: bool,
    ):
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("Robomimic demo BC requires h5py") from exc

        self.dataset_path = str(dataset_path)
        self.obs_key = str(obs_key)
        self.horizon_steps = int(horizon_steps)
        self.cond_steps = int(cond_steps)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.padded = bool(padded)
        if self.horizon_steps <= 0 or self.cond_steps <= 0:
            raise ValueError("horizon_steps and cond_steps must be > 0")
        if self.obs_dim <= 0 or self.action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be > 0")

        obs_stats = obs_normalization_stats.get(self.obs_key)
        if not isinstance(obs_stats, dict):
            raise KeyError(
                f"checkpoint is missing observation normalization stats for {self.obs_key!r}"
            )
        obs_offset, obs_scale = self._stats_vectors(
            obs_stats,
            expected_dim=self.obs_dim,
            name=f"obs_normalization_stats[{self.obs_key!r}]",
        )

        action_stats = None
        if isinstance(action_normalization_stats, dict):
            action_stats = action_normalization_stats.get("actions")
        if action_stats is None:
            action_offset = np.zeros((self.action_dim,), dtype=np.float32)
            action_scale = np.ones((self.action_dim,), dtype=np.float32)
        else:
            action_offset, action_scale = self._stats_vectors(
                action_stats,
                expected_dim=self.action_dim,
                name="action_normalization_stats['actions']",
            )

        with h5py.File(self.dataset_path, "r", swmr=True, libver="latest") as hdf5_file:
            if "data" not in hdf5_file:
                raise KeyError(f"robomimic demo file has no 'data' group: {self.dataset_path}")
            demos = list(hdf5_file["data"].keys())
            demos.sort(key=self._demo_sort_key)
            if not demos:
                raise ValueError(f"robomimic demo file has no episodes: {self.dataset_path}")

            lengths = []
            for demo in demos:
                group = hdf5_file[f"data/{demo}"]
                obs_path = f"obs/{self.obs_key}"
                if obs_path not in group or "actions" not in group:
                    raise KeyError(f"demo {demo!r} is missing {obs_path!r} or 'actions'")
                obs_shape = tuple(group[obs_path].shape)
                action_shape = tuple(group["actions"].shape)
                if len(obs_shape) != 2 or obs_shape[1] != self.obs_dim:
                    raise ValueError(
                        f"demo {demo!r} obs shape mismatch: {obs_shape}, "
                        f"expected [T,{self.obs_dim}]"
                    )
                if len(action_shape) != 2 or action_shape[1] != self.action_dim:
                    raise ValueError(
                        f"demo {demo!r} action shape mismatch: {action_shape}, "
                        f"expected [T,{self.action_dim}]"
                    )
                if obs_shape[0] != action_shape[0]:
                    raise ValueError(
                        f"demo {demo!r} length mismatch: obs={obs_shape[0]} "
                        f"actions={action_shape[0]}"
                    )
                lengths.append(int(action_shape[0]))

            self.demo_names = tuple(demos)
            self.traj_lengths = np.asarray(lengths, dtype=np.int64)
            self.episode_offsets = np.concatenate(
                [np.zeros((1,), dtype=np.int64), np.cumsum(self.traj_lengths[:-1])]
            )
            total_steps = int(np.sum(self.traj_lengths))
            self.states = np.empty((total_steps, self.obs_dim), dtype=np.float32)
            self.actions = np.empty((total_steps, self.action_dim), dtype=np.float32)
            for demo_idx, demo in enumerate(self.demo_names):
                start = int(self.episode_offsets[demo_idx])
                end = start + int(self.traj_lengths[demo_idx])
                group = hdf5_file[f"data/{demo}"]
                raw_states = np.asarray(group[f"obs/{self.obs_key}"][()], dtype=np.float32)
                raw_actions = np.asarray(group["actions"][()], dtype=np.float32)
                self.states[start:end] = (raw_states - obs_offset) / obs_scale
                self.actions[start:end] = (raw_actions - action_offset) / action_scale

        if not np.isfinite(self.states).all() or not np.isfinite(self.actions).all():
            raise ValueError(
                f"demo dataset contains non-finite normalized values: {self.dataset_path}"
            )

        if self.padded:
            self.window_counts = self.traj_lengths.copy()
        else:
            self.window_counts = np.maximum(
                self.traj_lengths - self.horizon_steps + 1,
                0,
            )
        self.valid_episode_indices = np.flatnonzero(self.window_counts > 0).astype(np.int64)
        if self.valid_episode_indices.size <= 0:
            raise ValueError(f"robomimic demo file has no sampleable windows: {self.dataset_path}")
        self.valid_window_counts = self.window_counts[self.valid_episode_indices]
        self.cumulative_window_counts = np.cumsum(self.valid_window_counts)
        self.num_windows = int(self.cumulative_window_counts[-1])
        for array in (
            self.states,
            self.actions,
            self.traj_lengths,
            self.episode_offsets,
            self.window_counts,
            self.valid_episode_indices,
            self.valid_window_counts,
            self.cumulative_window_counts,
        ):
            array.setflags(write=False)

    @staticmethod
    def _demo_sort_key(name: str):
        text = str(name)
        try:
            return (0, int(text.rsplit("_", 1)[-1]))
        except ValueError:
            return (1, text)

    @staticmethod
    def _stats_vectors(
        stats: Dict, *, expected_dim: int, name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        offset = np.asarray(stats["offset"], dtype=np.float32).reshape(-1)
        scale = np.asarray(stats["scale"], dtype=np.float32).reshape(-1)
        if offset.shape != (int(expected_dim),) or scale.shape != (int(expected_dim),):
            raise ValueError(
                f"{name} shape mismatch: offset={offset.shape} scale={scale.shape} "
                f"expected={(int(expected_dim),)}"
            )
        if not np.isfinite(offset).all() or not np.isfinite(scale).all():
            raise ValueError(f"{name} contains non-finite values")
        if np.any(scale <= 0.0):
            raise ValueError(f"{name} scale must be strictly positive")
        return offset, scale

    def __len__(self) -> int:
        return int(self.num_windows)

    @property
    def n_episodes(self) -> int:
        return int(len(self.demo_names))

    @property
    def n_transitions(self) -> int:
        return int(self.states.shape[0])

    def sample(self, batch_size: int, *, sampling: str = "uniform_episode") -> BCTensorBatch:
        bs = int(batch_size)
        if bs <= 0:
            raise ValueError("batch_size must be > 0")
        sampling_mode = str(sampling)
        if sampling_mode == "uniform_episode":
            episode_indices = np.random.choice(
                self.valid_episode_indices,
                size=bs,
                replace=True,
            )
            local_starts = np.asarray(
                [np.random.randint(0, int(self.window_counts[idx])) for idx in episode_indices],
                dtype=np.int64,
            )
        elif sampling_mode == "uniform_window":
            flat_indices = np.random.randint(0, self.num_windows, size=bs)
            valid_positions = np.searchsorted(
                self.cumulative_window_counts,
                flat_indices,
                side="right",
            )
            episode_indices = self.valid_episode_indices[valid_positions]
            previous = np.where(
                valid_positions > 0,
                self.cumulative_window_counts[np.maximum(valid_positions - 1, 0)],
                0,
            )
            local_starts = flat_indices - previous
        else:
            raise ValueError(
                f"unknown demo sampling mode {sampling_mode!r}; "
                "expected 'uniform_window' or 'uniform_episode'"
            )

        episode_starts = self.episode_offsets[episode_indices]
        episode_ends = episode_starts + self.traj_lengths[episode_indices]
        starts = episode_starts + local_starts

        history_offsets = np.arange(
            self.cond_steps - 1,
            -1,
            -1,
            dtype=np.int64,
        )
        condition_indices = np.maximum(
            starts[:, None] - history_offsets[None, :],
            episode_starts[:, None],
        )
        action_indices = (
            starts[:, None]
            + np.arange(
                self.horizon_steps,
                dtype=np.int64,
            )[None, :]
        )
        if self.padded:
            action_indices = np.minimum(action_indices, episode_ends[:, None] - 1)

        states = np.asarray(self.states[condition_indices], dtype=np.float32)
        actions = np.asarray(self.actions[action_indices], dtype=np.float32)
        return BCTensorBatch(
            actions=torch.as_tensor(actions, device=self.device, dtype=torch.float32),
            conditions={
                "state": torch.as_tensor(states, device=self.device, dtype=torch.float32),
            },
            episode_ids=torch.as_tensor(
                episode_indices,
                device=self.device,
                dtype=torch.long,
            ),
        )
