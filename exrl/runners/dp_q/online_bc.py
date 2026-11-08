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


def _canonical_state_key(key: str) -> str:
    return ".".join(part for part in str(key).split(".") if part != "_orig_mod")


def canonical_module_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a checkpoint portable between eager and torch.compile modules."""
    out = {}
    for key, value in module.state_dict().items():
        canonical_key = _canonical_state_key(key)
        if canonical_key in out:
            raise RuntimeError(f"duplicate canonical module state key: {canonical_key}")
        out[canonical_key] = value
    return out


def load_module_state_dict_compat(
    module: nn.Module,
    state: Dict[str, torch.Tensor],
    *,
    strict: bool,
) -> Tuple[List[str], List[str]]:
    """Load eager or compiled checkpoints without silently dropping parameters."""
    current_by_canonical = {}
    for key in module.state_dict():
        canonical_key = _canonical_state_key(key)
        if canonical_key in current_by_canonical:
            raise RuntimeError(
                f"duplicate current canonical module key: {canonical_key}"
            )
        current_by_canonical[canonical_key] = key

    incoming_by_canonical = {}
    for key, value in state.items():
        canonical_key = _canonical_state_key(key)
        if canonical_key in incoming_by_canonical:
            raise RuntimeError(
                f"duplicate checkpoint canonical module key: {canonical_key}"
            )
        incoming_by_canonical[canonical_key] = value

    missing = sorted(set(current_by_canonical) - set(incoming_by_canonical))
    unexpected = sorted(set(incoming_by_canonical) - set(current_by_canonical))
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"module state mismatch: missing={missing} unexpected={unexpected}"
        )
    remapped = {
        current_by_canonical[key]: value
        for key, value in incoming_by_canonical.items()
        if key in current_by_canonical
    }
    incompatible = module.load_state_dict(remapped, strict=False)
    runtime_missing = [_canonical_state_key(key) for key in incompatible.missing_keys]
    runtime_unexpected = [
        _canonical_state_key(key) for key in incompatible.unexpected_keys
    ]
    missing = sorted(set(missing).union(runtime_missing))
    unexpected = sorted(set(unexpected).union(runtime_unexpected))
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"module state load mismatch: missing={missing} unexpected={unexpected}"
        )
    return missing, unexpected


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
        raise ValueError(
            f"states/actions length mismatch: {states.shape[0]} vs {actions.shape[0]}"
        )
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
                np.zeros(
                    (0, int(horizon_steps), int(actions.shape[1])), dtype=np.float32
                ),
            )
        starts = range(0, t - int(horizon_steps) + 1)

    conds = []
    chunks = []
    for start in starts:
        conds.append(
            _make_state_condition(states, start, int(cond_steps), episode_start=0)
        )
        chunks.append(
            _make_action_window(actions, start, int(horizon_steps), padded=bool(padded))
        )
    return np.stack(conds, axis=0).astype(np.float32), np.stack(chunks, axis=0).astype(
        np.float32
    )


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
        storage: str = "cpu",
        cuda_max_fraction: float = 0.25,
    ):
        self.capacity = int(capacity)
        self.horizon_steps = int(horizon_steps)
        self.action_dim = int(action_dim)
        self.cond_steps = int(cond_steps)
        self.obs_dim = int(obs_dim)
        self.device = torch.device(device)
        self.padded = bool(padded)
        self.storage = str(storage).lower()
        self.cuda_max_fraction = float(cuda_max_fraction)
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        if min(self.horizon_steps, self.action_dim, self.cond_steps, self.obs_dim) <= 0:
            raise ValueError(
                "horizon_steps/action_dim/cond_steps/obs_dim must all be positive, got "
                f"{self.horizon_steps}/{self.action_dim}/{self.cond_steps}/{self.obs_dim}"
            )
        if self.storage not in {"cpu", "cuda_mirror"}:
            raise ValueError(
                f"unknown BC replay storage {storage!r}; expected 'cpu' or 'cuda_mirror'"
            )
        if self.storage == "cuda_mirror" and self.device.type != "cuda":
            raise ValueError("BC replay cuda_mirror requires a CUDA training device")
        if not 0.0 < self.cuda_max_fraction <= 1.0:
            raise ValueError("cuda_max_fraction must be in (0, 1]")
        self.conditions = np.zeros(
            (self.capacity, self.cond_steps, self.obs_dim), dtype=np.float32
        )
        self.actions = np.zeros(
            (self.capacity, self.horizon_steps, self.action_dim), dtype=np.float32
        )
        self.episode_ids = np.full((self.capacity,), -1, dtype=np.int64)
        self._device_conditions = None
        self._device_actions = None
        self._device_episode_ids = None
        self.device_storage_bytes = 0
        if self.storage == "cuda_mirror":
            self._allocate_cuda_mirror()
        self.pos = 0
        self.size = 0
        self.total_windows_added = 0
        self.success_episodes_added = 0
        self.next_episode_id = 0
        self._episode_slots: Dict[int, Set[int]] = {}
        self._episode_slot_arrays: Dict[int, np.ndarray] = {}
        self._active_episode_ids_cache: Optional[np.ndarray] = None

    def _allocate_cuda_mirror(self) -> None:
        self.device_storage_bytes = int(
            self.capacity
            * (
                self.cond_steps * self.obs_dim * np.dtype(np.float32).itemsize
                + self.horizon_steps * self.action_dim * np.dtype(np.float32).itemsize
                + np.dtype(np.int64).itemsize
            )
        )
        with torch.cuda.device(self.device):
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
        budget_bytes = int(int(free_bytes) * self.cuda_max_fraction)
        if self.device_storage_bytes > budget_bytes:
            raise MemoryError(
                "BC replay CUDA mirror exceeds its free-memory budget: "
                f"required={self.device_storage_bytes} budget={budget_bytes} "
                f"free={int(free_bytes)} fraction={self.cuda_max_fraction:.3f}"
            )
        self._device_conditions = torch.empty(
            (self.capacity, self.cond_steps, self.obs_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self._device_actions = torch.empty(
            (self.capacity, self.horizon_steps, self.action_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self._device_episode_ids = torch.empty(
            (self.capacity,), device=self.device, dtype=torch.long
        )

    def _sync_slots_to_cuda(self, slots: np.ndarray) -> None:
        if self.storage != "cuda_mirror" or int(len(slots)) == 0:
            return
        slot_tensor = torch.as_tensor(slots, device=self.device, dtype=torch.long)
        self._device_conditions.index_copy_(
            0,
            slot_tensor,
            torch.as_tensor(
                self.conditions[slots], device=self.device, dtype=torch.float32
            ),
        )
        self._device_actions.index_copy_(
            0,
            slot_tensor,
            torch.as_tensor(
                self.actions[slots], device=self.device, dtype=torch.float32
            ),
        )
        self._device_episode_ids.index_copy_(
            0,
            slot_tensor,
            torch.as_tensor(
                self.episode_ids[slots], device=self.device, dtype=torch.long
            ),
        )

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
        self.episode_ids[write_slots] = episode_id
        self._sync_slots_to_cuda(write_slots)
        self._episode_slots.setdefault(episode_id, set()).update(
            int(slot) for slot in write_slots
        )
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
        self, batch_size: int, *, sampling: str = "uniform_window"
    ) -> BCTensorBatch:
        bs = int(batch_size)
        if not self.can_sample(bs):
            raise RuntimeError(
                f"not enough BC replay windows: size={self.size} batch={bs}"
            )
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
        if self.storage == "cuda_mirror":
            idx_tensor = torch.as_tensor(idx, device=self.device, dtype=torch.long)
            actions = self._device_actions.index_select(0, idx_tensor)
            state = self._device_conditions.index_select(0, idx_tensor)
            episode_ids = self._device_episode_ids.index_select(0, idx_tensor)
        else:
            actions = torch.as_tensor(
                self.actions[idx], device=self.device, dtype=torch.float32
            )
            state = torch.as_tensor(
                self.conditions[idx], device=self.device, dtype=torch.float32
            )
            episode_ids = torch.as_tensor(
                self.episode_ids[idx], device=self.device, dtype=torch.long
            )
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
            "storage": str(self.storage),
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
        conditions = np.asarray(state.get("conditions", []), dtype=np.float32)
        actions = np.asarray(state.get("actions", []), dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.conditions.fill(0.0)
        self.actions.fill(0.0)
        self.episode_ids.fill(-1)
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
                episode_ids = np.asarray(stored_episode_ids, dtype=np.int64)
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
            self._sync_slots_to_cuda(np.arange(n, dtype=np.int64))
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
