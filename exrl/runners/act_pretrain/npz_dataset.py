from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

OBS_STATE = "observation.state"
OBS_ENV_STATE = "observation.environment_state"
ACTION = "action"


@dataclass
class DSRLNPZInfo:
    path: Path
    total_frames: int
    state_dim: int
    action_dim: int
    action_chunk_len: int
    direct_chunk_labels: bool
    traj_lengths: np.ndarray


def inspect_dsrl_npz(npz_path: str | Path) -> DSRLNPZInfo:
    path = Path(npz_path).expanduser().resolve()
    data = np.load(path, allow_pickle=False)
    required = ("states", "actions", "traj_lengths")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")

    states = np.asarray(data["states"])
    actions = np.asarray(data["actions"])
    traj_lengths = np.asarray(data["traj_lengths"], dtype=np.int64).reshape(-1)
    if states.ndim != 2:
        raise ValueError(f"{path}::states must be 2D [T, D], got {states.shape}")
    if actions.ndim not in (2, 3):
        raise ValueError(
            f"{path}::actions must be 2D [T, A] or 3D [T, H, A], got {actions.shape}"
        )
    total_frames = int(traj_lengths.sum())
    if total_frames != int(states.shape[0]) or total_frames != int(actions.shape[0]):
        raise ValueError(
            f"{path} traj_lengths sum mismatch: states={states.shape[0]} actions={actions.shape[0]} traj_sum={total_frames}"
        )
    if actions.ndim == 2:
        action_dim = int(actions.shape[1])
        action_chunk_len = 0
        direct_chunk_labels = False
    else:
        action_chunk_len = int(actions.shape[1])
        action_dim = int(actions.shape[2])
        direct_chunk_labels = True
    return DSRLNPZInfo(
        path=path,
        total_frames=total_frames,
        state_dim=int(states.shape[1]),
        action_dim=action_dim,
        action_chunk_len=action_chunk_len,
        direct_chunk_labels=direct_chunk_labels,
        traj_lengths=traj_lengths,
    )


def build_flat_obs_keys(
    state_obs_keys: Sequence[str], env_obs_keys: Sequence[str]
) -> List[str]:
    out: List[str] = []
    for key in list(state_obs_keys) + list(env_obs_keys):
        if key not in out:
            out.append(str(key))
    if not out:
        raise ValueError("at least one low-dim obs key is required")
    return out


def split_dsrl_state_batch(
    state_batch: np.ndarray,
    *,
    flat_obs_keys: Sequence[str],
    lowdim_dims: Dict[str, int],
) -> Dict[str, np.ndarray]:
    state_batch = np.asarray(state_batch, dtype=np.float32)
    if state_batch.ndim != 2:
        raise ValueError(f"state_batch must be 2D [B, D], got {state_batch.shape}")
    expected_dim = int(sum(int(lowdim_dims[key]) for key in flat_obs_keys))
    if int(state_batch.shape[1]) != expected_dim:
        raise ValueError(
            f"state dim mismatch: got {state_batch.shape[1]}, expected {expected_dim}"
        )
    out: Dict[str, np.ndarray] = {}
    start = 0
    for key in flat_obs_keys:
        dim = int(lowdim_dims[key])
        out[key] = state_batch[:, start : start + dim]
        start += dim
    return out


def _concat_parts(parts: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(part, dtype=np.float32) for part in parts]
    if not arrays:
        if len(parts) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        raise ValueError("cannot infer concat shape from empty parts")
    return np.concatenate(arrays, axis=-1).astype(np.float32)


class _StatsAccumulator:
    def __init__(self, feature_shape: tuple[int, ...]):
        self.feature_shape = tuple(feature_shape)
        self.feature_size = int(np.prod(self.feature_shape))
        self.sum = np.zeros((self.feature_size,), dtype=np.float64)
        self.sq_sum = np.zeros((self.feature_size,), dtype=np.float64)
        self.min = np.full((self.feature_size,), np.inf, dtype=np.float64)
        self.max = np.full((self.feature_size,), -np.inf, dtype=np.float64)
        self.count = 0

    def update(self, values: np.ndarray) -> None:
        if values.ndim == len(self.feature_shape):
            values = values.reshape(1, *self.feature_shape)
        flat = values.reshape(-1, self.feature_size).astype(np.float64)
        if flat.size == 0:
            return
        self.sum += flat.sum(axis=0)
        self.sq_sum += np.square(flat).sum(axis=0)
        self.min = np.minimum(self.min, flat.min(axis=0))
        self.max = np.maximum(self.max, flat.max(axis=0))
        self.count += int(flat.shape[0])

    def to_torch_stats(self) -> dict[str, torch.Tensor]:
        if self.count <= 0:
            raise ValueError("Cannot finalize stats with 0 samples")
        mean = self.sum / self.count
        var = np.maximum(self.sq_sum / self.count - np.square(mean), 1e-12)
        std = np.sqrt(var)
        return {
            "mean": torch.from_numpy(mean.reshape(self.feature_shape)).float(),
            "std": torch.from_numpy(std.reshape(self.feature_shape)).float(),
            "min": torch.from_numpy(self.min.reshape(self.feature_shape)).float(),
            "max": torch.from_numpy(self.max.reshape(self.feature_shape)).float(),
        }


def infer_dsrl_feature_shapes(
    npz_path: str | Path,
    *,
    state_obs_keys: Sequence[str],
    env_obs_keys: Sequence[str],
    lowdim_dims: Dict[str, int],
    use_state_as_env_when_missing: bool = False,
) -> dict[str, tuple[int, ...]]:
    info = inspect_dsrl_npz(npz_path)
    flat_obs_keys = build_flat_obs_keys(state_obs_keys, env_obs_keys)
    expected_dim = int(sum(int(lowdim_dims[key]) for key in flat_obs_keys))
    if info.state_dim != expected_dim:
        raise ValueError(
            f"{info.path} state_dim={info.state_dim} does not match split dims {expected_dim}"
        )

    state_dim = int(sum(int(lowdim_dims[key]) for key in state_obs_keys))
    env_dim = int(sum(int(lowdim_dims[key]) for key in env_obs_keys))
    shapes = {
        OBS_STATE: (state_dim,),
        ACTION: (info.action_dim,),
    }
    if env_dim > 0:
        shapes[OBS_ENV_STATE] = (env_dim,)
    elif use_state_as_env_when_missing:
        shapes[OBS_ENV_STATE] = (state_dim,)
    return shapes


def compute_dsrl_dataset_stats(
    npz_path: str | Path,
    *,
    state_obs_keys: Sequence[str],
    env_obs_keys: Sequence[str],
    lowdim_dims: Dict[str, int],
    use_state_as_env_when_missing: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    info = inspect_dsrl_npz(npz_path)
    data = np.load(info.path, allow_pickle=False)
    states = np.asarray(data["states"][: info.total_frames], dtype=np.float32)
    actions = np.asarray(data["actions"][: info.total_frames], dtype=np.float32)
    flat_obs_keys = build_flat_obs_keys(state_obs_keys, env_obs_keys)
    split = split_dsrl_state_batch(
        states, flat_obs_keys=flat_obs_keys, lowdim_dims=lowdim_dims
    )

    state_mat = _concat_parts([split[key] for key in state_obs_keys])
    if env_obs_keys:
        env_mat = _concat_parts([split[key] for key in env_obs_keys])
    else:
        env_mat = state_mat

    feature_shapes = infer_dsrl_feature_shapes(
        info.path,
        state_obs_keys=state_obs_keys,
        env_obs_keys=env_obs_keys,
        lowdim_dims=lowdim_dims,
        use_state_as_env_when_missing=use_state_as_env_when_missing,
    )
    accumulators: dict[str, _StatsAccumulator] = {
        OBS_STATE: _StatsAccumulator(feature_shapes[OBS_STATE]),
        ACTION: _StatsAccumulator(feature_shapes[ACTION]),
    }
    if OBS_ENV_STATE in feature_shapes:
        accumulators[OBS_ENV_STATE] = _StatsAccumulator(feature_shapes[OBS_ENV_STATE])

    accumulators[OBS_STATE].update(state_mat)
    accumulators[ACTION].update(actions)
    if OBS_ENV_STATE in accumulators:
        accumulators[OBS_ENV_STATE].update(env_mat)
    return {name: acc.to_torch_stats() for name, acc in accumulators.items()}


class DSRLNPZACTDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        *,
        state_obs_keys: Sequence[str],
        env_obs_keys: Sequence[str],
        lowdim_dims: Dict[str, int],
        action_delta_indices: Sequence[int],
        use_state_as_env_when_missing: bool = False,
    ):
        info = inspect_dsrl_npz(npz_path)
        data = np.load(info.path, allow_pickle=False)
        self.path = str(info.path)
        self.state_obs_keys = list(state_obs_keys)
        self.env_obs_keys = list(env_obs_keys)
        self.flat_obs_keys = build_flat_obs_keys(state_obs_keys, env_obs_keys)
        self.lowdim_dims = {key: int(lowdim_dims[key]) for key in self.flat_obs_keys}
        self.action_delta_indices = np.asarray(action_delta_indices, dtype=np.int64)
        self.use_state_as_env_when_missing = bool(use_state_as_env_when_missing)
        self.direct_chunk_labels = bool(info.direct_chunk_labels)
        self.action_chunk_len = int(info.action_chunk_len)

        self.states = np.asarray(data["states"][: info.total_frames], dtype=np.float32)
        self.actions = np.asarray(
            data["actions"][: info.total_frames], dtype=np.float32
        )
        if self.direct_chunk_labels:
            if self.action_chunk_len != int(len(self.action_delta_indices)):
                raise ValueError(
                    f"{info.path} direct chunk labels have H={self.action_chunk_len}, "
                    f"but requested chunk size is {len(self.action_delta_indices)}"
                )
            if "action_is_pad" in data.files:
                self.action_is_pad = np.asarray(
                    data["action_is_pad"][: info.total_frames], dtype=np.bool_
                )
            else:
                self.action_is_pad = np.zeros(
                    (info.total_frames, self.action_chunk_len), dtype=np.bool_
                )
        else:
            self.action_is_pad = None
        self.traj_lengths = np.asarray(data["traj_lengths"], dtype=np.int64).reshape(-1)
        self.traj_starts = np.concatenate(
            ([0], np.cumsum(self.traj_lengths[:-1], dtype=np.int64))
        )
        self.index_map: list[tuple[int, int, int]] = []
        for traj_start, traj_len in zip(
            self.traj_starts.tolist(), self.traj_lengths.tolist()
        ):
            for frame_idx in range(int(traj_len)):
                self.index_map.append((int(traj_start), int(traj_len), int(frame_idx)))

        expected_dim = int(
            sum(int(self.lowdim_dims[key]) for key in self.flat_obs_keys)
        )
        if self.states.shape[1] != expected_dim:
            raise ValueError(
                f"{info.path} state_dim={self.states.shape[1]} does not match split dims {expected_dim}"
            )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj_start, traj_len, frame_idx = self.index_map[idx]
        global_idx = traj_start + frame_idx
        split = split_dsrl_state_batch(
            self.states[global_idx : global_idx + 1],
            flat_obs_keys=self.flat_obs_keys,
            lowdim_dims=self.lowdim_dims,
        )

        state_vec = np.concatenate(
            [split[key][0] for key in self.state_obs_keys], axis=0
        ).astype(np.float32)
        batch: dict[str, torch.Tensor] = {
            OBS_STATE: torch.from_numpy(state_vec).float(),
        }
        if self.env_obs_keys:
            env_vec = np.concatenate(
                [split[key][0] for key in self.env_obs_keys], axis=0
            ).astype(np.float32)
            batch[OBS_ENV_STATE] = torch.from_numpy(env_vec).float()
        elif self.use_state_as_env_when_missing:
            batch[OBS_ENV_STATE] = batch[OBS_STATE].clone()

        if self.direct_chunk_labels:
            action_chunk = self.actions[global_idx]
            if action_chunk.ndim == 1:
                action_chunk = action_chunk.reshape(1, -1)
            is_pad = self.action_is_pad[global_idx]
        else:
            target_indices = frame_idx + self.action_delta_indices
            clipped = np.clip(target_indices, 0, traj_len - 1)
            global_targets = traj_start + clipped
            action_chunk = self.actions[global_targets]
            if action_chunk.ndim == 1:
                action_chunk = action_chunk.reshape(1, -1)
            is_pad = target_indices >= traj_len

        batch[ACTION] = torch.from_numpy(action_chunk.astype(np.float32)).float()
        batch["action_is_pad"] = torch.from_numpy(is_pad.astype(np.bool_))
        return batch
