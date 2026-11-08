from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

OBS_STATE = "observation.state"
OBS_ENV_STATE = "observation.environment_state"
ACTION = "action"


def _sorted_demo_keys(keys: list[str]) -> list[str]:
    def demo_index(name: str) -> int:
        if name.startswith("demo_"):
            try:
                return int(name[5:])
            except ValueError:
                return 0
        return 0

    return sorted(keys, key=demo_index)


@dataclass
class RobomimicHDF5Info:
    path: Path
    has_obs: bool
    demo_keys: list[str]
    total_frames: int
    action_dim: int
    obs_keys: list[str]
    env_args: dict | str | None


def inspect_robomimic_hdf5(hdf5_path: str | Path) -> RobomimicHDF5Info:
    path = Path(hdf5_path).expanduser().resolve()
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise ValueError(
                f"{path} is not a valid robomimic dataset (missing 'data' group)"
            )
        demo_keys = _sorted_demo_keys(list(f["data"].keys()))
        if not demo_keys:
            raise ValueError(f"No demonstrations found under {path}::data")
        first_demo = f[f"data/{demo_keys[0]}"]
        has_obs = "obs" in first_demo
        obs_keys = sorted(list(first_demo["obs"].keys())) if has_obs else []
        action_dim = int(first_demo["actions"].shape[-1])
        total_frames = 0
        for demo_key in demo_keys:
            total_frames += int(f[f"data/{demo_key}/actions"].shape[0])
        env_args = f["data"].attrs.get("env_args", None)
        if isinstance(env_args, bytes):
            env_args = env_args.decode("utf-8")
        if isinstance(env_args, str):
            try:
                env_args = json.loads(env_args)
            except json.JSONDecodeError:
                pass
        return RobomimicHDF5Info(
            path=path,
            has_obs=has_obs,
            demo_keys=demo_keys,
            total_frames=total_frames,
            action_dim=action_dim,
            obs_keys=obs_keys,
            env_args=env_args,
        )


def detect_image_obs_keys(hdf5_path: str | Path) -> list[str]:
    info = inspect_robomimic_hdf5(hdf5_path)
    if not info.has_obs:
        return []
    image_keys: list[str] = []
    with h5py.File(info.path, "r") as f:
        first_demo = f[f"data/{info.demo_keys[0]}"]
        for key in info.obs_keys:
            dset = first_demo["obs"][key]
            obs_shape = tuple(dset.shape[1:])
            looks_like_image = (
                len(obs_shape) == 3
                and obs_shape[-1] in (1, 3, 4)
                and np.issubdtype(dset.dtype, np.integer)
            )
            if looks_like_image or key.endswith("_image"):
                image_keys.append(key)
    return sorted(image_keys)


def infer_state_env_obs_keys(
    all_obs_keys: list[str],
    image_obs_keys: list[str],
) -> tuple[list[str], list[str]]:
    lowdim_keys = [k for k in all_obs_keys if k not in set(image_obs_keys)]
    robot_keys = [k for k in lowdim_keys if k.startswith("robot")]
    env_keys = [k for k in lowdim_keys if k not in set(robot_keys)]

    if not robot_keys and lowdim_keys:
        robot_keys = [lowdim_keys[0]]
        env_keys = [k for k in lowdim_keys if k not in set(robot_keys)]
    return sorted(robot_keys), sorted(env_keys)


def _robomimic_extract_script_path(project_root: Path) -> Path:
    return project_root / "robomimic" / "scripts" / "dataset_states_to_obs.py"


def ensure_dataset_has_obs(
    dataset_path: str | Path,
    auto_extract_obs: bool,
    extract_output_name: str | None,
    camera_names: list[str],
    camera_height: int,
    camera_width: int,
    done_mode: int = 2,
    copy_rewards: bool = False,
    copy_dones: bool = False,
) -> Path:
    info = inspect_robomimic_hdf5(dataset_path)
    if info.has_obs:
        return info.path
    if not auto_extract_obs:
        raise ValueError(
            "Input dataset has no 'obs' group (likely demo_v15.hdf5). "
            "Please set auto_extract_obs=True, or run robomimic dataset_states_to_obs.py first."
        )

    from exrl.paths import ROOT

    project_root = ROOT / "vendor" / "robomimic"
    script_path = _robomimic_extract_script_path(project_root)
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find extraction script: {script_path}")

    output_name = extract_output_name
    if output_name is None:
        stem = info.path.stem
        if camera_names:
            output_name = f"{stem}_obs_im{camera_width}.hdf5"
        else:
            output_name = f"{stem}_obs_lowdim.hdf5"
    output_path = info.path.parent / output_name
    if output_path.resolve() == info.path.resolve():
        raise ValueError(
            f"extract_output_name must be different from input file name. "
            f"Got output={output_path.name}, input={info.path.name}."
        )

    if output_path.exists():
        try:
            output_info = inspect_robomimic_hdf5(output_path)
            same_demo_count = len(output_info.demo_keys) == len(info.demo_keys)
            same_total_frames = output_info.total_frames == info.total_frames
            if output_info.has_obs and same_demo_count and same_total_frames:
                return output_path
            output_path.unlink(missing_ok=True)
        except Exception:
            # A previous failed extraction can leave a truncated / corrupted hdf5.
            # Remove it and regenerate to avoid "Unable to synchronously open file" errors.
            output_path.unlink(missing_ok=True)

    cmd = [
        sys.executable,
        str(script_path),
        "--dataset",
        str(info.path),
        "--output_name",
        output_name,
        "--done_mode",
        str(done_mode),
        "--exclude-next-obs",
        "--compress",
    ]
    if camera_names:
        cmd.extend(["--camera_names", *camera_names])
        cmd.extend(
            ["--camera_height", str(camera_height), "--camera_width", str(camera_width)]
        )
    if copy_rewards:
        cmd.append("--copy_rewards")
    if copy_dones:
        cmd.append("--copy_dones")

    env = os.environ.copy()
    extra_paths = [
        str(project_root),
    ]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        extra_paths + ([existing_pythonpath] if existing_pythonpath else [])
    )

    # Help robomimic/mujoco_py find MuJoCo runtime for dataset extraction from states.
    ld_library_paths = [
        p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p
    ]
    mujoco_candidates = [
        Path.home() / ".mujoco" / "mujoco210" / "bin",
        Path.home() / ".mujoco" / "mujoco200" / "bin",
        Path("/usr/lib/nvidia"),
    ]
    for candidate in mujoco_candidates:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in ld_library_paths:
            ld_library_paths.append(candidate_str)
    if ld_library_paths:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_library_paths)

    try:
        subprocess.run(cmd, check=True, cwd=str(project_root), env=env)
    except Exception:
        # Best-effort cleanup of partially written files from failed extraction.
        output_path.unlink(missing_ok=True)
        raise
    return output_path.resolve()


def _concat_lowdim_obs(
    obs_group: h5py.Group, frame_idx: int, obs_keys: list[str]
) -> np.ndarray:
    parts = [
        np.asarray(obs_group[k][frame_idx], dtype=np.float32).reshape(-1)
        for k in obs_keys
    ]
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _concat_lowdim_obs_all(obs_group: h5py.Group, obs_keys: list[str]) -> np.ndarray:
    parts = [
        np.asarray(obs_group[k][:], dtype=np.float32).reshape(obs_group[k].shape[0], -1)
        for k in obs_keys
    ]
    if not parts:
        num_steps = next(iter(obs_group.values())).shape[0]
        return np.zeros((num_steps, 0), dtype=np.float32)
    return np.concatenate(parts, axis=-1)


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
        if self.count == 0:
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


def infer_feature_shapes(
    hdf5_path: str | Path,
    state_obs_keys: list[str],
    env_obs_keys: list[str],
    image_obs_keys: list[str],
) -> dict[str, tuple[int, ...]]:
    info = inspect_robomimic_hdf5(hdf5_path)
    if not info.has_obs:
        raise ValueError("Cannot infer feature shapes because dataset has no obs group")
    with h5py.File(info.path, "r") as f:
        obs_group = f[f"data/{info.demo_keys[0]}/obs"]
        shapes: dict[str, tuple[int, ...]] = {}

        state_dim = int(sum(np.prod(obs_group[k].shape[1:]) for k in state_obs_keys))
        shapes[OBS_STATE] = (state_dim,)

        env_dim = int(sum(np.prod(obs_group[k].shape[1:]) for k in env_obs_keys))
        if env_dim > 0:
            shapes[OBS_ENV_STATE] = (env_dim,)

        for key in image_obs_keys:
            raw_shape = tuple(obs_group[key].shape[1:])  # H, W, C
            if len(raw_shape) != 3:
                raise ValueError(
                    f"Image key {key} has invalid shape {raw_shape} (expected HWC)"
                )
            h, w, c = raw_shape
            shapes[f"observation.images.{key}"] = (c, h, w)

        shapes[ACTION] = (info.action_dim,)
        return shapes


def compute_dataset_stats(
    hdf5_path: str | Path,
    state_obs_keys: list[str],
    env_obs_keys: list[str],
    image_obs_keys: list[str],
    action_key: str = "actions",
    include_image_stats: bool = False,
    use_state_as_env_when_missing: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    info = inspect_robomimic_hdf5(hdf5_path)
    if not info.has_obs:
        raise ValueError(
            "Cannot compute observation stats from dataset without obs group"
        )

    feature_shapes = infer_feature_shapes(
        hdf5_path=info.path,
        state_obs_keys=state_obs_keys,
        env_obs_keys=env_obs_keys,
        image_obs_keys=image_obs_keys,
    )

    accumulators: dict[str, _StatsAccumulator] = {
        OBS_STATE: _StatsAccumulator(feature_shapes[OBS_STATE]),
        ACTION: _StatsAccumulator(feature_shapes[ACTION]),
    }
    if OBS_ENV_STATE in feature_shapes:
        accumulators[OBS_ENV_STATE] = _StatsAccumulator(feature_shapes[OBS_ENV_STATE])
    elif use_state_as_env_when_missing:
        accumulators[OBS_ENV_STATE] = _StatsAccumulator(feature_shapes[OBS_STATE])

    if include_image_stats:
        for raw_key in image_obs_keys:
            feature_key = f"observation.images.{raw_key}"
            accumulators[feature_key] = _StatsAccumulator(feature_shapes[feature_key])

    with h5py.File(info.path, "r") as f:
        for demo_key in info.demo_keys:
            demo = f[f"data/{demo_key}"]
            obs_group = demo["obs"]

            state_mat = _concat_lowdim_obs_all(obs_group, state_obs_keys)
            accumulators[OBS_STATE].update(state_mat)

            if OBS_ENV_STATE in accumulators:
                if env_obs_keys:
                    env_mat = _concat_lowdim_obs_all(obs_group, env_obs_keys)
                else:
                    env_mat = state_mat
                accumulators[OBS_ENV_STATE].update(env_mat)

            action_arr = np.asarray(demo[action_key][:], dtype=np.float32)
            accumulators[ACTION].update(action_arr)

            if include_image_stats:
                for raw_key in image_obs_keys:
                    imgs = np.asarray(obs_group[raw_key][:], dtype=np.float32) / 255.0
                    imgs = np.transpose(imgs, (0, 3, 1, 2))  # THWC -> TCHW
                    accumulators[f"observation.images.{raw_key}"].update(imgs)

    return {name: acc.to_torch_stats() for name, acc in accumulators.items()}


class RobomimicACTDataset(Dataset):
    """Robomimic HDF5 dataset adapter that emits lerobot ACT-compatible batches."""

    def __init__(
        self,
        hdf5_path: str | Path,
        state_obs_keys: list[str],
        env_obs_keys: list[str],
        image_obs_keys: list[str],
        action_delta_indices: list[int],
        action_key: str = "actions",
        use_state_as_env_when_missing: bool = False,
    ):
        self.hdf5_path = str(Path(hdf5_path).expanduser().resolve())
        self.state_obs_keys = list(state_obs_keys)
        self.env_obs_keys = list(env_obs_keys)
        self.image_obs_keys = list(image_obs_keys)
        self.action_delta_indices = np.asarray(action_delta_indices, dtype=np.int64)
        self.action_key = action_key
        self.use_state_as_env_when_missing = use_state_as_env_when_missing

        info = inspect_robomimic_hdf5(self.hdf5_path)
        if not info.has_obs:
            raise ValueError(
                f"{self.hdf5_path} has no obs group. "
                "Please extract observations first or use ensure_dataset_has_obs()."
            )
        self.demo_keys = info.demo_keys
        self.index_map: list[tuple[str, int]] = []
        self.demo_num_samples: dict[str, int] = {}

        with h5py.File(self.hdf5_path, "r") as f:
            for demo_key in self.demo_keys:
                num_samples = int(f[f"data/{demo_key}/{self.action_key}"].shape[0])
                self.demo_num_samples[demo_key] = num_samples
                for frame_idx in range(num_samples):
                    self.index_map.append((demo_key, frame_idx))

        self._h5_file: h5py.File | None = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def _get_h5_file(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.hdf5_path, "r")
        return self._h5_file

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        h5_file = self._get_h5_file()
        demo_key, frame_idx = self.index_map[idx]
        demo = h5_file[f"data/{demo_key}"]
        obs_group = demo["obs"]
        num_samples = self.demo_num_samples[demo_key]

        state_vec = _concat_lowdim_obs(obs_group, frame_idx, self.state_obs_keys)
        batch: dict[str, torch.Tensor] = {
            OBS_STATE: torch.from_numpy(state_vec).float(),
        }

        if self.env_obs_keys:
            env_vec = _concat_lowdim_obs(obs_group, frame_idx, self.env_obs_keys)
            batch[OBS_ENV_STATE] = torch.from_numpy(env_vec).float()
        elif self.use_state_as_env_when_missing:
            batch[OBS_ENV_STATE] = batch[OBS_STATE].clone()

        for raw_key in self.image_obs_keys:
            img = np.asarray(obs_group[raw_key][frame_idx], dtype=np.float32)
            img = np.transpose(img, (2, 0, 1)) / 255.0
            batch[f"observation.images.{raw_key}"] = torch.from_numpy(img).float()

        target_indices = frame_idx + self.action_delta_indices
        clipped_indices = np.clip(target_indices, 0, num_samples - 1)
        max_idx = int(clipped_indices.max())
        action_slice = np.asarray(
            demo[self.action_key][frame_idx : max_idx + 1], dtype=np.float32
        )
        rel_indices = clipped_indices - frame_idx
        action_chunk = action_slice[rel_indices]
        if action_chunk.ndim == 1:
            action_chunk = action_chunk.reshape(1, -1)
        is_pad = target_indices >= num_samples

        batch[ACTION] = torch.from_numpy(action_chunk).float()
        batch["action_is_pad"] = torch.from_numpy(is_pad.astype(np.bool_))
        return batch

    def __del__(self):
        if getattr(self, "_h5_file", None) is not None:
            self._h5_file.close()
            self._h5_file = None
