import os

import numpy as np

try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces

from exrl.runners.act_sac.execution import (
    BatchedAdaptiveWrapper as ACTCompatBatchedAdaptiveWrapper,
    DSRL_NORMALIZED_STATE_KEY,
)


class ExecutionLengthEnv(ACTCompatBatchedAdaptiveWrapper):
    """
    Task-generic wrapper on top of the ACT-compat synchronous BatchedAdaptiveWrapper.

    Purpose:
    - Reuse existing chunk execution + ACT chunk inference behavior.
    - Remove tool_hang-specific low-dim shape assumptions.
    - Allow task-specific stats npz override via a neutral env var.
    """

    def __init__(self, *args, **kwargs):
        self._task_lowdim_keys = [str(key) for key in kwargs.get("lowdim_keys", ())]
        requested_dims = dict(kwargs.get("lowdim_dims", {}))
        self._task_lowdim_dims = {
            key: int(requested_dims[key]) for key in self._task_lowdim_keys
        }
        self._physical_obs_normalization = (
            str(kwargs.pop("physical_obs_normalization", "stats")).strip().lower()
        )
        self._action_obs_normalization = (
            str(kwargs.pop("action_obs_normalization", "stats")).strip().lower()
        )
        if self._physical_obs_normalization not in {"stats", "none"}:
            raise ValueError("physical_obs_normalization must be one of: stats, none")
        if self._action_obs_normalization not in {"stats", "none"}:
            raise ValueError("action_obs_normalization must be one of: stats, none")
        physical_obs_clip = float(kwargs.pop("physical_obs_clip", 10.0))
        self._physical_obs_clip = physical_obs_clip if physical_obs_clip > 0.0 else None
        os.environ["OPT_MIN_CHUNK_LEN"] = str(int(kwargs.get("min_chunk_len", 1)))
        self._stats_path_override = (
            os.environ.get("OPT_TASK_STATS_NPZ", "").strip()
            or os.environ.get("OPT_CAN_STATS_NPZ", "").strip()
        )
        super().__init__(*args, **kwargs)
        self._set_task_obs_layout()
        self._reload_stats_override()
        self._refresh_obs_dims_from_stats()
        self._rebuild_task_observation_space()
        clip_desc = (
            "disabled"
            if self._physical_obs_clip is None
            else str(self._physical_obs_clip)
        )
        print(f"[TASK] physical observation clip: {clip_desc}")
        print(
            "[TASK] DP-aligned RL observation normalization: "
            f"physical={self._physical_obs_normalization} "
            f"action={self._action_obs_normalization}"
        )

    def _set_task_obs_layout(self):
        if not self._task_lowdim_keys:
            raise ValueError("[ACT-COMPAT] lowdim_keys cannot be empty")
        if len(set(self._task_lowdim_keys)) != len(self._task_lowdim_keys):
            raise ValueError(
                f"[ACT-COMPAT] duplicate lowdim keys: {self._task_lowdim_keys}"
            )
        missing = [
            key for key in self._task_lowdim_keys if key not in self._task_lowdim_dims
        ]
        if missing:
            raise KeyError(f"[ACT-COMPAT] lowdim_dims missing keys: {missing}")
        invalid = {
            key: self._task_lowdim_dims[key]
            for key in self._task_lowdim_keys
            if self._task_lowdim_dims[key] <= 0
        }
        if invalid:
            raise ValueError(f"[ACT-COMPAT] invalid lowdim dimensions: {invalid}")
        self._obs_keys = list(self._task_lowdim_keys)
        self._obs_dims = dict(self._task_lowdim_dims)

    def _rebuild_task_observation_space(self):
        state_dim = int(sum(self._obs_dims[key] for key in self._obs_keys))
        history_dim = int(self.action_history_len) * int(self.action_dim)
        chunk_dim = int(self.max_chunk_len) * int(self.action_dim)
        self.total_obs_dim = state_dim + history_dim + chunk_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.total_obs_dim,),
            dtype=np.float32,
        )
        print(
            "[ACT-COMPAT] critic observation layout "
            f"keys={self._obs_keys} state_dim={state_dim} history_dim={history_dim} "
            f"chunk_dim={chunk_dim} total_obs_dim={self.total_obs_dim}"
        )

    def _reload_stats_override(self):
        if not self._stats_path_override:
            return
        if not os.path.exists(self._stats_path_override):
            print(f"[TASK][WARN] stats override not found: {self._stats_path_override}")
            return
        stats = np.load(self._stats_path_override)
        self.stats = stats
        self._action_mean = np.asarray(self.stats["action_mean"], dtype=np.float32)
        self._action_std = np.asarray(self.stats["action_std"], dtype=np.float32)
        print(f"[TASK] loaded stats from {self._stats_path_override}")

    def _refresh_obs_dims_from_stats(self):
        dims = {}
        for key in self._obs_keys:
            mean_key = f"{key}_mean"
            if mean_key in self.stats:
                dims[key] = int(np.asarray(self.stats[mean_key]).reshape(-1).shape[0])
        if dims:
            self._obs_dims = dims

    def _safe_norm_stats(self, key, dim):
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        if mean_key in self.stats and std_key in self.stats:
            mean = np.asarray(self.stats[mean_key], dtype=np.float32).reshape(-1)
            std = np.asarray(self.stats[std_key], dtype=np.float32).reshape(-1)
            if mean.shape[0] == dim and std.shape[0] == dim:
                std = np.where(std < 1e-8, 1.0, std)
                return mean.reshape(1, dim), std.reshape(1, dim)
        return np.zeros((1, dim), dtype=np.float32), np.ones((1, dim), dtype=np.float32)

    def _get_obs_vec(self, obs_dict):
        batch = {}
        for key in self._obs_keys:
            if key in obs_dict:
                batch[key] = np.asarray(obs_dict[key], dtype=np.float32).reshape(1, -1)
        if DSRL_NORMALIZED_STATE_KEY in obs_dict:
            batch[DSRL_NORMALIZED_STATE_KEY] = np.asarray(
                obs_dict[DSRL_NORMALIZED_STATE_KEY],
                dtype=np.float32,
            ).reshape(1, -1)
        return self._build_obs_matrix(batch)[0]

    def _build_obs_matrix(self, obs_batch):
        n = self.num_envs
        for key in self._obs_keys:
            if key in obs_batch:
                n = int(np.asarray(obs_batch[key]).shape[0])
                break
        if DSRL_NORMALIZED_STATE_KEY in obs_batch:
            n = int(np.asarray(obs_batch[DSRL_NORMALIZED_STATE_KEY]).shape[0])

        dims = {}
        for key in self._obs_keys:
            if key in obs_batch:
                dims[key] = int(
                    np.asarray(obs_batch[key], dtype=np.float32).reshape(n, -1).shape[1]
                )
            elif f"{key}_mean" in self.stats:
                dims[key] = int(
                    np.asarray(self.stats[f"{key}_mean"]).reshape(-1).shape[0]
                )
            else:
                dims[key] = int(self._obs_dims.get(key, 0))

        self._obs_dims = dims
        task_keys = getattr(self, "_task_lowdim_keys", self._obs_keys)
        task_dims = getattr(self, "_task_lowdim_dims", dims)
        expected_dim = int(sum(max(int(task_dims[key]), 0) for key in task_keys))
        if self._physical_obs_normalization == "none":
            if DSRL_NORMALIZED_STATE_KEY in obs_batch:
                obs = np.asarray(
                    obs_batch[DSRL_NORMALIZED_STATE_KEY],
                    dtype=np.float32,
                ).reshape(n, -1)
            else:
                missing = [
                    key
                    for key in self._obs_keys
                    if int(dims[key]) > 0 and key not in obs_batch
                ]
                if missing:
                    raise RuntimeError(
                        "DP-aligned physical observations require a complete state; "
                        f"missing keys: {missing}"
                    )
                raw_obs = np.concatenate(
                    [
                        np.asarray(obs_batch[key], dtype=np.float32).reshape(
                            n, int(dims[key])
                        )
                        for key in self._obs_keys
                        if int(dims[key]) > 0
                    ],
                    axis=1,
                )
                obs = self.venv.policy_state_to_dsrl(raw_obs)
            if int(obs.shape[1]) != expected_dim:
                raise RuntimeError(
                    "DP-aligned physical observation dim mismatch: "
                    f"got {obs.shape[1]}, expected {expected_dim}"
                )
            obs = np.asarray(obs, dtype=np.float32)
            if self._physical_obs_clip is not None:
                np.clip(obs, -self._physical_obs_clip, self._physical_obs_clip, out=obs)
            return obs

        parts = []
        for key in self._obs_keys:
            dim = int(dims[key])
            if dim <= 0:
                continue
            if key in obs_batch:
                values = np.asarray(obs_batch[key], dtype=np.float32).reshape(n, dim)
                mean, std = self._safe_norm_stats(key, dim)
                parts.append((values - mean) / std)
            else:
                parts.append(np.zeros((n, dim), dtype=np.float32))

        if not parts:
            return np.zeros((n, 0), dtype=np.float32)

        obs = np.concatenate(parts, axis=1).astype(np.float32)
        if self._physical_obs_clip is not None:
            np.clip(obs, -self._physical_obs_clip, self._physical_obs_clip, out=obs)
        return obs
