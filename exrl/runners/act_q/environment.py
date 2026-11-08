import os

import numpy as np

from exrl.runners.act_q.execution import (
    BatchedAdaptiveWrapper as ACTCompatBatchedAdaptiveWrapper,
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
        os.environ["OPT_MIN_CHUNK_LEN"] = str(int(kwargs.get("min_chunk_len", 1)))
        self._stats_path_override = (
            os.environ.get("OPT_TASK_STATS_NPZ", "").strip()
            or os.environ.get("OPT_CAN_STATS_NPZ", "").strip()
        )
        super().__init__(*args, **kwargs)
        self._reload_stats_override()
        self._refresh_obs_dims_from_stats()

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
        return self._build_obs_matrix(batch)[0]

    def _build_obs_matrix(self, obs_batch):
        n = self.num_envs
        for key in self._obs_keys:
            if key in obs_batch:
                n = int(np.asarray(obs_batch[key]).shape[0])
                break

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
        use_stats = self.critic_obs_state_mode in {"stats_clip", "stats_no_clip"}
        clip_stats = self.critic_obs_state_mode == "stats_clip"
        parts = []
        for key in self._obs_keys:
            dim = int(dims[key])
            if dim <= 0:
                continue
            if key in obs_batch:
                values = np.asarray(obs_batch[key], dtype=np.float32).reshape(n, dim)
                if use_stats:
                    mean, std = self._safe_norm_stats(key, dim)
                    values = (values - mean) / std
                parts.append(values)
            else:
                parts.append(np.zeros((n, dim), dtype=np.float32))

        if not parts:
            return np.zeros((n, 0), dtype=np.float32)

        obs = np.concatenate(parts, axis=1).astype(np.float32)
        if clip_stats:
            np.clip(obs, -10.0, 10.0, out=obs)
        return obs
