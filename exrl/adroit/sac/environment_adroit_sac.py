"""
Adroit ExRL-SAC AAC wrapper.

This file intentionally does not modify the existing robomimic / DSRL AAC
wrappers. It reuses the proven synchronous BatchedAdaptiveWrapper machinery for
DP inference + RL-observation construction, but replaces the env backend and the
success / reward semantics for Adroit Gym tasks.

The base DP emits normalized action chunks. This wrapper denormalizes those
chunks before physical Adroit execution, and normalizes raw chunks back to the
same DP/stat space when constructing critic observations.
"""

from __future__ import annotations

import copy
import os
import random
from functools import partial
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - remote env may still use gym
    import gym
    from gym import spaces

from stable_baselines3.common.vec_env import SubprocVecEnv

from exrl.adroit.sac.chunk_env import (
    BatchedAdaptiveWrapper as LegacyBatchedAdaptiveWrapper,
)
from exrl.adroit.sac.environment_asyncqueue import BatchedAdaptiveWrapperAsyncQueue


def _as_flat_bool_success(raw: Any) -> bool:
    if raw is None:
        return False
    if isinstance(raw, dict):
        if "task" in raw:
            return bool(raw.get("task", False))
        if "success" in raw:
            return bool(raw.get("success", False))
        if "is_success" in raw:
            return _as_flat_bool_success(raw.get("is_success"))
        return any(_as_flat_bool_success(v) for v in raw.values())
    if isinstance(raw, (list, tuple, np.ndarray)):
        arr = list(raw)
        if not arr:
            return False
        return _as_flat_bool_success(arr[-1])
    return bool(raw)


def extract_adroit_success(env: Any, info: Optional[Dict] = None) -> bool:
    """
    Success for Adroit must never be inferred from dense reward.

    Accepted sources, in priority order:
    - robomimic EnvGym / wrappers: env.is_success()
    - raw gym env: _is_success() / _check_success()
    - info['is_success'] / info['success']
    """
    info = info or {}

    candidates = []
    cur = env
    for _ in range(5):
        if cur is None:
            break
        candidates.append(cur)
        cur = getattr(cur, "env", None)

    for obj in candidates:
        is_success = getattr(obj, "is_success", None)
        if callable(is_success):
            try:
                if _as_flat_bool_success(is_success()):
                    return True
            except Exception:
                pass
        raw = getattr(obj, "unwrapped", None)
        for name in ("_is_success", "_check_success"):
            fn = getattr(raw, name, None) if raw is not None else None
            if callable(fn):
                try:
                    if _as_flat_bool_success(fn()):
                        return True
                except Exception:
                    pass

    if "is_success" in info and _as_flat_bool_success(info.get("is_success")):
        return True
    if "success" in info and _as_flat_bool_success(info.get("success")):
        return True
    return False


class RobomimicToGymWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(env.action_dimension),),
            dtype=np.float32,
        )
        obs = self.env.reset()
        self.observation_space = spaces.Dict(
            {
                k: spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=np.asarray(v).shape,
                    dtype=np.asarray(v).dtype,
                )
                for k, v in obs.items()
            }
        )

    def reset(self, seed=None, options=None):
        _ = seed, options
        return self.env.reset(), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        info = dict(info) if isinstance(info, dict) else {}
        # EnvGym stores Gymnasium terminated/truncated in info. Preserve the
        # distinction so horizon timeout bootstrapping remains correct.
        terminated = bool(info.get("terminated", False)) or (
            bool(done) and not bool(info.get("truncated", False))
        )
        truncated = bool(info.get("truncated", False))
        return obs, float(reward), terminated, truncated, info

    def render(self, mode="rgb_array", height=512, width=512):
        return self.env.render(mode=mode, height=height, width=width)


class InternalAdroitVariableChunkWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        *,
        max_steps: int = 200,
        gamma: float = 0.999,
        resample_penalty: float = -5.0,
        sparse_step_reward: float = -1.0,
        sparse_success_reward: float = 0.0,
        terminate_on_success: bool = True,
        timeout_on_horizon: bool = True,
        zero_action_fail_streak: int = 0,
        zero_action_fail_reward: float = -1000.0,
        env_id: int = 0,
    ):
        super().__init__(env)
        self.max_steps = int(max_steps)
        self.gamma = float(gamma)
        self.resample_penalty = float(resample_penalty)
        self.sparse_step_reward = float(sparse_step_reward)
        self.sparse_success_reward = float(sparse_success_reward)
        self.terminate_on_success = bool(terminate_on_success)
        self.timeout_on_horizon = bool(timeout_on_horizon)
        self.zero_action_fail_streak = int(zero_action_fail_streak)
        self.zero_action_fail_reward = float(zero_action_fail_reward)
        self.env_id = int(env_id)

        self.current_step = 0
        self.episode_count = 0
        self.resample_streak = 0
        self.current_obs = None
        self.success_flag = False
        self.episode_dense_return = 0.0
        self.episode_sparse_return = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_step = 0
        self.resample_streak = 0
        self.success_flag = False
        self.episode_dense_return = 0.0
        self.episode_sparse_return = 0.0
        self.current_obs = obs
        return obs, info

    def step(self, action_data):
        chunk = np.asarray(action_data.get("chunk", []), dtype=np.float32)
        k = int(action_data.get("k", 0))

        if k <= 0:
            self.resample_streak += 1
            info = {
                "exec_len": 0,
                "resampled": True,
                "is_success": bool(self.success_flag),
                "success": bool(self.success_flag),
                "dense_reward": 0.0,
                "dense_return": float(self.episode_dense_return),
                "sparse_return": float(self.episode_sparse_return),
                "resample_streak": int(self.resample_streak),
            }
            if (
                self.zero_action_fail_streak > 0
                and self.resample_streak >= self.zero_action_fail_streak
            ):
                info["forced_zero_action_failure"] = True
                info["forced_failure_reward"] = float(self.zero_action_fail_reward)
                info["terminal_observation"] = self.current_obs
                self.episode_sparse_return += float(self.zero_action_fail_reward)
                info["episode_dense_return"] = float(self.episode_dense_return)
                info["episode_sparse_return"] = float(self.episode_sparse_return)
                self.episode_count += 1
                return self.current_obs, float(self.zero_action_fail_reward), True, False, info
            self.episode_sparse_return += float(self.resample_penalty)
            return self.current_obs, float(self.resample_penalty), False, False, info

        self.resample_streak = 0
        total_sparse_reward = 0.0
        total_dense_reward = 0.0
        discount = 1.0
        actual_steps = 0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        for i in range(min(k, int(chunk.shape[0]))):
            obs, dense_reward, term, trunc, step_info = self.env.step(chunk[i])
            info = dict(step_info) if isinstance(step_info, dict) else {}
            dense_reward = float(dense_reward)
            self.current_step += 1
            actual_steps += 1
            total_dense_reward += dense_reward
            self.episode_dense_return += dense_reward
            self.current_obs = obs

            success = extract_adroit_success(self.env, info)
            self.success_flag = bool(self.success_flag or success)
            sparse_r = self.sparse_success_reward if success else self.sparse_step_reward
            total_sparse_reward += float(discount) * float(sparse_r)
            self.episode_sparse_return += float(discount) * float(sparse_r)
            discount *= float(self.gamma)

            raw_truncated = bool(
                trunc or info.get("truncated", False) or info.get("TimeLimit.truncated", False)
            )
            raw_terminated = bool(term or info.get("terminated", False)) and not raw_truncated

            if self.success_flag and self.terminate_on_success:
                terminated = True
            elif raw_terminated:
                terminated = True
            elif raw_truncated:
                truncated = True
                info["TimeLimit.truncated"] = True

            if self.current_step >= self.max_steps and not terminated:
                truncated = bool(self.timeout_on_horizon)
                info["TimeLimit.truncated"] = True

            if terminated or truncated:
                break

        info["is_success"] = bool(self.success_flag)
        info["success"] = bool(self.success_flag)
        info["exec_len"] = int(actual_steps)
        info["resampled"] = False
        info["dense_reward"] = float(total_dense_reward)
        info["dense_return"] = float(self.episode_dense_return)
        info["sparse_return"] = float(self.episode_sparse_return)
        if truncated:
            info["TimeLimit.truncated"] = True
        if terminated or truncated:
            info["episode_dense_return"] = float(self.episode_dense_return)
            info["episode_sparse_return"] = float(self.episode_sparse_return)
            info["terminal_observation"] = self.current_obs
            self.episode_count += 1

        return self.current_obs, float(total_sparse_reward), bool(terminated), bool(truncated), info


def make_adroit_raw_env(env_meta, env_id: int, args, render_offscreen: bool = False):
    base_seed = int(getattr(args, "seed", 42))
    worker_seed = base_seed + int(env_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)

    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    if ObsUtils.OBS_KEYS_TO_MODALITIES is None:
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["flat"], "rgb": []}})

    meta = copy.deepcopy(env_meta)
    meta.setdefault("env_kwargs", {})
    if int(getattr(args, "env_horizon_override", 0)) > 0:
        meta["env_kwargs"]["max_episode_steps"] = int(args.env_horizon_override)

    env = EnvUtils.create_env_from_metadata(
        env_meta=meta,
        render=False,
        render_offscreen=render_offscreen,
        use_image_obs=False,
        use_depth_obs=False,
    )
    gym_env = RobomimicToGymWrapper(env)

    # Best-effort Gymnasium seeding. EnvGym itself has no seed argument, so the
    # next reset through the wrapper still follows the seeded underlying RNG.
    try:
        env.env.reset(seed=worker_seed)
    except Exception:
        pass

    return InternalAdroitVariableChunkWrapper(
        gym_env,
        max_steps=int(getattr(args, "env_horizon_override", 200)),
        gamma=float(getattr(args, "gamma", 0.999)),
        resample_penalty=float(getattr(args, "resample_penalty", -5.0)),
        sparse_step_reward=float(getattr(args, "sparse_step_reward", -1.0)),
        sparse_success_reward=float(getattr(args, "sparse_success_reward", 0.0)),
        terminate_on_success=bool(getattr(args, "terminate_on_success", True)),
        timeout_on_horizon=bool(getattr(args, "timeout_on_horizon", True)),
        zero_action_fail_streak=int(getattr(args, "zero_action_fail_streak", 0)),
        zero_action_fail_reward=float(getattr(args, "zero_action_fail_reward", -1000.0)),
        env_id=int(env_id),
    )


class _AdroitSACMixin:
    def _finish_adroit_init(
        self, ckpt_dict: Dict, stats_npz_path: str, max_chunk_len: int, min_chunk_len: int
    ):
        shape_meta = ckpt_dict.get("shape_metadata", {})
        flat_shape = shape_meta.get("all_shapes", {}).get("flat", [39])
        obs_dim = int(np.prod(flat_shape))
        self._obs_keys = ["flat"]
        self._obs_dims = {"flat": obs_dim}
        self.action_dim = int(shape_meta.get("ac_dim", self.action_dim))
        self.min_chunk_len = int(min_chunk_len)
        self.action_space = spaces.Discrete(int(max_chunk_len) - self.min_chunk_len + 1)
        self.stats = np.load(stats_npz_path)
        self._action_mean = np.asarray(self.stats["action_mean"], dtype=np.float32)
        self._action_std = np.asarray(self.stats["action_std"], dtype=np.float32)

        self.total_obs_dim = (
            obs_dim
            + int(self.action_history_len) * int(self.action_dim)
            + int(max_chunk_len) * int(self.action_dim)
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(self.total_obs_dim),),
            dtype=np.float32,
        )
        print(
            f"[ADROIT-SAC] RL obs_dim={self.total_obs_dim} phys={obs_dim} "
            f"action_dim={self.action_space.n} k_range=[{self.min_chunk_len},{int(max_chunk_len)}]"
        )
        print(f"[ADROIT-SAC] actor/critic obs state={self.critic_obs_state_mode}")

    def __init__(
        self,
        *,
        policy,
        ckpt_dict: Dict,
        env_meta: Dict,
        stats_npz_path: str,
        max_chunk_len: int,
        device,
        n_envs: int,
        vecenv_start_method: str = "fork",
        action_history_len: int = 0,
        dp_infer_steps: int = 5,
        dp_sampler: str = "ddim",
        dp_amp: bool = False,
        dp_compile: bool = False,
        ipc_chunk_dtype: str = "fp16",
        dp_infer_gpus: str = "0",
        dp_grouping: str = "round_robin",
        dp_infer_parallel: bool = False,
        ipc_send_mode: str = "executed_only",
        perf_profile: bool = False,
        min_chunk_len: int = 0,
        resample_penalty: float = -5.0,
        td_target_next_obs_samples: int = 1,
        critic_obs_state_mode: str = "dp_raw",
        args=None,
    ):
        _ = td_target_next_obs_samples
        self._init_adroit_base(
            policy=policy,
            ckpt_dict=ckpt_dict,
            env_meta=env_meta,
            stats_npz_path=stats_npz_path,
            max_chunk_len=max_chunk_len,
            device=device,
            n_envs=n_envs,
            vecenv_start_method=vecenv_start_method,
            action_history_len=action_history_len,
            dp_infer_steps=dp_infer_steps,
            dp_sampler=dp_sampler,
            dp_amp=dp_amp,
            dp_compile=dp_compile,
            ipc_chunk_dtype=ipc_chunk_dtype,
            dp_infer_gpus=dp_infer_gpus,
            dp_grouping=dp_grouping,
            dp_infer_parallel=dp_infer_parallel,
            ipc_send_mode=ipc_send_mode,
            perf_profile=perf_profile,
            min_chunk_len=min_chunk_len,
            resample_penalty=resample_penalty,
            critic_obs_state_mode=critic_obs_state_mode,
            args=args,
        )

    def _init_adroit_base(
        self,
        *,
        policy,
        ckpt_dict: Dict,
        env_meta: Dict,
        stats_npz_path: str,
        max_chunk_len: int,
        device,
        n_envs: int,
        vecenv_start_method: str,
        action_history_len: int,
        dp_infer_steps: int,
        dp_sampler: str,
        dp_amp: bool,
        dp_compile: bool,
        ipc_chunk_dtype: str,
        dp_infer_gpus: str,
        dp_grouping: str,
        dp_infer_parallel: bool,
        ipc_send_mode: str,
        perf_profile: bool,
        min_chunk_len: int,
        resample_penalty: float,
        critic_obs_state_mode: str,
        args,
    ):
        os.environ["OPT_MIN_CHUNK_LEN"] = str(int(min_chunk_len))
        os.environ["OPT_TOOLHANG_STATS_NPZ"] = str(stats_npz_path)

        critic_obs_state_mode = str(critic_obs_state_mode).lower()
        valid_state_modes = {"stats_clip", "dp_raw"}
        if critic_obs_state_mode not in valid_state_modes:
            raise ValueError(
                f"[ADROIT-SAC] invalid critic_obs_state_mode={critic_obs_state_mode}, "
                f"expected one of {sorted(valid_state_modes)}"
            )
        # The parent constructor probes the overridden observation builder.
        self.critic_obs_state_mode = critic_obs_state_mode

        env_fns = [
            partial(make_adroit_raw_env, copy.deepcopy(env_meta), i, args, render_offscreen=False)
            for i in range(int(n_envs))
        ]
        raw_venv = SubprocVecEnv(env_fns, start_method=str(vecenv_start_method))

        super().__init__(
            venv=raw_venv,
            policy=policy,
            max_chunk_len=int(max_chunk_len),
            device=device,
            reward_offset=0.0,
            resample_penalty=float(resample_penalty),
            action_history_len=int(action_history_len),
            dp_infer_steps=int(dp_infer_steps),
            dp_sampler=str(dp_sampler),
            dp_amp=bool(dp_amp),
            dp_compile=bool(dp_compile),
            ipc_chunk_dtype=str(ipc_chunk_dtype),
            perf_profile=bool(perf_profile),
            dp_infer_gpus=str(dp_infer_gpus),
            dp_grouping=str(dp_grouping),
            dp_infer_parallel=bool(dp_infer_parallel),
            ipc_send_mode=str(ipc_send_mode),
        )
        self._finish_adroit_init(ckpt_dict, stats_npz_path, int(max_chunk_len), int(min_chunk_len))

    def _safe_norm_stats(self, key, dim):
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        if mean_key in self.stats and std_key in self.stats:
            mean = np.asarray(self.stats[mean_key], dtype=np.float32).reshape(-1)
            std = np.asarray(self.stats[std_key], dtype=np.float32).reshape(-1)
            if mean.shape[0] == int(dim) and std.shape[0] == int(dim):
                std = np.where(std < 1e-8, 1.0, std)
                return mean.reshape(1, int(dim)), std.reshape(1, int(dim))
        return np.zeros((1, int(dim)), dtype=np.float32), np.ones((1, int(dim)), dtype=np.float32)

    def _get_obs_vec(self, obs_dict):
        batch = {}
        if "flat" in obs_dict:
            batch["flat"] = np.asarray(obs_dict["flat"], dtype=np.float32).reshape(1, -1)
        return self._build_obs_matrix(batch)[0]

    def _build_obs_matrix(self, obs_batch):
        if "flat" in obs_batch:
            values = np.asarray(obs_batch["flat"], dtype=np.float32)
            n = int(values.shape[0])
            values = values.reshape(n, -1)
            dim = int(values.shape[1])
            self._obs_dims = {"flat": dim}
            mean, std = self._safe_norm_stats("flat", dim)
            obs = (values - mean) / std
        else:
            n = int(self.num_envs)
            dim = int(self._obs_dims.get("flat", 39))
            obs = np.zeros((n, dim), dtype=np.float32)
        obs = obs.astype(np.float32)
        if self.critic_obs_state_mode == "stats_clip":
            np.clip(obs, -10.0, 10.0, out=obs)
        return obs

    def _normalize_action_chunk(self, chunks, n: Optional[int] = None):
        chunks = np.asarray(chunks, dtype=np.float32)
        if chunks.ndim == 2:
            chunks = chunks[None, :, :]
        if n is None:
            n = int(chunks.shape[0])
        if chunks.shape[1] > int(self.max_chunk_len):
            chunks = chunks[:, : int(self.max_chunk_len), :]
        norm = (chunks - self._action_mean.reshape(1, 1, -1)) / self._action_std.reshape(1, 1, -1)
        if norm.shape[1] < int(self.max_chunk_len):
            pad = np.zeros(
                (n, int(self.max_chunk_len) - norm.shape[1], int(self.action_dim)), dtype=np.float32
            )
            norm = np.concatenate([norm, pad], axis=1)
        return norm.astype(np.float32)

    def _normalize_policy_obs_batch(self, obs_batch):
        if "flat" not in obs_batch:
            raise RuntimeError("[ADROIT-AAC] DP policy obs_batch missing 'flat'")
        values = np.asarray(obs_batch["flat"], dtype=np.float32)
        n = int(values.shape[0])
        values = values.reshape(n, -1)
        mean, std = self._safe_norm_stats("flat", values.shape[1])
        return {"flat": ((values - mean) / std).astype(np.float32)}

    def _get_batch_policy_plan(self, obs_batch):
        # RolloutPolicy normally normalizes observations before calling the algo.
        # AAC uses the lower-level trajectory API for batched planning, so Adroit
        # must reproduce that normalization explicitly.
        norm_obs_batch = self._normalize_policy_obs_batch(obs_batch)
        normalized_dp_actions = super()._get_batch_policy_plan(norm_obs_batch)
        normalized_dp_actions = np.asarray(normalized_dp_actions, dtype=np.float32)
        return (
            normalized_dp_actions * self._action_std.reshape(1, 1, -1)
            + self._action_mean.reshape(1, 1, -1)
        ).astype(np.float32)

    def _make_batch_rl_obs(self, obs_batch, chunks):
        obs_part = self._build_obs_matrix(obs_batch)
        n = int(obs_part.shape[0])
        history_part = np.zeros((n, 0), dtype=np.float32)
        if int(self.action_history_len) > 0:
            norm_history = (
                self.action_histories - self._action_mean.reshape(1, 1, -1)
            ) / self._action_std.reshape(1, 1, -1)
            history_part = norm_history.reshape(n, -1).astype(np.float32)
        chunk_part = self._normalize_action_chunk(chunks, n=n).reshape(n, -1).astype(np.float32)
        return np.concatenate([obs_part, history_part, chunk_part], axis=1).astype(np.float32)

    def _make_rl_obs(self, env_idx, obs_dict, chunk):
        obs_vec = self._get_obs_vec(obs_dict)
        history_part = np.zeros((0,), dtype=np.float32)
        if int(self.action_history_len) > 0:
            norm_history = (
                self.action_histories[int(env_idx)] - self._action_mean.reshape(1, -1)
            ) / self._action_std.reshape(1, -1)
            history_part = norm_history.reshape(-1).astype(np.float32)
        chunk_part = self._normalize_action_chunk(chunk, n=1).reshape(-1).astype(np.float32)
        return np.concatenate([obs_vec, history_part, chunk_part], axis=0).astype(np.float32)

    def get_terminal_rl_obs(self, terminal_obs_dict, env_idx):
        return self.get_terminal_rl_obs_samples(terminal_obs_dict, env_idx, 1)[0]

    def get_terminal_rl_obs_samples(self, terminal_obs_dict, env_idx, num_samples):
        """Return M independently sampled terminal RL observations for timeout bootstrapping."""
        num_samples = max(1, int(num_samples))
        obs_dict = terminal_obs_dict
        if isinstance(obs_dict, dict) and "terminal_observation" in obs_dict:
            obs_dict = obs_dict["terminal_observation"]
        if not isinstance(obs_dict, dict):
            obs_dict = {"flat": np.asarray(obs_dict, dtype=np.float32).reshape(-1)}
        if "flat" not in obs_dict and "obs" in obs_dict and isinstance(obs_dict["obs"], dict):
            obs_dict = obs_dict["obs"]
        obs_batch_for_dp = {"flat": np.asarray(obs_dict["flat"], dtype=np.float32).reshape(1, -1)}
        samples = []
        for _ in range(num_samples):
            chunk = self._get_batch_policy_plan(obs_batch_for_dp)[0]
            samples.append(
                self._make_rl_obs(int(env_idx), {"flat": obs_batch_for_dp["flat"][0]}, chunk)
            )
        return np.stack(samples, axis=0).astype(np.float32)


class BatchedAdaptiveWrapperAdroitSAC(_AdroitSACMixin, LegacyBatchedAdaptiveWrapper):
    pass


class BatchedAdaptiveWrapperAdroitSACAsyncQueue(_AdroitSACMixin, BatchedAdaptiveWrapperAsyncQueue):
    pass


BatchedAdaptiveWrapperAdroit = BatchedAdaptiveWrapperAdroitSAC
