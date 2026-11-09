from __future__ import annotations

import copy
import random
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces


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
        values = list(raw)
        if not values:
            return False
        return _as_flat_bool_success(values[-1])
    return bool(raw)


def extract_adroit_success(env: Any, info: Optional[Dict] = None) -> bool:
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
        return self.env.reset(seed=seed, options=options), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        info = dict(info) if isinstance(info, dict) else {}
        terminated = bool(info.get("terminated", False)) or (
            bool(done) and not bool(info.get("truncated", False))
        )
        truncated = bool(info.get("truncated", False))
        return obs, float(reward), terminated, truncated, info

    def render(self, mode="rgb_array", height=512, width=512):
        return self.env.render(mode=mode, height=height, width=width)

    def close(self):
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


class AdroitDSRLActionChunkWrapper(gym.Env):
    def __init__(
        self,
        env,
        *,
        obs_mean,
        obs_std,
        action_mean,
        action_std,
        action_dim: int,
        obs_dim: int,
        action_horizon: int,
        execute_steps: int,
        action_magnitude: float,
        max_episode_steps: int = 200,
        sparse_step_reward: float = -1.0,
        sparse_success_reward: float = 0.0,
        terminate_on_success: bool = True,
        timeout_on_horizon: bool = True,
    ):
        self.env = env
        self.obs_mean = np.asarray(obs_mean, dtype=np.float32).reshape(1, -1)
        self.obs_std = np.asarray(obs_std, dtype=np.float32).reshape(1, -1)
        self.obs_std = np.where(self.obs_std < 1e-8, 1.0, self.obs_std)
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
        self.action_mean = np.asarray(action_mean, dtype=np.float32).reshape(1, -1)
        self.action_std = np.asarray(action_std, dtype=np.float32).reshape(1, -1)
        self.action_std = np.where(self.action_std < 1e-8, 1.0, self.action_std)
        if self.action_mean.shape != (1, self.action_dim):
            raise ValueError(
                f"action_mean shape {self.action_mean.shape} != {(1, self.action_dim)}"
            )
        if self.action_std.shape != (1, self.action_dim):
            raise ValueError(f"action_std shape {self.action_std.shape} != {(1, self.action_dim)}")
        self.action_horizon = int(action_horizon)
        self.execute_steps = max(1, min(int(execute_steps), self.action_horizon))
        self.max_episode_steps = int(max_episode_steps)
        self.sparse_step_reward = float(sparse_step_reward)
        self.sparse_success_reward = float(sparse_success_reward)
        self.terminate_on_success = bool(terminate_on_success)
        self.timeout_on_horizon = bool(timeout_on_horizon)
        self.current_step = 0
        self.episode_success = False
        self.episode_dense_return = 0.0
        self.episode_sparse_return = 0.0
        self.current_obs = None

        noise_bound = float(action_magnitude)
        if not np.isfinite(noise_bound) or noise_bound <= 0:
            raise ValueError(f"action_magnitude must be finite and positive, got {noise_bound}")
        self.action_space = spaces.Box(
            low=-noise_bound * np.ones(self.action_horizon * self.action_dim, dtype=np.float32),
            high=noise_bound * np.ones(self.action_horizon * self.action_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    def _normalize_flat_obs(self, obs_dict) -> np.ndarray:
        if isinstance(obs_dict, dict):
            if "flat" not in obs_dict:
                raise KeyError(f"Adroit obs missing 'flat'; keys={list(obs_dict.keys())}")
            obs = np.asarray(obs_dict["flat"], dtype=np.float32).reshape(1, -1)
        else:
            obs = np.asarray(obs_dict, dtype=np.float32).reshape(1, -1)
        norm = (obs - self.obs_mean) / self.obs_std
        return norm.reshape(-1).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(int(seed))
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))
        raw_obs, info = self.env.reset(seed=seed, options=options)
        self.current_step = 0
        self.episode_success = False
        self.episode_dense_return = 0.0
        self.episode_sparse_return = 0.0
        self.current_obs = raw_obs
        return self._normalize_flat_obs(raw_obs), dict(info or {})

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(self.action_horizon, self.action_dim)
        total_sparse_reward = 0.0
        total_dense_reward = 0.0
        actual_steps = 0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        obs = self.current_obs

        for i in range(min(self.execute_steps, int(action.shape[0]))):
            physical_action = (
                action[i].reshape(1, -1) * self.action_std + self.action_mean
            ).reshape(-1)
            obs, dense_reward, term, trunc, step_info = self.env.step(physical_action)
            info = dict(step_info) if isinstance(step_info, dict) else {}
            dense_reward = float(dense_reward)
            self.current_step += 1
            actual_steps += 1
            total_dense_reward += dense_reward
            self.episode_dense_return += dense_reward
            self.current_obs = obs

            success = extract_adroit_success(self.env, info)
            self.episode_success = bool(self.episode_success or success)
            sparse_r = self.sparse_success_reward if success else self.sparse_step_reward
            # Match robomimic DSRL: a chunk is one MDP step with summed rewards.
            total_sparse_reward += float(sparse_r)
            self.episode_sparse_return += float(sparse_r)

            raw_truncated = bool(
                trunc or info.get("truncated", False) or info.get("TimeLimit.truncated", False)
            )
            raw_terminated = bool(term or info.get("terminated", False)) and not raw_truncated
            if self.episode_success and self.terminate_on_success:
                terminated = True
                truncated = False
            elif self.current_step >= self.max_episode_steps:
                truncated = bool(self.timeout_on_horizon)
                terminated = not truncated
            elif raw_truncated:
                truncated = bool(self.timeout_on_horizon)
                terminated = not truncated
            elif raw_terminated:
                terminated = True
                truncated = False

            if terminated or truncated:
                info["TimeLimit.truncated"] = bool(truncated)

            if terminated or truncated:
                break

        norm_obs = self._normalize_flat_obs(obs)
        info["is_success"] = bool(self.episode_success)
        info["success"] = bool(self.episode_success)
        info["chunk_env_steps"] = int(max(actual_steps, 1))
        info["exec_len"] = int(actual_steps)
        info["dense_reward"] = float(total_dense_reward)
        info["dense_return"] = float(self.episode_dense_return)
        info["sparse_return"] = float(self.episode_sparse_return)
        if terminated or truncated:
            info["TimeLimit.truncated"] = bool(truncated)
            info["terminal_observation"] = norm_obs
            info["episode_dense_return"] = float(self.episode_dense_return)
            info["episode_sparse_return"] = float(self.episode_sparse_return)
        return norm_obs, float(total_sparse_reward), bool(terminated), bool(truncated), info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


def make_adroit_raw_env(env_meta: Dict, env_id: int, cfg, render_offscreen: bool = False):
    base_seed = int(cfg.seed)
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
    meta["env_kwargs"]["max_episode_steps"] = int(cfg.env.max_episode_steps)
    env = EnvUtils.create_env_from_metadata(
        env_meta=meta,
        render=False,
        render_offscreen=render_offscreen,
        use_image_obs=False,
        use_depth_obs=False,
    )
    try:
        env.reset(seed=worker_seed)
    except Exception:
        pass
    return RobomimicToGymWrapper(env)


def make_adroit_dsrl_env(
    env_meta: Dict,
    obs_mean,
    obs_std,
    action_mean,
    action_std,
    cfg,
    env_id: int = 0,
):
    raw_env = make_adroit_raw_env(env_meta, env_id, cfg, render_offscreen=False)
    return AdroitDSRLActionChunkWrapper(
        raw_env,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
        action_dim=int(cfg.action_dim),
        obs_dim=int(cfg.obs_dim),
        action_horizon=int(getattr(cfg, "diffusion_horizon_steps", cfg.act_steps)),
        execute_steps=int(getattr(cfg, "execute_steps", cfg.act_steps)),
        action_magnitude=float(cfg.train.action_magnitude),
        max_episode_steps=int(cfg.env.max_episode_steps),
        sparse_step_reward=float(cfg.env.sparse_step_reward),
        sparse_success_reward=float(cfg.env.sparse_success_reward),
        terminate_on_success=bool(cfg.env.terminate_on_success),
        timeout_on_horizon=bool(cfg.env.timeout_on_horizon),
    )
