"""
Adroit ResFiT environment backend, copied from the Adroit ExRL-Q wrapper.

The primitive step, success, sparse reward and boundary rules are copied from
Adroit_Q. AdroitExecutor exposes the interface expected by the unchanged ResFiT
chunk collector. No adaptive K policy or online BC is used by this executor.
"""

from __future__ import annotations

import copy
import os
import random
from functools import partial
from types import SimpleNamespace
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
        _ = options
        if seed is not None:
            # Match the Adroit_Q factory: seed the raw env, then reset EnvGym.
            self.env.env.reset(seed=int(seed))
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

    def close(self):
        self.env.env.close()


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
        collect_success_bc: bool = False,
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
        self.collect_success_bc = bool(collect_success_bc)
        self.env_id = int(env_id)

        self.current_step = 0
        self.episode_count = 0
        self.resample_streak = 0
        self.current_obs = None
        self.success_flag = False
        self.episode_dense_return = 0.0
        self.episode_sparse_return = 0.0
        self._bc_episode_raw_states = []
        self._bc_episode_raw_actions = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_step = 0
        self.resample_streak = 0
        self.success_flag = False
        self.episode_dense_return = 0.0
        self.episode_sparse_return = 0.0
        self._bc_episode_raw_states = []
        self._bc_episode_raw_actions = []
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
            if self.collect_success_bc:
                if not isinstance(self.current_obs, dict) or "flat" not in self.current_obs:
                    raise RuntimeError(
                        "Adroit online BC requires pre-action observation key 'flat'"
                    )
                self._bc_episode_raw_states.append(
                    np.asarray(self.current_obs["flat"], dtype=np.float32).reshape(-1).copy()
                )
                self._bc_episode_raw_actions.append(
                    np.asarray(chunk[i], dtype=np.float32).reshape(-1).copy()
                )
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
            if self.collect_success_bc and self.success_flag:
                if len(self._bc_episode_raw_states) != len(self._bc_episode_raw_actions):
                    raise RuntimeError("Adroit online BC state/action trajectory length mismatch")
                if self._bc_episode_raw_states:
                    info["bc_success_episode_raw_states"] = np.stack(
                        self._bc_episode_raw_states,
                        axis=0,
                    ).astype(np.float32)
                    info["bc_success_episode_raw_actions"] = np.stack(
                        self._bc_episode_raw_actions,
                        axis=0,
                    ).astype(np.float32)
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
        collect_success_bc=bool(getattr(args, "collect_success_bc", False)),
        env_id=int(env_id),
    )


class AdroitExecutor:
    """Adroit backend for the existing ChunkResidualVecEnv interface."""

    def __init__(
        self,
        policy,
        scaler,
        n_envs,
        seed,
        horizon,
        gamma,
        start_method="fork",
        ipc_chunk_dtype="fp16",
    ):
        self.n_envs = int(n_envs)
        self.scaler = scaler
        self.gamma = float(gamma)
        self.horizon = int(horizon)
        if ipc_chunk_dtype not in {"fp16", "fp32"}:
            raise ValueError("invalid Adroit IPC action dtype")
        self.ipc_chunk_dtype = ipc_chunk_dtype
        args = SimpleNamespace(
            seed=int(seed),
            env_horizon_override=self.horizon,
            gamma=self.gamma,
            resample_penalty=0.0,
            sparse_step_reward=-1.0,
            sparse_success_reward=0.0,
            terminate_on_success=True,
            timeout_on_horizon=True,
            collect_success_bc=False,
        )
        self.venv = SubprocVecEnv(
            [
                partial(make_adroit_raw_env, copy.deepcopy(policy.env_meta), i, args)
                for i in range(self.n_envs)
            ],
            start_method=start_method,
        )

    def reset_all(self, base_seed=None):
        if base_seed is not None:
            self.venv.seed(int(base_seed))
        obs = self.venv.reset()
        return np.asarray(obs["flat"], dtype=np.float32).reshape(self.n_envs, -1)

    def step_variable_k(
        self, action_chunks, exec_lens, *, gamma, reward_offset, discount_within_chunk
    ):
        if float(gamma) != self.gamma or float(reward_offset) != 0.0 or not discount_within_chunk:
            raise ValueError("Adroit backend already supplies discounted sparse -1/0 rewards")
        chunks = np.asarray(self.scaler.dp_to_raw(action_chunks), dtype=np.float32)
        lengths = np.asarray(exec_lens, dtype=np.int64).reshape(-1)
        obs, rewards, dones, infos = self.venv.step(
            [
                {
                    "chunk": chunks[i].astype(
                        np.float16 if self.ipc_chunk_dtype == "fp16" else np.float32
                    ),
                    "k": int(lengths[i]),
                }
                for i in range(self.n_envs)
            ]
        )
        converted_infos = []
        for info in infos:
            item = dict(info)
            # Collector timeout bootstrap expects a raw flat state vector.
            if "terminal_observation" in item:
                terminal = item["terminal_observation"]
                if isinstance(terminal, dict):
                    terminal = terminal["flat"]
                item["terminal_observation"] = (
                    np.asarray(terminal, dtype=np.float32).reshape(-1).copy()
                )
            converted_infos.append(item)
        return SimpleNamespace(
            next_state=np.asarray(obs["flat"], dtype=np.float32).reshape(self.n_envs, -1),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=bool),
            infos=converted_infos,
            exec_lens=np.asarray([item["exec_len"] for item in infos], dtype=np.int64),
        )

    def close(self):
        self.venv.close()
