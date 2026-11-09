"""Adroit fixed-TA environment backend for the existing DICE chunk collector.

Primitive environment construction, success detection, sparse reward, and
termination/truncation handling are copied from the validated Adroit ExRL-Q
and ResFiT adapters.  This backend returns checkpoint-normalized flat states
and undiscounted ``-1 until success, 0 on success`` chunk rewards.
"""

from __future__ import annotations

import copy
import random
from functools import partial
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - legacy remote fallback
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
        return any(_as_flat_bool_success(value) for value in raw.values())
    if isinstance(raw, (list, tuple, np.ndarray)):
        values = list(raw)
        return _as_flat_bool_success(values[-1]) if values else False
    return bool(raw)


def extract_adroit_success(env: Any, info: Optional[Dict] = None) -> bool:
    """Read explicit Adroit success; dense reward is never a success signal."""

    info = info or {}
    candidates = []
    current = env
    for _ in range(5):
        if current is None:
            break
        candidates.append(current)
        current = getattr(current, "env", None)
    for candidate in candidates:
        is_success = getattr(candidate, "is_success", None)
        if callable(is_success):
            try:
                if _as_flat_bool_success(is_success()):
                    return True
            except Exception:
                pass
        raw = getattr(candidate, "unwrapped", None)
        for name in ("_is_success", "_check_success"):
            function = getattr(raw, name, None) if raw is not None else None
            if callable(function):
                try:
                    if _as_flat_bool_success(function()):
                        return True
                except Exception:
                    pass
    return _as_flat_bool_success(info.get("is_success")) or _as_flat_bool_success(
        info.get("success")
    )


class RobomimicToGymWrapper(gym.Env):
    """Expose robomimic EnvGym through Gymnasium's five-value API."""

    def __init__(self, env):
        self.env = env
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(env.action_dimension),),
            dtype=np.float32,
        )
        observation = self.env.reset()
        self.observation_space = spaces.Dict(
            {
                key: spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=np.asarray(value).shape,
                    dtype=np.asarray(value).dtype,
                )
                for key, value in observation.items()
            }
        )

    def reset(self, seed=None, options=None):
        del options
        if seed is not None:
            self.env.env.reset(seed=int(seed))
        return self.env.reset(), {}

    def step(self, action):
        observation, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        info = dict(info) if isinstance(info, dict) else {}
        truncated = bool(info.get("truncated", False))
        terminated = bool(info.get("terminated", False)) or (bool(done) and not truncated)
        return observation, float(reward), terminated, truncated, info

    def close(self):
        self.env.env.close()


class InternalAdroitFixedChunkWrapper(gym.Wrapper):
    """Execute one fixed action chunk with project sparse-reward semantics."""

    def __init__(self, env, *, max_steps: int = 200, env_id: int = 0):
        super().__init__(env)
        self.max_steps = int(max_steps)
        self.env_id = int(env_id)
        self.current_step = 0
        self.current_obs = None
        self.success_flag = False
        self.episode_sparse_return = 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.current_step = 0
        self.current_obs = observation
        self.success_flag = False
        self.episode_sparse_return = 0.0
        return observation, info

    def step(self, action_data):
        chunk = np.asarray(action_data["chunk"], dtype=np.float32)
        execution_length = int(action_data["k"])
        if chunk.ndim != 2 or execution_length <= 0 or execution_length > len(chunk):
            raise ValueError("invalid Adroit fixed action chunk")

        reward_sum = 0.0
        dense_reward_sum = 0.0
        actual_steps = 0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        for index in range(execution_length):
            observation, dense_reward, term, trunc, step_info = self.env.step(chunk[index])
            info = dict(step_info) if isinstance(step_info, dict) else {}
            self.current_step += 1
            actual_steps += 1
            self.current_obs = observation
            dense_reward_sum += float(dense_reward)

            success = extract_adroit_success(self.env, info)
            self.success_flag = bool(self.success_flag or success)
            primitive_reward = 0.0 if success else -1.0
            reward_sum += primitive_reward
            self.episode_sparse_return += primitive_reward

            raw_truncated = bool(
                trunc or info.get("truncated", False) or info.get("TimeLimit.truncated", False)
            )
            raw_terminated = bool(term or info.get("terminated", False)) and not (raw_truncated)
            if self.success_flag:
                terminated = True
            elif raw_terminated:
                terminated = True
            elif raw_truncated:
                truncated = True
            if self.current_step >= self.max_steps and not terminated:
                truncated = True
            if terminated or truncated:
                break

        info["is_success"] = bool(self.success_flag)
        info["success"] = bool(self.success_flag)
        info["exec_len"] = int(actual_steps)
        info["dense_reward"] = float(dense_reward_sum)
        info["sparse_return"] = float(self.episode_sparse_return)
        if truncated:
            info["TimeLimit.truncated"] = True
            info["truncated"] = True
        if terminated:
            info["terminated"] = True
        if terminated or truncated:
            info["terminal_observation"] = self.current_obs
        return (
            self.current_obs,
            float(reward_sum),
            bool(terminated),
            bool(truncated),
            info,
        )


def make_adroit_raw_env(
    env_meta: Dict[str, object], env_id: int, seed: int, max_episode_steps: int
):
    worker_seed = int(seed) + int(env_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    if ObsUtils.OBS_KEYS_TO_MODALITIES is None:
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["flat"], "rgb": []}})
    metadata = copy.deepcopy(env_meta)
    metadata.setdefault("env_kwargs", {})["max_episode_steps"] = int(max_episode_steps)
    metadata["env_kwargs"].pop("horizon", None)
    environment = EnvUtils.create_env_from_metadata(
        env_meta=metadata,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        use_depth_obs=False,
    )
    gym_environment = RobomimicToGymWrapper(environment)
    try:
        environment.env.reset(seed=worker_seed)
    except Exception:
        pass
    return InternalAdroitFixedChunkWrapper(
        gym_environment,
        max_steps=int(max_episode_steps),
        env_id=int(env_id),
    )


class AdroitDICEExecutor:
    """Adapter implementing ``RobomimicExecutor``'s used surface."""

    def __init__(
        self,
        *,
        env_meta: Dict[str, object],
        policy,
        n_envs: int,
        max_episode_steps: int,
        seed: int,
        start_method: str = "fork",
        ipc_chunk_dtype: str = "fp16",
    ):
        self.n_envs = int(n_envs)
        self.policy = policy
        self.max_episode_steps = int(max_episode_steps)
        if ipc_chunk_dtype not in ("fp16", "fp32"):
            raise ValueError("ipc_chunk_dtype must be fp16 or fp32")
        self.ipc_chunk_dtype = str(ipc_chunk_dtype)
        self.venv = SubprocVecEnv(
            [
                partial(
                    make_adroit_raw_env,
                    copy.deepcopy(env_meta),
                    index,
                    int(seed),
                    self.max_episode_steps,
                )
                for index in range(self.n_envs)
            ],
            start_method=str(start_method),
        )

    def _normalize_observations(self, observations) -> np.ndarray:
        if not isinstance(observations, dict) or "flat" not in observations:
            raise RuntimeError("Adroit vector environment did not return flat state")
        raw = np.asarray(observations["flat"], dtype=np.float32).reshape(self.n_envs, -1)
        normalized = self.policy.normalize_raw_state(raw)
        if normalized.shape != (self.n_envs, self.policy.obs_dim):
            raise RuntimeError("normalized Adroit vector state shape mismatch")
        return normalized

    def reset_all(self, base_seed=None):
        if base_seed is not None:
            self.venv.seed(int(base_seed))
        return self._normalize_observations(self.venv.reset())

    def step_variable_k(
        self,
        action_chunks,
        exec_lens,
        *,
        gamma,
        reward_offset,
        discount_within_chunk,
    ):
        if gamma is not None or float(reward_offset) != 0.0 or discount_within_chunk:
            raise ValueError("Adroit DICE backend supplies undiscounted sparse -1/0 chunk rewards")
        normalized_chunks = np.asarray(action_chunks, dtype=np.float32)
        if normalized_chunks.shape[:1] != (self.n_envs,):
            raise ValueError("Adroit action chunk batch mismatch")
        raw_chunks = self.policy.denormalize_action(normalized_chunks)
        lengths = np.asarray(exec_lens, dtype=np.int64).reshape(-1)
        if lengths.shape != (self.n_envs,):
            raise ValueError("Adroit execution-length batch mismatch")
        dtype = np.float16 if self.ipc_chunk_dtype == "fp16" else np.float32
        observations, rewards, dones, infos = self.venv.step(
            [
                {
                    "chunk": raw_chunks[index].astype(dtype),
                    "k": int(lengths[index]),
                }
                for index in range(self.n_envs)
            ]
        )
        converted_infos = []
        for info in infos:
            item = dict(info)
            terminal = item.get("terminal_observation")
            if terminal is not None:
                if isinstance(terminal, dict):
                    terminal = terminal["flat"]
                item["terminal_observation"] = self.policy.normalize_raw_state(
                    np.asarray(terminal, dtype=np.float32).reshape(-1)
                )
            converted_infos.append(item)
        return SimpleNamespace(
            next_state=self._normalize_observations(observations),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=bool),
            infos=converted_infos,
            exec_lens=np.asarray([item["exec_len"] for item in converted_infos], dtype=np.int64),
        )

    def close(self) -> None:
        self.venv.close()
