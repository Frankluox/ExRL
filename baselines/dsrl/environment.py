import os
import numpy as np
from omegaconf import OmegaConf
import torch
import hydra
import sys
import gym
import gymnasium
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper
import json

from env.gym_utils.wrapper import wrapper_dict
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def _extract_success_flag(info):
    raw = info.get("is_success", False)
    if isinstance(raw, dict):
        return bool(raw.get("task", False))
    if isinstance(raw, (list, tuple, np.ndarray)):
        if len(raw) == 0:
            return False
        raw = raw[-1]
        if isinstance(raw, dict):
            return bool(raw.get("task", False))
    return bool(raw)


def _query_env_success(env):
    seen = set()
    stack = [env]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        cur_id = id(cur)
        if cur_id in seen:
            continue
        seen.add(cur_id)
        if hasattr(cur, "is_success"):
            try:
                raw = cur.is_success()
            except Exception:
                raw = None
            if raw is not None:
                if isinstance(raw, dict):
                    return bool(raw.get("task", False))
                if isinstance(raw, (list, tuple, np.ndarray)):
                    if len(raw) == 0:
                        return False
                    raw = raw[-1]
                    if isinstance(raw, dict):
                        return bool(raw.get("task", False))
                return bool(raw)
        for attr in ("env", "unwrapped"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and id(nxt) not in seen:
                stack.append(nxt)
    return False


def make_robomimic_env(
    render=False, env="square", normalization_path=None, low_dim_keys=None, dppo_path=None
):
    wrappers = OmegaConf.create(
        {
            "robomimic_lowdim": {
                "normalization_path": normalization_path,
                "low_dim_keys": low_dim_keys,
            },
        }
    )
    obs_modality_dict = {
        "low_dim": (
            wrappers.robomimic_image.low_dim_keys
            if "robomimic_image" in wrappers
            else wrappers.robomimic_lowdim.low_dim_keys
        ),
        "rgb": (wrappers.robomimic_image.image_keys if "robomimic_image" in wrappers else None),
    }
    if obs_modality_dict["rgb"] is None:
        obs_modality_dict.pop("rgb")
    ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
    robomimic_env_cfg_path = os.path.join(os.path.dirname(normalization_path), f"{env}.json")
    with open(robomimic_env_cfg_path, "r") as f:
        env_meta = json.load(f)
    env_meta["reward_shaping"] = False
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=render,
        use_image_obs=False,
    )
    env.env.hard_reset = False
    for wrapper, args in wrappers.items():
        env = wrapper_dict[wrapper](env, **args)
    return env


class ObservationWrapperRobomimic(gym.Env):
    def __init__(
        self,
        env,
        reward_offset=1,
    ):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_offset = reward_offset
        self._episode_success = False

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, **kwargs):
        options = kwargs.get("options", {})
        new_seed = options.get("seed", None)
        if new_seed is not None:
            self.seed(seed=new_seed)
        raw_obs = self.env.reset()
        self._episode_success = False
        obs = raw_obs["state"].flatten()
        return obs

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        info = dict(info) if isinstance(info, dict) else {}
        success = _extract_success_flag(info)
        if not success:
            success = _query_env_success(self.env)
        self._episode_success = bool(self._episode_success or success)
        info["is_success"] = bool(self._episode_success)
        reward = reward - self.reward_offset
        obs = raw_obs["state"].flatten()
        return obs, reward, done, info

    def render(self, **kwargs):
        return self.env.render()


class ObservationWrapperGym(gym.Env):
    def __init__(
        self,
        env,
        normalization_path,
    ):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        normalization = np.load(normalization_path)
        self.obs_min = normalization["obs_min"]
        self.obs_max = normalization["obs_max"]
        self.action_min = normalization["action_min"]
        self.action_max = normalization["action_max"]

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, **kwargs):
        options = kwargs.get("options", {})
        new_seed = options.get("seed", None)
        if new_seed is not None:
            self.seed(seed=new_seed)
        raw_obs = self.env.reset()
        obs = self.normalize_obs(raw_obs)
        return obs

    def step(self, action):
        raw_action = self.unnormalize_action(action)
        raw_obs, reward, done, info = self.env.step(raw_action)
        obs = self.normalize_obs(raw_obs)
        return obs, reward, done, info

    def render(self, **kwargs):
        return self.env.render()

    def normalize_obs(self, obs):
        return 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)

    def unnormalize_action(self, action):
        action = (action + 1) / 2
        return action * (self.action_max - self.action_min) + self.action_min


class ActionChunkWrapper(gymnasium.Env):
    def __init__(self, env, cfg, max_episode_steps=300):
        self.max_episode_steps = max_episode_steps
        self.env = env
        self.act_steps = int(cfg.act_steps)
        self.rl_action_horizon = int(cfg.act_steps)
        self.action_horizon = int(getattr(cfg, "diffusion_horizon_steps", cfg.act_steps))
        self.action_dim = int(np.prod(env.action_space.shape))
        self.execute_steps = int(getattr(cfg, "execute_steps", self.act_steps))
        self.execute_steps = max(1, min(self.execute_steps, self.action_horizon))
        self.action_space = spaces.Box(
            low=np.tile(env.action_space.low, self.rl_action_horizon),
            high=np.tile(env.action_space.high, self.rl_action_horizon),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.ones(cfg.obs_dim), high=np.ones(cfg.obs_dim), dtype=np.float32
        )
        self.count = 0
        env_cfg = getattr(cfg, "env", None)
        self.terminate_on_success = bool(getattr(env_cfg, "terminate_on_success", False))
        self.timeout_on_horizon = bool(getattr(env_cfg, "timeout_on_horizon", False))
        self.episode_success = False

    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)
        self.count = 0
        self.episode_success = False
        return obs, {}

    def step(self, action):
        if len(action.shape) == 1:
            if action.size % self.action_dim == 0:
                action = action.reshape(-1, self.action_dim)
            else:
                action = action.reshape(self.action_horizon, -1)
        obs_ = []
        reward_ = []
        terminated_ = []
        truncated_ = []
        info_ = []
        exec_steps = min(self.execute_steps, action.shape[0])
        for i in range(exec_steps):
            self.count += 1
            obs_i, reward_i, done_i, info_i = self.env.step(action[i])
            info_i = dict(info_i) if isinstance(info_i, dict) else {}
            success_i = _extract_success_flag(info_i)
            self.episode_success = bool(self.episode_success or success_i)
            info_i["is_success"] = success_i
            terminated_i = bool(done_i)
            truncated_i = False
            if self.terminate_on_success and success_i:
                terminated_i = True
            if self.count >= self.max_episode_steps:
                if self.timeout_on_horizon and not success_i:
                    terminated_i = False
                    truncated_i = True
                else:
                    terminated_i = True
            obs_.append(obs_i)
            reward_.append(reward_i)
            terminated_.append(terminated_i)
            truncated_.append(truncated_i)
            info_.append(info_i)
            if terminated_i or truncated_i:
                break
        obs = obs_[-1]
        reward = sum(reward_)
        terminated = bool(np.max(terminated_)) if len(terminated_) > 0 else False
        truncated = bool(np.max(truncated_)) if len(truncated_) > 0 else False
        actual_exec_steps = len(reward_)
        info = info_[-1]
        info["is_success"] = bool(self.episode_success)
        if terminated or truncated:
            info["terminal_observation"] = obs
            info["TimeLimit.truncated"] = bool(truncated)
        info["chunk_env_steps"] = actual_exec_steps
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return


class DiffusionPolicyEnvWrapper(VecEnvWrapper):
    def __init__(self, env, cfg, base_policy):
        super().__init__(env)
        self.action_horizon = int(cfg.act_steps)
        self.diffusion_horizon = int(getattr(cfg, "diffusion_horizon_steps", cfg.act_steps))
        self.action_dim = cfg.action_dim
        self.action_magnitude = float(cfg.train.action_magnitude)
        self.action_space = spaces.Box(
            low=-self.action_magnitude * np.ones(self.action_dim * self.action_horizon),
            high=self.action_magnitude * np.ones(self.action_dim * self.action_horizon),
            dtype=np.float32,
        )
        self.obs_dim = cfg.obs_dim
        self.observation_space = spaces.Box(
            low=-np.ones(self.obs_dim), high=np.ones(self.obs_dim), dtype=np.float32
        )
        self.env = env
        self.device = cfg.model.device
        self.base_policy = base_policy
        self.obs = None

    def step_async(self, actions):
        actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
        actions = actions.view(-1, self.action_horizon, self.action_dim)
        if self.action_horizon != self.diffusion_horizon:
            full_actions = torch.randn(
                actions.shape[0],
                self.diffusion_horizon,
                self.action_dim,
                device=self.device,
                dtype=actions.dtype,
            )
            full_actions = torch.clamp(full_actions, -self.action_magnitude, self.action_magnitude)
            copy_steps = min(self.action_horizon, self.diffusion_horizon)
            full_actions[:, :copy_steps, :] = actions[:, :copy_steps, :]
            actions = full_actions
        diffused_actions = self.base_policy(self.obs, actions)
        self.venv.step_async(diffused_actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        obs_out = self.obs
        return obs_out.detach().cpu().numpy(), rewards, dones, infos

    def reset(self):
        obs = self.venv.reset()
        self.obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        obs_out = self.obs
        return obs_out.detach().cpu().numpy()
