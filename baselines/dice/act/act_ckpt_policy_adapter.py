"""
ACT checkpoint adapter for DICE-RL.

Goal: expose the same minimal `_get_action_trajectory` interface that the legacy
DICE-RL's frozen-prior wrapper expects, backed by the exact low-dimensional ACT
inference path used by the audited ACT ExRL-Q runs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE


class _CfgNode:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _AlgoConfig:
    def __init__(self, horizon_steps: int, cond_steps: int):
        self.horizon = _CfgNode(
            observation_horizon=int(cond_steps),
            action_horizon=int(horizon_steps),
            prediction_horizon=int(horizon_steps),
        )
        # ACT is not diffusion-based, but the legacy wrapper may inspect these
        # fields. Keep them present and disabled.
        self.ddpm = _CfgNode(enabled=False, num_train_timesteps=0, num_inference_timesteps=0)
        self.ddim = _CfgNode(enabled=False, num_train_timesteps=0, num_inference_timesteps=0)

    @contextmanager
    def values_unlocked(self):
        yield self


class ACTCkptPolicyAdapter(nn.Module):
    """
    Mimics the minimal robomimic policy interface used by AAC wrappers.
    """

    def __init__(
        self,
        *,
        policy: nn.Module,
        preprocessor,
        postprocessor,
        state_obs_keys: List[str],
        env_obs_keys: List[str],
        use_state_as_env_when_missing: bool,
        device: torch.device,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.policy_model = policy.to(self.device).eval()
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.state_obs_keys = list(state_obs_keys)
        self.env_obs_keys = list(env_obs_keys)
        self.use_state_as_env_when_missing = bool(use_state_as_env_when_missing)
        self.obs_shapes = {key: (0,) for key in (self.state_obs_keys + self.env_obs_keys)}

        action_feature = self.policy_model.config.action_feature
        if action_feature is None:
            raise ValueError("ACT config missing action feature")
        self.ac_dim = int(action_feature.shape[0])
        self.horizon_steps = int(self.policy_model.config.chunk_size)
        self.cond_steps = int(self.policy_model.config.n_obs_steps)
        self.obs_dim = int(
            (
                self.policy_model.config.robot_state_feature.shape[0]
                if self.policy_model.config.robot_state_feature
                else 0
            )
            + (
                self.policy_model.config.env_state_feature.shape[0]
                if self.policy_model.config.env_state_feature
                else 0
            )
        )

        self.algo_config = _AlgoConfig(horizon_steps=self.horizon_steps, cond_steps=self.cond_steps)
        self.use_action_scheduler = False
        self.use_ood_monitor = False
        self.ema = None
        self.nets = nn.ModuleDict({"policy": nn.ModuleDict()})

    def set_eval(self):
        self.eval()
        self.policy_model.eval()

    def reset(self):
        reset_fn = getattr(self.policy_model, "reset", None)
        if callable(reset_fn):
            reset_fn()
        return None

    def _concat_obs_parts(self, obs_dict: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
        pieces = []
        batch_size = None
        for key in keys:
            value = obs_dict.get(key, None)
            if value is None:
                continue
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, dtype=torch.float32)
            value = value.to(dtype=torch.float32)
            if value.ndim == 1:
                value = value.unsqueeze(0)
            else:
                value = value.reshape(value.shape[0], -1)
            batch_size = int(value.shape[0]) if batch_size is None else batch_size
            pieces.append(value)
        if not pieces:
            if batch_size is None:
                for value in obs_dict.values():
                    if torch.is_tensor(value):
                        batch_size = int(value.shape[0]) if value.ndim > 1 else 1
                        break
            if batch_size is None:
                batch_size = 1
            return torch.zeros((batch_size, 0), dtype=torch.float32)
        return torch.cat(pieces, dim=-1)

    def _build_policy_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_vec = self._concat_obs_parts(obs_dict, self.state_obs_keys)
        built = {OBS_STATE: state_vec}
        need_env_state = self.policy_model.config.env_state_feature is not None
        if need_env_state:
            if self.env_obs_keys:
                env_vec = self._concat_obs_parts(obs_dict, self.env_obs_keys)
            elif self.use_state_as_env_when_missing:
                env_vec = state_vec.clone()
            else:
                raise ValueError(
                    "ACT checkpoint expects observation.environment_state but env_obs_keys is empty"
                )
            built[OBS_ENV_STATE] = env_vec
        return built

    @torch.no_grad()
    def _get_action_trajectory(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        policy_obs = self._build_policy_obs(obs_dict)
        processed = self.preprocessor(policy_obs)
        action_chunk = self.policy_model.predict_action_chunk(processed)
        action_chunk = self.postprocessor(action_chunk)
        if not torch.is_tensor(action_chunk):
            action_chunk = torch.as_tensor(action_chunk, dtype=torch.float32)
        if action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(0)
        return action_chunk.to(self.device, dtype=torch.float32)


@dataclass
class ACTCkptPolicyBundle:
    policy: ACTCkptPolicyAdapter
    source_path: str
    state_obs_keys: List[str]
    env_obs_keys: List[str]
    meta: Dict


class ACTCkptPolicyWrapper:
    """
    Thin wrapper so callers can keep using `wrapper.policy`.
    """

    def __init__(self, bundle: ACTCkptPolicyBundle):
        self.policy = bundle.policy
        self._bundle = bundle

    @property
    def meta(self) -> Dict:
        return self._bundle.meta

    @property
    def source_path(self) -> str:
        return self._bundle.source_path

    @property
    def state_obs_keys(self) -> List[str]:
        return list(self._bundle.state_obs_keys)

    @property
    def env_obs_keys(self) -> List[str]:
        return list(self._bundle.env_obs_keys)
