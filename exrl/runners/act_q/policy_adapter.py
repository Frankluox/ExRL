"""Base-policy inference and imitation losses for chunked control."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


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
        self.ddpm = _CfgNode(
            enabled=False, num_train_timesteps=0, num_inference_timesteps=0
        )
        self.ddim = _CfgNode(
            enabled=False, num_train_timesteps=0, num_inference_timesteps=0
        )

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
        self.obs_shapes = {
            key: (0,) for key in (self.state_obs_keys + self.env_obs_keys)
        }

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

        self.algo_config = _AlgoConfig(
            horizon_steps=self.horizon_steps, cond_steps=self.cond_steps
        )
        self.use_action_scheduler = False
        self.use_ood_monitor = False
        self.ema = None
        self.nets = nn.ModuleDict({"policy": nn.ModuleDict()})

    def set_eval(self):
        self.eval()
        self.policy_model.eval()

    def set_train(self):
        self.train()
        self.policy_model.train()

    def reset(self):
        reset_fn = getattr(self.policy_model, "reset", None)
        if callable(reset_fn):
            reset_fn()
        return None

    def _concat_obs_parts(
        self, obs_dict: Dict[str, torch.Tensor], keys: List[str]
    ) -> torch.Tensor:
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

    def _build_policy_obs(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
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

    def _build_bc_policy_batch(
        self,
        actions: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
        action_is_pad: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if "state" not in conditions:
            raise KeyError("ACT BC conditions must contain 'state'")
        states = conditions["state"].to(self.device, dtype=torch.float32)
        actions = actions.to(self.device, dtype=torch.float32)
        if states.ndim != 3:
            raise ValueError(
                f"ACT BC states must be [B, C, D], got {tuple(states.shape)}"
            )
        if actions.ndim != 3:
            raise ValueError(
                f"ACT BC actions must be [B, H, A], got {tuple(actions.shape)}"
            )
        if int(states.shape[1]) != int(self.cond_steps):
            raise ValueError(
                f"ACT BC cond_steps mismatch: got {int(states.shape[1])}, expected {self.cond_steps}"
            )
        if int(states.shape[2]) != int(self.obs_dim):
            raise ValueError(
                f"ACT BC obs_dim mismatch: got {int(states.shape[2])}, expected {self.obs_dim}"
            )
        if tuple(actions.shape[1:]) != (int(self.horizon_steps), int(self.ac_dim)):
            raise ValueError(
                "ACT BC action shape mismatch: "
                f"got {tuple(actions.shape[1:])}, expected {(self.horizon_steps, self.ac_dim)}"
            )
        if int(self.cond_steps) != 1:
            raise ValueError(
                "ACT online BC currently requires n_obs_steps=1, matching the ACT AAC inference path; "
                f"checkpoint has n_obs_steps={self.cond_steps}"
            )

        current_state = states[:, -1]
        robot_dim = int(
            self.policy_model.config.robot_state_feature.shape[0]
            if self.policy_model.config.robot_state_feature is not None
            else 0
        )
        env_dim = int(
            self.policy_model.config.env_state_feature.shape[0]
            if self.policy_model.config.env_state_feature is not None
            else 0
        )
        if robot_dim + env_dim != int(current_state.shape[-1]):
            raise ValueError(
                "ACT BC state split mismatch: "
                f"robot={robot_dim} env={env_dim} total={int(current_state.shape[-1])}"
            )

        batch: Dict[str, torch.Tensor] = {}
        if robot_dim > 0:
            batch[OBS_STATE] = current_state[:, :robot_dim]
        if env_dim > 0:
            batch[OBS_ENV_STATE] = current_state[:, robot_dim : robot_dim + env_dim]
        batch[ACTION] = actions
        expected_pad_shape = (int(actions.shape[0]), int(actions.shape[1]))
        if action_is_pad is None:
            action_is_pad = torch.zeros(
                expected_pad_shape,
                dtype=torch.bool,
                device=self.device,
            )
        else:
            action_is_pad = action_is_pad.to(self.device, dtype=torch.bool)
            if tuple(action_is_pad.shape) != expected_pad_shape:
                raise ValueError(
                    "ACT BC action_is_pad shape mismatch: "
                    f"got {tuple(action_is_pad.shape)}, expected {expected_pad_shape}"
                )
        batch["action_is_pad"] = action_is_pad
        processed = self.preprocessor(batch)
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in processed.items()
        }

    def act_bc_loss(
        self,
        actions: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
        action_is_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss, _ = self.act_bc_loss_terms(
            actions, conditions, action_is_pad=action_is_pad
        )
        return loss

    def act_bc_loss_terms(
        self,
        actions: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
        *,
        action_is_pad: Optional[torch.Tensor] = None,
        teacher_policy_model: Optional[nn.Module] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self._build_bc_policy_batch(
            actions,
            conditions,
            action_is_pad=action_is_pad,
        )
        self.policy_model.train()
        bc_loss, _ = self.policy_model(batch)
        teacher_loss = bc_loss.new_zeros(())
        if teacher_policy_model is None:
            return bc_loss, teacher_loss

        # Keep the constraint on deterministic inference behavior. The native
        # ACT reconstruction/KL loss above still uses the training VAE path.
        inference_batch = {
            key: value
            for key, value in batch.items()
            if key not in {ACTION, "action_is_pad"}
        }
        student_model = self.policy_model.model
        was_training = bool(student_model.training)
        student_model.eval()
        teacher_policy_model.eval()
        student_actions = student_model(inference_batch)[0]
        with torch.no_grad():
            teacher_actions = teacher_policy_model.model(inference_batch)[0]
        teacher_loss = F.mse_loss(
            student_actions, teacher_actions.detach(), reduction="mean"
        )
        student_model.train(was_training)
        return bc_loss, teacher_loss

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
