"""Online-success BC adapter for native robomimic diffusion policies."""

from __future__ import annotations

import copy
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import robomimic.utils.tensor_utils as TensorUtils


def active_inference_policy_module(algo) -> nn.Module:
    """Return the exact module used by DiffusionPolicyUNet rollout inference."""
    ema = getattr(algo, "ema", None)
    averaged_model = getattr(ema, "averaged_model", None) if ema is not None else None
    if averaged_model is not None:
        return averaged_model
    nets = getattr(algo, "nets", None)
    if nets is None:
        raise RuntimeError("robomimic diffusion policy has neither EMA nor nets")
    return nets


class AdroitOnlineTeacherAdapter:
    """Train the checkpoint's live inference network with native DP epsilon loss.

    Adroit checkpoints were trained with robomimic EMA enabled, and rollout uses
    ``algo.ema.averaged_model`` rather than ``algo.nets``. This adapter therefore
    optimizes that exact live inference module. The pretraining EMA updater is
    intentionally not invoked during online BC.
    """

    def __init__(self, algo):
        self.algo = algo
        self.device = torch.device(algo.device)
        self.live_policy = active_inference_policy_module(algo)
        self.horizon_steps = int(algo.algo_config.horizon.prediction_horizon)
        self.cond_steps = int(algo.algo_config.horizon.observation_horizon)
        self.action_dim = int(algo.ac_dim)
        self.obs_key = "flat"
        obs_shape = tuple(algo.obs_shapes[self.obs_key])
        self.obs_dim = int(np.prod(obs_shape))

        if self.horizon_steps <= 0 or self.cond_steps <= 0:
            raise ValueError(
                f"invalid diffusion horizons: prediction={self.horizon_steps} "
                f"observation={self.cond_steps}"
            )
        if "policy" not in self.live_policy:
            raise RuntimeError("active robomimic policy is missing the 'policy' module")
        required = {"obs_encoder", "noise_pred_net"}
        missing = required.difference(self.live_policy["policy"].keys())
        if missing:
            raise RuntimeError(f"active robomimic policy is missing modules: {sorted(missing)}")

    def parameters(self):
        return self.live_policy["policy"]["noise_pred_net"].parameters()

    def state_dict(self):
        return self.live_policy.state_dict()

    def load_state_dict(self, state, strict: bool = True):
        return self.live_policy.load_state_dict(state, strict=strict)

    def set_train(self) -> None:
        # Match the formal one-stage implementation: the observation encoder
        # remains fixed and only the diffusion noise predictor is adapted.
        self.live_policy.eval()
        for parameter in self.live_policy.parameters():
            parameter.requires_grad = False
        noise_pred_net = self.live_policy["policy"]["noise_pred_net"]
        noise_pred_net.train()
        for parameter in noise_pred_net.parameters():
            parameter.requires_grad = True

    def set_eval(self, *, keep_trainable: bool = True) -> None:
        self.live_policy.eval()
        if not keep_trainable:
            for parameter in self.live_policy.parameters():
                parameter.requires_grad = False

    def frozen_teacher_copy(self) -> nn.Module:
        teacher = copy.deepcopy(self.live_policy["policy"]["noise_pred_net"]).to(self.device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        return teacher

    def _validate_batch(
        self,
        actions: torch.Tensor,
        conditions,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if "state" not in conditions:
            raise ValueError("Adroit online BC conditions require key 'state'")
        clean_actions = actions.to(self.device, dtype=torch.float32)
        states = conditions["state"]
        if not torch.is_tensor(states):
            states = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        else:
            states = states.to(self.device, dtype=torch.float32)
        if clean_actions.ndim != 3 or tuple(clean_actions.shape[1:]) != (
            self.horizon_steps,
            self.action_dim,
        ):
            raise ValueError(
                f"Adroit BC action shape mismatch: got {tuple(clean_actions.shape)}, "
                f"expected [B,{self.horizon_steps},{self.action_dim}]"
            )
        if states.ndim != 3 or tuple(states.shape[1:]) != (
            self.cond_steps,
            self.obs_dim,
        ):
            raise ValueError(
                f"Adroit BC state shape mismatch: got {tuple(states.shape)}, "
                f"expected [B,{self.cond_steps},{self.obs_dim}]"
            )
        if int(states.shape[0]) != int(clean_actions.shape[0]):
            raise ValueError(
                f"Adroit BC batch mismatch: states={int(states.shape[0])} "
                f"actions={int(clean_actions.shape[0])}"
            )
        if not torch.isfinite(clean_actions).all() or not torch.isfinite(states).all():
            raise ValueError("Adroit BC batch contains non-finite values")
        return clean_actions, states

    def _encode_condition(self, policy_module: nn.Module, states: torch.Tensor) -> torch.Tensor:
        inputs = {
            "obs": {self.obs_key: states},
            "goal": None,
        }
        features = TensorUtils.time_distributed(
            inputs,
            policy_module["policy"]["obs_encoder"],
            inputs_as_kwargs=True,
        )
        if features.ndim != 3:
            raise RuntimeError(f"unexpected Adroit DP encoded obs shape: {tuple(features.shape)}")
        return features.flatten(start_dim=1)

    def diffusion_bc_loss_multi_teacher_terms(
        self,
        actions: torch.Tensor,
        conditions,
        *,
        teacher_noise_pred_nets: Sequence[nn.Module],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Use native robomimic forward diffusion for student and teachers."""
        clean_actions, states = self._validate_batch(actions, conditions)
        batch_size = int(clean_actions.shape[0])
        noise_scheduler = self.algo.noise_scheduler
        # Keep native robomimic's RNG order: action noise is sampled before
        # the per-example diffusion timestep.
        noise = torch.randn_like(clean_actions)
        timesteps = torch.randint(
            0,
            int(noise_scheduler.config.num_train_timesteps),
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )
        noisy_actions = noise_scheduler.add_noise(clean_actions, noise, timesteps)

        # Encoder parameters are frozen, so teachers and student receive the
        # exact same condition just as in the formal DSRL one-stage path.
        student_cond = self._encode_condition(self.live_policy, states)
        student_noise = self.live_policy["policy"]["noise_pred_net"](
            noisy_actions,
            timesteps,
            global_cond=student_cond,
        )
        bc_loss = F.mse_loss(student_noise, noise)

        teacher_losses = []
        for teacher_noise_pred_net in teacher_noise_pred_nets:
            with torch.no_grad():
                teacher_noise = teacher_noise_pred_net(
                    noisy_actions,
                    timesteps,
                    global_cond=student_cond,
                )
            teacher_losses.append(F.mse_loss(student_noise, teacher_noise.detach()))
        return bc_loss, teacher_losses

    def diffusion_bc_loss(self, actions: torch.Tensor, conditions) -> torch.Tensor:
        loss, _ = self.diffusion_bc_loss_multi_teacher_terms(
            actions,
            conditions,
            teacher_noise_pred_nets=[],
        )
        return loss
