"""Native robomimic Adroit diffusion policy adapter for DP-native OGPO."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

os.environ.setdefault("ROBOMIMIC_ENABLE_LANG", "0")
WUHAO_DIR = os.path.dirname(os.path.abspath(__file__))

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
from baselines.ogpo.adroit.policy_adapter import active_inference_policy_module
from baselines.ogpo.adroit.adroit_task_specs import AdroitTaskContract
from baselines.ogpo.adroit.dp_core import DPDiffusionChainSampler, DPDiffusionSamplerConfig


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_stats(checkpoint: Dict) -> Dict[str, np.ndarray]:
    result = {}
    for source, key, prefix in (
        ("obs_normalization_stats", "flat", "flat"),
        ("action_normalization_stats", "actions", "action"),
    ):
        stats = checkpoint[source][key]
        offset = np.asarray(stats["offset"], dtype=np.float32).reshape(-1)
        scale = np.asarray(stats["scale"], dtype=np.float32).reshape(-1)
        if (
            offset.shape != scale.shape
            or not np.isfinite(offset).all()
            or not np.isfinite(scale).all()
        ):
            raise ValueError(f"invalid checkpoint normalization: {source}")
        result[f"{prefix}_mean"] = offset
        result[f"{prefix}_std"] = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
    return result


class AdroitStatePreprocessor:
    mode = "dp_raw"

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1)
        self.state_dim = int(self.mean.size)

    def transform(self, raw_state) -> np.ndarray:
        state = np.asarray(raw_state, dtype=np.float32)
        if state.shape[-1] != self.state_dim or not np.isfinite(state).all():
            raise ValueError(f"invalid raw Adroit state shape={state.shape}")
        return ((state - self.mean) / self.std).astype(np.float32)


class AdroitActionScaler:
    def __init__(self, offset: np.ndarray, scale: np.ndarray):
        self.offset = np.asarray(offset, dtype=np.float32).reshape(-1)
        self.scale = np.asarray(scale, dtype=np.float32).reshape(-1)
        if self.offset.shape != self.scale.shape:
            raise ValueError("Adroit action offset/scale shape mismatch")
        self.action_dim = int(self.offset.size)

    @staticmethod
    def _like(array: np.ndarray, value):
        if torch.is_tensor(value):
            return torch.as_tensor(array, device=value.device, dtype=value.dtype)
        return array

    def _validate(self, value, label: str):
        if int(value.shape[-1]) != self.action_dim:
            raise ValueError(
                f"{label} action dim mismatch: got {value.shape[-1]}, expected {self.action_dim}"
            )

    def dp_to_raw(self, action_dp):
        self._validate(action_dp, "dp_to_raw")
        return action_dp * self._like(self.scale, action_dp) + self._like(self.offset, action_dp)

    def raw_to_dp(self, action_raw):
        self._validate(action_raw, "raw_to_dp")
        return (action_raw - self._like(self.offset, action_raw)) / self._like(
            self.scale, action_raw
        )


class AdroitDiffusionActor(nn.Module):
    """Expose the active EMA U-Net through OGPO's ``network(x,t,cond)`` API."""

    def __init__(
        self, obs_encoder: nn.Module, noise_pred_net: nn.Module, alphas_cumprod: torch.Tensor
    ):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.noise_pred_net = noise_pred_net
        self.register_buffer(
            "alphas_cumprod",
            torch.as_tensor(alphas_cumprod, dtype=torch.float32).detach().clone(),
            persistent=True,
        )
        for parameter in self.obs_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.noise_pred_net.parameters():
            parameter.requires_grad_(True)
        self.train(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.obs_encoder.eval()
        self.noise_pred_net.train(mode)
        return self

    def _encode_condition(self, states: torch.Tensor) -> torch.Tensor:
        inputs = {"obs": {"flat": states}, "goal": None}
        features = TensorUtils.time_distributed(
            inputs,
            self.obs_encoder,
            inputs_as_kwargs=True,
        )
        if features.ndim != 3:
            raise RuntimeError(f"unexpected Adroit condition shape: {tuple(features.shape)}")
        return features.flatten(start_dim=1)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, cond) -> torch.Tensor:
        if "state" not in cond:
            raise ValueError("Adroit actor condition requires 'state'")
        states = cond["state"].to(device=sample.device, dtype=sample.dtype)
        global_cond = self._encode_condition(states)
        return self.noise_pred_net(sample=sample, timestep=timestep, global_cond=global_cond)

    def diffusion_bc_loss(self, actions: torch.Tensor, cond) -> torch.Tensor:
        """Match native robomimic DP epsilon loss, including RNG draw order."""
        clean_actions = actions.to(dtype=torch.float32)
        noise = torch.randn_like(clean_actions)
        timesteps = torch.randint(
            0,
            int(self.alphas_cumprod.numel()),
            (int(clean_actions.shape[0]),),
            device=clean_actions.device,
            dtype=torch.long,
        )
        alpha = self.alphas_cumprod.to(clean_actions).gather(0, timesteps).reshape(-1, 1, 1)
        noisy_actions = torch.sqrt(alpha) * clean_actions + torch.sqrt(1.0 - alpha) * noise
        predicted_noise = self(noisy_actions, timesteps, cond=cond)
        return F.mse_loss(predicted_noise, noise)


class RobomimicDDIMChainSampler(DPDiffusionChainSampler):
    """DDIM mean from diffusers, wrapped in the existing OGPO SDE process."""

    def __init__(
        self,
        horizon_steps: int,
        action_dim: int,
        alphas_cumprod: torch.Tensor,
        inference_timesteps: torch.Tensor,
        final_alpha_cumprod: torch.Tensor,
    ):
        super().__init__(horizon_steps, action_dim)
        self._native_alphas = (
            torch.as_tensor(alphas_cumprod, dtype=torch.float32).detach().cpu().clone()
        )
        self._native_timesteps = (
            torch.as_tensor(inference_timesteps, dtype=torch.long).detach().cpu().clone()
        )
        self._native_final_alpha = (
            torch.as_tensor(final_alpha_cumprod, dtype=torch.float32).detach().cpu().clone()
        )

    def _build_schedule(self, denoising_steps, use_ddim, ddim_steps, device, dtype):
        if not bool(use_ddim):
            raise ValueError("robomimic Adroit OGPO supports the checkpoint's DDIM path only")
        if int(denoising_steps) != int(self._native_alphas.numel()):
            raise ValueError(
                f"Adroit training timestep mismatch: config={denoising_steps} "
                f"checkpoint={self._native_alphas.numel()}"
            )
        if int(ddim_steps) != int(self._native_timesteps.numel()):
            raise ValueError(
                f"Adroit DDIM inference-step mismatch: config={ddim_steps} "
                f"checkpoint={self._native_timesteps.numel()}"
            )
        key = (int(denoising_steps), True, int(ddim_steps), str(device), str(dtype))
        if key in self._cache:
            return self._cache[key]
        timesteps = self._native_timesteps.to(device=device)
        alphas = self._native_alphas.to(device=device, dtype=dtype)
        alpha = alphas.gather(0, timesteps)
        prev_timesteps = torch.cat(
            [timesteps[1:], torch.tensor([-1], device=device, dtype=torch.long)]
        )
        alpha_prev = torch.where(
            prev_timesteps >= 0,
            alphas.gather(0, torch.clamp(prev_timesteps, min=0)),
            self._native_final_alpha.to(device=device, dtype=dtype),
        )
        schedule = {
            "ddim_t": timesteps,
            "ddim_alphas": alpha,
            "ddim_alphas_prev": alpha_prev,
            "ddim_sqrt_one_minus_alphas": torch.sqrt(1.0 - alpha),
            "ddim_sigmas": torch.zeros_like(alpha),
        }
        self._cache[key] = schedule
        return schedule

    @staticmethod
    def _ogpo_mean_std(x, native_mean, step_index, num_steps, cfg):
        if not cfg.use_ogpo_sde:
            return native_mean, torch.zeros_like(native_mean)
        dt = 1.0 / float(max(int(num_steps), 1))
        t_value = float(step_index) / float(max(int(num_steps), 1))
        t_flow = torch.full_like(native_mean, t_value)
        is_last = int(step_index) == int(num_steps) - 1
        drift = (native_mean - x) / dt
        if cfg.error_correct_sde_to_ode:
            sigma = float(cfg.constant_noise_std)
            correction = 0.5 * sigma * sigma * (t_flow * drift - x)
            if cfg.use_tapered_noise:
                drift = drift + correction
            elif cfg.use_constant_noise and not is_last:
                drift = drift + correction / max(1.0 - t_value, 1e-6)
        mean = x + drift * dt
        if cfg.denoised_clip_value is not None:
            mean = torch.clamp(
                mean, -float(cfg.denoised_clip_value), float(cfg.denoised_clip_value)
            )
        if cfg.use_tapered_noise:
            std = float(cfg.constant_noise_std) * math.sqrt(max(1.0 - t_value, 0.0))
        elif cfg.use_constant_noise:
            std = 0.0 if is_last else float(cfg.constant_noise_std)
        else:
            raise ValueError("Adroit OGPO DDIM SDE requires tapered or constant noise")
        return mean, torch.full_like(mean, std)

    @staticmethod
    def _ogpo_mean_std_batched(x, native_mean, step_indices, num_steps, cfg):
        if not cfg.use_ogpo_sde:
            return native_mean, torch.zeros_like(native_mean)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        dt = 1.0 / float(max(int(num_steps), 1))
        t_flow = (step_indices.to(dtype=x.dtype) / float(max(int(num_steps), 1))).reshape(shape)
        is_last = (step_indices == int(num_steps) - 1).reshape(shape)
        drift = (native_mean - x) / dt
        if cfg.error_correct_sde_to_ode:
            sigma = float(cfg.constant_noise_std)
            correction = 0.5 * sigma * sigma * (t_flow * drift - x)
            if cfg.use_tapered_noise:
                drift = drift + correction
            elif cfg.use_constant_noise:
                drift = drift + torch.where(
                    is_last,
                    torch.zeros_like(correction),
                    correction / torch.clamp(1.0 - t_flow, min=1e-6),
                )
        mean = x + drift * dt
        if cfg.denoised_clip_value is not None:
            mean = torch.clamp(
                mean, -float(cfg.denoised_clip_value), float(cfg.denoised_clip_value)
            )
        if cfg.use_tapered_noise:
            std = float(cfg.constant_noise_std) * torch.sqrt(torch.clamp(1.0 - t_flow, min=0.0))
            std = std.expand_as(mean)
        elif cfg.use_constant_noise:
            std = torch.where(
                is_last,
                torch.zeros_like(mean),
                torch.full_like(mean, float(cfg.constant_noise_std)),
            )
        else:
            raise ValueError("Adroit OGPO DDIM SDE requires tapered or constant noise")
        return mean, std

    @staticmethod
    def _native_mean(noise, x, alpha, alpha_prev, cfg):
        beta = 1.0 - alpha
        pred_original = (x - torch.sqrt(beta) * noise) / torch.sqrt(alpha)
        if cfg.denoised_clip_value is not None:
            pred_original = torch.clamp(
                pred_original,
                -float(cfg.denoised_clip_value),
                float(cfg.denoised_clip_value),
            )
        direction = torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)) * noise
        return torch.sqrt(alpha_prev) * pred_original + direction

    def _step_params(self, network, x, t_val, step_index, cond, sched, cfg, num_steps):
        batch = int(x.shape[0])
        timesteps = torch.full((batch,), int(t_val), device=x.device, dtype=torch.long)
        noise = network(x, timesteps, cond=cond)
        index = torch.full((batch,), int(step_index), device=x.device, dtype=torch.long)
        alpha = self._extract(sched["ddim_alphas"], index, x.shape)
        alpha_prev = self._extract(sched["ddim_alphas_prev"], index, x.shape)
        native_mean = self._native_mean(noise, x, alpha, alpha_prev, cfg)
        return self._ogpo_mean_std(x, native_mean, step_index, num_steps, cfg)

    def _step_params_batched(self, network, x, t_values, step_indices, cond, sched, cfg, num_steps):
        t_values = t_values.to(device=x.device, dtype=torch.long).reshape(-1)
        step_indices = step_indices.to(device=x.device, dtype=torch.long).reshape(-1)
        if t_values.shape[0] != x.shape[0] or step_indices.shape[0] != x.shape[0]:
            raise ValueError("batched Adroit DDIM indices must match x batch size")
        noise = network(x, t_values, cond=cond)
        alpha = self._extract(sched["ddim_alphas"], step_indices, x.shape)
        alpha_prev = self._extract(sched["ddim_alphas_prev"], step_indices, x.shape)
        native_mean = self._native_mean(noise, x, alpha, alpha_prev, cfg)
        return self._ogpo_mean_std_batched(x, native_mean, step_indices, num_steps, cfg)


@dataclass
class AdroitPolicyBundle:
    actor: AdroitDiffusionActor
    sampler: RobomimicDDIMChainSampler
    rollout_policy: object
    env_meta: Dict
    state_preprocessor: AdroitStatePreprocessor
    action_scaler: AdroitActionScaler
    metadata: Dict


def load_adroit_ogpo_policy(
    checkpoint_path: str,
    contract: AdroitTaskContract,
    device: torch.device,
    infer_steps: int = 5,
) -> AdroitPolicyBundle:
    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = FileUtils.maybe_dict_from_checkpoint(ckpt_path=checkpoint_path)
    shape = checkpoint["shape_metadata"]
    state_dim = int(np.prod(shape["all_shapes"]["flat"]))
    action_dim = int(shape["ac_dim"])
    if (state_dim, action_dim) != (contract.state_dim, contract.action_dim):
        raise ValueError(
            f"checkpoint dimensions mismatch: got={(state_dim, action_dim)} "
            f"expected={(contract.state_dim, contract.action_dim)}"
        )
    env_meta = copy.deepcopy(checkpoint["env_metadata"])
    if str(env_meta.get("env_name")) != str(contract.env_id):
        raise ValueError(
            f"checkpoint env mismatch: {env_meta.get('env_name')} != {contract.env_id}"
        )
    env_meta.setdefault("env_kwargs", {})["max_episode_steps"] = int(contract.env_horizon)
    env_meta["env_kwargs"].pop("horizon", None)

    stats = _checkpoint_stats(checkpoint)
    preprocessor = AdroitStatePreprocessor(stats["flat_mean"], stats["flat_std"])
    action_scaler = AdroitActionScaler(stats["action_mean"], stats["action_std"])
    if not np.array_equal(action_scaler.offset, np.zeros(action_dim, dtype=np.float32)):
        raise ValueError("formal Adroit checkpoint requires zero action offset")
    if not np.array_equal(action_scaler.scale, np.ones(action_dim, dtype=np.float32)):
        raise ValueError("formal Adroit checkpoint requires unit action scale")

    rollout_policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_dict=checkpoint,
        device=device,
        verbose=False,
    )
    algo = rollout_policy.policy if hasattr(rollout_policy, "policy") else rollout_policy
    pred_horizon = int(algo.algo_config.horizon.prediction_horizon)
    obs_horizon = int(algo.algo_config.horizon.observation_horizon)
    if pred_horizon != int(contract.base_horizon_steps) or obs_horizon != 1:
        raise ValueError(
            f"checkpoint horizon mismatch: prediction={pred_horizon} observation={obs_horizon}"
        )
    with algo.algo_config.values_unlocked():
        algo.algo_config.horizon.action_horizon = pred_horizon
        algo.algo_config.ddpm.enabled = False
        algo.algo_config.ddim.enabled = True
        algo.algo_config.ddim.num_inference_timesteps = int(infer_steps)
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    scheduler_cfg = algo.algo_config.ddim
    algo.noise_scheduler = DDIMScheduler(
        num_train_timesteps=int(scheduler_cfg.num_train_timesteps),
        beta_schedule=str(scheduler_cfg.beta_schedule),
        clip_sample=bool(scheduler_cfg.clip_sample),
        set_alpha_to_one=bool(scheduler_cfg.set_alpha_to_one),
        steps_offset=int(scheduler_cfg.steps_offset),
        prediction_type=str(scheduler_cfg.prediction_type),
    )
    algo.noise_scheduler.set_timesteps(int(infer_steps), device=device)
    if hasattr(algo, "ac_horizon"):
        algo.ac_horizon = pred_horizon
    if hasattr(algo, "use_action_scheduler"):
        algo.use_action_scheduler = False
    if hasattr(algo, "use_ood_monitor"):
        algo.use_ood_monitor = False
    algo.set_eval()
    rollout_policy.start_episode()

    live_policy = active_inference_policy_module(algo)
    if getattr(algo, "ema", None) is None:
        raise ValueError("formal Adroit OGPO requires EMA checkpoint weights")
    actor = AdroitDiffusionActor(
        copy.deepcopy(live_policy["policy"]["obs_encoder"]),
        copy.deepcopy(live_policy["policy"]["noise_pred_net"]),
        algo.noise_scheduler.alphas_cumprod,
    )
    sampler = RobomimicDDIMChainSampler(
        pred_horizon,
        action_dim,
        algo.noise_scheduler.alphas_cumprod,
        algo.noise_scheduler.timesteps,
        algo.noise_scheduler.final_alpha_cumprod,
    )
    metadata = {
        "task": contract.env_name,
        "env_name": contract.env_id,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_hash,
        "weight_source": "ema",
        "state_dim": state_dim,
        "action_dim": action_dim,
        "observation_horizon": obs_horizon,
        "prediction_horizon": pred_horizon,
        "train_denoising_steps": int(algo.noise_scheduler.config.num_train_timesteps),
        "ddim_inference_steps": int(infer_steps),
        "ddim_timesteps": [int(x) for x in algo.noise_scheduler.timesteps.detach().cpu().tolist()],
        "beta_schedule": str(algo.noise_scheduler.config.beta_schedule),
        "clip_sample": bool(algo.noise_scheduler.config.clip_sample),
        "set_alpha_to_one": bool(algo.noise_scheduler.config.set_alpha_to_one),
        "prediction_type": str(algo.noise_scheduler.config.prediction_type),
        "action_normalization_identity": True,
    }
    return AdroitPolicyBundle(
        actor=actor,
        sampler=sampler,
        rollout_policy=rollout_policy,
        env_meta=env_meta,
        state_preprocessor=preprocessor,
        action_scaler=action_scaler,
        metadata=metadata,
    )


def native_deterministic_actions(
    bundle: AdroitPolicyBundle, normalized_states: torch.Tensor
) -> torch.Tensor:
    """Reference the untouched robomimic EMA rollout path for parity tests."""
    algo = bundle.rollout_policy.policy
    obs = {"flat": normalized_states.to(device=algo.device, dtype=torch.float32)}
    return algo._get_action_trajectory(obs_dict=obs)


__all__ = [
    "AdroitActionScaler",
    "AdroitDiffusionActor",
    "AdroitPolicyBundle",
    "AdroitStatePreprocessor",
    "RobomimicDDIMChainSampler",
    "load_adroit_ogpo_policy",
    "native_deterministic_actions",
    "sha256_file",
]
