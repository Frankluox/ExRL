"""
PyTorch DP-native OGPO core.

This module keeps the OGPO online-RL structure but uses the existing DSRL
diffusion-policy backbone as the trainable actor. It is intentionally separate
from the AAC/SAC cores so the original worktrees remain comparable.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-9


def _prod(xs: Sequence[int]) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return int(out)


@dataclass
class DPDiffusionSamplerConfig:
    denoising_steps: int = 5
    use_ddim: bool = False
    ddim_steps: int = 5
    ft_denoising_steps: int = 0
    use_ogpo_sde: bool = False
    use_constant_noise: bool = False
    use_tapered_noise: bool = True
    error_correct_sde_to_ode: bool = True
    constant_noise_std: float = 0.05
    min_noise_std: float = 0.01
    denoised_clip_value: Optional[float] = 1.0
    randn_clip_value: Optional[float] = 3.0
    final_action_clip_value: Optional[float] = None
    eps_clip_value: Optional[float] = None
    include_initial_logprob: bool = True
    normalize_denoising_horizon: bool = True
    normalize_act_space_dimension: bool = True


@dataclass
class DPReverseChain:
    xs: torch.Tensor
    t_values: torch.Tensor
    log_prob: Optional[torch.Tensor]
    actions: torch.Tensor


class DPDiffusionChainSampler:
    """
    DDPM/DDIM sampler copied from the local DSRL ckpt adapter and extended to
    retain reverse-process chains and recompute DDPM or OGPO-style DDIM SDE
    transition log-probs.
    """

    def __init__(self, horizon_steps: int, action_dim: int):
        self.horizon_steps = int(horizon_steps)
        self.action_dim = int(action_dim)
        self._cache: Dict[Tuple[int, bool, int, str, str], Dict[str, torch.Tensor]] = {}

    @staticmethod
    def _cosine_beta_schedule(
        timesteps: int, s: float = 0.008, dtype=torch.float32
    ) -> torch.Tensor:
        steps = int(timesteps) + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, a_min=0.0, a_max=0.999)
        return torch.tensor(betas, dtype=dtype)

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        b = int(t.shape[0])
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def _build_schedule(
        self,
        denoising_steps: int,
        use_ddim: bool,
        ddim_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        key = (int(denoising_steps), bool(use_ddim), int(ddim_steps), str(device), str(dtype))
        if key in self._cache:
            return self._cache[key]

        betas = self._cosine_beta_schedule(int(denoising_steps), dtype=dtype).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device, dtype=dtype), alphas_cumprod[:-1]]
        )
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)
        ddpm_logvar_clipped = torch.log(
            torch.clamp(betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod), min=1e-20)
        )
        ddpm_mu_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ddpm_mu_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

        sched = {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "sqrt_recip_alphas_cumprod": sqrt_recip_alphas_cumprod,
            "sqrt_recipm1_alphas_cumprod": sqrt_recipm1_alphas_cumprod,
            "ddpm_logvar_clipped": ddpm_logvar_clipped,
            "ddpm_mu_coef1": ddpm_mu_coef1,
            "ddpm_mu_coef2": ddpm_mu_coef2,
        }

        if use_ddim:
            ddim_steps = max(1, int(ddim_steps))
            ratio = max(1, int(denoising_steps // ddim_steps))
            ddim_t = torch.arange(0, ddim_steps, device=device, dtype=torch.long) * ratio
            ddim_t = torch.clamp(ddim_t, max=int(denoising_steps) - 1)
            ddim_alphas = alphas_cumprod[ddim_t].clone().to(dtype)
            ddim_alphas_prev = torch.cat(
                [
                    torch.tensor([1.0], device=device, dtype=dtype),
                    alphas_cumprod[ddim_t[:-1]].to(dtype),
                ]
            )
            ddim_sqrt_one_minus_alphas = (1.0 - ddim_alphas) ** 0.5
            ddim_sigmas = torch.zeros_like(ddim_alphas)
            sched.update(
                {
                    "ddim_t": torch.flip(ddim_t, [0]),
                    "ddim_alphas": torch.flip(ddim_alphas, [0]),
                    "ddim_alphas_prev": torch.flip(ddim_alphas_prev, [0]),
                    "ddim_sqrt_one_minus_alphas": torch.flip(ddim_sqrt_one_minus_alphas, [0]),
                    "ddim_sigmas": torch.flip(ddim_sigmas, [0]),
                }
            )

        self._cache[key] = sched
        return sched

    def _cond_from_obs(
        self, observations: torch.Tensor, cond_steps: int, obs_dim: int
    ) -> Dict[str, torch.Tensor]:
        if observations.ndim == 3:
            state = observations
        elif observations.ndim == 2:
            state = observations.reshape(observations.shape[0], int(cond_steps), int(obs_dim))
        else:
            raise ValueError(
                f"observations must be [B,D] or [B,To,D], got {tuple(observations.shape)}"
            )
        return {"state": state}

    def _step_params(
        self,
        network: nn.Module,
        x: torch.Tensor,
        t_val: int,
        step_index: int,
        cond: Dict[str, torch.Tensor],
        sched: Dict[str, torch.Tensor],
        cfg: DPDiffusionSamplerConfig,
        num_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = int(x.shape[0])
        t_b = torch.full((b,), int(t_val), device=x.device, dtype=torch.long)
        noise = network(x, t_b, cond=cond)

        if cfg.use_ddim:
            index_b = torch.full((b,), int(step_index), device=x.device, dtype=torch.long)
            alpha = self._extract(sched["ddim_alphas"], index_b, x.shape)
            alpha_prev = self._extract(sched["ddim_alphas_prev"], index_b, x.shape)
            sqrt_one_minus_alpha = self._extract(
                sched["ddim_sqrt_one_minus_alphas"], index_b, x.shape
            )
            x_recon = (x - sqrt_one_minus_alpha * noise) / torch.sqrt(alpha)
        else:
            x_recon = (
                self._extract(sched["sqrt_recip_alphas_cumprod"], t_b, x.shape) * x
                - self._extract(sched["sqrt_recipm1_alphas_cumprod"], t_b, x.shape) * noise
            )

        if cfg.denoised_clip_value is not None:
            clip_v = float(cfg.denoised_clip_value)
            x_recon = torch.clamp(x_recon, -clip_v, clip_v)
            if cfg.use_ddim:
                noise = (x - torch.sqrt(alpha) * x_recon) / torch.clamp(
                    sqrt_one_minus_alpha, min=1e-8
                )

        if cfg.use_ddim and cfg.eps_clip_value is not None:
            noise = torch.clamp(noise, -float(cfg.eps_clip_value), float(cfg.eps_clip_value))

        if cfg.use_ddim:
            sigma = self._extract(sched["ddim_sigmas"], index_b, x.shape)
            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma**2, min=0.0)) * noise
            mean = torch.sqrt(alpha_prev) * x_recon + dir_xt
            if cfg.use_ogpo_sde:
                dt = 1.0 / float(max(int(num_steps), 1))
                t_flow_value = float(step_index) / float(max(int(num_steps), 1))
                t_flow = torch.full_like(mean, t_flow_value)
                is_last_step = int(step_index) == int(num_steps) - 1
                drift = (mean - x) / dt
                if cfg.error_correct_sde_to_ode:
                    sigma_base = float(cfg.constant_noise_std)
                    if cfg.use_tapered_noise:
                        drift = drift + 0.5 * sigma_base * sigma_base * (t_flow * drift - x)
                    elif cfg.use_constant_noise and not is_last_step:
                        denom = max(1.0 - t_flow_value, 1e-6)
                        drift = drift + 0.5 * sigma_base * sigma_base * (t_flow * drift - x) / denom
                mean = x + drift * dt
                if cfg.denoised_clip_value is not None:
                    clip_v = float(cfg.denoised_clip_value)
                    mean = torch.clamp(mean, -clip_v, clip_v)
                if cfg.use_tapered_noise:
                    std = float(cfg.constant_noise_std) * math.sqrt(max(1.0 - t_flow_value, 0.0))
                    std = torch.full_like(mean, std)
                elif cfg.use_constant_noise:
                    if is_last_step:
                        std = torch.zeros_like(mean)
                    else:
                        std = torch.full_like(mean, float(cfg.constant_noise_std))
                else:
                    raise ValueError("DP-native OGPO DDIM SDE requires tapered or constant noise")
            else:
                std = sigma
        else:
            mean = (
                self._extract(sched["ddpm_mu_coef1"], t_b, x.shape) * x_recon
                + self._extract(sched["ddpm_mu_coef2"], t_b, x.shape) * x
            )
            logvar = self._extract(sched["ddpm_logvar_clipped"], t_b, x.shape)
            std = torch.exp(0.5 * logvar)
            if int(t_val) == 0:
                std = torch.zeros_like(std)
            else:
                std = torch.clamp(std, min=1e-3)
        return mean, std

    def _step_params_batched(
        self,
        network: nn.Module,
        x: torch.Tensor,
        t_values: torch.Tensor,
        step_indices: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        sched: Dict[str, torch.Tensor],
        cfg: DPDiffusionSamplerConfig,
        num_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized counterpart of ``_step_params`` for independent chain steps."""
        t_values = t_values.to(device=x.device, dtype=torch.long).reshape(-1)
        step_indices = step_indices.to(device=x.device, dtype=torch.long).reshape(-1)
        if t_values.shape[0] != x.shape[0] or step_indices.shape[0] != x.shape[0]:
            raise ValueError("batched diffusion indices must match x batch size")
        noise = network(x, t_values, cond=cond)

        if cfg.use_ddim:
            alpha = self._extract(sched["ddim_alphas"], step_indices, x.shape)
            alpha_prev = self._extract(sched["ddim_alphas_prev"], step_indices, x.shape)
            sqrt_one_minus_alpha = self._extract(
                sched["ddim_sqrt_one_minus_alphas"], step_indices, x.shape
            )
            x_recon = (x - sqrt_one_minus_alpha * noise) / torch.sqrt(alpha)
        else:
            x_recon = (
                self._extract(sched["sqrt_recip_alphas_cumprod"], t_values, x.shape) * x
                - self._extract(sched["sqrt_recipm1_alphas_cumprod"], t_values, x.shape) * noise
            )

        if cfg.denoised_clip_value is not None:
            clip_v = float(cfg.denoised_clip_value)
            x_recon = torch.clamp(x_recon, -clip_v, clip_v)
            if cfg.use_ddim:
                noise = (x - torch.sqrt(alpha) * x_recon) / torch.clamp(
                    sqrt_one_minus_alpha, min=1e-8
                )

        if cfg.use_ddim and cfg.eps_clip_value is not None:
            noise = torch.clamp(noise, -float(cfg.eps_clip_value), float(cfg.eps_clip_value))

        if cfg.use_ddim:
            sigma = self._extract(sched["ddim_sigmas"], step_indices, x.shape)
            direction = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma**2, min=0.0)) * noise
            mean = torch.sqrt(alpha_prev) * x_recon + direction
            if cfg.use_ogpo_sde:
                dt = 1.0 / float(max(int(num_steps), 1))
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                t_flow = (step_indices.to(dtype=x.dtype) / float(max(int(num_steps), 1))).reshape(
                    shape
                )
                is_last = (step_indices == int(num_steps) - 1).reshape(shape)
                drift = (mean - x) / dt
                if cfg.error_correct_sde_to_ode:
                    sigma_base = float(cfg.constant_noise_std)
                    correction = 0.5 * sigma_base * sigma_base * (t_flow * drift - x)
                    if cfg.use_tapered_noise:
                        drift = drift + correction
                    elif cfg.use_constant_noise:
                        denom = torch.clamp(1.0 - t_flow, min=1e-6)
                        drift = drift + torch.where(
                            is_last, torch.zeros_like(correction), correction / denom
                        )
                mean = x + drift * dt
                if cfg.denoised_clip_value is not None:
                    mean = torch.clamp(
                        mean, -float(cfg.denoised_clip_value), float(cfg.denoised_clip_value)
                    )
                if cfg.use_tapered_noise:
                    std = float(cfg.constant_noise_std) * torch.sqrt(
                        torch.clamp(1.0 - t_flow, min=0.0)
                    )
                    std = std.expand_as(mean)
                elif cfg.use_constant_noise:
                    std = torch.where(
                        is_last,
                        torch.zeros_like(mean),
                        torch.full_like(mean, float(cfg.constant_noise_std)),
                    )
                else:
                    raise ValueError("DP-native OGPO DDIM SDE requires tapered or constant noise")
            else:
                std = sigma
        else:
            mean = (
                self._extract(sched["ddpm_mu_coef1"], t_values, x.shape) * x_recon
                + self._extract(sched["ddpm_mu_coef2"], t_values, x.shape) * x
            )
            logvar = self._extract(sched["ddpm_logvar_clipped"], t_values, x.shape)
            std = torch.exp(0.5 * logvar)
            std = torch.where(
                (t_values == 0).reshape((x.shape[0],) + (1,) * (x.ndim - 1)),
                torch.zeros_like(std),
                torch.clamp(std, min=1e-3),
            )
        return mean, std

    @staticmethod
    def _normal_log_prob(
        value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        var = std.pow(2)
        return -0.5 * (
            ((value - mean).pow(2) / var) + 2.0 * torch.log(std) + math.log(2.0 * math.pi)
        )

    def _normalize_log_prob(
        self, log_prob: torch.Tensor, terms: int, cfg: DPDiffusionSamplerConfig
    ) -> torch.Tensor:
        denom = 1.0
        if cfg.normalize_denoising_horizon:
            denom *= float(max(int(terms), 1))
        if cfg.normalize_act_space_dimension:
            denom *= float(self.horizon_steps * self.action_dim)
        return log_prob / denom

    @staticmethod
    def _ft_window(num_steps: int, cfg: DPDiffusionSamplerConfig) -> Tuple[int, int]:
        ft = int(cfg.ft_denoising_steps)
        effective_ft = ft if 0 < ft < int(num_steps) else int(num_steps)
        return int(num_steps) - effective_ft, effective_ft

    def sample_chain(
        self,
        network: nn.Module,
        observations: torch.Tensor,
        cfg: DPDiffusionSamplerConfig,
        *,
        cond_steps: int,
        obs_dim: int,
        deterministic: bool = False,
        noise_action: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> DPReverseChain:
        cond = self._cond_from_obs(observations, cond_steps=cond_steps, obs_dim=obs_dim)
        state = cond["state"]
        b = int(state.shape[0])
        sched = self._build_schedule(
            int(cfg.denoising_steps),
            bool(cfg.use_ddim),
            int(cfg.ddim_steps),
            state.device,
            state.dtype,
        )

        if noise_action is None:
            x = torch.randn(
                (b, self.horizon_steps, self.action_dim),
                device=state.device,
                dtype=state.dtype,
                generator=generator,
            )
        else:
            x = noise_action.to(device=state.device, dtype=state.dtype)
            expected = (b, self.horizon_steps, self.action_dim)
            if tuple(x.shape) != expected:
                raise ValueError(
                    f"noise_action shape mismatch: expected {expected}, got {tuple(x.shape)}"
                )

        if cfg.use_ddim:
            t_all = sched["ddim_t"]
        else:
            t_all = torch.arange(
                int(cfg.denoising_steps) - 1, -1, -1, device=state.device, dtype=torch.long
            )
        num_steps = int(t_all.shape[0])
        ft_start, _ = self._ft_window(num_steps, cfg)

        xs: List[torch.Tensor] = [x]
        log_prob = torch.zeros((b,), device=state.device, dtype=state.dtype)
        n_log_terms = 0
        if cfg.include_initial_logprob and ft_start == 0 and not deterministic:
            init_lp = -0.5 * (x.pow(2) + math.log(2.0 * math.pi))
            log_prob = log_prob + init_lp.flatten(1).sum(dim=1)
            n_log_terms += 1

        for i, t in enumerate(t_all):
            t_val = int(t.item())
            mean, std = self._step_params(network, x, t_val, i, cond, sched, cfg, num_steps)
            if deterministic:
                step_noise = torch.zeros_like(x)
            else:
                step_noise = torch.randn(
                    x.shape, device=x.device, dtype=x.dtype, generator=generator
                )
                if cfg.randn_clip_value is not None:
                    step_noise = torch.clamp(
                        step_noise, -float(cfg.randn_clip_value), float(cfg.randn_clip_value)
                    )
            x_next = mean + std * step_noise
            if (not deterministic) and i >= ft_start and bool((std > 0).any().item()):
                lp = self._normal_log_prob(x_next, mean, std).flatten(1).sum(dim=1)
                log_prob = log_prob + lp
                n_log_terms += 1
            xs.append(x_next)
            x = x_next

        if cfg.final_action_clip_value is not None:
            x = torch.clamp(
                x, -float(cfg.final_action_clip_value), float(cfg.final_action_clip_value)
            )
            xs[-1] = x

        lp_out = None if deterministic else self._normalize_log_prob(log_prob, n_log_terms, cfg)
        return DPReverseChain(
            xs=torch.stack(xs, dim=1),
            t_values=t_all.detach().clone(),
            log_prob=lp_out,
            actions=x,
        )

    def log_prob_chain(
        self,
        network: nn.Module,
        observations: torch.Tensor,
        chain: DPReverseChain,
        cfg: DPDiffusionSamplerConfig,
        *,
        cond_steps: int,
        obs_dim: int,
        fuse_steps: bool = False,
    ) -> torch.Tensor:
        if cfg.use_ddim and not cfg.use_ogpo_sde:
            raise ValueError("deterministic DDIM has no OGPO path log-prob; enable use_ogpo_sde")
        cond = self._cond_from_obs(observations, cond_steps=cond_steps, obs_dim=obs_dim)
        sched = self._build_schedule(
            int(cfg.denoising_steps),
            bool(cfg.use_ddim),
            int(cfg.ddim_steps),
            chain.xs.device,
            chain.xs.dtype,
        )
        b = int(chain.xs.shape[0])
        log_prob = torch.zeros((b,), device=chain.xs.device, dtype=chain.xs.dtype)
        n_log_terms = 0
        num_steps = int(chain.t_values.shape[0])
        ft_start, _ = self._ft_window(num_steps, cfg)
        if cfg.include_initial_logprob and ft_start == 0:
            x0 = chain.xs[:, 0]
            log_prob = log_prob + (-0.5 * (x0.pow(2) + math.log(2.0 * math.pi))).flatten(1).sum(
                dim=1
            )
            n_log_terms += 1

        if fuse_steps and ft_start < num_steps:
            selected = torch.arange(ft_start, num_steps, device=chain.xs.device, dtype=torch.long)
            step_count = int(selected.shape[0])
            x = chain.xs[:, ft_start:num_steps].reshape(
                b * step_count, self.horizon_steps, self.action_dim
            )
            x_next = chain.xs[:, ft_start + 1 : num_steps + 1].reshape_as(x)
            t_values = chain.t_values[selected].unsqueeze(0).expand(b, -1).reshape(-1)
            step_indices = selected.unsqueeze(0).expand(b, -1).reshape(-1)
            fused_cond = {
                key: value.unsqueeze(1)
                .expand(value.shape[0], step_count, *value.shape[1:])
                .reshape(b * step_count, *value.shape[1:])
                for key, value in cond.items()
            }
            mean, std = self._step_params_batched(
                network, x, t_values, step_indices, fused_cond, sched, cfg, num_steps
            )
            std = std.expand_as(mean)
            mean = mean.reshape(b, step_count, self.horizon_steps, self.action_dim)
            std = std.reshape_as(mean)
            x_next = x_next.reshape_as(mean)
            valid_steps = std[0].flatten(1).amax(dim=1) > 0
            if bool(valid_steps.any().item()):
                selected_log_prob = self._normal_log_prob(
                    x_next[:, valid_steps], mean[:, valid_steps], std[:, valid_steps]
                )
                log_prob = log_prob + selected_log_prob.flatten(2).sum(dim=2).sum(dim=1)
                n_log_terms += int(valid_steps.sum().item())
        else:
            for i, t in enumerate(chain.t_values):
                t_val = int(t.item())
                x = chain.xs[:, i]
                x_next = chain.xs[:, i + 1]
                mean, std = self._step_params(network, x, t_val, i, cond, sched, cfg, num_steps)
                if i >= ft_start and bool((std > 0).any().item()):
                    log_prob = log_prob + self._normal_log_prob(x_next, mean, std).flatten(1).sum(
                        dim=1
                    )
                    n_log_terms += 1
        return self._normalize_log_prob(log_prob, n_log_terms, cfg)


@dataclass
class OGPOSequenceBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    masks: torch.Tensor
    terminals: torch.Tensor
    valid: torch.Tensor
    next_observations: torch.Tensor
    is_success: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "OGPOSequenceBatch":
        return OGPOSequenceBatch(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            masks=self.masks.to(device),
            terminals=self.terminals.to(device),
            valid=self.valid.to(device),
            next_observations=self.next_observations.to(device),
            is_success=None if self.is_success is None else self.is_success.to(device),
        )


class _FenwickTree:
    """Dynamic prefix sums used only for score-weighted replay sampling."""

    def __init__(self, points: np.ndarray):
        points = np.asarray(points, dtype=np.float64).reshape(-1)
        self.size = int(points.shape[0])
        prefix = np.empty((self.size + 1,), dtype=np.float64)
        prefix[0] = 0.0
        np.cumsum(points, out=prefix[1:])
        indices = np.arange(1, self.size + 1, dtype=np.int64)
        lowbits = indices & -indices
        self.tree = np.zeros((self.size + 1,), dtype=np.float64)
        self.tree[1:] = prefix[indices] - prefix[indices - lowbits]

    def add(self, slot: int, delta: float) -> None:
        i = int(slot) + 1
        while i <= self.size:
            self.tree[i] += float(delta)
            i += i & -i

    def prefix_sum(self, end: int) -> float:
        total = 0.0
        i = int(end)
        while i > 0:
            total += float(self.tree[i])
            i -= i & -i
        return total

    def range_sum(self, start: int, end: int) -> float:
        return self.prefix_sum(end) - self.prefix_sum(start)

    def find_prefix(self, targets: np.ndarray) -> np.ndarray:
        """Return the first zero-based slot whose cumulative weight exceeds target."""
        remaining = np.asarray(targets, dtype=np.float64).copy()
        indices = np.zeros(remaining.shape, dtype=np.int64)
        bit = 1 << (max(self.size, 1).bit_length() - 1)
        while bit:
            candidate = indices + bit
            in_bounds = candidate <= self.size
            candidate_values = np.zeros_like(remaining)
            candidate_values[in_bounds] = self.tree[candidate[in_bounds]]
            advance = in_bounds & (candidate_values <= remaining)
            remaining[advance] -= candidate_values[advance]
            indices[advance] = candidate[advance]
            bit >>= 1
        return np.minimum(indices, max(self.size - 1, 0))


class _SequenceStartPool:
    """Chronological ring of physical replay indices supporting O(batch) draws."""

    def __init__(self, capacity: int, starts: np.ndarray, scores: np.ndarray):
        self.capacity = int(capacity)
        index_dtype = np.int32 if self.capacity <= np.iinfo(np.int32).max else np.int64
        self.items = np.empty((self.capacity,), dtype=index_dtype)
        self.head = 0
        self.size = int(len(starts))
        if self.size > 0:
            self.items[: self.size] = np.asarray(starts, dtype=index_dtype)

        initial_weights = np.asarray(scores, dtype=np.float64).reshape(-1)
        finite = np.isfinite(initial_weights)
        self._finite_weight_total = float(initial_weights[finite].sum(dtype=np.float64))
        self._nonfinite_weight_count = int((~finite).sum())
        self._negative_weight_count = int((finite & (initial_weights < 0.0)).sum())
        self._fenwick: Optional[_FenwickTree] = None

    def __len__(self) -> int:
        return int(self.size)

    def _active_segments(self) -> List[Tuple[int, int]]:
        if self.size <= 0:
            return []
        first_end = min(self.capacity, self.head + self.size)
        segments = [(self.head, first_end)]
        remaining = self.size - (first_end - self.head)
        if remaining > 0:
            segments.append((0, remaining))
        return segments

    def _update_weight_stats(self, weight: float, direction: float) -> None:
        value = float(weight)
        if np.isfinite(value):
            self._finite_weight_total += float(direction) * value
            if value < 0.0:
                self._negative_weight_count += int(direction)
        else:
            self._nonfinite_weight_count += int(direction)

    def _build_fenwick(self, score_values: np.ndarray) -> None:
        points = np.zeros((self.capacity,), dtype=np.float64)
        for start, end in self._active_segments():
            points[start:end] = np.asarray(score_values[self.items[start:end]], dtype=np.float64)
        self._fenwick = _FenwickTree(points)

    def append(self, physical_start: int, weight: float) -> None:
        if self.size >= self.capacity:
            raise RuntimeError("sequence-start pool capacity exceeded")
        slot = (self.head + self.size) % self.capacity
        self.items[slot] = int(physical_start)
        self.size += 1
        self._update_weight_stats(weight, 1.0)
        if self._fenwick is not None:
            if np.isfinite(weight) and float(weight) >= 0.0:
                self._fenwick.add(slot, float(weight))
            else:
                self._fenwick = None

    def popleft_expected(self, physical_start: int, weight: float) -> None:
        if self.size <= 0 or int(self.items[self.head]) != int(physical_start):
            actual = None if self.size <= 0 else int(self.items[self.head])
            raise RuntimeError(
                f"sequence-start cache out of sync: expected oldest={int(physical_start)}, actual={actual}"
            )
        slot = int(self.head)
        if self._fenwick is not None:
            if np.isfinite(weight) and float(weight) >= 0.0:
                self._fenwick.add(slot, -float(weight))
            else:
                self._fenwick = None
        self._update_weight_stats(weight, -1.0)
        self.head = (self.head + 1) % self.capacity
        self.size -= 1

    def active_items(self) -> np.ndarray:
        """Materialize starts in chronological order for diagnostics and tests."""
        if self.size <= 0:
            return np.empty((0,), dtype=np.int64)
        pieces = [self.items[start:end] for start, end in self._active_segments()]
        return np.concatenate(pieces).astype(np.int64, copy=False)

    def sample(
        self,
        rng: np.random.Generator,
        batch_size: int,
        *,
        by_score: bool,
        score_values: np.ndarray,
    ) -> np.ndarray:
        if self.size <= 0:
            raise ValueError("cannot sample from an empty sequence-start pool")
        b = int(batch_size)
        use_weights = (
            bool(by_score)
            and self._nonfinite_weight_count == 0
            and np.isfinite(self._finite_weight_total)
            and self._finite_weight_total > 0.0
        )
        if use_weights and self._negative_weight_count > 0:
            raise ValueError("probabilities are not non-negative")

        if not use_weights:
            relative = rng.choice(self.size, size=b)
            slots = (self.head + relative) % self.capacity
            return self.items[slots].astype(np.int64, copy=False)

        if self._fenwick is None:
            self._build_fenwick(score_values)
        first_length = min(self.size, self.capacity - self.head)
        first_end = self.head + first_length
        first_total = self._fenwick.range_sum(self.head, first_end)
        second_length = self.size - first_length
        second_total = self._fenwick.range_sum(0, second_length) if second_length > 0 else 0.0
        total = first_total + second_total
        if not np.isfinite(total) or total <= 0.0:
            relative = rng.choice(self.size, size=b)
            slots = (self.head + relative) % self.capacity
            return self.items[slots].astype(np.int64, copy=False)

        draws = rng.random(b) * total
        slots = np.empty((b,), dtype=np.int64)
        first_mask = draws < first_total
        if np.any(first_mask):
            base = self._fenwick.prefix_sum(self.head)
            slots[first_mask] = self._fenwick.find_prefix(base + draws[first_mask])
        if np.any(~first_mask):
            slots[~first_mask] = self._fenwick.find_prefix(draws[~first_mask] - first_total)
        return self.items[slots].astype(np.int64, copy=False)


class _SequenceStartCache:
    def __init__(
        self,
        capacity: int,
        sequence_length: int,
        starts: np.ndarray,
        success_flags: np.ndarray,
        scores: np.ndarray,
    ):
        self.sequence_length = int(sequence_length)
        flags = np.asarray(success_flags, dtype=bool)
        self.all_starts = _SequenceStartPool(capacity, starts, scores)
        self.success_starts = _SequenceStartPool(capacity, starts[flags], scores[flags])
        self.failure_starts = _SequenceStartPool(capacity, starts[~flags], scores[~flags])

    def pool(self, success_only: Optional[bool]) -> _SequenceStartPool:
        if success_only is None:
            return self.all_starts
        return self.success_starts if success_only else self.failure_starts

    def append(self, physical_start: int, is_success: bool, score: float) -> None:
        self.all_starts.append(physical_start, score)
        (self.success_starts if is_success else self.failure_starts).append(physical_start, score)

    def popleft(self, physical_start: int, is_success: bool, score: float) -> None:
        self.all_starts.popleft_expected(physical_start, score)
        (self.success_starts if is_success else self.failure_starts).popleft_expected(
            physical_start, score
        )


class OGPOSequenceReplayBuffer:
    """Single-step replay with OGPO_public-compatible length-T sampling."""

    def __init__(
        self, obs_dim: int, action_dim: int, capacity: int, device: torch.device, seed: int = 0
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(int(seed))
        self.pointer = 0
        self.size = 0
        self.trajectory_ids = np.full((self.capacity,), -1, dtype=np.int64)
        self._next_trajectory_id = 0
        self._sequence_caches: Dict[int, _SequenceStartCache] = {}

        self.observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.terminals = np.zeros((self.capacity,), dtype=np.float32)
        self.masks = np.ones((self.capacity,), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.is_success = np.zeros((self.capacity,), dtype=np.float32)
        self.score = np.zeros((self.capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return int(self.size)

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        # Sequence pools are derived data. Omitting them keeps checkpoints small
        # and avoids serializing stale physical indices across code versions.
        state.pop("_sequence_caches", None)
        return state

    def __setstate__(self, state: Dict) -> None:
        self.__dict__.update(state)
        self._sequence_caches = {}

    def _ensure_sequence_cache_state(self) -> None:
        if not hasattr(self, "_sequence_caches") or not isinstance(self._sequence_caches, dict):
            self._sequence_caches = {}

    def _chronological_indices(self) -> np.ndarray:
        """Physical buffer indices ordered from oldest to newest."""
        if self.size <= 0:
            return np.empty((0,), dtype=np.int64)
        if self.size < self.capacity:
            return np.arange(self.size, dtype=np.int64)
        return np.concatenate(
            [
                np.arange(self.pointer, self.capacity, dtype=np.int64),
                np.arange(0, self.pointer, dtype=np.int64),
            ]
        )

    def _ensure_trajectory_metadata(self) -> None:
        """Reconstruct trajectory ids when loading checkpoints made before this field existed."""
        self._ensure_sequence_cache_state()
        trajectory_ids = getattr(self, "trajectory_ids", None)
        if trajectory_ids is not None and np.asarray(trajectory_ids).shape == (self.capacity,):
            if not hasattr(self, "_next_trajectory_id"):
                logical = self._chronological_indices()
                if len(logical) == 0:
                    self._next_trajectory_id = 0
                else:
                    last = int(logical[-1])
                    self._next_trajectory_id = int(np.max(self.trajectory_ids[logical]))
                    if self.terminals[last] > 0.0:
                        self._next_trajectory_id += 1
            return

        self._sequence_caches.clear()
        self.trajectory_ids = np.full((self.capacity,), -1, dtype=np.int64)
        trajectory_id = 0
        for physical_idx in self._chronological_indices():
            i = int(physical_idx)
            self.trajectory_ids[i] = trajectory_id
            if self.terminals[i] > 0.0:
                trajectory_id += 1
        self._next_trajectory_id = trajectory_id

    def _chronological_position(self, physical_idx: int) -> Optional[int]:
        i = int(physical_idx)
        if i < 0 or i >= self.capacity:
            return None
        if self.size < self.capacity:
            return i if i < self.size else None
        return int((i - self.pointer) % self.capacity)

    def _is_valid_sequence_start(self, physical_start: int, sequence_length: int) -> bool:
        t = int(sequence_length)
        position = self._chronological_position(physical_start)
        if position is None or position > self.size - t:
            return False
        end = (int(physical_start) + t - 1) % self.capacity
        return int(self.trajectory_ids[int(physical_start)]) == int(self.trajectory_ids[end])

    def _build_sequence_cache(self, sequence_length: int) -> _SequenceStartCache:
        t = int(sequence_length)
        logical = self._chronological_indices()
        candidate_count = self.size - t + 1
        first = logical[:candidate_count]
        last = logical[t - 1 : t - 1 + candidate_count]
        same_trajectory = self.trajectory_ids[first] == self.trajectory_ids[last]
        starts = np.asarray(first[same_trajectory], dtype=np.int64)
        success_flags = self.is_success[starts] > 0.0
        scores = np.asarray(self.score[starts], dtype=np.float64)
        cache = _SequenceStartCache(self.capacity, t, starts, success_flags, scores)
        self._sequence_caches[t] = cache
        return cache

    def _get_sequence_cache(self, sequence_length: int) -> _SequenceStartCache:
        self._ensure_sequence_cache_state()
        t = int(sequence_length)
        cache = self._sequence_caches.get(t)
        return self._build_sequence_cache(t) if cache is None else cache

    def _remove_overwritten_starts(self, physical_idx: int) -> None:
        for t, cache in self._sequence_caches.items():
            if self._is_valid_sequence_start(physical_idx, t):
                cache.popleft(
                    physical_idx,
                    bool(self.is_success[physical_idx] > 0.0),
                    float(self.score[physical_idx]),
                )

    def _append_newest_starts(self) -> None:
        for t, cache in self._sequence_caches.items():
            if self.size < t:
                continue
            start_position = self.size - t
            if self.size < self.capacity:
                physical_start = int(start_position)
            else:
                physical_start = int((self.pointer + start_position) % self.capacity)
            if self._is_valid_sequence_start(physical_start, t):
                cache.append(
                    physical_start,
                    bool(self.is_success[physical_start] > 0.0),
                    float(self.score[physical_start]),
                )

    def add_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminal: float,
        mask: float,
        next_observation: np.ndarray,
        *,
        is_success: float = 0.0,
        score: Optional[float] = None,
    ) -> None:
        self._ensure_trajectory_metadata()
        i = int(self.pointer)
        if self.size == self.capacity:
            self._remove_overwritten_starts(i)
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        next_obs = np.asarray(next_observation, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.obs_dim or next_obs.shape[0] != self.obs_dim:
            raise ValueError(
                f"obs dim mismatch: got {obs.shape[0]} / {next_obs.shape[0]}, expected {self.obs_dim}"
            )
        if act.shape[0] != self.action_dim:
            raise ValueError(f"action dim mismatch: got {act.shape[0]}, expected {self.action_dim}")

        self.observations[i] = obs
        self.actions[i] = act
        self.rewards[i] = float(reward)
        self.terminals[i] = float(terminal)
        self.masks[i] = float(mask)
        self.next_observations[i] = next_obs
        self.is_success[i] = float(is_success)
        self.score[i] = float((1.0 if float(is_success) > 0.0 else EPS) if score is None else score)
        self.trajectory_ids[i] = int(self._next_trajectory_id)
        if float(terminal) > 0.0:
            self._next_trajectory_id += 1

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._append_newest_starts()

    def sample_sequence(
        self,
        batch_size: int,
        sequence_length: int,
        discount: float,
        *,
        success_only: Optional[bool] = None,
        by_score: bool = False,
        idxs: Optional[np.ndarray] = None,
    ) -> OGPOSequenceBatch:
        self._ensure_trajectory_metadata()
        b = int(batch_size)
        t = int(sequence_length)
        if t <= 0:
            raise ValueError(f"sequence_length must be positive, got {t}")
        max_start = int(self.size) - t
        if max_start < 0:
            raise ValueError(f"need at least {t} transitions, have {self.size}")

        if idxs is None:
            cache = self._get_sequence_cache(t)
            pool = cache.pool(success_only)
            if len(pool) == 0:
                raise ValueError(
                    "no complete single-trajectory sequences found matching sequence filter"
                )
            idxs = pool.sample(self.rng, b, by_score=by_score, score_values=self.score)
        else:
            idxs = np.asarray(idxs, dtype=np.int64).reshape(-1)
            b = int(idxs.shape[0])
            for physical_idx in idxs:
                position = self._chronological_position(int(physical_idx))
                if position is None or position > self.size - t:
                    raise ValueError(
                        f"sequence start {int(physical_idx)} does not have {t} newer transitions in replay"
                    )
                if not self._is_valid_sequence_start(int(physical_idx), t):
                    raise ValueError(
                        f"sequence start {int(physical_idx)} crosses a trajectory boundary for length {t}"
                    )
                is_success_start = bool(self.is_success[int(physical_idx)] > 0.0)
                if success_only is not None and is_success_start != bool(success_only):
                    raise ValueError(
                        f"sequence start {int(physical_idx)} does not match success_only={success_only}"
                    )
        idxs = np.asarray(idxs, dtype=np.int64).reshape(-1)
        offs = np.arange(t, dtype=np.int64)[None, :]
        seq_idxs = (idxs[:, None] + offs) % self.capacity
        actions_seq = self.actions[seq_idxs]
        rewards_seq = self.rewards[seq_idxs]
        masks_seq = self.masks[seq_idxs]
        terminals_seq = self.terminals[seq_idxs]

        masks_prefix = np.minimum.accumulate(masks_seq, axis=1)
        terminals_prefix = np.maximum.accumulate(terminals_seq, axis=1)
        valid = np.ones_like(masks_seq, dtype=np.float32)
        valid[:, 1:] = 1.0 - terminals_prefix[:, :-1]
        disc_pows = (float(discount) ** np.arange(t)).astype(rewards_seq.dtype, copy=False)
        rewards_prefix = np.cumsum(rewards_seq * disc_pows[None, :], axis=1)

        return OGPOSequenceBatch(
            observations=torch.as_tensor(
                self.observations[idxs], device=self.device, dtype=torch.float32
            ),
            actions=torch.as_tensor(actions_seq, device=self.device, dtype=torch.float32),
            rewards=torch.as_tensor(rewards_prefix, device=self.device, dtype=torch.float32),
            masks=torch.as_tensor(masks_prefix, device=self.device, dtype=torch.float32),
            terminals=torch.as_tensor(terminals_prefix, device=self.device, dtype=torch.float32),
            valid=torch.as_tensor(valid, device=self.device, dtype=torch.float32),
            next_observations=torch.as_tensor(
                self.next_observations[seq_idxs[:, -1]], device=self.device, dtype=torch.float32
            ),
            is_success=torch.as_tensor(
                self.is_success[seq_idxs], device=self.device, dtype=torch.float32
            ),
        )


class _MLP(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, hidden_dims: Sequence[int], layer_norm: bool = True
    ):
        super().__init__()
        dims = [int(in_dim)] + [int(x) for x in hidden_dims]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.Mish())
        layers.append(nn.Linear(dims[-1], int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnsembleQ(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        horizon_steps: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (512, 512, 512, 512),
        num_qs: int = 10,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.horizon_steps = int(horizon_steps)
        self.action_dim = int(action_dim)
        self.num_qs = int(num_qs)
        in_dim = self.obs_dim + self.horizon_steps * self.action_dim
        self.qs = nn.ModuleList(
            [_MLP(in_dim, 1, hidden_dims, layer_norm=layer_norm) for _ in range(self.num_qs)]
        )

    def forward(self, observations: torch.Tensor, action_chunks: torch.Tensor) -> torch.Tensor:
        if action_chunks.ndim == 3:
            action_flat = action_chunks.reshape(action_chunks.shape[0], -1)
        else:
            action_flat = action_chunks
        x = torch.cat([observations.reshape(observations.shape[0], -1), action_flat], dim=-1)
        return torch.cat([q(x) for q in self.qs], dim=-1)


class VectorizedEnsembleQ(nn.Module):
    """Equivalent independent Q heads using head-batched matrix multiplies."""

    def __init__(self, source: EnsembleQ):
        super().__init__()
        self.obs_dim = int(source.obs_dim)
        self.horizon_steps = int(source.horizon_steps)
        self.action_dim = int(source.action_dim)
        self.num_qs = int(source.num_qs)
        linears = [[layer for layer in q.net if isinstance(layer, nn.Linear)] for q in source.qs]
        norms = [[layer for layer in q.net if isinstance(layer, nn.LayerNorm)] for q in source.qs]
        if any(len(head) != len(linears[0]) for head in linears):
            raise ValueError("all critic heads must have the same linear layout")
        if any(len(head) != len(norms[0]) for head in norms):
            raise ValueError("all critic heads must have the same normalization layout")
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.stack([head[layer].weight.detach() for head in linears], dim=0))
                for layer in range(len(linears[0]))
            ]
        )
        self.biases = nn.ParameterList(
            [
                nn.Parameter(torch.stack([head[layer].bias.detach() for head in linears], dim=0))
                for layer in range(len(linears[0]))
            ]
        )
        self.norm_weights = nn.ParameterList(
            [
                nn.Parameter(torch.stack([head[layer].weight.detach() for head in norms], dim=0))
                for layer in range(len(norms[0]))
            ]
        )
        self.norm_biases = nn.ParameterList(
            [
                nn.Parameter(torch.stack([head[layer].bias.detach() for head in norms], dim=0))
                for layer in range(len(norms[0]))
            ]
        )
        self.norm_eps = [float(norms[0][layer].eps) for layer in range(len(norms[0]))]

    @classmethod
    def from_dimensions(
        cls,
        obs_dim: int,
        horizon_steps: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        num_qs: int,
        layer_norm: bool,
    ) -> "VectorizedEnsembleQ":
        return cls(EnsembleQ(obs_dim, horizon_steps, action_dim, hidden_dims, num_qs, layer_norm))

    def forward(self, observations: torch.Tensor, action_chunks: torch.Tensor) -> torch.Tensor:
        if action_chunks.ndim == 3:
            action_flat = action_chunks.reshape(action_chunks.shape[0], -1)
        else:
            action_flat = action_chunks
        base = torch.cat([observations.reshape(observations.shape[0], -1), action_flat], dim=-1)
        x = base.unsqueeze(0).expand(self.num_qs, -1, -1)
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = torch.bmm(x, weight.transpose(1, 2)) + bias[:, None, :]
            if layer != len(self.weights) - 1:
                if len(self.norm_weights):
                    x = F.layer_norm(x, (x.shape[-1],), eps=self.norm_eps[layer])
                    x = (
                        x * self.norm_weights[layer][:, None, :]
                        + self.norm_biases[layer][:, None, :]
                    )
                x = F.mish(x)
        return x.squeeze(-1).transpose(0, 1)


@dataclass
class DPNativeOGPOConfig:
    obs_dim: int
    action_dim: int
    horizon_steps: int
    cond_steps: int = 1
    discount: float = 0.99
    tau: float = 0.05
    actor_tau: float = 0.005
    num_qs: int = 10
    q_agg: str = "mean"
    critic_hidden_dims: Tuple[int, ...] = (512, 512, 512, 512)
    layer_norm: bool = True
    ppo_lr: float = 4.5e-5
    actor_lr: Optional[float] = None
    critic_lr: Optional[float] = None
    bc_coeff: float = 1.0
    update_mode: str = "ogpo"
    group_num_samples: int = 32
    actor_microbatch_size: int = 0
    inference_microbatch_size: int = 0
    fuse_log_prob_steps: bool = False
    vectorized_critic: bool = False
    compile_models: bool = False
    compile_mode: str = "default"
    clip_epsilon: float = 0.01
    entropy_coeff: float = 0.0
    best_of_n: int = 8
    subsample_bon: bool = True
    bon_subsample_size: int = 2
    use_success_buffer_q: bool = True
    max_grad_norm_actor: float = 1.0
    max_grad_norm_critic: float = 1.0
    train_denoising_steps: int = 5
    eval_ddim_steps: int = 5
    ft_denoising_steps: int = 0
    use_ogpo_sde: bool = True
    use_constant_noise: bool = False
    use_tapered_noise: bool = True
    error_correct_sde_to_ode: bool = True
    constant_noise_std: float = 0.05
    min_noise_std: float = 0.01
    denoised_clip_value: Optional[float] = 1.0
    randn_clip_value: Optional[float] = 3.0
    final_action_clip_value: Optional[float] = 1.0
    normalize_denoising_horizon: bool = True
    normalize_act_space_dimension: bool = True

    @property
    def per_obs_dim(self) -> int:
        if int(self.obs_dim) % int(self.cond_steps) != 0:
            raise ValueError(
                f"obs_dim={self.obs_dim} must be divisible by cond_steps={self.cond_steps}"
            )
        return int(self.obs_dim) // int(self.cond_steps)

    def rollout_sampler_config(self) -> DPDiffusionSamplerConfig:
        return DPDiffusionSamplerConfig(
            denoising_steps=int(self.train_denoising_steps),
            use_ddim=True,
            ddim_steps=int(self.eval_ddim_steps),
            ft_denoising_steps=int(self.ft_denoising_steps),
            use_ogpo_sde=bool(self.use_ogpo_sde),
            use_constant_noise=bool(self.use_constant_noise),
            use_tapered_noise=bool(self.use_tapered_noise),
            error_correct_sde_to_ode=bool(self.error_correct_sde_to_ode),
            constant_noise_std=float(self.constant_noise_std),
            min_noise_std=float(self.min_noise_std),
            denoised_clip_value=self.denoised_clip_value,
            randn_clip_value=self.randn_clip_value,
            final_action_clip_value=self.final_action_clip_value,
            include_initial_logprob=True,
            normalize_denoising_horizon=bool(self.normalize_denoising_horizon),
            normalize_act_space_dimension=bool(self.normalize_act_space_dimension),
        )

    def eval_sampler_config(self) -> DPDiffusionSamplerConfig:
        return DPDiffusionSamplerConfig(
            denoising_steps=int(self.train_denoising_steps),
            use_ddim=True,
            ddim_steps=int(self.eval_ddim_steps),
            ft_denoising_steps=0,
            use_ogpo_sde=False,
            use_constant_noise=False,
            use_tapered_noise=False,
            error_correct_sde_to_ode=False,
            constant_noise_std=float(self.constant_noise_std),
            min_noise_std=float(self.min_noise_std),
            denoised_clip_value=self.denoised_clip_value,
            randn_clip_value=self.randn_clip_value,
            final_action_clip_value=self.final_action_clip_value,
            include_initial_logprob=False,
            normalize_denoising_horizon=bool(self.normalize_denoising_horizon),
            normalize_act_space_dimension=bool(self.normalize_act_space_dimension),
        )


def aggregate_q_values(q_values: torch.Tensor, method: str = "mean") -> torch.Tensor:
    if method == "min":
        return q_values.min(dim=-1).values
    if method == "mean":
        return q_values.mean(dim=-1)
    raise ValueError(f"unsupported q aggregation: {method}")


def compute_td_target(
    rewards: torch.Tensor,
    masks: torch.Tensor,
    next_q: torch.Tensor,
    discount: float,
    horizon_length: int = 1,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> torch.Tensor:
    if rewards.ndim > 1:
        rewards = rewards[..., -1]
    if masks.ndim > 1:
        masks = masks[..., -1]
    target = rewards + (float(discount) ** int(horizon_length)) * masks * next_q
    if clip_min is not None or clip_max is not None:
        target = torch.clamp(
            target,
            min=-float("inf") if clip_min is None else float(clip_min),
            max=float("inf") if clip_max is None else float(clip_max),
        )
    return target


def compute_td_loss(
    q_pred: torch.Tensor,
    target_q: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if valid_mask is None:
        valid_mask = torch.ones_like(target_q)
    return (torch.square(q_pred - target_q[:, None]) * valid_mask[:, None]).mean()


def ppo_clipped_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
    clip_min_epsilon_multiplier: float = 1.0,
    adv_clip_min: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    lower_bound = 1.0 - float(clip_epsilon) * float(clip_min_epsilon_multiplier)
    upper_bound = 1.0 + float(clip_epsilon) * float(clip_min_epsilon_multiplier)
    clipped_ratio = torch.clamp(ratio, lower_bound, upper_bound)
    if adv_clip_min is not None:
        advantages = torch.clamp(advantages, min=float(adv_clip_min))
    loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
    stats = {
        "ratio": ratio.mean(),
        "ratio_std": ratio.std(unbiased=False),
        "ratio_min": ratio.min(),
        "ratio_max": ratio.max(),
        "approx_kl": approx_kl,
        "ratio_clipped_upper": (ratio > upper_bound).to(ratio.dtype).mean(),
        "ratio_clipped_lower": (ratio < lower_bound).to(ratio.dtype).mean(),
    }
    stats["ratio_clipped_total"] = stats["ratio_clipped_upper"] + stats["ratio_clipped_lower"]
    return loss, stats


def select_best_of_n_actions(
    actions: torch.Tensor,
    q_values: torch.Tensor,
    *,
    subsample_bon: bool = True,
    subsample_size: int = 2,
    subsample_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Match OGPO_public BoN scoring.

    actions:  [B, N, ...]
    q_values: [M, B, N]
    """
    if q_values.ndim != 3:
        raise ValueError(f"q_values must be [M,B,N], got {tuple(q_values.shape)}")
    m, b, n = q_values.shape
    if actions.shape[0] != b or actions.shape[1] != n:
        raise ValueError(
            f"actions shape {tuple(actions.shape)} incompatible with q_values {tuple(q_values.shape)}"
        )
    if subsample_bon:
        k = min(int(subsample_size), int(m))
        if subsample_indices is None:
            subsample_indices = torch.randint(0, m, (k, b, n), device=q_values.device)
        else:
            subsample_indices = subsample_indices.to(device=q_values.device, dtype=torch.long)
            if tuple(subsample_indices.shape) != (k, b, n):
                raise ValueError(
                    f"subsample_indices must be {(k, b, n)}, got {tuple(subsample_indices.shape)}"
                )
        min_qs = torch.gather(q_values, 0, subsample_indices).min(dim=0).values
    else:
        min_qs = q_values.min(dim=0).values
    best_idx = torch.argmax(min_qs, dim=1)
    best = actions[torch.arange(b, device=actions.device), best_idx]
    return best, best_idx


def conservative_group_advantages(q_values: torch.Tensor) -> torch.Tensor:
    """
    q_values: [B, G, M]. Returns [B, G] using OGPO conservative per-head
    centered advantages and safe sign agreement.
    """
    centered = q_values - q_values.mean(dim=1, keepdim=True)
    min_head = centered.min(dim=-1).values
    max_head = centered.max(dim=-1).values
    all_pos = min_head > 0
    all_neg = max_head < 0
    return torch.where(
        all_pos, min_head, torch.where(all_neg, max_head, torch.zeros_like(min_head))
    )


class DPNativeOGPOAgent:
    def __init__(
        self, actor: nn.Module, config: DPNativeOGPOConfig, device: Optional[torch.device] = None
    ):
        if config.actor_microbatch_size < 0 or config.inference_microbatch_size < 0:
            raise ValueError("microbatch sizes must be non-negative")
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = actor.to(self.device)
        self.target_actor = copy.deepcopy(actor).to(self.device)
        critic_args = dict(
            obs_dim=config.obs_dim,
            horizon_steps=config.horizon_steps,
            action_dim=config.action_dim,
            hidden_dims=config.critic_hidden_dims,
            num_qs=config.num_qs,
            layer_norm=config.layer_norm,
        )
        if config.vectorized_critic:
            self.critic = VectorizedEnsembleQ.from_dimensions(**critic_args).to(self.device)
        else:
            self.critic = EnsembleQ(**critic_args).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        if config.compile_models:
            compile_kwargs = {"dynamic": True}
            if str(config.compile_mode) != "default":
                compile_kwargs["mode"] = str(config.compile_mode)
            self.actor = torch.compile(self.actor, **compile_kwargs)
            self.target_actor = torch.compile(self.target_actor, **compile_kwargs)
            self.critic = torch.compile(self.critic, **compile_kwargs)
            self.target_critic = torch.compile(self.target_critic, **compile_kwargs)
        self.sampler = DPDiffusionChainSampler(config.horizon_steps, config.action_dim)

        actor_lr = float(config.ppo_lr if config.actor_lr is None else config.actor_lr)
        critic_lr = float(config.ppo_lr if config.critic_lr is None else config.critic_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.update_step = 0

    def _soft_update(self, source: nn.Module, target: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for p, p_t in zip(source.parameters(), target.parameters()):
                p_t.data.mul_(1.0 - float(tau)).add_(float(tau) * p.data)

    def _sample_chain(
        self,
        network: nn.Module,
        observations: torch.Tensor,
        cfg: DPDiffusionSamplerConfig,
        deterministic: bool = False,
    ) -> DPReverseChain:
        forward = network
        limit = int(self.config.inference_microbatch_size)
        if limit > 0 and observations.shape[0] > limit:
            # Split only network forwards; diffusion noise keeps its full-batch RNG order.
            def forward(x, time, cond):
                return torch.cat(
                    [
                        network(
                            x[start : start + limit],
                            time[start : start + limit],
                            cond={key: value[start : start + limit] for key, value in cond.items()},
                        )
                        for start in range(0, x.shape[0], limit)
                    ],
                    dim=0,
                )

        return self.sampler.sample_chain(
            forward,
            observations,
            cfg,
            cond_steps=self.config.cond_steps,
            obs_dim=self.config.per_obs_dim,
            deterministic=deterministic,
        )

    @torch.no_grad()
    def sample_actions(
        self, observations: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        obs = observations.to(self.device, dtype=torch.float32)
        cfg = (
            self.config.eval_sampler_config()
            if deterministic
            else self.config.rollout_sampler_config()
        )
        chain = self._sample_chain(self.target_actor, obs, cfg, deterministic=deterministic)
        return chain.actions

    @torch.no_grad()
    def compute_flow_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """OGPO-compatible ODE-style eval hook: pure deterministic DDIM."""
        return self.sample_actions(observations, deterministic=True)

    @torch.no_grad()
    def sample_actions_BON(
        self, observations: torch.Tensor, best_of_n: Optional[int] = None
    ) -> torch.Tensor:
        obs = observations.to(self.device, dtype=torch.float32)
        b = int(obs.shape[0])
        n = int(best_of_n or self.config.best_of_n)
        obs_rep = obs.repeat_interleave(n, dim=0)
        chain = self._sample_chain(
            self.target_actor, obs_rep, self.config.rollout_sampler_config(), deterministic=False
        )
        q_all = self.target_critic(obs_rep, chain.actions)
        q_values = q_all.reshape(b, n, self.config.num_qs).permute(2, 0, 1)
        actions_reshaped = chain.actions.reshape(
            b, n, self.config.horizon_steps, self.config.action_dim
        )
        if self.config.subsample_bon and self.config.num_qs > 1:
            actions_best, _ = select_best_of_n_actions(
                actions_reshaped,
                q_values,
                subsample_bon=True,
                subsample_size=self.config.bon_subsample_size,
            )
        else:
            actions_best, _ = select_best_of_n_actions(
                actions_reshaped, q_values, subsample_bon=False
            )
        return actions_best

    @torch.no_grad()
    def select_action_chunk_np(
        self, observations: np.ndarray, deterministic: bool = False, best_of_n: Optional[int] = None
    ) -> np.ndarray:
        obs = torch.as_tensor(np.asarray(observations, dtype=np.float32), device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if deterministic:
            actions = self.sample_actions(obs, deterministic=True)
        elif best_of_n is not None and int(best_of_n) <= 1:
            actions = self.sample_actions(obs, deterministic=False)
        else:
            actions = self.sample_actions_BON(obs, best_of_n=best_of_n)
        return actions.detach().cpu().numpy().astype(np.float32)

    def bc_epsilon_loss(self, batch: OGPOSequenceBatch) -> torch.Tensor:
        obs = batch.observations.to(self.device)
        actions = batch.actions.to(self.device)
        b = int(actions.shape[0])
        cfg = self.config.rollout_sampler_config()
        sched = self.sampler._build_schedule(
            int(cfg.denoising_steps), False, 0, actions.device, actions.dtype
        )
        t = torch.randint(
            0, int(cfg.denoising_steps), (b,), device=actions.device, dtype=torch.long
        )
        alpha_bar = sched["alphas_cumprod"].gather(0, t).reshape(b, 1, 1)
        eps = torch.randn_like(actions)
        noisy = torch.sqrt(alpha_bar) * actions + torch.sqrt(1.0 - alpha_bar) * eps
        cond = self.sampler._cond_from_obs(obs, self.config.cond_steps, self.config.per_obs_dim)
        pred = self.actor(noisy, t, cond=cond)
        return F.mse_loss(pred, eps)

    def _critic_loss(self, batch: OGPOSequenceBatch) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch = batch.to(self.device)
        with torch.no_grad():
            next_chain = self._sample_chain(
                self.target_actor,
                batch.next_observations,
                self.config.rollout_sampler_config(),
                deterministic=False,
            )
            next_q_all = self.target_critic(batch.next_observations, next_chain.actions)
            next_q = aggregate_q_values(next_q_all, self.config.q_agg)
            target = compute_td_target(
                batch.rewards,
                batch.masks,
                next_q,
                discount=self.config.discount,
                horizon_length=self.config.horizon_steps,
            )
            valid = batch.valid[:, -1]

        q_pred = self.critic(batch.observations, batch.actions)
        loss = compute_td_loss(q_pred, target, valid)
        info = {
            "critic/loss": float(loss.detach().item()),
            "critic/q_mean": float(q_pred.detach().mean().item()),
            "critic/target_q_mean": float(target.detach().mean().item()),
            "critic/valid_mean": float(valid.detach().mean().item()),
        }
        return loss, info

    def _critic_step(self, batch: OGPOSequenceBatch) -> Dict[str, float]:
        loss, info = self._critic_loss(batch)
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_grad_norm_critic > 0:
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), float(self.config.max_grad_norm_critic)
            )
        self.critic_opt.step()
        return info

    def _actor_step(
        self, batch: OGPOSequenceBatch, bc_batch: OGPOSequenceBatch
    ) -> Dict[str, float]:
        batch = batch.to(self.device)
        b = int(batch.observations.shape[0])
        g = int(self.config.group_num_samples)
        obs_rep = batch.observations.repeat_interleave(g, dim=0)
        cfg = self.config.rollout_sampler_config()

        with torch.no_grad():
            chain = self._sample_chain(self.target_actor, obs_rep, cfg, deterministic=False)
            if chain.log_prob is None:
                raise RuntimeError("stochastic chain must provide old log-probs")
            old_log_prob = chain.log_prob.detach()
            q = self.target_critic(obs_rep, chain.actions).reshape(b, g, self.config.num_qs)
            advantages = conservative_group_advantages(q).reshape(b * g).detach()

        if self.config.actor_microbatch_size > 0:
            return self._actor_step_microbatched(obs_rep, chain, advantages, bc_batch, cfg)

        cur_log_prob = self.sampler.log_prob_chain(
            self.actor,
            obs_rep,
            chain,
            cfg,
            cond_steps=self.config.cond_steps,
            obs_dim=self.config.per_obs_dim,
            fuse_steps=bool(self.config.fuse_log_prob_steps),
        )
        pg_loss, ppo_stats = ppo_clipped_loss(
            cur_log_prob,
            old_log_prob,
            advantages,
            clip_epsilon=self.config.clip_epsilon,
        )
        ratio = torch.exp(cur_log_prob - old_log_prob)
        entropy = -cur_log_prob.mean()
        bc_loss = self.bc_epsilon_loss(bc_batch.to(self.device))
        actor_loss = (
            pg_loss
            - float(self.config.entropy_coeff) * entropy
            + float(self.config.bc_coeff) * bc_loss
        )

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.config.max_grad_norm_actor > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), float(self.config.max_grad_norm_actor)
            )
        self.actor_opt.step()
        self._soft_update(self.actor, self.target_actor, self.config.actor_tau)

        return {
            "actor/loss": float(actor_loss.detach().item()),
            "actor/pg_loss": float(pg_loss.detach().item()),
            "actor/bc_loss": float(bc_loss.detach().item()),
            "actor/entropy": float(entropy.detach().item()),
            "actor/ratio_mean": float(ratio.detach().mean().item()),
            "actor/ratio_clipped_total": float(ppo_stats["ratio_clipped_total"].detach().item()),
            "actor/adv_mean": float(advantages.detach().mean().item()),
        }

    def _actor_step_microbatched(
        self, observations, chain, advantages, bc_batch, cfg
    ) -> Dict[str, float]:
        count = int(observations.shape[0])
        limit = int(self.config.actor_microbatch_size)
        totals = {
            key: torch.zeros((), device=self.device)
            for key in ("pg", "entropy", "ratio", "clipped")
        }
        self.actor_opt.zero_grad(set_to_none=True)
        # Advantages are already computed over complete groups. Only the separable
        # chain loss is split; there is one BC loss, gradient clip, and Adam step.
        for start in range(0, count, limit):
            stop = min(start + limit, count)
            part = DPReverseChain(
                chain.xs[start:stop],
                chain.t_values,
                chain.log_prob[start:stop],
                chain.actions[start:stop],
            )
            log_prob = self.sampler.log_prob_chain(
                self.actor,
                observations[start:stop],
                part,
                cfg,
                cond_steps=self.config.cond_steps,
                obs_dim=self.config.per_obs_dim,
                fuse_steps=bool(self.config.fuse_log_prob_steps),
            )
            pg_loss, stats = ppo_clipped_loss(
                log_prob,
                part.log_prob,
                advantages[start:stop],
                clip_epsilon=self.config.clip_epsilon,
            )
            entropy = -log_prob.mean()
            weight = (stop - start) / count
            ((pg_loss - float(self.config.entropy_coeff) * entropy) * weight).backward()
            totals["pg"] += pg_loss.detach() * weight
            totals["entropy"] += entropy.detach() * weight
            totals["ratio"] += stats["ratio"].detach() * weight
            totals["clipped"] += stats["ratio_clipped_total"].detach() * weight
        bc_loss = self.bc_epsilon_loss(bc_batch.to(self.device))
        (float(self.config.bc_coeff) * bc_loss).backward()
        if self.config.max_grad_norm_actor > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), float(self.config.max_grad_norm_actor)
            )
        self.actor_opt.step()
        self._soft_update(self.actor, self.target_actor, self.config.actor_tau)
        loss = (
            totals["pg"]
            - float(self.config.entropy_coeff) * totals["entropy"]
            + float(self.config.bc_coeff) * bc_loss.detach()
        )
        return {
            "actor/loss": float(loss.item()),
            "actor/pg_loss": float(totals["pg"].item()),
            "actor/bc_loss": float(bc_loss.detach().item()),
            "actor/entropy": float(totals["entropy"].item()),
            "actor/ratio_mean": float(totals["ratio"].item()),
            "actor/ratio_clipped_total": float(totals["clipped"].item()),
            "actor/adv_mean": float(advantages.mean().item()),
        }

    def bc_only_update(self, bc_batch: OGPOSequenceBatch) -> Dict[str, float]:
        self.update_step += 1
        bc_loss = self.bc_epsilon_loss(bc_batch.to(self.device))
        self.actor_opt.zero_grad(set_to_none=True)
        bc_loss.backward()
        if self.config.max_grad_norm_actor > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), float(self.config.max_grad_norm_actor)
            )
        self.actor_opt.step()
        self._soft_update(self.actor, self.target_actor, self.config.actor_tau)
        return {
            "actor/loss": float(bc_loss.detach().item()),
            "actor/bc_loss": float(bc_loss.detach().item()),
            "actor/pg_loss": 0.0,
            "actor/entropy": 0.0,
            "actor/ratio_mean": 1.0,
            "actor/ratio_clipped_total": 0.0,
            "actor/adv_mean": 0.0,
            "update_step": float(self.update_step),
            "update_mode": "bc_only",
        }

    def batch_update(
        self,
        batch: OGPOSequenceBatch,
        success_batch: Optional[OGPOSequenceBatch] = None,
        success_flag: bool = False,
    ) -> Dict[str, float]:
        if str(self.config.update_mode) == "bc_only":
            bc_batch = success_batch if success_batch is not None else batch
            return self.bc_only_update(bc_batch)
        self.update_step += 1
        bc_batch = success_batch if success_batch is not None else batch
        info = self._actor_step(batch, bc_batch)
        critic_info = self._critic_step(batch)
        info.update({f"rb_{k}": v for k, v in critic_info.items()})
        if bool(success_flag) and success_batch is not None and self.config.use_success_buffer_q:
            sb_info = self._critic_step(success_batch)
            info.update({f"sb_{k}": v for k, v in sb_info.items()})
        self._soft_update(self.critic, self.target_critic, self.config.tau)
        info["update_step"] = float(self.update_step)
        return info

    def state_dict(self) -> Dict:
        def module_state(module):
            return getattr(module, "_orig_mod", module).state_dict()

        return {
            "config": self.config,
            "actor": module_state(self.actor),
            "target_actor": module_state(self.target_actor),
            "critic": module_state(self.critic),
            "target_critic": module_state(self.target_critic),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "update_step": int(self.update_step),
        }

    def load_state_dict(self, state: Dict) -> None:
        getattr(self.actor, "_orig_mod", self.actor).load_state_dict(state["actor"])
        getattr(self.target_actor, "_orig_mod", self.target_actor).load_state_dict(
            state["target_actor"]
        )
        getattr(self.critic, "_orig_mod", self.critic).load_state_dict(state["critic"])
        getattr(self.target_critic, "_orig_mod", self.target_critic).load_state_dict(
            state["target_critic"]
        )
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.update_step = int(state.get("update_step", 0))


__all__ = [
    "DPDiffusionSamplerConfig",
    "DPDiffusionChainSampler",
    "DPReverseChain",
    "DPNativeOGPOConfig",
    "DPNativeOGPOAgent",
    "EnsembleQ",
    "VectorizedEnsembleQ",
    "OGPOSequenceBatch",
    "OGPOSequenceReplayBuffer",
    "aggregate_q_values",
    "compute_td_loss",
    "compute_td_target",
    "conservative_group_advantages",
    "ppo_clipped_loss",
    "select_best_of_n_actions",
]
