"""Base-policy inference and imitation losses for chunked control."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exrl.runners.dp_q.diffusion import (
    BackboneMeta,
    DiffusionSampler,
    DiffusionSamplerConfig,
    DiffusionPolicy,
)


class _CfgNode:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _AlgoConfig:
    def __init__(
        self,
        horizon_steps: int,
        cond_steps: int,
        train_denoising_steps: int = 5,
        default_inference_steps: int = 5,
    ):
        train_steps = max(1, int(train_denoising_steps))
        infer_steps = max(1, int(default_inference_steps))
        self.horizon = _CfgNode(
            observation_horizon=int(cond_steps),
            action_horizon=int(horizon_steps),
            prediction_horizon=int(horizon_steps),
        )
        # Keep fields compatible with existing wrapper runtime toggles.
        # Important: num_train_timesteps is the original diffusion horizon used
        # to build the noise schedule; num_inference_timesteps is how many DDIM
        # points to sample at runtime. These are not interchangeable.
        self.ddpm = _CfgNode(
            enabled=True,
            num_train_timesteps=train_steps,
            num_inference_timesteps=infer_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.ddim = _CfgNode(
            enabled=False,
            num_train_timesteps=train_steps,
            num_inference_timesteps=infer_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

    @contextmanager
    def values_unlocked(self):
        yield self


class DSRLCkptPolicyAdapter(nn.Module):
    """
    Mimics minimal interface used by adaptive_env_utils_parallel_opt wrappers.
    """

    def __init__(
        self,
        backbone: DiffusionPolicy,
        backbone_meta: BackboneMeta,
        obs_shapes: Dict[str, tuple],
        obs_key_order: List[str],
        device: torch.device,
        default_weight_source: str = "ema",
        train_denoising_steps: int = 5,
        default_inference_steps: int = 5,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.obs_shapes = dict(obs_shapes)
        self._obs_key_order = list(obs_key_order)
        self._obs_total_dim = int(
            sum(int(np.prod(self.obs_shapes[k])) for k in self._obs_key_order)
        )

        self.ac_dim = int(backbone_meta.action_dim)
        self.horizon_steps = int(backbone_meta.horizon_steps)
        self.cond_steps = int(backbone_meta.cond_steps)
        self.obs_dim = int(backbone_meta.obs_dim)

        self.noise_pred_net = backbone
        # Existing wrappers may access model.nets["policy"]["noise_pred_net"]
        self.nets = nn.ModuleDict(
            {"policy": nn.ModuleDict({"noise_pred_net": self.noise_pred_net})}
        )

        self.algo_config = _AlgoConfig(
            horizon_steps=self.horizon_steps,
            cond_steps=self.cond_steps,
            train_denoising_steps=int(train_denoising_steps),
            default_inference_steps=int(default_inference_steps),
        )
        self.sampler = DiffusionSampler(
            horizon_steps=self.horizon_steps, action_dim=self.ac_dim
        )

        self.use_action_scheduler = False
        self.use_ood_monitor = False
        self.use_cuda_graph_trajectory = False
        self.default_weight_source = str(default_weight_source)
        self.ema = None
        self.to(self.device)
        self.set_eval()

    def set_eval(self):
        self.eval()
        self.noise_pred_net.eval()

    def set_train(self):
        self.train()
        self.noise_pred_net.train()

    def reset(self):
        # Compatibility with wrappers that call reset on policy.
        return None

    def _make_state_cond(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        parts = []
        batch = None
        for k in self._obs_key_order:
            shp = self.obs_shapes[k]
            d = int(np.prod(shp))
            if k in obs_dict:
                v = obs_dict[k]
                if not torch.is_tensor(v):
                    v = torch.as_tensor(v, dtype=torch.float32)
                v = v.to(self.device, dtype=torch.float32)
                if v.ndim == 1:
                    v = v.view(1, -1)
                else:
                    v = v.reshape(v.shape[0], -1)
                if v.shape[1] != d:
                    raise ValueError(
                        f"[DSRL-ADAPTER] obs dim mismatch key={k}: expected {d}, got {int(v.shape[1])}"
                    )
                if batch is None:
                    batch = int(v.shape[0])
                elif int(v.shape[0]) != batch:
                    raise ValueError(
                        f"[DSRL-ADAPTER] batch mismatch key={k}: expected {batch}, got {int(v.shape[0])}"
                    )
                parts.append(v)
            else:
                if batch is None:
                    for vv in obs_dict.values():
                        if torch.is_tensor(vv):
                            batch = int(vv.shape[0]) if vv.ndim > 1 else 1
                            break
                    if batch is None:
                        raise ValueError(
                            "[DSRL-ADAPTER] cannot infer batch size from obs_dict"
                        )
                parts.append(
                    torch.zeros((batch, d), device=self.device, dtype=torch.float32)
                )

        flat = torch.cat(parts, dim=-1)
        if flat.shape[1] != self._obs_total_dim:
            raise ValueError(
                f"[DSRL-ADAPTER] obs concat mismatch: expected total={self._obs_total_dim}, got={int(flat.shape[1])}"
            )
        expected_cond = self.cond_steps * self.obs_dim
        if flat.shape[1] != expected_cond:
            raise ValueError(
                f"[DSRL-ADAPTER] cond shape mismatch: expected cond_steps*obs_dim={expected_cond}, got={int(flat.shape[1])}"
            )
        state = flat.reshape(flat.shape[0], self.cond_steps, self.obs_dim)
        return {"state": state}

    @torch.no_grad()
    def _get_action_trajectory(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        cond = self._make_state_cond(obs_dict)

        if bool(self.algo_config.ddim.enabled):
            sampler_cfg = DiffusionSamplerConfig(
                # Match official semantics: build the schedule from the
                # training diffusion horizon, then subsample DDIM steps.
                denoising_steps=int(self.algo_config.ddim.num_train_timesteps),
                use_ddim=True,
                ddim_steps=int(self.algo_config.ddim.num_inference_timesteps),
                denoised_clip_value=1.0,
                # Align with official DSRL / DPPO eval config.
                randn_clip_value=3.0,
                # Keep official eval semantics: no final action clip.
                final_action_clip_value=None,
                eps_clip_value=None,
            )
        else:
            sampler_cfg = DiffusionSamplerConfig(
                denoising_steps=int(self.algo_config.ddpm.num_train_timesteps),
                use_ddim=False,
                ddim_steps=0,
                denoised_clip_value=1.0,
                # Align with official DSRL / DPPO eval config.
                randn_clip_value=3.0,
                # Keep official eval semantics: no final action clip.
                final_action_clip_value=None,
                eps_clip_value=None,
            )

        if self.use_cuda_graph_trajectory:
            acts = self.sampler.sample_cuda_graph(
                network=self.nets["policy"]["noise_pred_net"],
                cond=cond,
                cfg=sampler_cfg,
            )
        else:
            acts = self.sampler.sample(
                network=self.nets["policy"]["noise_pred_net"],
                cond=cond,
                cfg=sampler_cfg,
                deterministic=True,
            )
        # Match official DSRL eval path: no final action clipping at adapter output.
        return acts

    def diffusion_bc_loss(
        self,
        actions: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        bc_loss, _ = self.diffusion_bc_loss_terms(actions, conditions)
        return bc_loss

    def diffusion_bc_loss_terms(
        self,
        actions: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
        *,
        teacher_noise_pred_net: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        teacher_nets = (
            [] if teacher_noise_pred_net is None else [teacher_noise_pred_net]
        )
        bc_loss, teacher_losses = self.diffusion_bc_loss_multi_teacher_terms(
            actions,
            conditions,
            teacher_noise_pred_nets=teacher_nets,
        )
        teacher_loss = teacher_losses[0] if teacher_losses else bc_loss.new_zeros(())
        return bc_loss, teacher_loss

    def diffusion_bc_loss_multi_teacher_terms(
        self,
        actions: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
        *,
        teacher_noise_pred_nets: Sequence[nn.Module],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Official DP/DSRL epsilon-prediction BC objective.

        This mirrors the pretrain loss used by DSRL diffusion policies:
        sample a random diffusion timestep, add noise to the ground-truth
        action horizon, and regress the predicted epsilon with MSE. Every
        teacher is evaluated on the exact same noisy action, timestep, and
        condition for directly comparable function-space constraints.
        """
        if "state" not in conditions:
            raise ValueError("[DSRL-ADAPTER] BC conditions require key 'state'")
        x_start = actions.to(self.device, dtype=torch.float32)
        cond = {
            k: (
                v.to(self.device, dtype=torch.float32)
                if torch.is_tensor(v)
                else torch.as_tensor(v, device=self.device, dtype=torch.float32)
            )
            for k, v in conditions.items()
        }
        if x_start.ndim != 3:
            raise ValueError(
                f"[DSRL-ADAPTER] BC actions must be [B,H,A], got {tuple(x_start.shape)}"
            )
        if tuple(x_start.shape[1:]) != (self.horizon_steps, self.ac_dim):
            raise ValueError(
                f"[DSRL-ADAPTER] BC action shape mismatch: got {tuple(x_start.shape[1:])}, "
                f"expected {(self.horizon_steps, self.ac_dim)}"
            )
        state = cond["state"]
        if state.ndim != 3 or tuple(state.shape[1:]) != (self.cond_steps, self.obs_dim):
            raise ValueError(
                f"[DSRL-ADAPTER] BC state condition mismatch: got {tuple(state.shape)}, "
                f"expected [B,{self.cond_steps},{self.obs_dim}]"
            )
        if int(state.shape[0]) != int(x_start.shape[0]):
            raise ValueError(
                f"[DSRL-ADAPTER] BC batch mismatch: actions={int(x_start.shape[0])} state={int(state.shape[0])}"
            )

        train_steps = int(getattr(self.algo_config.ddpm, "num_train_timesteps", 0) or 0)
        if train_steps <= 0:
            train_steps = int(
                getattr(self.algo_config.ddim, "num_train_timesteps", 0) or 0
            )
        train_steps = max(train_steps, 1)

        b = int(x_start.shape[0])
        t = torch.randint(0, train_steps, (b,), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        sched = self.sampler._build_schedule(
            denoising_steps=train_steps,
            use_ddim=False,
            ddim_steps=0,
            device=x_start.device,
        )
        alphas_cumprod = sched["alphas_cumprod"]
        sqrt_alpha = torch.sqrt(self.sampler._extract(alphas_cumprod, t, x_start.shape))
        sqrt_one_minus_alpha = torch.sqrt(
            torch.clamp(
                1.0 - self.sampler._extract(alphas_cumprod, t, x_start.shape), min=0.0
            )
        )
        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        pred_noise = self.nets["policy"]["noise_pred_net"](x_noisy, t, cond=cond)
        bc_loss = F.mse_loss(pred_noise, noise, reduction="mean")
        teacher_losses = []
        for teacher_noise_pred_net in teacher_noise_pred_nets:
            with torch.no_grad():
                teacher_pred_noise = teacher_noise_pred_net(x_noisy, t, cond=cond)
            teacher_losses.append(
                F.mse_loss(
                    pred_noise,
                    teacher_pred_noise.detach(),
                    reduction="mean",
                )
            )
        return bc_loss, teacher_losses


@dataclass
class DSRLCkptPolicyBundle:
    policy: DSRLCkptPolicyAdapter
    source_path: str
    weight_source: str
    meta: BackboneMeta


class DSRLCkptPolicyWrapper:
    """
    Thin wrapper so callers can use wrapper.policy just like robomimic FileUtils output.
    """

    def __init__(self, bundle: DSRLCkptPolicyBundle):
        self.policy = bundle.policy
        self._bundle = bundle

    @property
    def meta(self) -> BackboneMeta:
        return self._bundle.meta

    @property
    def source_path(self) -> str:
        return self._bundle.source_path

    @property
    def weight_source(self) -> str:
        return self._bundle.weight_source
