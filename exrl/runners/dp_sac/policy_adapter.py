"""Base-policy inference and imitation losses for chunked control."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from exrl.runners.dp_sac.diffusion import (
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
