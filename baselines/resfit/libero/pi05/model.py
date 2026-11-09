"""Pi0.5 pixel adapter around the unchanged formal ResFiT DP TD3 updates.

The actor conditions on the same normalized base plan as pi0.5 ExRL-Q. The
critic conditions on the executed continuous chunk, transformed into pi0.5's
conditioning coordinates. Neither network receives privileged object state.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, replace

import numpy as np
import torch
from torch import nn

from baselines.resfit.dp.td3_core import (
    ResFiTLearner,
    ResFiTLearnerConfig,
    ResidualActor,
    EnsembleContinuousCritic,
)


class ActionCoordinates:
    """ResFiT expansion of LIBERO's normalized controller command box.

    Pi0.5's checkpoint std is for RL conditioning, not an action range. LIBERO
    clips controller inputs to [-1, 1]. Residuals correct this effective base
    command in ResFiT's expanded box. Replay records the effective controller
    command, including that final clip. Exact-zero components preserve the
    original pi0.5 API command, including its out-of-range components.
    """

    def __init__(self, action_scale=0.1):
        if not np.isfinite(action_scale) or action_scale < 0:
            raise ValueError("action_scale must be finite and non-negative")
        self.action_scale = float(action_scale)
        self.half_range = np.float32(1 + action_scale)

    def encode(self, raw):
        return np.clip(np.asarray(raw, dtype=np.float32), -1, 1) / self.half_range

    def combine(self, base_raw, residual):
        base_raw = np.asarray(base_raw, dtype=np.float32)
        residual = np.asarray(residual, dtype=np.float32)
        if base_raw.shape != residual.shape or not np.isfinite(residual).all():
            raise ValueError("residual shape/values do not match the base chunk")
        corrected = np.clip(self.encode(base_raw) + residual, -1, 1)
        raw = corrected * self.half_range
        raw = np.where(residual == 0, base_raw, raw).astype(np.float32)
        return raw, self.encode(raw)

    def manifest(self):
        return {
            "mode": "libero_controller_box",
            "controller_min": [-1.0] * 7,
            "controller_max": [1.0] * 7,
            "expanded_half_range": float(self.half_range),
            "action_scale": self.action_scale,
            "max_actor_mean_raw_delta": self.action_scale * float(self.half_range),
            "replay_action": "controller_clipped_full_chunk_in_resfit_coordinates",
        }


class PixelEncoder(nn.Module):
    """Torch translation of the pinned small Encoder + PixelMultiplexer.

    Four VALID 3x3 convolutions (32 channels, strides 2,1,1,1), NHWC flatten,
    Dense(50), LayerNorm(eps=1e-6), tanh. Convolution initialization has the same
    fan-average variance as Flax default_init; RNG streams differ by framework.
    """

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList()
        for channels, stride in zip((3, 32, 32, 32), (2, 1, 1, 1)):
            layer = nn.Conv2d(channels, 32, 3, stride=stride)
            nn.init.xavier_uniform_(layer.weight, gain=2**0.25)
            nn.init.zeros_(layer.bias)
            self.convs.append(layer)
        self.projection = nn.Linear(25 * 25 * 32, 50)
        nn.init.xavier_normal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.norm = nn.LayerNorm(50, eps=1e-6)

    def forward(self, pixels):
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (64, 64, 3):
            raise ValueError("RL pixels must have shape [B,64,64,3]")
        x = pixels.to(torch.float32).permute(0, 3, 1, 2) / 255.0
        for layer in self.convs:
            x = torch.relu(layer(x))
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        return torch.tanh(self.norm(self.projection(x)))


class PixelResidualActor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = PixelEncoder()
        self.head = ResidualActor(
            58, config.action_dim, config.actor_hidden_dims, config.action_scale, use_layernorm=True
        )

    def forward(self, obs, base_action):
        if base_action.shape != obs["base_conditioning"].shape:
            raise ValueError("base chunk shape mismatch")
        state = torch.cat((self.encoder(obs["pixels"]), obs["state"]), -1)
        return self.head(state, obs["base_conditioning"])


class PixelContinuousCritic(nn.Module):
    def __init__(self, config, norm_stats):
        super().__init__()
        self.encoder = PixelEncoder()
        self.head = EnsembleContinuousCritic(
            58, config.action_dim, config.critic_hidden_dims, config.critic_ensemble_size, True
        )
        self.register_buffer("action_mean", torch.as_tensor(norm_stats["action_mean"]).float())
        self.register_buffer("action_std", torch.as_tensor(norm_stats["action_std"]).float())
        self.half_range = float(1 + config.action_scale)
        self.extra_delta_transform = bool(norm_stats["extra_delta_transform"])
        # Match the pi0.5 ExRL-Q MLP init / LayerNorm convention. The scalar
        # continuous head replaces its vector of discrete duration values.
        for layer in self.head.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=2**0.25)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                layer.eps = 1e-6

    def condition_action(self, obs, action):
        # Differentiable physical clipping is also used in actor/target Q.
        raw = torch.clamp(action.reshape(-1, 50, 7) * self.half_range, -1, 1)
        conditioning = raw
        if self.extra_delta_transform:
            conditioning = torch.cat(
                (raw[..., :6] - obs["raw_state"][:, None, :6], raw[..., 6:]), -1
            )
        return torch.clamp(
            (conditioning - self.action_mean) / (self.action_std + 1e-6), -10, 10
        ).flatten(1)

    def forward(self, obs, action):
        state = torch.cat((self.encoder(obs["pixels"]), obs["state"]), -1)
        return self.head(state, self.condition_action(obs, action))


class ReferenceAugmenter:
    """Run the formal pi0/pi0.5 JAX augmentations, with a separate saved RNG."""

    def __init__(self, seed):
        import jax
        import jax.numpy as jnp
        from baselines.resfit.libero.pi05.reference_augmentations import (
            batched_random_crop,
            color_transform,
        )

        self.key = jax.random.PRNGKey(seed)

        @jax.jit
        def augment(key, pixels):
            key, crop_key, color_key = jax.random.split(key, 3)
            pixels = batched_random_crop(crop_key, pixels[..., None])
            pixels = (color_transform(color_key, pixels.astype(jnp.float32) / 255.0) * 255).astype(
                jnp.uint8
            )
            return key, pixels[..., 0]

        self.augment = augment

    def __call__(self, pixels):
        import jax.dlpack

        # DLPack keeps small image batches on the selected GPU. No pi0
        # parameters or gradients are shared with the residual learner.
        image = jax.dlpack.from_dlpack(pixels.contiguous())
        self.key, image = self.augment(self.key, image)
        return torch.utils.dlpack.from_dlpack(image)

    def state_dict(self):
        return np.asarray(self.key).copy()

    def load_state_dict(self, state):
        import jax.numpy as jnp

        self.key = jnp.asarray(state, dtype=jnp.uint32)


class PixelResFiTLearner(ResFiTLearner):
    """Only the observation/network adapter changes; TD3 methods are inherited."""

    def __init__(self, config: ResFiTLearnerConfig, norm_stats, device, *, seed=0, augment=True):
        if config.state_dim != 8 or config.action_dim != 350:
            raise ValueError("pi0.5 LIBERO contract is proprio8 and chunk50x7")
        if config.td_loss_mode != "per_head_mse":
            raise ValueError("formal ResFiT uses per_head_mse")
        if not 0 < config.target_critic_subsample_size <= config.critic_ensemble_size:
            raise ValueError("invalid target ensemble size")
        self.config, self.device = config, torch.device(device)
        self.actor = PixelResidualActor(config).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = PixelContinuousCritic(config, norm_stats).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.train(False)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=config.critic_lr)
        self.augmenter = ReferenceAugmenter(seed + 30000) if augment else None

    @torch.no_grad()
    def act(self, states, base_actions, eval_mode, exploration_std=0.025, exploration_clip=0.3):
        obs = {k: torch.as_tensor(v, device=self.device) for k, v in states.items()}
        base = torch.as_tensor(base_actions, dtype=torch.float32, device=self.device)
        residual = self.actor(obs, base)
        if not eval_mode and exploration_std > 0:
            residual += torch.clamp(
                torch.randn_like(residual) * exploration_std, -exploration_clip, exploration_clip
            )
        return residual.clamp(-1 + 1e-6, 1 - 1e-6).cpu().numpy()

    def update(self, batch, update_actor):
        if self.augmenter is not None:
            batch = replace(
                batch,
                states={**batch.states, "pixels": self.augmenter(batch.states["pixels"])},
                next_states={
                    **batch.next_states,
                    "pixels": self.augmenter(batch.next_states["pixels"]),
                },
            )
        return super().update(batch, update_actor)

    def state_dict(self):
        return {
            **super().state_dict(),
            "pixel_adapter_version": 2,
            "augmentation_rng": None if self.augmenter is None else self.augmenter.state_dict(),
        }

    def load_state_dict(self, state):
        if state.get("pixel_adapter_version") != 2:
            raise ValueError("not a ResFiT pi0.5 checkpoint")
        if (state["augmentation_rng"] is None) != (self.augmenter is None):
            raise ValueError("augmentation configuration changed")
        super().load_state_dict(state)
        if self.augmenter is not None:
            self.augmenter.load_state_dict(state["augmentation_rng"])
