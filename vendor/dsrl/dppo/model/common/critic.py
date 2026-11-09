"""
Critic networks.

"""

import math
from typing import Union
import torch
import einops
from copy import deepcopy

from model.common.mlp import MLP, ResidualMLP, activation_dict
from model.common.modules import SpatialEmb, RandomShiftsAug


class CriticObs(torch.nn.Module):
    """State-only critic network."""

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        **kwargs,
    ):
        super().__init__()
        mlp_dims = [cond_dim] + mlp_dims + [1]
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.Q1 = model(
            mlp_dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )

    def forward(self, cond: Union[dict, torch.Tensor]):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            or (B, num_feature) from ViT encoder
        """
        if isinstance(cond, dict):
            B = len(cond["state"])

            # flatten history
            state = cond["state"].view(B, -1)
        else:
            state = cond
        q1 = self.Q1(state)
        return q1


class CriticObsAct(torch.nn.Module):
    """State-action double critic network."""

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        action_dim,
        action_steps=1,
        activation_type="Mish",
        use_layernorm=False,
        residual_tyle=False,
        double_q=True,
        **kwargs,
    ):
        super().__init__()
        mlp_dims = [cond_dim + action_dim * action_steps] + mlp_dims + [1]
        if residual_tyle:
            model = ResidualMLP
        else:
            model = MLP
        self.Q1 = model(
            mlp_dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )
        if double_q:
            self.Q2 = model(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )

    def forward(self, cond: dict, action):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        action: (B, Ta, Da)
        """
        B = len(cond["state"])

        # flatten history
        state = cond["state"].view(B, -1)

        # flatten action
        action = action.view(B, -1)

        x = torch.cat((state, action), dim=-1)
        if hasattr(self, "Q2"):
            q1 = self.Q1(x)
            q2 = self.Q2(x)
            return q1.squeeze(1), q2.squeeze(1)
        else:
            q1 = self.Q1(x)
            return q1.squeeze(1)


class ViTCritic(CriticObs):
    """ViT + MLP, state only"""

    def __init__(
        self,
        backbone,
        cond_dim,
        img_cond_steps=1,
        spatial_emb=128,
        dropout=0,
        augment=False,
        num_img=1,
        **kwargs,
    ):
        # update input dim to mlp
        mlp_obs_dim = spatial_emb * num_img + cond_dim
        super().__init__(cond_dim=mlp_obs_dim, **kwargs)
        self.backbone = backbone
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        if num_img > 1:
            self.compress1 = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
            self.compress2 = deepcopy(self.compress1)
        else:  # TODO: clean up
            self.compress = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment

    def forward(
        self,
        cond: dict,
        no_augment=False,
    ):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)
        no_augment: whether to skip augmentation

        TODO long term: more flexible handling of cond
        """
        B, T_rgb, C, H, W = cond["rgb"].shape

        # flatten history
        state = cond["state"].view(B, -1)

        # Take recent images --- sometimes we want to use fewer img_cond_steps than cond_steps (e.g., 1 image but 3 prio)
        rgb = cond["rgb"][:, -self.img_cond_steps :]

        # concatenate images in cond by channels
        if self.num_img > 1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")

        # convert rgb to float32 for augmentation
        rgb = rgb.float()

        # get vit output - pass in two images separately
        if self.num_img > 1:  # TODO: properly handle multiple images
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if self.augment and not no_augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)
            feat1 = self.compress1.forward(feat1, state)
            feat2 = self.compress2.forward(feat2, state)
            feat = torch.cat([feat1, feat2], dim=-1)
        else:  # single image
            if self.augment and not no_augment:
                rgb = self.aug(rgb)  # uint8 -> float32
            feat = self.backbone(rgb)
            feat = self.compress.forward(feat, state)
        feat = torch.cat([feat, state], dim=-1)
        return super().forward(feat)


class _DistributionalCriticMember(torch.nn.Module):
    """One member of a distributional state-action critic ensemble."""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        num_bins,
        activation_type="SiLU",
        use_layernorm=True,
        init_final_scale=1e-2,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layer = torch.nn.Linear(last_dim, hidden_dim)
            torch.nn.init.orthogonal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            self.norms.append(
                torch.nn.LayerNorm(hidden_dim) if use_layernorm else torch.nn.Identity()
            )
            last_dim = hidden_dim
        self.activation = activation_dict[activation_type]
        self.logits = torch.nn.Linear(last_dim, num_bins)
        torch.nn.init.orthogonal_(self.logits.weight, gain=init_final_scale)
        torch.nn.init.zeros_(self.logits.bias)

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = self.activation(norm(layer(x)))
        return self.logits(x)


class DistributionalCriticObsAct(torch.nn.Module):
    """State-action distributional critic ensemble.

    This mirrors PA-RL's JAX `DistributionalCritic`: each ensemble member emits
    logits over fixed Q bins, while the scalar Q used for ranking/backups is the
    expectation under the softmax probabilities.
    """

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        action_dim,
        action_steps=1,
        activation_type="SiLU",
        use_layernorm=True,
        critic_ensemble_size=10,
        q_min=-100.0,
        q_max=0.0,
        num_bins=128,
        init_final_scale=1e-2,
        **kwargs,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_steps = action_steps
        self.critic_ensemble_size = int(critic_ensemble_size)
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.num_bins = int(num_bins)
        input_dim = cond_dim + action_dim * action_steps
        self.members = torch.nn.ModuleList(
            [
                _DistributionalCriticMember(
                    input_dim=input_dim,
                    hidden_dims=mlp_dims,
                    num_bins=self.num_bins,
                    activation_type=activation_type,
                    use_layernorm=use_layernorm,
                    init_final_scale=init_final_scale,
                )
                for _ in range(self.critic_ensemble_size)
            ]
        )
        support_edges = torch.linspace(self.q_min, self.q_max, self.num_bins + 1)
        support_centers = 0.5 * (support_edges[:-1] + support_edges[1:])
        self.register_buffer("support_edges", support_edges)
        self.register_buffer("support_centers", support_centers)
        self.hl_gauss_sigma = 0.75 * (self.q_max - self.q_min) / self.num_bins

    def forward(self, cond: dict, action):
        B = len(cond["state"])
        state = cond["state"].view(B, -1)
        action = action.view(B, -1)
        x = torch.cat((state, action), dim=-1)
        logits = torch.stack([member(x) for member in self.members], dim=0)
        probs = torch.softmax(logits, dim=-1)
        q_values = torch.sum(probs * self.support_centers.view(1, 1, -1), dim=-1)
        return q_values, logits

    @staticmethod
    def _log1mexp(x):
        x = torch.abs(x)
        log_two = math.log(2.0)
        return torch.where(
            x < log_two,
            torch.log(-torch.expm1(-x)),
            torch.log1p(-torch.exp(-x)),
        )

    @classmethod
    def _log_sub_exp(cls, x, y):
        larger = torch.maximum(x, y)
        smaller = torch.minimum(x, y)
        return larger + cls._log1mexp(torch.clamp(larger - smaller, min=0.0))

    @classmethod
    def _normal_cdf_log_difference(cls, x, y):
        # Computes log(ndtr(x) - ndtr(y)) for x >= y using the same stable
        # branch as PA-RL's JAX HL-Gauss transform.
        y_positive = y >= 0
        x_hat = torch.where(y_positive, -y, x)
        y_hat = torch.where(y_positive, -x, y)
        return cls._log_sub_exp(
            torch.special.log_ndtr(x_hat),
            torch.special.log_ndtr(y_hat),
        )

    def scalar_target_to_probs(self, target):
        target = target.view(-1, 1)
        support = self.support_edges.to(device=target.device, dtype=target.dtype)
        sigma = torch.as_tensor(
            self.hl_gauss_sigma,
            device=target.device,
            dtype=target.dtype,
        )
        denom = math.sqrt(2.0) * sigma
        upper = (support[1:].view(1, -1) - target) / denom
        lower = (support[:-1].view(1, -1) - target) / denom
        bin_log_probs = self._normal_cdf_log_difference(upper, lower)
        log_z = self._normal_cdf_log_difference(
            (support[-1] - target) / denom,
            (support[0] - target) / denom,
        )
        probs = torch.exp(bin_log_probs - log_z)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        normalizer = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probs / normalizer

    def cross_entropy_loss_on_scalar(self, logits, target):
        target_probs = self.scalar_target_to_probs(target).detach()
        log_probs = torch.log_softmax(logits, dim=-1)
        return -torch.sum(target_probs.unsqueeze(0) * log_probs, dim=-1)
