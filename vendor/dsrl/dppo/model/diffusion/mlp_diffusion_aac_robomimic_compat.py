"""
MLP models for diffusion policies.

AAC / robomimic compat notes:
- Keep original DSRL classes intact in this copied file.
- Add a low-dim robomimic UNet backbone that obeys the same
  forward(x, time, cond) contract expected by DSRL diffusion samplers.
"""

import json

import torch
import torch.nn as nn
import logging
import einops
from copy import deepcopy
from collections import OrderedDict

from model.common.mlp import MLP, ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb
from model.common.modules import SpatialEmb, RandomShiftsAug

log = logging.getLogger(__name__)


class IdentityLowdimObsEncoder(nn.Module):
    """
    Exact low-dim identity path for robomimic configs with core_class=None.
    This matches the current stable low-dim checkpoints we are adapting:
    there are no obs_encoder weights, and conditioning is just ordered concat.
    """

    def __init__(self, lowdim_keys, state_key_dims):
        super().__init__()
        self.lowdim_keys = list(lowdim_keys)
        self.state_key_dims = OrderedDict((str(k), int(v)) for k, v in state_key_dims.items())
        self._output_dim = int(sum(self.state_key_dims[k] for k in self.lowdim_keys))

    def output_shape(self):
        return (self._output_dim,)

    def forward(self, obs):
        pieces = [obs[k] for k in self.lowdim_keys]
        return torch.cat(pieces, dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """
    Copied from our robomimic low-dim diffusion backbone so the DSRL compat
    branch stays self-contained and does not depend on another repo snapshot.
    """

    def __init__(
        self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.down_modules = down_modules
        self.up_modules = up_modules
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        print("number of parameters: {:e}".format(sum(p.numel() for p in self.parameters())))

    def forward(self, sample, timestep, global_cond=None):
        sample = sample.moveaxis(-1, -2)
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)


class VisionDiffusionMLP(nn.Module):
    """With ViT backbone"""

    def __init__(
        self,
        backbone,
        action_dim,
        horizon_steps,
        cond_dim,
        img_cond_steps=1,
        time_dim=16,
        mlp_dims=[256, 256],
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        spatial_emb=0,
        visual_feature_dim=128,
        dropout=0,
        num_img=1,
        augment=False,
    ):
        super().__init__()

        # vision
        self.backbone = backbone
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        if spatial_emb > 0:
            assert spatial_emb > 1, "this is the dimension"
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
            visual_feature_dim = spatial_emb * num_img
        else:
            self.compress = nn.Sequential(
                nn.Linear(self.backbone.repr_dim, visual_feature_dim),
                nn.LayerNorm(visual_feature_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )

        # diffusion
        input_dim = (
            time_dim + action_dim * horizon_steps + visual_feature_dim + cond_dim
        )
        output_dim = action_dim * horizon_steps
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.time_dim = time_dim

    def forward(
        self,
        x,
        time,
        cond: dict,
        **kwargs,
    ):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)

        TODO long term: more flexible handling of cond
        """
        B, Ta, Da = x.shape
        _, T_rgb, C, H, W = cond["rgb"].shape

        # flatten chunk
        x = x.view(B, -1)

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
            if self.augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)
            feat1 = self.compress1.forward(feat1, state)
            feat2 = self.compress2.forward(feat2, state)
            feat = torch.cat([feat1, feat2], dim=-1)
        else:  # single image
            if self.augment:
                rgb = self.aug(rgb)
            feat = self.backbone(rgb)

            # compress
            if isinstance(self.compress, SpatialEmb):
                feat = self.compress.forward(feat, state)
            else:
                feat = feat.flatten(1, -1)
                feat = self.compress(feat)
        cond_encoded = torch.cat([feat, state], dim=-1)

        # append time and cond
        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)
        x = torch.cat([x, time_emb, cond_encoded], dim=-1)

        # mlp
        out = self.mlp_mean(x)
        return out.view(B, Ta, Da)


class DiffusionMLP(nn.Module):

    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_dim,
        time_dim=16,
        mlp_dims=[256, 256],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    ):
        super().__init__()
        output_dim = action_dim * horizon_steps
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        if cond_mlp_dims is not None:
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            input_dim = time_dim + action_dim * horizon_steps + cond_mlp_dims[-1]
        else:
            input_dim = time_dim + action_dim * horizon_steps + cond_dim

        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.time_dim = time_dim

    def forward(
        self,
        x,
        time,
        cond,
        **kwargs,
    ):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        B, Ta, Da = x.shape

        # flatten chunk
        x = x.view(B, -1)

        # flatten history
        state = cond["state"].view(B, -1)

        # obs encoder
        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state)

        # append time and cond
        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)
        x = torch.cat([x, time_emb, state], dim=-1)

        # mlp head
        out = self.mlp_mean(x)
        return out.view(B, Ta, Da)


class RobomimicLowdimDiffusionUnetCompat(nn.Module):
    """
    DSRL-compatible low-dim backbone that preserves the current stable
    low-dim robomimic semantics:
    - state is split by explicit lowdim keys and concatenated in-order
    - noise prediction uses the same ConditionalUnet1D architecture
    while preserving DSRL diffusion sampler / RL logic.
    """

    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_steps,
        state_key_dims,
        lowdim_keys=None,
        obs_encoder_config=None,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        backbone_source="fresh_init",
        robomimic_ckpt_path=None,
        weight_source="ema",
        strict_ckpt_load=True,
        **kwargs,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.horizon_steps = int(horizon_steps)
        self.cond_steps = int(cond_steps)
        self.backbone_source = str(backbone_source)
        self.robomimic_ckpt_path = robomimic_ckpt_path
        self.weight_source = str(weight_source)
        self.strict_ckpt_load = bool(strict_ckpt_load)

        self.lowdim_keys, self.state_key_dims = self._normalize_key_dims(
            state_key_dims=state_key_dims,
            lowdim_keys=lowdim_keys,
        )
        self.obs_dim = int(sum(self.state_key_dims[k] for k in self.lowdim_keys))
        self.obs_encoder_config = self._normalize_obs_encoder_config(obs_encoder_config)
        self._validate_lowdim_encoder_config(self.obs_encoder_config)

        self.obs_encoder = IdentityLowdimObsEncoder(
            lowdim_keys=self.lowdim_keys,
            state_key_dims=self.state_key_dims,
        )
        self.obs_feature_dim = int(self.obs_encoder.output_shape()[0])
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_feature_dim * self.cond_steps,
            diffusion_step_embed_dim=int(diffusion_step_embed_dim),
            down_dims=list(down_dims),
            kernel_size=int(kernel_size),
            n_groups=int(n_groups),
        )

        if self.backbone_source == "robomimic_ckpt":
            if not self.robomimic_ckpt_path:
                raise ValueError("robomimic_ckpt_path is required when backbone_source=robomimic_ckpt")
            self.load_robomimic_ckpt(
                ckpt_path=self.robomimic_ckpt_path,
                weight_source=self.weight_source,
                strict=self.strict_ckpt_load,
            )
        elif self.backbone_source != "fresh_init":
            raise ValueError(f"unknown backbone_source={self.backbone_source}")

    @staticmethod
    def _normalize_key_dims(state_key_dims, lowdim_keys=None):
        if hasattr(state_key_dims, "items"):
            ordered_items = list(state_key_dims.items())
        elif isinstance(state_key_dims, (list, tuple)):
            if lowdim_keys is None:
                raise ValueError("lowdim_keys is required when state_key_dims is a list/tuple")
            if len(lowdim_keys) != len(state_key_dims):
                raise ValueError("lowdim_keys and state_key_dims length mismatch")
            ordered_items = list(zip(lowdim_keys, state_key_dims))
        else:
            raise TypeError("state_key_dims must be mapping-like or list/tuple")

        keys = [str(k) for k, _ in ordered_items]
        if lowdim_keys is not None:
            lowdim_keys = [str(k) for k in lowdim_keys]
            if keys != lowdim_keys:
                raise ValueError(
                    f"lowdim_keys/order mismatch: from_state_key_dims={keys}, explicit_lowdim_keys={lowdim_keys}"
                )
        dims = OrderedDict()
        for k, v in ordered_items:
            vv = int(v)
            if vv <= 0:
                raise ValueError(f"state_key_dims[{k}] must be > 0, got {vv}")
            dims[str(k)] = vv
        return list(dims.keys()), dims

    @staticmethod
    def _normalize_obs_encoder_config(obs_encoder_config):
        if obs_encoder_config is None:
            return {
                "low_dim": {
                    "core_class": None,
                    "core_kwargs": {},
                    "obs_randomizer_class": None,
                    "obs_randomizer_kwargs": {},
                }
            }
        if hasattr(obs_encoder_config, "items"):
            out = {}
            for mod, cfg in obs_encoder_config.items():
                if hasattr(cfg, "items"):
                    out[str(mod)] = {str(k): v for k, v in cfg.items()}
                else:
                    out[str(mod)] = cfg
            return out
        raise TypeError("obs_encoder_config must be mapping-like")

    @staticmethod
    def _validate_lowdim_encoder_config(obs_encoder_config):
        low_dim_cfg = obs_encoder_config.get("low_dim", {})
        if low_dim_cfg.get("core_class", None) is not None:
            raise ValueError(
                "RobomimicLowdimDiffusionUnetCompat only supports low_dim core_class=None "
                "because current adapted checkpoints use identity low-dim encoding."
            )
        if low_dim_cfg.get("obs_randomizer_class", None) is not None:
            raise ValueError(
                "RobomimicLowdimDiffusionUnetCompat does not support low-dim randomizers."
            )

    @staticmethod
    def _select_robomimic_state(raw_ckpt, weight_source):
        if not isinstance(raw_ckpt, dict) or "model" not in raw_ckpt:
            raise ValueError("expected robomimic checkpoint dict with top-level 'model'")
        model = raw_ckpt["model"]
        if not isinstance(model, dict):
            raise ValueError("robomimic checkpoint['model'] must be a dict")
        source = str(weight_source)
        if source == "ema":
            state = model.get("ema", None)
        elif source == "nets":
            state = model.get("nets", None)
        else:
            raise ValueError(f"unsupported weight_source={weight_source}")
        if state is None:
            raise ValueError(f"robomimic checkpoint missing model['{source}']")
        return state

    def _validate_against_robomimic_ckpt(self, raw_ckpt):
        try:
            config = json.loads(raw_ckpt["config"])
            horizon = config["algo"]["horizon"]
            if int(horizon["prediction_horizon"]) != self.horizon_steps:
                raise ValueError(
                    f"robomimic ckpt prediction_horizon={horizon['prediction_horizon']} != expected {self.horizon_steps}"
                )
            if int(horizon["observation_horizon"]) != self.cond_steps:
                raise ValueError(
                    f"robomimic ckpt observation_horizon={horizon['observation_horizon']} != expected {self.cond_steps}"
                )
            low_dim_cfg = config["observation"]["encoder"]["low_dim"]
            if low_dim_cfg.get("core_class", None) is not None:
                raise ValueError(
                    "robomimic ckpt low_dim encoder is not identity; this compat path only "
                    "supports core_class=None low-dim checkpoints."
                )
        except KeyError as exc:
            raise ValueError(f"robomimic ckpt missing expected config field: {exc}") from exc

        shape_meta = raw_ckpt.get("shape_metadata", {})
        shape_keys = shape_meta.get("all_obs_keys", None)
        if shape_keys is not None:
            shape_keys = [str(k) for k in shape_keys]
            if shape_keys != list(self.lowdim_keys):
                raise ValueError(
                    "robomimic ckpt lowdim key mismatch against shape_metadata order: "
                    f"ckpt={shape_keys}, expected={self.lowdim_keys}"
                )
        else:
            lowdim = config["observation"]["modalities"]["obs"]["low_dim"]
            if [str(k) for k in lowdim] != list(self.lowdim_keys):
                raise ValueError(
                    f"robomimic ckpt lowdim key mismatch: ckpt={lowdim}, expected={self.lowdim_keys}"
                )
        if int(shape_meta.get("ac_dim", self.action_dim)) != self.action_dim:
            raise ValueError(
                f"robomimic ckpt action dim mismatch: ckpt={shape_meta.get('ac_dim')} expected={self.action_dim}"
            )
        all_shapes = shape_meta.get("all_shapes", {})
        if isinstance(all_shapes, dict):
            for k in self.lowdim_keys:
                if k in all_shapes:
                    got = int(all_shapes[k][0]) if isinstance(all_shapes[k], (list, tuple)) else int(all_shapes[k])
                    exp = int(self.state_key_dims[k])
                    if got != exp:
                        raise ValueError(f"robomimic ckpt obs dim mismatch for {k}: ckpt={got}, expected={exp}")

    def _adopt_ckpt_lowdim_order_if_needed(self, raw_ckpt):
        shape_meta = raw_ckpt.get("shape_metadata", {})
        all_shapes = shape_meta.get("all_shapes", {})
        ckpt_keys = None
        if isinstance(all_shapes, dict) and len(all_shapes) > 0:
            ckpt_keys = [str(k) for k in all_shapes.keys()]
        elif shape_meta.get("all_obs_keys", None) is not None:
            ckpt_keys = [str(k) for k in shape_meta.get("all_obs_keys", [])]
        else:
            try:
                config = json.loads(raw_ckpt["config"])
                ckpt_keys = [
                    str(k)
                    for k in config["observation"]["modalities"]["obs"]["low_dim"]
                ]
            except Exception:
                ckpt_keys = None

        if ckpt_keys is None or ckpt_keys == list(self.lowdim_keys):
            return

        cur_keys = list(self.lowdim_keys)
        if set(ckpt_keys) != set(cur_keys):
            raise ValueError(
                "robomimic ckpt lowdim keys mismatch in set comparison: "
                f"ckpt={ckpt_keys}, current={cur_keys}"
            )

        new_dims = OrderedDict()
        for k in ckpt_keys:
            if k in all_shapes:
                got = (
                    int(all_shapes[k][0])
                    if isinstance(all_shapes[k], (list, tuple))
                    else int(all_shapes[k])
                )
            else:
                got = int(self.state_key_dims[k])
            exp = int(self.state_key_dims[k])
            if got != exp:
                raise ValueError(
                    f"robomimic ckpt obs dim mismatch while reordering {k}: ckpt={got}, expected={exp}"
                )
            new_dims[k] = exp

        self.lowdim_keys = list(new_dims.keys())
        self.state_key_dims = new_dims
        self.obs_encoder.lowdim_keys = list(new_dims.keys())
        self.obs_encoder.state_key_dims = new_dims

    def load_robomimic_ckpt(self, ckpt_path, weight_source="ema", strict=True):
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self._adopt_ckpt_lowdim_order_if_needed(raw)
        self._validate_against_robomimic_ckpt(raw)
        state = self._select_robomimic_state(raw, weight_source=weight_source)
        noise_state = {}
        for k, v in state.items():
            if k.startswith("policy.obs_encoder."):
                continue
            if k.startswith("policy.noise_pred_net."):
                noise_state[k.split("policy.noise_pred_net.", 1)[1]] = v
        missing, unexpected = self.noise_pred_net.load_state_dict(noise_state, strict=bool(strict))
        if strict and (len(missing) > 0 or len(unexpected) > 0):
            raise RuntimeError(
                f"noise_pred_net strict load mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
            )

    def _split_state(self, state):
        # DSRL RL rollout paths may pass a single-step state as [B, Do].
        # Treat that as cond_steps == 1 so the copied sampler / RL code can stay unchanged.
        if state.ndim == 2:
            state = state[:, None, :]
        if state.ndim != 3:
            raise ValueError(f"cond['state'] must be [B,To,Do] or [B,Do], got {tuple(state.shape)}")
        if int(state.shape[1]) != self.cond_steps:
            raise ValueError(f"cond_steps mismatch: got {int(state.shape[1])}, expected {self.cond_steps}")
        if int(state.shape[2]) != self.obs_dim:
            raise ValueError(f"obs_dim mismatch: got {int(state.shape[2])}, expected {self.obs_dim}")
        out = OrderedDict()
        start = 0
        for key in self.lowdim_keys:
            dim = int(self.state_key_dims[key])
            out[key] = state[:, :, start:start + dim]
            start += dim
        if start != self.obs_dim:
            raise ValueError(f"state split mismatch: consumed {start}, expected {self.obs_dim}")
        return out

    def forward(
        self,
        x,
        time,
        cond,
        **kwargs,
    ):
        if "state" not in cond:
            raise ValueError("cond must contain key 'state'")
        if x.ndim != 3:
            raise ValueError(f"x must be [B,H,A], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.horizon_steps or int(x.shape[2]) != self.action_dim:
            raise ValueError(
                f"action chunk shape mismatch: expected [B,{self.horizon_steps},{self.action_dim}], got {tuple(x.shape)}"
            )

        obs_dict = self._split_state(cond["state"])
        obs_features = self.obs_encoder(obs_dict)
        global_cond = obs_features.flatten(start_dim=1)
        return self.noise_pred_net(
            sample=x,
            timestep=time,
            global_cond=global_cond,
        )
