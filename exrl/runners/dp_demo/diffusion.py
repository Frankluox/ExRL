"""Low-dimensional diffusion policy with DDPM and DDIM sampling.

The MLP and checkpoint layout follow DPPO / DSRL.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _mish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tanh(torch.nn.functional.softplus(x))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        if half_dim <= 0:
            raise ValueError(f"invalid sinusoidal dim={self.dim}")
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=x.dtype) * (-emb))
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


@dataclass
class LinearLayerPack:
    weight: torch.Tensor
    bias: torch.Tensor
    norm_weight: Optional[torch.Tensor] = None
    norm_bias: Optional[torch.Tensor] = None


class ParsedMLP(nn.Module):
    def __init__(
        self, layers: List[LinearLayerPack], apply_mish_on_hidden: bool = True
    ):
        super().__init__()
        if len(layers) == 0:
            raise ValueError("ParsedMLP requires at least one linear layer")
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.has_norm: List[bool] = []
        self.apply_mish_on_hidden = bool(apply_mish_on_hidden)

        for pack in layers:
            out_dim, in_dim = int(pack.weight.shape[0]), int(pack.weight.shape[1])
            lin = nn.Linear(in_dim, out_dim)
            with torch.no_grad():
                lin.weight.copy_(pack.weight)
                lin.bias.copy_(pack.bias)
            self.layers.append(lin)

            if pack.norm_weight is not None and pack.norm_bias is not None:
                ln = nn.LayerNorm(out_dim)
                with torch.no_grad():
                    ln.weight.copy_(pack.norm_weight)
                    ln.bias.copy_(pack.norm_bias)
                self.norms.append(ln)
                self.has_norm.append(True)
            else:
                self.norms.append(nn.Identity())
                self.has_norm.append(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = len(self.layers) - 1
        for i, lin in enumerate(self.layers):
            x = lin(x)
            x = self.norms[i](x)
            if self.apply_mish_on_hidden and i != last:
                x = _mish(x)
        return x


class ParsedResidualBlock(nn.Module):
    """
    Match DSRL TwoLayerPreActivationResNetLinear:
        x -> (norm?) -> Mish -> l1 -> (norm?) -> Mish -> l2 -> + x
    """

    def __init__(
        self,
        l1_w: torch.Tensor,
        l1_b: torch.Tensor,
        l2_w: torch.Tensor,
        l2_b: torch.Tensor,
        norm1_w: Optional[torch.Tensor] = None,
        norm1_b: Optional[torch.Tensor] = None,
        norm2_w: Optional[torch.Tensor] = None,
        norm2_b: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        h1_out, h1_in = int(l1_w.shape[0]), int(l1_w.shape[1])
        h2_out, h2_in = int(l2_w.shape[0]), int(l2_w.shape[1])
        if not (h1_in == h1_out == h2_in == h2_out):
            raise ValueError(
                f"[DSRL-CKPT] residual block dim mismatch: "
                f"l1={tuple(l1_w.shape)} l2={tuple(l2_w.shape)}"
            )
        self.hidden_dim = h1_in
        self.l1 = nn.Linear(h1_in, h1_out)
        self.l2 = nn.Linear(h2_in, h2_out)
        with torch.no_grad():
            self.l1.weight.copy_(l1_w)
            self.l1.bias.copy_(l1_b)
            self.l2.weight.copy_(l2_w)
            self.l2.bias.copy_(l2_b)

        if norm1_w is not None and norm1_b is not None:
            self.norm1 = nn.LayerNorm(self.hidden_dim)
            with torch.no_grad():
                self.norm1.weight.copy_(norm1_w)
                self.norm1.bias.copy_(norm1_b)
        else:
            self.norm1 = None

        if norm2_w is not None and norm2_b is not None:
            self.norm2 = nn.LayerNorm(self.hidden_dim)
            with torch.no_grad():
                self.norm2.weight.copy_(norm2_w)
                self.norm2.bias.copy_(norm2_b)
        else:
            self.norm2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.l1(_mish(x))
        if self.norm2 is not None:
            x = self.norm2(x)
        x = self.l2(_mish(x))
        return x + x_in


class ParsedResidualMLP(nn.Module):
    """
    Match DSRL ResidualMLP:
        in_linear -> residual_blocks -> out_linear
    """

    def __init__(
        self,
        in_w: torch.Tensor,
        in_b: torch.Tensor,
        blocks: List[ParsedResidualBlock],
        out_w: torch.Tensor,
        out_b: torch.Tensor,
    ):
        super().__init__()
        h_out, h_in = int(in_w.shape[0]), int(in_w.shape[1])
        out_out, out_in = int(out_w.shape[0]), int(out_w.shape[1])
        if out_in != h_out:
            raise ValueError(
                f"[DSRL-CKPT] residual out layer input mismatch: out_in={out_in} hidden={h_out}"
            )
        self.input_dim = h_in
        self.hidden_dim = h_out
        self.output_dim = out_out
        self.num_blocks = len(blocks)

        self.in_linear = nn.Linear(h_in, h_out)
        with torch.no_grad():
            self.in_linear.weight.copy_(in_w)
            self.in_linear.bias.copy_(in_b)

        self.blocks = nn.ModuleList(blocks)
        for b in self.blocks:
            if getattr(b, "hidden_dim", None) != h_out:
                raise ValueError(
                    "[DSRL-CKPT] inconsistent hidden dim across residual blocks"
                )

        self.out_linear = nn.Linear(out_in, out_out)
        with torch.no_grad():
            self.out_linear.weight.copy_(out_w)
            self.out_linear.bias.copy_(out_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_linear(x)
        for b in self.blocks:
            x = b(x)
        x = self.out_linear(x)
        return x


class ParsedTimeEmbedding(nn.Module):
    def __init__(
        self, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor
    ):
        super().__init__()
        in1 = int(w1.shape[1])
        out1 = int(w1.shape[0])
        in2 = int(w2.shape[1])
        out2 = int(w2.shape[0])

        self.pos = SinusoidalPosEmb(in1)
        self.fc1 = nn.Linear(in1, out1)
        self.fc2 = nn.Linear(in2, out2)
        with torch.no_grad():
            self.fc1.weight.copy_(w1)
            self.fc1.bias.copy_(b1)
            self.fc2.weight.copy_(w2)
            self.fc2.bias.copy_(b2)
        self.time_dim = out2

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.pos(t)
        x = self.fc1(x)
        x = _mish(x)
        x = self.fc2(x)
        return x


@dataclass
class BackboneMeta:
    action_dim: int
    horizon_steps: int
    cond_dim: int
    cond_steps: int
    obs_dim: int
    time_dim: int
    mlp_dims: List[int]
    cond_mlp_dims: List[int]


class DiffusionPolicy(nn.Module):
    """
    Reconstructed low-dim DSRL DiffusionMLP from checkpoint tensors.
    """

    def __init__(
        self,
        action_dim: int,
        horizon_steps: int,
        cond_dim: int,
        cond_steps: int,
        time_embed: ParsedTimeEmbedding,
        mlp_mean: ParsedMLP,
        cond_mlp: Optional[ParsedMLP] = None,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.horizon_steps = int(horizon_steps)
        self.cond_dim = int(cond_dim)
        self.cond_steps = int(cond_steps)
        self.obs_dim = int(cond_dim // max(cond_steps, 1))

        self.time_embedding = time_embed
        self.cond_mlp = cond_mlp
        self.mlp_mean = mlp_mean

    @staticmethod
    def _strip_prefix_if_needed(
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        keys = list(state_dict.keys())
        if len(keys) == 0:
            return state_dict

        candidates = ["network.", "model.network.", "module.network."]
        for p in candidates:
            if sum(1 for k in keys if k.startswith(p)) >= max(1, len(keys) // 2):
                out = {}
                for k, v in state_dict.items():
                    nk = k[len(p) :] if k.startswith(p) else k
                    out[nk] = v
                return out
        return dict(state_dict)

    @staticmethod
    def _find_two_time_linears(
        state_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pats = re.compile(r"^time_embedding\.(\d+)\.weight$")
        idxs = []
        for k in state_dict.keys():
            m = pats.match(k)
            if m:
                idxs.append(int(m.group(1)))
        idxs = sorted(set(idxs))
        if len(idxs) < 2:
            raise ValueError("[DSRL-CKPT] cannot locate time_embedding linear weights")

        # Pick first two linear indices in sequence (official usually 1 and 3)
        i1, i2 = idxs[0], idxs[1]
        k1w, k1b = f"time_embedding.{i1}.weight", f"time_embedding.{i1}.bias"
        k2w, k2b = f"time_embedding.{i2}.weight", f"time_embedding.{i2}.bias"
        for k in [k1w, k1b, k2w, k2b]:
            if k not in state_dict:
                raise ValueError(f"[DSRL-CKPT] missing key: {k}")
        return (
            state_dict[k1w].float(),
            state_dict[k1b].float(),
            state_dict[k2w].float(),
            state_dict[k2b].float(),
        )

    @staticmethod
    def _build_residual_mlp_if_present(
        state_dict: Dict[str, torch.Tensor], prefix: str
    ) -> Optional[ParsedResidualMLP]:
        """
        Detect and reconstruct DSRL ResidualMLP layout:
            {prefix}.layers.0.(weight|bias)
            {prefix}.layers.<i>.l1.(weight|bias), {prefix}.layers.<i>.l2.(weight|bias)
            {prefix}.layers.<j>.(weight|bias)   # final projection
        """
        # quick detect
        has_res = any(
            k.startswith(prefix + ".layers.") and ".l1.weight" in k
            for k in state_dict.keys()
        )
        if not has_res:
            return None

        in_wk = f"{prefix}.layers.0.weight"
        in_bk = f"{prefix}.layers.0.bias"
        if in_wk not in state_dict or in_bk not in state_dict:
            raise ValueError(
                f"[DSRL-CKPT] residual mlp missing input linear: {in_wk}/{in_bk}"
            )
        in_w = state_dict[in_wk].float()
        in_b = state_dict[in_bk].float()

        # residual block indices
        blk_pat = re.compile(rf"^{re.escape(prefix)}\.layers\.(\d+)\.l1\.weight$")
        blk_idxs = sorted(
            {
                int(blk_pat.match(k).group(1))
                for k in state_dict.keys()
                if blk_pat.match(k)
            }
        )
        blocks: List[ParsedResidualBlock] = []
        for i in blk_idxs:
            l1_wk = f"{prefix}.layers.{i}.l1.weight"
            l1_bk = f"{prefix}.layers.{i}.l1.bias"
            l2_wk = f"{prefix}.layers.{i}.l2.weight"
            l2_bk = f"{prefix}.layers.{i}.l2.bias"
            for k in [l1_wk, l1_bk, l2_wk, l2_bk]:
                if k not in state_dict:
                    raise ValueError(f"[DSRL-CKPT] residual block missing key: {k}")
            n1_w = state_dict.get(f"{prefix}.layers.{i}.norm1.weight")
            n1_b = state_dict.get(f"{prefix}.layers.{i}.norm1.bias")
            n2_w = state_dict.get(f"{prefix}.layers.{i}.norm2.weight")
            n2_b = state_dict.get(f"{prefix}.layers.{i}.norm2.bias")
            blocks.append(
                ParsedResidualBlock(
                    l1_w=state_dict[l1_wk].float(),
                    l1_b=state_dict[l1_bk].float(),
                    l2_w=state_dict[l2_wk].float(),
                    l2_b=state_dict[l2_bk].float(),
                    norm1_w=None if n1_w is None else n1_w.float(),
                    norm1_b=None if n1_b is None else n1_b.float(),
                    norm2_w=None if n2_w is None else n2_w.float(),
                    norm2_b=None if n2_b is None else n2_b.float(),
                )
            )

        # output projection index: largest numeric layers.<idx>.weight without submodule suffix
        out_pat = re.compile(rf"^{re.escape(prefix)}\.layers\.(\d+)\.weight$")
        out_candidates = sorted(
            {
                int(out_pat.match(k).group(1))
                for k in state_dict.keys()
                if out_pat.match(k)
            }
        )
        # remove input index 0
        out_candidates = [i for i in out_candidates if i != 0]
        if len(out_candidates) == 0:
            raise ValueError(
                f"[DSRL-CKPT] residual mlp missing output projection under {prefix}.layers.*"
            )
        out_idx = max(out_candidates)
        out_wk = f"{prefix}.layers.{out_idx}.weight"
        out_bk = f"{prefix}.layers.{out_idx}.bias"
        if out_wk not in state_dict or out_bk not in state_dict:
            raise ValueError(
                f"[DSRL-CKPT] residual mlp missing output linear: {out_wk}/{out_bk}"
            )

        return ParsedResidualMLP(
            in_w=in_w,
            in_b=in_b,
            blocks=blocks,
            out_w=state_dict[out_wk].float(),
            out_b=state_dict[out_bk].float(),
        )

    @staticmethod
    def _collect_linear_packs(
        state_dict: Dict[str, torch.Tensor], prefix: str
    ) -> List[LinearLayerPack]:
        packs: List[LinearLayerPack] = []

        # Style A: moduleList.<i>.linear_1.weight
        pa = re.compile(rf"^{re.escape(prefix)}\.moduleList\.(\d+)\.linear_1\.weight$")
        idxs_a = sorted(
            {int(pa.match(k).group(1)) for k in state_dict.keys() if pa.match(k)}
        )
        if idxs_a:
            for i in idxs_a:
                wk = f"{prefix}.moduleList.{i}.linear_1.weight"
                bk = f"{prefix}.moduleList.{i}.linear_1.bias"
                if wk not in state_dict or bk not in state_dict:
                    raise ValueError(
                        f"[DSRL-CKPT] malformed {prefix} moduleList layer index={i}"
                    )
                n_w = state_dict.get(f"{prefix}.moduleList.{i}.norm_1.weight")
                n_b = state_dict.get(f"{prefix}.moduleList.{i}.norm_1.bias")
                packs.append(
                    LinearLayerPack(
                        weight=state_dict[wk].float(),
                        bias=state_dict[bk].float(),
                        norm_weight=None if n_w is None else n_w.float(),
                        norm_bias=None if n_b is None else n_b.float(),
                    )
                )
            return packs

        # Style B: layers.<i>.weight / layers.<i>.bias
        pb = re.compile(rf"^{re.escape(prefix)}\.layers\.(\d+)\.weight$")
        idxs_b = sorted(
            {int(pb.match(k).group(1)) for k in state_dict.keys() if pb.match(k)}
        )
        if idxs_b:
            for i in idxs_b:
                wk = f"{prefix}.layers.{i}.weight"
                bk = f"{prefix}.layers.{i}.bias"
                if wk not in state_dict or bk not in state_dict:
                    continue
                n_w = state_dict.get(f"{prefix}.layers.{i}.norm.weight")
                n_b = state_dict.get(f"{prefix}.layers.{i}.norm.bias")
                packs.append(
                    LinearLayerPack(
                        weight=state_dict[wk].float(),
                        bias=state_dict[bk].float(),
                        norm_weight=None if n_w is None else n_w.float(),
                        norm_bias=None if n_b is None else n_b.float(),
                    )
                )
            if packs:
                return packs

        # Style C: sequential numeric <i>.weight / <i>.bias
        pc = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.weight$")
        idxs_c = sorted(
            {int(pc.match(k).group(1)) for k in state_dict.keys() if pc.match(k)}
        )
        if idxs_c:
            for i in idxs_c:
                wk = f"{prefix}.{i}.weight"
                bk = f"{prefix}.{i}.bias"
                if wk in state_dict and bk in state_dict:
                    packs.append(
                        LinearLayerPack(
                            weight=state_dict[wk].float(), bias=state_dict[bk].float()
                        )
                    )
            if packs:
                return packs

        # Generic fallback: any linear-ish keys under prefix
        generic = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "."):
                continue
            if not k.endswith(".weight"):
                continue
            if ".norm" in k or ".bn" in k or "time_embedding" in k:
                continue
            b = k[:-7] + ".bias"
            if b not in state_dict:
                continue
            generic.append((k, b))
        if not generic:
            return []

        def _sort_key(pair: Tuple[str, str]) -> Tuple[int, str]:
            k = pair[0]
            nums = re.findall(r"(\d+)", k)
            return (int(nums[-1]) if nums else 10**9, k)

        generic.sort(key=_sort_key)
        for wk, bk in generic:
            packs.append(
                LinearLayerPack(
                    weight=state_dict[wk].float(), bias=state_dict[bk].float()
                )
            )
        return packs

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        action_dim: int,
        cond_steps: int = 1,
    ) -> Tuple["DiffusionPolicy", BackboneMeta]:
        sd = cls._strip_prefix_if_needed(state_dict)

        if int(action_dim) <= 0:
            raise ValueError(f"[DSRL-CKPT] invalid action_dim={action_dim}")
        action_dim = int(action_dim)
        cond_steps = int(cond_steps)
        if cond_steps <= 0:
            raise ValueError(f"[DSRL-CKPT] cond_steps must be > 0, got {cond_steps}")

        w1, b1, w2, b2 = cls._find_two_time_linears(sd)
        time_embed = ParsedTimeEmbedding(w1=w1, b1=b1, w2=w2, b2=b2)

        mlp_res = cls._build_residual_mlp_if_present(sd, "mlp_mean")
        if mlp_res is not None:
            mlp_mean = mlp_res
            output_dim = int(mlp_res.output_dim)
            input_dim = int(mlp_res.input_dim)
            hidden = [int(mlp_res.hidden_dim)] * (int(mlp_res.num_blocks) + 1)
        else:
            mlp_packs = cls._collect_linear_packs(sd, "mlp_mean")
            if len(mlp_packs) == 0:
                raise ValueError("[DSRL-CKPT] cannot find mlp_mean linear layers")
            mlp_mean = ParsedMLP(mlp_packs, apply_mish_on_hidden=True)
            output_dim = int(mlp_packs[-1].weight.shape[0])
            input_dim = int(mlp_packs[0].weight.shape[1])
            hidden = [int(p.weight.shape[0]) for p in mlp_packs[:-1]]

        cond_packs = cls._collect_linear_packs(sd, "cond_mlp")
        cond_mlp = (
            ParsedMLP(cond_packs, apply_mish_on_hidden=True)
            if len(cond_packs) > 0
            else None
        )

        if output_dim % action_dim != 0:
            raise ValueError(
                f"[DSRL-CKPT] output_dim ({output_dim}) not divisible by action_dim ({action_dim})"
            )
        horizon_steps = int(output_dim // action_dim)

        cond_encoded_dim = (
            int(cond_packs[-1].weight.shape[0]) if len(cond_packs) > 0 else None
        )
        if cond_encoded_dim is None:
            cond_dim = int(input_dim - output_dim - time_embed.time_dim)
        else:
            cond_dim = int(cond_packs[0].weight.shape[1])
            expected = int(time_embed.time_dim + output_dim + cond_encoded_dim)
            if input_dim != expected:
                raise ValueError(
                    f"[DSRL-CKPT] first mlp input mismatch: got={input_dim}, expected={expected}"
                )

        if cond_dim <= 0:
            raise ValueError(f"[DSRL-CKPT] inferred cond_dim invalid: {cond_dim}")
        if cond_dim % cond_steps != 0:
            raise ValueError(
                f"[DSRL-CKPT] cond_dim ({cond_dim}) not divisible by cond_steps ({cond_steps})"
            )
        obs_dim = int(cond_dim // cond_steps)

        model = cls(
            action_dim=action_dim,
            horizon_steps=horizon_steps,
            cond_dim=cond_dim,
            cond_steps=cond_steps,
            time_embed=time_embed,
            mlp_mean=mlp_mean,
            cond_mlp=cond_mlp,
        )

        cond_hidden = (
            [int(p.weight.shape[0]) for p in cond_packs] if len(cond_packs) > 0 else []
        )
        meta = BackboneMeta(
            action_dim=action_dim,
            horizon_steps=horizon_steps,
            cond_dim=cond_dim,
            cond_steps=cond_steps,
            obs_dim=obs_dim,
            time_dim=int(time_embed.time_dim),
            mlp_dims=hidden,
            cond_mlp_dims=cond_hidden,
        )
        return model, meta

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        if "state" not in cond:
            raise ValueError("[DSRL-CKPT] cond must contain key 'state'")
        if x.ndim != 3:
            raise ValueError(
                f"[DSRL-CKPT] x must be [B,H,A], got shape={tuple(x.shape)}"
            )

        b, h, a = x.shape
        if h != self.horizon_steps or a != self.action_dim:
            raise ValueError(
                f"[DSRL-CKPT] action shape mismatch, expected [B,{self.horizon_steps},{self.action_dim}], got {tuple(x.shape)}"
            )

        state = cond["state"]
        if state.ndim != 3:
            raise ValueError(
                f"[DSRL-CKPT] state must be [B,To,Do], got shape={tuple(state.shape)}"
            )
        if (
            int(state.shape[1]) != self.cond_steps
            or int(state.shape[2]) != self.obs_dim
        ):
            raise ValueError(
                f"[DSRL-CKPT] state shape mismatch, expected [B,{self.cond_steps},{self.obs_dim}], got {tuple(state.shape)}"
            )

        x_flat = x.reshape(b, -1)
        state_flat = state.reshape(b, -1)
        if self.cond_mlp is not None:
            state_flat = self.cond_mlp(state_flat)

        t = time.reshape(b).to(dtype=x.dtype)
        t_emb = self.time_embedding(t)

        model_in = torch.cat([x_flat, t_emb, state_flat], dim=-1)
        out = self.mlp_mean(model_in)
        return out.reshape(b, self.horizon_steps, self.action_dim)


@dataclass
class DiffusionSamplerConfig:
    denoising_steps: int = 100
    use_ddim: bool = True
    ddim_steps: int = 5
    denoised_clip_value: Optional[float] = None
    randn_clip_value: float = 10.0
    final_action_clip_value: Optional[float] = 1.0
    eps_clip_value: Optional[float] = None


class DiffusionSampler:
    def __init__(self, horizon_steps: int, action_dim: int):
        self.horizon_steps = int(horizon_steps)
        self.action_dim = int(action_dim)
        self._cache = {}

    @staticmethod
    def _cosine_beta_schedule(
        timesteps: int, s: float = 0.008, dtype=torch.float32
    ) -> torch.Tensor:
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, a_min=0.0, a_max=0.999)
        return torch.tensor(betas, dtype=dtype)

    @staticmethod
    def _extract(
        a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        b = int(t.shape[0])
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def _build_schedule(
        self,
        denoising_steps: int,
        use_ddim: bool,
        ddim_steps: int,
        device: torch.device,
    ):
        key = (int(denoising_steps), bool(use_ddim), int(ddim_steps), str(device))
        if key in self._cache:
            return self._cache[key]

        betas = self._cosine_beta_schedule(denoising_steps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), alphas_cumprod[:-1]]
        )
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)
        ddpm_logvar_clipped = torch.log(
            torch.clamp(
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod), min=1e-20
            )
        )
        ddpm_mu_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ddpm_mu_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

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
            ddim_t = (
                torch.arange(0, ddim_steps, device=device, dtype=torch.long) * ratio
            )
            ddim_t = torch.clamp(ddim_t, max=denoising_steps - 1)
            ddim_alphas = alphas_cumprod[ddim_t].clone().to(torch.float32)
            ddim_alphas_prev = torch.cat(
                [
                    torch.tensor([1.0], device=device, dtype=torch.float32),
                    alphas_cumprod[ddim_t[:-1]].to(torch.float32),
                ]
            )
            ddim_sqrt_one_minus_alphas = (1.0 - ddim_alphas) ** 0.5
            ddim_sigmas = torch.zeros_like(ddim_alphas)

            # reverse order for denoising
            sched.update(
                {
                    "ddim_t": torch.flip(ddim_t, [0]),
                    "ddim_alphas": torch.flip(ddim_alphas, [0]),
                    "ddim_alphas_prev": torch.flip(ddim_alphas_prev, [0]),
                    "ddim_sqrt_one_minus_alphas": torch.flip(
                        ddim_sqrt_one_minus_alphas, [0]
                    ),
                    "ddim_sigmas": torch.flip(ddim_sigmas, [0]),
                }
            )

        self._cache[key] = sched
        return sched

    @torch.no_grad()
    def sample(
        self,
        network: nn.Module,
        cond: Dict[str, torch.Tensor],
        cfg: DiffusionSamplerConfig,
        deterministic: bool = True,
        noise_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if "state" not in cond:
            raise ValueError("[DSRL-CKPT] cond requires key 'state'")
        state = cond["state"]
        if state.ndim != 3:
            raise ValueError(
                f"[DSRL-CKPT] cond['state'] must be [B,To,Do], got {tuple(state.shape)}"
            )

        device = state.device
        b = int(state.shape[0])
        sched = self._build_schedule(
            denoising_steps=int(cfg.denoising_steps),
            use_ddim=bool(cfg.use_ddim),
            ddim_steps=int(cfg.ddim_steps),
            device=device,
        )

        if noise_action is None:
            x = torch.randn((b, self.horizon_steps, self.action_dim), device=device)
        else:
            x = noise_action.to(device=device, dtype=state.dtype)
            if tuple(x.shape) != (b, self.horizon_steps, self.action_dim):
                raise ValueError(
                    f"[DSRL-CKPT] noise_action shape mismatch, expected {(b, self.horizon_steps, self.action_dim)}, got {tuple(x.shape)}"
                )

        if cfg.use_ddim:
            t_all = sched["ddim_t"]
        else:
            t_all = list(reversed(range(int(cfg.denoising_steps))))

        for i, t in enumerate(t_all):
            if isinstance(t, torch.Tensor):
                t_val = int(t.item())
            else:
                t_val = int(t)
            t_b = torch.full((b,), t_val, device=device, dtype=torch.long)

            noise = network(x, t_b, cond=cond)

            if cfg.use_ddim:
                index_b = torch.full((b,), i, device=device, dtype=torch.long)
                alpha = self._extract(sched["ddim_alphas"], index_b, x.shape)
                alpha_prev = self._extract(sched["ddim_alphas_prev"], index_b, x.shape)
                sqrt_one_minus_alpha = self._extract(
                    sched["ddim_sqrt_one_minus_alphas"], index_b, x.shape
                )
                x_recon = (x - sqrt_one_minus_alpha * noise) / torch.sqrt(alpha)
            else:
                x_recon = (
                    self._extract(sched["sqrt_recip_alphas_cumprod"], t_b, x.shape) * x
                    - self._extract(sched["sqrt_recipm1_alphas_cumprod"], t_b, x.shape)
                    * noise
                )

            if cfg.denoised_clip_value is not None:
                x_recon = torch.clamp(
                    x_recon,
                    -float(cfg.denoised_clip_value),
                    float(cfg.denoised_clip_value),
                )
                if cfg.use_ddim:
                    noise = (x - torch.sqrt(alpha) * x_recon) / torch.clamp(
                        sqrt_one_minus_alpha, min=1e-8
                    )

            if cfg.use_ddim and cfg.eps_clip_value is not None:
                noise = torch.clamp(
                    noise, -float(cfg.eps_clip_value), float(cfg.eps_clip_value)
                )

            if cfg.use_ddim:
                sigma = self._extract(sched["ddim_sigmas"], index_b, x.shape)
                dir_xt = (
                    torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma**2, min=0.0))
                    * noise
                )
                mean = torch.sqrt(alpha_prev) * x_recon + dir_xt
                std = sigma
            else:
                mean = (
                    self._extract(sched["ddpm_mu_coef1"], t_b, x.shape) * x_recon
                    + self._extract(sched["ddpm_mu_coef2"], t_b, x.shape) * x
                )
                logvar = self._extract(sched["ddpm_logvar_clipped"], t_b, x.shape)
                std = torch.exp(0.5 * logvar)
                if t_val == 0:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clamp(std, min=1e-3)

            if deterministic:
                step_noise = torch.zeros_like(x)
            else:
                step_noise = torch.randn_like(x)
                if cfg.randn_clip_value is not None:
                    step_noise = torch.clamp(
                        step_noise,
                        -float(cfg.randn_clip_value),
                        float(cfg.randn_clip_value),
                    )

            x = mean + std * step_noise

        if cfg.final_action_clip_value is not None:
            x = torch.clamp(
                x,
                -float(cfg.final_action_clip_value),
                float(cfg.final_action_clip_value),
            )
        return x
