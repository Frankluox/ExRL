"""Shared actor and critic networks over discrete execution lengths."""

import math
from typing import Dict, List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation_module(name: str):
    n = str(name).lower()
    if n == "relu":
        return nn.ReLU(inplace=True)
    if n == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
        use_layernorm: bool = False,
    ):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if bool(use_layernorm):
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(_activation_module(activation))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiscreteActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.backbone = MLP(
            self.obs_dim,
            self.action_dim,
            hidden_dims,
            activation=activation,
            use_layernorm=use_layernorm,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs)

    def distribution(self, obs: torch.Tensor, min_action_id: int = 0):
        logits = self.forward(obs)
        min_action_id = int(min_action_id)
        if min_action_id < 0 or min_action_id >= self.action_dim:
            if min_action_id != 0:
                raise ValueError(f"min_action_id out of range: {min_action_id}")
        if min_action_id > 0:
            logits = logits.clone()
            logits[:, :min_action_id] = -1e9
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        return logits, probs, log_probs

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, deterministic: bool = False, min_action_id: int = 0
    ) -> torch.Tensor:
        logits, _, _ = self.distribution(obs, min_action_id=min_action_id)
        if deterministic:
            return torch.argmax(logits, dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()


class TwinQCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.q1 = MLP(
            self.obs_dim,
            self.action_dim,
            hidden_dims,
            activation=activation,
            use_layernorm=use_layernorm,
        )
        self.q2 = MLP(
            self.obs_dim,
            self.action_dim,
            hidden_dims,
            activation=activation,
            use_layernorm=use_layernorm,
        )

    def forward(self, obs: torch.Tensor):
        return self.q1(obs), self.q2(obs)
