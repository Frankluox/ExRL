from __future__ import annotations

import abc
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import torch

OptimizerParams = Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]]


@dataclass
class OptimizerConfig(abc.ABC):
    lr: float
    weight_decay: float
    grad_clip_norm: float

    @abc.abstractmethod
    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        raise NotImplementedError


@dataclass
class AdamWConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    grad_clip_norm: float = 10.0

    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.AdamW(params, **kwargs)
