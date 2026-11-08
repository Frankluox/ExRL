from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import ACTION

from .converters import from_tensor_to_numpy, to_tensor
from .core import EnvTransition, PolicyAction, TransitionKey
from .pipeline import ProcessorStep, ProcessorStepRegistry, RobotObservation


def _restore_features(features: dict[str, Any]) -> dict[str, PolicyFeature]:
    restored: dict[str, PolicyFeature] = {}
    for key, feature in (features or {}).items():
        if isinstance(feature, PolicyFeature):
            restored[key] = feature
        else:
            restored[key] = PolicyFeature(type=FeatureType(feature["type"]), shape=tuple(feature["shape"]))
    return restored


def _restore_norm_map(norm_map: dict[Any, Any]) -> dict[FeatureType, NormalizationMode]:
    restored: dict[FeatureType, NormalizationMode] = {}
    for feature_type, mode in (norm_map or {}).items():
        restored[FeatureType(feature_type)] = NormalizationMode(mode)
    return restored


@dataclass
class _NormalizationMixin:
    features: dict[str, PolicyFeature]
    norm_map: dict[FeatureType, NormalizationMode]
    stats: dict[str, dict[str, Any]] | None = None
    device: torch.device | str | None = None
    eps: float = 1e-8
    _tensor_stats: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.features = _restore_features(self.features)
        self.norm_map = _restore_norm_map(self.norm_map)
        self.stats = self.stats or {}
        self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=torch.float32)

    def state_dict(self) -> dict[str, torch.Tensor]:
        flat: dict[str, torch.Tensor] = {}
        for key, sub_stats in self._tensor_stats.items():
            for stat_name, tensor in sub_stats.items():
                flat[f"{key}.{stat_name}"] = tensor.detach().cpu()
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if self.stats:
            self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=torch.float32)
            return
        self._tensor_stats = {}
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.rsplit(".", 1)
            self._tensor_stats.setdefault(key, {})[stat_name] = tensor.to(self.device)
        self.stats = {
            key: {stat_name: from_tensor_to_numpy(tensor) for stat_name, tensor in sub.items()}
            for key, sub in self._tensor_stats.items()
        }

    def get_config(self) -> dict[str, Any]:
        return {
            "features": {
                key: {"type": feature.type.value, "shape": list(feature.shape)}
                for key, feature in self.features.items()
            },
            "norm_map": {feature_type.value: mode.value for feature_type, mode in self.norm_map.items()},
            "eps": self.eps,
        }

    def _apply_transform(self, tensor: torch.Tensor, key: str, feature_type: FeatureType, inverse: bool) -> torch.Tensor:
        mode = self.norm_map.get(feature_type, NormalizationMode.IDENTITY)
        if mode == NormalizationMode.IDENTITY or key not in self._tensor_stats:
            return tensor
        stats = self._tensor_stats[key]
        mean = stats.get("mean")
        std = stats.get("std")
        if mean is None or std is None:
            return tensor
        mean = mean.to(device=tensor.device, dtype=tensor.dtype)
        std = std.to(device=tensor.device, dtype=tensor.dtype)
        std = torch.where(std.abs() < self.eps, torch.ones_like(std), std)
        if inverse:
            return tensor * std + mean
        return (tensor - mean) / std

    def _normalize_observation(self, observation: RobotObservation, inverse: bool) -> dict[str, torch.Tensor]:
        updated = dict(observation)
        for key, feature in self.features.items():
            if feature.type == FeatureType.ACTION or key not in updated:
                continue
            updated[key] = self._apply_transform(torch.as_tensor(updated[key]), key, feature.type, inverse)
        return updated

    def _normalize_action(self, action: torch.Tensor, inverse: bool) -> torch.Tensor:
        return self._apply_transform(action, ACTION, FeatureType.ACTION, inverse)


@ProcessorStepRegistry.register("normalizer_processor")
@dataclass
class NormalizerProcessorStep(_NormalizationMixin, ProcessorStep):
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        updated = dict(transition)
        observation = updated.get(TransitionKey.OBSERVATION)
        if observation is not None:
            updated[TransitionKey.OBSERVATION] = self._normalize_observation(observation, inverse=False)
        action = updated.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            updated[TransitionKey.ACTION] = self._normalize_action(action, inverse=False)
        return updated

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("unnormalizer_processor")
@dataclass
class UnnormalizerProcessorStep(_NormalizationMixin, ProcessorStep):
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        updated = dict(transition)
        observation = updated.get(TransitionKey.OBSERVATION)
        if observation is not None:
            updated[TransitionKey.OBSERVATION] = self._normalize_observation(observation, inverse=True)
        action = updated.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            updated[TransitionKey.ACTION] = self._normalize_action(action, inverse=True)
        return updated

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def hotswap_stats(processor: _NormalizationMixin, stats: dict[str, dict[str, Any]]) -> _NormalizationMixin:
    processor.stats = stats
    processor._tensor_stats = to_tensor(stats, device=processor.device, dtype=torch.float32)
    return processor
