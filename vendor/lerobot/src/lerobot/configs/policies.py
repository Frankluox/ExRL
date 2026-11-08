from __future__ import annotations

import abc
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim.optimizers import OptimizerConfig
from lerobot.utils.constants import ACTION, CONFIG_NAME, OBS_STATE
from lerobot.utils.utils import auto_select_torch_device, is_amp_available, is_torch_device_available

T = TypeVar("T", bound="PreTrainedConfig")


def _encode_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {k: _encode_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _encode_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_jsonable(v) for v in value]
    return value


def _decode_policy_features(features: dict[str, Any] | None) -> dict[str, PolicyFeature]:
    out: dict[str, PolicyFeature] = {}
    for key, value in (features or {}).items():
        if isinstance(value, PolicyFeature):
            out[key] = value
        else:
            out[key] = PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
    return out


@dataclass
class PreTrainedConfig(abc.ABC):
    _registry: ClassVar[dict[str, type["PreTrainedConfig"]]] = {}
    _choice_name: ClassVar[str | None] = None

    n_obs_steps: int = 1
    input_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    device: str | None = None
    use_amp: bool = False
    use_peft: bool = False
    push_to_hub: bool = False
    repo_id: str | None = None
    private: bool | None = None
    tags: list[str] | None = None
    license: str | None = None
    pretrained_path: Path | None = None

    @classmethod
    def register_subclass(cls, name: str):
        def decorator(subclass: type["PreTrainedConfig"]) -> type["PreTrainedConfig"]:
            cls._registry[name] = subclass
            subclass._choice_name = name
            return subclass

        return decorator

    @classmethod
    def get_choice_name(cls, subclass: type["PreTrainedConfig"]) -> str | None:
        return getattr(subclass, "_choice_name", None)

    def __post_init__(self) -> None:
        self.input_features = _decode_policy_features(self.input_features)
        self.output_features = _decode_policy_features(self.output_features)
        if not self.device or not is_torch_device_available(self.device):
            self.device = auto_select_torch_device().type
        if self.use_amp and not is_amp_available(self.device):
            self.use_amp = False
        if self.pretrained_path is not None and not isinstance(self.pretrained_path, Path):
            self.pretrained_path = Path(self.pretrained_path)

    @property
    def type(self) -> str:
        choice_name = self.get_choice_name(self.__class__)
        if choice_name is None:
            raise ValueError(f"Unregistered config class: {self.__class__.__name__}")
        return choice_name

    @property
    @abc.abstractmethod
    def observation_delta_indices(self) -> list | None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_delta_indices(self) -> list | None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reward_delta_indices(self) -> list | None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        for key, feature in (self.input_features or {}).items():
            if feature.type is FeatureType.STATE and key == OBS_STATE:
                return feature
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        for _, feature in (self.input_features or {}).items():
            if feature.type is FeatureType.ENV:
                return feature
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {
            key: feature
            for key, feature in (self.input_features or {}).items()
            if feature.type is FeatureType.VISUAL
        }

    @property
    def action_feature(self) -> PolicyFeature | None:
        for key, feature in (self.output_features or {}).items():
            if feature.type is FeatureType.ACTION and key == ACTION:
                return feature
        return None

    def to_dict(self) -> dict[str, Any]:
        payload = _encode_jsonable(self)
        payload["type"] = self.type
        return payload

    def _save_pretrained(self, save_directory: Path) -> None:
        save_directory.mkdir(parents=True, exist_ok=True)
        with open(save_directory / CONFIG_NAME, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def save_pretrained(self, save_directory: str | Path) -> None:
        self._save_pretrained(Path(save_directory))

    @classmethod
    def from_pretrained(cls: type[T], pretrained_name_or_path: str | Path, **policy_kwargs: Any) -> T:
        config_path = Path(pretrained_name_or_path)
        if config_path.is_dir():
            config_path = config_path / CONFIG_NAME
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        config_type = payload.pop("type", None)
        if config_type is None:
            raise ValueError(f"config missing 'type': {config_path}")
        subclass = cls._registry.get(config_type)
        if subclass is None:
            raise ValueError(f"unknown config type '{config_type}' in {config_path}")

        payload["input_features"] = _decode_policy_features(payload.get("input_features"))
        payload["output_features"] = _decode_policy_features(payload.get("output_features"))
        payload.update(policy_kwargs)
        return subclass(**payload)
