from __future__ import annotations

import abc
import builtins
from pathlib import Path
from typing import TypeVar

from safetensors.torch import load_model as load_model_as_safetensor, save_model as save_model_as_safetensor
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.constants import MODEL_FILE_NAME

T = TypeVar("T", bound="PreTrainedPolicy")


class PreTrainedPolicy(nn.Module, abc.ABC):
    config_class = None
    name = None

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        del inputs, kwargs
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise ValueError(f"Expected PreTrainedConfig, got {type(config)}")
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None):
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")
        if not getattr(cls, "name", None):
            raise TypeError(f"Class {cls.__name__} must define 'name'")

    def _save_pretrained(self, save_directory: Path) -> None:
        save_directory.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(save_directory / MODEL_FILE_NAME))

    def save_pretrained(self, save_directory: str | Path) -> None:
        self._save_pretrained(Path(save_directory))

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        model_file = Path(pretrained_name_or_path)
        if model_file.is_dir():
            model_file = model_file / MODEL_FILE_NAME
        if not model_file.exists():
            raise FileNotFoundError(f"model weights not found: {model_file}")
        load_model_as_safetensor(instance, str(model_file), strict=strict, device=config.device)
        instance.to(config.device)
        instance.eval()
        return instance

    @abc.abstractmethod
    def get_optim_params(self):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError

    @abc.abstractmethod
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError
