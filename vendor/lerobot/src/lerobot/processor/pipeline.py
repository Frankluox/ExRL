from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Sequence, TypeVar, cast

import torch
from safetensors.torch import load_file, save_file

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from .core import EnvTransition, PolicyAction, RobotObservation, TransitionKey

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class ProcessorStepRegistry:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str | None = None):
        def decorator(step_class: type) -> type:
            reg_name = name or step_class.__name__
            cls._registry[reg_name] = step_class
            step_class._registry_name = reg_name
            return step_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._registry:
            raise KeyError(f"Processor step '{name}' not registered")
        return cls._registry[name]


class ProcessorStep(ABC):
    _current_transition: EnvTransition | None = None

    @abstractmethod
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        del state

    def reset(self) -> None:
        return None

    @abstractmethod
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        raise NotImplementedError


class ObservationProcessorStep(ProcessorStep):
    @abstractmethod
    def observation(self, observation: RobotObservation) -> RobotObservation:
        raise NotImplementedError

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        updated = dict(transition)
        observation = updated.get(TransitionKey.OBSERVATION, None)
        if observation is not None:
            updated[TransitionKey.OBSERVATION] = self.observation(observation)
        return cast(EnvTransition, updated)


class PolicyActionProcessorStep(ProcessorStep):
    @abstractmethod
    def action(self, action: PolicyAction) -> PolicyAction:
        raise NotImplementedError

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        updated = dict(transition)
        action = updated.get(TransitionKey.ACTION, None)
        if isinstance(action, torch.Tensor):
            updated[TransitionKey.ACTION] = self.action(action)
        return cast(EnvTransition, updated)


class ComplementaryDataProcessorStep(ProcessorStep):
    @abstractmethod
    def complementary_data(self, complementary_data: dict) -> dict:
        raise NotImplementedError

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        updated = dict(transition)
        comp = updated.get(TransitionKey.COMPLEMENTARY_DATA, None)
        if comp is not None:
            updated[TransitionKey.COMPLEMENTARY_DATA] = self.complementary_data(comp)
        return cast(EnvTransition, updated)


@dataclass
class DataProcessorPipeline(Generic[TInput, TOutput]):
    steps: Sequence[ProcessorStep] = field(default_factory=list)
    name: str = "processor"
    to_transition: Callable[[TInput], EnvTransition] = field(
        default_factory=lambda: cast(Callable[[TInput], EnvTransition], batch_to_transition),
        repr=False,
    )
    to_output: Callable[[EnvTransition], TOutput] = field(
        default_factory=lambda: cast(Callable[[EnvTransition], TOutput], transition_to_batch),
        repr=False,
    )

    def __call__(self, value: TInput) -> TOutput:
        transition = self.to_transition(value)
        for step in self.steps:
            transition = step(transition)
        return self.to_output(transition)

    def reset(self) -> None:
        for step in self.steps:
            step.reset()

    def save_pretrained(self, save_directory: str | Path, config_filename: str | None = None) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        config_filename = config_filename or f"{self.name}.json"
        step_payloads = []
        payload = {
            "name": self.name,
            "steps": step_payloads,
        }
        with open(save_directory / config_filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        for idx, step in enumerate(self.steps):
            registry_name = getattr(step, "_registry_name", step.__class__.__name__)
            step_entry = {
                "registry_name": registry_name,
                "config": step.get_config(),
            }
            state = {key: tensor.detach().cpu() for key, tensor in step.state_dict().items()}
            if state:
                state_filename = f"{self.name}_step_{idx}_{registry_name}.safetensors"
                save_file(state, str(save_directory / state_filename))
                step_entry["state_file"] = state_filename
            step_payloads.append(step_entry)

        with open(save_directory / config_filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        config_filename: str,
        overrides: dict[str, Any] | None = None,
        to_transition: Callable[[Any], EnvTransition] | None = None,
        to_output: Callable[[EnvTransition], Any] | None = None,
    ) -> "DataProcessorPipeline":
        save_directory = Path(pretrained_model_name_or_path)
        config_path = save_directory / config_filename
        if not config_path.exists():
            raise FileNotFoundError(f"processor config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        steps = []
        overrides = overrides or {}
        step_states: list[dict[str, torch.Tensor] | None] = []
        for step_payload in payload.get("steps", []):
            step_name = step_payload.get("name") or step_payload.get("registry_name")
            if not step_name:
                raise KeyError(f"processor step missing name/registry_name: {step_payload}")
            step_config = dict(step_payload.get("config", {}))
            if step_name in overrides:
                step_config.update(overrides[step_name])
            step_cls = ProcessorStepRegistry.get(step_name)
            steps.append(step_cls(**step_config))
            state_file = step_payload.get("state_file")
            if state_file:
                state_path = save_directory / state_file
                step_states.append(load_file(str(state_path)) if state_path.exists() else {})
            else:
                step_states.append(None)

        combined_state: dict[str, torch.Tensor] = {}
        if any(step_state is None for step_state in step_states):
            state_filename = payload.get("state_filename", f"{payload.get('name', 'processor')}.safetensors")
            state_path = save_directory / state_filename
            combined_state = load_file(str(state_path)) if state_path.exists() else {}

        for idx, step in enumerate(steps):
            step_state = step_states[idx]
            if step_state is None:
                prefix = f"step{idx}."
                step_state = {k[len(prefix) :]: v for k, v in combined_state.items() if k.startswith(prefix)}
            step.load_state_dict(step_state)

        return cls(
            steps=steps,
            name=payload.get("name", config_path.stem),
            to_transition=to_transition or batch_to_transition,
            to_output=to_output or transition_to_batch,
        )


class PolicyProcessorPipeline(DataProcessorPipeline[TInput, TOutput]):
    pass


class IdentityProcessorStep(ProcessorStep):
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


ProcessorKwargs = dict[str, Any]
