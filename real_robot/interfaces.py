"""Small interfaces for connecting an existing robot and chunked policy."""

from __future__ import annotations

import importlib
from typing import Protocol

import numpy as np

from exrl.config import load_config


class Robot(Protocol):
    def observe(self, after: float | None = None) -> dict:
        """Return state, RGB cameras, and a local monotonic timestamp.

        When after is supplied, wait for an observation captured after that
        timestamp. Camera names are selected by the task's camera_keys.
        """
        ...

    def command(self, action: np.ndarray) -> None:
        """Submit one primitive action using the policy's action convention."""
        ...

    def close(self) -> None:
        """Release client connections and subscriptions."""
        ...


class ChunkPolicy(Protocol):
    def predict(self, observation: dict) -> np.ndarray:
        """Return an H x action_dim chunk in robot-command coordinates."""
        ...


def load_component(path, task):
    """Load a user factory without adding hardware dependencies to ExRL."""
    settings = load_config(path)
    module, name = settings["factory"].split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    return factory(settings.get("options", {}), task)


def policy_rng_state(policy):
    getter = getattr(policy, "rng_state", None)
    return getter() if getter is not None else None


def restore_policy_rng(policy, state):
    if state is not None:
        restore = getattr(policy, "restore_rng", None)
        if restore is None:
            raise ValueError("This checkpoint requires a policy adapter with restore_rng(state)")
        restore(state)
