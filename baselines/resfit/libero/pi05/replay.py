"""Pixel storage adapter for ResFiT's fixed-chunk replay and 50/50 sampler."""

from __future__ import annotations
from dataclasses import dataclass, fields

import numpy as np
import torch

from baselines.resfit.dp.td3_core import ReplayBatch


OBS_SPEC = {
    "pixels": ((64, 64, 3), np.uint8),
    "state": ((8,), np.float32),
    "raw_state": ((8,), np.float32),
    "base_conditioning": ((350,), np.float32),
}


@dataclass
class PixelTransition:
    state: dict
    base_action: np.ndarray
    action: np.ndarray
    reward: float
    discount: float
    next_state: dict
    next_base_action: np.ndarray
    terminated: bool
    truncated: bool


class PixelReplayBuffer:
    def __init__(self, capacity, seed=0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.position = self.size = 0
        self.rng = np.random.RandomState(seed)
        self.states = {
            k: np.empty((capacity, *shape), dtype) for k, (shape, dtype) in OBS_SPEC.items()
        }
        self.next_states = {k: np.empty_like(v) for k, v in self.states.items()}
        for name in ("base_actions", "actions", "next_base_actions"):
            setattr(self, name, np.empty((capacity, 350), np.float32))
        for name in ("rewards", "discounts"):
            setattr(self, name, np.empty(capacity, np.float32))
        for name in ("terminated", "truncated"):
            setattr(self, name, np.empty(capacity, bool))

    def __len__(self):
        return self.size

    def add(self, transition):
        if transition.terminated and transition.discount != 0:
            raise ValueError("success must have zero bootstrap discount")
        if not (0 <= transition.discount <= 1) or not np.isfinite(transition.reward):
            raise ValueError("invalid reward or discount")
        index = self.position
        for dest, obs in (
            (self.states, transition.state),
            (self.next_states, transition.next_state),
        ):
            if set(obs) != set(OBS_SPEC):
                raise ValueError("observation keys do not match the pi0.5 contract")
            for key, (shape, dtype) in OBS_SPEC.items():
                array = np.asarray(obs[key])
                if array.shape != shape or array.dtype != dtype or not np.isfinite(array).all():
                    raise ValueError("invalid observation: " + key)
                dest[key][index] = array
        for dest, src in (
            ("base_actions", "base_action"),
            ("actions", "action"),
            ("next_base_actions", "next_base_action"),
        ):
            value = np.asarray(getattr(transition, src)).reshape(-1)
            if value.shape != (350,) or not np.isfinite(value).all():
                raise ValueError("invalid action chunk")
            getattr(self, dest)[index] = value
        for dest, src in (
            ("rewards", "reward"),
            ("discounts", "discount"),
            ("terminated", "terminated"),
            ("truncated", "truncated"),
        ):
            getattr(self, dest)[index] = getattr(transition, src)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        if self.size < batch_size or batch_size <= 0:
            raise ValueError(f"replay has {self.size} items, cannot sample {batch_size}")
        indices = self.rng.randint(self.size, size=batch_size)

        def sample(value):
            if isinstance(value, dict):
                return {k: sample(v) for k, v in value.items()}
            return torch.as_tensor(value[indices], device=device)

        return ReplayBatch(**{f.name: sample(getattr(self, f.name)) for f in fields(ReplayBatch)})

    def state_dict(self):
        def snapshot(value):
            if isinstance(value, dict):
                return {k: snapshot(v) for k, v in value.items()}
            return value[: self.size].copy()

        return {
            "schema": 1,
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
            "rng": self.rng.get_state(),
            "arrays": {f.name: snapshot(getattr(self, f.name)) for f in fields(ReplayBatch)},
        }

    def load_state_dict(self, state):
        if state["schema"] != 1 or state["capacity"] != self.capacity:
            raise ValueError("replay contract changed")
        size, position = int(state["size"]), int(state["position"])
        if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
            raise ValueError("invalid replay cursor")
        for name, value in state["arrays"].items():
            dest = getattr(self, name)
            if isinstance(value, dict):
                for key in OBS_SPEC:
                    dest[key][:size] = value[key]
            else:
                dest[:size] = value
        self.size, self.position = size, position
        self.rng.set_state(state["rng"])


def sample_training_batch(online, offline, data_mode, batch_size, device):
    if data_mode == "online":
        if offline is not None:
            raise ValueError("online mode must not use offline data")
        return online.sample(batch_size, device)
    if data_mode != "mixed" or offline is None or batch_size % 2:
        raise ValueError("mixed mode requires a demo replay and an even batch size")
    a, b = online.sample(batch_size // 2, device), offline.sample(batch_size // 2, device)

    def cat(x, y):
        if isinstance(x, dict):
            return {k: cat(x[k], y[k]) for k in x}
        return torch.cat((x, y))

    return ReplayBatch(
        **{f.name: cat(getattr(a, f.name), getattr(b, f.name)) for f in fields(ReplayBatch)}
    )
