"""Replay and complete training snapshots for image-based ExRL."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np


def stack_rows(rows):
    if isinstance(rows[0], dict):
        return {key: stack_rows([row[key] for row in rows]) for key in rows[0]}
    return np.stack(rows)


class Replay:
    def __init__(self, capacity):
        self.capacity, self.cursor, self.rows = int(capacity), 0, []

    def extend(self, transitions):
        for row in transitions:
            if len(self.rows) < self.capacity:
                self.rows.append(row)
            else:
                self.rows[self.cursor] = row
            self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, batch_size, rng):
        if not self.rows:
            raise ValueError("Cannot sample empty replay")
        return stack_rows([self.rows[i] for i in rng.integers(len(self.rows), size=batch_size)])


def make_agent(config, observation):
    from exrl.pixel.agents.discrete.learner import PixelExRL

    kwargs = dict(config["learner"])
    for name in ("hidden_dims", "cnn_features", "cnn_strides"):
        if name in kwargs:
            kwargs[name] = tuple(kwargs[name])
    return PixelExRL(
        config["seed"],
        stack_rows([observation]),
        np.zeros((1, config["chunk_size"], 7), np.float32),
        num_discrete_actions=config["chunk_size"],
        action_id_offset=1,
        **kwargs,
    )


def save(path, agent, replay, counters, rng, config, **extra):
    from flax import serialization

    path = Path(path)
    temporary = path.with_suffix(".tmp")
    payload = dict(
        agent=serialization.to_bytes(agent._save_dict),
        replay=replay,
        counters=counters,
        rng=rng.bit_generator.state,
        config=config,
        **extra,
    )
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=4)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def restore(payload, agent):
    from flax import serialization

    state = serialization.from_bytes(agent._save_dict, payload["agent"])
    for name, value in state.items():
        target = (
            "_" + name
            if name in {"critic", "actor", "temp", "rng", "target_critic_params"}
            else name
        )
        setattr(agent, target, value)
