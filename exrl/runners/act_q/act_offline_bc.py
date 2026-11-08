"""Immutable demonstration sampling for ACT BC, separate from the RL replay.

NPZ states/actions are in the ACT adapter's *input* coordinates, not its final
mean/std-normalized coordinates. The adapter applies checkpoint normalization
exactly once. This loader deliberately supports only the DSRL-normalized
primitive-trajectory format; it does not guess another dataset's conventions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from exrl.runners.act_q.online_bc import BCTensorBatch, OnlineSuccessBCReplay


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_bc_counts(batch_size: int, offline_fraction: float) -> tuple[int, int]:
    if int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError("BC batch size must be a positive integer")
    fraction = float(offline_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("BC offline fraction must be finite and in [0, 1]")
    exact = int(batch_size) * fraction
    offline = int(round(exact))
    if not np.isclose(exact, offline, rtol=0, atol=1e-9):
        raise ValueError("BC offline fraction must give an integer sample count")
    return offline, int(batch_size) - offline


class ACTOfflineDemoPool:
    """Read-only primitive arrays, with lazy, episode-bounded action windows."""

    def __init__(
        self,
        path: str,
        *,
        horizon_steps: int,
        obs_dim: int,
        action_dim: int,
        cond_steps: int,
        device,
        padded: bool,
        observation_space_kind: str,
        action_space_kind: str,
    ):
        if (
            observation_space_kind != "dsrl_normalized"
            or action_space_kind != "dsrl_normalized"
        ):
            raise ValueError(
                "Offline ACT BC requires DSRL-normalized checkpoint input coordinates"
            )
        if cond_steps != 1:
            raise ValueError("Offline ACT BC currently requires cond_steps=1")
        if min(horizon_steps, obs_dim, action_dim) <= 0:
            raise ValueError("Offline ACT BC dimensions must be positive")
        self.path = str(Path(path).expanduser().resolve(strict=True))
        self.sha256 = file_sha256(self.path)
        self.horizon_steps = int(horizon_steps)
        self.device = torch.device(device)
        self.padded = bool(padded)
        with np.load(self.path, allow_pickle=False) as data:
            for key in ("states", "actions", "traj_lengths"):
                if key not in data.files:
                    raise ValueError(f"Offline demo missing {key}")
            self.states = np.asarray(data["states"], dtype=np.float32)
            self.actions = np.asarray(data["actions"], dtype=np.float32)
            lengths = np.asarray(data["traj_lengths"])
        if lengths.ndim != 1 or lengths.size == 0 or lengths.dtype.kind not in "iu":
            raise ValueError("Offline traj_lengths must be a nonempty integer vector")
        self.lengths = lengths.astype(np.int64)
        if np.any(self.lengths <= 0):
            raise ValueError("Offline trajectories must have positive lengths")
        count = int(self.lengths.sum())
        if self.states.shape != (count, int(obs_dim)) or self.actions.shape != (
            count,
            int(action_dim),
        ):
            raise ValueError(
                f"Offline shape/length mismatch: states={self.states.shape} actions={self.actions.shape} total={count}"
            )
        if not np.isfinite(self.states).all() or not np.isfinite(self.actions).all():
            raise ValueError("Offline states/actions must be finite")
        # Original normalized demonstrations should be within their min/max
        # range. Fail closed instead of silently clipping a raw-coordinate file.
        if np.abs(self.states).max() > 1.001 or np.abs(self.actions).max() > 1.001:
            raise ValueError("Offline demo is outside DSRL-normalized bounds")
        self.starts = np.concatenate(([0], np.cumsum(self.lengths[:-1])))
        self.ends = self.starts + self.lengths
        self.window_counts = (
            self.lengths
            if self.padded
            else np.maximum(self.lengths - self.horizon_steps + 1, 0)
        )
        self.eligible_episodes = np.flatnonzero(self.window_counts > 0)
        if self.eligible_episodes.size == 0:
            raise ValueError("Offline demo has no eligible BC windows")
        self.window_cumsum = np.cumsum(self.window_counts)
        self.size = int(self.window_counts.sum())
        for array in (
            self.states,
            self.actions,
            self.lengths,
            self.starts,
            self.ends,
            self.window_counts,
            self.eligible_episodes,
            self.window_cumsum,
        ):
            array.setflags(write=False)

    def __len__(self):
        return self.size

    def contract(self) -> dict:
        # No absolute path: moving the exact dataset to another host is safe.
        return {
            "format": "act_offline_bc_dsrl_primitive_v1",
            "sha256": self.sha256,
            "episodes": int(self.lengths.size),
            "frames": int(self.states.shape[0]),
            "windows": self.size,
            "horizon_steps": self.horizon_steps,
            "obs_dim": int(self.states.shape[1]),
            "action_dim": int(self.actions.shape[1]),
            "cond_steps": 1,
            "padded": self.padded,
            "coordinates": "dsrl_normalized_before_ACT_mean_std",
        }

    def batch_at(self, episode_ids, offsets) -> BCTensorBatch:
        episodes = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
        offsets = np.asarray(offsets, dtype=np.int64).reshape(-1)
        if episodes.size == 0 or episodes.shape != offsets.shape:
            raise ValueError("Offline batch indices must be nonempty and aligned")
        if np.any(episodes < 0) or np.any(episodes >= self.lengths.size):
            raise ValueError("Offline episode index out of bounds")
        if np.any(offsets < 0) or np.any(offsets >= self.window_counts[episodes]):
            raise ValueError("Offline frame offset out of bounds")
        indices = self.starts[episodes] + offsets
        targets = indices[:, None] + np.arange(self.horizon_steps)[None, :]
        ends = self.ends[episodes, None]
        pads = targets >= ends
        clipped = np.minimum(targets, ends - 1)
        return BCTensorBatch(
            actions=torch.as_tensor(self.actions[clipped], device=self.device),
            conditions={
                "state": torch.as_tensor(
                    self.states[indices, None, :], device=self.device
                )
            },
            action_is_pad=torch.as_tensor(pads, dtype=torch.bool, device=self.device),
            episode_ids=torch.as_tensor(episodes, device=self.device),
        )

    def sample(
        self, batch_size: int, *, sampling: str = "uniform_episode"
    ) -> BCTensorBatch:
        if batch_size <= 0:
            raise ValueError("Offline batch size must be positive")
        if sampling == "uniform_episode":
            episodes = np.random.choice(
                self.eligible_episodes, size=batch_size, replace=True
            )
            offsets = np.random.randint(0, self.window_counts[episodes])
        elif sampling == "uniform_window":
            windows = np.random.randint(0, self.size, size=batch_size)
            episodes = np.searchsorted(self.window_cumsum, windows, side="right")
            previous = self.window_cumsum[episodes] - self.window_counts[episodes]
            offsets = windows - previous
        else:
            raise ValueError(f"Invalid offline sampling mode: {sampling}")
        return self.batch_at(episodes, offsets)


def bc_sources_ready(
    online: OnlineSuccessBCReplay,
    offline: Optional[ACTOfflineDemoPool],
    *,
    batch_size: int,
    offline_fraction: float,
    start_mode: str,
) -> bool:
    n_offline, n_online = split_bc_counts(batch_size, offline_fraction)
    if start_mode not in ("match_online", "immediate"):
        raise ValueError(f"Invalid offline BC start mode: {start_mode}")
    if n_offline and offline is None:
        raise ValueError("Offline BC samples requested without an offline pool")
    # Match the original online-only readiness gate in the controlled ablation,
    # even for the demo-only arm. Do not silently fill missing online samples
    # with demos: that would change the configured ratio.
    required_online = batch_size if start_mode == "match_online" else n_online
    return required_online == 0 or online.can_sample(required_online)


def sample_bc_sources(
    online, offline, *, batch_size, offline_fraction, online_sampling, offline_sampling
):
    """Return source batches and weights; zero-demo mode preserves legacy RNG."""
    n_offline, n_online = split_bc_counts(batch_size, offline_fraction)
    sources = []
    if n_online:
        sources.append(
            (
                "online",
                online.sample(n_online, sampling=online_sampling),
                n_online / batch_size,
            )
        )
    if n_offline:
        if offline is None:
            raise ValueError("Missing offline pool")
        sources.append(
            (
                "offline",
                offline.sample(n_offline, sampling=offline_sampling),
                n_offline / batch_size,
            )
        )
    return sources


def validate_resume_bc_contract(saved: Optional[dict], current: Optional[dict]) -> None:
    if saved != current:
        raise ValueError(
            f"BC data contract mismatch on resume: saved={saved} current={current}"
        )
