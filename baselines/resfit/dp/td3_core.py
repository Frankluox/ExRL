"""State-based ResFiT learner shared by primitive- and chunk-level adapters.

This module intentionally contains no AAC, entropy, CQL, or Gaussian SAC
policy. ``action_dim`` is the physical action dimension for the preserved
primitive adapter or ``TA * physical_action_dim`` for the fixed-TA chunk
adapter. In both cases the critic learns
Q(state, *executed full action*) and the actor predicts a deterministic
residual conditioned on state and the corresponding complete base action.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from exrl.common.networks import MLP as ExRLMLP


def _as_float_array(value: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float32).reshape(-1)
    if out.size == 0 or not np.all(np.isfinite(out)):
        raise ValueError("%s must be a non-empty finite vector" % name)
    return out


class ResFiTActionScaler:
    """Per-dimension bridge between DSRL, raw, and ResFiT action coordinates.

    DSRL normalizes actions with the dataset min/max.  ResFiT expands the raw
    half-range by ``1 + action_scale`` and predicts a residual in that expanded
    coordinate system.  Keeping the raw coordinate in the middle is important:
    action dimensions do not generally have the same physical scale.
    """

    def __init__(
        self,
        action_min: np.ndarray,
        action_max: np.ndarray,
        action_scale: float = 0.1,
        min_range_per_dim: float = 0.1,
    ):
        lo = _as_float_array(action_min, "action_min")
        hi = _as_float_array(action_max, "action_max")
        if lo.shape != hi.shape:
            raise ValueError(
                "action_min/action_max shape mismatch: %s vs %s" % (lo.shape, hi.shape)
            )
        if np.any(hi <= lo):
            raise ValueError("every DSRL action range must be strictly positive")
        if float(action_scale) < 0.0:
            raise ValueError("action_scale must be non-negative")
        if float(min_range_per_dim) <= 0.0:
            raise ValueError("min_range_per_dim must be positive")

        self.action_min = lo
        self.action_max = hi
        self.action_scale = float(action_scale)
        self.min_range_per_dim = float(min_range_per_dim)
        self.mid = (lo + hi) / 2.0
        self.dataset_half_range = (hi - lo) / 2.0
        self.half_range = np.maximum(self.dataset_half_range, self.min_range_per_dim / 2.0)
        self.expanded_half_range = self.half_range * (1.0 + self.action_scale)
        self.expanded_min = self.mid - self.expanded_half_range
        self.expanded_max = self.mid + self.expanded_half_range

    @classmethod
    def from_npz(
        cls,
        path: str,
        action_scale: float = 0.1,
        min_range_per_dim: float = 0.1,
    ) -> "ResFiTActionScaler":
        data = np.load(path)
        missing = [key for key in ("action_min", "action_max") if key not in data]
        if missing:
            raise KeyError("normalization npz is missing: %s" % missing)
        return cls(
            action_min=data["action_min"],
            action_max=data["action_max"],
            action_scale=action_scale,
            min_range_per_dim=min_range_per_dim,
        )

    @property
    def action_dim(self) -> int:
        return int(self.action_min.shape[0])

    @staticmethod
    def _like(value: np.ndarray, reference):
        if torch.is_tensor(reference):
            return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        return value.astype(np.asarray(reference).dtype, copy=False)

    @staticmethod
    def _clip(value, low: float, high: float):
        if torch.is_tensor(value):
            return torch.clamp(value, low, high)
        return np.clip(value, low, high)

    def _check_last_dim(self, value, name: str) -> None:
        if int(value.shape[-1]) != self.action_dim:
            raise ValueError(
                "%s last dim must be %d, got %s" % (name, self.action_dim, tuple(value.shape))
            )

    def dp_to_raw(self, action_dp):
        self._check_last_dim(action_dp, "action_dp")
        lo = self._like(self.action_min, action_dp)
        hi = self._like(self.action_max, action_dp)
        return lo + (action_dp + 1.0) * (hi - lo) / 2.0

    def raw_to_dp(self, action_raw):
        self._check_last_dim(action_raw, "action_raw")
        lo = self._like(self.action_min, action_raw)
        hi = self._like(self.action_max, action_raw)
        return 2.0 * (action_raw - lo) / (hi - lo) - 1.0

    def raw_to_resfit(self, action_raw, clamp: bool = True):
        self._check_last_dim(action_raw, "action_raw")
        mid = self._like(self.mid, action_raw)
        expanded = self._like(self.expanded_half_range, action_raw)
        out = (action_raw - mid) / expanded
        return self._clip(out, -1.0, 1.0) if clamp else out

    def resfit_to_raw(self, action_resfit, clamp: bool = True):
        self._check_last_dim(action_resfit, "action_resfit")
        value = self._clip(action_resfit, -1.0, 1.0) if clamp else action_resfit
        mid = self._like(self.mid, value)
        expanded = self._like(self.expanded_half_range, value)
        return mid + expanded * value

    def dp_to_resfit(self, action_dp, clamp: bool = True):
        return self.raw_to_resfit(self.dp_to_raw(action_dp), clamp=clamp)

    def resfit_to_dp(self, action_resfit, clamp: bool = True):
        return self.raw_to_dp(self.resfit_to_raw(action_resfit, clamp=clamp))

    def combine(
        self, base_action_dp, residual_resfit, replay_action_mode: str = "executed_clamped"
    ):
        """Combine base and residual while retaining both release and fixed replay semantics."""
        if replay_action_mode not in {"executed_clamped", "release_preclamp"}:
            raise ValueError("unsupported replay_action_mode=%s" % replay_action_mode)
        base_resfit = self.dp_to_resfit(base_action_dp)
        preclamp_resfit = base_resfit + residual_resfit
        full_resfit = self._clip(preclamp_resfit, -1.0, 1.0)
        replay_resfit = full_resfit if replay_action_mode == "executed_clamped" else preclamp_resfit
        converted_raw = self.resfit_to_raw(full_resfit)
        converted_dp = self.raw_to_dp(converted_raw)

        # A zero residual must execute the frozen DP action bit-for-bit.  The
        # expanded ResFiT affine map is mathematically invertible, but its two
        # float32 round trips can otherwise move the asymmetric gripper
        # dimension by several ULPs.  Preserve zero components directly while
        # retaining the official ResFiT coordinate path for non-zero residuals.
        base_raw = self.dp_to_raw(base_action_dp)
        zero_residual = residual_resfit == 0.0
        if torch.is_tensor(zero_residual):
            action_raw = torch.where(zero_residual, base_raw, converted_raw)
            action_dp = torch.where(zero_residual, base_action_dp, converted_dp)
        else:
            action_raw = np.where(zero_residual, base_raw, converted_raw)
            action_dp = np.where(zero_residual, base_action_dp, converted_dp)
        return {
            "base_resfit": base_resfit,
            "preclamp_resfit": preclamp_resfit,
            "full_resfit": full_resfit,
            "replay_resfit": replay_resfit,
            "action_raw": action_raw,
            "action_dp": action_dp,
        }

    def manifest(self) -> Dict[str, object]:
        return {
            "action_scale": self.action_scale,
            "min_range_per_dim": self.min_range_per_dim,
            "action_min": self.action_min.tolist(),
            "action_max": self.action_max.tolist(),
            "mid": self.mid.tolist(),
            "dataset_half_range": self.dataset_half_range.tolist(),
            "half_range": self.half_range.tolist(),
            "expanded_half_range": self.expanded_half_range.tolist(),
            "expanded_min": self.expanded_min.tolist(),
            "expanded_max": self.expanded_max.tolist(),
            "max_actor_mean_raw_delta": (self.action_scale * self.expanded_half_range).tolist(),
        }


class StatePreprocessor:
    """Reproduce the formal ExRL state coordinate and optional ResFiT raw z-score."""

    VALID_MODES = {"stats_no_clip", "dsrl_raw", "resfit_raw_zscore"}

    def __init__(
        self,
        obs_min: np.ndarray,
        obs_max: np.ndarray,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        mode: str = "stats_no_clip",
        min_state_std: float = 0.1,
    ):
        self.obs_min = _as_float_array(obs_min, "obs_min")
        self.obs_max = _as_float_array(obs_max, "obs_max")
        self.state_mean = _as_float_array(state_mean, "state_mean")
        self.state_std = _as_float_array(state_std, "state_std")
        shapes = {
            self.obs_min.shape,
            self.obs_max.shape,
            self.state_mean.shape,
            self.state_std.shape,
        }
        if len(shapes) != 1:
            raise ValueError("state preprocessing vectors must share a shape")
        if np.any(self.obs_max <= self.obs_min):
            raise ValueError("obs normalization ranges must be positive")
        mode = str(mode).lower()
        if mode not in self.VALID_MODES:
            raise ValueError("unsupported state mode=%s" % mode)
        self.mode = mode
        self.min_state_std = float(min_state_std)
        self.formal_std = np.where(self.state_std < 1e-8, 1.0, self.state_std)
        self.resfit_std = np.maximum(self.state_std, self.min_state_std)

    @classmethod
    def from_npz(
        cls,
        normalization_path: str,
        stats_path: str,
        lowdim_keys: Sequence[str],
        mode: str = "stats_no_clip",
        min_state_std: float = 0.1,
    ) -> "StatePreprocessor":
        norm = np.load(normalization_path)
        stats = np.load(stats_path)
        means: List[np.ndarray] = []
        stds: List[np.ndarray] = []
        for key in lowdim_keys:
            mean_key = "%s_mean" % key
            std_key = "%s_std" % key
            if mean_key not in stats or std_key not in stats:
                raise KeyError("stats npz missing %s/%s" % (mean_key, std_key))
            means.append(np.asarray(stats[mean_key], dtype=np.float32).reshape(-1))
            stds.append(np.asarray(stats[std_key], dtype=np.float32).reshape(-1))
        return cls(
            obs_min=norm["obs_min"],
            obs_max=norm["obs_max"],
            state_mean=np.concatenate(means),
            state_std=np.concatenate(stds),
            mode=mode,
            min_state_std=min_state_std,
        )

    @property
    def state_dim(self) -> int:
        return int(self.state_mean.shape[0])

    def dp_to_raw(self, state_dp: np.ndarray) -> np.ndarray:
        state = np.asarray(state_dp, dtype=np.float32)
        if int(state.shape[-1]) != self.state_dim:
            raise ValueError(
                "state last dim must be %d, got %s" % (self.state_dim, tuple(state.shape))
            )
        # Exact inverse of DSRL's 2 * ((raw-min)/(max-min+1e-6) - .5).
        return self.obs_min + (state + 1.0) * (self.obs_max - self.obs_min + 1e-6) / 2.0

    def transform(self, state_dp: np.ndarray) -> np.ndarray:
        state = np.asarray(state_dp, dtype=np.float32)
        if int(state.shape[-1]) != self.state_dim:
            raise ValueError(
                "state last dim must be %d, got %s" % (self.state_dim, tuple(state.shape))
            )
        if self.mode == "stats_no_clip":
            # Deliberately match the 915 formal run: apply raw task statistics
            # directly to the already DSRL-normalized state, with no clipping.
            out = (state - self.state_mean) / self.formal_std
        elif self.mode == "dsrl_raw":
            # Transport's selected DP ExRL-Q runs feed the normalized state
            # emitted by the DSRL wrapper directly to the RL networks.
            out = state
        else:
            raw = self.dp_to_raw(state)
            out = (raw - self.state_mean) / self.resfit_std
        return np.asarray(out, dtype=np.float32)

    def manifest(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "min_state_std": self.min_state_std,
            "state_dim": self.state_dim,
            "formal_no_clip": self.mode == "stats_no_clip",
            "identity_dsrl_state": self.mode == "dsrl_raw",
        }


def _init_resfit_actor_linear(layer: nn.Linear) -> None:
    # Match ResFiT's ``orth_weight_init`` for actor intermediate Linear
    # layers: orthogonal gain 1 (not the ReLU gain) and zero bias.
    nn.init.orthogonal_(layer.weight, gain=1.0)
    nn.init.zeros_(layer.bias)


class ResidualActor(nn.Module):
    """Deterministic ResFiT actor over either one action or a flattened chunk."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (2048, 2048, 2048),
        action_scale: float = 0.1,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.action_scale = float(action_scale)
        dims = [self.state_dim + self.action_dim] + [int(x) for x in hidden_dims]
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            linear = nn.Linear(in_dim, out_dim)
            _init_resfit_actor_linear(linear)
            layers.append(linear)
            if use_layernorm:
                layers.append(nn.LayerNorm(out_dim))
            # Keep the official ResFiT actor block ordering exactly:
            # Linear -> LayerNorm -> Dropout(0) -> ReLU.  Dropout(0) is an
            # identity numerically, but retaining the module makes the state
            # actor structurally auditable against the upstream image actor.
            layers.append(nn.Dropout(0.0))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)
        self.output = nn.Linear(dims[-1], self.action_dim)
        # Exact-zero initialization is the key residual-policy safety property.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.state_dim or base_action.shape[-1] != self.action_dim:
            raise ValueError("ResidualActor input shape mismatch")
        features = torch.cat([state, base_action], dim=-1)
        return self.action_scale * torch.tanh(self.output(self.trunk(features)))


class ContinuousQMember(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        use_layernorm: bool,
    ):
        super().__init__()
        # Reuse the ExRL state-based MLP itself rather than maintaining a
        # look-alike.  This preserves its native nn.Linear initialization and
        # exact Linear -> LayerNorm -> ReLU(inplace=True) ordering.
        self.backbone = ExRLMLP(
            in_dim=int(state_dim + action_dim),
            out_dim=1,
            hidden_dims=tuple(int(x) for x in hidden_dims),
            activation="relu",
            use_layernorm=bool(use_layernorm),
        )

    @property
    def output(self) -> nn.Linear:
        return self.backbone.net[-1]

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.backbone(torch.cat([state, action], dim=-1))


class EnsembleContinuousCritic(nn.Module):
    """Independent ExRL-style continuous Q members, returned as [K, B]."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (512, 512, 512),
        ensemble_size: int = 10,
        use_layernorm: bool = True,
    ):
        super().__init__()
        if int(ensemble_size) <= 0:
            raise ValueError("ensemble_size must be positive")
        self.ensemble_size = int(ensemble_size)
        self.members = nn.ModuleList(
            [
                ContinuousQMember(state_dim, action_dim, hidden_dims, use_layernorm)
                for _ in range(self.ensemble_size)
            ]
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.stack([member(state, action).squeeze(-1) for member in self.members], dim=0)

    def forward_members(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        member_indices: Sequence[int],
    ) -> torch.Tensor:
        indices = [int(index) for index in member_indices]
        if not indices or any(index < 0 or index >= self.ensemble_size for index in indices):
            raise ValueError("critic member indices are empty or out of range")
        return torch.stack(
            [self.members[index](state, action).squeeze(-1) for index in indices],
            dim=0,
        )


@dataclass
class PrimitiveTransition:
    state: np.ndarray
    base_action: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    next_base_action: np.ndarray
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        """Episode boundary (success terminal or time-limit truncation)."""
        return bool(self.terminated or self.truncated)

    @property
    def bootstrap_terminal(self) -> bool:
        """Only true terminals suppress the critic bootstrap."""
        return bool(self.terminated)


@dataclass
class AggregatedTransition:
    state: np.ndarray
    base_action: np.ndarray
    action: np.ndarray
    reward: float
    discount: float
    next_state: np.ndarray
    next_base_action: np.ndarray
    terminated: bool
    truncated: bool


class PrimitiveNStepAccumulator:
    """Per-environment primitive n-step accumulator with terminal flushing."""

    def __init__(self, n_step: int, gamma: float, n_envs: int):
        if int(n_step) <= 0 or int(n_envs) <= 0:
            raise ValueError("n_step and n_envs must be positive")
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.queues: List[Deque[PrimitiveTransition]] = [deque() for _ in range(int(n_envs))]

    def _aggregate_front(self, queue_: Deque[PrimitiveTransition]) -> AggregatedTransition:
        if not queue_:
            raise RuntimeError("cannot aggregate an empty queue")
        items = list(queue_)[: self.n_step]
        reward = 0.0
        final = items[0]
        steps = 0
        for index, transition in enumerate(items):
            reward += (self.gamma**index) * float(transition.reward)
            final = transition
            steps += 1
            if transition.done:
                break
        # Match ExRL timeout handling: a truncation ends/flushed the episode,
        # but the target bootstraps from its terminal observation. Success is
        # true task terminals in the formal robomimic environments.
        discount = 0.0 if final.bootstrap_terminal else self.gamma**steps
        first = items[0]
        return AggregatedTransition(
            state=np.asarray(first.state, dtype=np.float32),
            base_action=np.asarray(first.base_action, dtype=np.float32),
            action=np.asarray(first.action, dtype=np.float32),
            reward=float(reward),
            discount=float(discount),
            next_state=np.asarray(final.next_state, dtype=np.float32),
            next_base_action=np.asarray(final.next_base_action, dtype=np.float32),
            terminated=bool(final.terminated),
            truncated=bool(final.truncated),
        )

    def append(self, env_index: int, transition: PrimitiveTransition) -> List[AggregatedTransition]:
        queue_ = self.queues[int(env_index)]
        queue_.append(transition)
        emitted: List[AggregatedTransition] = []
        if transition.done:
            while queue_:
                emitted.append(self._aggregate_front(queue_))
                queue_.popleft()
        elif len(queue_) >= self.n_step:
            emitted.append(self._aggregate_front(queue_))
            queue_.popleft()
        return emitted

    def flush(self, env_index: Optional[int] = None) -> List[AggregatedTransition]:
        indices: Iterable[int] = range(len(self.queues)) if env_index is None else [int(env_index)]
        emitted: List[AggregatedTransition] = []
        for index in indices:
            queue_ = self.queues[index]
            while queue_:
                emitted.append(self._aggregate_front(queue_))
                queue_.popleft()
        return emitted

    def reset(self, env_index: Optional[int] = None) -> None:
        indices: Iterable[int] = range(len(self.queues)) if env_index is None else [int(env_index)]
        for index in indices:
            self.queues[index].clear()


@dataclass
class ReplayBatch:
    states: torch.Tensor
    base_actions: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    next_states: torch.Tensor
    next_base_actions: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class PrimitiveReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int, seed: int = 0):
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.states = np.empty((self.capacity, self.state_dim), dtype=np.float32)
        self.base_actions = np.empty((self.capacity, self.action_dim), dtype=np.float32)
        self.actions = np.empty((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.empty((self.capacity,), dtype=np.float32)
        self.discounts = np.empty((self.capacity,), dtype=np.float32)
        self.next_states = np.empty((self.capacity, self.state_dim), dtype=np.float32)
        self.next_base_actions = np.empty((self.capacity, self.action_dim), dtype=np.float32)
        self.terminated = np.empty((self.capacity,), dtype=np.bool_)
        self.truncated = np.empty((self.capacity,), dtype=np.bool_)
        self.position = 0
        self.size = 0
        self.rng = np.random.RandomState(int(seed))

    def __len__(self) -> int:
        return int(self.size)

    def add(self, transition: AggregatedTransition) -> None:
        index = self.position
        self.states[index] = transition.state
        self.base_actions[index] = transition.base_action
        self.actions[index] = transition.action
        self.rewards[index] = transition.reward
        self.discounts[index] = transition.discount
        self.next_states[index] = transition.next_state
        self.next_base_actions[index] = transition.next_base_action
        self.terminated[index] = transition.terminated
        self.truncated[index] = transition.truncated
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def extend(self, transitions: Iterable[AggregatedTransition]) -> int:
        count = 0
        for transition in transitions:
            self.add(transition)
            count += 1
        return count

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if self.size < int(batch_size):
            raise RuntimeError("replay has %d items, cannot sample %d" % (self.size, batch_size))
        indices = self.rng.randint(0, self.size, size=int(batch_size))

        def tensor(array, dtype=torch.float32):
            return torch.as_tensor(array[indices], dtype=dtype, device=device)

        return ReplayBatch(
            states=tensor(self.states),
            base_actions=tensor(self.base_actions),
            actions=tensor(self.actions),
            rewards=tensor(self.rewards),
            discounts=tensor(self.discounts),
            next_states=tensor(self.next_states),
            next_base_actions=tensor(self.next_base_actions),
            terminated=tensor(self.terminated, dtype=torch.bool),
            truncated=tensor(self.truncated, dtype=torch.bool),
        )

    def state_dict(self) -> Dict[str, object]:
        size = int(self.size)
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "capacity": self.capacity,
            "position": self.position,
            "size": size,
            "states": self.states[:size].copy(),
            "base_actions": self.base_actions[:size].copy(),
            "actions": self.actions[:size].copy(),
            "rewards": self.rewards[:size].copy(),
            "discounts": self.discounts[:size].copy(),
            "next_states": self.next_states[:size].copy(),
            "next_base_actions": self.next_base_actions[:size].copy(),
            "terminated": self.terminated[:size].copy(),
            "truncated": self.truncated[:size].copy(),
            "rng_state": self.rng.get_state(),
        }

    def load_state_dict(
        self,
        state: Dict[str, object],
        strict_capacity: bool = True,
    ) -> None:
        if int(state["state_dim"]) != self.state_dim or int(state["action_dim"]) != self.action_dim:
            raise ValueError("replay shape mismatch")
        saved_capacity = int(state["capacity"])
        if strict_capacity and saved_capacity != self.capacity:
            raise ValueError(
                "replay capacity mismatch: checkpoint=%d current=%d"
                % (saved_capacity, self.capacity)
            )
        size = int(state["size"])
        if size > self.capacity:
            raise ValueError("saved replay is larger than current capacity")
        saved_position = int(state["position"])
        if not strict_capacity and saved_position != size % saved_capacity:
            raise ValueError(
                "capacity-independent replay load requires a contiguous, non-wrapped payload"
            )
        for name in (
            "states",
            "base_actions",
            "actions",
            "rewards",
            "discounts",
            "next_states",
            "next_base_actions",
            "terminated",
            "truncated",
        ):
            getattr(self, name)[:size] = state[name]
        self.size = size
        self.position = saved_position % self.capacity if strict_capacity else size % self.capacity
        self.rng.set_state(state["rng_state"])


def concatenate_batches(first: ReplayBatch, second: ReplayBatch) -> ReplayBatch:
    def cat(name: str):
        return torch.cat([getattr(first, name), getattr(second, name)], dim=0)

    return ReplayBatch(
        states=cat("states"),
        base_actions=cat("base_actions"),
        actions=cat("actions"),
        rewards=cat("rewards"),
        discounts=cat("discounts"),
        next_states=cat("next_states"),
        next_base_actions=cat("next_base_actions"),
        terminated=cat("terminated"),
        truncated=cat("truncated"),
    )


def sample_training_batch(
    online_replay: PrimitiveReplayBuffer,
    offline_replay: Optional[PrimitiveReplayBuffer],
    data_mode: str,
    batch_size: int,
    device: torch.device,
) -> ReplayBatch:
    mode = str(data_mode).lower()
    if mode == "online":
        if offline_replay is not None:
            raise ValueError("online mode must not receive an offline replay")
        return online_replay.sample(int(batch_size), device)
    if mode != "mixed":
        raise ValueError("data_mode must be mixed or online")
    if offline_replay is None:
        raise ValueError("mixed mode requires an offline replay")
    if int(batch_size) % 2 != 0:
        raise ValueError("mixed mode requires an even batch size")
    half = int(batch_size) // 2
    return concatenate_batches(
        online_replay.sample(half, device), offline_replay.sample(half, device)
    )


@dataclass
class ResFiTLearnerConfig:
    state_dim: int = 23
    action_dim: int = 7
    actor_hidden_dims: Tuple[int, ...] = (2048, 2048, 2048)
    critic_hidden_dims: Tuple[int, ...] = (512, 512, 512)
    critic_ensemble_size: int = 10
    target_critic_subsample_size: int = 2
    critic_use_layernorm: bool = True
    action_scale: float = 0.1
    actor_lr: float = 1e-6
    critic_lr: float = 1e-4
    tau: float = 0.005
    actor_grad_clip: float = 1.0
    critic_grad_clip: float = 1.0
    target_noise_std: float = 0.05
    target_noise_clip: float = 0.3
    td_loss_mode: str = "per_head_mse"
    # The trainer resolves its default bounds from the reward contract;
    # None disables an individual bound at the learner level.
    target_q_clip_max: Optional[float] = None
    target_q_clip_min: Optional[float] = None
    consolidate_metric_syncs: bool = False
    target_critic_subset_compute: bool = False
    foreach_soft_update: bool = False
    freeze_critic_during_actor_update: bool = False
    fused_adamw: bool = False


def critic_td_loss(errors: torch.Tensor, mode: str) -> torch.Tensor:
    """Keep the corrected MSE and the release-compatible ablation explicit."""
    if mode == "per_head_mse":
        return errors.square().mean()
    if mode == "release_mean_abs_square":
        return errors.abs().mean(dim=0).square().mean()
    raise ValueError("unsupported td_loss_mode=%s" % mode)


class ResFiTLearner:
    VALID_TD_LOSSES = {"per_head_mse", "release_mean_abs_square"}

    def __init__(self, config: ResFiTLearnerConfig, device: torch.device):
        self.config = config
        self.device = torch.device(device)
        if config.td_loss_mode not in self.VALID_TD_LOSSES:
            raise ValueError("unsupported td_loss_mode=%s" % config.td_loss_mode)
        if config.target_q_clip_max is not None and not math.isfinite(config.target_q_clip_max):
            raise ValueError("target_q_clip_max must be finite or None")
        if config.target_q_clip_min is not None and not math.isfinite(config.target_q_clip_min):
            raise ValueError("target_q_clip_min must be finite or None")
        if (
            config.target_q_clip_min is not None
            and config.target_q_clip_max is not None
            and config.target_q_clip_min > config.target_q_clip_max
        ):
            raise ValueError("target_q_clip_min must not exceed target_q_clip_max")
        if not (0 < int(config.target_critic_subsample_size) <= int(config.critic_ensemble_size)):
            raise ValueError("invalid target critic subsample size")
        self.actor = ResidualActor(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.actor_hidden_dims,
            action_scale=config.action_scale,
            use_layernorm=True,
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic = EnsembleContinuousCritic(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.critic_hidden_dims,
            ensemble_size=config.critic_ensemble_size,
            use_layernorm=bool(config.critic_use_layernorm),
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        # Official ResFiT uses AdamW with PyTorch defaults, including the
        # default decoupled weight decay of 0.01, for both actor and critic.
        optimizer_kwargs = {}
        if bool(config.fused_adamw):
            if self.device.type != "cuda":
                raise ValueError("fused AdamW requires a CUDA learner")
            optimizer_kwargs["fused"] = True
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=float(config.actor_lr), **optimizer_kwargs
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=float(config.critic_lr), **optimizer_kwargs
        )
        # Match QAgent: target critic is always inference-only, while the
        # target actor remains in train mode (its dropout probability is 0).
        self.critic_target.train(False)

    @torch.no_grad()
    def act(
        self,
        states: np.ndarray,
        base_actions: np.ndarray,
        eval_mode: bool,
        exploration_std: float = 0.05,
        exploration_clip: float = 0.3,
    ) -> np.ndarray:
        state_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        base_t = torch.as_tensor(base_actions, dtype=torch.float32, device=self.device)
        residual = self.actor(state_t, base_t)
        if not eval_mode and float(exploration_std) > 0.0:
            noise = torch.randn_like(residual) * float(exploration_std)
            noise = torch.clamp(noise, -float(exploration_clip), float(exploration_clip))
            residual = residual + noise
        # Exact TruncatedNormal support used by ResFiT.
        return torch.clamp(residual, -1.0 + 1e-6, 1.0 - 1e-6).cpu().numpy().astype(np.float32)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        tau = float(self.config.tau)
        with torch.no_grad():
            source_params = list(source.parameters())
            target_params = list(target.parameters())
            if bool(self.config.foreach_soft_update):
                torch._foreach_mul_(target_params, 1.0 - tau)
                torch._foreach_add_(target_params, source_params, alpha=tau)
            else:
                for source_param, target_param in zip(source_params, target_params):
                    target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)

    def _target_min_q(self, batch: ReplayBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            next_residual = self.actor_target(batch.next_states, batch.next_base_actions)
            noise = torch.randn_like(next_residual) * float(self.config.target_noise_std)
            noise = torch.clamp(
                noise,
                -float(self.config.target_noise_clip),
                float(self.config.target_noise_clip),
            )
            next_residual = torch.clamp(
                next_residual + noise,
                -1.0 + 1e-6,
                1.0 - 1e-6,
            )
            next_full_action = torch.clamp(batch.next_base_actions + next_residual, -1.0, 1.0)
            indices = torch.randperm(int(self.config.critic_ensemble_size), device=self.device)[
                : int(self.config.target_critic_subsample_size)
            ]
            if bool(self.config.target_critic_subset_compute):
                selected_q = self.critic_target.forward_members(
                    batch.next_states,
                    next_full_action,
                    indices.detach().cpu().tolist(),
                )
            else:
                all_target_q = self.critic_target(batch.next_states, next_full_action)
                selected_q = all_target_q.index_select(0, indices)
            target_min_q = selected_q.min(dim=0).values
            return target_min_q, next_full_action

    @staticmethod
    def _metrics_with_one_device_sync(items) -> Dict[str, float]:
        names = [name for name, _value in items]
        values = torch.stack([value.detach().reshape(()) for _name, value in items]).cpu().tolist()
        return {name: float(value) for name, value in zip(names, values)}

    def update_critic(self, batch: ReplayBatch, collect_metrics: bool = True) -> Dict[str, float]:
        target_min_q, next_full_action = self._target_min_q(batch)
        unclipped_target_q = batch.rewards + batch.discounts * target_min_q
        target_q = unclipped_target_q
        if self.config.target_q_clip_max is not None or self.config.target_q_clip_min is not None:
            # Clip the complete Bellman target, not the bootstrap Q or the
            # target-action smoothing noise. Omitted bounds remain disabled.
            target_q = target_q.clamp(
                min=self.config.target_q_clip_min, max=self.config.target_q_clip_max
            )
        current_q = self.critic(batch.states, batch.actions)
        errors = current_q - target_q.unsqueeze(0)
        critic_loss = critic_td_loss(errors, self.config.td_loss_mode)
        if not torch.isfinite(critic_loss):
            raise RuntimeError("critic loss is NaN/Inf")

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.critic_grad_clip
        )
        self.critic_optimizer.step()
        self._soft_update(self.critic, self.critic_target)
        if not collect_metrics:
            return {}

        lower_clipped = None
        upper_clipped = None
        if self.config.target_q_clip_max is not None or self.config.target_q_clip_min is not None:
            lower_clipped = (
                unclipped_target_q < self.config.target_q_clip_min
                if self.config.target_q_clip_min is not None
                else torch.zeros_like(unclipped_target_q, dtype=torch.bool)
            )
            upper_clipped = (
                unclipped_target_q > self.config.target_q_clip_max
                if self.config.target_q_clip_max is not None
                else torch.zeros_like(unclipped_target_q, dtype=torch.bool)
            )

        if self.config.consolidate_metric_syncs:
            items = [
                ("train/critic_loss", critic_loss),
                ("train/critic_grad_norm", grad_norm),
                ("train/q_mean", current_q.mean()),
                ("train/q_min", current_q.min()),
                ("train/q_max", current_q.max()),
                ("train/target_q_mean", target_q.mean()),
                ("train/next_action_abs_mean", next_full_action.abs().mean()),
            ]
            if lower_clipped is not None and upper_clipped is not None:
                items.extend(
                    [
                        ("train/target_q_min", target_q.min()),
                        ("train/target_q_max", target_q.max()),
                        ("train/target_q_unclipped_min", unclipped_target_q.min()),
                        ("train/target_q_unclipped_max", unclipped_target_q.max()),
                        (
                            "train/target_q_clip_fraction",
                            (lower_clipped | upper_clipped).float().mean(),
                        ),
                        ("train/target_q_clip_lower_fraction", lower_clipped.float().mean()),
                        ("train/target_q_clip_upper_fraction", upper_clipped.float().mean()),
                    ]
                )
            return self._metrics_with_one_device_sync(items)

        metrics = {
            "train/critic_loss": float(critic_loss.detach().item()),
            "train/critic_grad_norm": float(grad_norm.detach().item()),
            "train/q_mean": float(current_q.detach().mean().item()),
            "train/q_min": float(current_q.detach().min().item()),
            "train/q_max": float(current_q.detach().max().item()),
            "train/target_q_mean": float(target_q.detach().mean().item()),
            "train/next_action_abs_mean": float(next_full_action.detach().abs().mean().item()),
        }
        if lower_clipped is not None and upper_clipped is not None:
            metrics.update(
                {
                    "train/target_q_min": float(target_q.detach().min().item()),
                    "train/target_q_max": float(target_q.detach().max().item()),
                    "train/target_q_unclipped_min": float(unclipped_target_q.detach().min().item()),
                    "train/target_q_unclipped_max": float(unclipped_target_q.detach().max().item()),
                    "train/target_q_clip_fraction": float(
                        (lower_clipped | upper_clipped).float().mean().item()
                    ),
                    "train/target_q_clip_lower_fraction": float(
                        lower_clipped.float().mean().item()
                    ),
                    "train/target_q_clip_upper_fraction": float(
                        upper_clipped.float().mean().item()
                    ),
                }
            )
        return metrics

    def update_actor(self, batch: ReplayBatch, collect_metrics: bool = True) -> Dict[str, float]:
        critic_parameters = list(self.critic.parameters())
        freeze_critic = bool(self.config.freeze_critic_during_actor_update)
        if freeze_critic:
            for parameter in critic_parameters:
                parameter.requires_grad_(False)
        try:
            residual = self.actor(batch.states, batch.base_actions)
            full_action = torch.clamp(batch.base_actions + residual, -1.0, 1.0)
            actor_q = self.critic(batch.states, full_action).mean(dim=0)
            actor_loss = -actor_q.mean()
            if not torch.isfinite(actor_loss):
                raise RuntimeError("actor loss is NaN/Inf")
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.actor_grad_clip
            )
            self.actor_optimizer.step()
        finally:
            if freeze_critic:
                for parameter in critic_parameters:
                    parameter.requires_grad_(True)
        self._soft_update(self.actor, self.actor_target)
        if not collect_metrics:
            return {}
        if self.config.consolidate_metric_syncs:
            return self._metrics_with_one_device_sync(
                [
                    ("train/actor_loss", actor_loss),
                    ("train/actor_grad_norm", grad_norm),
                    ("train/residual_abs_mean", residual.abs().mean()),
                    ("train/residual_l2_mean", residual.square().mean()),
                    (
                        "train/full_action_saturation",
                        (full_action.abs() >= 0.999999).float().mean(),
                    ),
                ]
            )
        return {
            "train/actor_loss": float(actor_loss.detach().item()),
            "train/actor_grad_norm": float(grad_norm.detach().item()),
            "train/residual_abs_mean": float(residual.detach().abs().mean().item()),
            "train/residual_l2_mean": float(residual.detach().square().mean().item()),
            "train/full_action_saturation": float(
                (full_action.detach().abs() >= 0.999999).float().mean().item()
            ),
        }

    def update(
        self,
        batch: ReplayBatch,
        update_actor: bool,
        collect_metrics: bool = True,
    ) -> Dict[str, float]:
        metrics = self.update_critic(batch, collect_metrics=collect_metrics)
        if update_actor:
            metrics.update(self.update_actor(batch, collect_metrics=collect_metrics))
        return metrics

    def state_dict(self) -> Dict[str, object]:
        return {
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        saved_config = state.get("config")
        if isinstance(saved_config, dict):
            saved_config = dict(saved_config)
            # Legacy checkpoints remain loadable only with the same effective
            # unclipped configuration.
            saved_config.setdefault("target_q_clip_max", None)
            saved_config.setdefault("target_q_clip_min", None)
            saved_config.setdefault("consolidate_metric_syncs", False)
            saved_config.setdefault("target_critic_subset_compute", False)
            saved_config.setdefault("foreach_soft_update", False)
            saved_config.setdefault("freeze_critic_during_actor_update", False)
            saved_config.setdefault("fused_adamw", False)
        if saved_config != asdict(self.config):
            raise ValueError(
                "learner config mismatch: checkpoint=%r current=%r"
                % (saved_config, asdict(self.config))
            )
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        # Checkpoints are loaded on CPU first so they remain portable. Move
        # Adam's moment tensors to the learner device before the next update.
        for optimizer in (self.actor_optimizer, self.critic_optimizer):
            for optimizer_state in optimizer.state.values():
                for key, value in optimizer_state.items():
                    if torch.is_tensor(value):
                        optimizer_state[key] = value.to(self.device)


class TransitionUpdateScheduler:
    """Convert collected policy decisions into vectorization-invariant updates."""

    def __init__(
        self, critic_updates_per_transition: float = 4.0, actor_updates_per_transition: float = 1.0
    ):
        if critic_updates_per_transition <= 0.0 or actor_updates_per_transition < 0.0:
            raise ValueError("invalid update rates")
        self.critic_rate = float(critic_updates_per_transition)
        self.actor_rate = float(actor_updates_per_transition)
        self.critic_credit = 0.0
        self.actor_credit = 0.0

    def plan(self, decision_transitions: int) -> List[bool]:
        if int(decision_transitions) < 0:
            raise ValueError("decision_transitions must be non-negative")
        self.critic_credit += int(decision_transitions) * self.critic_rate
        self.actor_credit += int(decision_transitions) * self.actor_rate
        critic_updates = int(math.floor(self.critic_credit + 1e-12))
        actor_updates = int(math.floor(self.actor_credit + 1e-12))
        self.critic_credit -= critic_updates
        self.actor_credit -= actor_updates
        if actor_updates > critic_updates:
            raise RuntimeError("actor update credit exceeds critic updates")
        flags = [False] * critic_updates
        if actor_updates > 0:
            # Place actor updates at evenly spaced critic-update boundaries.
            for index in range(actor_updates):
                position = int(math.ceil((index + 1) * critic_updates / float(actor_updates))) - 1
                flags[position] = True
        return flags

    def state_dict(self) -> Dict[str, float]:
        return {
            "critic_rate": self.critic_rate,
            "actor_rate": self.actor_rate,
            "critic_credit": self.critic_credit,
            "actor_credit": self.actor_credit,
        }

    def load_state_dict(self, state: Dict[str, float]) -> None:
        self.critic_credit = float(state["critic_credit"])
        self.actor_credit = float(state["actor_credit"])
