"""Can-only DICE-RL learner on a frozen state-based DSRL diffusion policy.

The transfer deliberately reuses the audited ResFiT state implementation's
network blocks, CPU replay layout, strict checkpoint state, and update-credit
accounting.  The learning equations are DICE-RL: the actor consumes ``(s, z)``,
the frozen diffusion policy maps the same ``(s, z)`` to a base chunk, and the
critic scores the complete ``base + residual`` action chunk.

Upstream references:
  - DICE-RL commit 98d936e52523878f0172e606ea32403176497100
  - ResFiT transfer run commit 70d8794b0120aed288c950fb23b3d97dd832619b
  - DP ExRL-Q formal base 22f7cae576116a19c6fdceb9d210550a8630f7f1
"""

from __future__ import annotations

import copy
import contextlib
import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exrl.common.networks import MLP as ExRLMLP


class FrozenDPNoiseSampler(Protocol):
    """Minimal frozen-prior interface required by :class:`DICERLLearner`."""

    horizon_steps: int
    ac_dim: int

    def sample_from_state_noise(
        self, state: torch.Tensor, noise_action: torch.Tensor
    ) -> torch.Tensor: ...


def _init_resfit_actor_linear(layer: nn.Linear) -> None:
    """Exact hidden-linear initialization used by the audited ResFiT actor."""

    nn.init.orthogonal_(layer.weight, gain=1.0)
    nn.init.zeros_(layer.bias)


class DICEResidualActor(nn.Module):
    """Residual chunk actor ``s_theta(s, z)`` using the ResFiT MLP block."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (2048, 2048, 2048),
        use_layernorm: bool = True,
        squash_output: bool = False,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.squash_output = bool(squash_output)
        self.residual_scale = float(residual_scale)
        dims = [self.state_dim + self.action_dim] + [int(x) for x in hidden_dims]
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            linear = nn.Linear(in_dim, out_dim)
            _init_resfit_actor_linear(linear)
            layers.append(linear)
            if use_layernorm:
                layers.append(nn.LayerNorm(out_dim))
            # Preserve the audited ResFiT block byte-for-byte in module order.
            layers.append(nn.Dropout(0.0))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)
        self.output = nn.Linear(dims[-1], self.action_dim)
        # A zero residual makes the initial DICE policy exactly the frozen DP.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.state_dim or noise.shape[-1] != self.action_dim:
            raise ValueError("DICEResidualActor input shape mismatch")
        residual = self.output(self.trunk(torch.cat([state, noise], dim=-1)))
        if self.squash_output:
            residual = torch.tanh(residual)
        return residual * self.residual_scale


class DICEQMember(nn.Module):
    """One ExRL-style critic member over ``(state, full_action_chunk)``."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        use_layernorm: bool,
    ):
        super().__init__()
        self.backbone = ExRLMLP(
            in_dim=int(state_dim + action_dim),
            out_dim=1,
            hidden_dims=tuple(int(x) for x in hidden_dims),
            activation="relu",
            use_layernorm=bool(use_layernorm),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.backbone(torch.cat([state, action], dim=-1))

    def forward_input(self, critic_input: torch.Tensor) -> torch.Tensor:
        return self.backbone(critic_input)


class DICEEnsembleCritic(nn.Module):
    """Ten independent critic members, returned as ``[ensemble, batch]``."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (512, 512, 512),
        ensemble_size: int = 10,
        use_layernorm: bool = False,
        shared_input: bool = False,
    ):
        super().__init__()
        if int(ensemble_size) <= 0:
            raise ValueError("ensemble_size must be positive")
        self.ensemble_size = int(ensemble_size)
        self.shared_input = bool(shared_input)
        self.members = nn.ModuleList(
            [
                DICEQMember(state_dim, action_dim, hidden_dims, use_layernorm)
                for _ in range(self.ensemble_size)
            ]
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if not self.shared_input:
            return torch.stack(
                [member(state, action).squeeze(-1) for member in self.members], dim=0
            )
        critic_input = torch.cat([state, action], dim=-1)
        return torch.stack(
            [member.forward_input(critic_input).squeeze(-1) for member in self.members],
            dim=0,
        )


class DICEBatchedEnsembleCritic(nn.Module):
    """The same independent critics evaluated with batched GEMMs.

    The parameters are copied from a freshly initialized legacy ensemble, so
    baseline and optimized runs begin from the same ten member weights under
    the same seed. Adam still maintains independent elementwise state for each
    stacked member parameter.
    """

    def __init__(
        self,
        linear_weights: Sequence[torch.Tensor],
        linear_biases: Sequence[torch.Tensor],
        norm_weights: Sequence[torch.Tensor],
        norm_biases: Sequence[torch.Tensor],
    ):
        super().__init__()
        self.linear_weights = nn.ParameterList(
            [nn.Parameter(value.detach().clone()) for value in linear_weights]
        )
        self.linear_biases = nn.ParameterList(
            [nn.Parameter(value.detach().clone()) for value in linear_biases]
        )
        self.norm_weights = nn.ParameterList(
            [nn.Parameter(value.detach().clone()) for value in norm_weights]
        )
        self.norm_biases = nn.ParameterList(
            [nn.Parameter(value.detach().clone()) for value in norm_biases]
        )
        self.ensemble_size = int(self.linear_weights[0].shape[0])

    @classmethod
    def from_legacy(cls, critic: DICEEnsembleCritic):
        member_modules = [list(member.backbone.net) for member in critic.members]
        linear_positions = [
            index for index, module in enumerate(member_modules[0]) if isinstance(module, nn.Linear)
        ]
        norm_positions = [
            index
            for index, module in enumerate(member_modules[0])
            if isinstance(module, nn.LayerNorm)
        ]
        for modules in member_modules[1:]:
            if [type(module) for module in modules] != [
                type(module) for module in member_modules[0]
            ]:
                raise ValueError("critic ensemble member architecture mismatch")
        return cls(
            linear_weights=[
                torch.stack([modules[index].weight for modules in member_modules])
                for index in linear_positions
            ],
            linear_biases=[
                torch.stack([modules[index].bias for modules in member_modules])
                for index in linear_positions
            ],
            norm_weights=[
                torch.stack([modules[index].weight for modules in member_modules])
                for index in norm_positions
            ],
            norm_biases=[
                torch.stack([modules[index].bias for modules in member_modules])
                for index in norm_positions
            ],
        )

    @staticmethod
    def _linear(value: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        if value.ndim == 2:
            return torch.einsum("bi,eoi->ebo", value, weight) + bias[:, None, :]
        return torch.bmm(value, weight.transpose(1, 2)) + bias[:, None, :]

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        value = torch.cat([state, action], dim=-1)
        norm_index = 0
        for index, (weight, bias) in enumerate(zip(self.linear_weights, self.linear_biases)):
            value = self._linear(value, weight, bias)
            if index != len(self.linear_weights) - 1:
                if norm_index < len(self.norm_weights):
                    value = F.layer_norm(value, (value.shape[-1],))
                    value = (
                        value * self.norm_weights[norm_index][:, None, :]
                        + self.norm_biases[norm_index][:, None, :]
                    )
                    norm_index += 1
                value = F.relu(value, inplace=True)
        return value.squeeze(-1)


@dataclass
class MacroTransition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool

    @property
    def boundary(self) -> bool:
        return bool(self.terminated or self.truncated)


@dataclass
class DICETransition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    discount: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool
    mc_return: float


class CompleteEpisodeAccumulator:
    """Materialize macro transitions only after an episode has completed.

    This retains the official DICE requirement that the replay can provide a
    Monte-Carlo return for the BC filter while storing exactly one transition
    per issued DP chunk (not the release code's overlapping primitive windows).
    """

    def __init__(self, n_step: int, gamma: float, n_envs: int):
        if int(n_step) <= 0 or int(n_envs) <= 0:
            raise ValueError("n_step and n_envs must be positive")
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.episodes: List[List[MacroTransition]] = [[] for _ in range(int(n_envs))]

    def append(self, env_index: int, transition: MacroTransition) -> List[DICETransition]:
        episode = self.episodes[int(env_index)]
        episode.append(transition)
        if not transition.boundary:
            return []
        emitted = self._finish_episode(episode)
        episode.clear()
        return emitted

    def _finish_episode(self, episode: Sequence[MacroTransition]) -> List[DICETransition]:
        if not episode or not episode[-1].boundary:
            raise RuntimeError("cannot finish an incomplete episode")
        mc_returns = np.zeros((len(episode),), dtype=np.float64)
        running = 0.0
        for index in range(len(episode) - 1, -1, -1):
            running = float(episode[index].reward) + self.gamma * running
            mc_returns[index] = running

        emitted: List[DICETransition] = []
        for start, first in enumerate(episode):
            reward = 0.0
            multiplier = 1.0
            final = first
            steps = 0
            for item in episode[start : start + self.n_step]:
                reward += multiplier * float(item.reward)
                final = item
                steps += 1
                if item.boundary:
                    break
                multiplier *= self.gamma
            discount = 0.0 if final.terminated else self.gamma**steps
            emitted.append(
                DICETransition(
                    state=np.asarray(first.state, dtype=np.float32),
                    action=np.asarray(first.action, dtype=np.float32),
                    reward=float(reward),
                    discount=float(discount),
                    next_state=np.asarray(final.next_state, dtype=np.float32),
                    terminated=bool(final.terminated),
                    truncated=bool(final.truncated),
                    mc_return=float(mc_returns[start]),
                )
            )
        return emitted

    def reset(self, env_index: Optional[int] = None) -> None:
        indices: Iterable[int] = (
            range(len(self.episodes)) if env_index is None else [int(env_index)]
        )
        for index in indices:
            self.episodes[index].clear()

    def pending_lengths(self) -> List[int]:
        return [len(episode) for episode in self.episodes]


@dataclass
class DICEReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    mc_returns: torch.Tensor
    data_source: torch.Tensor


class DICEReplayBuffer:
    """CPU NumPy ring buffer copied from the audited ResFiT placement model."""

    def __init__(self, state_dim: int, action_dim: int, capacity: int, seed: int = 0):
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.states = np.empty((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.empty((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.empty((self.capacity,), dtype=np.float32)
        self.discounts = np.empty((self.capacity,), dtype=np.float32)
        self.next_states = np.empty((self.capacity, self.state_dim), dtype=np.float32)
        self.terminated = np.empty((self.capacity,), dtype=np.bool_)
        self.truncated = np.empty((self.capacity,), dtype=np.bool_)
        self.mc_returns = np.empty((self.capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.RandomState(int(seed))
        self._device_cache = None
        self._device_cache_device = None

    def __len__(self) -> int:
        return int(self.size)

    def add(self, transition: DICETransition) -> None:
        state = np.asarray(transition.state, dtype=np.float32)
        action = np.asarray(transition.action, dtype=np.float32)
        next_state = np.asarray(transition.next_state, dtype=np.float32)
        if state.shape != (self.state_dim,) or next_state.shape != (self.state_dim,):
            raise ValueError("replay transition state shape mismatch")
        if action.shape != (self.action_dim,):
            raise ValueError("replay transition action shape mismatch")
        if not (
            np.all(np.isfinite(state))
            and np.all(np.isfinite(action))
            and np.all(np.isfinite(next_state))
            and np.isfinite(float(transition.reward))
            and np.isfinite(float(transition.discount))
            and np.isfinite(float(transition.mc_return))
        ):
            raise ValueError("replay transition contains NaN/Inf")
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = transition.reward
        self.discounts[index] = transition.discount
        self.next_states[index] = next_state
        self.terminated[index] = transition.terminated
        self.truncated[index] = transition.truncated
        self.mc_returns[index] = transition.mc_return
        if self._device_cache is not None:
            values = {
                "states": state,
                "actions": action,
                "rewards": transition.reward,
                "discounts": transition.discount,
                "next_states": next_state,
                "terminated": transition.terminated,
                "truncated": transition.truncated,
                "mc_returns": transition.mc_return,
            }
            for name, value in values.items():
                self._device_cache[name][index].copy_(
                    torch.as_tensor(value, device=self._device_cache_device)
                )
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def extend(self, transitions: Iterable[DICETransition]) -> int:
        count = 0
        for transition in transitions:
            self.add(transition)
            count += 1
        return count

    def enable_device_cache(self, device: torch.device) -> None:
        """Mirror replay arrays once so hot-path sampling stays on the GPU."""

        device = torch.device(device)
        if device.type != "cuda":
            return
        specifications = {
            "states": (self.states, torch.float32),
            "actions": (self.actions, torch.float32),
            "rewards": (self.rewards, torch.float32),
            "discounts": (self.discounts, torch.float32),
            "next_states": (self.next_states, torch.float32),
            "terminated": (self.terminated, torch.bool),
            "truncated": (self.truncated, torch.bool),
            "mc_returns": (self.mc_returns, torch.float32),
        }
        cache = {}
        for name, (array, dtype) in specifications.items():
            cache[name] = torch.empty(array.shape, dtype=dtype, device=device)
            if self.size:
                cache[name][: self.size].copy_(
                    torch.as_tensor(array[: self.size], device=device, dtype=dtype)
                )
        self._device_cache = cache
        self._device_cache_device = device

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        data_source: int,
        use_device_cache: bool = False,
    ) -> DICEReplayBatch:
        if self.size <= 0:
            raise RuntimeError("cannot sample an empty replay")
        indices = self.rng.randint(0, self.size, size=int(batch_size))

        if use_device_cache:
            if self._device_cache is None or torch.device(device) != self._device_cache_device:
                raise RuntimeError("replay device cache is not enabled on requested device")
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)

            def tensor(array, dtype=torch.float32):
                del dtype
                name = next(
                    key
                    for key, candidate in (
                        ("states", self.states),
                        ("actions", self.actions),
                        ("rewards", self.rewards),
                        ("discounts", self.discounts),
                        ("next_states", self.next_states),
                        ("terminated", self.terminated),
                        ("truncated", self.truncated),
                        ("mc_returns", self.mc_returns),
                    )
                    if candidate is array
                )
                return torch.index_select(self._device_cache[name], 0, index_tensor)

        else:

            def tensor(array, dtype=torch.float32):
                return torch.as_tensor(array[indices], dtype=dtype, device=device)

        return DICEReplayBatch(
            states=tensor(self.states),
            actions=tensor(self.actions),
            rewards=tensor(self.rewards),
            discounts=tensor(self.discounts),
            next_states=tensor(self.next_states),
            terminated=tensor(self.terminated, dtype=torch.bool),
            truncated=tensor(self.truncated, dtype=torch.bool),
            mc_returns=tensor(self.mc_returns),
            data_source=torch.full(
                (int(batch_size),), int(data_source), dtype=torch.long, device=device
            ),
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
            "actions": self.actions[:size].copy(),
            "rewards": self.rewards[:size].copy(),
            "discounts": self.discounts[:size].copy(),
            "next_states": self.next_states[:size].copy(),
            "terminated": self.terminated[:size].copy(),
            "truncated": self.truncated[:size].copy(),
            "mc_returns": self.mc_returns[:size].copy(),
            "rng_state": self.rng.get_state(),
        }

    def load_state_dict(self, state: Dict[str, object], strict_capacity: bool = True) -> None:
        if int(state["state_dim"]) != self.state_dim or int(state["action_dim"]) != self.action_dim:
            raise ValueError("replay shape mismatch")
        saved_capacity = int(state["capacity"])
        if strict_capacity and saved_capacity != self.capacity:
            raise ValueError("replay capacity mismatch")
        size = int(state["size"])
        if size > self.capacity:
            raise ValueError("saved replay exceeds configured capacity")
        saved_position = int(state["position"])
        if not strict_capacity and saved_position != size % saved_capacity:
            raise ValueError(
                "capacity-independent replay load requires a contiguous, non-wrapped payload"
            )
        for name in (
            "states",
            "actions",
            "rewards",
            "discounts",
            "next_states",
            "terminated",
            "truncated",
            "mc_returns",
        ):
            getattr(self, name)[:size] = state[name]
        self.size = size
        self.position = saved_position % self.capacity if strict_capacity else size % self.capacity
        self.rng.set_state(state["rng_state"])
        self._device_cache = None
        self._device_cache_device = None


def concatenate_batches(first: DICEReplayBatch, second: DICEReplayBatch) -> DICEReplayBatch:
    def cat(name: str):
        return torch.cat([getattr(first, name), getattr(second, name)], dim=0)

    return DICEReplayBatch(
        states=cat("states"),
        actions=cat("actions"),
        rewards=cat("rewards"),
        discounts=cat("discounts"),
        next_states=cat("next_states"),
        terminated=cat("terminated"),
        truncated=cat("truncated"),
        mc_returns=cat("mc_returns"),
        data_source=cat("data_source"),
    )


def sample_rlpd_batch(
    online_replay: DICEReplayBuffer,
    offline_replay: Optional[DICEReplayBuffer],
    batch_size: int,
    offline_ratio: float,
    device: torch.device,
    use_device_cache: bool = False,
) -> DICEReplayBatch:
    """Match release integer splitting while keeping both replays on CPU."""

    ratio = float(offline_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("offline_ratio must be in [0, 1]")
    offline_count = int(int(batch_size) * ratio)
    online_count = int(batch_size) - offline_count
    if offline_count == 0:
        return online_replay.sample(
            online_count, device, data_source=0, use_device_cache=use_device_cache
        )
    if offline_replay is None:
        raise ValueError("positive offline_ratio requires an offline replay")
    if online_count == 0:
        return offline_replay.sample(
            offline_count, device, data_source=1, use_device_cache=use_device_cache
        )
    offline = offline_replay.sample(
        offline_count, device, data_source=1, use_device_cache=use_device_cache
    )
    online = online_replay.sample(
        online_count, device, data_source=0, use_device_cache=use_device_cache
    )
    return concatenate_batches(offline, online)


class LinearOfflineRatioSchedule:
    """Linear RLPD schedule indexed by actual primitive transitions."""

    def __init__(self, start: float = 0.7, end: float = 0.1, decay_steps: int = 160_000):
        if not (0.0 <= float(end) <= float(start) <= 1.0):
            raise ValueError("invalid offline ratio endpoints")
        if int(decay_steps) <= 0:
            raise ValueError("decay_steps must be positive")
        self.start = float(start)
        self.end = float(end)
        self.decay_steps = int(decay_steps)

    def value(self, primitive_transitions: int) -> float:
        progress = min(max(int(primitive_transitions), 0) / self.decay_steps, 1.0)
        return self.start + progress * (self.end - self.start)


class TransitionUpdateScheduler:
    """ResFiT credit scheduler, used here to realize DICE's actual UTD.

    Fractional per-transition rates are accumulated exactly across collection
    steps. Actor updates are spread evenly through each critic burst instead of
    being released as one contiguous block; long-run update counts are exact.
    """

    def __init__(
        self,
        critic_updates_per_transition: float = 2.5,
        actor_updates_per_transition: float = 1.25,
    ):
        if critic_updates_per_transition <= 0.0 or not (
            0.0 <= actor_updates_per_transition <= critic_updates_per_transition
        ):
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
        for index in range(actor_updates):
            position = int(math.ceil((index + 1) * critic_updates / actor_updates)) - 1
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
        if (
            float(state["critic_rate"]) != self.critic_rate
            or float(state["actor_rate"]) != self.actor_rate
        ):
            raise ValueError("update scheduler rate mismatch")
        self.critic_credit = float(state["critic_credit"])
        self.actor_credit = float(state["actor_credit"])


@dataclass
class DICERLLearnerConfig:
    state_dim: int = 23
    action_dim: int = 7
    horizon_steps: int = 32
    actor_hidden_dims: Tuple[int, ...] = (2048, 2048, 2048)
    critic_hidden_dims: Tuple[int, ...] = (512, 512, 512)
    critic_ensemble_size: int = 10
    critic_use_layernorm: bool = False
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    actor_weight_decay: float = 1e-5
    critic_weight_decay: float = 1e-5
    tau: float = 0.01
    beta: float = 50.0
    actor_num_noise_samples: int = 16
    target_num_noise_samples: int = 4
    rollout_num_noise_samples: int = 16
    q_filter_enabled: bool = True
    q_filter_epsilon: float = -0.5
    q_filter_warmup_primitive_steps: int = 32_000
    best_of_n_warmup_primitive_steps: int = 64_000
    q_normalization: bool = True
    disable_actor_q_loss_for_offline: bool = True
    always_retain_bc_for_offline: bool = False
    actor_grad_clip: float = 1.0
    critic_grad_clip: float = 1.0
    squash_residual: bool = False
    residual_scale: float = 1.0
    prior_microbatch_size: int = 512
    optimization_level: str = "baseline"
    phase2_amp: bool = True
    phase2_compile: bool = True

    @property
    def flat_action_dim(self) -> int:
        return int(self.action_dim * self.horizon_steps)


class DICERLLearner:
    """DICE-RL actor/critic around a frozen explicit-noise DP sampler."""

    def __init__(
        self,
        config: DICERLLearnerConfig,
        frozen_sampler: FrozenDPNoiseSampler,
        device: torch.device,
    ):
        self.config = config
        self.device = torch.device(device)
        self.frozen_sampler = frozen_sampler
        if config.optimization_level not in ("baseline", "phase1", "phase2"):
            raise ValueError("invalid optimization_level")
        self.runtime_optimized = config.optimization_level != "baseline"
        self.mixed_precision = config.optimization_level == "phase2" and bool(config.phase2_amp)
        self.compile_enabled = config.optimization_level == "phase2" and bool(config.phase2_compile)
        self._amp_dtype = None
        if self.mixed_precision and self.device.type == "cuda":
            major, _minor = torch.cuda.get_device_capability(self.device)
            self._amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        if int(frozen_sampler.horizon_steps) != int(config.horizon_steps):
            raise ValueError("frozen DP / learner horizon mismatch")
        if int(frozen_sampler.ac_dim) != int(config.action_dim):
            raise ValueError("frozen DP / learner action dimension mismatch")
        self.actor = DICEResidualActor(
            state_dim=config.state_dim,
            action_dim=config.flat_action_dim,
            hidden_dims=config.actor_hidden_dims,
            use_layernorm=True,
            squash_output=config.squash_residual,
            residual_scale=config.residual_scale,
        ).to(self.device)
        legacy_critic = DICEEnsembleCritic(
            state_dim=config.state_dim,
            action_dim=config.flat_action_dim,
            hidden_dims=config.critic_hidden_dims,
            ensemble_size=config.critic_ensemble_size,
            use_layernorm=config.critic_use_layernorm,
            shared_input=self.runtime_optimized,
        )
        self.critic = (
            DICEBatchedEnsembleCritic.from_legacy(legacy_critic)
            if self.runtime_optimized
            else legacy_critic
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.train(False)
        optimizer_options = {}
        if self.runtime_optimized and self.device.type == "cuda":
            optimizer_options["fused"] = True
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=float(config.actor_lr),
            weight_decay=float(config.actor_weight_decay),
            **optimizer_options,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(config.critic_lr),
            weight_decay=float(config.critic_weight_decay),
            **optimizer_options,
        )
        scaler_enabled = self._amp_dtype == torch.float16
        self._grad_scaling = scaler_enabled
        if hasattr(torch.amp, "GradScaler"):
            self.actor_scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
            self.critic_scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:
            self.actor_scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
            self.critic_scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        self._actor_call = self.actor
        self._critic_call = self.critic
        self._critic_target_call = self.critic_target
        if self.compile_enabled and self.device.type == "cuda" and hasattr(torch, "compile"):
            compile_options = {
                "dynamic": False,
                "fullgraph": False,
                "mode": "reduce-overhead",
            }
            self._actor_call = torch.compile(self.actor, **compile_options)
            self._critic_call = torch.compile(self.critic, **compile_options)
            self._critic_target_call = torch.compile(self.critic_target, **compile_options)

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            enabled=self._amp_dtype is not None and self.device.type == "cuda",
        )

    def _assert_finite(self, value: torch.Tensor, label: str) -> None:
        finite = torch.isfinite(value).all()
        if self.runtime_optimized and hasattr(torch, "_assert_async"):
            torch._assert_async(finite, "%s is NaN/Inf" % label)
        elif not bool(finite):
            raise RuntimeError("%s is NaN/Inf" % label)

    @staticmethod
    def _make_gradients_contiguous(parameters) -> None:
        for parameter in parameters:
            if parameter.grad is not None and (
                parameter.grad.layout != parameter.layout
                or parameter.grad.stride() != parameter.stride()
            ):
                matching = torch.empty_strided(
                    parameter.shape,
                    parameter.stride(),
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
                matching.copy_(parameter.grad)
                parameter.grad = matching

    @contextlib.contextmanager
    def _critic_parameters_frozen(self):
        parameters = tuple(self.critic.parameters())
        previous = tuple(parameter.requires_grad for parameter in parameters)
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
            yield
        finally:
            for parameter, requires_grad in zip(parameters, previous):
                parameter.requires_grad_(requires_grad)

    def _flat_noise(self, noise: torch.Tensor) -> torch.Tensor:
        return noise.reshape(noise.shape[0], self.config.flat_action_dim)

    @torch.no_grad()
    def _sample_prior(self, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError("prior state must be [B, state_dim]")
        expected = (
            state.shape[0],
            self.config.horizon_steps,
            self.config.action_dim,
        )
        if tuple(noise.shape) != expected:
            raise ValueError("prior noise shape mismatch")
        outputs = []
        micro = max(1, int(self.config.prior_microbatch_size))
        for start in range(0, state.shape[0], micro):
            stop = min(start + micro, state.shape[0])
            outputs.append(
                self.frozen_sampler.sample_from_state_noise(state[start:stop], noise[start:stop])
            )
        action = torch.cat(outputs, dim=0)
        if tuple(action.shape) != expected:
            raise RuntimeError("frozen DP returned invalid action chunks")
        self._assert_finite(action, "frozen DP action")
        return action.reshape(action.shape[0], self.config.flat_action_dim)

    def _candidates(
        self, state: torch.Tensor, count: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = state.shape[0]
        count = int(count)
        noise = torch.randn(
            (batch, count, self.config.horizon_steps, self.config.action_dim),
            device=self.device,
        )
        repeated_state = (
            state[:, None, :].expand(-1, count, -1).reshape(batch * count, self.config.state_dim)
        )
        flat_noise_chunk = noise.reshape(
            batch * count, self.config.horizon_steps, self.config.action_dim
        )
        base = self._sample_prior(repeated_state, flat_noise_chunk)
        with self._autocast():
            residual = self._actor_call(repeated_state, self._flat_noise(flat_noise_chunk))
        residual = residual.to(torch.float32)
        action = base + residual
        return (
            action.reshape(batch, count, self.config.flat_action_dim),
            base.reshape(batch, count, self.config.flat_action_dim),
            residual.reshape(batch, count, self.config.flat_action_dim),
        )

    @torch.no_grad()
    def select_action(
        self,
        states: np.ndarray,
        primitive_transitions: int,
        eval_mode: bool = False,
        candidate_count_override: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        state = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        use_best = bool(eval_mode) or int(primitive_transitions) > int(
            self.config.best_of_n_warmup_primitive_steps
        )
        count = (
            int(candidate_count_override)
            if candidate_count_override is not None
            else (self.config.rollout_num_noise_samples if use_best else 1)
        )
        if count <= 0:
            raise ValueError("candidate_count_override must be positive")
        actions, bases, residuals = self._candidates(state, count)
        if count == 1:
            selected = torch.zeros((state.shape[0],), dtype=torch.long, device=self.device)
            with self._autocast():
                selected_q = self._critic_call(state, actions[:, 0]).min(dim=0).values
        else:
            flat_state = (
                state[:, None, :]
                .expand(-1, count, -1)
                .reshape(state.shape[0] * count, self.config.state_dim)
            )
            with self._autocast():
                all_q = self._critic_call(
                    flat_state, actions.reshape(-1, self.config.flat_action_dim)
                )
            conservative = all_q.min(dim=0).values.reshape(state.shape[0], count)
            selected = conservative.argmax(dim=1)
            selected_q = conservative.gather(1, selected[:, None]).squeeze(1)
        rows = torch.arange(state.shape[0], device=self.device)
        return {
            "action": actions[rows, selected]
            .reshape(-1, self.config.horizon_steps, self.config.action_dim)
            .cpu()
            .numpy()
            .astype(np.float32),
            "base_action": bases[rows, selected]
            .reshape(-1, self.config.horizon_steps, self.config.action_dim)
            .cpu()
            .numpy()
            .astype(np.float32),
            "residual": residuals[rows, selected]
            .reshape(-1, self.config.horizon_steps, self.config.action_dim)
            .cpu()
            .numpy()
            .astype(np.float32),
            "selected_q": selected_q.cpu().numpy().astype(np.float32),
            "candidate_count": np.full((state.shape[0],), count, dtype=np.int64),
        }

    def _target_value(self, next_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            actions, _, _ = self._candidates(next_states, self.config.target_num_noise_samples)
            batch, count, _ = actions.shape
            state = (
                next_states[:, None, :]
                .expand(-1, count, -1)
                .reshape(batch * count, self.config.state_dim)
            )
            with self._autocast():
                all_q = self._critic_target_call(
                    state, actions.reshape(batch * count, self.config.flat_action_dim)
                )
            conservative = all_q.min(dim=0).values.reshape(batch, count)
            return conservative.mean(dim=1)

    def update_critic(
        self, batch: DICEReplayBatch, collect_metrics: bool = True
    ) -> Dict[str, float]:
        target_value = self._target_value(batch.next_states)
        target_q = batch.rewards + batch.discounts * target_value
        with self._autocast():
            current_q = self._critic_call(batch.states, batch.actions)
        # Official DICE sums the independent member losses.
        critic_loss = (current_q.float() - target_q.float().unsqueeze(0)).square().mean(dim=1).sum()
        self._assert_finite(critic_loss, "critic loss")
        self.critic_optimizer.zero_grad(set_to_none=True)
        if self._grad_scaling:
            self.critic_scaler.scale(critic_loss).backward()
            self.critic_scaler.unscale_(self.critic_optimizer)
        else:
            critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), float(self.config.critic_grad_clip)
        )
        if self.runtime_optimized and self.device.type == "cuda":
            self._make_gradients_contiguous(self.critic.parameters())
        if self._grad_scaling:
            self.critic_scaler.step(self.critic_optimizer)
            self.critic_scaler.update()
        else:
            self.critic_optimizer.step()
        if not collect_metrics:
            return {}
        return {
            "train/critic_loss": float(critic_loss.detach()),
            "train/critic_grad_norm": float(grad_norm.detach()),
            "train/q_mean": float(current_q.detach().mean()),
            "train/q_min": float(current_q.detach().min()),
            "train/q_max": float(current_q.detach().max()),
            "train/target_q_mean": float(target_q.detach().mean()),
        }

    def update_actor(
        self,
        batch: DICEReplayBatch,
        primitive_transitions: int,
        collect_metrics: bool = True,
    ) -> Dict[str, float]:
        actions, bases, residuals = self._candidates(
            batch.states, self.config.actor_num_noise_samples
        )
        batch_size, count, _ = actions.shape
        state = (
            batch.states[:, None, :]
            .expand(-1, count, -1)
            .reshape(batch_size * count, self.config.state_dim)
        )
        freeze_context = (
            self._critic_parameters_frozen() if self.runtime_optimized else contextlib.nullcontext()
        )
        with freeze_context:
            with self._autocast():
                current_q = (
                    self._critic_call(
                        state,
                        actions.reshape(batch_size * count, self.config.flat_action_dim),
                    )
                    .min(dim=0)
                    .values.reshape(batch_size, count)
                )
            current_q = current_q.float()

            q_per_state = current_q.mean(dim=1)
            q_mask = torch.ones_like(q_per_state)
            if self.config.disable_actor_q_loss_for_offline:
                q_mask = (batch.data_source == 0).float()
            q_loss = -(q_per_state * q_mask).mean()
            online_count = q_mask.sum()
            if self.runtime_optimized:
                masked_q_abs = (current_q.abs().mean(dim=1) * q_mask).sum() / torch.clamp(
                    online_count, min=1.0
                )
                q_abs = torch.where(
                    online_count > 0.0,
                    masked_q_abs,
                    current_q.abs().mean(),
                )
            elif float(online_count.detach().item()) > 0.0:
                q_abs = (current_q.abs().mean(dim=1) * q_mask).sum() / online_count
            else:
                q_abs = current_q.abs().mean()
            if self.config.q_normalization:
                if self.runtime_optimized:
                    q_loss = torch.where(
                        q_abs.detach() > 1e-8,
                        q_loss / torch.clamp(q_abs.detach(), min=1e-8),
                        q_loss,
                    )
                elif float(q_abs.detach().item()) > 1e-8:
                    q_loss = q_loss / q_abs.detach()

            residual_mse = residuals.float().square().mean(dim=-1)
            bc_keep = torch.ones_like(residual_mse)
            filter_active = bool(self.config.q_filter_enabled) and int(primitive_transitions) > int(
                self.config.q_filter_warmup_primitive_steps
            )
            if filter_active:
                with torch.no_grad(), self._autocast():
                    base_q = (
                        self._critic_call(
                            state,
                            bases.reshape(batch_size * count, self.config.flat_action_dim),
                        )
                        .min(dim=0)
                        .values.reshape(batch_size, count)
                    )
                    base_q = base_q.float()
                    consistent = current_q - batch.mc_returns[:, None] <= float(
                        self.config.q_filter_epsilon
                    )
                    improved = current_q >= base_q
                    disable_bc = improved & consistent
                    bc_keep = (~disable_bc).float()
                    if self.config.always_retain_bc_for_offline:
                        bc_keep = torch.where(
                            (batch.data_source == 1)[:, None],
                            torch.ones_like(bc_keep),
                            bc_keep,
                        )
            bc_loss = (bc_keep * residual_mse).mean()
            actor_loss = q_loss + float(self.config.beta) * bc_loss
            self._assert_finite(actor_loss, "actor loss")
            self.actor_optimizer.zero_grad(set_to_none=True)
            if self._grad_scaling:
                self.actor_scaler.scale(actor_loss).backward()
                self.actor_scaler.unscale_(self.actor_optimizer)
            else:
                actor_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), float(self.config.actor_grad_clip)
            )
            if self.runtime_optimized and self.device.type == "cuda":
                self._make_gradients_contiguous(self.actor.parameters())
            if self._grad_scaling:
                self.actor_scaler.step(self.actor_optimizer)
                self.actor_scaler.update()
            else:
                self.actor_optimizer.step()
        if not collect_metrics:
            return {}
        return {
            "train/actor_loss": float(actor_loss.detach()),
            "train/actor_q_loss": float(q_loss.detach()),
            "train/actor_bc_loss": float(bc_loss.detach()),
            "train/actor_grad_norm": float(grad_norm.detach()),
            "train/residual_abs_mean": float(residuals.detach().abs().mean()),
            "train/residual_l2_mean": float(residuals.detach().square().mean()),
            "train/bc_keep_fraction": float(bc_keep.detach().mean()),
            "train/q_filter_active": float(filter_active),
        }

    def _soft_update_target(self) -> None:
        tau = float(self.config.tau)
        with torch.no_grad():
            source = [parameter.data for parameter in self.critic.parameters()]
            target = [parameter.data for parameter in self.critic_target.parameters()]
            if self.runtime_optimized:
                torch._foreach_mul_(target, 1.0 - tau)
                torch._foreach_add_(target, source, alpha=tau)
            else:
                for source_parameter, target_parameter in zip(source, target):
                    target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)

    def update(
        self,
        batch: DICEReplayBatch,
        update_actor: bool,
        update_target: bool,
        primitive_transitions: int,
        collect_metrics: bool = True,
    ) -> Dict[str, float]:
        metrics = self.update_critic(batch, collect_metrics=collect_metrics)
        if update_actor:
            metrics.update(
                self.update_actor(
                    batch,
                    primitive_transitions,
                    collect_metrics=collect_metrics,
                )
            )
        if update_target:
            self._soft_update_target()
        return metrics

    def state_dict(self) -> Dict[str, object]:
        return {
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_scaler": self.actor_scaler.state_dict(),
            "critic_scaler": self.critic_scaler.state_dict(),
        }

    @staticmethod
    def _algorithm_config(config: Dict[str, object]) -> Dict[str, object]:
        output = dict(config)
        for name in (
            "optimization_level",
            "phase2_amp",
            "phase2_compile",
            "prior_microbatch_size",
        ):
            output.pop(name, None)
        return output

    def _legacy_critic_template(self) -> DICEEnsembleCritic:
        return DICEEnsembleCritic(
            state_dim=self.config.state_dim,
            action_dim=self.config.flat_action_dim,
            hidden_dims=self.config.critic_hidden_dims,
            ensemble_size=self.config.critic_ensemble_size,
            use_layernorm=self.config.critic_use_layernorm,
            shared_input=True,
        )

    @staticmethod
    def _batched_to_legacy_parameter_names(
        legacy: DICEEnsembleCritic,
    ) -> Dict[str, List[str]]:
        modules = list(legacy.members[0].backbone.net)
        linear_positions = [
            index for index, module in enumerate(modules) if isinstance(module, nn.Linear)
        ]
        norm_positions = [
            index for index, module in enumerate(modules) if isinstance(module, nn.LayerNorm)
        ]
        mapping = {}
        for parameter_group, positions in (
            ("linear", linear_positions),
            ("norm", norm_positions),
        ):
            for index, position in enumerate(positions):
                for suffix in ("weight", "bias"):
                    collection = "weights" if suffix == "weight" else "biases"
                    mapping["%s_%s.%d" % (parameter_group, collection, index)] = [
                        "members.%d.backbone.net.%d.%s" % (member, position, suffix)
                        for member in range(legacy.ensemble_size)
                    ]
        return mapping

    def _convert_legacy_critic_optimizer_state(
        self,
        saved_optimizer: Dict[str, object],
        legacy: DICEEnsembleCritic,
    ) -> Dict[str, object]:
        old_names = [name for name, _parameter in legacy.named_parameters()]
        old_ids = [
            parameter_id
            for group in saved_optimizer["param_groups"]
            for parameter_id in group["params"]
        ]
        if len(old_names) != len(old_ids):
            raise ValueError("legacy critic optimizer parameter count mismatch")
        old_state_by_name = {
            name: saved_optimizer["state"].get(parameter_id, {})
            for name, parameter_id in zip(old_names, old_ids)
        }

        current = self.critic_optimizer.state_dict()
        current_names = [name for name, _parameter in self.critic.named_parameters()]
        current_ids = [
            parameter_id for group in current["param_groups"] for parameter_id in group["params"]
        ]
        mapping = self._batched_to_legacy_parameter_names(legacy)
        converted_state = {}
        for current_name, current_id in zip(current_names, current_ids):
            source_names = mapping[current_name]
            source_states = [old_state_by_name[name] for name in source_names]
            if not source_states or not source_states[0]:
                converted_state[current_id] = {}
                continue
            if any(set(value) != set(source_states[0]) for value in source_states):
                raise ValueError("legacy critic optimizer state keys mismatch")
            converted = {}
            for key in source_states[0]:
                values = [value[key] for value in source_states]
                if torch.is_tensor(values[0]) and values[0].ndim > 0:
                    converted[key] = torch.stack(values, dim=0)
                elif torch.is_tensor(values[0]):
                    if not all(torch.equal(values[0], value) for value in values[1:]):
                        raise ValueError("legacy optimizer scalar state mismatch")
                    converted[key] = values[0].clone()
                else:
                    if not all(values[0] == value for value in values[1:]):
                        raise ValueError("legacy optimizer state mismatch")
                    converted[key] = values[0]
            converted_state[current_id] = converted

        if len(saved_optimizer["param_groups"]) != len(current["param_groups"]):
            raise ValueError("legacy critic optimizer group count mismatch")
        for current_group, saved_group in zip(
            current["param_groups"], saved_optimizer["param_groups"]
        ):
            for key in ("lr", "betas", "eps", "weight_decay", "amsgrad"):
                if key in saved_group:
                    current_group[key] = saved_group[key]
        current["state"] = converted_state
        return current

    def load_state_dict(
        self,
        state: Dict[str, object],
        allow_runtime_migration: bool = False,
    ) -> None:
        saved_config = dict(state.get("config", {}))
        current_config = asdict(self.config)
        exact_config = saved_config == current_config
        migrate_legacy_runtime = (
            bool(allow_runtime_migration)
            and self.runtime_optimized
            and self._algorithm_config(saved_config) == self._algorithm_config(current_config)
            and any(str(key).startswith("members.") for key in state["critic"])
        )
        if not exact_config and not migrate_legacy_runtime:
            raise ValueError("learner config mismatch")
        self.actor.load_state_dict(state["actor"])
        if migrate_legacy_runtime:
            legacy = self._legacy_critic_template()
            legacy.load_state_dict(state["critic"])
            converted = DICEBatchedEnsembleCritic.from_legacy(legacy)
            self.critic.load_state_dict(converted.state_dict())
            legacy_target = self._legacy_critic_template()
            legacy_target.load_state_dict(state["critic_target"])
            converted_target = DICEBatchedEnsembleCritic.from_legacy(legacy_target)
            self.critic_target.load_state_dict(converted_target.state_dict())
        else:
            self.critic.load_state_dict(state["critic"])
            self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if migrate_legacy_runtime:
            self.critic_optimizer.load_state_dict(
                self._convert_legacy_critic_optimizer_state(state["critic_optimizer"], legacy)
            )
        else:
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if "actor_scaler" in state:
            self.actor_scaler.load_state_dict(state["actor_scaler"])
        if "critic_scaler" in state:
            self.critic_scaler.load_state_dict(state["critic_scaler"])
        for optimizer in (self.actor_optimizer, self.critic_optimizer):
            if self.runtime_optimized and self.device.type == "cuda":
                for group in optimizer.param_groups:
                    group["fused"] = True
                    group["foreach"] = None
            for optimizer_state in optimizer.state.values():
                for key, value in optimizer_state.items():
                    if torch.is_tensor(value):
                        optimizer_state[key] = value.to(self.device)
