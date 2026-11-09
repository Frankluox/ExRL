"""Four-task DICE-RL learner on a frozen state-based ACT policy.

The transfer deliberately reuses the audited ResFiT state implementation's
network blocks, CPU replay layout, strict checkpoint state, and update-credit
accounting.  The learning equations are DICE-RL: the actor consumes ``(s, z)``,
the frozen ACT policy maps ``s`` to its deterministic zero-latent base chunk,
and the critic scores the complete ``base + residual`` action chunk. The same
Gaussian ``z`` still conditions the residual actor; ACT intentionally ignores
it so zero residual exactly recovers the audited ACT ExRL-Q policy for both VAE
and no-VAE checkpoints.

Upstream references:
  - DICE-RL commit 98d936e52523878f0172e606ea32403176497100
  - ResFiT transfer run commit 70d8794b0120aed288c950fb23b3d97dd832619b
  - ACT ExRL-Q formal base 22f7cae576116a19c6fdceb9d210550a8630f7f1
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from exrl.common.networks import MLP as ExRLMLP


class FrozenACTNoiseSampler(Protocol):
    """Minimal frozen-prior interface required by :class:`DICERLLearner`."""

    horizon_steps: int
    ac_dim: int
    noise_conditioned: bool

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
        # A zero residual makes the initial DICE policy exactly the frozen ACT.
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


class DICEEnsembleCritic(nn.Module):
    """Ten independent critic members, returned as ``[ensemble, batch]``."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (512, 512, 512),
        ensemble_size: int = 10,
        use_layernorm: bool = False,
    ):
        super().__init__()
        if int(ensemble_size) <= 0:
            raise ValueError("ensemble_size must be positive")
        self.ensemble_size = int(ensemble_size)
        self.members = nn.ModuleList(
            [
                DICEQMember(state_dim, action_dim, hidden_dims, use_layernorm)
                for _ in range(self.ensemble_size)
            ]
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.stack([member(state, action).squeeze(-1) for member in self.members], dim=0)


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
    per issued frozen-prior chunk (not the release code's overlapping windows).
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
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def extend(self, transitions: Iterable[DICETransition]) -> int:
        count = 0
        for transition in transitions:
            self.add(transition)
            count += 1
        return count

    def sample(self, batch_size: int, device: torch.device, data_source: int) -> DICEReplayBatch:
        if self.size <= 0:
            raise RuntimeError("cannot sample an empty replay")
        indices = self.rng.randint(0, self.size, size=int(batch_size))

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
    offline_replay: DICEReplayBuffer,
    batch_size: int,
    offline_ratio: float,
    device: torch.device,
) -> DICEReplayBatch:
    """Match release integer splitting while keeping both replays on CPU."""

    ratio = float(offline_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("offline_ratio must be in [0, 1]")
    offline_count = int(int(batch_size) * ratio)
    online_count = int(batch_size) - offline_count
    if offline_count == 0:
        return online_replay.sample(online_count, device, data_source=0)
    if online_count == 0:
        return offline_replay.sample(offline_count, device, data_source=1)
    offline = offline_replay.sample(offline_count, device, data_source=1)
    online = online_replay.sample(online_count, device, data_source=0)
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

    Can uses four environments and ten critic steps per vector collection step,
    hence 2.5 critic updates per collected macro transition.  The release's
    delayed actor frequency of two gives 1.25 actor/target updates per macro
    transition.  Actor updates are spread evenly instead of being burst in
    alternating groups of ten; the long-run update counts are identical.
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

    @property
    def flat_action_dim(self) -> int:
        return int(self.action_dim * self.horizon_steps)


class DICERLLearner:
    """DICE-RL actor/critic around a frozen deterministic ACT sampler."""

    def __init__(
        self,
        config: DICERLLearnerConfig,
        frozen_sampler: FrozenACTNoiseSampler,
        device: torch.device,
    ):
        self.config = config
        self.device = torch.device(device)
        self.frozen_sampler = frozen_sampler
        if int(frozen_sampler.horizon_steps) != int(config.horizon_steps):
            raise ValueError("frozen ACT / learner horizon mismatch")
        if int(frozen_sampler.ac_dim) != int(config.action_dim):
            raise ValueError("frozen ACT / learner action dimension mismatch")
        self.actor = DICEResidualActor(
            state_dim=config.state_dim,
            action_dim=config.flat_action_dim,
            hidden_dims=config.actor_hidden_dims,
            use_layernorm=True,
            squash_output=config.squash_residual,
            residual_scale=config.residual_scale,
        ).to(self.device)
        self.critic = DICEEnsembleCritic(
            state_dim=config.state_dim,
            action_dim=config.flat_action_dim,
            hidden_dims=config.critic_hidden_dims,
            ensemble_size=config.critic_ensemble_size,
            use_layernorm=config.critic_use_layernorm,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.train(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=float(config.actor_lr),
            weight_decay=float(config.actor_weight_decay),
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(config.critic_lr),
            weight_decay=float(config.critic_weight_decay),
        )

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
        if tuple(action.shape) != expected or not bool(torch.isfinite(action).all()):
            raise RuntimeError("frozen ACT returned invalid action chunks")
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
        if bool(getattr(self.frozen_sampler, "noise_conditioned", True)):
            base = self._sample_prior(repeated_state, flat_noise_chunk)
        else:
            # Standard ACT inference is deterministic and ignores DICE z. Avoid
            # running the transformer K identical times per state.
            base_once = self._sample_prior(state, noise[:, 0])
            base = (
                base_once[:, None, :]
                .expand(-1, count, -1)
                .reshape(batch * count, self.config.flat_action_dim)
            )
        residual = self.actor(repeated_state, self._flat_noise(flat_noise_chunk))
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
            selected_q = self.critic(state, actions[:, 0]).min(dim=0).values
        else:
            flat_state = (
                state[:, None, :]
                .expand(-1, count, -1)
                .reshape(state.shape[0] * count, self.config.state_dim)
            )
            all_q = self.critic(flat_state, actions.reshape(-1, self.config.flat_action_dim))
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
            all_q = self.critic_target(
                state, actions.reshape(batch * count, self.config.flat_action_dim)
            )
            conservative = all_q.min(dim=0).values.reshape(batch, count)
            return conservative.mean(dim=1)

    def update_critic(self, batch: DICEReplayBatch) -> Dict[str, float]:
        target_value = self._target_value(batch.next_states)
        target_q = batch.rewards + batch.discounts * target_value
        current_q = self.critic(batch.states, batch.actions)
        # Official DICE sums the independent member losses.
        critic_loss = (current_q - target_q.unsqueeze(0)).square().mean(dim=1).sum()
        if not bool(torch.isfinite(critic_loss)):
            raise RuntimeError("critic loss is NaN/Inf")
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), float(self.config.critic_grad_clip)
        )
        self.critic_optimizer.step()
        return {
            "train/critic_loss": float(critic_loss.detach()),
            "train/critic_grad_norm": float(grad_norm.detach()),
            "train/q_mean": float(current_q.detach().mean()),
            "train/q_min": float(current_q.detach().min()),
            "train/q_max": float(current_q.detach().max()),
            "train/target_q_mean": float(target_q.detach().mean()),
        }

    def update_actor(self, batch: DICEReplayBatch, primitive_transitions: int) -> Dict[str, float]:
        actions, bases, residuals = self._candidates(
            batch.states, self.config.actor_num_noise_samples
        )
        batch_size, count, _ = actions.shape
        state = (
            batch.states[:, None, :]
            .expand(-1, count, -1)
            .reshape(batch_size * count, self.config.state_dim)
        )
        current_q = (
            self.critic(state, actions.reshape(batch_size * count, self.config.flat_action_dim))
            .min(dim=0)
            .values.reshape(batch_size, count)
        )

        q_per_state = current_q.mean(dim=1)
        q_mask = torch.ones_like(q_per_state)
        if self.config.disable_actor_q_loss_for_offline:
            q_mask = (batch.data_source == 0).float()
        q_loss = -(q_per_state * q_mask).mean()
        online_count = q_mask.sum()
        if float(online_count.detach().item()) > 0.0:
            q_abs = (current_q.abs().mean(dim=1) * q_mask).sum() / online_count
        else:
            # This is unreachable under the formal 0.7 -> 0.1 RLPD schedule,
            # but matches the DICE release fallback for an all-offline batch.
            q_abs = current_q.abs().mean()
        if self.config.q_normalization and float(q_abs.detach().item()) > 1e-8:
            q_loss = q_loss / q_abs.detach()

        residual_mse = residuals.square().mean(dim=-1)
        bc_keep = torch.ones_like(residual_mse)
        filter_active = bool(self.config.q_filter_enabled) and int(primitive_transitions) > int(
            self.config.q_filter_warmup_primitive_steps
        )
        if filter_active:
            with torch.no_grad():
                base_q = (
                    self.critic(
                        state, bases.reshape(batch_size * count, self.config.flat_action_dim)
                    )
                    .min(dim=0)
                    .values.reshape(batch_size, count)
                )
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
        if not bool(torch.isfinite(actor_loss)):
            raise RuntimeError("actor loss is NaN/Inf")
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), float(self.config.actor_grad_clip)
        )
        self.actor_optimizer.step()
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
            for source, target in zip(self.critic.parameters(), self.critic_target.parameters()):
                target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)

    def update(
        self,
        batch: DICEReplayBatch,
        update_actor: bool,
        update_target: bool,
        primitive_transitions: int,
    ) -> Dict[str, float]:
        metrics = self.update_critic(batch)
        if update_actor:
            metrics.update(self.update_actor(batch, primitive_transitions))
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
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if state.get("config") != asdict(self.config):
            raise ValueError("learner config mismatch")
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        for optimizer in (self.actor_optimizer, self.critic_optimizer):
            for optimizer_state in optimizer.state.values():
                for key, value in optimizer_state.items():
                    if torch.is_tensor(value):
                        optimizer_state[key] = value.to(self.device)
