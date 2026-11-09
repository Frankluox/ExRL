import math
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def smdp_backup(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    durations: torch.Tensor,
    v_next: torch.Tensor,
) -> torch.Tensor:
    """
    SMDP backup with duration-aware discount:
    target = r + (1-done) * (gamma ** k) * V(s')
    """
    discount = torch.pow(
        torch.as_tensor(gamma, dtype=rewards.dtype, device=rewards.device), durations
    )
    return rewards + (1.0 - dones) * discount * v_next


def _activation_module(name: str):
    n = str(name).lower()
    if n == "relu":
        return nn.ReLU(inplace=True)
    if n == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
        use_layernorm: bool = False,
    ):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if bool(use_layernorm):
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(_activation_module(activation))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiscreteActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.backbone = MLP(
            self.obs_dim,
            self.action_dim,
            hidden_dims,
            activation=activation,
            use_layernorm=use_layernorm,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs)

    def distribution(self, obs: torch.Tensor, min_action_id: int = 0):
        logits = self.forward(obs)
        min_action_id = int(min_action_id)
        if min_action_id < 0 or min_action_id >= self.action_dim:
            if min_action_id != 0:
                raise ValueError(f"min_action_id out of range: {min_action_id}")
        if min_action_id > 0:
            logits = logits.clone()
            logits[:, :min_action_id] = -1e9
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        return logits, probs, log_probs

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, deterministic: bool = False, min_action_id: int = 0
    ) -> torch.Tensor:
        logits, _, _ = self.distribution(obs, min_action_id=min_action_id)
        if deterministic:
            return torch.argmax(logits, dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()


class TwinQCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.q1 = MLP(
            self.obs_dim,
            self.action_dim,
            hidden_dims,
            activation=activation,
            use_layernorm=use_layernorm,
        )
        self.q2 = MLP(
            self.obs_dim,
            self.action_dim,
            hidden_dims,
            activation=activation,
            use_layernorm=use_layernorm,
        )

    def forward(self, obs: torch.Tensor):
        return self.q1(obs), self.q2(obs)


class EnsembleQCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
        use_layernorm: bool = False,
        ensemble_size: int = 2,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.ensemble_size = int(ensemble_size)
        if self.ensemble_size <= 0:
            raise ValueError(f"ensemble_size must be positive, got {ensemble_size}")
        self.members = nn.ModuleList(
            [
                MLP(
                    self.obs_dim,
                    self.action_dim,
                    hidden_dims,
                    activation=activation,
                    use_layernorm=use_layernorm,
                )
                for _ in range(self.ensemble_size)
            ]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.stack([member(obs) for member in self.members], dim=0)


class _DistributionalDiscreteQMember(nn.Module):
    """
    PA-RL-style distributional critic member.

    The small final logits initialization matters: otherwise random initial
    logits can make the implied return distribution too sharp before the
    critic has learned anything useful.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
        use_layernorm: bool = False,
        init_final_scale: float = 1e-2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        last_dim = int(in_dim)
        for hidden_dim in hidden_dims:
            layer = nn.Linear(last_dim, int(hidden_dim))
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            self.norms.append(
                nn.LayerNorm(int(hidden_dim)) if bool(use_layernorm) else nn.Identity()
            )
            last_dim = int(hidden_dim)
        self.activation = _activation_module(activation)
        self.logits = nn.Linear(last_dim, int(out_dim))
        nn.init.orthogonal_(self.logits.weight, gain=float(init_final_scale))
        nn.init.zeros_(self.logits.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, norm in zip(self.layers, self.norms):
            x = self.activation(norm(layer(x)))
        return self.logits(x)


class DistributionalDiscreteQCritic(nn.Module):
    """
    Discrete-k distributional critic.

    Each ensemble member emits one return distribution per action id:
        logits: [E, B, action_dim, num_bins]
        q:      [E, B, action_dim]
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
        use_layernorm: bool = False,
        ensemble_size: int = 2,
        q_min: float = -300.0,
        q_max: float = 0.0,
        num_bins: int = 128,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.ensemble_size = int(ensemble_size)
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.num_bins = int(num_bins)
        if self.ensemble_size <= 0:
            raise ValueError(f"ensemble_size must be positive, got {ensemble_size}")
        if self.num_bins <= 1:
            raise ValueError(f"num_bins must be > 1, got {num_bins}")
        if not (self.q_min < self.q_max):
            raise ValueError(f"q_min must be < q_max, got {q_min} >= {q_max}")
        self.members = nn.ModuleList(
            [
                _DistributionalDiscreteQMember(
                    self.obs_dim,
                    self.action_dim * self.num_bins,
                    hidden_dims,
                    activation=activation,
                    use_layernorm=use_layernorm,
                    init_final_scale=1e-2,
                )
                for _ in range(self.ensemble_size)
            ]
        )
        edges = torch.linspace(self.q_min, self.q_max, self.num_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.register_buffer("support_edges", edges)
        self.register_buffer("support_centers", centers)
        self.hl_gauss_sigma = 0.75 * (self.q_max - self.q_min) / float(self.num_bins)

    def forward(self, obs: torch.Tensor):
        logits = torch.stack(
            [
                member(obs).view(obs.shape[0], self.action_dim, self.num_bins)
                for member in self.members
            ],
            dim=0,
        )
        probs = torch.softmax(logits, dim=-1)
        q_values = torch.sum(
            probs * self.support_centers.to(device=obs.device, dtype=obs.dtype).view(1, 1, 1, -1),
            dim=-1,
        )
        return q_values, logits

    @staticmethod
    def _log1mexp(x: torch.Tensor) -> torch.Tensor:
        x = torch.abs(x)
        log_two = math.log(2.0)
        return torch.where(
            x < log_two,
            torch.log(-torch.expm1(-x)),
            torch.log1p(-torch.exp(-x)),
        )

    @classmethod
    def _log_sub_exp(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        larger = torch.maximum(x, y)
        smaller = torch.minimum(x, y)
        return larger + cls._log1mexp(torch.clamp(larger - smaller, min=0.0))

    @classmethod
    def _normal_cdf_log_difference(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Stable log(ndtr(x) - ndtr(y)), matching PA-RL's HL-Gauss transform.
        y_positive = y >= 0
        x_hat = torch.where(y_positive, -y, x)
        y_hat = torch.where(y_positive, -x, y)
        return cls._log_sub_exp(torch.special.log_ndtr(x_hat), torch.special.log_ndtr(y_hat))

    def scalar_target_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        target = target.view(-1, 1)
        edges = self.support_edges.to(device=target.device, dtype=target.dtype)
        sigma = torch.as_tensor(self.hl_gauss_sigma, device=target.device, dtype=target.dtype)
        denom = torch.clamp(math.sqrt(2.0) * sigma, min=1e-6)
        upper = (edges[1:].view(1, -1) - target) / denom
        lower = (edges[:-1].view(1, -1) - target) / denom
        bin_log_probs = self._normal_cdf_log_difference(upper, lower)
        log_z = self._normal_cdf_log_difference(
            (edges[-1] - target) / denom, (edges[0] - target) / denom
        )
        probs = torch.exp(bin_log_probs - log_z)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        normalizer = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probs / normalizer

    def cross_entropy_loss_on_scalar(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        target_probs = self.scalar_target_to_probs(target).detach()
        log_probs = torch.log_softmax(logits, dim=-1)
        return -torch.sum(target_probs.unsqueeze(0) * log_probs, dim=-1)


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    durations: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    mc_returns: Optional[torch.Tensor] = None
    mc_return_valid: Optional[torch.Tensor] = None
    next_obs_samples: Optional[torch.Tensor] = None


@dataclass
class PendingSMDPTransition:
    obs: np.ndarray
    action_id: int
    reward: float
    duration_k: float
    next_obs: np.ndarray
    done: bool
    episode_done: bool
    next_obs_samples: Optional[np.ndarray] = None


@dataclass
class AggregatedSMDPTransition:
    obs: np.ndarray
    action_id: int
    reward: float
    duration_k: float
    next_obs: np.ndarray
    done: bool
    episode_done: bool
    next_obs_samples: Optional[np.ndarray] = None


@dataclass
class IngestedEnvStepResult:
    duration_k: float
    truncated: bool
    done_for_backup: bool
    replay_next_obs: np.ndarray
    emitted_transitions: int


def build_smdp_nstep_transition(
    transitions: Sequence[PendingSMDPTransition],
    gamma: float,
    n_step: int,
) -> AggregatedSMDPTransition:
    """
    Build one duration-aware SMDP n-step transition from the oldest element in
    `transitions`.

    If a true terminal or timeout boundary appears before n steps, accumulate
    only up to that boundary. For timeouts, `done` must remain False so SAC
    still bootstraps from the stored terminal-step observation.
    """
    if len(transitions) <= 0:
        raise ValueError("transitions must be non-empty")
    n_step = int(n_step)
    if n_step <= 0:
        raise ValueError(f"n_step must be >= 1, got {n_step}")

    first = transitions[0]
    discount = 1.0
    reward_acc = 0.0
    duration_acc = 0.0
    next_obs = np.asarray(first.next_obs, dtype=np.float32).reshape(-1)
    next_obs_samples = first.next_obs_samples
    done = bool(first.done)
    episode_done = bool(first.episode_done)

    for tr in list(transitions)[:n_step]:
        reward_acc += discount * float(tr.reward)
        duration_acc += float(tr.duration_k)
        next_obs = np.asarray(tr.next_obs, dtype=np.float32).reshape(-1)
        next_obs_samples = tr.next_obs_samples
        done = bool(tr.done)
        episode_done = bool(tr.episode_done)
        if bool(tr.episode_done):
            break
        discount *= float(gamma) ** float(tr.duration_k)

    return AggregatedSMDPTransition(
        obs=np.asarray(first.obs, dtype=np.float32).reshape(-1),
        action_id=int(first.action_id),
        reward=float(reward_acc),
        duration_k=float(duration_acc),
        next_obs=next_obs,
        done=done,
        episode_done=episode_done,
        next_obs_samples=None
        if next_obs_samples is None
        else np.asarray(next_obs_samples, dtype=np.float32),
    )


def drain_smdp_nstep_queue_to_replay(
    pending_queue: Deque[PendingSMDPTransition],
    replay: "SMDPReplayBuffer",
    gamma: float,
    n_step: int,
    action_dim: Optional[int] = None,
    env_id: Optional[int] = None,
    flush_all: bool = False,
) -> int:
    """
    Emit ready duration-aware n-step transitions from one per-env queue into replay.

    - Normal online case: emit once queue length reaches n.
    - Episode boundary case (`flush_all=True`): flush every remaining prefix so
      tail transitions are preserved with truncated n-step targets.
    """
    emitted = 0
    n_step = int(n_step)
    if n_step <= 0:
        raise ValueError(f"n_step must be >= 1, got {n_step}")

    while len(pending_queue) > 0 and (flush_all or len(pending_queue) >= n_step):
        agg = build_smdp_nstep_transition(pending_queue, gamma=gamma, n_step=n_step)
        replay.add(
            obs=agg.obs,
            action_id=int(agg.action_id),
            reward=float(agg.reward),
            duration_k=float(agg.duration_k),
            next_obs=agg.next_obs,
            done=bool(agg.done),
            action_dim=action_dim,
            next_obs_samples=agg.next_obs_samples,
            episode_done=bool(agg.episode_done),
            env_id=env_id,
        )
        pending_queue.popleft()
        emitted += 1

    return emitted


def _coerce_next_obs_samples(
    samples,
    fallback_next_obs: np.ndarray,
    num_samples: int,
) -> Optional[np.ndarray]:
    num_samples = int(num_samples)
    if num_samples <= 1:
        return None
    fallback = np.asarray(fallback_next_obs, dtype=np.float32).reshape(-1)
    if samples is None:
        return np.repeat(fallback.reshape(1, -1), num_samples, axis=0).astype(np.float32)
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[-1] != fallback.shape[0]:
        return np.repeat(fallback.reshape(1, -1), num_samples, axis=0).astype(np.float32)
    if arr.shape[0] >= num_samples:
        return arr[:num_samples].astype(np.float32, copy=False)
    pad = np.repeat(arr[-1:].astype(np.float32, copy=False), num_samples - arr.shape[0], axis=0)
    return np.concatenate([arr.astype(np.float32, copy=False), pad], axis=0)


def extract_bootstrap_next_obs_from_venv(
    venv,
    env_idx: int,
    info: Dict,
    fallback_next_obs: np.ndarray,
) -> np.ndarray:
    """
    For timeout / truncation transitions, replay should bootstrap from the
    actual truncation-boundary RL observation, not the reset observation
    returned by the collector VecEnv.
    """
    term_obs = info.get("terminal_observation", None)
    if term_obs is None:
        return np.asarray(fallback_next_obs, dtype=np.float32).reshape(-1)
    if isinstance(term_obs, dict):
        base = getattr(venv, "venv", venv)
        get_terminal_rl_obs = getattr(base, "get_terminal_rl_obs", None)
        if callable(get_terminal_rl_obs):
            try:
                rl_obs = get_terminal_rl_obs(term_obs, int(env_idx))
                rl_obs = np.asarray(rl_obs, dtype=np.float32).reshape(-1)
                if (
                    rl_obs.shape
                    == np.asarray(fallback_next_obs, dtype=np.float32).reshape(-1).shape
                ):
                    return rl_obs
            except Exception:
                pass
    return np.asarray(fallback_next_obs, dtype=np.float32).reshape(-1)


def extract_bootstrap_next_obs_samples_from_venv(
    venv,
    env_idx: int,
    info: Dict,
    fallback_next_obs: np.ndarray,
    num_samples: int,
) -> Optional[np.ndarray]:
    num_samples = int(num_samples)
    if num_samples <= 1:
        return None
    term_obs = info.get("terminal_observation", None)
    if term_obs is not None and isinstance(term_obs, dict):
        base = getattr(venv, "venv", venv)
        get_terminal_rl_obs_samples = getattr(base, "get_terminal_rl_obs_samples", None)
        if callable(get_terminal_rl_obs_samples):
            try:
                samples = get_terminal_rl_obs_samples(term_obs, int(env_idx), num_samples)
                coerced = _coerce_next_obs_samples(samples, fallback_next_obs, num_samples)
                if coerced is not None:
                    return coerced
            except Exception:
                pass
    return _coerce_next_obs_samples(None, fallback_next_obs, num_samples)


def ingest_smdp_env_step_to_replay(
    *,
    venv,
    env_idx: int,
    obs_i: np.ndarray,
    next_obs_i: np.ndarray,
    reward_i: float,
    done_i: bool,
    info: Dict,
    action_id: int,
    pending_queue: Deque[PendingSMDPTransition],
    replay: "SMDPReplayBuffer",
    gamma: float,
    n_step: int,
    reward_duration_mode: str,
    max_chunk_len: int,
    action_dim: Optional[int] = None,
    target_next_obs_samples: int = 1,
) -> IngestedEnvStepResult:
    k = float(info.get("exec_len", 1.0))
    if (k < 0.0) or (k > float(max_chunk_len)):
        raise RuntimeError(f"duration_k out of range: {k} not in [0, {max_chunk_len}]")
    truncated = bool(info.get("TimeLimit.truncated", False))
    done_for_backup = bool(done_i) and (not truncated)

    replay_next_obs = np.asarray(next_obs_i, dtype=np.float32).reshape(-1)
    target_next_obs_samples = max(1, int(target_next_obs_samples))
    replay_next_obs_samples = _coerce_next_obs_samples(
        info.get("target_next_observations", None),
        replay_next_obs,
        target_next_obs_samples,
    )
    if bool(done_i) and (not done_for_backup):
        replay_next_obs = extract_bootstrap_next_obs_from_venv(
            venv=venv,
            env_idx=int(env_idx),
            info=info,
            fallback_next_obs=replay_next_obs,
        )
        replay_next_obs_samples = extract_bootstrap_next_obs_samples_from_venv(
            venv=venv,
            env_idx=int(env_idx),
            info=info,
            fallback_next_obs=replay_next_obs,
            num_samples=target_next_obs_samples,
        )

    reward_for_backup = (
        float(reward_i) if reward_duration_mode == "sum" else float(reward_i) / max(float(k), 1.0)
    )

    pending_queue.append(
        PendingSMDPTransition(
            obs=np.asarray(obs_i, dtype=np.float32).reshape(-1),
            action_id=int(action_id),
            reward=float(reward_for_backup),
            duration_k=float(k),
            next_obs=replay_next_obs,
            done=bool(done_for_backup),
            episode_done=bool(done_i),
            next_obs_samples=replay_next_obs_samples,
        )
    )
    emitted_transitions = drain_smdp_nstep_queue_to_replay(
        pending_queue,
        replay=replay,
        gamma=float(gamma),
        n_step=int(n_step),
        action_dim=action_dim,
        env_id=int(env_idx),
        flush_all=bool(done_i),
    )
    return IngestedEnvStepResult(
        duration_k=float(k),
        truncated=bool(truncated),
        done_for_backup=bool(done_for_backup),
        replay_next_obs=np.asarray(replay_next_obs, dtype=np.float32).reshape(-1),
        emitted_transitions=int(emitted_transitions),
    )


class SMDPReplayBuffer:
    """
    Replay fields are fixed to:
    (obs, action_id, reward_chunk, duration_k, next_obs, done)
    """

    def __init__(
        self,
        obs_dim: int,
        capacity: int,
        device: torch.device,
        target_next_obs_samples: int = 1,
    ):
        self.obs_dim = int(obs_dim)
        self.capacity = int(capacity)
        self.device = device
        self.target_next_obs_samples = max(1, int(target_next_obs_samples))
        self.pos = 0
        self.size = 0

        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.durations = np.zeros((self.capacity,), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs_samples = (
            np.zeros((self.capacity, self.target_next_obs_samples, self.obs_dim), dtype=np.float32)
            if self.target_next_obs_samples > 1
            else None
        )
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.episode_dones = np.zeros((self.capacity,), dtype=np.float32)
        self.env_ids = np.full((self.capacity,), -1, dtype=np.int64)
        self.mc_returns = np.zeros((self.capacity,), dtype=np.float32)
        self.mc_return_valid = np.zeros((self.capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return int(self.size)

    def state_dict(self) -> Dict:
        """
        Store only initialized replay slots when the buffer is not full.
        Once full, the physical ring layout and pos are preserved so resumed
        sampling sees the same index-to-transition mapping.
        """
        stored_size = int(self.capacity if self.size >= self.capacity else self.size)
        return {
            "obs_dim": int(self.obs_dim),
            "capacity": int(self.capacity),
            "pos": int(self.pos),
            "size": int(self.size),
            "stored_size": int(stored_size),
            "target_next_obs_samples": int(self.target_next_obs_samples),
            "obs": self.obs[:stored_size].copy(),
            "actions": self.actions[:stored_size].copy(),
            "rewards": self.rewards[:stored_size].copy(),
            "durations": self.durations[:stored_size].copy(),
            "next_obs": self.next_obs[:stored_size].copy(),
            "next_obs_samples": (
                None
                if self.next_obs_samples is None
                else self.next_obs_samples[:stored_size].copy()
            ),
            "dones": self.dones[:stored_size].copy(),
            "episode_dones": self.episode_dones[:stored_size].copy(),
            "env_ids": self.env_ids[:stored_size].copy(),
            "mc_returns": self.mc_returns[:stored_size].copy(),
            "mc_return_valid": self.mc_return_valid[:stored_size].copy(),
        }

    def load_state_dict(self, state: Dict) -> None:
        ckpt_obs_dim = int(state.get("obs_dim", self.obs_dim))
        ckpt_capacity = int(state.get("capacity", self.capacity))
        if ckpt_obs_dim != int(self.obs_dim):
            raise ValueError(f"Replay obs_dim mismatch: ckpt={ckpt_obs_dim} current={self.obs_dim}")
        if ckpt_capacity != int(self.capacity):
            raise ValueError(
                f"Replay capacity mismatch: ckpt={ckpt_capacity} current={self.capacity}"
            )

        size = int(state.get("size", 0))
        pos = int(state.get("pos", 0))
        stored_size = int(
            state.get("stored_size", self.capacity if size >= self.capacity else size)
        )
        if not (0 <= size <= self.capacity):
            raise ValueError(f"Replay size out of range: {size} capacity={self.capacity}")
        if not (0 <= stored_size <= self.capacity):
            raise ValueError(
                f"Replay stored_size out of range: {stored_size} capacity={self.capacity}"
            )

        self.obs.fill(0.0)
        self.actions.fill(0)
        self.rewards.fill(0.0)
        self.durations.fill(0.0)
        self.next_obs.fill(0.0)
        if self.next_obs_samples is not None:
            self.next_obs_samples.fill(0.0)
        self.dones.fill(0.0)
        self.episode_dones.fill(0.0)
        self.env_ids.fill(-1)
        self.mc_returns.fill(0.0)
        self.mc_return_valid.fill(0.0)

        if stored_size > 0:
            self.obs[:stored_size] = np.asarray(state["obs"], dtype=np.float32)
            self.actions[:stored_size] = np.asarray(state["actions"], dtype=np.int64)
            self.rewards[:stored_size] = np.asarray(state["rewards"], dtype=np.float32)
            self.durations[:stored_size] = np.asarray(state["durations"], dtype=np.float32)
            self.next_obs[:stored_size] = np.asarray(state["next_obs"], dtype=np.float32)
            if self.next_obs_samples is not None:
                state_samples = state.get("next_obs_samples", None)
                if state_samples is None:
                    self.next_obs_samples[:stored_size] = np.repeat(
                        self.next_obs[:stored_size, None, :],
                        self.target_next_obs_samples,
                        axis=1,
                    )
                else:
                    arr = np.asarray(state_samples, dtype=np.float32)
                    if arr.ndim != 3 or arr.shape[0] != stored_size or arr.shape[2] != self.obs_dim:
                        raise ValueError(
                            f"Replay next_obs_samples shape mismatch: got {arr.shape}, "
                            f"expected [{stored_size}, M, {self.obs_dim}]"
                        )
                    m = min(int(arr.shape[1]), int(self.target_next_obs_samples))
                    self.next_obs_samples[:stored_size, :m, :] = arr[:, :m, :]
                    if m < int(self.target_next_obs_samples):
                        self.next_obs_samples[:stored_size, m:, :] = arr[:, m - 1 : m, :]
            self.dones[:stored_size] = np.asarray(state["dones"], dtype=np.float32)
            self.episode_dones[:stored_size] = np.asarray(state["episode_dones"], dtype=np.float32)
            self.env_ids[:stored_size] = np.asarray(state["env_ids"], dtype=np.int64)
            self.mc_returns[:stored_size] = np.asarray(state["mc_returns"], dtype=np.float32)
            self.mc_return_valid[:stored_size] = np.asarray(
                state["mc_return_valid"], dtype=np.float32
            )

        self.size = int(size)
        self.pos = int(pos % max(int(self.capacity), 1))

    def add(
        self,
        obs: np.ndarray,
        action_id: int,
        reward: float,
        duration_k: float,
        next_obs: np.ndarray,
        done: bool,
        action_dim: Optional[int] = None,
        episode_done: Optional[bool] = None,
        env_id: Optional[int] = None,
        mc_return: Optional[float] = None,
        mc_return_valid: bool = False,
        next_obs_samples: Optional[np.ndarray] = None,
    ) -> int:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.obs_dim or next_obs.shape[0] != self.obs_dim:
            raise ValueError(
                f"Replay obs dim mismatch: got {obs.shape[0]} / {next_obs.shape[0]}, expect {self.obs_dim}"
            )

        action_id = int(action_id)
        if action_dim is not None:
            if action_id < 0 or action_id >= int(action_dim):
                raise ValueError(
                    f"action_id out of range: {action_id} not in [0, {int(action_dim) - 1}]"
                )

        duration_k = float(duration_k)
        if duration_k < 0:
            raise ValueError(f"duration_k must be >= 0, got {duration_k}")

        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action_id
        self.rewards[i] = float(reward)
        self.durations[i] = duration_k
        self.next_obs[i] = next_obs
        if self.next_obs_samples is not None:
            coerced = _coerce_next_obs_samples(
                next_obs_samples,
                next_obs,
                self.target_next_obs_samples,
            )
            if coerced is None:
                coerced = np.repeat(next_obs.reshape(1, -1), self.target_next_obs_samples, axis=0)
            self.next_obs_samples[i] = coerced
        self.dones[i] = 1.0 if bool(done) else 0.0
        self.episode_dones[i] = 1.0 if bool(done if episode_done is None else episode_done) else 0.0
        self.env_ids[i] = -1 if env_id is None else int(env_id)
        self.mc_returns[i] = 0.0 if mc_return is None else float(mc_return)
        self.mc_return_valid[i] = 1.0 if bool(mc_return_valid) else 0.0

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return int(i)

    def compute_mc_returns_in_place(self, gamma: float) -> int:
        """
        Compute SMDP Monte-Carlo return-to-go for currently stored transitions.

        This is intended for offline phase-0 buffers. It respects env_id when
        available, so vector-env interleaving does not corrupt returns.
        """
        if self.size <= 0:
            return 0
        gamma = float(gamma)
        if self.size < self.capacity:
            chronological = np.arange(self.size, dtype=np.int64)
        else:
            chronological = np.concatenate(
                [
                    np.arange(self.pos, self.capacity, dtype=np.int64),
                    np.arange(0, self.pos, dtype=np.int64),
                ]
            )
        returns_by_env: Dict[int, float] = {}
        valid_by_env: Dict[int, bool] = {}
        count = 0
        for idx in chronological[::-1]:
            env_id = int(self.env_ids[idx])
            if env_id < 0:
                env_id = 0
            if bool(self.episode_dones[idx]):
                returns_by_env[env_id] = 0.0
                valid_by_env[env_id] = True
            if not bool(valid_by_env.get(env_id, False)):
                self.mc_returns[idx] = 0.0
                self.mc_return_valid[idx] = 0.0
                continue
            ret = float(self.rewards[idx]) + (gamma ** float(self.durations[idx])) * float(
                returns_by_env.get(env_id, 0.0)
            )
            returns_by_env[env_id] = ret
            self.mc_returns[idx] = ret
            self.mc_return_valid[idx] = 1.0
            count += 1
        return int(count)

    def sample(self, batch_size: int) -> ReplayBatch:
        if self.size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        batch_size = int(batch_size)
        idx = np.random.randint(0, self.size, size=batch_size)
        return ReplayBatch(
            obs=torch.as_tensor(self.obs[idx], device=self.device, dtype=torch.float32),
            actions=torch.as_tensor(self.actions[idx], device=self.device, dtype=torch.long),
            rewards=torch.as_tensor(self.rewards[idx], device=self.device, dtype=torch.float32),
            durations=torch.as_tensor(self.durations[idx], device=self.device, dtype=torch.float32),
            next_obs=torch.as_tensor(self.next_obs[idx], device=self.device, dtype=torch.float32),
            dones=torch.as_tensor(self.dones[idx], device=self.device, dtype=torch.float32),
            mc_returns=torch.as_tensor(
                self.mc_returns[idx], device=self.device, dtype=torch.float32
            ),
            mc_return_valid=torch.as_tensor(
                self.mc_return_valid[idx], device=self.device, dtype=torch.float32
            ),
            next_obs_samples=(
                None
                if self.next_obs_samples is None
                else torch.as_tensor(
                    self.next_obs_samples[idx], device=self.device, dtype=torch.float32
                )
            ),
        )


class SMDPDiscreteSAC:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (512, 512, 512),
        gamma: float = 0.999,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        init_alpha: float = 0.1,
        fixed_alpha: Optional[float] = None,
        max_alpha: Optional[float] = None,
        target_entropy: Optional[float] = None,
        activation: str = "relu",
        actor_use_layernorm: bool = False,
        critic_use_layernorm: bool = False,
        actor_enabled: bool = True,
        target_policy_mode: str = "actor",
        critic_backup_temp: float = 1.0,
        cql_alpha: float = 0.0,
        cql_temp: float = 1.0,
        use_mc_lower_bound: bool = False,
        mc_lower_bound_scope: str = "topk_non_data",
        mc_lower_bound_topk: int = 10,
        critic_ensemble_size: int = 2,
        target_critic_subsample_size: int = 0,
        critic_bootstrap_mask_prob: float = 1.0,
        distributional_critic: bool = False,
        distributional_q_min: float = -300.0,
        distributional_q_max: float = 0.0,
        distributional_num_bins: int = 128,
        td_loss_type: str = "mse",
        td_huber_delta: float = 10.0,
        target_q_clip_min: Optional[float] = None,
        target_q_clip_max: Optional[float] = None,
        critic_q_upper_bound: Optional[float] = None,
        critic_q_upper_penalty: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.activation = str(activation)
        self.actor_use_layernorm = bool(actor_use_layernorm)
        self.critic_use_layernorm = bool(critic_use_layernorm)
        self.actor_enabled = bool(actor_enabled)
        self.target_policy_mode = str(target_policy_mode).lower()
        self.critic_backup_temp = float(critic_backup_temp)
        self.cql_alpha = float(cql_alpha)
        self.cql_temp = float(cql_temp)
        self.use_mc_lower_bound = bool(use_mc_lower_bound)
        self.mc_lower_bound_scope = str(mc_lower_bound_scope).lower()
        self.mc_lower_bound_topk = int(mc_lower_bound_topk)
        self.critic_ensemble_size = int(critic_ensemble_size)
        self.target_critic_subsample_size = int(target_critic_subsample_size)
        self.critic_bootstrap_mask_prob = float(critic_bootstrap_mask_prob)
        self.distributional_critic = bool(distributional_critic)
        self.distributional_q_min = float(distributional_q_min)
        self.distributional_q_max = float(distributional_q_max)
        self.distributional_num_bins = int(distributional_num_bins)
        self.td_loss_type = str(td_loss_type).lower()
        self.td_huber_delta = float(td_huber_delta)
        self.target_q_clip_min = None if target_q_clip_min is None else float(target_q_clip_min)
        self.target_q_clip_max = None if target_q_clip_max is None else float(target_q_clip_max)
        self.critic_q_upper_bound = (
            None if critic_q_upper_bound is None else float(critic_q_upper_bound)
        )
        self.critic_q_upper_penalty = float(critic_q_upper_penalty)
        if self.target_policy_mode not in {"actor", "q_soft", "q_greedy"}:
            raise ValueError(f"Unsupported target_policy_mode: {target_policy_mode}")
        if (not self.actor_enabled) and self.target_policy_mode == "actor":
            raise ValueError(
                "actor_enabled=False requires target_policy_mode to be q_soft or q_greedy"
            )
        if self.critic_backup_temp <= 0.0:
            raise ValueError(f"critic_backup_temp must be > 0, got {critic_backup_temp}")
        if self.cql_alpha < 0.0:
            raise ValueError(f"cql_alpha must be >= 0, got {cql_alpha}")
        if self.cql_temp <= 0.0:
            raise ValueError(f"cql_temp must be > 0, got {cql_temp}")
        if self.mc_lower_bound_scope not in {"all", "non_data", "data", "topk_non_data"}:
            raise ValueError(
                f"mc_lower_bound_scope must be one of all/non_data/data/topk_non_data, got {mc_lower_bound_scope}"
            )
        if self.mc_lower_bound_topk <= 0:
            raise ValueError(f"mc_lower_bound_topk must be positive, got {mc_lower_bound_topk}")
        if self.critic_ensemble_size <= 0:
            raise ValueError(f"critic_ensemble_size must be positive, got {critic_ensemble_size}")
        if self.target_critic_subsample_size < 0:
            raise ValueError(
                f"target_critic_subsample_size must be >= 0, got {target_critic_subsample_size}"
            )
        if not (0.0 < self.critic_bootstrap_mask_prob <= 1.0):
            raise ValueError(
                f"critic_bootstrap_mask_prob must be in (0, 1], got {critic_bootstrap_mask_prob}"
            )
        if self.td_loss_type not in {"mse", "huber"}:
            raise ValueError(f"td_loss_type must be mse or huber, got {td_loss_type}")
        if self.td_huber_delta <= 0.0:
            raise ValueError(f"td_huber_delta must be > 0, got {td_huber_delta}")
        if (
            self.target_q_clip_min is not None
            and self.target_q_clip_max is not None
            and self.target_q_clip_min > self.target_q_clip_max
        ):
            raise ValueError(
                f"target_q_clip_min ({self.target_q_clip_min}) cannot exceed "
                f"target_q_clip_max ({self.target_q_clip_max})"
            )
        if self.critic_q_upper_penalty < 0.0:
            raise ValueError(f"critic_q_upper_penalty must be >= 0, got {critic_q_upper_penalty}")

        self.actor = (
            DiscreteActor(
                self.obs_dim,
                self.action_dim,
                hidden_dims,
                activation=self.activation,
                use_layernorm=self.actor_use_layernorm,
            ).to(self.device)
            if self.actor_enabled
            else None
        )
        self.critic = self._build_critic(hidden_dims).to(self.device)
        self.critic_target = self._build_critic(hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = (
            torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
            if self.actor is not None
            else None
        )
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(critic_lr))

        self.fixed_alpha = None if fixed_alpha is None else float(fixed_alpha)
        self.max_alpha = None if max_alpha is None else float(max_alpha)
        if self.max_alpha is not None and self.max_alpha <= 0.0:
            raise ValueError(f"max_alpha must be > 0 when set, got {max_alpha}")
        if self.fixed_alpha is not None:
            if self.fixed_alpha < 0.0:
                raise ValueError(f"fixed_alpha must be >= 0, got {fixed_alpha}")
            if self.max_alpha is not None and self.fixed_alpha > self.max_alpha:
                raise ValueError(
                    f"fixed_alpha ({self.fixed_alpha}) cannot exceed max_alpha ({self.max_alpha})"
                )
            alpha_for_state = max(self.fixed_alpha, 1e-12)
            self.log_alpha = torch.nn.Parameter(
                torch.tensor(
                    np.log(float(alpha_for_state)), dtype=torch.float32, device=self.device
                ),
                requires_grad=False,
            )
            self.alpha_opt = None
        elif not self.actor_enabled:
            self.fixed_alpha = 0.0
            self.log_alpha = torch.nn.Parameter(
                torch.tensor(np.log(1e-12), dtype=torch.float32, device=self.device),
                requires_grad=False,
            )
            self.alpha_opt = None
        else:
            if float(init_alpha) <= 0.0:
                raise ValueError(f"init_alpha must be > 0, got {init_alpha}")
            if self.max_alpha is not None and float(init_alpha) > self.max_alpha:
                raise ValueError(
                    f"init_alpha ({init_alpha}) cannot exceed max_alpha ({self.max_alpha})"
                )
            self.log_alpha = torch.nn.Parameter(
                torch.tensor(np.log(float(init_alpha)), dtype=torch.float32, device=self.device)
            )
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=float(alpha_lr))

        if target_entropy is None:
            target_entropy = 0.98 * np.log(float(self.action_dim))
        self.target_entropy = float(target_entropy)

    def _build_critic(self, hidden_dims: Sequence[int]) -> nn.Module:
        if self.distributional_critic:
            return DistributionalDiscreteQCritic(
                self.obs_dim,
                self.action_dim,
                hidden_dims,
                activation=self.activation,
                use_layernorm=self.critic_use_layernorm,
                ensemble_size=self.critic_ensemble_size,
                q_min=self.distributional_q_min,
                q_max=self.distributional_q_max,
                num_bins=self.distributional_num_bins,
            )
        if self.critic_ensemble_size == 2:
            return TwinQCritic(
                self.obs_dim,
                self.action_dim,
                hidden_dims,
                activation=self.activation,
                use_layernorm=self.critic_use_layernorm,
            )
        return EnsembleQCritic(
            self.obs_dim,
            self.action_dim,
            hidden_dims,
            activation=self.activation,
            use_layernorm=self.critic_use_layernorm,
            ensemble_size=self.critic_ensemble_size,
        )

    @property
    def alpha(self) -> torch.Tensor:
        if self.fixed_alpha is not None:
            return torch.as_tensor(self.fixed_alpha, dtype=torch.float32, device=self.device)
        alpha = self.log_alpha.exp()
        if self.max_alpha is not None:
            alpha = torch.clamp(alpha, max=float(self.max_alpha))
        return alpha

    def _clamp_log_alpha_(self) -> None:
        if self.max_alpha is None:
            return
        max_log_alpha = float(np.log(max(float(self.max_alpha), 1e-12)))
        with torch.no_grad():
            self.log_alpha.data.clamp_(max=max_log_alpha)

    @torch.no_grad()
    def select_action(
        self, obs_np: np.ndarray, deterministic: bool = False, min_action_id: int = 0
    ) -> np.ndarray:
        if self.actor is None:
            return self.select_critic_action(
                obs_np,
                deterministic=deterministic,
                min_action_id=min_action_id,
                mode=("q_soft" if not deterministic else "q_greedy"),
            )
        obs = torch.as_tensor(np.asarray(obs_np, dtype=np.float32), device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        acts = self.actor.act(obs, deterministic=deterministic, min_action_id=min_action_id)
        return acts.detach().cpu().numpy().astype(np.int64)

    def _critic_output(self, critic: nn.Module, obs: torch.Tensor):
        out = critic(obs)
        if isinstance(out, tuple) and len(out) == 2:
            first, second = out
            if torch.is_tensor(second) and second.ndim == 4:
                return first, second
            return torch.stack([first, second], dim=0), None
        if torch.is_tensor(out):
            if out.ndim != 3:
                raise RuntimeError(f"Unsupported critic tensor shape: {tuple(out.shape)}")
            return out, None
        raise RuntimeError(f"Unsupported critic output type: {type(out)}")

    def _subsample_critic_values(
        self, q_values: torch.Tensor, q_logits: Optional[torch.Tensor] = None
    ):
        n = int(q_values.shape[0])
        subsample_size = int(self.target_critic_subsample_size)
        if subsample_size <= 0 or subsample_size >= n:
            return q_values, q_logits
        idx = torch.randint(low=0, high=n, size=(subsample_size,), device=q_values.device)
        q_values = q_values.index_select(0, idx)
        if q_logits is not None:
            q_logits = q_logits.index_select(0, idx)
        return q_values, q_logits

    @staticmethod
    def _mask_q_values(q: torch.Tensor, min_action_id: int) -> torch.Tensor:
        min_action_id = int(min_action_id)
        if min_action_id <= 0:
            return q
        out = q.clone()
        out[..., :min_action_id] = -1e9
        return out

    def _critic_value_from_q(
        self, q: torch.Tensor, mode: str, min_action_id: int = 0
    ) -> torch.Tensor:
        mode = str(mode).lower()
        q = self._mask_q_values(q, min_action_id=min_action_id)
        if mode == "q_greedy":
            return q.max(dim=-1).values
        if mode == "q_soft":
            temp = float(self.critic_backup_temp)
            return temp * torch.logsumexp(q / temp, dim=-1)
        raise ValueError(f"Unsupported critic value mode: {mode}")

    @torch.no_grad()
    def select_critic_action(
        self,
        obs_np: np.ndarray,
        deterministic: bool = True,
        min_action_id: int = 0,
        mode: str = "q_greedy",
        reduce: str = "min",
        ucb_beta: float = 0.0,
        thompson_heads: Optional[np.ndarray] = None,
        return_info: bool = False,
    ) -> np.ndarray:
        obs = torch.as_tensor(np.asarray(obs_np, dtype=np.float32), device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        q_values, _ = self._critic_output(self.critic, obs)
        mode = str(mode).lower()
        reduce = str(reduce).lower()
        info: Dict[str, float] = {}

        if mode == "q_ucb":
            q_mean = q_values.mean(dim=0)
            q_std = q_values.std(dim=0, unbiased=False)
            q = self._mask_q_values(q_mean + float(ucb_beta) * q_std, min_action_id=min_action_id)
            acts = torch.argmax(q, dim=-1)
            selected = acts.view(-1, 1)
            member_argmax = torch.argmax(
                self._mask_q_values(q_values, min_action_id=min_action_id), dim=-1
            )
            disagree = member_argmax.ne(member_argmax[:1]).any(dim=0).float()
            info = {
                "q_ensemble_std_selected": float(q_std.gather(1, selected).mean().detach().item()),
                "q_ensemble_std_max": float(q_std.max(dim=-1).values.mean().detach().item()),
                "q_ucb_beta": float(ucb_beta),
                "q_ucb_bonus_selected": float(
                    (float(ucb_beta) * q_std.gather(1, selected)).mean().detach().item()
                ),
                "q_argmax_member_disagree_rate": float(disagree.mean().detach().item()),
            }
        elif mode == "q_thompson":
            if thompson_heads is None:
                head_idx = torch.randint(
                    low=0,
                    high=int(q_values.shape[0]),
                    size=(int(q_values.shape[1]),),
                    device=q_values.device,
                )
            else:
                heads_np = np.asarray(thompson_heads, dtype=np.int64).reshape(-1)
                if heads_np.shape[0] != int(q_values.shape[1]):
                    raise ValueError(
                        f"thompson_heads must have batch size {q_values.shape[1]}, got {heads_np.shape[0]}"
                    )
                head_idx = torch.as_tensor(heads_np, dtype=torch.long, device=q_values.device)
                head_idx = torch.remainder(head_idx, int(q_values.shape[0]))
            batch_idx = torch.arange(int(q_values.shape[1]), device=q_values.device)
            q = q_values[head_idx, batch_idx, :]
            q = self._mask_q_values(q, min_action_id=min_action_id)
            acts = torch.argmax(q, dim=-1)
            info = {
                "q_thompson_head_mean": float(head_idx.float().mean().detach().item()),
                "q_thompson_head_unique": float(torch.unique(head_idx).numel()),
            }
        else:
            if reduce == "min":
                q = q_values.min(dim=0).values
            elif reduce == "mean":
                q = q_values.mean(dim=0)
            else:
                raise ValueError(f"Unsupported critic action reduce: {reduce}")
            q = self._mask_q_values(q, min_action_id=min_action_id)
            if deterministic or mode == "q_greedy":
                acts = torch.argmax(q, dim=-1)
            elif mode == "q_soft":
                temp = float(self.critic_backup_temp)
                probs = torch.softmax(q / temp, dim=-1)
                acts = torch.distributions.Categorical(probs=probs).sample()
            else:
                raise ValueError(f"Unsupported critic action mode: {mode}")

        actions_np = acts.detach().cpu().numpy().astype(np.int64)
        if return_info:
            return actions_np, info
        return actions_np

    def _check_batch(self, batch: ReplayBatch) -> None:
        if torch.any(batch.actions < 0) or torch.any(batch.actions >= self.action_dim):
            raise RuntimeError("Batch action_id out of range")
        if torch.any(batch.durations < 0.0):
            raise RuntimeError("Batch duration_k must be >= 0")
        tensors = [batch.obs, batch.next_obs, batch.rewards, batch.durations, batch.dones]
        if batch.next_obs_samples is not None:
            tensors.append(batch.next_obs_samples)
        for t in tensors:
            if not torch.isfinite(t).all():
                raise RuntimeError("NaN/Inf detected in replay batch")

    def _soft_update_target(self) -> None:
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def _target_v_from_next_obs(
        self,
        next_obs: torch.Tensor,
        legal_min_action_id: int,
    ) -> torch.Tensor:
        next_q_values_t, _ = self._critic_output(self.critic_target, next_obs)
        if self.target_policy_mode == "actor":
            if self.actor is None:
                raise RuntimeError("target_policy_mode=actor requires actor_enabled=True")
            _, next_probs, next_log_probs = self.actor.distribution(
                next_obs, min_action_id=legal_min_action_id
            )
            next_min_q_t = next_q_values_t.min(dim=0).values
            alpha_detached = self.alpha.detach()
            return torch.sum(next_probs * (next_min_q_t - alpha_detached * next_log_probs), dim=1)
        next_q_values_t, _ = self._subsample_critic_values(next_q_values_t)
        next_min_q_t = next_q_values_t.min(dim=0).values
        return self._critic_value_from_q(
            next_min_q_t,
            mode=self.target_policy_mode,
            min_action_id=legal_min_action_id,
        )

    def _distributional_td_loss(
        self,
        current_logits: torch.Tensor,
        actions: torch.Tensor,
        target_q: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self.critic, "cross_entropy_loss_on_scalar"):
            raise RuntimeError("distributional_critic=True requires cross_entropy_loss_on_scalar")
        action_idx = actions.view(1, -1, 1, 1).expand(
            current_logits.shape[0], -1, 1, current_logits.shape[-1]
        )
        logits_a = current_logits.gather(2, action_idx).squeeze(2)
        per_member_loss = self.critic.cross_entropy_loss_on_scalar(logits_a, target_q)
        return (weights.view(1, -1) * per_member_loss).mean()

    def _cql_loss(
        self,
        q_values: torch.Tensor,
        actions: torch.Tensor,
        batch: ReplayBatch,
    ):
        action_idx = actions.view(1, -1, 1).expand(q_values.shape[0], -1, 1)
        q_data = q_values.gather(2, action_idx).squeeze(2)
        q_for_lse = q_values
        mc_bound_rate = torch.zeros((), device=q_values.device, dtype=torch.float32)
        mc_bound_actions_per_valid = torch.zeros((), device=q_values.device, dtype=torch.float32)
        if self.use_mc_lower_bound:
            if batch.mc_returns is None or batch.mc_return_valid is None:
                raise RuntimeError("use_mc_lower_bound=True requires replay mc_returns")
            valid = batch.mc_return_valid.view(1, -1, 1) > 0.5
            mc = batch.mc_returns.view(1, -1, 1)
            action_ids = torch.arange(self.action_dim, device=q_values.device).view(1, 1, -1)
            data_action = actions.view(1, -1, 1)
            is_data_action = action_ids == data_action
            if self.mc_lower_bound_scope == "all":
                scope_mask = torch.ones_like(is_data_action, dtype=torch.bool)
            elif self.mc_lower_bound_scope == "non_data":
                # Closest discrete analogue of PA-RL CalQL: bound OOD CQL
                # candidates, but leave the replay/data action itself unchanged.
                scope_mask = ~is_data_action
            elif self.mc_lower_bound_scope == "data":
                scope_mask = is_data_action
            elif self.mc_lower_bound_scope == "topk_non_data":
                # PA-RL bounds policy-like current/next candidate actions but
                # not uniformly random CQL actions. In discrete-k AAC, top-k
                # current-Q actions are the closest exact analogue.
                k = min(int(self.mc_lower_bound_topk), int(self.action_dim))
                rank_q = q_values.detach().mean(dim=0)
                topk_actions = torch.topk(rank_q, k=k, dim=-1).indices
                topk_mask = torch.zeros_like(is_data_action, dtype=torch.bool)
                topk_mask.scatter_(dim=2, index=topk_actions.unsqueeze(0), value=True)
                scope_mask = topk_mask & (~is_data_action)
            else:
                raise RuntimeError(f"Unexpected mc_lower_bound_scope: {self.mc_lower_bound_scope}")
            apply_mask = valid & scope_mask
            bound_events = (q_for_lse < mc) & apply_mask
            denom = apply_mask.expand_as(q_for_lse).float().sum().clamp_min(1.0)
            mc_bound_rate = bound_events.float().sum() / denom
            valid_count = valid.float().sum().clamp_min(1.0)
            mc_bound_actions_per_valid = apply_mask.float().sum() / valid_count
            q_for_lse = torch.where(apply_mask, torch.maximum(q_for_lse, mc), q_for_lse)
        q_for_lse, q_data = self._subsample_critic_values(q_for_lse, q_data)
        temp = float(self.cql_temp)
        cql_ood_unnormalized = temp * torch.logsumexp(q_for_lse / temp, dim=-1)
        cql_ood = cql_ood_unnormalized - temp * math.log(float(q_for_lse.shape[-1]))
        cql_diff = cql_ood - q_data
        cql_loss = cql_diff.mean()
        return cql_loss, {
            "train/cql_loss": float(cql_loss.detach().item()),
            "train/cql_loss_unnormalized": float(
                (cql_ood_unnormalized - q_data).mean().detach().item()
            ),
            "train/cql_ood_q_mean": float(cql_ood.detach().mean().item()),
            "train/cql_ood_q_mean_unnormalized": float(cql_ood_unnormalized.detach().mean().item()),
            "train/cql_data_q_mean": float(q_data.detach().mean().item()),
            "train/cql_ensemble_size": float(q_for_lse.shape[0]),
            "train/mc_bound_rate": float(mc_bound_rate.detach().item()),
            "train/mc_bound_actions_per_valid": float(mc_bound_actions_per_valid.detach().item()),
        }

    def _td_element_loss(self, q_a: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
        target = target_q.view(1, -1).expand_as(q_a)
        if self.td_loss_type == "huber":
            return F.smooth_l1_loss(
                q_a,
                target,
                beta=float(self.td_huber_delta),
                reduction="none",
            )
        return (q_a - target) ** 2

    def update(
        self,
        batch: ReplayBatch,
        update_actor: bool = True,
        update_alpha: bool = True,
        transition_weight_mode: str = "none",
        actor_prior_coeff: float = 0.0,
        actor_prior_min_action_id: int = 0,
        legal_min_action_id: int = 0,
    ) -> Dict[str, float]:
        self._check_batch(batch)

        obs = batch.obs
        next_obs = batch.next_obs
        actions = batch.actions
        rewards = batch.rewards
        durations = batch.durations
        dones = batch.dones

        mode = str(transition_weight_mode).lower()
        if mode == "none":
            weights = torch.ones_like(durations)
        elif mode == "duration":
            weights = durations
        elif mode == "sqrt_duration":
            weights = torch.sqrt(durations)
        else:
            raise ValueError(f"Unknown transition_weight_mode: {transition_weight_mode}")
        weights = weights / torch.clamp(weights.mean(), min=1e-6)
        legal_min_action_id = int(legal_min_action_id)
        if legal_min_action_id < 0 or legal_min_action_id >= self.action_dim:
            if legal_min_action_id != 0:
                raise RuntimeError(f"legal_min_action_id out of range: {legal_min_action_id}")

        # ----- Critic target -----
        target_next_obs_sample_count = 1
        target_next_v_sample_std = torch.zeros((), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            if batch.next_obs_samples is not None:
                sample_obs = batch.next_obs_samples
                if (
                    sample_obs.ndim != 3
                    or sample_obs.shape[0] != next_obs.shape[0]
                    or sample_obs.shape[2] != next_obs.shape[1]
                ):
                    raise RuntimeError(
                        f"next_obs_samples must have shape [B, M, obs_dim], got {tuple(sample_obs.shape)}"
                    )
                target_next_obs_sample_count = int(sample_obs.shape[1])
                flat_next_obs = sample_obs.reshape(-1, sample_obs.shape[-1])
                v_each = self._target_v_from_next_obs(flat_next_obs, legal_min_action_id).view(
                    next_obs.shape[0], target_next_obs_sample_count
                )
                v_next = v_each.mean(dim=1)
                target_next_v_sample_std = v_each.std(dim=1, unbiased=False).mean()
            else:
                v_next = self._target_v_from_next_obs(next_obs, legal_min_action_id)
            target_q = smdp_backup(
                rewards=rewards,
                dones=dones,
                gamma=self.gamma,
                durations=durations,
                v_next=v_next,
            )
            if self.target_q_clip_min is not None or self.target_q_clip_max is not None:
                target_q = torch.clamp(
                    target_q,
                    min=self.target_q_clip_min,
                    max=self.target_q_clip_max,
                )

        # ----- Critic update -----
        q_values, q_logits = self._critic_output(self.critic, obs)
        action_idx = actions.view(1, -1, 1).expand(q_values.shape[0], -1, 1)
        q_a = q_values.gather(2, action_idx).squeeze(2)
        mask_metrics = {
            "train/critic_bootstrap_mask_prob": float(self.critic_bootstrap_mask_prob),
            "train/critic_bootstrap_mask_mean": 1.0,
        }
        if q_logits is not None:
            td_loss = self._distributional_td_loss(q_logits, actions, target_q, weights)
        else:
            member_weights = weights.view(1, -1)
            td_element_loss = self._td_element_loss(q_a, target_q)
            if float(self.critic_bootstrap_mask_prob) < 1.0 and q_a.shape[0] > 1:
                mask = (
                    torch.rand_like(q_a, dtype=torch.float32)
                    < float(self.critic_bootstrap_mask_prob)
                ).float()
                empty_members = mask.sum(dim=1) <= 0.0
                if bool(empty_members.any().item()):
                    forced_idx = torch.randint(
                        low=0,
                        high=int(mask.shape[1]),
                        size=(int(empty_members.sum().item()),),
                        device=mask.device,
                    )
                    mask[empty_members, forced_idx] = 1.0
                member_weights = member_weights * mask
                denom = torch.clamp(member_weights.sum(), min=1.0)
                td_loss = (member_weights * td_element_loss).sum() / denom
                mask_metrics["train/critic_bootstrap_mask_mean"] = float(
                    mask.mean().detach().item()
                )
            else:
                td_loss = (member_weights * td_element_loss).mean()
        critic_loss = td_loss
        q_cap_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        if self.critic_q_upper_bound is not None and float(self.critic_q_upper_penalty) > 0.0:
            q_cap_loss = F.relu(q_values - float(self.critic_q_upper_bound)).pow(2).mean()
            critic_loss = critic_loss + float(self.critic_q_upper_penalty) * q_cap_loss
        cql_metrics = {
            "train/cql_loss": 0.0,
            "train/cql_loss_unnormalized": 0.0,
            "train/cql_ood_q_mean": 0.0,
            "train/cql_ood_q_mean_unnormalized": 0.0,
            "train/cql_data_q_mean": 0.0,
            "train/cql_ensemble_size": 0.0,
            "train/mc_bound_rate": 0.0,
            "train/mc_bound_actions_per_valid": 0.0,
        }
        if float(self.cql_alpha) > 0.0:
            cql_loss, cql_metrics = self._cql_loss(q_values, actions, batch)
            critic_loss = critic_loss + float(self.cql_alpha) * cql_loss
        if not torch.isfinite(critic_loss):
            raise RuntimeError("critic_loss is NaN/Inf")

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # ----- Actor / Alpha update -----
        probs = None
        log_probs = None
        entropy = torch.zeros((), device=self.device, dtype=torch.float32)
        if self.actor is not None:
            _, probs, log_probs = self.actor.distribution(obs, min_action_id=legal_min_action_id)
            with torch.no_grad():
                entropy_per_state = -(probs * log_probs).sum(dim=1)
                entropy = (weights * entropy_per_state).mean()

        actor_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        q_span_mean = torch.zeros((), device=self.device, dtype=torch.float32)
        q_argmax_k_mean = torch.zeros((), device=self.device, dtype=torch.float32)
        actor_prior_ce = torch.zeros((), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            min_q_diag = q_values.detach().min(dim=0).values
            if legal_min_action_id > 0:
                masked_min_q_diag = min_q_diag.clone()
                masked_min_q_diag[:, :legal_min_action_id] = -1e9
                q_span_slice = min_q_diag[:, legal_min_action_id:]
            else:
                masked_min_q_diag = min_q_diag
                q_span_slice = min_q_diag
            q_span_mean = (q_span_slice.max(dim=1).values - q_span_slice.min(dim=1).values).mean()
            q_argmax_k_mean = (torch.argmax(masked_min_q_diag, dim=1).float() + 1.0).mean()
        if update_actor:
            if self.actor is None or probs is None or log_probs is None:
                raise RuntimeError("update_actor=True requires actor_enabled=True")
            q_values_pi, _ = self._critic_output(self.critic, obs)
            min_q_pi = q_values_pi.min(dim=0).values
            alpha_detached = self.alpha.detach()
            actor_loss_per_state = torch.sum(probs * (alpha_detached * log_probs - min_q_pi), dim=1)
            actor_loss = (weights * actor_loss_per_state).mean()
            if float(actor_prior_coeff) > 0.0:
                min_id = int(actor_prior_min_action_id)
                if not (0 <= min_id < self.action_dim):
                    raise RuntimeError(f"actor_prior_min_action_id out of range: {min_id}")
                prior = torch.zeros_like(probs)
                prior[:, min_id:] = 1.0 / float(self.action_dim - min_id)
                actor_prior_ce = (weights * (-(prior * log_probs).sum(dim=1))).mean()
                actor_loss = actor_loss + float(actor_prior_coeff) * actor_prior_ce
            if not torch.isfinite(actor_loss):
                raise RuntimeError("actor_loss is NaN/Inf")

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()

        alpha_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        if update_alpha and self.alpha_opt is not None:
            alpha_loss = (self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
            if not torch.isfinite(alpha_loss):
                raise RuntimeError("alpha_loss is NaN/Inf")

            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            self._clamp_log_alpha_()

        self._soft_update_target()

        return {
            "train/critic_loss": float(critic_loss.item()),
            "train/td_loss": float(td_loss.detach().item()),
            "train/q_cap_loss": float(q_cap_loss.detach().item()),
            "train/q_cap_penalty": float(self.critic_q_upper_penalty),
            "train/q_upper_bound": float(
                self.critic_q_upper_bound if self.critic_q_upper_bound is not None else 0.0
            ),
            "train/td_loss_type_id": float({"mse": 0, "huber": 1}[self.td_loss_type]),
            "train/target_q_clip_min": float(
                self.target_q_clip_min if self.target_q_clip_min is not None else -1.0e9
            ),
            "train/target_q_clip_max": float(
                self.target_q_clip_max if self.target_q_clip_max is not None else 1.0e9
            ),
            "train/actor_loss": float(actor_loss.item()),
            "train/alpha_loss": float(alpha_loss.item()),
            "train/alpha": float(self.alpha.detach().item()),
            "train/entropy": float(entropy.detach().item()),
            "train/transition_weight_mean": float(weights.detach().mean().item()),
            "train/q1_mean": float(q_a[0].detach().mean().item()),
            "train/q2_mean": float(q_a[min(1, q_a.shape[0] - 1)].detach().mean().item()),
            "train/q_ensemble_min_mean": float(q_a.detach().min(dim=0).values.mean().item()),
            "train/q_ensemble_size": float(q_values.shape[0]),
            "train/target_q_mean": float(target_q.detach().mean().item()),
            "train/td_target_next_obs_samples": float(target_next_obs_sample_count),
            "train/td_target_v_sample_std_mean": float(target_next_v_sample_std.detach().item()),
            "train/q_span_mean": float(q_span_mean.detach().item()),
            "train/q_argmax_k_mean": float(q_argmax_k_mean.detach().item()),
            "train/actor_prior_ce": float(actor_prior_ce.detach().item()),
            "train/target_policy_mode_id": float(
                {"actor": 0, "q_soft": 1, "q_greedy": 2}[self.target_policy_mode]
            ),
            "train/distributional_critic": float(q_logits is not None),
            **mask_metrics,
            **cql_metrics,
        }

    def state_dict(self) -> Dict:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "tau": self.tau,
            "target_entropy": self.target_entropy,
            "fixed_alpha": self.fixed_alpha,
            "max_alpha": self.max_alpha,
            "activation": self.activation,
            "actor_use_layernorm": self.actor_use_layernorm,
            "critic_use_layernorm": self.critic_use_layernorm,
            "actor_enabled": self.actor_enabled,
            "target_policy_mode": self.target_policy_mode,
            "critic_backup_temp": self.critic_backup_temp,
            "cql_alpha": self.cql_alpha,
            "cql_temp": self.cql_temp,
            "use_mc_lower_bound": self.use_mc_lower_bound,
            "mc_lower_bound_scope": self.mc_lower_bound_scope,
            "mc_lower_bound_topk": self.mc_lower_bound_topk,
            "critic_ensemble_size": self.critic_ensemble_size,
            "target_critic_subsample_size": self.target_critic_subsample_size,
            "critic_bootstrap_mask_prob": self.critic_bootstrap_mask_prob,
            "distributional_critic": self.distributional_critic,
            "distributional_q_min": self.distributional_q_min,
            "distributional_q_max": self.distributional_q_max,
            "distributional_num_bins": self.distributional_num_bins,
            "td_loss_type": self.td_loss_type,
            "td_huber_delta": self.td_huber_delta,
            "target_q_clip_min": self.target_q_clip_min,
            "target_q_clip_max": self.target_q_clip_max,
            "critic_q_upper_bound": self.critic_q_upper_bound,
            "critic_q_upper_penalty": self.critic_q_upper_penalty,
            "actor": None if self.actor is None else self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": None if self.actor_opt is None else self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_opt": None if self.alpha_opt is None else self.alpha_opt.state_dict(),
        }

    def load_state_dict(self, state: Dict) -> None:
        if int(state["obs_dim"]) != self.obs_dim or int(state["action_dim"]) != self.action_dim:
            raise ValueError("Model shape mismatch while loading SAC state")
        if self.actor is not None and state.get("actor") is not None:
            self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        if self.actor_opt is not None and state.get("actor_opt") is not None:
            self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.fixed_alpha = state.get("fixed_alpha", self.fixed_alpha)
        self.max_alpha = state.get("max_alpha", self.max_alpha)
        self.target_policy_mode = state.get("target_policy_mode", self.target_policy_mode)
        self.critic_backup_temp = float(state.get("critic_backup_temp", self.critic_backup_temp))
        self.cql_alpha = float(state.get("cql_alpha", self.cql_alpha))
        self.cql_temp = float(state.get("cql_temp", self.cql_temp))
        self.use_mc_lower_bound = bool(state.get("use_mc_lower_bound", self.use_mc_lower_bound))
        self.mc_lower_bound_scope = str(
            state.get("mc_lower_bound_scope", self.mc_lower_bound_scope)
        ).lower()
        self.mc_lower_bound_topk = int(state.get("mc_lower_bound_topk", self.mc_lower_bound_topk))
        self.target_critic_subsample_size = int(
            state.get("target_critic_subsample_size", self.target_critic_subsample_size)
        )
        self.critic_bootstrap_mask_prob = float(
            state.get("critic_bootstrap_mask_prob", self.critic_bootstrap_mask_prob)
        )
        self.td_loss_type = str(state.get("td_loss_type", self.td_loss_type)).lower()
        self.td_huber_delta = float(state.get("td_huber_delta", self.td_huber_delta))
        self.target_q_clip_min = state.get("target_q_clip_min", self.target_q_clip_min)
        self.target_q_clip_max = state.get("target_q_clip_max", self.target_q_clip_max)
        self.critic_q_upper_bound = state.get("critic_q_upper_bound", self.critic_q_upper_bound)
        self.critic_q_upper_penalty = float(
            state.get("critic_q_upper_penalty", self.critic_q_upper_penalty)
        )
        with torch.no_grad():
            self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self._clamp_log_alpha_()
        if self.alpha_opt is not None and state.get("alpha_opt") is not None:
            self.alpha_opt.load_state_dict(state["alpha_opt"])


def run_core_sanity_checks(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Lightweight sanity checks for:
    - SMDP backup math
    - action/duration range checks
    - alpha loss detach path can run end-to-end
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sac = SMDPDiscreteSAC(obs_dim=12, action_dim=32, hidden_dims=(64, 64), device=dev)
    rb = SMDPReplayBuffer(obs_dim=12, capacity=1024, device=dev)

    # Replay shape/range checks
    for i in range(256):
        obs = np.random.randn(12).astype(np.float32)
        next_obs = np.random.randn(12).astype(np.float32)
        action = np.random.randint(0, 32)
        reward = float(np.random.randn())
        duration = float(np.random.randint(1, 33))
        done = bool(np.random.rand() < 0.1)
        rb.add(obs, action, reward, duration, next_obs, done, action_dim=32)

    batch = rb.sample(128)
    metrics = sac.update(batch)
    actorless = SMDPDiscreteSAC(
        obs_dim=12,
        action_dim=32,
        hidden_dims=(64, 64),
        actor_enabled=False,
        target_policy_mode="q_soft",
        cql_alpha=0.005,
        device=dev,
    )
    metrics_actorless = actorless.update(batch, update_actor=False, update_alpha=False)
    if not np.isfinite(metrics_actorless["train/critic_loss"]):
        raise RuntimeError("actorless critic loss not finite")

    dist = SMDPDiscreteSAC(
        obs_dim=12,
        action_dim=32,
        hidden_dims=(64,),
        actor_enabled=False,
        target_policy_mode="q_greedy",
        distributional_critic=True,
        distributional_q_min=-40.0,
        distributional_q_max=5.0,
        distributional_num_bins=32,
        device=dev,
    )
    metrics_dist = dist.update(batch, update_actor=False, update_alpha=False)
    if not np.isfinite(metrics_dist["train/critic_loss"]):
        raise RuntimeError("distributional critic loss not finite")

    # Closed-form backup check
    r = torch.tensor([1.0], device=dev)
    d = torch.tensor([0.0], device=dev)
    k = torch.tensor([2.0], device=dev)
    v = torch.tensor([3.0], device=dev)
    target = smdp_backup(r, d, gamma=0.5, durations=k, v_next=v)
    expected = 1.0 + (0.5**2) * 3.0
    if abs(float(target.item()) - float(expected)) > 1e-6:
        raise RuntimeError("SMDP backup sanity check failed")

    return {"alpha": metrics["train/alpha"], "critic_loss": metrics["train/critic_loss"]}
