from dataclasses import dataclass
from collections import deque
import math
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


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    durations: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


@dataclass
class PendingSMDPTransition:
    obs: np.ndarray
    action_id: int
    reward: float
    duration_k: float
    next_obs: np.ndarray
    done: bool
    episode_done: bool


@dataclass
class AggregatedSMDPTransition:
    obs: np.ndarray
    action_id: int
    reward: float
    duration_k: float
    next_obs: np.ndarray
    done: bool


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
    done = bool(first.done)

    for tr in list(transitions)[:n_step]:
        reward_acc += discount * float(tr.reward)
        duration_acc += float(tr.duration_k)
        next_obs = np.asarray(tr.next_obs, dtype=np.float32).reshape(-1)
        done = bool(tr.done)
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
    )


def drain_smdp_nstep_queue_to_replay(
    pending_queue: Deque[PendingSMDPTransition],
    replay: "SMDPReplayBuffer",
    gamma: float,
    n_step: int,
    action_dim: Optional[int] = None,
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
        )
        pending_queue.popleft()
        emitted += 1

    return emitted


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
) -> IngestedEnvStepResult:
    k = float(info.get("exec_len", 1.0))
    if (k < 0.0) or (k > float(max_chunk_len)):
        raise RuntimeError(f"duration_k out of range: {k} not in [0, {max_chunk_len}]")
    truncated = bool(info.get("TimeLimit.truncated", False))
    done_for_backup = bool(done_i) and (not truncated)

    replay_next_obs = np.asarray(next_obs_i, dtype=np.float32).reshape(-1)
    if bool(done_i) and (not done_for_backup):
        replay_next_obs = extract_bootstrap_next_obs_from_venv(
            venv=venv,
            env_idx=int(env_idx),
            info=info,
            fallback_next_obs=replay_next_obs,
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
        )
    )
    emitted_transitions = drain_smdp_nstep_queue_to_replay(
        pending_queue,
        replay=replay,
        gamma=float(gamma),
        n_step=int(n_step),
        action_dim=action_dim,
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

    def __init__(self, obs_dim: int, capacity: int, device: torch.device):
        self.obs_dim = int(obs_dim)
        self.capacity = int(capacity)
        self.device = device
        self.pos = 0
        self.size = 0

        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.durations = np.zeros((self.capacity,), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return int(self.size)

    def add(
        self,
        obs: np.ndarray,
        action_id: int,
        reward: float,
        duration_k: float,
        next_obs: np.ndarray,
        done: bool,
        action_dim: Optional[int] = None,
    ) -> None:
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
        self.dones[i] = 1.0 if bool(done) else 0.0

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

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
        )

    def state_dict(self) -> Dict:
        stored_size = int(self.size)
        return {
            "obs_dim": int(self.obs_dim),
            "capacity": int(self.capacity),
            "pos": int(self.pos),
            "size": stored_size,
            "obs": self.obs[:stored_size].copy(),
            "actions": self.actions[:stored_size].copy(),
            "rewards": self.rewards[:stored_size].copy(),
            "durations": self.durations[:stored_size].copy(),
            "next_obs": self.next_obs[:stored_size].copy(),
            "dones": self.dones[:stored_size].copy(),
        }

    def load_state_dict(self, state: Dict) -> None:
        if int(state.get("obs_dim", -1)) != int(self.obs_dim):
            raise ValueError("Replay obs_dim mismatch while loading state")
        stored_size = int(state.get("size", 0))
        if stored_size < 0 or stored_size > int(self.capacity):
            raise ValueError(f"Replay size out of range while loading state: {stored_size}")
        if int(state.get("capacity", self.capacity)) > int(self.capacity):
            raise ValueError(
                f"Replay capacity mismatch while loading state: checkpoint={state.get('capacity')} current={self.capacity}"
            )

        obs = np.asarray(state["obs"], dtype=np.float32)
        next_obs = np.asarray(state["next_obs"], dtype=np.float32)
        if obs.shape != (stored_size, self.obs_dim) or next_obs.shape != (
            stored_size,
            self.obs_dim,
        ):
            raise ValueError("Replay observation shape mismatch while loading state")

        self.obs.fill(0.0)
        self.actions.fill(0)
        self.rewards.fill(0.0)
        self.durations.fill(0.0)
        self.next_obs.fill(0.0)
        self.dones.fill(0.0)
        if stored_size > 0:
            self.obs[:stored_size] = obs
            self.actions[:stored_size] = np.asarray(state["actions"], dtype=np.int64)[:stored_size]
            self.rewards[:stored_size] = np.asarray(state["rewards"], dtype=np.float32)[
                :stored_size
            ]
            self.durations[:stored_size] = np.asarray(state["durations"], dtype=np.float32)[
                :stored_size
            ]
            self.next_obs[:stored_size] = next_obs
            self.dones[:stored_size] = np.asarray(state["dones"], dtype=np.float32)[:stored_size]
        self.size = stored_size
        self.pos = int(state.get("pos", stored_size % max(int(self.capacity), 1))) % int(
            self.capacity
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
        critic_backup_temp: float = 1.0,
        cql_alpha: float = 0.0,
        cql_temp: float = 1.0,
        critic_ensemble_size: int = 2,
        target_critic_subsample_size: int = 0,
        critic_bootstrap_mask_prob: float = 1.0,
        actor_target_critic_subsample: bool = False,
        actor_loss_critic_subsample: bool = False,
        target_q_clip_min: Optional[float] = None,
        target_q_clip_max: Optional[float] = None,
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
        self.critic_backup_temp = float(critic_backup_temp)
        self.cql_alpha = float(cql_alpha)
        self.cql_temp = float(cql_temp)
        self.critic_ensemble_size = int(critic_ensemble_size)
        self.target_critic_subsample_size = int(target_critic_subsample_size)
        self.critic_bootstrap_mask_prob = float(critic_bootstrap_mask_prob)
        self.actor_target_critic_subsample = bool(actor_target_critic_subsample)
        self.actor_loss_critic_subsample = bool(actor_loss_critic_subsample)
        self.target_q_clip_min = None if target_q_clip_min is None else float(target_q_clip_min)
        self.target_q_clip_max = None if target_q_clip_max is None else float(target_q_clip_max)
        if self.critic_backup_temp <= 0.0:
            raise ValueError(f"critic_backup_temp must be > 0, got {critic_backup_temp}")
        if self.cql_alpha < 0.0:
            raise ValueError(f"cql_alpha must be >= 0, got {cql_alpha}")
        if self.cql_temp <= 0.0:
            raise ValueError(f"cql_temp must be > 0, got {cql_temp}")
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
        if (
            self.target_q_clip_min is not None
            and self.target_q_clip_max is not None
            and self.target_q_clip_min > self.target_q_clip_max
        ):
            raise ValueError(
                f"target_q_clip_min ({self.target_q_clip_min}) cannot exceed "
                f"target_q_clip_max ({self.target_q_clip_max})"
            )

        self.actor = DiscreteActor(
            self.obs_dim,
            self.action_dim,
            hidden_dims,
            activation=self.activation,
            use_layernorm=self.actor_use_layernorm,
        ).to(self.device)
        self.critic = self._build_critic(hidden_dims).to(self.device)
        self.critic_target = self._build_critic(hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
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
        obs = torch.as_tensor(np.asarray(obs_np, dtype=np.float32), device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        acts = self.actor.act(obs, deterministic=deterministic, min_action_id=min_action_id)
        return acts.detach().cpu().numpy().astype(np.int64)

    def _critic_output(self, critic: nn.Module, obs: torch.Tensor):
        out = critic(obs)
        if isinstance(out, tuple) and len(out) == 2:
            first, second = out
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

    def _mask_q_values(self, q_values: torch.Tensor, min_action_id: int) -> torch.Tensor:
        min_action_id = int(min_action_id)
        if min_action_id < 0 or min_action_id >= int(self.action_dim):
            if min_action_id != 0:
                raise RuntimeError(f"min_action_id out of range: {min_action_id}")
        if min_action_id <= 0:
            return q_values
        masked = q_values.clone()
        masked[..., :min_action_id] = -1e9
        return masked

    def _critic_value_from_q(
        self,
        q_values: torch.Tensor,
        target_policy_mode: str,
        min_action_id: int,
    ) -> torch.Tensor:
        mode = str(target_policy_mode).lower()
        masked_q = self._mask_q_values(q_values, min_action_id=min_action_id)
        if mode == "q_greedy":
            return masked_q.max(dim=1).values
        if mode == "q_soft":
            temp = float(self.critic_backup_temp)
            return temp * torch.logsumexp(masked_q / temp, dim=1)
        raise ValueError(f"Unknown target_policy_mode: {target_policy_mode}")

    @torch.no_grad()
    def select_critic_action(
        self,
        obs_np: np.ndarray,
        deterministic: bool = True,
        min_action_id: int = 0,
        mode: str = "q_greedy",
        reduce: str = "min",
        return_info: bool = False,
    ):
        obs = torch.as_tensor(np.asarray(obs_np, dtype=np.float32), device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        q_values_ensemble, _ = self._critic_output(self.critic, obs)
        reduce_mode = str(reduce).lower()
        if reduce_mode == "min":
            q_values = q_values_ensemble.min(dim=0).values
        elif reduce_mode == "mean":
            q_values = q_values_ensemble.mean(dim=0)
        elif reduce_mode == "q1":
            q_values = q_values_ensemble[0]
        elif reduce_mode == "q2":
            q_values = q_values_ensemble[min(1, int(q_values_ensemble.shape[0]) - 1)]
        else:
            raise ValueError(f"Unknown critic reduce mode: {reduce}")

        mode = str(mode).lower()
        masked_q = self._mask_q_values(q_values, min_action_id=min_action_id)
        if deterministic or mode == "q_greedy":
            acts = torch.argmax(masked_q, dim=1)
        elif mode == "q_soft":
            probs = F.softmax(masked_q, dim=1)
            acts = torch.distributions.Categorical(probs=probs).sample()
        else:
            raise ValueError(f"Unknown critic policy mode: {mode}")

        acts_np = acts.detach().cpu().numpy().astype(np.int64)
        if not return_info:
            return acts_np
        info = {
            "critic/q_argmax_k_mean": float((acts.float() + 1.0).mean().item()),
            "critic/q_value_mean": float(q_values.detach().mean().item()),
            "critic/q_value_max_mean": float(masked_q.max(dim=1).values.detach().mean().item()),
        }
        return acts_np, info

    def _check_batch(self, batch: ReplayBatch) -> None:
        if torch.any(batch.actions < 0) or torch.any(batch.actions >= self.action_dim):
            raise RuntimeError("Batch action_id out of range")
        if torch.any(batch.durations < 0.0):
            raise RuntimeError("Batch duration_k must be >= 0")
        tensors = [batch.obs, batch.next_obs, batch.rewards, batch.durations, batch.dones]
        for t in tensors:
            if not torch.isfinite(t).all():
                raise RuntimeError("NaN/Inf detected in replay batch")

    def _soft_update_target(self) -> None:
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def _cql_loss(
        self,
        q_values: torch.Tensor,
        actions: torch.Tensor,
    ):
        action_idx = actions.view(1, -1, 1).expand(q_values.shape[0], -1, 1)
        q_data = q_values.gather(2, action_idx).squeeze(2)
        q_values, q_data = self._subsample_critic_values(q_values, q_data)
        temp = float(self.cql_temp)
        cql_ood_unnormalized = temp * torch.logsumexp(q_values / temp, dim=-1)
        cql_ood = cql_ood_unnormalized - temp * math.log(float(q_values.shape[-1]))
        cql_diff = cql_ood - q_data
        cql_diff_unnormalized = cql_ood_unnormalized - q_data
        cql_loss = cql_diff.mean()
        return cql_loss, {
            "train/cql_loss": float(cql_loss.detach().item()),
            "train/cql_loss_unnormalized": float(cql_diff_unnormalized.mean().detach().item()),
            "train/cql_margin": float(cql_diff.detach().mean().item()),
            "train/cql_margin_unnormalized": float(cql_diff_unnormalized.detach().mean().item()),
            "train/cql_ood_q_mean": float(cql_ood.detach().mean().item()),
            "train/cql_ood_q_mean_unnormalized": float(cql_ood_unnormalized.detach().mean().item()),
            "train/cql_data_q_mean": float(q_data.detach().mean().item()),
            "train/cql_ensemble_size": float(q_values.shape[0]),
        }

    def update(
        self,
        batch: ReplayBatch,
        update_actor: bool = True,
        update_alpha: bool = True,
        update_critic: bool = True,
        transition_weight_mode: str = "none",
        actor_prior_coeff: float = 0.0,
        actor_prior_min_action_id: int = 0,
        legal_min_action_id: int = 0,
        target_policy_mode: str = "actor",
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

        target_policy_mode = str(target_policy_mode).lower()
        if target_policy_mode not in ("actor", "q_greedy", "q_soft"):
            raise ValueError(f"Unknown target_policy_mode: {target_policy_mode}")

        cql_metrics = {
            "train/cql_loss": 0.0,
            "train/cql_loss_unnormalized": 0.0,
            "train/cql_margin": 0.0,
            "train/cql_margin_unnormalized": 0.0,
            "train/cql_ood_q_mean": 0.0,
            "train/cql_ood_q_mean_unnormalized": 0.0,
            "train/cql_data_q_mean": 0.0,
            "train/cql_ensemble_size": 0.0,
        }
        if update_critic:
            # ----- Critic target -----
            with torch.no_grad():
                next_q_values_t, _ = self._critic_output(self.critic_target, next_obs)
                if target_policy_mode == "actor":
                    if self.actor_target_critic_subsample:
                        next_q_values_t, _ = self._subsample_critic_values(next_q_values_t)
                    next_min_q_t = next_q_values_t.min(dim=0).values
                    _, next_probs, next_log_probs = self.actor.distribution(
                        next_obs, min_action_id=legal_min_action_id
                    )
                    alpha_detached = self.alpha.detach()
                    v_next = torch.sum(
                        next_probs * (next_min_q_t - alpha_detached * next_log_probs), dim=1
                    )
                else:
                    next_q_values_t, _ = self._subsample_critic_values(next_q_values_t)
                    next_min_q_t = next_q_values_t.min(dim=0).values
                    v_next = self._critic_value_from_q(
                        next_min_q_t,
                        target_policy_mode=target_policy_mode,
                        min_action_id=legal_min_action_id,
                    )
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
            q_values, _ = self._critic_output(self.critic, obs)
            action_idx = actions.view(1, -1, 1).expand(q_values.shape[0], -1, 1)
            q_a = q_values.gather(2, action_idx).squeeze(2)
            mask_metrics = {
                "train/critic_bootstrap_mask_prob": float(self.critic_bootstrap_mask_prob),
                "train/critic_bootstrap_mask_mean": 1.0,
            }
            legacy_twin_td = (
                int(self.critic_ensemble_size) == 2
                and int(self.target_critic_subsample_size) == 0
                and float(self.critic_bootstrap_mask_prob) == 1.0
            )
            if legacy_twin_td:
                td_loss = (weights * F.mse_loss(q_a[0], target_q, reduction="none")).mean() + (
                    weights * F.mse_loss(q_a[min(1, q_a.shape[0] - 1)], target_q, reduction="none")
                ).mean()
            else:
                member_weights = weights.view(1, -1)
                td_element_loss = (q_a - target_q.view(1, -1).expand_as(q_a)) ** 2
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
            if float(self.cql_alpha) > 0.0:
                cql_loss, cql_metrics = self._cql_loss(q_values, actions)
                critic_loss = critic_loss + float(self.cql_alpha) * cql_loss
            if not torch.isfinite(critic_loss):
                raise RuntimeError("critic_loss is NaN/Inf")

            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_opt.step()
        else:
            with torch.no_grad():
                q_values, _ = self._critic_output(self.critic, obs)
                action_idx = actions.view(1, -1, 1).expand(q_values.shape[0], -1, 1)
                q_a = q_values.gather(2, action_idx).squeeze(2)
            target_q = torch.zeros_like(q_a[0])
            td_loss = torch.zeros((), device=self.device, dtype=torch.float32)
            critic_loss = torch.zeros((), device=self.device, dtype=torch.float32)
            mask_metrics = {
                "train/critic_bootstrap_mask_prob": float(self.critic_bootstrap_mask_prob),
                "train/critic_bootstrap_mask_mean": 1.0,
            }

        # ----- Actor / Alpha update -----
        _, probs, log_probs = self.actor.distribution(obs, min_action_id=legal_min_action_id)
        with torch.no_grad():
            entropy_per_state = -(probs * log_probs).sum(dim=1)
            entropy = (weights * entropy_per_state).mean()

        actor_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        q_span_mean = torch.zeros((), device=self.device, dtype=torch.float32)
        q_argmax_k_mean = torch.zeros((), device=self.device, dtype=torch.float32)
        actor_prior_ce = torch.zeros((), device=self.device, dtype=torch.float32)
        diagnostic_metrics = {}
        with torch.no_grad():
            weights_detached = weights.detach()
            q_values_detached = q_values.detach()
            target_q_detached = target_q.detach()
            q_ensemble_std = q_values_detached.std(dim=0, unbiased=False)
            min_q_diag = q_values.detach().min(dim=0).values
            if legal_min_action_id > 0:
                masked_min_q_diag = min_q_diag.clone()
                masked_min_q_diag[:, :legal_min_action_id] = -1e9
                q_span_slice = min_q_diag[:, legal_min_action_id:]
            else:
                masked_min_q_diag = min_q_diag
                q_span_slice = min_q_diag
            q_span_mean = (q_span_slice.max(dim=1).values - q_span_slice.min(dim=1).values).mean()
            q_argmax_k = torch.argmax(masked_min_q_diag, dim=1).float() + 1.0
            q_argmax_k_mean = q_argmax_k.mean()

            action_ids = (
                torch.arange(
                    self.action_dim,
                    device=self.device,
                    dtype=torch.float32,
                )
                + 1.0
            )
            probs_detached = probs.detach()
            actor_expected_k_per_state = (probs_detached * action_ids.view(1, -1)).sum(dim=1)
            actor_mode_k = torch.argmax(probs_detached, dim=1).float() + 1.0
            actor_top1_prob = probs_detached.max(dim=1).values
            valid_action_count = max(int(self.action_dim) - int(legal_min_action_id), 1)
            entropy_norm = (
                entropy / float(math.log(float(valid_action_count)))
                if valid_action_count > 1
                else torch.zeros_like(entropy)
            )
            prob_k_ge_25 = probs_detached[:, action_ids >= 25.0].sum(dim=1)
            prob_k_ge_40 = probs_detached[:, action_ids >= 40.0].sum(dim=1)

            diagnostic_metrics = {
                "train/q_mean": float(q_values_detached.mean().item()),
                "train/q_std": float(q_values_detached.std(unbiased=False).item()),
                "train/q_min": float(q_values_detached.min().item()),
                "train/q_max": float(q_values_detached.max().item()),
                "train/q_abs_max": float(q_values_detached.abs().max().item()),
                "train/q_data_mean": float(q_a.detach().mean().item()),
                "train/q_data_std": float(q_a.detach().std(unbiased=False).item()),
                "train/q_ensemble_std_mean": float(q_ensemble_std.mean().item()),
                "train/q_ensemble_std_max": float(q_ensemble_std.max().item()),
                "train/target_q_abs_max": float(target_q_detached.abs().max().item()),
                "train/actor_entropy": float(entropy.detach().item()),
                "train/actor_entropy_norm": float(entropy_norm.detach().item()),
                "train/actor_top1_prob": float((weights_detached * actor_top1_prob).mean().item()),
                "train/actor_expected_k": float(
                    (weights_detached * actor_expected_k_per_state).mean().item()
                ),
                "train/actor_mode_k_mean": float(actor_mode_k.mean().item()),
                "train/actor_prob_k_ge_25": float((weights_detached * prob_k_ge_25).mean().item()),
                "train/actor_prob_k_ge_40": float((weights_detached * prob_k_ge_40).mean().item()),
                "train/q_argmax_k_std": float(q_argmax_k.std(unbiased=False).item()),
                "train/q_argmax_k_frac_le_3": float((q_argmax_k <= 3.0).float().mean().item()),
                "train/q_argmax_k_frac_ge_25": float((q_argmax_k >= 25.0).float().mean().item()),
                "train/q_argmax_k_frac_ge_40": float((q_argmax_k >= 40.0).float().mean().item()),
                "train/actor_expected_k_minus_q_argmax_k": float(
                    (actor_expected_k_per_state - q_argmax_k).mean().item()
                ),
                "train/actor_mode_k_minus_q_argmax_k": float(
                    (actor_mode_k - q_argmax_k).mean().item()
                ),
                "train/actor_mode_eq_q_argmax_frac": float(
                    (actor_mode_k == q_argmax_k).float().mean().item()
                ),
            }
        if update_actor:
            critic_requires_grad = [p.requires_grad for p in self.critic.parameters()]
            for p in self.critic.parameters():
                p.requires_grad_(False)
            q_values_pi, _ = self._critic_output(self.critic, obs)
            if self.actor_loss_critic_subsample:
                q_values_pi, _ = self._subsample_critic_values(q_values_pi)
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
            try:
                actor_loss.backward()
                self.actor_opt.step()
            finally:
                for p, requires_grad in zip(self.critic.parameters(), critic_requires_grad):
                    p.requires_grad_(requires_grad)

        alpha_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        if update_alpha and self.alpha_opt is not None:
            alpha_loss = (self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
            if not torch.isfinite(alpha_loss):
                raise RuntimeError("alpha_loss is NaN/Inf")

            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            self._clamp_log_alpha_()

        if update_critic:
            self._soft_update_target()

        return {
            "train/critic_loss": float(critic_loss.item()),
            "train/td_loss": float(td_loss.detach().item()),
            "train/actor_loss": float(actor_loss.item()),
            "train/alpha_loss": float(alpha_loss.item()),
            "train/alpha": float(self.alpha.detach().item()),
            "train/entropy": float(entropy.detach().item()),
            "train/transition_weight_mean": float(weights.detach().mean().item()),
            "train/q1_mean": float(q_a[0].detach().mean().item()),
            "train/q2_mean": float(q_a[min(1, q_a.shape[0] - 1)].detach().mean().item()),
            "train/q_ensemble_min_mean": float(q_a.detach().min(dim=0).values.mean().item()),
            "train/q_ensemble_size": float(q_values.shape[0]),
            "train/actor_target_critic_subsample": float(bool(self.actor_target_critic_subsample)),
            "train/actor_loss_critic_subsample": float(bool(self.actor_loss_critic_subsample)),
            "train/actor_loss_critic_ensemble_size": float(q_values_pi.shape[0])
            if update_actor
            else 0.0,
            "train/target_q_clip_min": float(
                self.target_q_clip_min if self.target_q_clip_min is not None else -1.0e9
            ),
            "train/target_q_clip_max": float(
                self.target_q_clip_max if self.target_q_clip_max is not None else 1.0e9
            ),
            "train/target_q_mean": float(target_q.detach().mean().item()),
            "train/q_span_mean": float(q_span_mean.detach().item()),
            "train/q_argmax_k_mean": float(q_argmax_k_mean.detach().item()),
            "train/actor_prior_ce": float(actor_prior_ce.detach().item()),
            "train/target_policy_mode_id": float(
                {"actor": 0, "q_greedy": 1, "q_soft": 2}[target_policy_mode]
            ),
            "train/update_critic": float(bool(update_critic)),
            "train/update_actor": float(bool(update_actor)),
            "train/update_alpha": float(bool(update_alpha and self.alpha_opt is not None)),
            **diagnostic_metrics,
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
            "critic_backup_temp": self.critic_backup_temp,
            "cql_alpha": self.cql_alpha,
            "cql_temp": self.cql_temp,
            "critic_ensemble_size": self.critic_ensemble_size,
            "target_critic_subsample_size": self.target_critic_subsample_size,
            "critic_bootstrap_mask_prob": self.critic_bootstrap_mask_prob,
            "actor_target_critic_subsample": self.actor_target_critic_subsample,
            "actor_loss_critic_subsample": self.actor_loss_critic_subsample,
            "target_q_clip_min": self.target_q_clip_min,
            "target_q_clip_max": self.target_q_clip_max,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_opt": None if self.alpha_opt is None else self.alpha_opt.state_dict(),
        }

    def load_state_dict(self, state: Dict) -> None:
        if int(state["obs_dim"]) != self.obs_dim or int(state["action_dim"]) != self.action_dim:
            raise ValueError("Model shape mismatch while loading SAC state")
        loaded_critic_ensemble_size = int(state.get("critic_ensemble_size", 2))
        if loaded_critic_ensemble_size != int(self.critic_ensemble_size):
            raise ValueError(
                "Critic ensemble size mismatch while loading SAC state: "
                f"checkpoint={loaded_critic_ensemble_size}, current={self.critic_ensemble_size}"
            )
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.fixed_alpha = state.get("fixed_alpha", self.fixed_alpha)
        self.max_alpha = state.get("max_alpha", self.max_alpha)
        self.critic_backup_temp = float(state.get("critic_backup_temp", self.critic_backup_temp))
        self.cql_alpha = float(state.get("cql_alpha", self.cql_alpha))
        self.cql_temp = float(state.get("cql_temp", self.cql_temp))
        self.target_critic_subsample_size = int(
            state.get("target_critic_subsample_size", self.target_critic_subsample_size)
        )
        self.critic_bootstrap_mask_prob = float(
            state.get("critic_bootstrap_mask_prob", self.critic_bootstrap_mask_prob)
        )
        self.actor_target_critic_subsample = bool(
            state.get("actor_target_critic_subsample", self.actor_target_critic_subsample)
        )
        self.actor_loss_critic_subsample = bool(
            state.get("actor_loss_critic_subsample", self.actor_loss_critic_subsample)
        )
        self.target_q_clip_min = state.get("target_q_clip_min", self.target_q_clip_min)
        self.target_q_clip_max = state.get("target_q_clip_max", self.target_q_clip_max)
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
    offline_metrics = sac.update(
        rb.sample(128),
        update_actor=False,
        update_alpha=False,
        target_policy_mode="q_greedy",
    )
    cql_sac = SMDPDiscreteSAC(
        obs_dim=12,
        action_dim=32,
        hidden_dims=(64, 64),
        cql_alpha=0.005,
        device=dev,
    )
    cql_metrics = cql_sac.update(rb.sample(128), update_actor=False, update_alpha=False)
    if not np.isfinite(cql_metrics["train/cql_loss"]):
        raise RuntimeError("CQL loss sanity check failed")
    critic_actions = sac.select_critic_action(
        np.random.randn(4, 12).astype(np.float32), min_action_id=0
    )
    if critic_actions.shape != (4,):
        raise RuntimeError("critic action selection sanity check failed")

    rb2 = SMDPReplayBuffer(obs_dim=12, capacity=1024, device=dev)
    rb2.load_state_dict(rb.state_dict())
    if len(rb2) != len(rb):
        raise RuntimeError("Replay state_dict sanity check failed")

    # Closed-form backup check
    r = torch.tensor([1.0], device=dev)
    d = torch.tensor([0.0], device=dev)
    k = torch.tensor([2.0], device=dev)
    v = torch.tensor([3.0], device=dev)
    target = smdp_backup(r, d, gamma=0.5, durations=k, v_next=v)
    expected = 1.0 + (0.5**2) * 3.0
    if abs(float(target.item()) - float(expected)) > 1e-6:
        raise RuntimeError("SMDP backup sanity check failed")

    return {
        "alpha": metrics["train/alpha"],
        "critic_loss": metrics["train/critic_loss"],
        "offline_critic_loss": offline_metrics["train/critic_loss"],
        "cql_loss": cql_metrics["train/cql_loss"],
    }
