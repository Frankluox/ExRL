"""
Build offline ExRL replay transitions from flat DSRL demo npz files.

The official low-dim DSRL datasets store one state per demo action and do not
store the post-terminal observation. We therefore only use full action-window
starts and mask bootstrap at the final successful action.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class OfflineDemoReplaySummary:
    dataset_path: str
    window_mode: str
    critic_obs_state_mode: str
    critic_obs_action_mode: str
    critic_obs_unnormalize_dp_chunk: bool
    episodes_available: int
    episodes_used: int
    state_count: int
    action_count: int
    max_chunk_len: int
    min_chunk_len: int
    full_window_starts: int
    transitions_added: int
    terminal_count: int
    nonterminal_count: int
    skipped_short_episodes: int
    target_next_obs_samples: int
    cached_target_states: int
    duration_sum: float
    reward_sum: float
    reward_min: float
    reward_max: float
    max_transitions_hit: bool

    def as_metrics(self, prefix: str = "offline_demo") -> Dict[str, float]:
        data = asdict(self)
        out = {}
        for key, value in data.items():
            if isinstance(value, (int, float, bool, np.integer, np.floating)):
                out[f"{prefix}/{key}"] = float(value)
        return out


class OfflineDemoRLObsEncoder:
    def __init__(
        self,
        *,
        stats_npz: str,
        normalization_npz: str,
        lowdim_keys: Sequence[str],
        lowdim_dims: Dict[str, int],
        action_dim: int,
        max_chunk_len: int,
        action_history_len: int = 0,
        critic_obs_state_mode: str,
        critic_obs_action_mode: str,
        critic_obs_unnormalize_dp_chunk: bool,
    ):
        if int(action_history_len) != 0:
            raise ValueError(
                "offline demo replay currently requires --action_history_len 0"
            )
        if not os.path.exists(stats_npz):
            raise FileNotFoundError(f"stats npz not found: {stats_npz}")
        self.stats_npz = str(stats_npz)
        self.stats = np.load(stats_npz)
        self.lowdim_keys = list(lowdim_keys)
        self.lowdim_dims = {str(k): int(v) for k, v in lowdim_dims.items()}
        self.action_dim = int(action_dim)
        self.max_chunk_len = int(max_chunk_len)
        self.action_history_len = int(action_history_len)
        self.critic_obs_state_mode = str(critic_obs_state_mode).lower()
        valid_state_modes = {"stats_clip", "stats_no_clip", "dsrl_raw"}
        if self.critic_obs_state_mode not in valid_state_modes:
            raise ValueError(
                f"invalid critic_obs_state_mode={self.critic_obs_state_mode}, "
                f"expected one of {sorted(valid_state_modes)}"
            )

        self.critic_obs_action_mode = str(critic_obs_action_mode).lower()
        if bool(critic_obs_unnormalize_dp_chunk):
            if self.critic_obs_action_mode not in {"stats", "dsrl_unnorm_stats"}:
                raise ValueError(
                    "critic_obs_unnormalize_dp_chunk conflicts with "
                    f"critic_obs_action_mode={self.critic_obs_action_mode}"
                )
            self.critic_obs_action_mode = "dsrl_unnorm_stats"
        valid_action_modes = {"stats", "dsrl_unnorm_stats", "dp_raw"}
        if self.critic_obs_action_mode not in valid_action_modes:
            raise ValueError(
                f"invalid critic_obs_action_mode={self.critic_obs_action_mode}, "
                f"expected one of {sorted(valid_action_modes)}"
            )
        self.critic_obs_unnormalize_dp_chunk = (
            self.critic_obs_action_mode == "dsrl_unnorm_stats"
        )

        self.action_mean = np.asarray(
            self.stats["action_mean"], dtype=np.float32
        ).reshape(1, 1, -1)
        self.action_std = np.asarray(
            self.stats["action_std"], dtype=np.float32
        ).reshape(1, 1, -1)
        self.action_std = np.where(self.action_std < 1e-8, 1.0, self.action_std)
        if self.action_mean.shape[-1] != self.action_dim:
            raise ValueError(
                f"action dim mismatch: stats action_dim={self.action_mean.shape[-1]} expected={self.action_dim}"
            )
        self.action_min = None
        self.action_max = None
        if self.critic_obs_unnormalize_dp_chunk:
            if not os.path.exists(normalization_npz):
                raise FileNotFoundError(
                    f"normalization npz not found: {normalization_npz}"
                )
            normalization = np.load(normalization_npz)
            missing = [
                key for key in ("action_min", "action_max") if key not in normalization
            ]
            if missing:
                raise KeyError(f"normalization npz missing keys: {missing}")
            self.action_min = np.asarray(
                normalization["action_min"], dtype=np.float32
            ).reshape(1, 1, -1)
            self.action_max = np.asarray(
                normalization["action_max"], dtype=np.float32
            ).reshape(1, 1, -1)
            if (
                self.action_min.shape[-1] != self.action_dim
                or self.action_max.shape[-1] != self.action_dim
            ):
                raise ValueError(
                    "normalization action bounds dim mismatch: "
                    f"min={self.action_min.shape[-1]} max={self.action_max.shape[-1]} "
                    f"expected={self.action_dim}"
                )

    @property
    def flat_state_dim(self) -> int:
        return int(sum(int(self.lowdim_dims[k]) for k in self.lowdim_keys))

    def split_states(self, flat_states: np.ndarray) -> Dict[str, np.ndarray]:
        states = np.asarray(flat_states, dtype=np.float32)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        if states.ndim != 2 or states.shape[1] != self.flat_state_dim:
            raise ValueError(
                f"state shape mismatch: got {states.shape}, expected [N,{self.flat_state_dim}]"
            )
        out: Dict[str, np.ndarray] = {}
        cursor = 0
        for key in self.lowdim_keys:
            dim = int(self.lowdim_dims[key])
            out[key] = states[:, cursor : cursor + dim].astype(np.float32)
            cursor += dim
        return out

    def build_obs_matrix(self, flat_states: np.ndarray) -> np.ndarray:
        obs_batch = self.split_states(flat_states)
        n = next(iter(obs_batch.values())).shape[0] if obs_batch else 0
        parts: List[np.ndarray] = []
        use_stats = self.critic_obs_state_mode in {"stats_clip", "stats_no_clip"}
        for key in self.lowdim_keys:
            dim = int(self.lowdim_dims[key])
            values = np.asarray(obs_batch[key], dtype=np.float32).reshape(n, dim)
            if use_stats:
                mean = np.asarray(self.stats[f"{key}_mean"], dtype=np.float32).reshape(
                    1, dim
                )
                std = np.asarray(self.stats[f"{key}_std"], dtype=np.float32).reshape(
                    1, dim
                )
                std = np.where(std < 1e-8, 1.0, std)
                values = (values - mean) / std
            parts.append(values)
        obs = (
            np.concatenate(parts, axis=1).astype(np.float32)
            if parts
            else np.zeros((n, 0), dtype=np.float32)
        )
        if self.critic_obs_state_mode == "stats_clip":
            np.clip(obs, -10.0, 10.0, out=obs)
        return obs

    def normalize_actions_for_critic_obs(self, actions: np.ndarray) -> np.ndarray:
        critic_actions = np.asarray(actions, dtype=np.float32)
        if self.critic_obs_unnormalize_dp_chunk:
            critic_actions = (
                (critic_actions + 1.0) * 0.5 * (self.action_max - self.action_min)
                + self.action_min
            ).astype(np.float32)
        if self.critic_obs_action_mode == "dp_raw":
            return critic_actions.astype(np.float32)
        return ((critic_actions - self.action_mean) / self.action_std).astype(
            np.float32
        )

    def build_rl_obs(self, flat_states: np.ndarray, chunks: np.ndarray) -> np.ndarray:
        states = np.asarray(flat_states, dtype=np.float32)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        chunks = np.asarray(chunks, dtype=np.float32)
        if chunks.ndim == 2:
            chunks = chunks.reshape(1, chunks.shape[0], chunks.shape[1])
        if chunks.ndim != 3:
            raise ValueError(f"chunks must have shape [N,T,A], got {chunks.shape}")
        if chunks.shape[0] != states.shape[0]:
            raise ValueError(
                f"state/chunk batch mismatch: {states.shape[0]} vs {chunks.shape[0]}"
            )
        if chunks.shape[2] != self.action_dim:
            raise ValueError(
                f"chunk action dim mismatch: got {chunks.shape[2]}, expected {self.action_dim}"
            )
        if chunks.shape[1] > self.max_chunk_len:
            chunks = chunks[:, : self.max_chunk_len, :]

        encoded_chunks = self.normalize_actions_for_critic_obs(chunks)
        if encoded_chunks.shape[1] < self.max_chunk_len:
            pad = np.zeros(
                (
                    encoded_chunks.shape[0],
                    self.max_chunk_len - encoded_chunks.shape[1],
                    self.action_dim,
                ),
                dtype=np.float32,
            )
            encoded_chunks = np.concatenate([encoded_chunks, pad], axis=1)

        obs_part = self.build_obs_matrix(states)
        chunk_part = encoded_chunks.reshape(encoded_chunks.shape[0], -1).astype(
            np.float32
        )
        return np.concatenate([obs_part, chunk_part], axis=1).astype(np.float32)


def _load_demo_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"offline demo npz not found: {path}")
    data = np.load(path)
    required = ["states", "actions", "traj_lengths"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"offline demo npz missing keys: {missing}")
    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    traj_lengths = np.asarray(data["traj_lengths"], dtype=np.int64).reshape(-1)
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError(
            f"states/actions must be rank-2, got {states.shape} / {actions.shape}"
        )
    if states.shape[0] != actions.shape[0]:
        raise ValueError(
            f"states/actions length mismatch: {states.shape[0]} vs {actions.shape[0]}"
        )
    if int(traj_lengths.sum()) != int(states.shape[0]):
        raise ValueError(
            f"traj_lengths sum mismatch: sum={int(traj_lengths.sum())} states={int(states.shape[0])}"
        )
    return states, actions, traj_lengths


def _iter_full_window_starts(
    traj_lengths: np.ndarray,
    *,
    max_chunk_len: int,
    max_episodes: int,
) -> Iterable[Tuple[int, int, int, int]]:
    base = 0
    used_episodes = 0
    for ep_id, length_raw in enumerate(
        np.asarray(traj_lengths, dtype=np.int64).reshape(-1)
    ):
        length = int(length_raw)
        use_ep = max_episodes <= 0 or used_episodes < int(max_episodes)
        if use_ep:
            used_episodes += 1
            if length >= int(max_chunk_len):
                for offset in range(0, length - int(max_chunk_len) + 1):
                    yield int(ep_id), int(base), int(length), int(offset)
        base += length


def _count_used_episodes(traj_lengths: np.ndarray, max_episodes: int) -> int:
    if int(max_episodes) <= 0:
        return int(len(traj_lengths))
    return int(min(int(max_episodes), len(traj_lengths)))


def _discounted_success_reward(
    *,
    start_offset: int,
    requested_k: int,
    episode_len: int,
    gamma: float,
    reward_offset: float,
    reward_duration_mode: str,
) -> Tuple[float, int, bool]:
    final_action_offset = int(episode_len) - 1
    effective_duration = min(int(requested_k), int(episode_len) - int(start_offset))
    total = 0.0
    discount = 1.0
    for j in range(effective_duration):
        action_offset = int(start_offset) + int(j)
        raw_reward = 1.0 if action_offset >= final_action_offset else 0.0
        total += discount * (raw_reward - float(reward_offset))
        discount *= float(gamma)
    if str(reward_duration_mode) == "mean":
        total /= max(float(effective_duration), 1.0)
    elif str(reward_duration_mode) != "sum":
        raise ValueError(f"unknown reward_duration_mode: {reward_duration_mode}")
    done = (int(start_offset) + int(requested_k) - 1) >= final_action_offset
    return float(total), int(effective_duration), bool(done)


def _unwrap_policy_adapter(policy):
    return getattr(policy, "policy", policy)


def _configure_policy_sampler(
    policy_adapter, *, dp_sampler: str, dp_infer_steps: int
) -> None:
    cfg = getattr(policy_adapter, "algo_config", None)
    if cfg is None:
        return
    ctx = cfg.values_unlocked() if hasattr(cfg, "values_unlocked") else None
    if ctx is None:
        _apply_policy_sampler_config(
            cfg, dp_sampler=dp_sampler, dp_infer_steps=dp_infer_steps
        )
    else:
        with ctx:
            _apply_policy_sampler_config(
                cfg, dp_sampler=dp_sampler, dp_infer_steps=dp_infer_steps
            )


def _apply_policy_sampler_config(cfg, *, dp_sampler: str, dp_infer_steps: int) -> None:
    steps = max(1, int(dp_infer_steps))
    if str(dp_sampler) == "ddim":
        if hasattr(cfg, "ddpm"):
            cfg.ddpm.enabled = False
        if hasattr(cfg, "ddim"):
            cfg.ddim.enabled = True
            cfg.ddim.num_inference_timesteps = steps
    elif str(dp_sampler) == "ddpm":
        if hasattr(cfg, "ddpm"):
            cfg.ddpm.enabled = True
            cfg.ddpm.num_inference_timesteps = steps
        if hasattr(cfg, "ddim"):
            cfg.ddim.enabled = False
    else:
        raise ValueError(f"unknown dp_sampler: {dp_sampler}")


def _sample_policy_chunks(
    policy_adapter,
    encoder: OfflineDemoRLObsEncoder,
    flat_states: np.ndarray,
    *,
    max_chunk_len: int,
) -> np.ndarray:
    obs_batch_np = encoder.split_states(flat_states)
    obs_batch = {
        k: torch.as_tensor(v, dtype=torch.float32) for k, v in obs_batch_np.items()
    }
    if hasattr(policy_adapter, "set_eval"):
        policy_adapter.set_eval()
    with torch.no_grad():
        chunks = policy_adapter._get_action_trajectory(obs_batch)
    if torch.is_tensor(chunks):
        chunks = chunks.detach().cpu().numpy()
    chunks = np.asarray(chunks, dtype=np.float32)
    if chunks.ndim != 3:
        raise ValueError(f"policy chunks must have shape [N,T,A], got {chunks.shape}")
    if chunks.shape[1] > int(max_chunk_len):
        chunks = chunks[:, : int(max_chunk_len), :]
    return chunks.astype(np.float32)


def _collect_target_state_indices(
    traj_lengths: np.ndarray,
    *,
    max_chunk_len: int,
    max_episodes: int,
    max_transitions: int,
) -> Tuple[List[int], int, int, bool]:
    target_indices = set()
    transitions = 0
    full_starts = 0
    max_hit = False
    for _, ep_start, ep_len, offset in _iter_full_window_starts(
        traj_lengths,
        max_chunk_len=max_chunk_len,
        max_episodes=max_episodes,
    ):
        full_starts += 1
        for requested_k in range(1, int(max_chunk_len) + 1):
            if int(max_transitions) > 0 and transitions >= int(max_transitions):
                max_hit = True
                return (
                    sorted(target_indices),
                    int(full_starts),
                    int(transitions),
                    bool(max_hit),
                )
            terminal = (int(offset) + requested_k - 1) >= (int(ep_len) - 1)
            if not terminal:
                target_indices.add(int(ep_start) + int(offset) + requested_k)
            transitions += 1
    return sorted(target_indices), int(full_starts), int(transitions), bool(max_hit)


def _build_target_cache(
    *,
    policy,
    encoder: OfflineDemoRLObsEncoder,
    states: np.ndarray,
    target_indices: Sequence[int],
    target_next_obs_samples: int,
    policy_batch_size: int,
    dp_sampler: str,
    dp_infer_steps: int,
    max_chunk_len: int,
) -> Dict[int, np.ndarray]:
    sample_count = max(1, int(target_next_obs_samples))
    if len(target_indices) == 0:
        return {}
    policy_adapter = _unwrap_policy_adapter(policy)
    if policy_adapter is None or not hasattr(policy_adapter, "_get_action_trajectory"):
        raise ValueError(
            "offline demo target sampling requires a policy with _get_action_trajectory"
        )
    _configure_policy_sampler(
        policy_adapter, dp_sampler=dp_sampler, dp_infer_steps=dp_infer_steps
    )

    cache: Dict[int, np.ndarray] = {}
    unique_batch = max(1, int(policy_batch_size) // sample_count)
    for start in range(0, len(target_indices), unique_batch):
        batch_indices = list(target_indices[start : start + unique_batch])
        batch_states = np.asarray(
            states[np.asarray(batch_indices, dtype=np.int64)], dtype=np.float32
        )
        repeated_states = np.repeat(batch_states, sample_count, axis=0)
        chunks = _sample_policy_chunks(
            policy_adapter,
            encoder,
            repeated_states,
            max_chunk_len=max_chunk_len,
        )
        rl_obs = encoder.build_rl_obs(repeated_states, chunks)
        rl_obs = rl_obs.reshape(len(batch_indices), sample_count, -1)
        for local_i, state_idx in enumerate(batch_indices):
            cache[int(state_idx)] = rl_obs[local_i].astype(np.float32)
        print(
            f"[OFFLINE-DEMO] sampled target chunks {min(start + unique_batch, len(target_indices))}/{len(target_indices)}",
            flush=True,
        )
    return cache


def populate_replay_from_demo_npz(
    *,
    replay,
    policy,
    demo_npz: str,
    stats_npz: str,
    normalization_npz: str,
    lowdim_keys: Sequence[str],
    lowdim_dims: Dict[str, int],
    action_dim: int,
    max_chunk_len: int,
    min_chunk_len: int,
    gamma: float,
    reward_offset: float,
    reward_duration_mode: str,
    target_next_obs_samples: int,
    policy_batch_size: int,
    dp_sampler: str,
    dp_infer_steps: int,
    action_history_len: int = 0,
    critic_obs_state_mode: str,
    critic_obs_action_mode: str,
    critic_obs_unnormalize_dp_chunk: bool,
    max_episodes: int = 0,
    max_transitions: int = 0,
    window_mode: str = "full_only",
    progress_every: int = 50000,
) -> OfflineDemoReplaySummary:
    if str(window_mode) != "full_only":
        raise ValueError(
            "offline demo replay currently supports only window_mode=full_only"
        )
    if int(min_chunk_len) != 1:
        raise ValueError("offline demo replay currently requires min_chunk_len=1")
    if int(max_chunk_len) <= 0:
        raise ValueError("max_chunk_len must be positive")
    if int(target_next_obs_samples) < 1:
        raise ValueError("target_next_obs_samples must be >= 1")

    states, actions, traj_lengths = _load_demo_npz(demo_npz)
    encoder = OfflineDemoRLObsEncoder(
        stats_npz=stats_npz,
        normalization_npz=normalization_npz,
        lowdim_keys=lowdim_keys,
        lowdim_dims=lowdim_dims,
        action_dim=action_dim,
        max_chunk_len=max_chunk_len,
        action_history_len=action_history_len,
        critic_obs_state_mode=critic_obs_state_mode,
        critic_obs_action_mode=critic_obs_action_mode,
        critic_obs_unnormalize_dp_chunk=critic_obs_unnormalize_dp_chunk,
    )
    print(
        "[OFFLINE-DEMO] critic obs encoder "
        f"state={encoder.critic_obs_state_mode} "
        f"action={encoder.critic_obs_action_mode} "
        f"unnormalize_dp_chunk={encoder.critic_obs_unnormalize_dp_chunk}",
        flush=True,
    )
    if states.shape[1] != encoder.flat_state_dim:
        raise ValueError(
            f"demo state dim={states.shape[1]} but stats/lowdim dims imply {encoder.flat_state_dim}"
        )
    if actions.shape[1] != int(action_dim):
        raise ValueError(
            f"demo action dim={actions.shape[1]} expected={int(action_dim)}"
        )

    target_indices, full_starts, transition_plan_count, max_hit = (
        _collect_target_state_indices(
            traj_lengths,
            max_chunk_len=max_chunk_len,
            max_episodes=max_episodes,
            max_transitions=max_transitions,
        )
    )
    if transition_plan_count <= 0:
        raise ValueError("offline demo replay produced zero transitions")
    if int(getattr(replay, "size", len(replay))) + int(transition_plan_count) > int(
        replay.capacity
    ):
        raise ValueError(
            f"demo transitions would exceed replay capacity: current={len(replay)} "
            f"planned={transition_plan_count} capacity={replay.capacity}"
        )

    print(
        f"[OFFLINE-DEMO] loading demo={demo_npz} "
        f"states={states.shape} actions={actions.shape} episodes={len(traj_lengths)}",
        flush=True,
    )
    print(
        f"[OFFLINE-DEMO] full_window_starts={full_starts} planned_transitions={transition_plan_count} "
        f"target_states={len(target_indices)} target_samples={int(target_next_obs_samples)}",
        flush=True,
    )

    target_cache = _build_target_cache(
        policy=policy,
        encoder=encoder,
        states=states,
        target_indices=target_indices,
        target_next_obs_samples=target_next_obs_samples,
        policy_batch_size=policy_batch_size,
        dp_sampler=dp_sampler,
        dp_infer_steps=dp_infer_steps,
        max_chunk_len=max_chunk_len,
    )

    transitions = 0
    terminal_count = 0
    nonterminal_count = 0
    duration_sum = 0.0
    reward_sum = 0.0
    reward_min = float("inf")
    reward_max = float("-inf")
    used_episodes = _count_used_episodes(traj_lengths, int(max_episodes))
    skipped_short = int(
        sum(
            1
            for x in np.asarray(traj_lengths[:used_episodes], dtype=np.int64)
            if int(x) < int(max_chunk_len)
        )
    )
    obs_cache: Dict[int, np.ndarray] = {}
    chunk_cache: Dict[int, np.ndarray] = {}

    for ep_id, ep_start, ep_len, offset in _iter_full_window_starts(
        traj_lengths,
        max_chunk_len=max_chunk_len,
        max_episodes=max_episodes,
    ):
        start_idx = int(ep_start) + int(offset)
        if start_idx not in obs_cache:
            current_chunk = np.asarray(
                actions[start_idx : start_idx + int(max_chunk_len)], dtype=np.float32
            )
            chunk_cache[start_idx] = current_chunk
            obs_cache[start_idx] = encoder.build_rl_obs(
                states[start_idx], current_chunk
            )[0]
        current_obs = obs_cache[start_idx]

        for requested_k in range(1, int(max_chunk_len) + 1):
            if int(max_transitions) > 0 and transitions >= int(max_transitions):
                max_hit = True
                break
            reward, duration, done = _discounted_success_reward(
                start_offset=int(offset),
                requested_k=int(requested_k),
                episode_len=int(ep_len),
                gamma=float(gamma),
                reward_offset=float(reward_offset),
                reward_duration_mode=str(reward_duration_mode),
            )
            if done:
                next_state_idx = int(ep_start) + int(ep_len) - 1
                next_chunk = chunk_cache[start_idx]
                next_obs = encoder.build_rl_obs(states[next_state_idx], next_chunk)[0]
                next_samples = None
                terminal_count += 1
            else:
                next_state_idx = int(ep_start) + int(offset) + int(requested_k)
                next_samples = target_cache[int(next_state_idx)]
                next_obs = next_samples[0]
                nonterminal_count += 1

            replay.add(
                current_obs,
                int(requested_k) - int(min_chunk_len),
                float(reward),
                float(duration),
                next_obs,
                bool(done),
                action_dim=int(max_chunk_len) - int(min_chunk_len) + 1,
                episode_done=bool(done),
                env_id=int(ep_id),
                next_obs_samples=next_samples,
            )
            transitions += 1
            duration_sum += float(duration)
            reward_sum += float(reward)
            reward_min = min(reward_min, float(reward))
            reward_max = max(reward_max, float(reward))
            if int(progress_every) > 0 and transitions % int(progress_every) == 0:
                print(
                    f"[OFFLINE-DEMO] added {transitions}/{transition_plan_count} transitions "
                    f"replay={len(replay)}",
                    flush=True,
                )
        if int(max_transitions) > 0 and transitions >= int(max_transitions):
            break

    if transitions <= 0:
        reward_min = 0.0
        reward_max = 0.0

    summary = OfflineDemoReplaySummary(
        dataset_path=str(demo_npz),
        window_mode=str(window_mode),
        critic_obs_state_mode=str(encoder.critic_obs_state_mode),
        critic_obs_action_mode=str(encoder.critic_obs_action_mode),
        critic_obs_unnormalize_dp_chunk=bool(encoder.critic_obs_unnormalize_dp_chunk),
        episodes_available=int(len(traj_lengths)),
        episodes_used=int(used_episodes),
        state_count=int(states.shape[0]),
        action_count=int(actions.shape[0]),
        max_chunk_len=int(max_chunk_len),
        min_chunk_len=int(min_chunk_len),
        full_window_starts=int(full_starts),
        transitions_added=int(transitions),
        terminal_count=int(terminal_count),
        nonterminal_count=int(nonterminal_count),
        skipped_short_episodes=int(skipped_short),
        target_next_obs_samples=int(target_next_obs_samples),
        cached_target_states=int(len(target_cache)),
        duration_sum=float(duration_sum),
        reward_sum=float(reward_sum),
        reward_min=float(reward_min),
        reward_max=float(reward_max),
        max_transitions_hit=bool(max_hit),
    )
    print(f"[OFFLINE-DEMO] summary={asdict(summary)}", flush=True)
    return summary
