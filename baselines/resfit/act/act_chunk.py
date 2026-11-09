"""Fixed-TA chunk-level ACT adapter for state-based ResFiT.

The frozen ACT policy plans one ``[TA, action_dim]`` base chunk at each
macro decision.  The residual actor sees that complete chunk, predicts one
residual chunk, and the executor consumes the command with ``exec_len=TA``.
Success and time-limit boundaries may shorten the *actual* duration, but never
cause an intermediate ACT replan.

Like the standard ACT ExRL-Q compatibility wrapper, policy outputs in the
DSRL-normalized action space are clipped to ``[-1, 1]`` before execution.

Historical ``*_dp`` field names denote the shared DSRL-normalized coordinate
system used by both standard DP and ACT runs; they do not select a DP policy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from exrl.runners.dp_q.vector_env import (
    RobomimicExecutor,
    _extract_state_from_obs,
    split_state_to_obs_dict,
)
from baselines.resfit.act.td3_core import (
    AggregatedTransition,
    PrimitiveReplayBuffer,
    ResFiTActionScaler,
    StatePreprocessor,
)


class ACTChunkPlanner:
    """Sample complete fixed-horizon chunks from the frozen ACT policy."""

    def __init__(
        self,
        policy,
        lowdim_keys: Sequence[str],
        lowdim_dims: Dict[str, int],
        horizon_steps: int,
        action_dim: int,
    ):
        self.policy = policy
        self.lowdim_keys = list(lowdim_keys)
        self.lowdim_dims = {key: int(value) for key, value in lowdim_dims.items()}
        self.horizon_steps = int(horizon_steps)
        self.action_dim = int(action_dim)
        if self.horizon_steps <= 0 or self.action_dim <= 0:
            raise ValueError("planner dimensions must be positive")
        self.sample_count = 0
        self.bootstrap_sample_count = 0

    @torch.no_grad()
    def plan(self, state_dp: np.ndarray, *, bootstrap: bool = False) -> np.ndarray:
        state = np.asarray(state_dp, dtype=np.float32)
        if state.ndim != 2 or state.shape[0] <= 0:
            raise ValueError("state_dp must be a non-empty [batch, state_dim] matrix")
        obs_dict = split_state_to_obs_dict(state, self.lowdim_keys, self.lowdim_dims)
        actions = self.policy._get_action_trajectory(obs_dict)
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        chunks = np.asarray(actions, dtype=np.float32)
        expected = (state.shape[0], self.horizon_steps, self.action_dim)
        if chunks.shape != expected:
            raise RuntimeError(
                "ACT chunk shape mismatch: expected %s, got %s" % (expected, chunks.shape)
            )
        if not np.all(np.isfinite(chunks)):
            raise RuntimeError("ACT produced NaN/Inf action chunks")
        chunks = np.clip(chunks, -1.0, 1.0).astype(np.float32, copy=False)
        if bootstrap:
            self.bootstrap_sample_count += int(state.shape[0])
        else:
            self.sample_count += int(state.shape[0])
        return chunks.copy()


@dataclass
class ChunkResidualObservation:
    state_dp: np.ndarray
    state_actor: np.ndarray
    base_action_dp: np.ndarray
    base_action_resfit: np.ndarray


@dataclass
class ChunkResidualStepBatch:
    observation: ChunkResidualObservation
    replay_next_state: np.ndarray
    replay_next_base_action: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    dones: np.ndarray
    infos: List[Dict[str, object]]
    exec_lens: np.ndarray
    base_action_resfit: np.ndarray
    residual_resfit: np.ndarray
    preclamp_action_resfit: np.ndarray
    full_action_resfit: np.ndarray
    replay_action_resfit: np.ndarray
    action_raw: np.ndarray
    action_dp: np.ndarray
    queue_resamples: int


@dataclass
class SMDPTransition:
    """One macro decision with an already discounted within-chunk reward."""

    state: np.ndarray
    base_action: np.ndarray
    action: np.ndarray
    reward: float
    discount: float
    next_state: np.ndarray
    next_base_action: np.ndarray
    terminated: bool
    truncated: bool

    @property
    def boundary(self) -> bool:
        return bool(self.terminated or self.truncated)


class SMDPNStepAccumulator:
    """Duration-aware optional n-step aggregation over macro decisions."""

    def __init__(self, n_step: int, n_envs: int):
        if int(n_step) <= 0 or int(n_envs) <= 0:
            raise ValueError("n_step and n_envs must be positive")
        self.n_step = int(n_step)
        self.queues: List[Deque[SMDPTransition]] = [deque() for _ in range(int(n_envs))]

    def _aggregate_front(self, queue_: Deque[SMDPTransition]) -> AggregatedTransition:
        if not queue_:
            raise RuntimeError("cannot aggregate an empty SMDP queue")
        items = list(queue_)[: self.n_step]
        reward = 0.0
        cumulative_discount = 1.0
        final = items[0]
        for transition in items:
            reward += cumulative_discount * float(transition.reward)
            cumulative_discount *= float(transition.discount)
            final = transition
            if transition.boundary:
                break
        first = items[0]
        return AggregatedTransition(
            state=np.asarray(first.state, dtype=np.float32),
            base_action=np.asarray(first.base_action, dtype=np.float32),
            action=np.asarray(first.action, dtype=np.float32),
            reward=float(reward),
            discount=float(cumulative_discount),
            next_state=np.asarray(final.next_state, dtype=np.float32),
            next_base_action=np.asarray(final.next_base_action, dtype=np.float32),
            terminated=bool(final.terminated),
            truncated=bool(final.truncated),
        )

    def append(self, env_index: int, transition: SMDPTransition) -> List[AggregatedTransition]:
        queue_ = self.queues[int(env_index)]
        queue_.append(transition)
        emitted: List[AggregatedTransition] = []
        if transition.boundary:
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


class ChunkResidualVecEnv:
    """Execute one complete residual-corrected TA chunk per RL decision."""

    def __init__(
        self,
        executor: RobomimicExecutor,
        planner: ACTChunkPlanner,
        action_scaler: ResFiTActionScaler,
        state_preprocessor: StatePreprocessor,
        gamma: float,
        reward_offset: float = 1.0,
        replay_action_mode: str = "executed_clamped",
        bootstrap_timeouts: bool = True,
    ):
        self.executor = executor
        self.planner = planner
        # Keep the telemetry attribute shared with the primitive adapter.
        self.queue = planner
        self.action_scaler = action_scaler
        self.state_preprocessor = state_preprocessor
        self.gamma = float(gamma)
        self.reward_offset = float(reward_offset)
        self.replay_action_mode = str(replay_action_mode)
        self.bootstrap_timeouts = bool(bootstrap_timeouts)
        self._observation: Optional[ChunkResidualObservation] = None

    @property
    def n_envs(self) -> int:
        return int(self.executor.n_envs)

    @property
    def horizon_steps(self) -> int:
        return int(self.planner.horizon_steps)

    @property
    def flat_action_dim(self) -> int:
        return int(self.horizon_steps * self.action_scaler.action_dim)

    def _build_observation(
        self, state_dp: np.ndarray, base_dp: np.ndarray
    ) -> ChunkResidualObservation:
        return ChunkResidualObservation(
            state_dp=np.asarray(state_dp, dtype=np.float32).copy(),
            state_actor=self.state_preprocessor.transform(state_dp),
            base_action_dp=np.asarray(base_dp, dtype=np.float32).copy(),
            base_action_resfit=np.asarray(
                self.action_scaler.dp_to_resfit(base_dp), dtype=np.float32
            ),
        )

    def reset(self, base_seed: Optional[int] = None) -> ChunkResidualObservation:
        state_dp = self.executor.reset_all(base_seed=base_seed)
        base_dp = self.planner.plan(state_dp)
        self._observation = self._build_observation(state_dp, base_dp)
        return self._observation

    @staticmethod
    def _time_limit_truncated(info: Dict[str, object]) -> bool:
        value = info.get("TimeLimit.truncated", False)
        if isinstance(value, (list, tuple, np.ndarray)):
            return bool(value[-1]) if len(value) else False
        return bool(value)

    @staticmethod
    def _terminal_state_dp(info: Dict[str, object], expected_dim: int) -> np.ndarray:
        terminal_observation = info.get("terminal_observation")
        if terminal_observation is None:
            raise RuntimeError(
                "timeout is missing terminal_observation; refusing reset-state bootstrap"
            )
        if isinstance(terminal_observation, dict):
            state = _extract_state_from_obs(terminal_observation)[0]
        else:
            state = np.asarray(terminal_observation, dtype=np.float32).reshape(-1)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (int(expected_dim),) or not np.all(np.isfinite(state)):
            raise RuntimeError(
                "invalid timeout terminal state: expected (%d,), got %s"
                % (int(expected_dim), state.shape)
            )
        return state

    def step(self, residual_resfit: np.ndarray) -> ChunkResidualStepBatch:
        if self._observation is None:
            raise RuntimeError("environment must be reset before step")
        residual = np.asarray(residual_resfit, dtype=np.float32)
        if residual.shape == (self.n_envs, self.flat_action_dim):
            residual = residual.reshape(
                self.n_envs, self.horizon_steps, self.action_scaler.action_dim
            )
        expected = (
            self.n_envs,
            self.horizon_steps,
            self.action_scaler.action_dim,
        )
        if residual.shape != expected:
            raise ValueError("residual shape must be %s, got %s" % (expected, residual.shape))
        if not np.all(np.isfinite(residual)):
            raise ValueError("residual contains NaN/Inf")

        current = self._observation
        combined = self.action_scaler.combine(
            current.base_action_dp,
            residual,
            replay_action_mode=self.replay_action_mode,
        )
        action_dp = np.asarray(combined["action_dp"], dtype=np.float32)
        requested = np.full((self.n_envs,), self.horizon_steps, dtype=np.int64)
        before_samples = int(self.planner.sample_count)
        result = self.executor.step_variable_k(
            action_dp,
            requested,
            gamma=self.gamma,
            reward_offset=self.reward_offset,
            discount_within_chunk=True,
        )
        raw_exec_lens = np.asarray(result.exec_lens).reshape(-1)
        exec_lens = raw_exec_lens.astype(np.int64)
        if (
            exec_lens.shape != (self.n_envs,)
            or np.any(raw_exec_lens != exec_lens)
            or np.any(exec_lens <= 0)
            or np.any(exec_lens > self.horizon_steps)
        ):
            raise RuntimeError("fixed-TA decision returned invalid durations: %s" % exec_lens)
        dones = np.asarray(result.dones, dtype=bool).reshape(-1)
        rewards = np.asarray(result.rewards, dtype=np.float32).reshape(-1)
        if (
            dones.shape != (self.n_envs,)
            or rewards.shape != (self.n_envs,)
            or len(result.infos) != self.n_envs
        ):
            raise RuntimeError("executor returned a malformed fixed-TA batch")
        if np.any((~dones) & (exec_lens != self.horizon_steps)):
            raise RuntimeError(
                "a live fixed-TA worker executed fewer than %d actions: %s"
                % (self.horizon_steps, exec_lens)
            )
        truncated = (
            np.asarray([self._time_limit_truncated(info) for info in result.infos], dtype=bool)
            & dones
        )
        terminated = dones & (~truncated)

        # The executor auto-resets completed workers.  Plan the next live chunk
        # from those reset states (or from the post-TA state for live episodes).
        next_state_dp = np.asarray(result.next_state, dtype=np.float32)
        next_base_dp = self.planner.plan(next_state_dp)
        next_observation = self._build_observation(next_state_dp, next_base_dp)

        replay_next_state = next_observation.state_actor.copy()
        replay_next_base = next_observation.base_action_resfit.copy()
        replay_next_state[terminated] = 0.0
        replay_next_base[terminated] = 0.0
        if not self.bootstrap_timeouts:
            replay_next_state[truncated] = 0.0
            replay_next_base[truncated] = 0.0
        else:
            timeout_indices = np.flatnonzero(truncated).tolist()
            if timeout_indices:
                terminal_states = np.stack(
                    [
                        self._terminal_state_dp(result.infos[index], next_state_dp.shape[1])
                        for index in timeout_indices
                    ],
                    axis=0,
                )
                timeout_base_dp = self.planner.plan(terminal_states, bootstrap=True)
                replay_next_state[timeout_indices] = self.state_preprocessor.transform(
                    terminal_states
                )
                replay_next_base[timeout_indices] = self.action_scaler.dp_to_resfit(timeout_base_dp)

        self._observation = next_observation
        return ChunkResidualStepBatch(
            observation=next_observation,
            replay_next_state=replay_next_state,
            replay_next_base_action=replay_next_base,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            dones=dones,
            infos=[dict(info) for info in result.infos],
            exec_lens=exec_lens,
            base_action_resfit=np.asarray(combined["base_resfit"], dtype=np.float32),
            residual_resfit=residual.copy(),
            preclamp_action_resfit=np.asarray(combined["preclamp_resfit"], dtype=np.float32),
            full_action_resfit=np.asarray(combined["full_resfit"], dtype=np.float32),
            replay_action_resfit=np.asarray(combined["replay_resfit"], dtype=np.float32),
            action_raw=np.asarray(combined["action_raw"], dtype=np.float32),
            action_dp=action_dp,
            queue_resamples=int(self.planner.sample_count) - before_samples,
        )

    def close(self) -> None:
        self.executor.close()


def add_chunk_step_to_replay(
    previous: ChunkResidualObservation,
    step: ChunkResidualStepBatch,
    accumulator: SMDPNStepAccumulator,
    replay: PrimitiveReplayBuffer,
    gamma: float,
) -> int:
    if not np.allclose(
        np.asarray(step.base_action_resfit),
        np.asarray(previous.base_action_resfit),
        rtol=0.0,
        atol=2e-7,
    ):
        raise RuntimeError("replay base chunk drifted from the chunk used for execution")
    if not np.all(np.isfinite(step.replay_action_resfit)):
        raise RuntimeError("replay action chunk contains NaN/Inf")
    emitted = 0
    for env_index in range(previous.state_actor.shape[0]):
        duration = int(step.exec_lens[env_index])
        discount = 0.0 if bool(step.terminated[env_index]) else float(gamma) ** duration
        transition = SMDPTransition(
            state=previous.state_actor[env_index],
            base_action=np.asarray(step.base_action_resfit[env_index]).reshape(-1),
            action=np.asarray(step.replay_action_resfit[env_index]).reshape(-1),
            reward=float(step.rewards[env_index]),
            discount=float(discount),
            next_state=step.replay_next_state[env_index],
            next_base_action=np.asarray(step.replay_next_base_action[env_index]).reshape(-1),
            terminated=bool(step.terminated[env_index]),
            truncated=bool(step.truncated[env_index]),
        )
        emitted += replay.extend(accumulator.append(env_index, transition))
    return emitted
