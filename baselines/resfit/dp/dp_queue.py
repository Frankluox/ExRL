"""Primitive-step DP queue and residual-action environment adapter.

The frozen diffusion policy still plans a TA-step chunk, but ResFiT consumes
exactly one queued base action and predicts one physical-action residual per environment
step.  This module is deliberately independent from ExRL-Q's variable-k logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from exrl.runners.dp_q.vector_env import (
    RobomimicExecutor,
    _extract_state_from_obs,
    split_state_to_obs_dict,
)
from baselines.resfit.dp.td3_core import ResFiTActionScaler, StatePreprocessor


class DPChunkQueue:
    """Maintain one independent frozen-DP action queue per vector environment."""

    def __init__(
        self,
        policy,
        lowdim_keys: Sequence[str],
        lowdim_dims: Dict[str, int],
        n_envs: int,
        horizon_steps: int,
        action_dim: int,
    ):
        self.policy = policy
        self.lowdim_keys = list(lowdim_keys)
        self.lowdim_dims = {key: int(value) for key, value in lowdim_dims.items()}
        self.n_envs = int(n_envs)
        self.horizon_steps = int(horizon_steps)
        self.action_dim = int(action_dim)
        if self.n_envs <= 0 or self.horizon_steps <= 0 or self.action_dim <= 0:
            raise ValueError("queue dimensions must be positive")
        self.chunks = np.zeros((self.n_envs, self.horizon_steps, self.action_dim), dtype=np.float32)
        self.cursors = np.zeros((self.n_envs,), dtype=np.int64)
        self.valid = np.zeros((self.n_envs,), dtype=bool)
        self.resample_count = 0
        self.bootstrap_resample_count = 0

    def reset(self, env_indices: Optional[Sequence[int]] = None) -> None:
        indices = range(self.n_envs) if env_indices is None else [int(x) for x in env_indices]
        for index in indices:
            self.valid[index] = False
            self.cursors[index] = 0
            self.chunks[index].fill(0.0)

    @torch.no_grad()
    def _sample(self, state_dp: np.ndarray, indices: Sequence[int]) -> None:
        ids = [int(x) for x in indices]
        if not ids:
            return
        state = np.asarray(state_dp, dtype=np.float32)
        if state.ndim != 2 or state.shape[0] != self.n_envs:
            raise ValueError("state_dp must be [n_envs, state_dim]")
        subset = state[ids]
        obs_dict = split_state_to_obs_dict(subset, self.lowdim_keys, self.lowdim_dims)
        actions = self.policy._get_action_trajectory(obs_dict)
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        chunks = np.asarray(actions, dtype=np.float32)
        expected = (len(ids), self.horizon_steps, self.action_dim)
        if chunks.shape != expected:
            raise RuntimeError(
                "DP chunk shape mismatch: expected %s, got %s" % (expected, chunks.shape)
            )
        if not np.all(np.isfinite(chunks)):
            raise RuntimeError("DP produced NaN/Inf action chunks")
        for row, env_index in enumerate(ids):
            self.chunks[env_index] = chunks[row]
            self.cursors[env_index] = 0
            self.valid[env_index] = True
        self.resample_count += len(ids)

    def current(self, state_dp: np.ndarray) -> np.ndarray:
        missing = np.flatnonzero(~self.valid).tolist()
        self._sample(state_dp, missing)
        rows = np.arange(self.n_envs, dtype=np.int64)
        return self.chunks[rows, self.cursors].copy()

    def advance(self, next_state_dp: np.ndarray, dones: np.ndarray) -> np.ndarray:
        done = np.asarray(dones, dtype=bool).reshape(-1)
        if done.shape[0] != self.n_envs:
            raise ValueError("dones size mismatch")
        for index in range(self.n_envs):
            if done[index]:
                self.valid[index] = False
                self.cursors[index] = 0
                continue
            self.cursors[index] += 1
            if self.cursors[index] >= self.horizon_steps:
                self.valid[index] = False
                self.cursors[index] = 0
        return self.current(next_state_dp)

    @torch.no_grad()
    def bootstrap_after_current(
        self,
        terminal_state_dp: np.ndarray,
        env_indices: Sequence[int],
    ) -> np.ndarray:
        """Return the base action after the current primitive for timeouts.

        This is a read-only view of the execution queue. If the current action
        is the final queued item, a new fixed-TA chunk is sampled from the terminal state only for
        the bootstrap target; the live queue is still reset for the new episode.
        """
        ids = [int(value) for value in env_indices]
        states = np.asarray(terminal_state_dp, dtype=np.float32)
        if states.ndim != 2 or states.shape[0] != self.n_envs:
            raise ValueError("terminal_state_dp must be [n_envs, state_dim]")
        output = np.empty((len(ids), self.action_dim), dtype=np.float32)
        rollover_rows = []
        rollover_ids = []
        for row, env_index in enumerate(ids):
            if not self.valid[env_index]:
                raise RuntimeError("cannot bootstrap from an invalid DP queue")
            next_cursor = int(self.cursors[env_index]) + 1
            if next_cursor < self.horizon_steps:
                output[row] = self.chunks[env_index, next_cursor]
            else:
                rollover_rows.append(row)
                rollover_ids.append(env_index)
        if rollover_ids:
            obs_dict = split_state_to_obs_dict(
                states[rollover_ids], self.lowdim_keys, self.lowdim_dims
            )
            actions = self.policy._get_action_trajectory(obs_dict)
            if torch.is_tensor(actions):
                actions = actions.detach().cpu().numpy()
            chunks = np.asarray(actions, dtype=np.float32)
            expected = (len(rollover_ids), self.horizon_steps, self.action_dim)
            if chunks.shape != expected or not np.all(np.isfinite(chunks)):
                raise RuntimeError(
                    "timeout bootstrap DP chunk mismatch: expected %s, got %s"
                    % (expected, chunks.shape)
                )
            for row, chunk in zip(rollover_rows, chunks):
                output[row] = chunk[0]
            self.bootstrap_resample_count += len(rollover_ids)
        return output

    def state_dict(self) -> Dict[str, object]:
        return {
            "n_envs": self.n_envs,
            "horizon_steps": self.horizon_steps,
            "action_dim": self.action_dim,
            "chunks": self.chunks.copy(),
            "cursors": self.cursors.copy(),
            "valid": self.valid.copy(),
            "resample_count": int(self.resample_count),
            "bootstrap_resample_count": int(self.bootstrap_resample_count),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if (
            int(state["n_envs"]) != self.n_envs
            or int(state["horizon_steps"]) != self.horizon_steps
            or int(state["action_dim"]) != self.action_dim
        ):
            raise ValueError("DP queue shape mismatch")
        self.chunks[...] = state["chunks"]
        self.cursors[...] = state["cursors"]
        self.valid[...] = state["valid"]
        self.resample_count = int(state["resample_count"])
        self.bootstrap_resample_count = int(state.get("bootstrap_resample_count", 0))


@dataclass
class ResidualObservation:
    state_dp: np.ndarray
    state_actor: np.ndarray
    base_action_dp: np.ndarray
    base_action_resfit: np.ndarray


@dataclass
class ResidualStepBatch:
    observation: ResidualObservation
    replay_next_state: np.ndarray
    replay_next_base_action: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    dones: np.ndarray
    infos: List[Dict[str, object]]
    base_action_resfit: np.ndarray
    residual_resfit: np.ndarray
    preclamp_action_resfit: np.ndarray
    full_action_resfit: np.ndarray
    replay_action_resfit: np.ndarray
    action_raw: np.ndarray
    action_dp: np.ndarray
    queue_resamples: int


class PrimitiveResidualVecEnv:
    """Compose the official DSRL executor with a fixed-chunk base queue."""

    def __init__(
        self,
        executor: RobomimicExecutor,
        queue: DPChunkQueue,
        action_scaler: ResFiTActionScaler,
        state_preprocessor: StatePreprocessor,
        reward_offset: float = 1.0,
        replay_action_mode: str = "executed_clamped",
        bootstrap_timeouts: bool = True,
    ):
        if executor.n_envs != queue.n_envs:
            raise ValueError("executor/queue n_envs mismatch")
        self.executor = executor
        self.queue = queue
        self.action_scaler = action_scaler
        self.state_preprocessor = state_preprocessor
        self.reward_offset = float(reward_offset)
        self.replay_action_mode = str(replay_action_mode)
        self.bootstrap_timeouts = bool(bootstrap_timeouts)
        self._observation: Optional[ResidualObservation] = None

    @property
    def n_envs(self) -> int:
        return int(self.executor.n_envs)

    def _build_observation(self, state_dp: np.ndarray, base_dp: np.ndarray) -> ResidualObservation:
        return ResidualObservation(
            state_dp=np.asarray(state_dp, dtype=np.float32).copy(),
            state_actor=self.state_preprocessor.transform(state_dp),
            base_action_dp=np.asarray(base_dp, dtype=np.float32).copy(),
            base_action_resfit=np.asarray(
                self.action_scaler.dp_to_resfit(base_dp), dtype=np.float32
            ),
        )

    def reset(self, base_seed: Optional[int] = None) -> ResidualObservation:
        state_dp = self.executor.reset_all(base_seed=base_seed)
        self.queue.reset()
        base_dp = self.queue.current(state_dp)
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

    def step(self, residual_resfit: np.ndarray) -> ResidualStepBatch:
        if self._observation is None:
            raise RuntimeError("environment must be reset before step")
        residual = np.asarray(residual_resfit, dtype=np.float32)
        expected = (self.n_envs, self.action_scaler.action_dim)
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
        action_chunks = action_dp[:, None, :]
        exec_lens = np.ones((self.n_envs,), dtype=np.int64)
        before_resamples = int(self.queue.resample_count)
        result = self.executor.step_variable_k(action_chunks, exec_lens)
        dones = np.asarray(result.dones, dtype=bool)
        truncated = (
            np.asarray([self._time_limit_truncated(info) for info in result.infos], dtype=bool)
            & dones
        )
        terminated = dones & (~truncated)
        rewards = np.asarray(result.rewards, dtype=np.float32) - self.reward_offset

        next_state_dp = np.asarray(result.next_state, dtype=np.float32)
        terminal_state_dp = next_state_dp.copy()
        timeout_indices = np.flatnonzero(truncated).tolist()
        bootstrap_timeout_indices = timeout_indices if self.bootstrap_timeouts else []
        for env_index in bootstrap_timeout_indices:
            terminal_state_dp[env_index] = self._terminal_state_dp(
                result.infos[env_index], next_state_dp.shape[1]
            )
        timeout_next_base_dp = self.queue.bootstrap_after_current(
            terminal_state_dp, bootstrap_timeout_indices
        )
        next_base_dp = self.queue.advance(next_state_dp, dones)
        next_observation = self._build_observation(next_state_dp, next_base_dp)

        # The executor auto-resets finished workers. Success terminals get
        # explicit zeros; timeouts bootstrap from the actual terminal state and
        # the base action that the fixed-TA queue would execute next.
        replay_next_state = next_observation.state_actor.copy()
        replay_next_base = next_observation.base_action_resfit.copy()
        replay_next_state[terminated] = 0.0
        replay_next_base[terminated] = 0.0
        if not self.bootstrap_timeouts:
            replay_next_state[truncated] = 0.0
            replay_next_base[truncated] = 0.0
        if bootstrap_timeout_indices:
            replay_next_state[bootstrap_timeout_indices] = self.state_preprocessor.transform(
                terminal_state_dp[bootstrap_timeout_indices]
            )
            replay_next_base[bootstrap_timeout_indices] = self.action_scaler.dp_to_resfit(
                timeout_next_base_dp
            )

        self._observation = next_observation
        return ResidualStepBatch(
            observation=next_observation,
            replay_next_state=replay_next_state,
            replay_next_base_action=replay_next_base,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            dones=dones,
            infos=[dict(info) for info in result.infos],
            base_action_resfit=np.asarray(combined["base_resfit"], dtype=np.float32),
            residual_resfit=residual.copy(),
            preclamp_action_resfit=np.asarray(combined["preclamp_resfit"], dtype=np.float32),
            full_action_resfit=np.asarray(combined["full_resfit"], dtype=np.float32),
            replay_action_resfit=np.asarray(combined["replay_resfit"], dtype=np.float32),
            action_raw=np.asarray(combined["action_raw"], dtype=np.float32),
            action_dp=action_dp,
            queue_resamples=int(self.queue.resample_count) - before_resamples,
        )

    def close(self) -> None:
        self.executor.close()
