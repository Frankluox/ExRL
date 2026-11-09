"""Fixed-TA Can environment adapter for DICE-RL on the frozen DSRL DP.

The structure is copied from the audited ResFiT chunk adapter, but DICE selects
the complete ``base + residual`` chunk before entering this adapter. One replay
transition is stored per issued chunk. Rewards use the existing project rule
``raw_sparse_reward - 1`` at every executed primitive step and are summed
without within-chunk discounting, matching DICE Eq. 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from baselines.dice.dp.core import (
    CompleteEpisodeAccumulator,
    DICEReplayBuffer,
    MacroTransition,
)
from exrl.runners.dp_q.vector_env import RobomimicExecutor, _extract_state_from_obs


@dataclass
class DICEChunkStepBatch:
    next_state: np.ndarray
    replay_next_state: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    dones: np.ndarray
    infos: List[Dict[str, object]]
    exec_lens: np.ndarray
    actions: np.ndarray
    base_actions: np.ndarray
    residuals: np.ndarray


class DICEChunkVecEnv:
    """Execute one complete TA32/TA50 DICE action per macro decision."""

    def __init__(
        self,
        executor: RobomimicExecutor,
        horizon_steps: int,
        action_dim: int,
        reward_offset: float = 1.0,
        bootstrap_timeouts: bool = True,
    ):
        self.executor = executor
        self.horizon_steps = int(horizon_steps)
        self.action_dim = int(action_dim)
        self.reward_offset = float(reward_offset)
        self.bootstrap_timeouts = bool(bootstrap_timeouts)
        self._state: Optional[np.ndarray] = None
        if self.horizon_steps <= 0 or self.action_dim <= 0:
            raise ValueError("invalid chunk dimensions")

    @property
    def n_envs(self) -> int:
        return int(self.executor.n_envs)

    def reset(self, base_seed: Optional[int] = None) -> np.ndarray:
        self._state = np.asarray(self.executor.reset_all(base_seed=base_seed), dtype=np.float32)
        return self._state.copy()

    @staticmethod
    def _time_limit_truncated(info: Dict[str, object]) -> bool:
        value = info.get("TimeLimit.truncated", False)
        if isinstance(value, (list, tuple, np.ndarray)):
            return bool(value[-1]) if len(value) else False
        return bool(value)

    @staticmethod
    def _terminal_state(info: Dict[str, object], expected_dim: int) -> np.ndarray:
        observation = info.get("terminal_observation")
        if observation is None:
            raise RuntimeError("timeout is missing terminal_observation")
        if isinstance(observation, dict):
            state = _extract_state_from_obs(observation)[0]
        else:
            state = np.asarray(observation, dtype=np.float32).reshape(-1)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (int(expected_dim),) or not np.all(np.isfinite(state)):
            raise RuntimeError("invalid timeout terminal state")
        return state

    def step(
        self,
        actions: np.ndarray,
        base_actions: np.ndarray,
        residuals: np.ndarray,
    ) -> DICEChunkStepBatch:
        if self._state is None:
            raise RuntimeError("environment must be reset before step")
        expected = (self.n_envs, self.horizon_steps, self.action_dim)
        action = np.asarray(actions, dtype=np.float32)
        base = np.asarray(base_actions, dtype=np.float32)
        residual = np.asarray(residuals, dtype=np.float32)
        if action.shape != expected or base.shape != expected or residual.shape != expected:
            raise ValueError("DICE action/base/residual shape mismatch")
        if not (
            np.all(np.isfinite(action))
            and np.all(np.isfinite(base))
            and np.all(np.isfinite(residual))
        ):
            raise ValueError("DICE action contains NaN/Inf")
        if not np.allclose(action, base + residual, rtol=0.0, atol=2e-6):
            raise RuntimeError("DICE action is not base + residual")

        requested = np.full((self.n_envs,), self.horizon_steps, dtype=np.int64)
        # Ask the shared executor for the undiscounted raw sparse-reward sum,
        # then apply the established reward_offset once per actual primitive.
        result = self.executor.step_variable_k(
            action,
            requested,
            gamma=None,
            reward_offset=0.0,
            discount_within_chunk=False,
        )
        raw_exec_lens = np.asarray(result.exec_lens).reshape(-1)
        exec_lens = raw_exec_lens.astype(np.int64)
        if (
            exec_lens.shape != (self.n_envs,)
            or np.any(raw_exec_lens != exec_lens)
            or np.any(exec_lens <= 0)
            or np.any(exec_lens > self.horizon_steps)
        ):
            raise RuntimeError("fixed-TA execution returned invalid durations")
        dones = np.asarray(result.dones, dtype=bool).reshape(-1)
        raw_rewards = np.asarray(result.rewards, dtype=np.float32).reshape(-1)
        if dones.shape != (self.n_envs,) or raw_rewards.shape != (self.n_envs,):
            raise RuntimeError("fixed-TA execution returned malformed reward/done arrays")
        rewards = raw_rewards - self.reward_offset * exec_lens.astype(np.float32)
        if np.any((~dones) & (exec_lens != self.horizon_steps)):
            raise RuntimeError("live worker executed a partial fixed-TA chunk")
        infos = [dict(info) for info in result.infos]
        if len(infos) != self.n_envs:
            raise RuntimeError("fixed-TA execution returned malformed infos")
        truncated = (
            np.asarray([self._time_limit_truncated(info) for info in infos], dtype=bool) & dones
        )
        terminated = dones & (~truncated)

        next_state = np.asarray(result.next_state, dtype=np.float32)
        if (
            next_state.ndim != 2
            or next_state.shape[0] != self.n_envs
            or not np.all(np.isfinite(next_state))
        ):
            raise RuntimeError("fixed-TA execution returned invalid next state")
        replay_next_state = next_state.copy()
        replay_next_state[terminated] = 0.0
        if not self.bootstrap_timeouts:
            replay_next_state[truncated] = 0.0
        else:
            for index in np.flatnonzero(truncated).tolist():
                replay_next_state[index] = self._terminal_state(infos[index], next_state.shape[1])
        self._state = next_state.copy()
        return DICEChunkStepBatch(
            next_state=next_state,
            replay_next_state=replay_next_state,
            rewards=rewards.astype(np.float32),
            terminated=terminated,
            truncated=truncated,
            dones=dones,
            infos=infos,
            exec_lens=exec_lens,
            actions=action.copy(),
            base_actions=base.copy(),
            residuals=residual.copy(),
        )

    def close(self) -> None:
        self.executor.close()


def add_chunk_step_to_replay(
    previous_state: np.ndarray,
    step: DICEChunkStepBatch,
    accumulator: CompleteEpisodeAccumulator,
    replay: DICEReplayBuffer,
) -> int:
    """Append four macro decisions and materialize any newly finished episodes."""

    states = np.asarray(previous_state, dtype=np.float32)
    if states.ndim != 2 or states.shape[0] != step.actions.shape[0]:
        raise ValueError("previous state batch mismatch")
    emitted = 0
    for env_index in range(states.shape[0]):
        transition = MacroTransition(
            state=states[env_index],
            action=step.actions[env_index].reshape(-1),
            reward=float(step.rewards[env_index]),
            next_state=step.replay_next_state[env_index],
            terminated=bool(step.terminated[env_index]),
            truncated=bool(step.truncated[env_index]),
        )
        emitted += replay.extend(accumulator.append(env_index, transition))
    return emitted


def action_audit_metrics(step: DICEChunkStepBatch) -> Dict[str, float]:
    action = np.asarray(step.actions, dtype=np.float32)
    base = np.asarray(step.base_actions, dtype=np.float32)
    residual = np.asarray(step.residuals, dtype=np.float32)
    return {
        "action/base_abs_mean": float(np.mean(np.abs(base))),
        "action/residual_abs_mean": float(np.mean(np.abs(residual))),
        "action/residual_l2_mean": float(np.mean(np.square(residual))),
        "action/full_abs_mean": float(np.mean(np.abs(action))),
        "action/out_of_dp_range_rate": float(np.mean(np.abs(action) > 1.0)),
        "action/max_abs": float(np.max(np.abs(action))),
    }
