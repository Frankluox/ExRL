"""Vectorized Robomimic execution of action-chunk prefixes."""

from __future__ import annotations

import os
import sys
import json
import tempfile
import queue
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class AttrDict(dict):
    """Dict with attribute access and recursive conversion."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value

    @staticmethod
    def convert(obj):
        if isinstance(obj, AttrDict):
            return obj
        if isinstance(obj, dict):
            out = AttrDict()
            for k, v in obj.items():
                out[k] = AttrDict.convert(v)
            return out
        if isinstance(obj, list):
            return [AttrDict.convert(v) for v in obj]
        return obj


def _as_np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _extract_state_from_obs(obs: Any) -> np.ndarray:
    """
    Convert official DSRL wrapped observation to state matrix [B, obs_dim].
    Supports:
    - list[dict(state=...)] from reset_arg
    - dict(state=...) from vectorized outputs
    - single dict(state=...) for one env
    """
    if isinstance(obs, list):
        if len(obs) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        rows = []
        for item in obs:
            rows.append(_extract_state_from_obs(item)[0])
        return np.asarray(rows, dtype=np.float32)

    if not isinstance(obs, dict) or "state" not in obs:
        raise ValueError(f"[ExRL] unsupported obs format: type={type(obs)}")

    s = _as_np(obs["state"])
    # Single env: [To, obs_dim] or [obs_dim]
    if s.ndim == 1:
        return s.reshape(1, -1)
    if s.ndim == 2:
        # could be [To, obs_dim] (single env) or [B, obs_dim]
        # DSRL low-dim with MultiStep typically gives [To, obs_dim] for single env.
        if s.shape[0] <= 8 and s.shape[1] > s.shape[0]:
            return s[-1].reshape(1, -1)
        return s.reshape(s.shape[0], -1)
    if s.ndim == 3:
        # [B, To, obs_dim]
        return s[:, -1, :].reshape(s.shape[0], -1)

    raise ValueError(f"[ExRL] unexpected state ndim={s.ndim}, shape={tuple(s.shape)}")


def split_state_to_obs_dict(
    state_batch: np.ndarray,
    lowdim_keys: Sequence[str],
    lowdim_dims: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """
    Split flat state [B, obs_dim] back to keyed obs dict required by adapter.
    """
    state_batch = np.asarray(state_batch, dtype=np.float32)
    if state_batch.ndim != 2:
        raise ValueError(
            f"state_batch must be 2D [B, D], got shape={state_batch.shape}"
        )
    total = int(sum(int(lowdim_dims[k]) for k in lowdim_keys))
    if state_batch.shape[1] != total:
        raise ValueError(
            f"state dim mismatch: got {state_batch.shape[1]}, expected sum(lowdim_dims)={total}"
        )
    out: Dict[str, np.ndarray] = {}
    st = 0
    for k in lowdim_keys:
        d = int(lowdim_dims[k])
        out[k] = state_batch[:, st : st + d]
        st += d
    return out


def check_environment_dependencies(vendor_root: str) -> None:
    """Dependencies are installed once with ``pip install -e .``."""
    import importlib.util

    for module in ("env", "stable_baselines3", "robomimic", "robosuite"):
        if importlib.util.find_spec(module) is None:
            raise ImportError(
                f"Missing {module}; install the selected requirements and run pip install -e ."
            )


@dataclass
class StepBatch:
    next_state: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    infos: List[Dict[str, Any]]
    exec_lens: np.ndarray


class RobomimicExecutor:
    """
    Variable-k execution runtime backed by official DSRL env wrappers.

    The underlying env stack is:
      robomimic_lowdim(normalization) -> multi_step(n_action_steps=1, reset_within_step=True)
      -> AsyncVectorEnv
    """

    def __init__(
        self,
        vendor_root: str,
        env_name: str,
        env_meta_json: str,
        normalization_path: str,
        lowdim_keys: Sequence[str],
        n_envs: int,
        max_episode_steps: int,
        seed: int = 42,
        asynchronous: bool = True,
        terminate_on_success: bool = False,
        timeout_on_horizon: bool = False,
    ):
        check_environment_dependencies(vendor_root)

        from env.gym_utils import make_async  # type: ignore

        wrappers = AttrDict.convert(
            {
                "robomimic_lowdim": {
                    "normalization_path": normalization_path,
                    "low_dim_keys": list(lowdim_keys),
                    "clamp_obs": False,
                },
                "multi_step": {
                    "n_obs_steps": 1,
                    "n_action_steps": 1,
                    "max_episode_steps": int(max_episode_steps),
                    "reset_within_step": True,
                    "reward_agg_method": "sum",
                    "prev_action": True,
                    "terminate_on_success": bool(terminate_on_success),
                    "timeout_on_horizon": bool(timeout_on_horizon),
                },
            }
        )

        # obs_dim/action_dim are only used by dummy env fallback.
        # For robomimic path they are not the effective source of truth.
        env_meta_path = self._prepare_env_meta_json(str(env_meta_json))

        self.venv = make_async(
            id=str(env_name),
            num_envs=int(n_envs),
            asynchronous=bool(asynchronous),
            wrappers=wrappers,
            max_episode_steps=int(max_episode_steps),
            robomimic_env_cfg_path=env_meta_path,
            use_image_obs=False,
            render=False,
            render_offscreen=False,
            obs_dim=23,
            action_dim=7,
        )
        self.n_envs = int(n_envs)
        self.seed = int(seed)
        self.max_episode_steps = int(max_episode_steps)
        self.lowdim_keys = list(lowdim_keys)
        self.terminate_on_success = bool(terminate_on_success)
        self.timeout_on_horizon = bool(timeout_on_horizon)

        # Important: seed each env differently.
        self.venv.seed([self.seed + i for i in range(self.n_envs)])
        self.current_state: Optional[np.ndarray] = None
        self._episode_success = np.zeros((self.n_envs,), dtype=bool)
        self._tmp_env_meta_json = (
            env_meta_path if env_meta_path != str(env_meta_json) else None
        )

    def _prepare_env_meta_json(self, env_meta_json: str) -> str:
        """
        Patch robosuite controller config for newer robosuite versions when needed.
        Return either original env_meta path or a temporary patched json path.
        """
        try:
            import robosuite
        except Exception:
            return env_meta_json

        try:
            with open(env_meta_json, "r") as f:
                env_meta = json.load(f)
        except Exception:
            return env_meta_json

        if not isinstance(env_meta, dict):
            return env_meta_json
        env_kwargs = env_meta.get("env_kwargs", {})
        if not isinstance(env_kwargs, dict):
            return env_meta_json
        ccfg = env_kwargs.get("controller_configs", None)
        if not isinstance(ccfg, dict):
            return env_meta_json
        ctype = str(ccfg.get("type", "")).upper()
        if ctype in {
            "BASIC",
            "HYBRID_MOBILE_BASE",
            "WHOLE_BODY_COMPOSITE",
            "WHOLE_BODY_IK",
        }:
            return env_meta_json

        robots = env_kwargs.get("robots", ["Panda"])
        robot = robots[0] if isinstance(robots, list) and len(robots) > 0 else "Panda"
        try:
            env_kwargs["controller_configs"] = (
                robosuite.load_composite_controller_config(robot=robot)
            )
        except Exception:
            return env_meta_json

        fd, tmp = tempfile.mkstemp(prefix="env_meta_patched_", suffix=".json")
        os.close(fd)
        with open(tmp, "w") as f:
            json.dump(env_meta, f)
        return tmp

    def close(self):
        try:
            self.venv.close()
        except Exception:
            pass
        if self._tmp_env_meta_json and os.path.exists(self._tmp_env_meta_json):
            try:
                os.remove(self._tmp_env_meta_json)
            except Exception:
                pass

    def reset_all(self, base_seed: Optional[int] = None) -> np.ndarray:
        options = []
        for i in range(self.n_envs):
            opt = {}
            if base_seed is not None:
                opt["seed"] = int(base_seed) + i
            options.append(opt)
        obs = self.venv.reset_arg(options_list=options)
        self.current_state = _extract_state_from_obs(obs).astype(np.float32)
        self._episode_success[:] = False
        return self.current_state.copy()

    def _reset_one(self, env_idx: int) -> np.ndarray:
        obs = self.venv.reset_one_arg(env_ind=int(env_idx), options={})
        self._episode_success[int(env_idx)] = False
        return _extract_state_from_obs(obs)[0].astype(np.float32)

    @staticmethod
    def _extract_success(info: Dict[str, Any]) -> bool:
        raw = info.get("is_success", None)
        if raw is None:
            episode = info.get("episode", {})
            if isinstance(episode, dict):
                raw = episode.get("is_success", None)
        if raw is None:
            raw = info.get("success", False)
        # MultiStep dict_take_last_n may return arrays/lists.
        if isinstance(raw, (list, tuple, np.ndarray)):
            if len(raw) == 0:
                return False
            raw = raw[-1]
        if isinstance(raw, dict):
            return bool(raw.get("task", False))
        return bool(raw)

    def step_variable_k(
        self,
        action_chunks: np.ndarray,
        exec_lens: np.ndarray,
        *,
        gamma: Optional[float] = None,
        reward_offset: float = 0.0,
        discount_within_chunk: bool = False,
    ) -> StepBatch:
        """
        Execute variable number of actions per env.
        Inputs:
          action_chunks: [N, H, A] normalized physical actions in [-1, 1]
          exec_lens:     [N] integer k in [0, H]
        Reward:
          - default (backward compatible): simple sum of step rewards
          - when discount_within_chunk=True and gamma is provided:
                sum_t (gamma^t) * (r_t - reward_offset)
        Info contract (for PPO timeout bootstrap compatibility):
          - when an env is done at this decision step, info contains:
                info["terminal_observation"] : raw obs dict at terminal step
                info["TimeLimit.truncated"]   : bool(truncation)
        """
        chunks = np.asarray(action_chunks, dtype=np.float32)
        ks = np.asarray(exec_lens, dtype=np.int64).reshape(-1)
        if chunks.ndim != 3:
            raise ValueError(f"action_chunks must be [N,H,A], got shape={chunks.shape}")
        n, h, _ = chunks.shape
        if n != self.n_envs or ks.shape[0] != self.n_envs:
            raise ValueError(
                f"batch mismatch: chunks_n={n}, ks_n={ks.shape[0]}, expected n_envs={self.n_envs}"
            )
        if np.any(ks < 0) or np.any(ks > h):
            raise ValueError(f"exec_lens out of range [0,{h}]: {ks}")

        if self.current_state is None:
            raise RuntimeError("Executor must be reset before step_variable_k")

        next_state = self.current_state.copy()
        rewards = np.zeros((self.n_envs,), dtype=np.float32)
        dones = np.zeros((self.n_envs,), dtype=bool)
        exec_actual = np.zeros((self.n_envs,), dtype=np.int64)
        infos: List[Dict[str, Any]] = [
            dict(is_success=False) for _ in range(self.n_envs)
        ]
        discount_t = np.ones((self.n_envs,), dtype=np.float32)
        use_discounted = bool(discount_within_chunk) and (gamma is not None)
        gamma_f = float(gamma) if gamma is not None else 1.0

        max_k = int(np.max(ks))
        for t in range(max_k):
            idx = [i for i in range(self.n_envs) if (not dones[i]) and (t < int(ks[i]))]
            if not idx:
                continue

            actions_t = [chunks[i, t] for i in idx]
            results = self.venv.call_sync_arg("step", "action", actions_t, indices=idx)
            for local_j, i in enumerate(idx):
                payload, ok = results[local_j]
                if not ok:
                    err_detail = ""
                    try:
                        idx, exctype, value = self.venv.error_queue.get_nowait()
                        err_detail = f" worker={idx} exctype={getattr(exctype, '__name__', exctype)} value={value}"
                    except queue.Empty:
                        pass
                    raise RuntimeError(f"[ExRL] env {i} step failed.{err_detail}")
                obs_i, rew_i, term_i, trunc_i, info_i = payload
                rew_val = float(rew_i)
                if use_discounted:
                    rewards[i] += (rew_val - float(reward_offset)) * float(
                        discount_t[i]
                    )
                    discount_t[i] *= gamma_f
                else:
                    rewards[i] += rew_val
                exec_actual[i] += 1
                infos[i] = dict(info_i) if isinstance(info_i, dict) else {}
                # Keep success sticky within an episode.
                #
                # Important:
                # `rew_val` here comes from the official DPPO env stack
                # (robomimic_lowdim + multi_step) before this compat layer applies
                # any RL reward_offset subtraction. For robomimic sparse tasks,
                # raw reward >= 1 means the task has succeeded. Some tasks
                # (notably square) do not always surface a reliable
                # info["is_success"] on the compat path, while the official eval
                # still counts success from raw reward. Use the raw sparse reward
                # as a fallback so compat success matches official eval semantics.
                step_success = bool(self._extract_success(infos[i]))
                if (not step_success) and (rew_val >= 1.0):
                    step_success = True
                self._episode_success[i] = bool(
                    self._episode_success[i] or step_success
                )
                infos[i]["is_success"] = bool(self._episode_success[i])
                done_i = bool(term_i) or bool(trunc_i)
                if self.terminate_on_success and bool(infos[i]["is_success"]):
                    done_i = True
                if done_i:
                    # Preserve terminal-step observation/truncation semantics for PPO timeout bootstrap.
                    infos[i]["terminal_observation"] = infos[i].get("final_obs", obs_i)
                    infos[i]["TimeLimit.truncated"] = bool(
                        infos[i].get("TimeLimit.truncated", False) or trunc_i
                    )
                dones[i] = done_i
                next_state[i] = _extract_state_from_obs(obs_i)[0]

        # Reset finished envs to keep collector running.
        for i in range(self.n_envs):
            if dones[i]:
                next_state[i] = self._reset_one(i)

        self.current_state = next_state.copy()
        return StepBatch(
            next_state=next_state,
            rewards=rewards,
            dones=dones,
            infos=infos,
            exec_lens=exec_actual.astype(np.float32),
        )
