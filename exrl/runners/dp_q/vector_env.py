"""Vectorized Robomimic execution of action-chunk prefixes."""

from __future__ import annotations

import os
import sys
import json
import tempfile
import queue
import types
import multiprocessing
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


def _current_worker_cpu_id(worker_cpu_ids: Sequence[int]) -> Optional[int]:
    cpu_ids = [int(cpu) for cpu in worker_cpu_ids]
    if not cpu_ids:
        return None
    identity = getattr(multiprocessing.current_process(), "_identity", ())
    worker_ordinal = int(identity[-1] - 1) if identity else int(os.getpid())
    return cpu_ids[worker_ordinal % len(cpu_ids)]


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


class WorkerChunkRPCWrapper:
    """Execute a variable-length chunk inside one AsyncVectorEnv worker.

    The wrapped object remains the official ``MultiStep(n_action_steps=1)``
    instance. Calling that exact ``step`` method once per primitive action keeps
    reset, timeout, success, observation and reward semantics identical to the
    historical parent-side loop while replacing up to ``k`` pipe round trips
    with one.
    """

    def __init__(
        self,
        env,
        terminate_on_success: bool = False,
        prune_unused_observables: bool = False,
        required_observable_names: Optional[Sequence[str]] = None,
        worker_cpu_ids: Optional[Sequence[int]] = None,
        cache_static_robot_metadata: bool = False,
        **kwargs,
    ):
        _ = kwargs
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})
        self.worker_cpu_id = None
        cpu_ids = list(worker_cpu_ids or ())
        self.worker_cpu_id = _current_worker_cpu_id(cpu_ids)
        if self.worker_cpu_id is not None:
            os.sched_setaffinity(0, {self.worker_cpu_id})
        self.terminate_on_success = bool(terminate_on_success)
        self.current_obs = None
        self.prune_unused_observables = bool(prune_unused_observables)
        if self.prune_unused_observables:
            self._prune_unused_observables(required_observable_names or ())
        self.cache_static_robot_metadata = bool(cache_static_robot_metadata)
        if self.cache_static_robot_metadata:
            self._cache_static_robot_metadata()

    def _find_robosuite_env(self):
        cursor = self.env
        seen = set()
        while cursor is not None and id(cursor) not in seen:
            seen.add(id(cursor))
            if hasattr(cursor, "_observables") and hasattr(cursor, "sim"):
                return cursor
            cursor = getattr(cursor, "env", None)
        return None

    def _prune_unused_observables(
        self, required_observable_names: Sequence[str]
    ) -> None:
        """Disable sensors that cannot contribute to the requested low-dim state."""
        suite_env = self._find_robosuite_env()
        if suite_env is None:
            raise RuntimeError("[ExRL] could not locate robosuite observables to prune")
        required = set(str(name) for name in required_observable_names)
        keep_object_modality = "object" in required
        missing = {
            name
            for name in required
            if name != "object" and name not in suite_env._observables
        }
        if missing:
            raise RuntimeError(
                "[ExRL] required low-dim observables are missing: " f"{sorted(missing)}"
            )
        for name, observable in suite_env._observables.items():
            keep = name in required
            if keep_object_modality and observable.modality == "object":
                # Keep the enabled dependency graph as well as active outputs:
                # e.g. Can_to_eef consumes the inactive-but-enabled
                # world_pose_in_gripper cache entry.
                keep = keep or bool(observable.is_enabled() or observable.is_active())
            if not keep:
                observable.set_active(False)
                observable.set_enabled(False)

    def _cache_static_robot_metadata(self) -> None:
        """Cache immutable action bounds and gripper actuator metadata."""
        suite_env = self._find_robosuite_env()
        if suite_env is None:
            raise RuntimeError("[ExRL] could not locate robosuite robots to cache")
        for robot in suite_env.robots:
            if getattr(robot, "_aac_static_metadata_cached", False):
                continue
            action_limits = tuple(
                np.asarray(value).copy() for value in robot.action_limits
            )
            torque_limits = tuple(
                np.asarray(value).copy() for value in robot.torque_limits
            )
            robot._aac_action_limits = action_limits
            robot._aac_torque_limits = torque_limits
            original_class = robot.__class__

            class _CachedStaticRobotMetadata(original_class):
                @property
                def action_limits(self):
                    return self._aac_action_limits

                @property
                def torque_limits(self):
                    return self._aac_torque_limits

            _CachedStaticRobotMetadata.__name__ = f"AACCached{original_class.__name__}"
            robot.__class__ = _CachedStaticRobotMetadata

            gripper = getattr(robot, "gripper", None)
            if callable(getattr(robot, "grip_action", None)) and hasattr(
                gripper, "actuators"
            ):
                actuator_indices = [
                    robot.sim.model.actuator_name2id(name) for name in gripper.actuators
                ]
                ctrl_range = np.asarray(
                    robot.sim.model.actuator_ctrlrange[actuator_indices]
                ).copy()
                bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
                weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])

                def _cached_grip_action(
                    _robot,
                    gripper,
                    gripper_action,
                    _indices=actuator_indices,
                    _bias=bias,
                    _weight=weight,
                ):
                    gripper_action_actual = gripper.format_action(gripper_action)
                    applied_gripper_action = _bias + _weight * gripper_action_actual
                    _robot.sim.data.ctrl[_indices] = applied_gripper_action

                robot.grip_action = types.MethodType(_cached_grip_action, robot)
            robot._aac_static_metadata_cached = True

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        self.current_obs = obs
        return obs

    def step(self, action):
        result = self.env.step(action)
        self.current_obs = result[0]
        return result

    def step_chunk(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        chunk = np.asarray(payload.get("action_chunk", []), dtype=np.float32)
        exec_len = int(payload.get("exec_len", 0))
        gamma = float(payload.get("gamma", 1.0))
        reward_offset = float(payload.get("reward_offset", 0.0))
        discounted = bool(payload.get("discount_within_chunk", False))
        collect_trace = bool(payload.get("collect_trace", False))
        if chunk.ndim != 2:
            raise ValueError(f"action_chunk must be [H,A], got shape={chunk.shape}")
        if exec_len < 0 or exec_len > int(chunk.shape[0]):
            raise ValueError(
                f"exec_len out of range for worker chunk: exec_len={exec_len}, shape={chunk.shape}"
            )

        obs = self.current_obs
        info: Dict[str, Any] = {}
        reward_total = np.float32(0.0)
        discount = np.float32(1.0)
        actual = 0
        terminated = False
        truncated = False
        trace_states: List[np.ndarray] = []
        trace_actions: List[np.ndarray] = []

        for t in range(exec_len):
            if collect_trace:
                trace_states.append(
                    _extract_state_from_obs(obs)[0].astype(np.float32, copy=True)
                )
                trace_actions.append(chunk[t].astype(np.float32, copy=True))
            obs, reward, terminated, truncated, raw_info = self.env.step(chunk[t])
            self.current_obs = obs
            info = dict(raw_info) if isinstance(raw_info, dict) else {}
            reward_value = float(reward)
            if discounted:
                reward_total = np.float32(
                    float(reward_total)
                    + (reward_value - reward_offset) * float(discount)
                )
                discount = np.float32(float(discount) * gamma)
            else:
                reward_total = np.float32(float(reward_total) + reward_value)
            actual += 1

            # The parent loop historically treated a sparse raw reward as a
            # success fallback even if robosuite did not expose is_success.
            success = bool(RobomimicExecutor._extract_success(info))
            if (not success) and reward_value >= 1.0:
                success = True
                info["is_success"] = True
            if self.terminate_on_success and success:
                terminated = True
                truncated = False
            if bool(terminated) or bool(truncated):
                break

        result = {
            "obs": obs,
            "reward": float(reward_total),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
            "exec_len": int(actual),
        }
        if collect_trace:
            result["trace_states"] = trace_states
            result["trace_actions"] = trace_actions
        return result


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
        zero_action_fail_streak: int = 0,
        zero_action_fail_reward: float = -20.0,
        terminate_on_success: bool = False,
        timeout_on_horizon: bool = False,
        collect_success_episodes: bool = True,
        variable_k_runtime: str = "parent_loop",
        prune_unused_observables: bool = False,
        worker_cpu_ids: Optional[Sequence[int]] = None,
        cache_static_robot_metadata: bool = False,
    ):
        check_environment_dependencies(vendor_root)

        from env.gym_utils import make_async  # type: ignore
        from env.gym_utils import wrapper as wrapper_module  # type: ignore

        self.variable_k_runtime = str(variable_k_runtime).lower()
        if self.variable_k_runtime not in {"parent_loop", "worker_chunk"}:
            raise ValueError(
                "variable_k_runtime must be one of {'parent_loop', 'worker_chunk'}, "
                f"got {variable_k_runtime!r}"
            )
        if self.variable_k_runtime == "worker_chunk":
            wrapper_module.wrapper_dict["worker_chunk_rpc"] = WorkerChunkRPCWrapper

        wrapper_config = {
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
        if self.variable_k_runtime == "worker_chunk":
            # Keep the official MultiStep wrapper directly below this RPC
            # adapter so every primitive env.step follows the legacy path.
            wrapper_config["worker_chunk_rpc"] = {
                "terminate_on_success": bool(terminate_on_success),
                "prune_unused_observables": bool(prune_unused_observables),
                "required_observable_names": list(lowdim_keys),
                "worker_cpu_ids": list(worker_cpu_ids or ()),
                "cache_static_robot_metadata": bool(cache_static_robot_metadata),
            }
        wrappers = AttrDict.convert(wrapper_config)

        # obs_dim/action_dim are only used by dummy env fallback.
        # For robomimic path they are not the effective source of truth.
        env_meta_path = self._prepare_env_meta_json(str(env_meta_json))

        cpu_ids = [int(cpu) for cpu in (worker_cpu_ids or ())]
        env_utils_module = None
        original_create_env = None
        if bool(asynchronous) and cpu_ids:
            import robomimic.utils.env_utils as env_utils_module  # type: ignore

            original_create_env = env_utils_module.create_env_from_metadata

            def _create_env_after_worker_pin(*args, **kwargs):
                if multiprocessing.current_process().name != "MainProcess":
                    cpu_id = _current_worker_cpu_id(cpu_ids)
                    if cpu_id is not None:
                        os.sched_setaffinity(0, {cpu_id})
                return original_create_env(*args, **kwargs)

            env_utils_module.create_env_from_metadata = _create_env_after_worker_pin
        try:
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
        finally:
            if env_utils_module is not None and original_create_env is not None:
                env_utils_module.create_env_from_metadata = original_create_env
        self.n_envs = int(n_envs)
        self.seed = int(seed)
        self.max_episode_steps = int(max_episode_steps)
        self.lowdim_keys = list(lowdim_keys)
        self.zero_action_fail_streak = max(int(zero_action_fail_streak), 0)
        self.zero_action_fail_reward = float(zero_action_fail_reward)
        self.terminate_on_success = bool(terminate_on_success)
        self.timeout_on_horizon = bool(timeout_on_horizon)
        self.collect_success_episodes = bool(collect_success_episodes)

        # Important: seed each env differently.
        self.venv.seed([self.seed + i for i in range(self.n_envs)])
        self.current_state: Optional[np.ndarray] = None
        self._episode_success = np.zeros((self.n_envs,), dtype=bool)
        self._zero_action_streak = np.zeros((self.n_envs,), dtype=np.int64)
        self._bc_episode_states: List[List[np.ndarray]] = [
            [] for _ in range(self.n_envs)
        ]
        self._bc_episode_actions: List[List[np.ndarray]] = [
            [] for _ in range(self.n_envs)
        ]
        self._bc_episode_decision_ids: List[List[int]] = [
            [] for _ in range(self.n_envs)
        ]
        self._bc_next_decision_id = np.zeros((self.n_envs,), dtype=np.int64)
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
        self._zero_action_streak[:] = 0
        if self.collect_success_episodes:
            self._bc_episode_states = [[] for _ in range(self.n_envs)]
            self._bc_episode_actions = [[] for _ in range(self.n_envs)]
            self._bc_episode_decision_ids = [[] for _ in range(self.n_envs)]
            self._bc_next_decision_id[:] = 0
        return self.current_state.copy()

    def _reset_one(self, env_idx: int) -> np.ndarray:
        obs = self.venv.reset_one_arg(env_ind=int(env_idx), options={})
        self._episode_success[int(env_idx)] = False
        self._zero_action_streak[int(env_idx)] = 0
        if self.collect_success_episodes:
            self._bc_episode_states[int(env_idx)] = []
            self._bc_episode_actions[int(env_idx)] = []
            self._bc_episode_decision_ids[int(env_idx)] = []
            self._bc_next_decision_id[int(env_idx)] = 0
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

    def _step_variable_k_worker_chunk(
        self,
        chunks: np.ndarray,
        ks: np.ndarray,
        *,
        gamma: Optional[float],
        reward_offset: float,
        discount_within_chunk: bool,
    ) -> StepBatch:
        """One RPC per active env, with primitive stepping kept in MultiStep."""
        next_state = self.current_state.copy()
        rewards = np.zeros((self.n_envs,), dtype=np.float32)
        dones = np.zeros((self.n_envs,), dtype=bool)
        exec_actual = np.zeros((self.n_envs,), dtype=np.int64)
        infos: List[Dict[str, Any]] = [
            dict(is_success=False) for _ in range(self.n_envs)
        ]
        gamma_f = float(gamma) if gamma is not None else 1.0
        decision_ids = self._bc_next_decision_id.copy()
        if self.collect_success_episodes:
            self._bc_next_decision_id += 1

        active_indices: List[int] = []
        active_payloads: List[Dict[str, Any]] = []
        for i in range(self.n_envs):
            k_i = int(ks[i])
            if k_i <= 0:
                self._zero_action_streak[i] += 1
                infos[i] = {
                    "exec_len": 0,
                    "resampled": True,
                    "resample_streak": int(self._zero_action_streak[i]),
                    "is_success": bool(self._episode_success[i]),
                }
                if (
                    self.zero_action_fail_streak > 0
                    and int(self._zero_action_streak[i]) >= self.zero_action_fail_streak
                ):
                    rewards[i] = float(self.zero_action_fail_reward)
                    dones[i] = True
                    infos[i]["is_success"] = False
                    infos[i]["forced_failure_due_to_zero_action_streak"] = True
                    infos[i]["forced_failure_reward"] = float(
                        self.zero_action_fail_reward
                    )
                    infos[i]["terminal_observation"] = self.current_state[i].copy()
                    infos[i]["TimeLimit.truncated"] = False
                continue

            self._zero_action_streak[i] = 0
            active_indices.append(i)
            active_payloads.append(
                {
                    "action_chunk": chunks[i, :k_i],
                    "exec_len": k_i,
                    "gamma": gamma_f,
                    "reward_offset": float(reward_offset),
                    "discount_within_chunk": bool(
                        discount_within_chunk and gamma is not None
                    ),
                    "collect_trace": bool(self.collect_success_episodes),
                }
            )

        if active_indices:
            results = self.venv.call_sync_arg(
                "step_chunk", "payload", active_payloads, indices=active_indices
            )
            for local_index, env_index in enumerate(active_indices):
                payload, ok = results[local_index]
                if not ok:
                    err_detail = ""
                    try:
                        failed_idx, exctype, value = self.venv.error_queue.get_nowait()
                        err_detail = (
                            f" worker={failed_idx} "
                            f"exctype={getattr(exctype, '__name__', exctype)} value={value}"
                        )
                    except queue.Empty:
                        pass
                    raise RuntimeError(
                        f"[ExRL] env {env_index} step_chunk failed.{err_detail}"
                    )
                if not isinstance(payload, dict):
                    raise RuntimeError(
                        f"[ExRL] env {env_index} step_chunk returned {type(payload)}"
                    )

                obs_i = payload["obs"]
                rewards[env_index] = float(payload.get("reward", 0.0))
                exec_actual[env_index] = int(payload.get("exec_len", 0))
                raw_info = payload.get("info", {})
                infos[env_index] = dict(raw_info) if isinstance(raw_info, dict) else {}
                if self.collect_success_episodes:
                    trace_pairs = list(
                        zip(
                            payload.get("trace_states", []),
                            payload.get("trace_actions", []),
                        )
                    )
                    for state_t, action_t in trace_pairs:
                        self._bc_episode_states[env_index].append(
                            np.asarray(state_t, dtype=np.float32).copy()
                        )
                        self._bc_episode_actions[env_index].append(
                            np.asarray(action_t, dtype=np.float32).copy()
                        )
                    self._bc_episode_decision_ids[env_index].extend(
                        [int(decision_ids[env_index])] * len(trace_pairs)
                    )

                step_success = bool(self._extract_success(infos[env_index]))
                self._episode_success[env_index] = bool(
                    self._episode_success[env_index] or step_success
                )
                infos[env_index]["is_success"] = bool(self._episode_success[env_index])
                done_i = bool(payload.get("terminated", False)) or bool(
                    payload.get("truncated", False)
                )
                if self.terminate_on_success and bool(infos[env_index]["is_success"]):
                    done_i = True
                if done_i:
                    infos[env_index]["terminal_observation"] = infos[env_index].get(
                        "final_obs", obs_i
                    )
                    infos[env_index]["TimeLimit.truncated"] = bool(
                        infos[env_index].get("TimeLimit.truncated", False)
                        or payload.get("truncated", False)
                    )
                    if (
                        self.collect_success_episodes
                        and bool(infos[env_index].get("is_success", False))
                        and len(self._bc_episode_actions[env_index]) > 0
                    ):
                        infos[env_index]["bc_success_episode_states"] = np.asarray(
                            self._bc_episode_states[env_index], dtype=np.float32
                        )
                        infos[env_index]["bc_success_episode_actions"] = np.asarray(
                            self._bc_episode_actions[env_index], dtype=np.float32
                        )
                        infos[env_index]["bc_success_episode_decision_ids"] = (
                            np.asarray(
                                self._bc_episode_decision_ids[env_index], dtype=np.int64
                            )
                        )
                        infos[env_index]["bc_success_episode_len"] = int(
                            len(self._bc_episode_actions[env_index])
                        )
                dones[env_index] = done_i
                next_state[env_index] = _extract_state_from_obs(obs_i)[0]

        # This explicit reset is intentional: the historical parent loop also
        # resets once more after MultiStep(reset_within_step=True) returns done.
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

        if self.variable_k_runtime == "worker_chunk":
            return self._step_variable_k_worker_chunk(
                chunks,
                ks,
                gamma=gamma,
                reward_offset=reward_offset,
                discount_within_chunk=discount_within_chunk,
            )

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
        decision_ids = self._bc_next_decision_id.copy()
        if self.collect_success_episodes:
            self._bc_next_decision_id += 1

        zero_idx = [i for i in range(self.n_envs) if int(ks[i]) <= 0]
        pos_idx = [i for i in range(self.n_envs) if int(ks[i]) > 0]
        for i in pos_idx:
            self._zero_action_streak[i] = 0
        for i in zero_idx:
            self._zero_action_streak[i] += 1
            infos[i] = {
                "exec_len": 0,
                "resampled": True,
                "resample_streak": int(self._zero_action_streak[i]),
                "is_success": bool(self._episode_success[i]),
            }
            if (
                self.zero_action_fail_streak > 0
                and int(self._zero_action_streak[i]) >= self.zero_action_fail_streak
            ):
                rewards[i] = float(self.zero_action_fail_reward)
                dones[i] = True
                infos[i]["is_success"] = False
                infos[i]["forced_failure_due_to_zero_action_streak"] = True
                infos[i]["forced_failure_reward"] = float(self.zero_action_fail_reward)
                infos[i]["terminal_observation"] = self.current_state[i].copy()
                infos[i]["TimeLimit.truncated"] = False

        max_k = int(np.max(ks))
        for t in range(max_k):
            idx = [i for i in range(self.n_envs) if (not dones[i]) and (t < int(ks[i]))]
            if not idx:
                continue

            if self.collect_success_episodes:
                states_before_t = {i: next_state[i].copy() for i in idx}
                actions_before_t = {i: chunks[i, t].copy() for i in idx}
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
                if self.collect_success_episodes:
                    self._bc_episode_states[i].append(
                        states_before_t[i].astype(np.float32, copy=True)
                    )
                    self._bc_episode_actions[i].append(
                        actions_before_t[i].astype(np.float32, copy=True)
                    )
                    self._bc_episode_decision_ids[i].append(int(decision_ids[i]))
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
                    if (
                        self.collect_success_episodes
                        and bool(infos[i].get("is_success", False))
                        and len(self._bc_episode_actions[i]) > 0
                    ):
                        infos[i]["bc_success_episode_states"] = np.asarray(
                            self._bc_episode_states[i], dtype=np.float32
                        )
                        infos[i]["bc_success_episode_actions"] = np.asarray(
                            self._bc_episode_actions[i], dtype=np.float32
                        )
                        infos[i]["bc_success_episode_decision_ids"] = np.asarray(
                            self._bc_episode_decision_ids[i], dtype=np.int64
                        )
                        infos[i]["bc_success_episode_len"] = int(
                            len(self._bc_episode_actions[i])
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
