"""Execute action-chunk prefixes and construct critic observations."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

import numpy as np

try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces

from exrl.runners.dp_q.chunk_env import (
    BatchedAdaptiveWrapper as LegacyBatchedAdaptiveWrapper,
    DSRLStylePolicy,
    RobomimicToGymWrapper,
)
from exrl.runners.dp_q.vector_env import RobomimicExecutor, split_state_to_obs_dict


class _DSRLExecutorVecAdapter:
    """
    Minimal Vec-like adapter so legacy BatchedAdaptiveWrapper can run unchanged.
    Exposes reset/step_async/step_wait/get_attr/set_attr/env_method/env_is_wrapped/seed/close.
    """

    def __init__(
        self,
        *,
        vendor_root: str,
        env_name: str,
        env_meta_json: str,
        normalization_path: str,
        lowdim_keys: Sequence[str],
        lowdim_dims: Dict[str, int],
        n_envs: int,
        max_episode_steps: int,
        seed: int,
        gamma: float,
        reward_offset: float,
        zero_action_fail_streak: int,
        zero_action_fail_reward: float,
        terminate_on_success: bool,
        timeout_on_horizon: bool,
        collect_success_episodes: bool,
        variable_k_runtime: str,
        prune_unused_observables: bool,
        worker_cpu_ids: Sequence[int],
        cache_static_robot_metadata: bool,
    ):
        self.lowdim_keys = list(lowdim_keys)
        self.lowdim_dims = {k: int(v) for k, v in dict(lowdim_dims).items()}
        self.num_envs = int(n_envs)
        self.gamma = float(gamma)
        self.reward_offset = float(reward_offset)
        self.seed_value = int(seed)

        self.executor = RobomimicExecutor(
            vendor_root=str(vendor_root),
            env_name=str(env_name),
            env_meta_json=str(env_meta_json),
            normalization_path=str(normalization_path),
            lowdim_keys=self.lowdim_keys,
            n_envs=self.num_envs,
            max_episode_steps=int(max_episode_steps),
            seed=self.seed_value,
            asynchronous=True,
            zero_action_fail_streak=int(zero_action_fail_streak),
            zero_action_fail_reward=float(zero_action_fail_reward),
            terminate_on_success=bool(terminate_on_success),
            timeout_on_horizon=bool(timeout_on_horizon),
            collect_success_episodes=bool(collect_success_episodes),
            variable_k_runtime=str(variable_k_runtime),
            prune_unused_observables=bool(prune_unused_observables),
            worker_cpu_ids=list(worker_cpu_ids),
            cache_static_robot_metadata=bool(cache_static_robot_metadata),
        )

        self._pending_payloads = None
        self._last_states = None

    def reset(self):
        states = self.executor.reset_all(base_seed=None)
        states = np.asarray(states, dtype=np.float32)
        self._last_states = states
        return split_state_to_obs_dict(
            states, lowdim_keys=self.lowdim_keys, lowdim_dims=self.lowdim_dims
        )

    def step_async(self, payloads):
        self._pending_payloads = payloads

    def step_wait(self):
        if self._pending_payloads is None:
            raise RuntimeError("[DSRL-ADAPTER] step_wait called before step_async")

        payloads = self._pending_payloads
        self._pending_payloads = None

        ks = np.asarray([int(p.get("k", 0)) for p in payloads], dtype=np.int64)
        # old main wrapper expects k in [0, K]
        if np.any(ks < 0):
            raise RuntimeError(f"[DSRL-ADAPTER] negative k in payload: {ks.tolist()}")

        # Build dense action chunk tensor [N, L, A] from variable-length payload chunks.
        chunks_raw = [
            np.asarray(p.get("chunk", []), dtype=np.float32) for p in payloads
        ]
        action_dim = 0
        max_len = 0
        for c in chunks_raw:
            if c.ndim == 2:
                action_dim = max(action_dim, int(c.shape[1]))
                max_len = max(max_len, int(c.shape[0]))
        max_len = max(max_len, int(np.max(ks)) if ks.size > 0 else 0)

        if action_dim <= 0:
            # Fallback: infer from first non-empty state by lowdim split impossible; hard fail is better.
            raise RuntimeError(
                "[DSRL-ADAPTER] cannot infer action_dim from payload chunk"
            )

        dense = np.zeros((self.num_envs, max_len, action_dim), dtype=np.float32)
        for i, c in enumerate(chunks_raw):
            if c.ndim != 2 or c.shape[0] == 0:
                continue
            l = min(int(c.shape[0]), max_len)
            dense[i, :l, :] = c[:l, :]

        step = self.executor.step_variable_k(
            action_chunks=dense,
            exec_lens=ks,
            gamma=self.gamma,
            reward_offset=self.reward_offset,
            discount_within_chunk=True,
        )

        next_states = np.asarray(step.next_state, dtype=np.float32)
        self._last_states = next_states

        rews = np.asarray(step.rewards, dtype=np.float32).reshape(-1)
        dones = np.asarray(step.dones, dtype=bool).reshape(-1)
        exec_lens = np.asarray(step.exec_lens, dtype=np.int64).reshape(-1)

        infos: List[Dict[str, Any]] = []
        for i in range(self.num_envs):
            info_i = (
                dict(step.infos[i])
                if i < len(step.infos) and isinstance(step.infos[i], dict)
                else {}
            )
            info_i["exec_len"] = int(exec_lens[i])
            info_i["resampled"] = bool(exec_lens[i] == 0)
            info_i.setdefault("TimeLimit.truncated", False)
            infos.append(info_i)

        obs_batch = split_state_to_obs_dict(
            next_states, lowdim_keys=self.lowdim_keys, lowdim_dims=self.lowdim_dims
        )
        return obs_batch, rews, dones, infos

    def get_attr(self, attr_name, indices=None):
        _ = indices
        val = getattr(self, attr_name)
        return [val for _ in range(self.num_envs)]

    def set_attr(self, attr_name, value, indices=None):
        _ = indices
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        _ = indices
        method = getattr(self, method_name)
        ret = method(*method_args, **method_kwargs)
        return [ret for _ in range(self.num_envs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        _ = wrapper_class
        _ = indices
        return [False for _ in range(self.num_envs)]

    def seed(self, seed=None):
        if seed is not None:
            self.seed_value = int(seed)
        return [int(self.seed_value + i) for i in range(self.num_envs)]

    def close(self):
        try:
            self.executor.close()
        except Exception:
            pass


class BatchedAdaptiveWrapper(LegacyBatchedAdaptiveWrapper):
    """
    Legacy wrapper + DSRL env backend.

    This extends the batched chunk wrapper,
    and only replaces the raw vecenv source with _DSRLExecutorVecAdapter.
    """

    def __init__(
        self,
        venv,
        policy,
        max_chunk_len,
        device,
        reward_offset=0.0,
        resample_penalty=-0.1,
        save_all_videos=False,
        video_dir=None,
        action_history_len=0,
        dp_infer_steps=5,
        dp_sampler="ddim",
        dp_amp=False,
        dp_compile=False,
        dp_cuda_graph=False,
        ipc_chunk_dtype="fp32",
        perf_profile=False,
        dp_infer_gpus="",
        dp_grouping="round_robin",
        dp_infer_parallel=False,
        ipc_send_mode="executed_only",
        td_target_next_obs_samples=1,
        critic_obs_unnormalize_dp_chunk=False,
        critic_obs_action_mode="stats",
        critic_obs_state_mode="stats_clip",
        *,
        vendor_root: str,
        env_name: str,
        env_meta_json: str,
        normalization_path: str,
        stats_npz_path: str,
        lowdim_keys: Sequence[str],
        lowdim_dims: Dict[str, int],
        n_envs: int,
        max_episode_steps: int,
        gamma: float,
        seed: int = 42,
        min_chunk_len: int = 1,
        zero_action_fail_streak: int = 0,
        zero_action_fail_reward: float = -20.0,
        terminate_on_success: bool = False,
        timeout_on_horizon: bool = False,
        collect_success_episodes: bool = True,
        dsrl_variable_k_runtime: str = "parent_loop",
        dsrl_prune_unused_observables: bool = False,
        dsrl_worker_cpu_ids: Sequence[int] = (),
        dsrl_cache_static_robot_metadata: bool = False,
    ):
        _ = venv

        min_chunk_len = int(min_chunk_len)
        if min_chunk_len < 0 or min_chunk_len > int(max_chunk_len):
            raise ValueError(
                f"[ExRL] invalid min_chunk_len={min_chunk_len}, "
                f"expected in [0, {int(max_chunk_len)}]"
            )
        os.environ["OPT_MIN_CHUNK_LEN"] = str(min_chunk_len)

        # Ensure legacy wrapper reads the intended task stats (can/lift/...)
        os.environ["OPT_TOOLHANG_STATS_NPZ"] = str(stats_npz_path)
        self._stats_path_override = str(stats_npz_path)

        critic_obs_state_mode = str(critic_obs_state_mode).lower()
        valid_state_modes = {"stats_clip", "stats_no_clip", "dsrl_raw"}
        if critic_obs_state_mode not in valid_state_modes:
            raise ValueError(
                f"[ExRL] invalid critic_obs_state_mode={critic_obs_state_mode}, "
                f"expected one of {sorted(valid_state_modes)}"
            )
        # The legacy constructor probes the observation space through the
        # overridden _build_obs_matrix, so this must be set before super().
        self.critic_obs_state_mode = critic_obs_state_mode
        if self.critic_obs_state_mode == "dsrl_raw":
            print(
                "[ExRL] critic obs uses DSRL-normalized state directly (no stats z-score/clip)"
            )
        elif self.critic_obs_state_mode == "stats_no_clip":
            print("[ExRL] critic obs uses task-stats z-score without clip")

        critic_obs_action_min = None
        critic_obs_action_max = None
        critic_obs_action_mode = str(critic_obs_action_mode).lower()
        if bool(critic_obs_unnormalize_dp_chunk):
            if critic_obs_action_mode not in ("stats", "dsrl_unnorm_stats"):
                raise ValueError(
                    "[ExRL] critic_obs_unnormalize_dp_chunk conflicts with "
                    f"critic_obs_action_mode={critic_obs_action_mode}"
                )
            critic_obs_action_mode = "dsrl_unnorm_stats"
        if critic_obs_action_mode == "dsrl_unnorm_stats":
            norm_data = np.load(str(normalization_path))
            missing = [k for k in ("action_min", "action_max") if k not in norm_data]
            if missing:
                raise KeyError(
                    f"[ExRL] normalization npz missing keys for critic obs action unnormalize: {missing}"
                )
            critic_obs_action_min = np.asarray(
                norm_data["action_min"], dtype=np.float32
            )
            critic_obs_action_max = np.asarray(
                norm_data["action_max"], dtype=np.float32
            )

        raw_venv = _DSRLExecutorVecAdapter(
            vendor_root=vendor_root,
            env_name=env_name,
            env_meta_json=env_meta_json,
            normalization_path=normalization_path,
            lowdim_keys=lowdim_keys,
            lowdim_dims=lowdim_dims,
            n_envs=n_envs,
            max_episode_steps=max_episode_steps,
            seed=seed,
            gamma=gamma,
            reward_offset=reward_offset,
            zero_action_fail_streak=zero_action_fail_streak,
            zero_action_fail_reward=zero_action_fail_reward,
            terminate_on_success=terminate_on_success,
            timeout_on_horizon=timeout_on_horizon,
            collect_success_episodes=collect_success_episodes,
            variable_k_runtime=str(dsrl_variable_k_runtime),
            prune_unused_observables=bool(dsrl_prune_unused_observables),
            worker_cpu_ids=list(dsrl_worker_cpu_ids),
            cache_static_robot_metadata=bool(dsrl_cache_static_robot_metadata),
        )

        super().__init__(
            venv=raw_venv,
            policy=policy,
            max_chunk_len=max_chunk_len,
            device=device,
            reward_offset=reward_offset,
            resample_penalty=resample_penalty,
            save_all_videos=save_all_videos,
            video_dir=video_dir,
            action_history_len=action_history_len,
            dp_infer_steps=dp_infer_steps,
            dp_sampler=dp_sampler,
            dp_amp=dp_amp,
            dp_compile=dp_compile,
            dp_cuda_graph=dp_cuda_graph,
            ipc_chunk_dtype=ipc_chunk_dtype,
            perf_profile=perf_profile,
            dp_infer_gpus=dp_infer_gpus,
            dp_grouping=dp_grouping,
            dp_infer_parallel=dp_infer_parallel,
            ipc_send_mode=ipc_send_mode,
            td_target_next_obs_samples=td_target_next_obs_samples,
            critic_obs_unnormalize_dp_chunk=bool(
                critic_obs_action_mode == "dsrl_unnorm_stats"
            ),
            critic_obs_action_mode=critic_obs_action_mode,
            critic_obs_action_min=critic_obs_action_min,
            critic_obs_action_max=critic_obs_action_max,
        )

        self.min_chunk_len = min_chunk_len
        self.action_space = type(self.action_space)(
            int(max_chunk_len) - self.min_chunk_len + 1
        )
        got_dim = int(self.action_space.n)
        print(
            f"[ExRL] action space mapped to k in [{self.min_chunk_len}, {int(max_chunk_len)}], "
            f"policy_dim={got_dim}"
        )

        # The legacy wrapper hard-codes single-arm keys while probing its observation space.
        # Replace that layout with the task's complete key set before training starts.
        self._set_task_obs_layout(lowdim_keys=lowdim_keys, lowdim_dims=lowdim_dims)
        self._reload_stats_override()
        self._refresh_obs_dims_from_stats()
        self._rebuild_task_observation_space()

    def _set_task_obs_layout(self, *, lowdim_keys, lowdim_dims):
        keys = [str(key) for key in lowdim_keys]
        if not keys:
            raise ValueError("[ExRL] lowdim_keys cannot be empty")
        if len(set(keys)) != len(keys):
            raise ValueError(f"[ExRL] duplicate lowdim keys: {keys}")
        missing = [key for key in keys if key not in lowdim_dims]
        if missing:
            raise KeyError(f"[ExRL] lowdim_dims missing keys: {missing}")
        dims = {key: int(lowdim_dims[key]) for key in keys}
        invalid = {key: dim for key, dim in dims.items() if dim <= 0}
        if invalid:
            raise ValueError(f"[ExRL] invalid lowdim dimensions: {invalid}")
        self._obs_keys = keys
        self._obs_dims = dims

    def _rebuild_task_observation_space(self):
        state_dim = int(sum(self._obs_dims[key] for key in self._obs_keys))
        history_dim = int(self.action_history_len) * int(self.action_dim)
        chunk_dim = int(self.max_chunk_len) * int(self.action_dim)
        self.total_obs_dim = state_dim + history_dim + chunk_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.total_obs_dim,),
            dtype=np.float32,
        )
        print(
            "[ExRL] critic observation layout "
            f"keys={self._obs_keys} state_dim={state_dim} history_dim={history_dim} "
            f"chunk_dim={chunk_dim} total_obs_dim={self.total_obs_dim}"
        )

    def _reload_stats_override(self):
        if not self._stats_path_override:
            return
        if not os.path.exists(self._stats_path_override):
            print(f"[ExRL][WARN] stats override not found: {self._stats_path_override}")
            return
        stats = np.load(self._stats_path_override)
        self.stats = stats
        self._action_mean = np.asarray(self.stats["action_mean"], dtype=np.float32)
        self._action_std = np.asarray(self.stats["action_std"], dtype=np.float32)

    def _refresh_obs_dims_from_stats(self):
        dims = {}
        for k in self._obs_keys:
            mk = f"{k}_mean"
            if mk in self.stats:
                dims[k] = int(np.asarray(self.stats[mk]).reshape(-1).shape[0])
        if dims:
            self._obs_dims = dims

    def _safe_norm_stats(self, key, dim):
        mean_k = f"{key}_mean"
        std_k = f"{key}_std"
        if mean_k in self.stats and std_k in self.stats:
            mean = np.asarray(self.stats[mean_k], dtype=np.float32).reshape(-1)
            std = np.asarray(self.stats[std_k], dtype=np.float32).reshape(-1)
            if mean.shape[0] == dim and std.shape[0] == dim:
                std = np.where(std < 1e-8, 1.0, std)
                return mean.reshape(1, dim), std.reshape(1, dim)
        return np.zeros((1, dim), dtype=np.float32), np.ones((1, dim), dtype=np.float32)

    # Override only to support task-specific lowdim dims; keep old normalization semantics.
    def _build_obs_matrix(self, obs_batch):
        n = self.num_envs
        for k in self._obs_keys:
            if k in obs_batch:
                n = int(np.asarray(obs_batch[k]).shape[0])
                break

        dims = {}
        for k in self._obs_keys:
            if k in obs_batch:
                dims[k] = int(
                    np.asarray(obs_batch[k], dtype=np.float32).reshape(n, -1).shape[1]
                )
            elif f"{k}_mean" in self.stats:
                dims[k] = int(np.asarray(self.stats[f"{k}_mean"]).reshape(-1).shape[0])
            else:
                dims[k] = int(self._obs_dims.get(k, 0))

        self._obs_dims = dims
        use_stats = self.critic_obs_state_mode in {"stats_clip", "stats_no_clip"}
        clip_stats = self.critic_obs_state_mode == "stats_clip"
        parts = []
        for k in self._obs_keys:
            dim = int(dims[k])
            if dim <= 0:
                continue
            if k in obs_batch:
                v = np.asarray(obs_batch[k], dtype=np.float32).reshape(n, dim)
                if use_stats:
                    mean, std = self._safe_norm_stats(k, dim)
                    v = (v - mean) / std
                parts.append(v)
            else:
                parts.append(np.zeros((n, dim), dtype=np.float32))

        if not parts:
            return np.zeros((n, 0), dtype=np.float32)

        obs = np.concatenate(parts, axis=1).astype(np.float32)
        if clip_stats:
            np.clip(obs, -10.0, 10.0, out=obs)
        return obs

    def _split_state_vec_to_obs_dict(
        self, state_vec: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Convert flat state vector [D] into legacy obs dict keyed by self._obs_keys.
        """
        x = np.asarray(state_vec, dtype=np.float32).reshape(-1)
        total = int(sum(int(self._obs_dims.get(k, 0)) for k in self._obs_keys))
        if x.shape[0] != total:
            raise RuntimeError(
                f"[ExRL] terminal state dim mismatch: got {x.shape[0]}, expected {total}"
            )
        out: Dict[str, np.ndarray] = {}
        st = 0
        for k in self._obs_keys:
            d = int(self._obs_dims.get(k, 0))
            if d <= 0:
                continue
            out[k] = x[st : st + d].astype(np.float32)
            st += d
        return out

    def _coerce_terminal_obs_dict(self, terminal_obs: Any) -> Dict[str, np.ndarray]:
        """
        Accept multiple terminal_obs formats and return legacy keyed dict:
        - {robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, object}
        - {"state": ...} where state may be [D], [T,D], [B,T,D], ...
        - {"obs": <nested dict above>}
        """
        if isinstance(terminal_obs, dict):
            # Legacy keyed obs dict already
            if any(k in terminal_obs for k in self._obs_keys):
                out = {}
                for k in self._obs_keys:
                    if k in terminal_obs:
                        v = np.asarray(terminal_obs[k], dtype=np.float32)
                        if v.ndim > 1:
                            v = v.reshape(-1, v.shape[-1])[-1]
                        out[k] = v.reshape(-1).astype(np.float32)
                if out:
                    return out

            # Nested under "obs"
            if "obs" in terminal_obs and isinstance(terminal_obs["obs"], dict):
                return self._coerce_terminal_obs_dict(terminal_obs["obs"])

            # DSRL/official format: dict(state=...)
            if "state" in terminal_obs:
                s = np.asarray(terminal_obs["state"], dtype=np.float32)
                if s.ndim == 1:
                    state_vec = s
                elif s.ndim == 2:
                    state_vec = s[-1]
                elif s.ndim >= 3:
                    state_vec = s.reshape(-1, s.shape[-1])[-1]
                else:
                    raise RuntimeError(
                        f"[ExRL] unsupported terminal state ndim={s.ndim}"
                    )
                return self._split_state_vec_to_obs_dict(state_vec)

        # Raw array fallback
        arr = np.asarray(terminal_obs, dtype=np.float32)
        if arr.ndim == 1:
            return self._split_state_vec_to_obs_dict(arr)
        if arr.ndim >= 2:
            return self._split_state_vec_to_obs_dict(arr.reshape(-1, arr.shape[-1])[-1])
        raise RuntimeError(
            f"[ExRL] unsupported terminal_obs type: {type(terminal_obs)}"
        )

    def get_terminal_rl_obs_samples(self, terminal_obs_dict, env_idx, num_samples):
        """
        Return M independently sampled terminal RL observations for timeout bootstrapping.
        """
        obs_dict = self._coerce_terminal_obs_dict(terminal_obs_dict)
        obs_batch_for_dp = {}
        valid_keys = list(self.algo_instance.obs_shapes.keys())
        for k in valid_keys:
            if k in obs_dict:
                val = np.asarray(obs_dict[k], dtype=np.float32)
                if val.ndim == 1:
                    val = val[None, :]
                elif val.ndim > 2:
                    val = val.reshape(-1, val.shape[-1])[-1][None, :]
                obs_batch_for_dp[k] = val
        if not obs_batch_for_dp:
            raise RuntimeError("[ExRL] empty terminal obs after coercion")
        samples = []
        for _ in range(max(1, int(num_samples))):
            chunk = self._get_batch_policy_plan(obs_batch_for_dp)[0]
            samples.append(self._make_rl_obs(int(env_idx), obs_dict, chunk))
        return np.stack(samples, axis=0).astype(np.float32)

    def get_terminal_rl_obs(self, terminal_obs_dict, env_idx):
        """
        Override legacy method for DSRL env-stack terminal_observation compatibility.
        """
        return self.get_terminal_rl_obs_samples(terminal_obs_dict, env_idx, 1)[0]
