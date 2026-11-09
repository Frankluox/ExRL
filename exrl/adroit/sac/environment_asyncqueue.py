import os
import time
from typing import Dict, List, Optional

import numpy as np

from exrl.adroit.sac.chunk_env import BatchedAdaptiveWrapper
from exrl.adroit.sac.env_pool import AsyncEnvPool
from exrl.adroit.sac.infer_pool import AsyncInferPool, InferRequest


class BatchedAdaptiveWrapperAsyncQueue(BatchedAdaptiveWrapper):
    """
    Async-queue runtime for per-env stepping + micro-batch DP inference.

    Notes:
    - Strict on-policy is enforced at collector level (no actor-learner overlap).
    - This wrapper exposes per-env async primitives used by RolloutAsyncCollector.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.infer_microbatch_max = int(os.environ.get("OPT_ASYNCQUEUE_INFER_MICROBATCH_MAX", "64"))
        self.infer_microbatch_timeout_ms = int(
            os.environ.get("OPT_ASYNCQUEUE_INFER_MICROBATCH_TIMEOUT_MS", "3")
        )
        self.infer_wait_timeout_sec = float(
            os.environ.get("OPT_ASYNCQUEUE_INFER_WAIT_TIMEOUT_SEC", "5.0")
        )
        self.infer_queue_size = int(os.environ.get("OPT_ASYNCQUEUE_INFER_QUEUE_SIZE", "4096"))
        self.infer_result_queue_size = int(
            os.environ.get("OPT_ASYNCQUEUE_INFER_RESULT_QUEUE_SIZE", "4096")
        )
        self.env_poll_timeout_ms = int(os.environ.get("OPT_ASYNCQUEUE_ENV_POLL_TIMEOUT_MS", "5"))
        self.ready_queue_size = int(os.environ.get("OPT_ASYNCQUEUE_READY_QUEUE_SIZE", "8192"))
        self.max_inflight_per_env = int(os.environ.get("OPT_ASYNCQUEUE_MAX_INFLIGHT_PER_ENV", "1"))
        self.async_rollout_mode = os.environ.get("OPT_ASYNCQUEUE_ROLLOUT_MODE", "strict_on_policy")
        self.log_full_chunk_in_pkl = os.environ.get("OPT_LOG_FULL_CHUNK_IN_PKL", "0") == "1"

        allow_multi_gpu = os.environ.get("OPT_ASYNCQUEUE_ALLOW_MULTI_GPU_INFER", "0") == "1"
        if (len(self.dp_models) > 1) and (not allow_multi_gpu):
            print(
                "[ASYNCQUEUE][SAFEGUARD] multi-GPU DP inference is disabled by default "
                "for stability in async runtime. Falling back to first DP device. "
                "Set OPT_ASYNCQUEUE_ALLOW_MULTI_GPU_INFER=1 to override."
            )
            self.dp_models = [self.dp_models[0]]
            self.dp_devices = [self.dp_devices[0]]
            self.dp_parallel_workers = 1

        self._env_pool = AsyncEnvPool(self.venv, poll_timeout_ms=self.env_poll_timeout_ms)
        # Multi-GPU + background infer thread can introduce unstable behavior in async mode.
        # Use inline inference as the safe default when multiple DP replicas are active.
        force_threaded = os.environ.get("OPT_ASYNCQUEUE_FORCE_THREADED_INFER", "0") == "1"
        self._infer_inline = (len(self.dp_models) > 1) and (not force_threaded)
        self._infer_pool = None
        if self._infer_inline:
            print("[ASYNCQUEUE] Using inline DP inference (multi-GPU safe mode).")
        else:
            self._infer_pool = AsyncInferPool(
                infer_batch_fn=self._infer_batch_sync,
                microbatch_max=self.infer_microbatch_max,
                microbatch_timeout_ms=self.infer_microbatch_timeout_ms,
                request_queue_size=self.infer_queue_size,
                result_queue_size=self.infer_result_queue_size,
                name="infer_pool_asyncqueue",
            )
            self._infer_pool.start()

        self._aq_policy_obs_cache: Optional[np.ndarray] = None
        self._aq_last_requested_actions = np.zeros(self.num_envs, dtype=np.int32)
        self._aq_env_busy_start: Dict[int, float] = {}
        self._aq_perf = {
            "env_busy_sec": 0.0,
            "env_idle_sec": 0.0,
            "infer_queue_wait_sec": 0.0,
            "infer_compute_sec": 0.0,
            "infer_batch_size_sum": 0.0,
            "infer_batch_count": 0.0,
        }

    def close(self):
        try:
            if self._infer_pool is not None:
                self._infer_pool.stop()
        except Exception:
            pass
        return super().close()

    def reset(self):
        obs = super().reset()
        self._aq_policy_obs_cache = np.asarray(obs, dtype=np.float32)
        self._aq_last_requested_actions.fill(0)
        self._aq_env_busy_start.clear()
        for k in self._aq_perf:
            self._aq_perf[k] = 0.0
        return obs

    def _infer_batch_sync(self, obs_batch: Dict[str, np.ndarray]) -> np.ndarray:
        # Reuse parent multi-GPU inference implementation.
        return self._get_batch_policy_plan(obs_batch)

    def _obs_dict_for_env(self, env_id: int) -> Dict[str, np.ndarray]:
        out = {}
        for k, v in self.current_obs_dicts.items():
            out[k] = np.asarray(v[env_id], dtype=np.float32)
        return out

    def _assign_obs_for_env(self, env_id: int, obs_dict: Dict):
        for k in self.current_obs_dicts.keys():
            if k in obs_dict:
                self.current_obs_dicts[k][env_id] = np.asarray(obs_dict[k], dtype=np.float32)

    def get_policy_obs(self, env_ids: np.ndarray) -> np.ndarray:
        if self._aq_policy_obs_cache is None:
            raise RuntimeError("[AsyncQueue] reset() must be called before get_policy_obs()")
        return np.asarray(self._aq_policy_obs_cache[env_ids], dtype=np.float32)

    def dispatch_policy_actions(self, env_ids: np.ndarray, actions: np.ndarray):
        if len(env_ids) != len(actions):
            raise ValueError("env_ids and actions size mismatch")
        if self.max_inflight_per_env != 1:
            raise RuntimeError(
                "[AsyncQueue] only max_inflight_per_env=1 is supported in strict_on_policy mode"
            )

        max_k = (
            int(self.last_chunks.shape[1])
            if isinstance(self.last_chunks, np.ndarray)
            else int(self.max_chunk_len)
        )
        action_ids = np.asarray(actions, dtype=np.int64)
        requested = np.clip(action_ids + self.min_chunk_len, self.min_chunk_len, max_k)

        send_dtype = np.float16 if self.ipc_chunk_dtype == "fp16" else np.float32
        now = time.perf_counter()
        for idx, env_id in enumerate(env_ids.tolist()):
            env_id = int(env_id)
            k_i = int(requested[idx])
            chunk_full = np.asarray(self.last_chunks[env_id], dtype=send_dtype)
            if self.ipc_send_mode == "executed_only":
                chunk_send = chunk_full[:k_i] if k_i > 0 else chunk_full[:0]
            else:
                chunk_send = chunk_full
            payload = {"chunk": chunk_send, "k": k_i}
            if self.log_full_chunk_in_pkl:
                payload["full_chunk"] = np.asarray(self.last_chunks[env_id], dtype=np.float32)
            self._env_pool.step_env(env_id, payload)
            self._aq_last_requested_actions[env_id] = k_i
            self._aq_env_busy_start[env_id] = now

    def _refresh_next_policy_obs(
        self, ready_env_ids: List[int], rollout_id: int = 0, policy_version: int = 0
    ):
        if not ready_env_ids:
            return

        if self._infer_inline:
            obs_batch = {}
            first_env = int(ready_env_ids[0])
            for k in self.current_obs_dicts.keys():
                sample = np.asarray(self.current_obs_dicts[k][first_env], dtype=np.float32)
                obs_batch[k] = np.zeros((len(ready_env_ids),) + sample.shape, dtype=np.float32)
            for row, env_id in enumerate(ready_env_ids):
                env_id = int(env_id)
                for k in obs_batch.keys():
                    obs_batch[k][row] = np.asarray(
                        self.current_obs_dicts[k][env_id], dtype=np.float32
                    )

            t0 = time.perf_counter()
            chunks = self._infer_batch_sync(obs_batch)
            infer_sec = time.perf_counter() - t0
            self._aq_perf["infer_batch_count"] += 1.0
            self._aq_perf["infer_batch_size_sum"] += float(len(ready_env_ids))
            self._aq_perf["infer_compute_sec"] += float(infer_sec)
            for row, env_id in enumerate(ready_env_ids):
                self.last_chunks[int(env_id)] = np.asarray(chunks[row], dtype=np.float32)
        else:
            reqs = []
            t_submit = time.perf_counter()
            for env_id in ready_env_ids:
                reqs.append(
                    InferRequest(
                        env_id=int(env_id),
                        obs_dict=self._obs_dict_for_env(int(env_id)),
                        rollout_id=int(rollout_id),
                        policy_version=int(policy_version),
                        t_submit=t_submit,
                    )
                )

            infer_results = self._infer_pool.infer_many(
                reqs, timeout_sec=self.infer_wait_timeout_sec
            )
            batch_size = len(infer_results)
            self._aq_perf["infer_batch_count"] += 1.0
            self._aq_perf["infer_batch_size_sum"] += float(batch_size)

            for r in infer_results:
                env_id = int(r.env_id)
                self.last_chunks[env_id] = np.asarray(r.chunk, dtype=np.float32)
                self._aq_perf["infer_queue_wait_sec"] += float(r.queue_wait_sec)
                self._aq_perf["infer_compute_sec"] += float(r.compute_sec)

        if self._aq_policy_obs_cache is None:
            self._aq_policy_obs_cache = np.zeros(
                (self.num_envs, self.observation_space.shape[0]), dtype=np.float32
            )

        for env_id in ready_env_ids:
            env_id = int(env_id)
            obs_dict = self._obs_dict_for_env(env_id)
            self._aq_policy_obs_cache[env_id] = self._make_rl_obs(
                env_id, obs_dict, self.last_chunks[env_id]
            )

    def poll_step_results(
        self,
        min_ready: int = 1,
        timeout_ms: Optional[int] = None,
        max_events: int = 0,
        rollout_id: int = 0,
        policy_version: int = 0,
    ) -> List[Dict]:
        min_ready = max(int(min_ready), 1)
        t0 = time.perf_counter()
        results: List[Dict] = []

        take_n = int(max_events) if int(max_events) > 0 else self.num_envs
        while len(results) < min_ready:
            got = self._env_pool.poll_ready(max_events=take_n, timeout_ms=timeout_ms)
            if not got:
                self._aq_perf["env_idle_sec"] += max(0.0, time.perf_counter() - t0)
                break
            results.extend(got)
            if len(results) >= min_ready:
                break

        if not results:
            return []

        ready_env_ids: List[int] = []
        out: List[Dict] = []
        now = time.perf_counter()
        for item in results:
            env_id = int(item["env_id"])
            obs = item["obs"]
            rew = float(item["reward"])
            done = bool(item["done"])
            info = item["info"]

            t_sent = self._aq_env_busy_start.pop(env_id, None)
            if t_sent is not None:
                self._aq_perf["env_busy_sec"] += max(0.0, now - t_sent)

            k_executed = int(info.get("exec_len", 0))
            if k_executed > 0:
                actions_taken = self.last_chunks[env_id][:k_executed]
                self._update_action_history(env_id, actions_taken)
            if done:
                self.action_histories[env_id] = 0.0

            if int(self._aq_last_requested_actions[env_id]) == 0:
                rew = float(self.resample_penalty)

            self._assign_obs_for_env(env_id, obs)
            ready_env_ids.append(env_id)
            out.append(
                {
                    "env_id": env_id,
                    "reward": rew,
                    "done": done,
                    "info": info,
                }
            )

        self._refresh_next_policy_obs(
            ready_env_ids, rollout_id=rollout_id, policy_version=policy_version
        )
        for item in out:
            env_id = item["env_id"]
            item["next_obs"] = np.asarray(self._aq_policy_obs_cache[env_id], dtype=np.float32)

        return out

    def pull_perf_stats(self, reset: bool = True):
        out = dict(self._perf_acc)

        if self._infer_pool is not None:
            infer_stats = self._infer_pool.pull_perf_stats(reset=reset)
            out.update(
                {
                    "infer_queue_wait_sec": float(infer_stats.get("infer_queue_wait_sec", 0.0)),
                    "infer_compute_sec": float(infer_stats.get("infer_compute_sec", 0.0)),
                    "ready_queue_depth_avg": float(infer_stats.get("ready_queue_depth_avg", 0.0)),
                    "infer_batch_size_avg": float(infer_stats.get("infer_batch_size_avg", 0.0)),
                }
            )
        else:
            batch_count = max(float(self._aq_perf.get("infer_batch_count", 0.0)), 1.0)
            batch_sum = float(self._aq_perf.get("infer_batch_size_sum", 0.0))
            out.update(
                {
                    "infer_queue_wait_sec": float(self._aq_perf.get("infer_queue_wait_sec", 0.0)),
                    "infer_compute_sec": float(self._aq_perf.get("infer_compute_sec", 0.0)),
                    "ready_queue_depth_avg": 0.0,
                    "infer_batch_size_avg": batch_sum / batch_count,
                }
            )

        out["env_busy_sec"] = float(self._aq_perf.get("env_busy_sec", 0.0))
        out["env_idle_sec"] = float(self._aq_perf.get("env_idle_sec", 0.0))

        if reset:
            for k in self._perf_acc:
                self._perf_acc[k] = 0.0 if isinstance(self._perf_acc[k], float) else 0
            for k in self._aq_perf:
                self._aq_perf[k] = 0.0

        return out
