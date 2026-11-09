import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class InferRequest:
    env_id: int
    obs_dict: Dict[str, np.ndarray]
    rollout_id: int
    policy_version: int
    t_submit: float


@dataclass
class InferResult:
    env_id: int
    chunk: np.ndarray
    rollout_id: int
    policy_version: int
    t_submit: float
    t_ready: float
    queue_wait_sec: float
    compute_sec: float
    batch_size: int


class AsyncInferPool:
    """
    Single dispatcher + micro-batch inference queue.

    We intentionally keep one dispatcher thread so model usage remains safe,
    while allowing request aggregation into larger micro-batches.
    """

    def __init__(
        self,
        infer_batch_fn: Callable[[Dict[str, np.ndarray]], np.ndarray],
        microbatch_max: int = 64,
        microbatch_timeout_ms: int = 3,
        request_queue_size: int = 4096,
        result_queue_size: int = 4096,
        name: str = "infer_pool",
    ):
        self.infer_batch_fn = infer_batch_fn
        self.microbatch_max = max(int(microbatch_max), 1)
        self.microbatch_timeout_ms = max(int(microbatch_timeout_ms), 1)
        self.req_q: "queue.Queue[InferRequest]" = queue.Queue(
            maxsize=max(int(request_queue_size), 1)
        )
        self.res_q: "queue.Queue[InferResult]" = queue.Queue(maxsize=max(int(result_queue_size), 1))
        self.name = name

        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        self._perf_lock = threading.Lock()
        self._perf = {
            "infer_queue_wait_sec": 0.0,
            "infer_compute_sec": 0.0,
            "infer_batch_size_sum": 0.0,
            "infer_batch_count": 0.0,
            "infer_req_count": 0.0,
            "infer_req_drop_count": 0.0,
            "infer_stale_result_drop_count": 0.0,
            "ready_queue_depth_sum": 0.0,
            "ready_queue_depth_count": 0.0,
        }

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, name=self.name, daemon=True)
        self._thread.start()

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
        self._thread = None

    def submit(self, req: InferRequest) -> bool:
        try:
            self.req_q.put_nowait(req)
            return True
        except queue.Full:
            with self._perf_lock:
                self._perf["infer_req_drop_count"] += 1.0
            return False

    def poll_results(self, max_items: int = 0, timeout_sec: float = 0.0) -> List[InferResult]:
        out: List[InferResult] = []
        if max_items <= 0:
            max_items = 1 << 30

        deadline = time.perf_counter() + max(timeout_sec, 0.0)
        while len(out) < max_items:
            now = time.perf_counter()
            remaining = max(0.0, deadline - now)
            wait_t = remaining if (timeout_sec > 0 and len(out) == 0) else 0.0
            try:
                item = self.res_q.get(timeout=wait_t)
            except queue.Empty:
                break
            out.append(item)
            if timeout_sec > 0 and now >= deadline:
                break
        return out

    def infer_many(self, reqs: List[InferRequest], timeout_sec: float = 5.0) -> List[InferResult]:
        if not reqs:
            return []
        for r in reqs:
            ok = self.submit(r)
            if not ok:
                raise RuntimeError("[AsyncInferPool] request queue is full")

        pending = {r.env_id: r for r in reqs}
        out: Dict[int, InferResult] = {}
        deadline = time.perf_counter() + max(timeout_sec, 0.1)
        while pending:
            now = time.perf_counter()
            if now > deadline:
                raise TimeoutError(
                    f"[AsyncInferPool] timeout waiting results, pending={sorted(pending.keys())}"
                )
            got = self.poll_results(max_items=len(pending), timeout_sec=0.05)
            for r in got:
                req = pending.get(r.env_id)
                if req is None:
                    # Unknown/late result from previous calls.
                    with self._perf_lock:
                        self._perf["infer_stale_result_drop_count"] += 1.0
                    continue
                if (r.rollout_id != req.rollout_id) or (r.policy_version != req.policy_version):
                    # Same env_id but stale generation; drop explicitly.
                    with self._perf_lock:
                        self._perf["infer_stale_result_drop_count"] += 1.0
                    continue
                out[r.env_id] = r
                del pending[r.env_id]
        return [out[req.env_id] for req in reqs]

    def pull_perf_stats(self, reset: bool = True) -> Dict[str, float]:
        with self._perf_lock:
            d = dict(self._perf)
            if reset:
                for k in self._perf:
                    self._perf[k] = 0.0
        batch_cnt = max(d.get("infer_batch_count", 0.0), 1.0)
        depth_cnt = max(d.get("ready_queue_depth_count", 0.0), 1.0)
        req_cnt = max(d.get("infer_req_count", 0.0), 1.0)
        d["infer_batch_size_avg"] = d.get("infer_batch_size_sum", 0.0) / batch_cnt
        d["ready_queue_depth_avg"] = d.get("ready_queue_depth_sum", 0.0) / depth_cnt
        d["infer_queue_wait_sec_per_req"] = d.get("infer_queue_wait_sec", 0.0) / req_cnt
        d["infer_compute_sec_per_batch"] = d.get("infer_compute_sec", 0.0) / batch_cnt
        return d

    def _loop(self) -> None:
        timeout_sec = self.microbatch_timeout_ms / 1000.0
        while not self._stop_evt.is_set():
            try:
                first = self.req_q.get(timeout=0.1)
            except queue.Empty:
                continue

            batch = [first]
            t_deadline = time.perf_counter() + timeout_sec
            while len(batch) < self.microbatch_max:
                remain = t_deadline - time.perf_counter()
                if remain <= 0:
                    break
                try:
                    item = self.req_q.get(timeout=remain)
                except queue.Empty:
                    break
                batch.append(item)

            t_compute_start = time.perf_counter()
            obs_batch = self._stack_obs(batch)
            chunks = self.infer_batch_fn(obs_batch)
            t_compute_end = time.perf_counter()

            bsz = len(batch)
            compute_sec = t_compute_end - t_compute_start
            with self._perf_lock:
                self._perf["infer_batch_count"] += 1.0
                self._perf["infer_batch_size_sum"] += float(bsz)
                self._perf["infer_compute_sec"] += float(compute_sec)

            for i, req in enumerate(batch):
                t_ready = time.perf_counter()
                queue_wait = max(0.0, t_compute_start - req.t_submit)
                res = InferResult(
                    env_id=req.env_id,
                    chunk=np.asarray(chunks[i], dtype=np.float32),
                    rollout_id=req.rollout_id,
                    policy_version=req.policy_version,
                    t_submit=req.t_submit,
                    t_ready=t_ready,
                    queue_wait_sec=queue_wait,
                    compute_sec=compute_sec,
                    batch_size=bsz,
                )
                self.res_q.put(res)
                with self._perf_lock:
                    self._perf["infer_req_count"] += 1.0
                    self._perf["infer_queue_wait_sec"] += float(queue_wait)
                    self._perf["ready_queue_depth_sum"] += float(self.res_q.qsize())
                    self._perf["ready_queue_depth_count"] += 1.0

    @staticmethod
    def _stack_obs(reqs: List[InferRequest]) -> Dict[str, np.ndarray]:
        keys = list(reqs[0].obs_dict.keys())
        out: Dict[str, np.ndarray] = {}
        for k in keys:
            out[k] = np.stack([np.asarray(r.obs_dict[k], dtype=np.float32) for r in reqs], axis=0)
        return out
