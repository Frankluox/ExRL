import time
from typing import Dict, List, Optional

try:
    from multiprocessing.connection import wait as mp_wait
except Exception:  # pragma: no cover
    mp_wait = None


class AsyncEnvPool:
    """Per-env async step manager over SubprocVecEnv remotes."""

    def __init__(self, venv, poll_timeout_ms: int = 5):
        remotes = getattr(venv, "remotes", None)
        if remotes is None:
            raise RuntimeError("[AsyncEnvPool] venv.remotes is required (SubprocVecEnv expected)")
        if mp_wait is None:
            raise RuntimeError("[AsyncEnvPool] multiprocessing.connection.wait is unavailable")

        self.venv = venv
        self.remotes = remotes
        self.poll_timeout_sec = max(int(poll_timeout_ms), 1) / 1000.0
        self._fileno_to_idx = {}
        for i, r in enumerate(self.remotes):
            self._fileno_to_idx[r.fileno()] = i
        self._inflight = set()

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    def step_env(self, env_id: int, payload: Dict) -> None:
        if env_id in self._inflight:
            raise RuntimeError(f"[AsyncEnvPool] env {env_id} already inflight")
        self.remotes[env_id].send(("step", payload))
        self._inflight.add(env_id)

    def poll_ready(self, max_events: int = 0, timeout_ms: Optional[int] = None) -> List[Dict]:
        if not self._inflight:
            return []

        if max_events <= 0:
            max_events = len(self._inflight)

        timeout_sec = (
            self.poll_timeout_sec if timeout_ms is None else max(int(timeout_ms), 1) / 1000.0
        )
        wait_set = [self.remotes[i] for i in sorted(self._inflight)]
        ready_remotes = mp_wait(wait_set, timeout=timeout_sec)
        if not ready_remotes:
            return []

        out = []
        for r in ready_remotes[:max_events]:
            env_id = self._fileno_to_idx.get(r.fileno(), None)
            if env_id is None:
                continue
            payload = r.recv()
            obs, rew, done, info = self._parse_payload(payload)
            out.append(
                {
                    "env_id": env_id,
                    "obs": obs,
                    "reward": float(rew),
                    "done": bool(done),
                    "info": info,
                    "t_recv": time.perf_counter(),
                }
            )
            self._inflight.discard(env_id)

        # Keep VecEnv internal waiting flag consistent when we bypass step_wait.
        if getattr(self.venv, "waiting", False) and (len(self._inflight) == 0):
            self.venv.waiting = False

        return out

    @staticmethod
    def _parse_payload(payload):
        if isinstance(payload, tuple):
            if len(payload) >= 4:
                obs, rew, done, info = payload[0], payload[1], payload[2], payload[3]
                return obs, rew, done, info
        raise RuntimeError(f"[AsyncEnvPool] unexpected worker payload type={type(payload)}")
