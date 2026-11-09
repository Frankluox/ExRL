"""Small RPC client for the frozen Cosmos LIBERO policy server."""

from __future__ import annotations

import pickle
import socket
import struct
import time
from typing import Dict, Optional

import numpy as np


_HEADER = struct.Struct("!Q")


def _recvall(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = int(size)
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Cosmos policy server closed the socket")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _request(sock: socket.socket, payload: Dict) -> Dict:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(data)))
    sock.sendall(data)
    (size,) = _HEADER.unpack(_recvall(sock, _HEADER.size))
    response = pickle.loads(_recvall(sock, size))
    if "error" in response:
        raise RuntimeError(response["error"])
    return response


class CosmosPolicyClient:
    """Length-prefixed pickle RPC client with bounded startup retries."""

    def __init__(
        self,
        host="127.0.0.1",
        port=19057,
        timeout=600.0,
        *,
        wait_for_server=True,
        startup_timeout=600.0,
        retry_interval=2.0,
    ):
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)
        self.startup_timeout = float(startup_timeout)
        self.retry_interval = float(retry_interval)
        self._metadata: Optional[Dict] = None
        if wait_for_server:
            self._metadata = self._request_with_retry({"method": "metadata"})

    def _request_once(self, payload: Dict) -> Dict:
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            sock.settimeout(self.timeout)
            return _request(sock, payload)

    def _request_with_retry(self, payload: Dict) -> Dict:
        deadline = time.monotonic() + self.startup_timeout
        last_error = None
        while time.monotonic() < deadline:
            try:
                return self._request_once(payload)
            except (ConnectionRefusedError, TimeoutError, OSError) as exc:
                last_error = exc
                time.sleep(self.retry_interval)
        raise TimeoutError(
            f"Cosmos policy server {self.host}:{self.port} was not ready within "
            f"{self.startup_timeout:.1f}s"
        ) from last_error

    def get_server_metadata(self) -> Dict:
        if self._metadata is None:
            self._metadata = self._request_with_retry({"method": "metadata"})
        return dict(self._metadata)

    @staticmethod
    def _decode(response, prefix="actions") -> np.ndarray:
        if f"{prefix}_bytes" in response:
            value = (
                np.frombuffer(
                    response[f"{prefix}_bytes"], dtype=np.dtype(response[f"{prefix}_dtype"])
                )
                .reshape(tuple(response[f"{prefix}_shape"]))
                .copy()
            )
        else:
            value = np.asarray(response[prefix], dtype=np.float32)
        return np.asarray(value, dtype=np.float32)

    def infer(self, obs: Dict) -> Dict:
        started = time.perf_counter()
        response = self._request_once({"method": "infer", "obs": obs})
        timing = dict(response.get("policy_timing", {}))
        timing["roundtrip_sec"] = float(time.perf_counter() - started)
        return {
            "actions": self._decode(response),
            "policy_timing": timing,
            "fast_decode_failed": False,
        }

    def infer_batch(self, observations, padded_batch_size=0):
        observations = list(observations)
        if not observations:
            return []
        started = time.perf_counter()
        response = self._request_once(
            {
                "method": "infer_batch",
                "observations": observations,
                "padded_batch_size": int(padded_batch_size),
            }
        )
        timing = dict(response.get("policy_timing", {}))
        timing["roundtrip_sec"] = float(time.perf_counter() - started)
        actions = self._decode(response, "actions_batch")
        if actions.shape != (len(observations), 50, 7):
            raise ValueError(
                f"expected Cosmos batch [{len(observations)},50,7], got {actions.shape}"
            )
        return [
            {"actions": action.copy(), "policy_timing": timing, "fast_decode_failed": False}
            for action in actions
        ]

    def reset(self):
        pass

    def close(self):
        pass
