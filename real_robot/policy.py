"""Optional Piper / OpenPI policy adapter."""

from __future__ import annotations

import numpy as np


class PolicyClient:
    def __init__(
        self,
        host="localhost",
        port=8000,
        prompt="",
        chunk_size=50,
        seed=0,
        action_mode="state_delta",
    ):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(host=host, port=int(port))
        self.prompt, self.chunk_size = prompt, int(chunk_size)
        metadata = self.client.get_server_metadata()
        if metadata.get("horizon") != self.chunk_size or not metadata.get("exrl_seeded"):
            raise ValueError("Start the matching real_robot.serve_policy server before training")
        if metadata.get("action_mode") != action_mode:
            raise ValueError("The policy server and critic must use the same action representation")
        self.rng = np.random.default_rng(seed)

    def rng_state(self):
        return self.rng.bit_generator.state

    def restore_rng(self, state):
        self.rng.bit_generator.state = state

    def predict(self, frame):
        result = self.client.infer(
            {
                "observation/image": frame["image"],
                "observation/wrist_image": frame["wrist_image"],
                "observation/state": frame["state"],
                "prompt": self.prompt,
                "exrl_seed": int(self.rng.integers(0, 2**32)),
            }
        )
        actions = np.asarray(result["actions"], dtype=np.float32)
        if (
            actions.ndim != 2
            or actions.shape[0] < self.chunk_size
            or actions.shape[1] < 7
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Policy server returned an invalid action chunk")
        return actions[: self.chunk_size, :7].copy()


def make_piper_policy(options, task):
    return PolicyClient(
        host=options.get("host", "localhost"),
        port=options.get("port", 8000),
        prompt=task["prompt"],
        chunk_size=task["chunk_size"],
        seed=task["seed"],
        action_mode=task["action_mode"],
    )
