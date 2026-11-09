"""Serve a frozen Piper π₀.₅ checkpoint from an OpenPI environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import threading

import numpy as np


class SeededPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.lock = threading.Lock()

    def infer(self, observation):
        observation = dict(observation)
        seed = observation.pop("exrl_seed", None)
        with self.lock:
            if seed is None:
                return self.policy.infer(observation)
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError("exrl_seed must be an unsigned 32-bit integer")
            import jax

            previous = self.policy._rng
            try:
                self.policy._rng = jax.random.key(seed)
                return self.policy.infer(observation)
            finally:
                self.policy._rng = previous


def make_config(asset_id, prompt, action_mode="state_delta"):
    from openpi.models import pi0_config
    from openpi.training import config as c
    from exrl.pretraining.piper import PiperData

    return c.TrainConfig(
        name="exrl_pi05_piper",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=PiperData(repo_id=asset_id, prompt=prompt, action_mode=action_mode),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--asset-id", help="Normalization asset directory inside checkpoint/assets")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--action-mode", choices=("state_delta", "absolute"), default="state_delta")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    asset_id = args.asset_id
    if asset_id is None:
        stats = list((checkpoint / "assets").glob("**/norm_stats.json"))
        if len(stats) != 1:
            raise ValueError(
                "Specify --asset-id when the checkpoint has multiple normalization assets"
            )
        asset_id = str(stats[0].parent.relative_to(checkpoint / "assets"))
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    policy = policy_config.create_trained_policy(
        make_config(asset_id, args.prompt, args.action_mode), checkpoint, default_prompt=args.prompt
    )
    WebsocketPolicyServer(
        SeededPolicy(policy),
        host="127.0.0.1",
        port=args.port,
        metadata=dict(
            family="pi05", robot="piper", horizon=50, action_mode=args.action_mode, exrl_seeded=True
        ),
    ).serve_forever()


if __name__ == "__main__":
    main()
