"""Serve a frozen LIBERO checkpoint from a separate OpenPI environment."""

import argparse
from exrl.libero.policies import load_openpi_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("pi0", "pi05", "pi0fast"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    policy, _ = load_openpi_policy(args.family, args.checkpoint)

    class Policy:
        def infer(self, observation, **kwargs):
            observation = dict(observation)
            noise = observation.pop("exrl_noise", None)
            if noise is not None:
                kwargs["noise"] = noise
            return policy.infer(observation, **kwargs)

    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    WebsocketPolicyServer(
        Policy(), host=args.host, port=args.port, metadata=dict(family=args.family, horizon=50)
    ).serve_forever()


if __name__ == "__main__":
    main()
