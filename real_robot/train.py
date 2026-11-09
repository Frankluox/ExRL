"""Train or evaluate ExRL-Q using robot and base-policy adapters."""

from __future__ import annotations

import argparse
import pickle
import json
from pathlib import Path

import numpy as np

from exrl.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--robot", required=True, help="Robot adapter YAML")
    parser.add_argument(
        "--normalization",
        required=True,
        help="State/action mean-std JSON or OpenPI norm_stats.json",
    )
    parser.add_argument("--policy", required=True, help="Base-policy adapter YAML")
    parser.add_argument("--output", default="outputs/real_robot")
    parser.add_argument("--resume")
    parser.add_argument("--evaluate", type=int, metavar="EPISODES")
    parser.add_argument("--fixed-k", type=int, help="Evaluate a fixed length instead of the critic")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check inputs and policy inference without publishing actions",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config["learner"]["num_cameras"] = len(config["camera_keys"])
    from real_robot.observations import load_normalization, critic_observation
    from real_robot.interfaces import load_component
    from real_robot.training import make_agent, train, restore, stack_rows

    stats = load_normalization(args.normalization)
    robot = load_component(args.robot, config)
    try:
        policy = load_component(args.policy, config)
        frame = robot.observe()
        chunk = policy.predict(frame)
        observation = critic_observation(
            frame,
            chunk,
            stats,
            config["image_size"],
            config["action_mode"],
            config["camera_keys"],
            config.get("action_state_indices", []),
        )
        if args.check:
            print("Camera, feedback, normalization and policy inference are ready.")
            print({k: v.shape for k, v in observation.items()})
            return
        agent = make_agent(config, observation)
        from real_robot.operator import Operator
        from real_robot.collector import Collector

        collector = Collector(robot, policy, Operator(config["stages"]), config, stats)
        if args.evaluate:
            if args.fixed_k is None:
                if not args.resume:
                    raise ValueError("Critic evaluation requires --resume")
                restore(pickle.loads(Path(args.resume).read_bytes()), agent)
            results = []
            for _ in range(args.evaluate):
                _, result = collector.rollout(
                    lambda obs: (
                        args.fixed_k
                        if args.fixed_k is not None
                        else int(
                            agent.eval_q_action_ids(stack_rows([obs]), critic_reduction="mean")[0]
                        )
                        + 1
                    )
                )
                results.append(result)
                output = Path(args.output)
                output.mkdir(parents=True, exist_ok=True)
                (output / "evaluation.json").write_text(json.dumps(results, indent=2))
                print(f"Successes: {sum(r['success'] for r in results)}/{len(results)}", flush=True)
        else:
            train(collector, agent, config, args.output, args.resume)
    finally:
        if "policy" in locals() and hasattr(policy, "close"):
            policy.close()
        robot.close()


if __name__ == "__main__":
    main()
