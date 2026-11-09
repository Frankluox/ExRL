"""Episode-boundary updates for real-robot adaptation."""

from pathlib import Path
import pickle
import numpy as np
from exrl.pixel.training import Replay, make_agent, restore, save, stack_rows
from real_robot.interfaces import policy_rng_state, restore_policy_rng


def train(collector, agent, config, output, resume=None):
    import json

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["seed"])
    replay = Replay(config["replay_capacity"])
    counters = dict(episodes=0, transitions=0, updates=0, warmup_complete=False)
    if resume:
        payload = pickle.loads(Path(resume).read_bytes())
        for key in (
            "chunk_size",
            "discount",
            "stages",
            "control_hz",
            "episode_seconds",
            "camera_keys",
            "action_state_indices",
            "image_size",
            "action_mode",
            "learner",
        ):
            if payload["config"].get(key) != config.get(key):
                raise ValueError(f"Cannot resume with a changed {key}")
        restore(payload, agent)
        replay, counters = payload["replay"], payload["counters"]
        rng.bit_generator.state = payload["rng"]
        if payload.get("policy_rng") is not None:
            restore_policy_rng(collector.policy, payload["policy_rng"])

    def update(count):
        from flax.core import freeze

        for _ in range(int(count)):
            agent.update(
                freeze(replay.sample(config["batch_size"], rng)),
                update_actor=False,
                update_alpha=False,
            )
            counters["updates"] += 1

    try:
        while counters["episodes"] < config["episodes"]:
            warming = not counters["warmup_complete"]

            def choose(obs):
                if warming:
                    return rng.integers(1, config["chunk_size"] + 1)
                return (
                    int(agent.eval_q_action_ids(stack_rows([obs]), critic_reduction="mean")[0]) + 1
                )

            transitions, result = collector.rollout(choose)
            replay.extend(transitions)
            counters["episodes"] += 1
            counters["transitions"] += len(transitions)
            if warming:
                if (
                    counters["episodes"] >= config["warmup_episodes"]
                    and len(replay.rows) >= config["warmup_transitions"]
                ):
                    update(config["warmup_updates"])
                    counters["warmup_complete"] = True
            else:
                update(config.get("updates_per_episode") or len(transitions) * config["utd"])
            result.update(counters)
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print(result, flush=True)
            save(
                output / "latest.pkl",
                agent,
                replay,
                counters,
                rng,
                config,
                policy_rng=policy_rng_state(collector.policy),
            )
    finally:
        save(
            output / "latest.pkl",
            agent,
            replay,
            counters,
            rng,
            config,
            policy_rng=policy_rng_state(collector.policy),
        )
