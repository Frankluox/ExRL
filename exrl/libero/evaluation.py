"""Evaluate a saved LIBERO execution-length learner."""

from pathlib import Path
import pickle

from exrl.config import merge
from exrl.pixel.training import make_agent, restore
from exrl.libero.train import Environment, evaluate


def evaluate_checkpoint(checkpoint, config, episodes):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    payload = pickle.loads(Path(checkpoint).read_bytes())
    config = merge(config, config.get("training", {}))
    for key in ("base_policy", "method", "chunk_size", "learner"):
        if config[key] != payload["config"][key]:
            raise ValueError(f"Evaluation configuration differs from checkpoint: {key}")
    env = Environment(config, config["checkpoint"], config["output"], config["seed"] + 10000)
    try:
        agent = make_agent(config, env.reset())
        restore(payload, agent)
        return evaluate(env, agent, config, episodes)
    finally:
        env.close()
