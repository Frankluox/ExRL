"""ExRL-Q and discrete SAC with frozen image-conditioned base policies."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import pickle
import random

import numpy as np

from exrl.pixel.training import Replay, make_agent, restore, save, stack_rows


class Environment:
    def __init__(self, config, checkpoint, output, seed, policy_bundle=None):
        self.config = config
        family = config["base_policy"]
        runtime = importlib.import_module(f"baselines.resfit.libero.{family}.runtime")
        self.reference = importlib.import_module(f"baselines.resfit.libero.{family}.reference")
        runtime.configure_runtime(None, Path(output) / "libero_config")
        kwargs = dict(config.get("policy_server", {}))
        policy, stats = policy_bundle or runtime.load_policy(checkpoint, **kwargs)
        if policy_bundle is None and hasattr(policy, "_rng"):
            import jax

            policy._rng = jax.random.key(seed)
        self.policy_bundle = (policy, stats)
        self.owns_policy = policy_bundle is None
        self.settings = runtime.variant(stats)
        self.settings.max_timesteps = config["horizon"]
        self.settings.discount = config["discount"]
        cls = getattr(
            runtime,
            {
                "pi0": "Pi0Planner",
                "pi05": "Pi05Planner",
                "pi0fast": "Pi0FastPlanner",
                "cosmos": "CosmosPlanner",
            }[family],
        )
        self.planner = cls(policy, self.settings)
        self.env = runtime.make_env(seed)
        self.steps = 0

    def encode(self, raw, chunk):
        return self.reference.build_aac_observation(raw, chunk, [], self.settings)

    def reset(self):
        self.steps = 0
        self.raw = self.env.reset()
        self.chunk = self.planner.plan(self.raw)
        self.obs = self.encode(self.raw, self.chunk)
        return self.obs

    def step(self, k):
        if not 1 <= k <= self.config["chunk_size"]:
            raise ValueError("Invalid execution length")
        if self.steps >= self.config["horizon"]:
            raise RuntimeError("Reset the environment after the episode ends")
        previous = self.obs
        reward, duration, terminal, success = 0.0, 0, False, False
        for action in self.chunk[: min(k, self.config["horizon"] - self.steps)]:
            raw, _, done, info = self.env.step(action.tolist())
            success = bool(self.env.check_success())
            reward += self.config["discount"] ** duration * (0.0 if success else -1.0)
            duration += 1
            self.steps += 1
            terminal = success or bool(done)
            if terminal:
                break
        timeout = self.steps >= self.config["horizon"]
        self.raw = raw
        self.chunk = np.zeros_like(self.chunk) if terminal else self.planner.plan(raw)
        self.obs = self.encode(raw, self.chunk)
        row = dict(
            observations=previous,
            next_observations=self.obs,
            actions=np.int32(k - 1),
            rewards=np.float32(reward),
            masks=np.float32(not terminal),
            durations=np.float32(duration),
            discount=np.float32(self.config["discount"] ** duration),
        )
        return row, terminal or timeout, success

    def close(self):
        self.env.close()
        close = getattr(self.planner.policy, "close", None)
        if close and self.owns_policy:
            close()


def choose(agent, observation, method, deterministic=False):
    batch = stack_rows([observation])
    if method == "exrl_q":
        return int(agent.eval_q_action_ids(batch, critic_reduction="mean")[0]) + 1
    return (
        int((agent.eval_action_ids(batch) if deterministic else agent.sample_action_ids(batch))[0])
        + 1
    )


def evaluate(environment, agent, config, episodes):
    """Evaluation never advances the learner's or base policy's RNG stream."""
    policy = environment.planner.policy
    agent_rng = agent._rng
    policy_rng = getattr(policy, "_rng", None)
    np_state, py_state = np.random.get_state(), random.getstate()
    successes = 0
    try:
        for _ in range(episodes):
            obs, done = environment.reset(), False
            while not done:
                row, done, success = environment.step(
                    choose(agent, obs, config["method"], deterministic=True)
                )
                obs = row["next_observations"]
            successes += success
    finally:
        agent._rng = agent_rng
        if policy_rng is not None:
            policy._rng = policy_rng
        np.random.set_state(np_state)
        random.setstate(py_state)
    return dict(successes=successes, episodes=episodes, success_rate=successes / episodes)


def run(config, cli):
    from flax.core import freeze
    from exrl.config import merge

    config = merge(config, config.get("training", {}))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    env = Environment(config, config["checkpoint"], output, config["seed"])
    evaluation = None
    try:
        observation = env.reset()
        agent = make_agent(config, observation)
        if cli.check:
            print("Environment, base-policy inference, and image critic are ready.")
            return
        rng = np.random.default_rng(config["seed"])
        replay = Replay(config["replay_capacity"])
        counters = dict(
            env_steps=0, transitions=0, updates=0, warmup_complete=False, next_eval=0, next_save=0
        )
        if config.get("resume"):
            payload = pickle.loads(Path(config["resume"]).read_bytes())
            for key in ("base_policy", "method", "chunk_size", "discount", "learner"):
                if payload["config"][key] != config[key]:
                    raise ValueError(f"Cannot resume with a changed {key}")
            restore(payload, agent)
            replay, counters = payload["replay"], payload["counters"]
            rng.bit_generator.state = payload["rng"]
            if payload.get("policy_rng") is not None:
                import jax

                env.planner.policy._rng = jax.random.wrap_key_data(payload["policy_rng"])
            observation = env.reset()
        evaluation = Environment(
            config,
            config["checkpoint"],
            output,
            config["seed"] + 10000,
            policy_bundle=env.policy_bundle,
        )

        def checkpoint():
            import jax

            key = getattr(env.planner.policy, "_rng", None)
            save(
                output / "latest.pkl",
                agent,
                replay,
                counters,
                rng,
                config,
                policy_rng=None if key is None else np.asarray(jax.random.key_data(key)),
            )

        def update(count):
            for _ in range(count):
                actor = config["method"] == "exrl_sac"
                agent.update(
                    freeze(replay.sample(config["batch_size"], rng)),
                    update_actor=actor,
                    update_alpha=actor,
                )
                counters["updates"] += 1

        try:
            while counters["env_steps"] < config["env_steps"]:
                if counters["env_steps"] >= counters["next_eval"]:
                    result = evaluate(evaluation, agent, config, config["eval_episodes"])
                    result["env_steps"] = counters["env_steps"]
                    with (output / "evaluation.jsonl").open("a") as stream:
                        stream.write(json.dumps(result) + "\n")
                    print(result, flush=True)
                    counters["next_eval"] = (
                        (counters["env_steps"] // config["eval_interval"]) + 1
                    ) * config["eval_interval"]
                warming = not counters["warmup_complete"]
                k = (
                    int(rng.integers(1, config["chunk_size"] + 1))
                    if warming
                    else choose(agent, observation, config["method"])
                )
                row, done, success = env.step(k)
                replay.extend([row])
                counters["transitions"] += 1
                counters["env_steps"] += int(row["durations"])
                if warming:
                    if len(replay.rows) >= config["warmup_transitions"]:
                        update(config["warmup_updates"])
                        counters["warmup_complete"] = True
                else:
                    update(config["utd"])
                observation = env.reset() if done else row["next_observations"]
                if counters["env_steps"] >= counters["next_save"]:
                    counters["next_save"] = counters["env_steps"] + config["save_interval"]
                    checkpoint()
            result = evaluate(evaluation, agent, config, config["eval_episodes"])
            result["env_steps"] = counters["env_steps"]
            with (output / "evaluation.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
        finally:
            checkpoint()
    finally:
        env.close()
        if evaluation is not None:
            evaluation.close()
