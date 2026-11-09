"""Exercise variable-length collection without a simulator or model server."""

from types import SimpleNamespace

import numpy as np
import pytest

from exrl.libero.train import Environment, evaluate


class FakeSimulator:
    def __init__(self, success_at=None):
        self.t = 0
        self.success_at = success_at

    def step(self, action):
        self.t += 1
        return self.t, 0.0, False, {}

    def check_success(self):
        return self.t == self.success_at


def environment(horizon=5, success_at=None):
    env = object.__new__(Environment)
    env.config = dict(chunk_size=3, horizon=horizon, discount=0.9)
    env.env = FakeSimulator(success_at)
    env.steps = 0
    env.chunk = np.ones((3, 7), np.float32)
    env.obs = dict(state=np.zeros((1, 1), np.float32))
    env.planner = SimpleNamespace(plan=lambda raw: np.ones((3, 7), np.float32))
    env.encode = lambda raw, chunk: dict(state=np.array([[raw]], np.float32))
    return env


def test_timeout_keeps_bootstrap_and_actual_duration():
    env = environment(horizon=5)
    first, done, success = env.step(3)
    assert not done and not success
    assert first["rewards"] == pytest.approx(-1 - 0.9 - 0.9**2)
    last, done, success = env.step(3)
    assert done and not success
    assert last["actions"] == 2  # Chosen k=3, although only two steps remain.
    assert last["durations"] == 2
    assert last["discount"] == pytest.approx(0.9**2)
    assert last["masks"] == 1
    with pytest.raises(RuntimeError):
        env.step(1)


def test_success_is_terminal_and_stops_chunk():
    env = environment(success_at=2)
    row, done, success = env.step(3)
    assert done and success
    assert row["durations"] == 2
    assert row["rewards"] == -1  # Zero reward on the successful second step.
    assert row["masks"] == 0
    assert not env.chunk.any()


def test_evaluation_restores_sampling_streams():
    policy = SimpleNamespace(_rng=7)
    agent = SimpleNamespace(_rng=11)

    def action(obs, critic_reduction):
        agent._rng += 1
        return np.array([0])

    agent.eval_q_action_ids = action

    def reset():
        policy._rng += 1
        return dict(state=np.zeros((1, 1)))

    env = SimpleNamespace(
        planner=SimpleNamespace(policy=policy),
        reset=reset,
        step=lambda k: (dict(next_observations=dict(state=np.zeros((1, 1)))), True, True),
    )
    result = evaluate(env, agent, dict(method="exrl_q"), 2)
    assert result["successes"] == 2
    assert agent._rng == 11 and policy._rng == 7
