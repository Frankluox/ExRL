"""Collector tests use a simulated clock and never import ROS."""
from types import SimpleNamespace
import numpy as np
import pytest
from real_robot.collector import Collector, transition_return
from real_robot.observations import state_from_raw, action_to_raw

class Clock:
    def __init__(self): self.t = 10.
    def monotonic(self): return self.t
    def sleep(self, dt): self.t += dt

class Robot:
    def __init__(self, clock): self.clock, self.commands = clock, []
    def observe(self, after=None):
        return dict(time=self.clock.t, state=np.zeros(7), image=np.zeros((16,16,3),np.uint8), wrist_image=np.zeros((16,16,3),np.uint8))
    def command(self, action): self.commands.append(action.copy())

class Operator:
    def __init__(self, clock, success_at=None): self.clock, self.success_at, self.ended = clock, success_at, False
    def start_episode(self): pass
    def end_episode(self): self.ended = True
    def snapshot(self):
        events = [self.success_at] if self.success_at is not None and self.clock.t >= self.success_at else []
        return events, None, False

class Policy:
    def __init__(self, clock, latency): self.clock, self.latency = clock, latency
    def predict(self, frame):
        self.clock.sleep(self.latency)
        return np.zeros((4,7),np.float32)

def run(seconds, latency, success_at=None):
    c = Clock(); r = Robot(c); op = Operator(c, success_at)
    cfg = dict(chunk_size=4, control_hz=20, image_size=16, action_mode="absolute", discount=.99, stages=1, reward="elapsed_time", episode_seconds=seconds)
    stats = {f"{p}_{k}": np.ones(7) if k=="std" else np.zeros(7) for p in ["state","action"] for k in ["mean","std"]}
    rows, result = Collector(r, Policy(c,latency),op,cfg,stats,c).rollout(lambda obs:4)
    assert op.ended
    return r, rows, result

def test_timeout_stops_commands_but_keeps_bootstrap():
    robot, rows, result = run(.3,.12)
    assert len(robot.commands)==4
    assert result["timeout"] and not result["success"]
    assert rows[-1]["masks"]==1
    assert rows[-1]["discount"] == pytest.approx(.99**6)

def test_success_during_inference_has_time_cost_and_no_command():
    robot, rows, result = run(1,.3,10.2)
    assert result["success"] and len(robot.commands)==0
    assert rows[-1]["masks"]==0
    assert result["seconds"]==pytest.approx(.2)
    assert rows[-1]["rewards"]==pytest.approx(-(1-.99**4)/.01)

def test_piper_coordinate_roundtrip():
    raw=np.array([1000,-20000,30000,-50000,100000,45000,35000])
    offsets=np.array([.01,.02,-.02,0,0,0])
    np.testing.assert_array_equal(action_to_raw(state_from_raw(raw,offsets),offsets),raw)

def test_fold_stage_cost_includes_rpc_interval():
    cfg=dict(discount=1.,stages=3,reward="elapsed_time",control_hz=20)
    reward,discount=transition_return(cfg,10.,11.,[],[10.25,10.75])
    assert reward==pytest.approx(-40.)
    assert discount==1.
