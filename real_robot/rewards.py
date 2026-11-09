"""Physical-time costs and one-host monotonic rollout deadlines (no hardware)."""

import math
import time


def stage_count(value):
    if type(value) is not int or value < 1:
        raise ValueError("num_stages must be a positive integer")
    return value


def time_cost(elapsed_seconds, gamma=0.997, tick_seconds=0.05, cost_per_tick=1.0):
    values = tuple(map(float, (elapsed_seconds, gamma, tick_seconds, cost_per_tick)))
    elapsed, gamma, tick, cost = values
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite physical-time reward parameters")
    if elapsed < 0 or tick <= 0 or cost <= 0 or not 0 < gamma <= 1:
        raise ValueError("Invalid physical-time reward parameters")
    if gamma == 1:
        return -cost * elapsed / tick, 1.0
    exponent = math.log(gamma) * elapsed / tick
    return -cost * (-math.expm1(exponent)) / (1.0 - gamma), math.exp(exponent)


def staged_time_cost(
    elapsed_seconds,
    num_stages,
    stage_before=0,
    event_offsets=(),
    gamma=0.997,
    tick_seconds=0.05,
    cost_per_tick=1.0,
):
    """Integrate remaining-stage cost; discount depends only on physical time."""
    stages = stage_count(num_stages)
    if type(stage_before) is not int or not 0 <= stage_before <= stages:
        raise ValueError("Invalid completed-stage count")
    elapsed = float(elapsed_seconds)
    _, discount = time_cost(elapsed, gamma, tick_seconds, cost_per_tick)
    offsets = list(map(float, event_offsets))
    if (
        len(offsets) > stages - stage_before
        or any(not math.isfinite(t) or not 0 <= t <= elapsed for t in offsets)
        or offsets != sorted(offsets)
    ):
        raise ValueError("Invalid physical stage event offsets")
    reward, prefix_discount, cursor = 0.0, 1.0, 0.0
    remaining = stages - stage_before
    for boundary in offsets + [elapsed]:
        part, part_discount = time_cost(boundary - cursor, gamma, tick_seconds, cost_per_tick)
        reward += prefix_discount * remaining * part
        prefix_discount *= part_discount
        cursor = boundary
        remaining -= 1
    return reward, discount


class PhysicalDeadline:
    """Only formal observation capture arms the clock; warmup/reset cannot."""

    def __init__(self, seconds, now=time.monotonic):
        self.seconds, self.now = float(seconds), now
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("Physical horizon must be positive")
        self.started = None

    def start(self, observation_monotonic):
        if self.started is not None:
            raise RuntimeError("Rollout clock is already armed")
        self.started = float(observation_monotonic)
        if not math.isfinite(self.started):
            raise ValueError("Invalid monotonic observation timestamp")

    @property
    def deadline(self):
        return None if self.started is None else self.started + self.seconds

    def expired(self):
        return self.deadline is not None and self.now() >= self.deadline

    def elapsed(self):
        return 0.0 if self.started is None else max(0.0, self.now() - self.started)

    def require_before_deadline(self):
        if self.expired():
            raise PhysicalTimeout("Physical rollout deadline reached")


class PhysicalTimeout(Exception):
    """Stops model publication; never issues a hold, reset, or disable command."""
