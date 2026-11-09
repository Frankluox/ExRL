"""Collect variable-length transitions without updating the base policy."""

from __future__ import annotations

import time
import numpy as np

from real_robot.observations import critic_observation
from real_robot.rewards import PhysicalDeadline, staged_time_cost


def transition_return(config, start, end, events, failure=None):
    """Charge elapsed time, weighted by the number of unfinished stages."""
    gamma, stages = float(config["discount"]), int(config["stages"])
    completed = sum(t <= start for t in events)
    offsets = [t - start for t in events if start < t <= end]
    reward, discount = staged_time_cost(
        end - start, stages, completed, offsets, gamma, 1.0 / config["control_hz"]
    )
    if failure is not None and config.get("absorbing_failure", False):
        if gamma == 1:
            raise ValueError("Absorbing failure costs require discount < 1")
        remaining = stages - sum(t <= failure for t in events)
        reward -= discount * remaining / (1 - gamma)
    return float(reward), float(discount)


class Collector:
    def __init__(self, robot, policy, operator, config, stats, clock=time):
        self.robot, self.policy, self.operator = robot, policy, operator
        self.config, self.stats, self.clock = config, stats, clock

    def encode(self, frame, actions):
        if np.asarray(actions).ndim != 2 or len(actions) != self.config["chunk_size"]:
            raise ValueError("The policy must return exactly chunk_size primitive actions")
        return critic_observation(
            frame,
            actions,
            self.stats,
            self.config["image_size"],
            self.config["action_mode"],
            self.config.get("camera_keys", ["image"]),
            self.config.get("action_state_indices", []),
        )

    def rollout(self, select_length):
        self.operator.start_episode()
        frame = self.robot.observe()
        start = frame["time"]
        timer = PhysicalDeadline(self.config["episode_seconds"], self.clock.monotonic)
        timer.start(start)
        deadline = timer.deadline
        period = 1 / self.config["control_hz"]
        chunk = self.policy.predict(frame)
        observation = self.encode(frame, chunk)
        transitions, steps = [], 0
        try:
            while True:
                k = int(select_length(observation))
                if not 1 <= k <= self.config["chunk_size"]:
                    raise ValueError("Execution length is outside the predicted chunk")
                issued = []
                for action in chunk[:k]:
                    events, failure, cancelled = self.operator.snapshot()
                    if cancelled:
                        raise KeyboardInterrupt
                    if failure is not None or len(events) >= self.config["stages"]:
                        break
                    if timer.expired():
                        break
                    self.robot.command(action)
                    issued.append(self.clock.monotonic())
                    steps += 1
                    self.clock.sleep(max(0, issued[-1] + period - self.clock.monotonic()))
                boundary = min(self.clock.monotonic(), deadline)
                next_frame = self.robot.observe(after=boundary)
                events, failure, cancelled = self.operator.snapshot()
                if cancelled:
                    raise KeyboardInterrupt
                events = [t for t in events if t <= deadline]
                success = len(events) >= self.config["stages"]
                failed = failure is not None and failure <= deadline
                terminal = success or failed
                timeout = next_frame["time"] >= deadline
                reward_end = min(next_frame["time"], deadline)
                if success:
                    reward_end = min(reward_end, events[self.config["stages"] - 1])
                if failed:
                    reward_end = min(reward_end, failure)
                reward, discount = transition_return(
                    self.config,
                    frame["time"],
                    max(frame["time"], reward_end),
                    events,
                    failure if failed else None,
                )
                # A timeout keeps the next-state bootstrap. Its chunk is queried
                # after the rollout stops and is never executed.
                next_chunk = np.zeros_like(chunk) if terminal else self.policy.predict(next_frame)
                next_observation = self.encode(next_frame, next_chunk)
                transitions.append(
                    dict(
                        observations=observation,
                        next_observations=next_observation,
                        actions=np.int32(k - 1),
                        rewards=np.float32(reward),
                        discount=np.float32(discount),
                        masks=np.float32(not terminal),
                        durations=np.float32(len(issued)),
                    )
                )
                if terminal or timeout:
                    return transitions, dict(
                        success=success,
                        completed_stages=len(events),
                        steps=steps,
                        seconds=max(0, reward_end - start),
                        timeout=timeout and not terminal,
                    )
                frame, chunk, observation = next_frame, next_chunk, next_observation
        finally:
            self.operator.end_episode()
