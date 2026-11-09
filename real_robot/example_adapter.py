"""A software-only example of the robot and policy interfaces.

Replace these methods with calls to your existing controller and policy.
The example deliberately uses a single RGB view and four action dimensions.
"""

import time
import numpy as np


class ExampleRobot:
    def __init__(self):
        self.state = np.zeros(4, np.float32)

    def observe(self, after=None):
        if after is not None:
            time.sleep(max(0, after - time.monotonic()))
        return dict(
            time=time.monotonic(), state=self.state.copy(), image=np.zeros((128, 128, 3), np.uint8)
        )

    def command(self, action):
        self.state = np.asarray(action, np.float32).copy()

    def close(self):
        pass


class ExamplePolicy:
    def __init__(self, horizon):
        self.horizon = horizon

    def predict(self, observation):
        return np.repeat(observation["state"][None], self.horizon, axis=0)


def make_robot(options, task):
    return ExampleRobot()


def make_policy(options, task):
    return ExamplePolicy(task["chunk_size"])
