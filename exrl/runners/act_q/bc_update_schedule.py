"""BC update budget measured from the first eligible imitation update."""

import math


class LinearBCUpdateSchedule:
    """Accumulate fractional updates rather than rounding each RL update."""

    def __init__(self, initial, final, duration):
        if not all(math.isfinite(x) and x >= 0 for x in (initial, final, duration)):
            raise ValueError("BC schedule values must be finite and nonnegative")
        self.initial = float(initial)
        self.final = float(final)
        self.duration = int(duration)
        self.start_step = None
        self.remainder = 0.0

    def rate(self, env_step):
        if not self.duration or self.start_step is None:
            return self.initial
        progress = min(max((env_step - self.start_step) / self.duration, 0.0), 1.0)
        return self.initial + progress * (self.final - self.initial)

    def next_updates(self, env_step):
        # Call only after the start delay and successful-data readiness checks.
        if self.start_step is None:
            self.start_step = int(env_step)
        budget = self.remainder + self.rate(env_step)
        updates = math.floor(budget + 1e-12)
        self.remainder = max(budget - updates, 0.0)
        return updates

    def state_dict(self):
        return dict(
            initial=self.initial,
            final=self.final,
            duration=self.duration,
            start_step=self.start_step,
            remainder=self.remainder,
        )

    def load_state_dict(self, state):
        for key in ("initial", "final", "duration"):
            if state[key] != getattr(self, key):
                raise ValueError(f"BC update schedule mismatch on resume: {key}")
        if not 0 <= state["remainder"] < 1:
            raise ValueError("Invalid BC fractional update remainder")
        self.start_step = state["start_step"]
        self.remainder = float(state["remainder"])
