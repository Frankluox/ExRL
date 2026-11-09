"""Exact update budgeting per eligible OGPO vector step."""

import argparse
from fractions import Fraction


def parse_utd(value):
    try:
        rate = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError("UTD must be a positive number or fraction") from exc
    if rate <= 0:
        raise argparse.ArgumentTypeError("UTD must be positive")
    return rate


class OGPOUpdateSchedule:
    def __init__(self, utd, start_training, batch_size, horizon_steps):
        self.rate = parse_utd(utd)
        self.start_training = int(start_training)
        required = int(batch_size) * int(horizon_steps) * max(1, self.rate)
        self.min_replay_size = (
            required.numerator + required.denominator - 1
        ) // required.denominator
        self.eligible_steps = 0
        self.credit = 0

    def advance(self, env_steps, replay_size):
        if env_steps < self.start_training or replay_size < self.min_replay_size:
            return 0
        # Warmup and insufficient replay never accumulate an update backlog.
        self.eligible_steps += 1
        self.credit += self.rate.numerator
        updates, self.credit = divmod(self.credit, self.rate.denominator)
        return updates

    def state_dict(self):
        return {
            "rate_numerator": int(self.rate.numerator),
            "rate_denominator": int(self.rate.denominator),
            "start_training": int(self.start_training),
            "min_replay_size": int(self.min_replay_size),
            "eligible_steps": int(self.eligible_steps),
            "credit": int(self.credit),
        }

    def load_state_dict(self, state):
        expected = {
            "rate_numerator": int(self.rate.numerator),
            "rate_denominator": int(self.rate.denominator),
            "start_training": int(self.start_training),
            "min_replay_size": int(self.min_replay_size),
        }
        actual = {key: int(state[key]) for key in expected}
        if actual != expected:
            raise ValueError(f"update schedule mismatch: expected={expected}, checkpoint={actual}")
        eligible_steps = int(state["eligible_steps"])
        credit = int(state["credit"])
        if eligible_steps < 0 or not 0 <= credit < self.rate.denominator:
            raise ValueError(
                f"invalid update schedule counters: eligible_steps={eligible_steps}, credit={credit}"
            )
        expected_credit = (eligible_steps * self.rate.numerator) % self.rate.denominator
        if credit != expected_credit:
            raise ValueError(
                f"update schedule credit mismatch: expected={expected_credit}, checkpoint={credit}"
            )
        self.eligible_steps = eligible_steps
        self.credit = credit
