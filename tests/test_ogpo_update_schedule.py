import argparse
import unittest
from fractions import Fraction

from baselines.ogpo.robomimic.update_schedule import OGPOUpdateSchedule, parse_utd


class TestOGPOUpdateSchedule(unittest.TestCase):
    def test_parse_fraction_decimal_and_invalid_values(self):
        self.assertEqual(parse_utd("1/16"), Fraction(1, 16))
        self.assertEqual(parse_utd("0.0625"), Fraction(1, 16))
        for value in ("0", "-1", "nan", "inf", "1/0", "bad"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_utd(value)

    def test_fraction_does_not_shrink_replay_warmup(self):
        schedule = OGPOUpdateSchedule("1/16", 10000, 256, 32)
        self.assertEqual(schedule.min_replay_size, 8192)
        for step in range(1, 10000):
            self.assertEqual(schedule.advance(step, 9000), 0)
        for step in range(10000, 10100):
            self.assertEqual(schedule.advance(step, 8191), 0)
        self.assertEqual(schedule.eligible_steps, 0)
        fired = [step for step in range(10100, 10148) if schedule.advance(step, 9000)]
        self.assertEqual(fired, [10115, 10131, 10147])

    def test_integer_utd_preserves_previous_loop_and_replay_gate(self):
        for utd in (1, 2, 4):
            schedule = OGPOUpdateSchedule(utd, 10000, 256, 32)
            for step in range(1, 40000, 4):
                replay_size = step - step % 300
                expected = utd if step >= 10000 and replay_size >= 256 * utd * 32 else 0
                self.assertEqual(schedule.advance(step, replay_size), expected)

    def test_fraction_long_run_exact_update_count(self):
        for rate in ("1/16", "1/3", "3/2"):
            schedule = OGPOUpdateSchedule(rate, 0, 1, 1)
            counts = [schedule.advance(i, 100) for i in range(48000)]
            self.assertEqual(sum(counts), int(Fraction(rate) * 48000))
            self.assertEqual(schedule.credit, 0)

    def test_fraction_one_million_step_count(self):
        schedule = OGPOUpdateSchedule("1/16", 10000, 256, 32)
        updates = sum(schedule.advance(step, 10000) for step in range(1, 1000001))
        self.assertEqual(updates, 61875)
        self.assertEqual(schedule.eligible_steps, 990001)

    def test_state_round_trip_preserves_fractional_credit(self):
        uninterrupted = OGPOUpdateSchedule("1/16", 10000, 256, 32)
        before = sum(uninterrupted.advance(step, 10000) for step in range(1, 2000001))
        self.assertEqual(before, 124375)
        self.assertEqual(uninterrupted.credit, 1)

        resumed = OGPOUpdateSchedule("1/16", 10000, 256, 32)
        resumed.load_state_dict(uninterrupted.state_dict())
        after_uninterrupted = [uninterrupted.advance(step, 10000) for step in range(2000001, 2000101)]
        after_resumed = [resumed.advance(step, 10000) for step in range(2000001, 2000101)]
        self.assertEqual(after_resumed, after_uninterrupted)
        self.assertEqual(resumed.state_dict(), uninterrupted.state_dict())

    def test_state_restore_rejects_protocol_and_credit_mismatches(self):
        schedule = OGPOUpdateSchedule("1/16", 10000, 256, 32)
        state = schedule.state_dict()
        bad_rate = dict(state, rate_denominator=64)
        with self.assertRaisesRegex(ValueError, "schedule mismatch"):
            schedule.load_state_dict(bad_rate)
        bad_credit = dict(state, eligible_steps=1, credit=0)
        with self.assertRaisesRegex(ValueError, "credit mismatch"):
            schedule.load_state_dict(bad_credit)


if __name__ == "__main__":
    unittest.main()
