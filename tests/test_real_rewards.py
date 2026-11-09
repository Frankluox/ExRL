import math
import unittest

from real_robot.rewards import PhysicalDeadline, PhysicalTimeout, time_cost, staged_time_cost


class TimeMathTests(unittest.TestCase):
    def test_three_stage_cost_splits_at_exact_times_not_command_indices(self):
        reward, discount = staged_time_cost(0.83, 3, 0, [0.123, 0.517])
        r1, d1 = time_cost(0.123)
        r2, d2 = time_cost(0.394)
        r3, _ = time_cost(0.313)
        self.assertAlmostEqual(reward, 3 * r1 + d1 * 2 * r2 + d1 * d2 * r3, places=10)
        self.assertAlmostEqual(discount, time_cost(0.83)[1], places=12)
        self.assertAlmostEqual(
            staged_time_cost(0.83, 3, 0, [0.123, 0.517], gamma=1)[0],
            -(3 * 0.123 + 2 * 0.394 + 0.313) / 0.05,
        )

    def test_stage_partition_preserves_return_and_single_stage_compatibility(self):
        whole = staged_time_cost(0.83, 3, 0, [0.123, 0.517])
        first = staged_time_cost(0.4, 3, 0, [0.123])
        second = staged_time_cost(0.43, 3, 1, [0.117])
        self.assertAlmostEqual(whole[0], first[0] + first[1] * second[0], places=10)
        for elapsed in (0.05, 0.214, 16.0):
            self.assertEqual(staged_time_cost(elapsed, 1), time_cost(elapsed))
            self.assertEqual(staged_time_cost(elapsed, 1, 0, [elapsed]), time_cost(elapsed))

    def test_invalid_stage_events_are_rejected(self):
        for args in (
            (0.5, 0),
            (0.5, True),
            (0.5, 3, 4),
            (0.5, 3, 0, [0.4, 0.2]),
            (0.5, 3, 0, [0.6]),
            (0.5, 3, 2, [0.1, 0.2]),
            (0.5, 3, 0, [math.nan]),
        ):
            with self.assertRaises(ValueError):
                staged_time_cost(*args)

    def test_matches_original_scale_at_fifty_milliseconds(self):
        for k in (1, 10, 50):
            reward, discount = time_cost(k * 0.05)
            self.assertAlmostEqual(reward, -sum(0.997**i for i in range(k)), places=10)
            self.assertAlmostEqual(discount, 0.997**k, places=12)

    def test_arbitrary_interval_partition_is_additive(self):
        r1, d1 = time_cost(0.163)
        r2, d2 = time_cost(0.517)
        r, d = time_cost(0.680)
        self.assertAlmostEqual(r, r1 + d1 * r2, places=10)
        self.assertAlmostEqual(d, d1 * d2, places=12)

    def test_inference_time_is_charged_and_discounted(self):
        fast, fast_d = time_cost(0.05)
        slow, slow_d = time_cost(0.05 + 0.164)
        self.assertLess(slow, fast)
        self.assertLess(slow_d, fast_d)

    def test_undiscounted_limit_and_invalid_values(self):
        reward, discount = time_cost(0.7, gamma=1)
        self.assertAlmostEqual(reward, -14.0)
        self.assertEqual(discount, 1.0)
        for kwargs in (
            {"elapsed_seconds": -1},
            {"elapsed_seconds": math.nan},
            {"elapsed_seconds": 0.1, "tick_seconds": 0},
        ):
            with self.assertRaises(ValueError):
                time_cost(**kwargs)

    def test_physical_deadline_does_not_depend_on_action_count(self):
        now = [100.0]
        clock = PhysicalDeadline(12, now=lambda: now[0])
        now[0] += 20  # Model warmup / reset before formal observation.
        self.assertFalse(clock.expired())
        self.assertEqual(clock.elapsed(), 0)
        clock.start(now[0])
        self.assertEqual(clock.deadline, 132.0)
        now[0] = 131.999
        clock.require_before_deadline()
        now[0] = 132.0
        with self.assertRaises(PhysicalTimeout):
            clock.require_before_deadline()
        with self.assertRaises(RuntimeError):
            clock.start(now[0])


if __name__ == "__main__":
    unittest.main()
