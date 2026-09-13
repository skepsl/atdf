#!/usr/bin/env python3
"""Deterministic simulation deadline tests without ROS or wall-clock sleeps."""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atdf_timing import Deadline


class Clocks:
    def __init__(self, ros=10.0, wall=100.0):
        self.ros = ros
        self.wall = wall

    def deadline(self, timeout_s, **kwargs):
        return Deadline(timeout_s, ros_now=lambda: self.ros,
                        wall_now=lambda: self.wall, **kwargs)

    def advance(self, ros, wall):
        self.ros += ros
        self.wall += wall


class DeadlineTest(unittest.TestCase):
    def test_slow_simulation_does_not_spend_wall_time_budget(self):
        clocks = Clocks()
        deadline = clocks.deadline(120.0, clock="ros", stall_timeout_s=60.0)
        # Matches the reported run: 120 wall seconds advanced ROS by only 30.
        clocks.advance(30.0, 120.0)
        self.assertFalse(deadline.expired())
        self.assertEqual(deadline.elapsed_ros_s, 30.0)
        self.assertEqual(deadline.elapsed_wall_s, 120.0)
        self.assertEqual(deadline.real_time_factor, 0.25)
        clocks.advance(90.0, 360.0)
        self.assertTrue(deadline.expired())
        self.assertIn("ROS time limit", deadline.reason)

    def test_fast_simulation_expires_by_simulation_seconds(self):
        clocks = Clocks()
        deadline = clocks.deadline(20.0, clock="ros")
        clocks.advance(20.0, 5.0)
        self.assertTrue(deadline.expired())
        self.assertEqual(deadline.real_time_factor, 4.0)

    def test_wall_mode_ignores_ros_stalls_and_resets(self):
        clocks = Clocks()
        deadline = clocks.deadline(5.0, clock="wall", stall_timeout_s=1.0)
        clocks.advance(0.0, 2.0)
        self.assertFalse(deadline.expired())
        clocks.advance(-10.0, 2.0)
        self.assertFalse(deadline.expired())
        clocks.advance(50.0, 1.0)
        self.assertTrue(deadline.expired())
        self.assertIn("wall time limit", deadline.reason)

    def test_frozen_ros_clock_expires_at_watchdog_boundary(self):
        clocks = Clocks(ros=0.0)
        deadline = clocks.deadline(300.0, clock="ros", stall_timeout_s=120.0)
        clocks.advance(0.0, 119.0)
        self.assertFalse(deadline.expired())
        clocks.advance(0.0, 1.0)
        self.assertTrue(deadline.expired())
        self.assertIn("ROS clock stopped advancing", deadline.reason)
        self.assertEqual(deadline.elapsed_ros_s, 0.0)
        self.assertEqual(deadline.real_time_factor, 0.0)

    def test_each_positive_clock_advance_resets_stall_watchdog(self):
        clocks = Clocks()
        deadline = clocks.deadline(0.0, clock="ros", stall_timeout_s=2.0)
        clocks.advance(0.0, 1.5)
        self.assertFalse(deadline.expired())
        clocks.advance(0.01, 0.4)
        self.assertFalse(deadline.expired())
        clocks.advance(0.0, 1.5)
        self.assertFalse(deadline.expired())
        clocks.advance(0.0, 0.6)
        self.assertTrue(deadline.expired())
        self.assertIn("ROS clock stopped advancing", deadline.reason)

    def test_ros_backwards_jump_fails_even_above_start_stamp(self):
        clocks = Clocks()
        deadline = clocks.deadline(0.0, clock="ros")
        clocks.advance(10.0, 1.0)
        self.assertFalse(deadline.expired())
        clocks.advance(-1.0, 1.0)
        self.assertTrue(deadline.expired())
        self.assertIn("ROS clock moved backwards from 20.000000 to 19.000000", deadline.reason)

    def test_zero_duration_and_watchdog_allow_unlimited_wait(self):
        for clock in ("ros", "wall"):
            with self.subTest(clock=clock):
                clocks = Clocks()
                deadline = clocks.deadline(0.0, clock=clock, stall_timeout_s=0.0)
                clocks.advance(0.0, 1e6)
                self.assertFalse(deadline.expired())
                clocks.advance(1e6, 1e6)
                self.assertFalse(deadline.expired())

    def test_expiration_reason_and_snapshot_persist_without_polling(self):
        clocks = Clocks()
        deadline = clocks.deadline(1.0, clock="ros")
        clocks.advance(1.0, 4.0)
        self.assertTrue(deadline.expired())
        reason = deadline.reason
        clocks.ros = float("nan")
        clocks.wall = float("nan")
        self.assertTrue(deadline.expired())
        self.assertEqual(deadline.reason, reason)
        self.assertEqual(deadline.elapsed_ros_s, 1.0)
        self.assertEqual(deadline.elapsed_wall_s, 4.0)
        self.assertEqual(deadline.real_time_factor, 0.25)

    def test_initial_timing_snapshot_is_finite(self):
        deadline = Clocks().deadline(1.0)
        self.assertIsNone(deadline.reason)
        self.assertEqual(deadline.elapsed_ros_s, 0.0)
        self.assertEqual(deadline.elapsed_wall_s, 0.0)
        self.assertTrue(math.isfinite(deadline.real_time_factor))
        self.assertFalse(deadline.expired())

    def test_invalid_limits_and_clock_are_rejected(self):
        for name in ("timeout_s", "stall_timeout_s"):
            for value in (-1.0, float("nan"), float("inf"), None, "bad"):
                with self.subTest(name=name, value=value):
                    args = {"timeout_s": 1.0, name: value}
                    with self.assertRaises(ValueError):
                        Clocks().deadline(**args)
        with self.assertRaises(ValueError):
            Clocks().deadline(1.0, clock="sim")
        with self.assertRaises(ValueError):
            Deadline(1.0)

    def test_invalid_clock_values_are_rejected_at_start_or_expire_on_poll(self):
        for name in ("ros", "wall"):
            with self.subTest(clock=name):
                clocks = Clocks()
                setattr(clocks, name, float("inf"))
                with self.assertRaises(ValueError):
                    clocks.deadline(1.0)
                clocks = Clocks()
                deadline = clocks.deadline(1.0)
                setattr(clocks, name, float("nan"))
                self.assertTrue(deadline.expired())
                self.assertIn("nonfinite timestamp", deadline.reason)

    def test_broken_monotonic_clock_fails_clearly(self):
        clocks = Clocks()
        deadline = clocks.deadline(1.0)
        clocks.advance(0.0, -1.0)
        self.assertTrue(deadline.expired())
        self.assertIn("wall clock moved backwards", deadline.reason)


if __name__ == "__main__":
    unittest.main()
