"""Deadline accounting for ROS simulation time and monotonic wall time.

Callers provide clocks explicitly so navigation can use simulation seconds while
a separate watchdog notices a paused or disconnected simulation. This module
does not depend on ROS and never sleeps.
"""

import math


class Deadline:
    """Track a duration limit and an optional stalled ROS clock watchdog.

    A zero ``timeout_s`` disables the duration limit. A zero
    ``stall_timeout_s`` disables the watchdog, which applies only in ROS mode.
    Timing attributes are snapshots updated by ``expired()``. Once a deadline
    expires, its reason and final timing snapshot remain available unchanged.
    """

    def __init__(self, timeout_s, clock="wall", stall_timeout_s=0.0,
                 ros_now=None, wall_now=None):
        self.timeout_s = self._duration(timeout_s, "timeout_s")
        self.stall_timeout_s = self._duration(stall_timeout_s, "stall_timeout_s")
        if clock not in ("ros", "wall"):
            raise ValueError("clock must be 'ros' or 'wall'")
        if not callable(ros_now) or not callable(wall_now):
            raise ValueError("ros_now and wall_now must be clock callbacks")
        self.clock = clock
        self._ros_now = ros_now
        self._wall_now = wall_now
        self._start_ros = self._timestamp(ros_now(), "ROS")
        self._start_wall = self._timestamp(wall_now(), "wall")
        self._last_ros = self._start_ros
        self._last_wall = self._start_wall
        self._last_ros_advance_wall = self._start_wall
        self.reason = None
        self.elapsed_ros_s = 0.0
        self.elapsed_wall_s = 0.0
        self.real_time_factor = 0.0

    @staticmethod
    def _duration(value, name):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("%s must be finite and nonnegative" % name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("%s must be finite and nonnegative" % name)
        return value

    @staticmethod
    def _timestamp(value, name):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("%s clock returned an invalid timestamp" % name)
        if not math.isfinite(value):
            raise ValueError("%s clock returned a nonfinite timestamp" % name)
        return value

    def expired(self):
        """Poll both clocks and return whether either applicable limit expired."""
        if self.reason is not None:
            return True
        try:
            current_ros = self._timestamp(self._ros_now(), "ROS")
            current_wall = self._timestamp(self._wall_now(), "wall")
        except ValueError as exc:
            self.reason = str(exc)
            return True

        self.elapsed_ros_s = max(0.0, current_ros - self._start_ros)
        self.elapsed_wall_s = max(0.0, current_wall - self._start_wall)
        self.real_time_factor = (self.elapsed_ros_s / self.elapsed_wall_s
                                 if self.elapsed_wall_s > 0.0 else 0.0)

        if current_wall < self._last_wall:
            self.reason = "monotonic wall clock moved backwards"
        elif self.clock == "ros" and current_ros < self._last_ros:
            self.reason = ("ROS clock moved backwards from %.6f to %.6f s; "
                           "the simulation may have reset" %
                           (self._last_ros, current_ros))

        if current_ros > self._last_ros:
            self._last_ros_advance_wall = current_wall
        self._last_ros = current_ros
        self._last_wall = current_wall
        if self.reason is not None:
            return True

        elapsed = self.elapsed_ros_s if self.clock == "ros" else self.elapsed_wall_s
        if self.timeout_s > 0.0 and elapsed >= self.timeout_s:
            self.reason = "%s time limit of %.3f s reached" % (
                "ROS" if self.clock == "ros" else "wall", self.timeout_s)
        elif (self.clock == "ros" and self.stall_timeout_s > 0.0 and
              current_wall - self._last_ros_advance_wall >= self.stall_timeout_s):
            self.reason = ("ROS clock stopped advancing for %.3f wall seconds; "
                           "check /clock and whether the simulation is paused" %
                           (current_wall - self._last_ros_advance_wall))
        return self.reason is not None
