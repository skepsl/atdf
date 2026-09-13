"""Check navigation completion and RF settling without a running ROS master.

Run after sourcing /opt/ros/noetic/setup.bash. The actual ATDF methods and ROS
messages are used, but no model, GPU, simulator, or action server is started.
"""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import torch


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src/atdf_video/src"
sys.path.insert(0, str(SOURCE_DIR))
import atdf_video as atdf


ROS_TIME = atdf.rospy.Time if atdf.ROS_AVAILABLE else None


class Clock:
    """Wall time always advances; simulation time follows supplied samples."""

    def __init__(self):
        self.wall = 0.0
        self.ros_seconds = 100.0
        self.on_sleep = None

    def monotonic(self):
        return self.wall

    def now(self):
        return ROS_TIME.from_sec(self.ros_seconds)

    def sleep(self, seconds):
        self.wall += seconds
        if self.on_sleep is not None:
            self.on_sleep()


@unittest.skipUnless(atdf.ROS_AVAILABLE, "Source the ROS Noetic environment first")
class NavigationMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        fake_rospy = SimpleNamespace(
            Time=SimpleNamespace(now=self.clock.now),
            is_shutdown=mock.Mock(return_value=False),
            logwarn_throttle=mock.Mock(),
            loginfo=mock.Mock(),
            logwarn=mock.Mock(),
        )
        self.patch_rospy = mock.patch.object(atdf, "rospy", fake_rospy)
        self.patch_time = mock.patch.object(atdf, "time", self.clock)
        self.patch_rospy.start()
        self.patch_time.start()
        self.addCleanup(self.patch_rospy.stop)
        self.addCleanup(self.patch_time.stop)

        # Skip loading the neural predictor and particle filter entirely.
        self.finder = atdf.ATDF.__new__(atdf.ATDF)
        self.finder.device = torch.device("cpu")
        self.finder._ros_ready = True
        self.finder.goal_frame = "map"
        self.finder.snap_rx_pose_to_valid = lambda pose: pose
        self.finder._goal_pub = mock.Mock()
        self.finder._goal_marker_pub = mock.Mock()
        self.finder._goal_cancel_pub = mock.Mock()
        self.finder._move_base_client = mock.Mock()
        self.goal = torch.tensor([1.0, 2.0, 0.0])
        self.finder.wait_for_robot_pose = mock.Mock(return_value=self.goal, side_effect=self.capture_current_pose)
        self.finder.measurement_settle_s = 0.4
        self.finder.measurement_settle_timeout_s = 0.5
        self.finder.stopped_linear_speed = 0.03
        self.finder.stopped_angular_speed = 0.04
        self.finder.motion_max_age_s = 0.5
        self.finder._latest_motion = None

    def capture_current_pose(self, timeout):
        self.finder._measurement_stamp = self.clock.now()
        return self.finder.wait_for_robot_pose.return_value

    def configure_tf(self, stamp_seconds=99.8):
        self.finder.robot_pose_source = "tf"
        self.finder.robot_pose_parent_frame = "map"
        self.finder.robot_pose_child_frame = "base_link"
        self.finder.robot_pose_max_age_s = 0.5
        self.finder._tf_listener = mock.Mock()
        self.finder._tf_listener.getLatestCommonTime.return_value = ROS_TIME.from_sec(stamp_seconds)
        self.finder._tf_listener.lookupTransform.return_value = (
            (1.2, 2.3, 0.0), (0.0, 0.0, 0.0, 1.0))

    def test_fresh_tf_pose_preserves_transform_timestamp_for_rf_measurement(self):
        self.configure_tf()
        stamp = self.finder._tf_listener.getLatestCommonTime.return_value

        # Invoke the real wait method instead of the pose mock used by action tests.
        pose = atdf.ATDF.wait_for_robot_pose(self.finder, timeout=0.1)

        self.assertTrue(torch.allclose(pose, torch.tensor([1.2, 2.3, 0.0])))
        self.finder._tf_listener.getLatestCommonTime.assert_called_once_with("map", "base_link")
        self.finder._tf_listener.lookupTransform.assert_called_once_with("map", "base_link", stamp)
        self.assertEqual(self.finder._measurement_stamp, stamp)
        self.assertNotEqual(self.finder._measurement_stamp, self.clock.now())

    def test_stale_and_future_tf_are_rejected_before_pose_lookup(self):
        for stamp_seconds in (99.0, 101.0):
            with self.subTest(stamp=stamp_seconds):
                self.configure_tf(stamp_seconds)
                previous_measurement = ROS_TIME.from_sec(90.0)
                self.finder._measurement_stamp = previous_measurement

                self.assertIsNone(self.finder._lookup_robot_pose_tf())

                self.finder._tf_listener.lookupTransform.assert_not_called()
                self.assertEqual(self.finder._measurement_stamp, previous_measurement)

    def test_missing_tf_listener_or_module_produces_no_pose(self):
        self.finder._tf_listener = None
        self.assertIsNone(self.finder._lookup_robot_pose_tf())

        self.configure_tf()
        with mock.patch.object(atdf, "tf", None):
            self.assertIsNone(self.finder._lookup_robot_pose_tf())
        self.finder._tf_listener.getLatestCommonTime.assert_not_called()

    def test_missing_tf_connection_produces_no_pose(self):
        self.configure_tf()
        self.finder._tf_listener.getLatestCommonTime.side_effect = RuntimeError("Frames are disconnected")

        self.assertIsNone(self.finder._lookup_robot_pose_tf())

        self.finder._tf_listener.lookupTransform.assert_not_called()
        atdf.rospy.logwarn_throttle.assert_called_once()

    def feed_motion(self, samples):
        """Each sample is (ROS now, odometry stamp, linear speed, yaw rate)."""
        remaining = iter(samples)

        def advance():
            sample = next(remaining, None)
            if sample is not None:
                now, stamp, linear, angular = sample
                self.clock.ros_seconds = now
                self.finder._latest_motion = (ROS_TIME.from_sec(stamp), linear, angular)

        advance()
        self.clock.on_sleep = advance

    def use_simulation_clock(self, rate=0.25, stall_timeout=1.0):
        self.finder.goal_timeout_clock = "ros"
        self.finder.clock_stall_timeout_s = stall_timeout
        start_ros = self.clock.ros_seconds
        start_wall = self.clock.wall

        def advance():
            self.clock.ros_seconds = start_ros + (self.clock.wall - start_wall) * rate

        self.clock.on_sleep = advance
        return advance

    @staticmethod
    def rendered_log(call):
        args = call.args
        return args[0] % args[1:] if len(args) > 1 else args[0]

    def test_action_goal_is_sent_once_without_simple_goal_duplication(self):
        client = self.finder._move_base_client
        client.get_state.return_value = atdf.GoalStatus.SUCCEEDED

        self.assertTrue(self.finder.send_robot_goal_and_wait(self.goal))

        client.send_goal.assert_called_once()
        sent = client.send_goal.call_args[0][0].target_pose
        self.assertEqual(sent.header.frame_id, "map")
        self.assertEqual((sent.pose.position.x, sent.pose.position.y), (1.0, 2.0))
        self.finder._goal_marker_pub.publish.assert_called_once_with(sent)
        self.finder._goal_pub.publish.assert_not_called()
        client.cancel_goal.assert_not_called()

    def test_nearby_pose_does_not_cancel_active_action_before_success(self):
        client = self.finder._move_base_client
        client.get_state.side_effect = [atdf.GoalStatus.ACTIVE, atdf.GoalStatus.ACTIVE,
                                        atdf.GoalStatus.SUCCEEDED]
        self.finder.wait_for_robot_pose.return_value = self.goal.clone()

        self.assertTrue(self.finder.send_robot_goal_and_wait(self.goal))

        self.assertEqual(client.get_state.call_count, 3)
        self.assertGreaterEqual(self.clock.wall, 0.1)
        client.cancel_goal.assert_not_called()
        self.finder._goal_pub.publish.assert_not_called()

    def test_action_timeout_cancels_and_reports_failure_despite_nearby_pose(self):
        client = self.finder._move_base_client
        client.get_state.return_value = atdf.GoalStatus.ACTIVE

        self.assertFalse(self.finder.send_robot_goal_and_wait(self.goal, timeout=0.12))

        client.cancel_goal.assert_called_once_with()
        self.assertGreaterEqual(self.clock.wall, 0.12)
        self.finder._goal_pub.publish.assert_not_called()

    def test_slow_simulation_goal_succeeds_after_old_wall_time_limit(self):
        self.use_simulation_clock()
        client = self.finder._move_base_client
        client.get_state.side_effect = lambda: (
            atdf.GoalStatus.SUCCEEDED if self.clock.wall >= 2.0
            else atdf.GoalStatus.ACTIVE)

        self.assertTrue(self.finder.send_robot_goal_and_wait(self.goal, timeout=1.0))

        self.assertGreater(self.clock.wall, 1.0)
        self.assertLess(self.clock.ros_seconds - 100.0, 1.0)
        client.cancel_goal.assert_not_called()
        message = self.rendered_log(atdf.rospy.loginfo.call_args)
        self.assertIn("Navigation succeeded", message)
        self.assertIn("real_time_factor=0.25", message)

    def test_simulation_goal_budget_cancels_after_full_simulation_duration(self):
        self.use_simulation_clock()
        client = self.finder._move_base_client
        client.get_state.return_value = atdf.GoalStatus.ACTIVE

        self.assertFalse(self.finder.send_robot_goal_and_wait(self.goal, timeout=0.2))

        self.assertGreaterEqual(self.clock.wall, 0.8)
        self.assertLess(self.clock.wall, 0.9)
        client.cancel_goal.assert_called_once_with()
        message = self.rendered_log(atdf.rospy.logwarn.call_args)
        self.assertIn("ROS time limit", message)
        self.assertIn("real_time_factor=0.25", message)

    def test_paused_simulation_goal_cancels_with_clock_watchdog_reason(self):
        self.use_simulation_clock(rate=0.0, stall_timeout=0.2)
        client = self.finder._move_base_client
        client.get_state.return_value = atdf.GoalStatus.ACTIVE

        self.assertFalse(self.finder.send_robot_goal_and_wait(self.goal, timeout=1.0))

        self.assertEqual(self.clock.ros_seconds, 100.0)
        self.assertGreaterEqual(self.clock.wall, 0.2)
        client.cancel_goal.assert_called_once_with()
        message = self.rendered_log(atdf.rospy.logwarn.call_args)
        self.assertIn("ROS clock stopped advancing", message)
        self.assertIn("/clock", message)
        self.assertIn("real_time_factor=0.00", message)

    def test_unlimited_goal_duration_still_honors_action_failure(self):
        self.use_simulation_clock(stall_timeout=0.2)
        client = self.finder._move_base_client
        client.get_state.side_effect = lambda: (
            atdf.GoalStatus.ABORTED if self.clock.wall >= 0.5
            else atdf.GoalStatus.ACTIVE)
        client.get_goal_status_text.return_value = "No valid local plan"

        self.assertFalse(self.finder.send_robot_goal_and_wait(self.goal, timeout=0.0))

        self.assertGreaterEqual(self.clock.wall, 0.5)
        self.assertGreater(client.get_state.call_count, 1)
        client.cancel_goal.assert_not_called()
        message = self.rendered_log(atdf.rospy.logwarn.call_args)
        self.assertIn("No valid local plan", message)
        self.assertIn("state=%s" % atdf.GoalStatus.ABORTED, message)

    def test_unlimited_goal_duration_still_honors_stalled_clock_watchdog(self):
        self.use_simulation_clock(rate=0.0, stall_timeout=0.2)
        client = self.finder._move_base_client
        client.get_state.return_value = atdf.GoalStatus.ACTIVE

        self.assertFalse(self.finder.send_robot_goal_and_wait(self.goal, timeout=0.0))

        client.cancel_goal.assert_called_once_with()
        self.assertIn("ROS clock stopped advancing",
                      self.rendered_log(atdf.rospy.logwarn.call_args))

    def test_invalid_goal_timeout_is_rejected_before_sending_goal(self):
        for timeout in (-1.0, float("nan"), float("inf")):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    self.finder.send_robot_goal_and_wait(self.goal, timeout=timeout)
                self.finder._move_base_client.send_goal.assert_not_called()
                self.finder._goal_pub.publish.assert_not_called()
                self.finder._goal_marker_pub.publish.assert_not_called()

    def test_action_terminal_failure_is_not_converted_to_pose_success(self):
        client = self.finder._move_base_client
        for state in (atdf.GoalStatus.ABORTED, atdf.GoalStatus.REJECTED,
                      atdf.GoalStatus.PREEMPTED, atdf.GoalStatus.RECALLED,
                      atdf.GoalStatus.LOST):
            with self.subTest(state=state):
                client.get_state.return_value = state
                self.assertFalse(self.finder.send_robot_goal_and_wait(self.goal))

    def test_simple_goal_fallback_publishes_once_and_cancels_on_arrival(self):
        self.finder._move_base_client = None
        self.finder.wait_for_robot_pose.side_effect = [torch.tensor([0.0, 0.0, 0.0]), self.goal]

        self.assertTrue(self.finder.send_robot_goal_and_wait(self.goal))

        self.finder._goal_pub.publish.assert_called_once()
        self.finder._goal_cancel_pub.publish.assert_called_once()
        self.assertIsInstance(self.finder._goal_cancel_pub.publish.call_args[0][0], atdf.GoalID)

    def test_simple_goal_fallback_timeout_cancels(self):
        self.finder._move_base_client = None
        self.finder.wait_for_robot_pose.return_value = None

        self.assertFalse(self.finder.send_robot_goal_and_wait(self.goal, timeout=0.12))

        self.finder._goal_pub.publish.assert_called_once()
        self.finder._goal_cancel_pub.publish.assert_called_once()

    def test_motion_callback_uses_planar_speed_and_absolute_yaw_rate(self):
        msg = atdf.Odometry()
        msg.header.stamp = ROS_TIME.from_sec(12.0)
        msg.twist.twist.linear.x = -0.3
        msg.twist.twist.linear.y = 0.4
        msg.twist.twist.angular.z = -0.2

        self.finder._motion_callback(msg)

        stamp, linear, angular = self.finder._latest_motion
        self.assertEqual(stamp, msg.header.stamp)
        self.assertAlmostEqual(linear, 0.5)
        self.assertAlmostEqual(angular, 0.2)

    def test_moving_stale_future_nonfinite_and_frozen_samples_cannot_settle(self):
        scenarios = {
            "linear motion": (100.0, 100.0, 0.1, 0.0),
            "rotation": (100.0, 100.0, 0.0, 0.1),
            "stale": (100.0, 99.0, 0.0, 0.0),
            "future": (100.0, 101.0, 0.0, 0.0),
            "nonfinite linear": (100.0, 100.0, float("nan"), 0.0),
            "nonfinite angular": (100.0, 100.0, 0.0, float("inf")),
            "frozen zero velocity": (100.0, 100.0, 0.0, 0.0),
        }
        for name, sample in scenarios.items():
            with self.subTest(scenario=name):
                self.feed_motion([sample])
                with self.assertRaisesRegex(RuntimeError, "fresh stopped odometry"):
                    self.finder.wait_for_measurement_pose()
                self.finder.wait_for_robot_pose.assert_not_called()

    def test_missing_odometry_times_out_without_pose_capture(self):
        with self.assertRaisesRegex(RuntimeError, "fresh stopped odometry"):
            self.finder.wait_for_measurement_pose()
        self.finder.wait_for_robot_pose.assert_not_called()

    def test_slow_clock_settling_allows_full_simulation_time_to_stop_and_capture(self):
        advance_clock = self.use_simulation_clock()
        self.finder.measurement_settle_timeout_s = 0.8
        self.finder._latest_motion = (self.clock.now(), 0.2, 0.0)

        def advance_motion():
            advance_clock()
            speed = 0.2 if self.clock.ros_seconds < 100.2 else 0.0
            self.finder._latest_motion = (self.clock.now(), speed, 0.0)

        self.clock.on_sleep = advance_motion

        self.assertIs(self.finder.wait_for_measurement_pose(), self.goal)

        self.assertGreater(self.clock.wall, self.finder.measurement_settle_timeout_s)
        self.assertGreaterEqual(self.clock.ros_seconds - 100.0, 0.6)
        self.assertLess(self.clock.ros_seconds - 100.0, 0.8)
        self.finder.wait_for_robot_pose.assert_called_once()
        self.assertEqual(self.finder._measurement_stamp, self.clock.now())

    def test_simulation_settling_timeout_reports_missing_motion_after_full_budget(self):
        self.use_simulation_clock()
        self.finder.measurement_settle_timeout_s = 0.2

        with self.assertRaisesRegex(RuntimeError, "ROS time limit.*real_time_factor=0.25"):
            self.finder.wait_for_measurement_pose()

        self.assertGreaterEqual(self.clock.wall, 0.8)
        self.assertLess(self.clock.wall, 0.9)
        self.finder.wait_for_robot_pose.assert_not_called()

    def test_capture_uses_pose_after_continuous_advancing_stopped_samples(self):
        self.feed_motion([
            (100.00, 100.00, 0.2, 0.0),
            (100.10, 100.10, 0.0, 0.0),
            (100.25, 100.25, 0.0, 0.0),
            (100.35, 100.35, 0.2, 0.0),  # Motion restarts the settling interval.
            (100.40, 100.40, 0.0, 0.0),
            (100.60, 100.60, 0.0, 0.0),
            (100.79, 100.79, 0.0, 0.0),
            (100.81, 100.81, 0.0, 0.0),
        ])
        measured = torch.tensor([1.02, 2.01, 0.01])
        capture_times = []

        def capture(timeout):
            capture_times.append(self.clock.ros_seconds)
            self.finder._measurement_stamp = self.clock.now()
            return measured

        self.finder.wait_for_robot_pose.side_effect = capture

        self.assertIs(self.finder.wait_for_measurement_pose(), measured)
        self.assertEqual(capture_times, [100.81])

    def test_stale_gap_restarts_settling_interval(self):
        self.feed_motion([
            (100.00, 100.00, 0.0, 0.0),
            (100.60, 100.00, 0.0, 0.0),  # Cached zero sample is now stale.
            (100.70, 100.70, 0.0, 0.0),
            (100.90, 100.90, 0.0, 0.0),
            (101.11, 101.11, 0.0, 0.0),
        ])
        capture_times = []

        def capture(timeout):
            capture_times.append(self.clock.ros_seconds)
            self.finder._measurement_stamp = self.clock.now()
            return self.goal

        self.finder.wait_for_robot_pose.side_effect = capture

        self.assertIs(self.finder.wait_for_measurement_pose(), self.goal)
        self.assertEqual(capture_times, [101.11])

    def test_missing_pose_is_not_accepted_after_odometry_settles(self):
        self.feed_motion([(100.0, 100.0, 0.0, 0.0), (100.5, 100.5, 0.0, 0.0)])
        self.finder.wait_for_robot_pose.return_value = None

        with self.assertRaisesRegex(RuntimeError, "fresh stopped odometry and pose"):
            self.finder.wait_for_measurement_pose()
        self.finder.wait_for_robot_pose.assert_called()

    def test_pose_from_before_stopping_cannot_satisfy_measurement_gate(self):
        self.feed_motion([(100.0, 100.0, 0.0, 0.0), (100.5, 100.5, 0.0, 0.0)])

        def capture_old_pose(timeout):
            # Recent under the general 2 s freshness limit, but before stopping.
            self.finder._measurement_stamp = ROS_TIME.from_sec(99.9)
            return self.goal

        self.finder.wait_for_robot_pose.side_effect = capture_old_pose

        with self.assertRaisesRegex(RuntimeError, "fresh stopped odometry and pose"):
            self.finder.wait_for_measurement_pose()
        self.finder.wait_for_robot_pose.assert_called()

    def test_waits_for_new_pose_after_rejecting_pose_from_before_stopping(self):
        self.feed_motion([(100.0, 100.0, 0.0, 0.0), (100.5, 100.5, 0.0, 0.0),
                          (100.6, 100.6, 0.0, 0.0)])
        capture_times = []

        def capture(timeout):
            capture_times.append(self.clock.ros_seconds)
            self.finder._measurement_stamp = (ROS_TIME.from_sec(99.9) if len(capture_times) == 1
                                               else self.clock.now())
            return self.goal

        self.finder.wait_for_robot_pose.side_effect = capture

        self.assertIs(self.finder.wait_for_measurement_pose(), self.goal)
        self.assertEqual(capture_times, [100.5, 100.6])

    def test_motion_resuming_during_pose_acquisition_invalidates_measurement(self):
        self.feed_motion([(100.0, 100.0, 0.0, 0.0), (100.5, 100.5, 0.0, 0.0)])

        def capture_after_motion_resumes(timeout):
            self.finder._measurement_stamp = self.clock.now()
            self.finder._latest_motion = (self.clock.now(), 0.2, 0.0)
            return self.goal

        self.finder.wait_for_robot_pose.side_effect = capture_after_motion_resumes

        with self.assertRaisesRegex(RuntimeError, "fresh stopped odometry and pose"):
            self.finder.wait_for_measurement_pose()
        self.finder.wait_for_robot_pose.assert_called()


if __name__ == "__main__":
    unittest.main()
