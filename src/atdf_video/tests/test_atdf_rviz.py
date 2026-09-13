#!/usr/bin/env python3
"""Marker regression tests using ROS message classes, without a ROS master.

Run: source /opt/ros/noetic/setup.bash
     /usr/bin/python3 -m unittest discover -s src/atdf_video/tests -p test_atdf_rviz.py
"""

import io
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rospy
from visualization_msgs.msg import Marker

from atdf_rviz import LocalizationMarkers


class CapturingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        # Serialize as well: malformed scales, colors or stamps must not pass.
        message.serialize(io.BytesIO())
        self.messages.append(message)


class LocalizationMarkersTest(unittest.TestCase):
    def setUp(self):
        self.now = patch("atdf_rviz.rospy.Time.now", return_value=rospy.Time(10))
        self.now.start()
        self.addCleanup(self.now.stop)
        self.publisher = CapturingPublisher()
        self.overlay = LocalizationMarkers(publisher=self.publisher)

    def markers(self):
        return {marker.ns: marker for marker in self.publisher.messages[-1].markers}

    def headings(self):
        return [marker for marker in self.publisher.messages[-1].markers
                if marker.ns == "measurement_heading"]

    def test_publisher_is_latched(self):
        with patch("atdf_rviz.rospy.Publisher", return_value=self.publisher) as create:
            LocalizationMarkers()
        self.assertEqual(create.call_args.args[0], "/atdf_video/localization_markers")
        self.assertTrue(create.call_args.kwargs["latch"])
        self.assertEqual(create.call_args.kwargs["queue_size"], 1)

    def test_histories_preserve_repeated_steps_positions_and_stamps(self):
        steps = [((1, 2), (4, 5), rospy.Time(20)),
                 ((1, 2), (3, 4), rospy.Time(25)),
                 ((2, 3), (3, 3), rospy.Time(30))]
        for robot, estimate, stamp in steps:
            self.assertTrue(self.overlay.record_step(robot, estimate, stamp=stamp))
        markers = self.markers()
        for namespace, expected in (("measurement", [step[0] for step in steps]),
                                    ("estimate", [step[1] for step in steps])):
            self.assertEqual([(point.x, point.y)
                              for point in markers[namespace + "_path"].points], expected)
            self.assertEqual(len(markers[namespace + "_points"].points), 3)
            self.assertEqual(markers[namespace + "_path"].action, Marker.ADD)
            self.assertIn("RF step 3", markers[namespace + "_current_label"].text)
        self.assertEqual(self.overlay.measurement_stamps, [step[2] for step in steps])
        self.assertEqual(self.overlay.robot_yaw_history, [None, None, None])
        self.assertEqual(self.headings(), [])
        self.overlay.publish()
        self.assertEqual(len(self.overlay.robot_history), 3)
        for marker in self.publisher.messages[-1].markers:
            self.assertEqual(marker.header.frame_id, "map")
            self.assertEqual(marker.header.stamp, rospy.Time(30))
            self.assertEqual(marker.lifetime, rospy.Duration(0))

    def test_measured_headings_are_anchored_at_base_and_preserve_in_place_turns(self):
        self.overlay.record_step((2, 3, 0), (4, 5), stamp=rospy.Time(20))
        self.overlay.record_step((2, 3, math.pi / 2), (4, 4), stamp=rospy.Time(25))
        arrows = self.headings()
        self.assertEqual([arrow.id for arrow in arrows], [0, 1])
        for arrow in arrows:
            self.assertEqual(arrow.type, Marker.ARROW)
            self.assertEqual(arrow.action, Marker.ADD)
            self.assertEqual((arrow.points[0].x, arrow.points[0].y), (2, 3))
            self.assertGreater(arrow.points[0].z, 0.11)
            self.assertEqual((arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a),
                             self.overlay.ROBOT_COLOR)
        self.assertAlmostEqual(arrows[0].points[1].x, 2.35)
        self.assertAlmostEqual(arrows[0].points[1].y, 3)
        self.assertAlmostEqual(arrows[1].points[1].x, 2)
        self.assertAlmostEqual(arrows[1].points[1].y, 3.35)
        self.assertEqual(self.overlay.robot_history, [(2, 3), (2, 3)])
        self.assertEqual(self.overlay.robot_yaw_history, [0, math.pi / 2])
        self.assertEqual(self.overlay.measurement_stamps, [rospy.Time(20), rospy.Time(25)])
        self.assertEqual(len(self.markers()["measurement_points"].points), 2)
        self.assertEqual(len(self.markers()["measurement_path"].points), 2)
        self.overlay.publish()
        self.assertEqual(self.headings(), arrows)

    def test_heading_length_can_be_configured_and_visibility_toggled(self):
        overlay = LocalizationMarkers(publisher=self.publisher, heading_length=0.6)
        overlay.record_step((1, 2, math.pi), (3, 4))
        arrow = self.headings()[0]
        self.assertAlmostEqual(arrow.points[1].x, 0.4)
        self.assertAlmostEqual(arrow.points[1].y, 2)
        overlay.show_orientations = False
        overlay.publish()
        self.assertEqual(self.headings(), [])
        self.assertEqual(self.publisher.messages[-1].markers[0].action, Marker.DELETEALL)
        overlay.show_orientations = True
        overlay.publish()
        self.assertEqual(self.headings(), [arrow])

    def test_invalid_heading_does_not_change_histories_or_particle_cloud(self):
        self.overlay.record_step((1, 2, 0), (3, 4), particles=[(5, 6)], stamp=rospy.Time(20))
        with patch("atdf_rviz.rospy.logwarn"):
            for yaw in (float("nan"), float("inf"), None, "invalid"):
                self.assertFalse(self.overlay.record_step((2, 3, yaw), (4, 5),
                                                          particles=[(8, 9)]))
        self.assertEqual(self.overlay.robot_history, [(1, 2)])
        self.assertEqual(self.overlay.robot_yaw_history, [0])
        self.assertEqual(self.overlay.estimate_history, [(3, 4)])
        self.assertEqual(self.overlay.measurement_stamps, [rospy.Time(20)])
        self.assertEqual(self.overlay.particles, [(5, 6)])

    def test_latest_scene_deletes_old_headings_even_when_reset_message_is_missed(self):
        visible = {}

        def receive(message):
            for marker in message.markers:
                key = (marker.ns, marker.id)
                if marker.action == Marker.DELETEALL:
                    visible.clear()
                elif marker.action == Marker.DELETE:
                    visible.pop(key, None)
                else:
                    visible[key] = marker

        for restart in (False, True):
            for index in range(3):
                self.overlay.record_step((index, 2, index * math.pi / 2), (3, 4))
            receive(self.publisher.messages[-1])
            self.assertGreaterEqual(sum(ns == "measurement_heading" for ns, _ in visible), 3)
            if restart:
                self.overlay = LocalizationMarkers(publisher=self.publisher)
            else:
                self.overlay.reset()
            self.assertEqual(self.overlay.robot_yaw_history, [])
            # Model RViz receiving only the newest latched array after reset.
            self.overlay.record_step((0, 0, 0), (1, 1))
            receive(self.publisher.messages[-1])
            self.assertEqual([key for key in visible if key[0] == "measurement_heading"],
                             [("measurement_heading", 0)])

    def test_heading_length_rejects_nonpositive_or_nonfinite_values(self):
        for length in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                LocalizationMarkers(publisher=self.publisher, heading_length=length)

    def test_reset_clears_old_cloud_histories_and_labels(self):
        self.overlay.record_step((1, 2), (3, 4), particles=[(5, 6)])
        self.overlay.record_step((2, 2), (4, 4), particles=[(6, 6)])
        message = self.overlay.reset(ground_truth=(4, 6), stamp=rospy.Time(40))
        self.assertEqual(message.markers[0].action, Marker.DELETEALL)
        self.assertEqual(self.overlay.robot_history, [])
        self.assertEqual(self.overlay.robot_yaw_history, [])
        self.assertEqual(self.overlay.estimate_history, [])
        self.assertEqual(self.overlay.measurement_stamps, [])
        markers = self.markers()
        for namespace in ("measurement_path", "measurement_points", "measurement_current_label",
                          "estimate_path", "estimate_current_label", "RF_posterior_particles"):
            self.assertEqual(markers[namespace].action, Marker.DELETE)
        self.assertEqual(markers["ground_truth"].action, Marker.ADD)
        self.assertEqual(markers["ground_truth_label"].text, "RF source (ground truth)")
        self.assertAlmostEqual(sum(point.x for point in markers["ground_truth"].points) / 4, 4)
        self.assertAlmostEqual(sum(point.y for point in markers["ground_truth"].points) / 4, 6)
        self.overlay.record_step((0, 0), (1, 1), stamp=rospy.Time(45))
        self.assertEqual(self.overlay.robot_history, [(0, 0)])
        self.assertEqual(self.markers()["measurement_path"].action, Marker.DELETE)

    def test_invalid_positions_do_not_create_partial_measurements(self):
        self.overlay.record_step((1, 2), (3, 4), stamp=rospy.Time(20))
        with patch("atdf_rviz.rospy.logwarn"):
            self.assertFalse(self.overlay.record_step((float("nan"), 2), (3, 4)))
            self.assertFalse(self.overlay.record_step((1, 2), (3, float("inf"))))
            self.assertFalse(self.overlay.record_step((1,), (3, 4)))
        self.assertEqual(self.overlay.robot_history, [(1, 2)])
        self.assertEqual(self.overlay.estimate_history, [(3, 4)])
        self.assertEqual(self.overlay.measurement_stamps, [rospy.Time(20)])

    def test_particles_are_finite_purple_and_low_weights_remain_visible(self):
        with patch("atdf_rviz.rospy.logwarn"):
            self.overlay.record_step((1, 2), (3, 4),
                                     particles=[(4, 5), (float("nan"), 1), (6, 7)],
                                     weights=[0, 0.2, 0.8])
        cloud = self.markers()["RF_posterior_particles"]
        self.assertEqual([(point.x, point.y) for point in cloud.points], [(4, 5), (6, 7)])
        self.assertEqual(len(cloud.colors), 2)
        for color in cloud.colors:
            self.assertGreater(color.r, color.g)
            self.assertGreater(color.b, color.g)
            self.assertGreater(color.a, 0)
        self.assertGreater(cloud.colors[1].a, cloud.colors[0].a)
        self.assertNotEqual(self.overlay.PARTICLE_COLOR, self.overlay.ROBOT_COLOR)
        self.assertNotEqual(self.overlay.PARTICLE_COLOR, self.overlay.ESTIMATE_COLOR)
        self.overlay.show_particles = False
        self.overlay.publish()
        self.assertEqual(self.markers()["RF_posterior_particles"].action, Marker.DELETE)

    def test_particle_cap_is_display_only_and_truth_can_be_removed(self):
        self.overlay.max_particles = 3
        particles = [(index, index + 1) for index in range(10)]
        self.overlay.record_step((1, 2), (3, 4), particles=particles)
        self.assertEqual(len(self.markers()["RF_posterior_particles"].points), 3)
        self.assertEqual(particles, [(index, index + 1) for index in range(10)])
        self.overlay.set_ground_truth((4, 6))
        self.overlay.set_ground_truth(None)
        self.assertEqual(self.markers()["ground_truth"].action, Marker.DELETE)
        self.assertEqual(self.markers()["ground_truth_label"].action, Marker.DELETE)


if __name__ == "__main__":
    unittest.main()
