#!/usr/bin/env python3
"""Persistent RViz overlays for the RF localization experiment.

All inputs are already in ``frame_id``. In particular, ``robot_xy`` is the
measured robot BASE pose, not the antenna position or a navigation goal.
Its optional third coordinate is the measured yaw in radians.
Call ``record_step`` once after each successful RF posterior update. Histories
are never distance-filtered: two measurements at the same pose are two steps.
The latched, complete MarkerArray also works when RViz joins after a run.
"""

import math

import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class LocalizationMarkers:
    """ROS-only visualization; callers convert tensors to CPU arrays first."""

    ROBOT_COLOR = (0.0, 0.40, 0.80, 1.0)
    ESTIMATE_COLOR = (0.95, 0.48, 0.0, 1.0)
    TRUTH_COLOR = (0.85, 0.05, 0.12, 1.0)
    PARTICLE_COLOR = (0.64, 0.12, 0.82, 0.45)

    def __init__(self, frame_id="map", topic="/atdf_video/localization_markers",
                 show_particles=True, max_particles=1000, publisher=None,
                 show_orientations=True, heading_length=0.35):
        self.frame_id = frame_id
        self.show_particles = bool(show_particles)
        self.max_particles = max(1, int(max_particles))
        self.show_orientations = bool(show_orientations)
        self.heading_length = float(heading_length)
        if not math.isfinite(self.heading_length) or self.heading_length <= 0.0:
            raise ValueError("heading_length must be finite and positive")
        self.publisher = publisher if publisher is not None else rospy.Publisher(
            topic, MarkerArray, queue_size=1, latch=True)
        self.robot_history = []
        self.robot_yaw_history = []
        self.estimate_history = []
        self.measurement_stamps = []
        self.particles = []
        self.particle_alphas = []
        self.ground_truth = None
        self._stamp = None
        self.reset()

    @staticmethod
    def _xy(position):
        try:
            xy = (float(position[0]), float(position[1]))
        except (TypeError, ValueError, IndexError, OverflowError) as exc:
            raise ValueError("position must contain finite x and y coordinates") from exc
        if not all(math.isfinite(value) for value in xy):
            raise ValueError("position must contain finite x and y coordinates")
        return xy

    @staticmethod
    def _yaw(position):
        """Return an optional measured yaw; do not infer it from the path."""
        try:
            value = position[2]
        except IndexError:
            return None
        try:
            yaw = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("robot yaw must be finite when supplied") from exc
        if not math.isfinite(yaw):
            raise ValueError("robot yaw must be finite when supplied")
        return yaw

    def reset(self, ground_truth=None, stamp=None):
        """Start a new run, deleting every old marker and measurement."""
        truth = None if ground_truth is None else self._xy(ground_truth)
        self.robot_history.clear()
        self.robot_yaw_history.clear()
        self.estimate_history.clear()
        self.measurement_stamps.clear()
        self.particles.clear()
        self.particle_alphas.clear()
        self.ground_truth = truth
        self._stamp = stamp if stamp is not None else rospy.Time.now()
        return self.publish(clear=True)

    def set_ground_truth(self, position, stamp=None):
        """Set the simulation source, or pass None to remove it."""
        self.ground_truth = None if position is None else self._xy(position)
        return self.publish(stamp=stamp)

    def record_step(self, robot_xy, estimate_xy, particles=None, stamp=None,
                    weights=None):
        """Append one paired measurement/posterior and publish the full scene.

        ``stamp`` is the measurement acquisition time, retained in
        ``measurement_stamps`` even when positions repeat. Invalid paired
        positions or robot yaw leave all histories unchanged. Pass robot
        ``[x, y, yaw]`` to draw its heading, or ``[x, y]`` without a heading.
        Nonfinite particles are
        skipped; finite particles remain visible even with zero weight.
        ``particles=None`` retains the previous posterior cloud.
        """
        try:
            robot = self._xy(robot_xy)
            yaw = self._yaw(robot_xy)
            estimate = self._xy(estimate_xy)
        except ValueError as exc:
            rospy.logwarn("Skipping invalid RF visualization step: %s", exc)
            return False
        if particles is not None:
            self._set_particles(particles, weights)
        self._stamp = stamp if stamp is not None else rospy.Time.now()
        self.robot_history.append(robot)
        self.robot_yaw_history.append(yaw)
        self.estimate_history.append(estimate)
        self.measurement_stamps.append(self._stamp)
        self.publish()
        return True

    def _set_particles(self, particles, weights):
        rows = list(particles)
        weight_values = list(weights) if weights is not None else []
        weighted = len(weight_values) == len(rows)
        valid = []
        invalid_count = 0
        for index, position in enumerate(rows):
            try:
                xy = self._xy(position)
            except ValueError:
                invalid_count += 1
                continue
            weight = 1.0
            if weighted:
                try:
                    weight = float(weight_values[index])
                except (TypeError, ValueError, OverflowError):
                    weight = 0.0
                if not math.isfinite(weight) or weight < 0.0:
                    weight = 0.0
            valid.append((xy, weight))
        if invalid_count:
            rospy.logwarn("Omitted %d nonfinite RF particles from RViz", invalid_count)
        if len(valid) > self.max_particles:
            # Spread a display-only subset over the input; never alter the PF.
            count = len(valid)
            valid = [valid[index * count // self.max_particles]
                     for index in range(self.max_particles)]
        maximum = max((weight for _, weight in valid), default=0.0)
        self.particles = [xy for xy, _ in valid]
        self.particle_alphas = [
            0.18 + 0.47 * math.sqrt(weight / maximum)
            if weighted and maximum > 0.0 else self.PARTICLE_COLOR[3]
            for _, weight in valid
        ]

    @staticmethod
    def _point(xy, z):
        return Point(x=xy[0], y=xy[1], z=z)

    def _marker(self, namespace, marker_type, color, scale):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self._stamp
        marker.ns = namespace
        marker.id = 0
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = scale
        marker.color = ColorRGBA(*color)
        marker.lifetime = rospy.Duration(0)
        marker.frame_locked = False
        return marker

    def _series(self, namespace, positions, color, z):
        line = self._marker(namespace + "_path", Marker.LINE_STRIP,
                            color, (0.045, 0.0, 0.0))
        line.points = [self._point(xy, z) for xy in positions]
        if len(positions) < 2:
            line.action = Marker.DELETE
        points = self._marker(namespace + "_points", Marker.SPHERE_LIST,
                              color, (0.13, 0.13, 0.05))
        points.points = [self._point(xy, z + 0.01) for xy in positions]
        if not positions:
            points.action = Marker.DELETE
        markers = [line, points]
        label_name = "Robot" if namespace == "measurement" else "Estimate"
        for endpoint, marker_type, size in (("start", Marker.CUBE, 0.20),
                                            ("current", Marker.SPHERE, 0.25)):
            endpoint_marker = self._marker(namespace + "_" + endpoint,
                                           marker_type, color, (size, size, 0.06))
            label = self._marker(namespace + "_" + endpoint + "_label",
                                 Marker.TEXT_VIEW_FACING, color, (0.0, 0.0, 0.25))
            if positions:
                xy = positions[0] if endpoint == "start" else positions[-1]
                endpoint_marker.pose.position = self._point(xy, z + 0.03)
                label.pose.position = self._point((xy[0], xy[1] + 0.30), z + 0.25)
                label.text = (label_name + " start" if endpoint == "start" else
                              "%s | RF step %d" % (label_name, len(positions)))
                if len(positions) == 1 and endpoint == "current":
                    endpoint_marker.action = Marker.DELETE
                    label.action = Marker.DELETE
            else:
                endpoint_marker.action = Marker.DELETE
                label.action = Marker.DELETE
            markers.extend((endpoint_marker, label))
        return markers

    def _measurement_headings(self):
        markers = []
        if not self.show_orientations:
            return markers
        length = self.heading_length
        scale = (min(0.045, 0.15 * length), min(0.10, 0.30 * length),
                 min(0.10, 0.30 * length))
        for index, (xy, yaw) in enumerate(zip(self.robot_history, self.robot_yaw_history)):
            if yaw is None:
                continue
            arrow = self._marker("measurement_heading", Marker.ARROW,
                                 self.ROBOT_COLOR, scale)
            arrow.id = index
            arrow.points = [self._point(xy, 0.16), self._point(
                (xy[0] + length * math.cos(yaw), xy[1] + length * math.sin(yaw)),
                0.16)]
            markers.append(arrow)
        return markers

    def publish(self, stamp=None, clear=False):
        """Republish without appending a step; returns the published array.

        Keep the owning ROS node alive after convergence to preserve the latch.
        The last RF timestamp is retained unless an explicit stamp is supplied.
        Every message replaces the whole scene, including old heading arrows
        after a reset or node restart. ``clear`` is retained for compatibility.
        """
        if stamp is not None:
            self._stamp = stamp
        # A reset-only message may be replaced in the publisher's queue/latch
        # before RViz receives it. Make every scene independently complete.
        delete = self._marker("reset", Marker.SPHERE, self.TRUTH_COLOR,
                              (1.0, 1.0, 1.0))
        delete.action = Marker.DELETEALL
        markers = [delete]
        cloud = self._marker("RF_posterior_particles", Marker.POINTS,
                             self.PARTICLE_COLOR, (0.055, 0.055, 0.0))
        if self.show_particles and self.particles:
            cloud.points = [self._point(xy, 0.035) for xy in self.particles]
            cloud.colors = [ColorRGBA(*self.PARTICLE_COLOR[:3], alpha)
                            for alpha in self.particle_alphas]
        else:
            cloud.action = Marker.DELETE
        markers.append(cloud)
        markers.extend(self._series("measurement", self.robot_history,
                                    self.ROBOT_COLOR, 0.08))
        markers.extend(self._measurement_headings())
        markers.extend(self._series("estimate", self.estimate_history,
                                    self.ESTIMATE_COLOR, 0.12))
        truth = self._marker("ground_truth", Marker.LINE_LIST,
                             self.TRUTH_COLOR, (0.08, 0.0, 0.0))
        label = self._marker("ground_truth_label", Marker.TEXT_VIEW_FACING,
                             self.TRUTH_COLOR, (0.0, 0.0, 0.28))
        if self.ground_truth is None:
            truth.action = Marker.DELETE
            label.action = Marker.DELETE
        else:
            x, y = self.ground_truth
            truth.points = [self._point(xy, 0.20) for xy in
                            ((x - 0.23, y - 0.23), (x + 0.23, y + 0.23),
                             (x - 0.23, y + 0.23), (x + 0.23, y - 0.23))]
            label.pose.position = self._point((x, y - 0.40), 0.32)
            label.text = "RF source (ground truth)"
        markers.extend((truth, label))
        message = MarkerArray(markers=markers)
        self.publisher.publish(message)
        return message
