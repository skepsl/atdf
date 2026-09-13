"""Check physical RF query coordinates and their corresponding RViz snapshot."""

from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import torch


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src/atdf_video/src"
sys.path.insert(0, str(SOURCE_DIR))
import atdf_video as atdf


class FakeTester:
    def __init__(self):
        self.calls = []

    def inquire_ray(self, rx, tx):
        self.calls.append((rx.copy(), tx.copy()))
        count = len(rx) * len(tx)
        return torch.zeros(count, 4), [torch.ones(1, 6) for _ in range(count)]


class MeasurementCoordinateTests(unittest.TestCase):
    def setUp(self):
        self.finder = atdf.ATDF.__new__(atdf.ATDF)
        self.finder.device = torch.device("cpu")
        self.finder.tester = FakeTester()
        self.finder.rt_rx_chunk_size = 1
        self.finder.rt_tx_chunk_size = 1
        self.finder.snap_forward_rx_to_valid = True
        # Make any unintended map snapping immediately visible in query data.
        self.finder.snap_points_to_valid = mock.Mock(
            side_effect=lambda xy, role: xy + (10.0 if role == "rx" else -10.0))
        self.rx = torch.tensor([1.25, 2.5, 0.75])
        self.tx = torch.tensor([3.25, 4.5])
        self.finder.nrp_ray_top_k = 1
        self.finder.nrp_predict_batch = mock.Mock(return_value=(torch.ones(1, 1), torch.ones(1, 1, 6)))
        self.finder.nrp_outputs_to_ray_list = mock.Mock(return_value=[torch.ones(1, 6)])

    def test_sionna_measurement_uses_physical_source_and_antenna_coordinates(self):
        rays = self.finder.measurement_sionna_fn(self.rx, self.tx)
        self.assertEqual(tuple(rays.shape), (1, 6))
        self.assertEqual(len(self.finder.tester.calls), 1)
        rx_sent, tx_sent = self.finder.tester.calls[0]
        np.testing.assert_array_equal(rx_sent, self.rx[:2].numpy()[None, :])
        np.testing.assert_array_equal(tx_sent, self.tx.numpy()[None, :])
        self.finder.snap_points_to_valid.assert_not_called()
        np.testing.assert_array_equal(self.rx.numpy(), [1.25, 2.5, 0.75])
        np.testing.assert_array_equal(self.tx.numpy(), [3.25, 4.5])

    def test_sionna_hypothetical_forward_query_retains_map_snapping(self):
        self.finder.query_rt_rays_cross(self.rx, self.tx)
        rx_sent, tx_sent = self.finder.tester.calls[0]
        np.testing.assert_array_equal(rx_sent, (self.rx[:2] + 10.0).numpy()[None, :])
        np.testing.assert_array_equal(tx_sent, (self.tx - 10.0).numpy()[None, :])
        self.assertEqual(self.finder.snap_points_to_valid.call_count, 2)

    def test_nrp_measurement_uses_physical_source_and_antenna_coordinates(self):
        self.finder.measurement_nrp_fn(self.rx, self.tx)
        tx_sent, rx_sent = self.finder.nrp_predict_batch.call_args[0]
        torch.testing.assert_close(rx_sent, self.rx[:2].view(1, 2))
        torch.testing.assert_close(tx_sent, self.tx.view(1, 2))
        self.finder.snap_points_to_valid.assert_not_called()

    def test_nrp_hypothetical_forward_query_retains_map_snapping(self):
        self.finder.query_nrp_rays_cross(self.rx, self.tx)
        tx_sent, rx_sent = self.finder.nrp_predict_batch.call_args[0]
        torch.testing.assert_close(rx_sent, (self.rx[:2] + 10.0).view(1, 2))
        torch.testing.assert_close(tx_sent, (self.tx - 10.0).view(1, 2))
        self.assertEqual(self.finder.snap_points_to_valid.call_count, 2)


@unittest.skipUnless(atdf.ROS_AVAILABLE, "Source the ROS Noetic environment first")
class MeasurementVisualizationTests(unittest.TestCase):
    def test_debug_snapshot_keeps_full_belief_base_pose_and_acquisition_stamp(self):
        finder = atdf.ATDF.__new__(atdf.ATDF)
        finder.device = torch.device("cpu")
        finder._ros_ready = True
        finder.goal_frame = "map"
        finder.antenna_offset_x_m = 0.0
        finder.antenna_offset_y_m = -0.30
        finder.antenna_yaw_offset_rad = -np.pi / 2
        finder._rviz_markers = mock.Mock()
        finder._antenna_pose_pub = mock.Mock()
        finder._estimate_pub = mock.Mock()
        finder._particle_pub = mock.Mock()
        robot = torch.tensor([1.0, 2.0, np.pi / 2])
        estimate = torch.tensor([4.0, 6.0])
        belief = {"mu": torch.arange(2000, dtype=torch.float32).reshape(1000, 2),
                  "w": torch.arange(1, 1001, dtype=torch.float32)}
        stamp = atdf.rospy.Time(12, 345)

        with mock.patch.object(atdf.rospy.Time, "now", return_value=atdf.rospy.Time(99)):
            finder.publish_ros_debug(belief, estimate, robot, max_particles=7,
                                     measurement_stamp=stamp)

        finder._rviz_markers.record_step.assert_called_once()
        args, kwargs = finder._rviz_markers.record_step.call_args
        np.testing.assert_array_equal(args[0], robot.numpy())
        np.testing.assert_array_equal(args[1], estimate.numpy())
        np.testing.assert_array_equal(kwargs["particles"], belief["mu"].numpy())
        np.testing.assert_array_equal(kwargs["weights"], belief["w"].numpy())
        self.assertEqual(kwargs["stamp"], stamp)

        antenna_msg = finder._antenna_pose_pub.publish.call_args[0][0]
        self.assertAlmostEqual(antenna_msg.pose.position.x, 1.3, places=6)
        self.assertAlmostEqual(antenna_msg.pose.position.y, 2.0, places=6)
        self.assertAlmostEqual(antenna_msg.pose.orientation.z, 0.0, places=6)
        estimate_msg = finder._estimate_pub.publish.call_args[0][0]
        self.assertEqual((estimate_msg.pose.position.x, estimate_msg.pose.position.y), (4.0, 6.0))
        particles_msg = finder._particle_pub.publish.call_args[0][0]
        # The legacy PoseArray display cap does not truncate the marker belief.
        self.assertEqual(len(particles_msg.poses), 7)
        for message in (antenna_msg, estimate_msg, particles_msg):
            self.assertEqual(message.header.frame_id, "map")
            self.assertEqual(message.header.stamp, stamp)


if __name__ == "__main__":
    unittest.main()
