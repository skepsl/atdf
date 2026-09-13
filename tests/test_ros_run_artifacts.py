"""Exercise artifact persistence in the ROS loop without ROS or a model."""

import inspect
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src/atdf_video/src"
sys.path.insert(0, str(SOURCE_DIR))
import atdf_video as atdf


class RosRunArtifactsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fake_ros = SimpleNamespace(
            is_shutdown=mock.Mock(return_value=False),
            loginfo=mock.Mock(), logwarn=mock.Mock(), logerr=mock.Mock(),
            get_param=lambda name, default=None: default,
            Time=SimpleNamespace(now=lambda: self.stamp(99.0)),
        )
        patcher = mock.patch.object(atdf, "rospy", self.fake_ros)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.finder = atdf.ATDF.__new__(atdf.ATDF)
        finder = self.finder
        finder.device = torch.device("cpu")
        finder._ros_ready = True
        finder._rviz_markers = None
        finder.rand_idx = 42
        finder.forward_model = "nrp"
        finder.measurement_source = "real_iq"
        finder.goal_frame = "map"
        finder.antenna_offset_x_m = 0.0
        finder.antenna_offset_y_m = -0.3
        finder.antenna_yaw_offset_rad = -np.pi / 2
        self.robot = torch.tensor([1., 2., np.pi / 2], dtype=torch.float32)
        self.truth = torch.tensor([4., 6.])
        self.prior = {
            "mu": torch.tensor([[1., 3.], [2., 4.], [3., 5.]]),
            "w": torch.tensor([0.25, 0.25, 0.5]),
            "Sigma": torch.eye(2).repeat(3, 1, 1),
            "prior_bounds": torch.tensor([-4., 5., 0., 8.]),
        }
        self.posterior = {
            "mu": self.prior["mu"] + 0.25,
            "w": torch.tensor([0.1, 0.2, 0.7]),
            "Sigma": self.prior["Sigma"] * 0.2,
            "prior_bounds": self.prior["prior_bounds"].clone(),
        }
        self.iq = torch.tensor([[1. + 2.j, 3. - 4.j], [-5. + 6.j, 7. + 8.j]],
                               dtype=torch.complex64)
        finder.init_belief = mock.Mock(return_value=self.prior)
        finder.wait_for_robot_pose = mock.Mock(return_value=self.robot.clone())
        finder.send_robot_goal_and_wait = mock.Mock(return_value=True)
        finder.snap_rx_pose_to_valid = lambda pose: pose
        finder.wait_for_measurement_pose = mock.Mock(side_effect=self.measurement_pose)
        finder.measurement_from_source = mock.Mock(return_value=self.iq)
        finder.particle_filter_update = mock.Mock(return_value=(self.posterior, {
            "ll_min": -2., "ll_max": -1., "resampled": 0.,
        }))
        finder.particle_estimate = lambda belief: (belief["mu"] * belief["w"][:, None]).sum(0)
        finder.uncertainty_scalar = lambda belief: 2.
        finder.mixture_mean_and_cov = lambda belief: (finder.particle_estimate(belief), torch.eye(2))
        finder.publish_ros_debug = mock.Mock()
        finder.plot_valid_mask = mock.Mock()
        finder.plot_state = mock.Mock()
        finder.generate_candidates = mock.Mock(return_value=self.robot.view(1, 3))
        finder.select_next_robot_pose = mock.Mock(return_value=self.robot.clone())
        finder.path_occupancy_stats = mock.Mock(return_value={"dist": 0.})

    @staticmethod
    def stamp(seconds):
        return SimpleNamespace(to_sec=lambda: seconds, secs=int(seconds),
                               nsecs=int(round((seconds % 1) * 1e9)))

    def measurement_pose(self):
        step = self.finder.wait_for_measurement_pose.call_count
        self.finder._measurement_stamp = self.stamp(12.25 + step - 1)
        return self.robot + torch.tensor([float(step - 1), 0., 0.])

    def run_finder(self, **kwargs):
        options = dict(tx_true_xy=self.truth, rx_init_pose=torch.tensor([0., 0., 1.]),
                       N_p=3, T_max=1, plot_dir=str(self.root), planning_mode="none",
                       verbose=False)
        options.update(kwargs)
        return self.finder.run_ros_active_localization(**options)

    def run_directory(self):
        directories = list(self.root.glob("run_*"))
        self.assertEqual(len(directories), 1)
        return directories[0]

    def assert_snapshot(self, directory, step, belief):
        path = directory / "step_{:06d}.npz".format(step)
        self.assertTrue(path.with_suffix(".json").is_file())
        with np.load(path, allow_pickle=False) as saved:
            for key in ("mu", "w", "Sigma"):
                np.testing.assert_array_equal(saved["belief_" + key], belief[key].numpy())
            np.testing.assert_array_equal(saved["prior_bounds"], self.prior["prior_bounds"].numpy())
            np.testing.assert_array_equal(saved["tx_true_xy"], self.truth.numpy())
            return {key: saved[key] for key in saved.files}

    def test_disabling_pdfs_still_preserves_full_particles_iq_and_heading(self):
        result = self.run_finder(plot_every=0, init_bounds=self.prior["prior_bounds"])
        directory = Path(result["artifact_dir"])
        self.assertEqual(directory, self.run_directory())
        saved = self.assert_snapshot(directory, 1, self.posterior)
        np.testing.assert_array_equal(saved["measurement"], self.iq.numpy())
        self.assertEqual(saved["measurement"].dtype, np.complex64)
        np.testing.assert_array_equal(saved["robot_traj"], self.robot.numpy()[None])
        antenna = self.finder.base_to_antenna_pose(self.robot).numpy()[None]
        np.testing.assert_array_equal(saved["antenna_traj"], antenna)
        self.assertAlmostEqual(saved["robot_traj"][0, 2], np.pi / 2, places=6)
        self.assertAlmostEqual(saved["antenna_traj"][0, 2], 0., places=6)
        np.testing.assert_array_equal(saved["measurement_stamps"], [12.25])
        for key in ("est_traj", "unc_traj", "err_traj", "ess_traj"):
            np.testing.assert_array_equal(saved[key], result[key].numpy())
        self.finder.plot_state.assert_not_called()
        self.finder.plot_valid_mask.assert_not_called()

    def test_initial_prior_is_saved_before_first_pose_wait_or_navigation(self):
        def inspect_before_navigation(**kwargs):
            self.assert_snapshot(self.run_directory(), 0, self.prior)
            return self.robot.clone()
        self.finder.wait_for_robot_pose.side_effect = inspect_before_navigation
        self.run_finder(plot_every=0)
        self.finder.wait_for_robot_pose.assert_called()
        self.finder.send_robot_goal_and_wait.assert_called_once()

    def test_measurement_snapshot_exists_before_debug_publication(self):
        def inspect_before_publication(*args, **kwargs):
            saved = self.assert_snapshot(self.run_directory(), 1, self.posterior)
            np.testing.assert_array_equal(saved["measurement"], self.iq.numpy())
        self.finder.publish_ros_debug.side_effect = inspect_before_publication
        self.run_finder(plot_every=0)
        self.finder.publish_ros_debug.assert_called_once()

    def test_plot_failure_cannot_destroy_completed_measurement_snapshot(self):
        self.finder.plot_state.side_effect = [None, RuntimeError("render failed")]
        try:
            self.run_finder(plot_every=1)
        except RuntimeError as exc:
            self.assertEqual(str(exc), "render failed")
        self.assertEqual(self.finder.plot_state.call_count, 2)
        saved = self.assert_snapshot(self.run_directory(), 1, self.posterior)
        np.testing.assert_array_equal(saved["measurement"], self.iq.numpy())

    def test_unwritable_output_aborts_before_pose_wait_and_navigation(self):
        blocker = self.root / "regular_file"
        blocker.write_text("Existing user data")
        with self.assertRaises(OSError):
            self.run_finder(plot_dir=str(blocker / "plots"), plot_every=0)
        self.finder.wait_for_robot_pose.assert_not_called()
        self.finder.send_robot_goal_and_wait.assert_not_called()
        self.finder.measurement_from_source.assert_not_called()
        self.assertEqual(blocker.read_text(), "Existing user data")

    def test_interruption_on_next_measurement_preserves_completed_step(self):
        original = self.measurement_pose
        def interrupt_second_measurement():
            if self.finder.wait_for_measurement_pose.call_count > 1:
                raise RuntimeError("measurement interrupted")
            return original()
        self.finder.wait_for_measurement_pose.side_effect = interrupt_second_measurement
        with self.assertRaisesRegex(RuntimeError, "measurement interrupted"):
            self.run_finder(T_max=2, plot_every=0)
        directory = self.run_directory()
        saved = self.assert_snapshot(directory, 1, self.posterior)
        np.testing.assert_array_equal(saved["robot_traj"], self.robot.numpy()[None])
        self.assertFalse((directory / "step_000002.npz").exists())

    def test_pdf_saving_defaults_to_every_measurement(self):
        signature = inspect.signature(atdf.ATDF.run_ros_active_localization)
        self.assertEqual(signature.parameters["plot_every"].default, 1)
        self.run_finder()
        self.assertEqual(self.finder.plot_state.call_count, 2)


if __name__ == "__main__":
    unittest.main()
