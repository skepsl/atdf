"""Check physical coordinates and uncertainty survive paper-axis formatting."""

from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.quiver import Quiver
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/atdf_video/src"))
from atdf_plot import make_reference_plot


class ReferencePlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_axis_swap_keeps_map_points_and_heading_in_same_frame(self):
        pixels = np.arange(6).reshape(2, 3) / 5.0
        robot = np.array([[1., 2., 0.], [3., 4., np.pi / 2]])
        fig = make_reference_plot(pixels, [-1, 5, -2, 6], robot,
                                  estimate_traj=[[2., 3.], [4., 5.]], ground_truth=[0., 1.],
                                  roi_bounds=[-1, 4, None, 5])
        ax = fig.axes[0]
        np.testing.assert_array_equal(ax.images[0].get_array(), pixels.T)
        np.testing.assert_array_equal(ax.images[0].get_extent(), [-2, 6, -1, 5])
        np.testing.assert_array_equal(ax.lines[0].get_xydata(), [[3, 2], [5, 4]])
        np.testing.assert_array_equal(ax.lines[1].get_xydata(), [[2, 1], [4, 3]])
        arrows = next(c for c in ax.collections if isinstance(c, Quiver))
        np.testing.assert_allclose(arrows.U, [0, .35], atol=1e-12)
        np.testing.assert_allclose(arrows.V, [.35, 0], atol=1e-12)
        box = ax.patches[0]
        self.assertEqual(box.get_xy(), (-2., -1.))
        self.assertEqual(box.get_width(), 7.)
        self.assertEqual(box.get_height(), 5.)

    def test_two_sigma_covariance_is_transformed_with_its_mean(self):
        fig = make_reference_plot(np.ones((2, 2)), [0, 10, 0, 10], [],
                                  particles=[[3., 4.]], weights=[1.],
                                  covariances=[[[4., 0.], [0., 1.]]])
        collection = next(c for c in fig.axes[0].collections if isinstance(c, PatchCollection))
        extent = collection.get_paths()[0].get_extents()
        # World sigma_x=2 and sigma_y=1 become displayed vertical/horizontal.
        np.testing.assert_allclose([extent.xmin, extent.xmax, extent.ymin, extent.ymax],
                                   [2, 6, -1, 7], atol=1e-10)

    def test_bootstrap_particles_have_no_fabricated_covariance(self):
        fig = make_reference_plot(np.ones((2, 2)), [0, 10, 0, 10], [],
                                  particles=[[3., 4.]], weights=[1.], covariances=np.zeros((1, 2, 2)))
        self.assertFalse(any(isinstance(c, PatchCollection) for c in fig.axes[0].collections))
        labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        self.assertIn("Source particles", labels)
        self.assertNotIn("Particle Cov. Ours", labels)

    def test_plotting_does_not_change_random_state_or_drop_history(self):
        state = np.random.get_state()
        robot = np.array([[1., 2., 0.], [1., 2., 1.], [3., 4., 2.]])
        fig = make_reference_plot(np.ones((2, 2)), [0, 10, 0, 10], robot,
                                  particles=np.arange(40).reshape(20, 2) / 4,
                                  weights=np.ones(20), max_particles=3)
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        np.testing.assert_array_equal(fig.axes[0].lines[0].get_xydata(), robot[:, [1, 0]])


if __name__ == "__main__":
    unittest.main()
