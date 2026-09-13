"""Verify source support through real PF operations without loading the NRP."""

from pathlib import Path
import sys
import unittest
from unittest import mock

import torch


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src/atdf_video/src"
sys.path.insert(0, str(SOURCE_DIR))
import atdf_video as atdf


class ParticleBoundsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.finder = atdf.ATDF.__new__(atdf.ATDF)
        self.finder.device = torch.device("cpu")
        self.finder.default_init_prior_bounds = None
        self.finder.default_init_x_range = None
        self.finder.default_init_y_range = None
        self.finder.forward_model = "nrp"
        self.finder.likelihood_temperature = 1.0
        axis = torch.arange(-2.0, 2.01, 0.5)
        self.finder.valid_tx_points = torch.cartesian_prod(axis, axis)
        self.finder.valid_rx_points = self.finder.valid_tx_points.clone()
        self.finder.is_valid_tx_xy = lambda xy: (
            torch.isfinite(xy).all(dim=-1) & (xy.abs() <= 2.0).all(dim=-1)
        )
        self.finder.is_valid_rx_xy = self.finder.is_valid_tx_xy
        self.bounds = (-0.5, 0.5, 0.0, 1.0)

    def assert_bounded(self, xy, bounds=None):
        bounds = self.bounds if bounds is None else bounds
        self.assertTrue(bool(self.finder.points_in_bounds(xy, bounds).all()))
        self.assertTrue(bool(self.finder.is_valid_tx_xy(xy).all()))

    def test_uniform_and_gaussian_initialization_obey_source_bounds(self):
        original_rx = self.finder.valid_rx_points.clone()
        original_tx = self.finder.valid_tx_points.clone()
        for center in (None, torch.tensor([0.0, 0.5])):
            with self.subTest(center=center):
                belief = self.finder.init_belief(
                    200, init_bounds=self.bounds, init_center=center, sigma_init=10.0
                )
                self.assert_bounded(belief["mu"])
                self.assertEqual(self.finder._tensor_to_bounds(belief["prior_bounds"]), self.bounds)
        torch.testing.assert_close(self.finder.valid_rx_points, original_rx)
        torch.testing.assert_close(self.finder.valid_tx_points, original_tx)
        # Robot support remains available outside the source rectangle.
        rx = self.finder.snap_points_to_valid(torch.tensor([2.0, 2.0]), role="rx")
        torch.testing.assert_close(rx, torch.tensor([2.0, 2.0]))

    def test_unbounded_default_retains_full_source_support(self):
        belief = self.finder.init_belief(1000)
        self.assertNotIn("prior_bounds", belief)
        self.assertFalse(bool(self.finder.points_in_bounds(belief["mu"], self.bounds).all()))
        self.assertEqual(float(belief["mu"].min()), -2.0)
        self.assertEqual(float(belief["mu"].max()), 2.0)

    def test_invalid_limits_fail_clearly(self):
        invalid = (
            (1.0, -1.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
            (-1.0, 1.0, 2.0, 1.0),
            (-1.0, 1.0, 1.0, 1.0),
            (float("nan"), 1.0, 0.0, 1.0),
            (-1.0, float("inf"), 0.0, 1.0),
            (-1.0, 1.0, float("-inf"), 1.0),
            {"x": [1.0, -1.0], "y": [0.0, 1.0]},
        )
        for limits in invalid:
            with self.subTest(limits=limits):
                with self.assertRaisesRegex(ValueError, "finite|minimum must be less"):
                    self.finder.init_belief(20, init_bounds=limits)

    def test_empty_valid_support_fails_for_uniform_and_gaussian_initialization(self):
        for center in (None, torch.tensor([3.25, 3.25])):
            with self.subTest(center=center):
                with self.assertRaisesRegex(ValueError, "No valid TX points inside requested bounds"):
                    self.finder.init_belief(
                        20, init_bounds=(3.0, 3.5, 3.0, 3.5), init_center=center
                    )

    def test_explicit_rectangle_overrides_configured_axis_defaults(self):
        self.finder.default_init_x_range = (-2.0, 2.0)
        self.finder.default_init_y_range = (-2.0, 2.0)
        self.assertEqual(self.finder.combine_prior_bounds(init_bounds=self.bounds), self.bounds)
        refined = self.finder.combine_prior_bounds(
            init_bounds=self.bounds, init_x_range=(-0.25, 0.25)
        )
        self.assertEqual(refined, (-0.25, 0.25, 0.0, 1.0))

    def test_center_outside_rectangle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "init_center .* outside init bounds"):
            self.finder.init_belief(20, init_bounds=self.bounds, init_center=torch.tensor([2.0, 2.0]))

    def test_estimate_snapping_around_obstacle_stays_in_rectangle(self):
        # Two valid in-bounds particles average into an obstacle. The closest
        # free map point is outside the rectangle, so unconstrained map snapping
        # would produce a source estimate outside the requested support.
        self.bounds = (-0.5, 0.5, 0.0, 1.0)
        support = torch.tensor([[-0.5, 0.0], [0.5, 0.0], [0.0, -0.1]])
        self.finder.valid_tx_points = support
        self.finder.is_valid_tx_xy = lambda xy: torch.isclose(
            xy[..., None, :], support, atol=1e-6
        ).all(dim=-1).any(dim=-1)
        for weights in (torch.tensor([0.5, 0.5]), torch.tensor([0.6, 0.4])):
            with self.subTest(weights=weights):
                belief = {
                    "mu": support[:2], "w": weights,
                    "prior_bounds": self.finder._bounds_to_tensor(self.bounds, "cpu"),
                }
                estimate = self.finder.particle_estimate(belief)
                self.assert_bounded(estimate)
                self.assertEqual(float(estimate[1]), 0.0)

    def test_repeated_updates_preserve_bounds_during_roughening_and_rejuvenation(self):
        self.finder.query_forward_rays_cross = mock.Mock(
            side_effect=lambda **kwargs: kwargs["tx_xy_batch"]
        )

        def concentrated_likelihood(observed, predicted):
            self.assert_bounded(predicted)
            likelihood = torch.full((len(predicted),), -100.0)
            likelihood[0] = 0.0
            return likelihood

        self.finder.ray_loglik_batch = concentrated_likelihood
        belief = self.finder.init_belief(100, init_bounds=self.bounds)
        # An imported invalid particle is repaired before the forward query.
        belief["mu"][0] = torch.tensor([2.0, 2.0])
        with mock.patch.object(
            self.finder, "sample_valid_points", wraps=self.finder.sample_valid_points
        ) as sampling:
            for _ in range(4):
                belief, stats = self.finder.particle_filter_update(
                    belief, torch.zeros(1, 6), torch.tensor([2.0, 2.0, 0.0]),
                    likelihood_mode="ray", resample_threshold=0.9,
                    roughening_scale=100.0, global_rejuvenation_frac=0.5,
                )
                self.assertEqual(stats["resampled"], 1.0)
                self.assert_bounded(belief["mu"])
                self.assert_bounded(self.finder.particle_estimate(belief))
                self.assertEqual(self.finder._tensor_to_bounds(belief["prior_bounds"]), self.bounds)
                self.assertAlmostEqual(float(belief["w"].sum()), 1.0, places=6)
        # A half-population global replacement happens at every update.
        global_calls = [call for call in sampling.call_args_list if call.args == (50,)]
        self.assertGreaterEqual(len(global_calls), 4)
        for call in sampling.call_args_list:
            self.assertEqual(call.kwargs["bounds"], self.bounds)


if __name__ == "__main__":
    unittest.main()
