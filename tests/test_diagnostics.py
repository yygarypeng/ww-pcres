import unittest

import numpy as np

from physics.diagnostics import (
    component_calibration,
    histogram_total_variation,
    transverse_sharing_fraction,
)


class ComponentCalibrationTest(unittest.TestCase):
    def test_reports_affine_bias_width_and_event_accuracy(self):
        truth = np.array([[-1.0], [0.0], [1.0]])
        prediction = 2.0 * truth + 1.0

        metrics = component_calibration(prediction, truth)

        np.testing.assert_allclose(metrics["bias"], [1.0])
        np.testing.assert_allclose(metrics["width_ratio"], [2.0])
        np.testing.assert_allclose(metrics["slope"], [2.0])
        np.testing.assert_allclose(metrics["intercept"], [1.0])
        np.testing.assert_allclose(metrics["correlation"], [1.0])
        np.testing.assert_allclose(metrics["rmse"], [np.sqrt(5.0 / 3.0)])

    def test_ignores_nonfinite_prediction_truth_pairs(self):
        prediction = np.array([[1.0], [np.nan], [3.0]])
        truth = np.array([[1.0], [2.0], [np.inf]])

        metrics = component_calibration(prediction, truth)

        np.testing.assert_allclose(metrics["bias"], [0.0])
        np.testing.assert_allclose(metrics["rmse"], [0.0])
        self.assertTrue(np.isnan(metrics["width_ratio"][0]))


class DistributionDiagnosticsTest(unittest.TestCase):
    def test_transverse_sharing_fraction_preserves_event_alignment(self):
        nu0 = np.array([[3.0, 0.0], [0.0, 0.0], [-1.0, 1.0]])
        nu1 = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])

        alpha = transverse_sharing_fraction(nu0, nu1)

        np.testing.assert_allclose(alpha[[0, 2]], [0.75, 0.5])
        self.assertTrue(np.isnan(alpha[1]))

    def test_histogram_total_variation_is_zero_for_equal_samples(self):
        values = np.array([0.1, 0.2, 0.8, 0.9])

        distance = histogram_total_variation(values, values, bins=np.linspace(0.0, 1.0, 6))

        self.assertEqual(distance, 0.0)

    def test_histogram_total_variation_is_one_for_disjoint_samples(self):
        prediction = np.array([0.1, 0.2])
        truth = np.array([0.8, 0.9])

        distance = histogram_total_variation(prediction, truth, bins=np.linspace(0.0, 1.0, 6))

        self.assertEqual(distance, 1.0)


if __name__ == "__main__":
    unittest.main()
