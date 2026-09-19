import unittest

import numpy as np

from physics.diagnostics import component_calibration


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
