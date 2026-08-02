import unittest

import numpy as np

from data.load_data import (
    _valid_truth_w_rows,
    compute_mmd_condition_stats,
    mmd_condition_features,
)


class MMDConditionFeaturesTest(unittest.TestCase):
    def test_builds_periodic_condition_features_in_documented_order(self):
        features = np.zeros((2, 22), dtype=np.float32)
        features[:, 18:22] = [
            [10.0, -0.5, 0.0, np.pi / 2.0],
            [20.0, 0.5, np.pi, -np.pi / 2.0],
        ]

        condition = mmd_condition_features(features)

        expected = np.array([
            [10.0, -0.5, 0.0, 1.0, 1.0, 0.0],
            [20.0, 0.5, 0.0, -1.0, -1.0, 0.0],
        ])
        np.testing.assert_allclose(condition, expected, atol=1.0e-6)

    def test_statistics_are_fitted_to_transformed_training_features(self):
        features = np.zeros((3, 22), dtype=np.float32)
        features[:, 18:22] = [
            [10.0, -1.0, -0.5, -1.0],
            [20.0, 0.0, 0.0, 0.0],
            [30.0, 1.0, 0.5, 1.0],
        ]

        mean, scale = compute_mmd_condition_stats(features)
        transformed = mmd_condition_features(features)

        np.testing.assert_allclose(mean, transformed.mean(axis=0))
        np.testing.assert_allclose(scale, transformed.std(axis=0))

    def test_condition_features_require_all_four_observables(self):
        with self.assertRaisesRegex(ValueError, "at least 22 features"):
            mmd_condition_features(np.zeros((2, 21), dtype=np.float32))


class ValidTruthWRowsTest(unittest.TestCase):
    def test_accepts_positive_energy_timelike_vectors(self):
        target = np.array(
            [[3.0, 4.0, 0.0, 13.0, -3.0, -4.0, 0.0, 13.0, 12.0, 12.0]]
        )

        np.testing.assert_array_equal(_valid_truth_w_rows(target), [True])

    def test_rejects_negative_energy_vector(self):
        target = np.array(
            [[3.0, 4.0, 0.0, -13.0, -3.0, -4.0, 0.0, 30.0, 12.0, np.sqrt(875.0)]]
        )

        np.testing.assert_array_equal(_valid_truth_w_rows(target), [False])


if __name__ == "__main__":
    unittest.main()
