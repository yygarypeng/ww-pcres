import unittest
from unittest.mock import patch

import numpy as np

from data.load_data import (
    _valid_truth_w_rows,
    load_data,
)


class ValidTruthWRowsTest(unittest.TestCase):
    def test_accepts_positive_energy_timelike_vectors(self):
        target = np.array([[3.0, 4.0, 0.0, 13.0, -3.0, -4.0, 0.0, 13.0, 12.0, 12.0]])

        np.testing.assert_array_equal(_valid_truth_w_rows(target), [True])

    def test_rejects_negative_energy_vector(self):
        target = np.array([[3.0, 4.0, 0.0, -13.0, -3.0, -4.0, 0.0, 30.0, 12.0, np.sqrt(875.0)]])

        np.testing.assert_array_equal(_valid_truth_w_rows(target), [False])


class TestInputEnergyValidation(unittest.TestCase):
    @staticmethod
    def _category():
        count = 2

        def group(**values):
            return {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}

        category = {
            "pos_lep": group(
                px=[3, 3], py=[0, 0], pz=[0, 0], energy=[5, 5], pt=[3, 3], eta=[0, 0], phi=[0, 0]
            ),
            "neg_lep": group(
                px=[-2, -2], py=[0, 0], pz=[0, 0], energy=[5, 5], pt=[3, 3], eta=[0, 0], phi=[1, 1]
            ),
            "met": group(px=[0, 0], py=[0, 0]),
            "jets": {
                "px": np.array([[0, 0], [1, 0]], dtype=np.float64),
                "py": np.zeros((count, 2)),
                "pz": np.zeros((count, 2)),
                "energy": np.array([[0, 0], [-1, 0]], dtype=np.float64),
            },
        }
        truth = dict(px=[0, 0], py=[0, 0], pz=[0, 0], energy=[12, 12], m=[12, 12])
        category["truth_pos_w"] = group(**truth)
        category["truth_neg_w"] = group(**truth)
        return category

    def test_filters_invalid_input_energy_rows_before_fitting_statistics(self):
        captured = {}

        def capture_stats(train_obj, target_obj):
            captured["train_obj"] = train_obj.copy()
            return ((np.zeros(18), np.ones(18)), (np.zeros(10), np.ones(10)))

        with (
            patch(
                "data.load_data.load_particles_from_h5", return_value={"sample": self._category()}
            ),
            patch("data.load_data.compute_standardization_stats", side_effect=capture_stats),
        ):
            train_obj, target_obj, _, _ = load_data("unused.h5")

        self.assertEqual(train_obj.shape, (1, 18))
        self.assertEqual(target_obj.shape, (1, 10))
        np.testing.assert_array_equal(train_obj[:, 8:12], np.zeros((1, 4)))
        np.testing.assert_array_equal(captured["train_obj"], train_obj)


if __name__ == "__main__":
    unittest.main()
