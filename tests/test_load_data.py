import unittest
from unittest.mock import patch

import numpy as np

from data.load_data import (
    _load_filtered_arrays,
    _valid_dilepton_mass_rows,
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


class ValidDileptonMassRowsTest(unittest.TestCase):
    @staticmethod
    def _train_obj(*lepton_energies):
        """Back-to-back massless lepton pairs, so m_ll is twice the energy."""
        rows = np.zeros((len(lepton_energies), 18))
        for row, energy in enumerate(lepton_energies):
            rows[row, :4] = (energy, 0.0, 0.0, energy)
            rows[row, 4:8] = (-energy, 0.0, 0.0, energy)
        return rows

    def test_keeps_rows_below_the_bound(self):
        train_obj = self._train_obj(50.0, 62.0)

        np.testing.assert_array_equal(_valid_dilepton_mass_rows(train_obj, 125.0), [True, True])

    def test_rejects_rows_at_or_above_the_bound(self):
        train_obj = self._train_obj(62.5, 70.0)

        np.testing.assert_array_equal(_valid_dilepton_mass_rows(train_obj, 125.0), [False, False])


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


    def test_filters_rows_above_the_dilepton_mass_bound(self):
        # the surviving pair is (3, 0, 0, 5) and (-2, 0, 0, 5), so m_ll is 9.95 GeV
        with patch(
            "data.load_data.load_particles_from_h5", return_value={"sample": self._category()}
        ):
            unbounded, _ = _load_filtered_arrays("unused.h5", None, None)
            kept, _ = _load_filtered_arrays("unused.h5", None, None, max_dilepton_mass=10.0)
            dropped, _ = _load_filtered_arrays("unused.h5", None, None, max_dilepton_mass=9.0)

        self.assertEqual(unbounded.shape, (1, 18))
        self.assertEqual(kept.shape, (1, 18))
        self.assertEqual(dropped.shape, (0, 18))


if __name__ == "__main__":
    unittest.main()
