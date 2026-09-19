import unittest
from unittest.mock import patch

import numpy as np

from data.load_data import (
    _valid_dilepton_mass_rows,
    _valid_truth_w_rows,
    load_data,
    load_presplit_data,
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

    def test_keeps_rows_below_the_higgs_mass(self):
        train_obj = self._train_obj(50.0, 62.0)

        np.testing.assert_array_equal(_valid_dilepton_mass_rows(train_obj), [True, True])

    def test_rejects_rows_at_or_above_the_higgs_mass(self):
        train_obj = self._train_obj(62.5, 70.0)

        np.testing.assert_array_equal(_valid_dilepton_mass_rows(train_obj), [False, False])


class TestInputEnergyValidation(unittest.TestCase):
    @staticmethod
    def _category():
        count = 2

        def group(**values):
            return {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}

        category = {
            "pos_lep": group(
                px=[3, 3], py=[0, 0], pz=[0, 0], energy=[5, 5]
            ),
            "neg_lep": group(
                px=[-2, -2], py=[0, 0], pz=[0, 0], energy=[5, 5]
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

    def test_drops_rows_with_invalid_input_energies(self):
        with patch(
            "data.load_data.load_particles_from_h5", return_value={"sample": self._category()}
        ):
            train_obj, target_obj = load_data("unused.h5", ["sample"])

        self.assertEqual(train_obj.shape, (1, 18))
        self.assertEqual(target_obj.shape, (1, 10))
        np.testing.assert_array_equal(train_obj[:, 8:12], np.zeros((1, 4)))

    def test_always_applies_the_higgs_mass_bound(self):
        # the surviving pair is (3, 0, 0, 5) and (-2, 0, 0, 5), so m_ll is 9.95 GeV
        over_bound = self._category()
        for lepton, sign in (("pos_lep", 1.0), ("neg_lep", -1.0)):
            over_bound[lepton]["px"] = np.array([sign * 3.0, sign * 80.0])
            over_bound[lepton]["energy"] = np.array([5.0, 80.0])

        with patch("data.load_data.load_particles_from_h5", return_value={"sample": over_bound}):
            kept, _ = load_data("unused.h5", ["sample"])

        # only the 9.95 GeV pair survives; the 160 GeV pair is above the Higgs mass
        self.assertEqual(kept.shape, (1, 18))
        np.testing.assert_allclose(kept[0, :4], [3.0, 0.0, 0.0, 5.0])


class LoadPresplitDataTest(unittest.TestCase):
    @staticmethod
    def _arrays(_path, _categories, _max_events=None):
        return np.zeros((1, 18)), np.zeros((1, 10))

    def test_reads_the_fixed_ggf_groups_in_split_order(self):
        calls = []

        def record(path, categories, max_events=None):
            calls.append(list(categories))
            return self._arrays(path, categories, max_events)

        with patch("data.load_data.load_data", side_effect=record):
            splits = load_presplit_data("unused.h5")

        self.assertEqual(calls, [["ggF_train"], ["ggF_val"], ["ggF_test"]])
        self.assertEqual(len(splits), 6)

    def test_forwards_max_events_per_category(self):
        with patch("data.load_data.load_data", side_effect=self._arrays) as fake:
            load_presplit_data("unused.h5", {"max_events_per_category": 7})

        self.assertEqual({call.args[2] for call in fake.call_args_list}, {7})

    def test_rejects_unknown_data_config_keys(self):
        with patch("data.load_data.load_data", side_effect=self._arrays) as fake:
            with self.assertRaises(ValueError) as error:
                load_presplit_data("unused.h5", {"val_frac": 0.05, "categories": None})

        message = str(error.exception)
        self.assertIn("val_frac", message)
        self.assertIn("categories", message)
        fake.assert_not_called()
