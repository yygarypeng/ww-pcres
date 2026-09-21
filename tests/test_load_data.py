import unittest
from unittest.mock import patch

import numpy as np

from data.load_data import _valid_truth_w_rows, load_data, load_presplit_data


class ValidTruthWRowsTest(unittest.TestCase):
    def test_accepts_positive_energy_timelike_vectors(self):
        target = np.array([[3.0, 4.0, 0.0, 13.0, -3.0, -4.0, 0.0, 13.0, 12.0, 12.0]])

        np.testing.assert_array_equal(_valid_truth_w_rows(target), [True])

    def test_rejects_negative_energy_vector(self):
        target = np.array([[3.0, 4.0, 0.0, -13.0, -3.0, -4.0, 0.0, 30.0, 12.0, np.sqrt(875.0)]])

        np.testing.assert_array_equal(_valid_truth_w_rows(target), [False])


class DileptonMassBoundTest(unittest.TestCase):
    """The m_ll bound load_data applies, exercised through the public loader."""

    @staticmethod
    def _kept_lepton_energies(*lepton_energies):
        """Load back-to-back massless lepton pairs, so m_ll is twice the energy."""
        count = len(lepton_energies)
        energies = np.asarray(lepton_energies)
        zeros = np.zeros(count)
        truth = dict(
            px=zeros, py=zeros, pz=zeros, energy=np.full(count, 12.0), m=np.full(count, 12.0)
        )
        category = {
            "pos_lep": dict(px=energies, py=zeros, pz=zeros, energy=energies),
            "neg_lep": dict(px=-energies, py=zeros, pz=zeros, energy=energies),
            "met": dict(px=zeros, py=zeros),
            "jets": {field: np.zeros((count, 2)) for field in ("px", "py", "pz", "energy")},
            "truth_pos_w": truth,
            "truth_neg_w": truth,
        }
        with patch("data.load_data.load_particles_from_h5", return_value={"sample": category}):
            train_obj, _ = load_data("unused.h5", ["sample"])
        return list(train_obj[:, 3])

    def test_keeps_rows_below_the_higgs_mass(self):
        self.assertEqual(self._kept_lepton_energies(50.0, 62.0), [50.0, 62.0])

    def test_rejects_rows_at_or_above_the_higgs_mass(self):
        # 62.5 GeV each puts m_ll exactly on the 125 GeV bound, which the cut excludes
        self.assertEqual(self._kept_lepton_energies(62.5, 70.0), [])


class TestInputEnergyValidation(unittest.TestCase):
    @staticmethod
    def _category():
        count = 2

        def group(**values):
            return {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}

        category = {
            "pos_lep": group(px=[3, 3], py=[0, 0], pz=[0, 0], energy=[5, 5]),
            "neg_lep": group(px=[-2, -2], py=[0, 0], pz=[0, 0], energy=[5, 5]),
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

    def test_event_numbers_drop_the_same_rows_as_the_inputs(self):
        category = self._category()
        category["event"] = {"eventNumber": np.array([41, 42], dtype=np.uint64)}

        with patch("data.load_data.load_particles_from_h5", return_value={"sample": category}):
            train_obj, _, event_numbers = load_data(
                "unused.h5", ["sample"], with_event_numbers=True
            )

        # The second row carries a negative jet energy, so only event 41 survives.
        self.assertEqual(train_obj.shape, (1, 18))
        np.testing.assert_array_equal(event_numbers, [41])

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
    def _arrays(_path, _categories, _max_events=None, with_event_numbers=False):
        arrays = (np.zeros((1, 18)), np.zeros((1, 10)))
        return arrays + (np.zeros(1, dtype=np.uint64),) if with_event_numbers else arrays

    def test_reads_the_fixed_ggf_groups_in_split_order(self):
        calls = []

        def record(path, categories, max_events=None, with_event_numbers=False):
            calls.append(list(categories))
            return self._arrays(path, categories, max_events, with_event_numbers)

        with patch("data.load_data.load_data", side_effect=record):
            splits = load_presplit_data("unused.h5")

        self.assertEqual(calls, [["ggF_train"], ["ggF_val"], ["ggF_test"]])
        self.assertEqual(len(splits), 6)

    def test_returns_event_numbers_per_split_when_requested(self):
        with patch("data.load_data.load_data", side_effect=self._arrays):
            splits = load_presplit_data("unused.h5", with_event_numbers=True)

        self.assertEqual(len(splits), 9)
        self.assertEqual([split.shape for split in splits[:3]], [(1, 18), (1, 10), (1,)])

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
