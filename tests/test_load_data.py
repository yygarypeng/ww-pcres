import unittest

import numpy as np

from data.load_data import _valid_truth_w_rows


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
