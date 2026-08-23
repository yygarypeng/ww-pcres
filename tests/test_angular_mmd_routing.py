import unittest

import numpy as np

from model import LightningWBoson


class AngularMmdRoutingTest(unittest.TestCase):
    def test_absolute_bandwidths_are_routed_to_angular_mmd(self):
        model = LightningWBoson(
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(21, dtype=np.float32),
            std_scale_train=np.ones(21, dtype=np.float32),
            mmd_config={"angular": {"bandwidths": [0.5, 1.0, 2.0, 4.0]}},
            attention_blocks=1,
            attention_dropout=0.0,
            decoder_dropout=0.0,
        )

        angular = model._mmd_kwargs("angular")
        mass = model._mmd_kwargs("mass")

        self.assertEqual(angular["bandwidths"], [0.5, 1.0, 2.0, 4.0])
        self.assertEqual(set(angular), {"kernel", "bandwidths"})
        self.assertEqual(set(mass), {"kernel", "bandwidths"})


if __name__ == "__main__":
    unittest.main()
