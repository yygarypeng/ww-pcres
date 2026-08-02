import unittest

import numpy as np
import torch

from model.model import WBosonRegressor
from model.layers import WBosonFourVectorLayer


class SymmetricFourVectorLayerTest(unittest.TestCase):
    def test_transverse_neutrino_sum_matches_met_minus_dmet(self):
        layer = WBosonFourVectorLayer()
        lep0 = torch.tensor([[2.0, 3.0, 4.0, 6.0]])
        lep1 = torch.tensor([[-1.0, 5.0, -2.0, 6.0]])
        params = torch.tensor([[8.0, -6.0, 3.0, -4.0, 1.5, -2.5]])
        met = torch.tensor([[20.0, 10.0]])

        prediction = layer(lep0, lep1, params, met)
        nu0_t = prediction[:, :2] - lep0[:, :2]
        nu1_t = prediction[:, 4:6] - lep1[:, :2]

        expected_total = met - params[:, 4:6]
        torch.testing.assert_close(nu0_t, 0.5 * (expected_total + params[:, :2]))
        torch.testing.assert_close(nu1_t, 0.5 * (expected_total - params[:, :2]))
        torch.testing.assert_close(nu0_t + nu1_t, met - params[:, 4:6])
        torch.testing.assert_close(nu0_t - nu1_t, params[:, :2])

    def test_swapping_w_slots_only_reverses_transverse_difference(self):
        layer = WBosonFourVectorLayer()
        lep0 = torch.tensor([[2.0, 3.0, 4.0, 6.0]])
        lep1 = torch.tensor([[-1.0, 5.0, -2.0, 6.0]])
        params = torch.tensor([[8.0, -6.0, 3.0, -4.0, 1.5, -2.5]])
        met = torch.tensor([[20.0, 10.0]])

        prediction = layer(lep0, lep1, params, met).reshape(-1, 2, 4)
        swapped_params = torch.cat(
            [-params[:, :2], params[:, 3:4], params[:, 2:3], params[:, 4:6]],
            dim=-1,
        )
        swapped = layer(lep1, lep0, swapped_params, met).reshape(-1, 2, 4)

        torch.testing.assert_close(swapped, prediction.flip(1))


class FlattenedAggregationTest(unittest.TestCase):
    @staticmethod
    def make_model(input_dim=18, mmd_cond_mean=None, mmd_cond_scale=None):
        return WBosonRegressor(
            input_dim=input_dim,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(input_dim, dtype=np.float32),
            std_scale_train=np.ones(input_dim, dtype=np.float32),
            mmd_cond_mean_train=mmd_cond_mean,
            mmd_cond_scale_train=mmd_cond_scale,
            attention_blocks=1,
            attention_dropout=0.0,
            decoder_dropout=0.0,
        )

    def test_aggregation_flattens_all_refined_tokens(self):
        model = self.make_model()
        aggregated = model.global_feature_aggregation(torch.randn(3, 18))

        self.assertFalse(hasattr(model, "pooling_mode"))
        self.assertFalse(hasattr(model, "event_pool"))
        self.assertFalse(hasattr(model, "pool_norm"))
        self.assertEqual(aggregated.shape, (3, model.num_tokens * 8))

    def test_both_lepton_embedding_paths_receive_gradients(self):
        model = self.make_model()

        model(torch.randn(4, 18)).square().mean().backward()

        gradients = [
            model.lep0_embed.weight.grad,
            model.lep1_embed.weight.grad,
        ]
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_regressor_output_shape_with_and_without_high_level_features(self):
        for input_dim in (18, 22):
            with self.subTest(input_dim=input_dim):
                model = self.make_model(input_dim=input_dim)

                self.assertEqual(model(torch.randn(2, input_dim)).shape, (2, 8))

    def test_empty_jet_embedding_values_are_masked_before_flattening(self):
        torch.manual_seed(3)
        model = self.make_model().eval()
        missing_jet = torch.randn(2, 18)
        missing_jet[:, 8:12] = 0.0
        present_jet = missing_jet.clone()
        present_jet[:, 8:12] = torch.tensor([1.0, 2.0, 3.0, 4.0])

        with torch.no_grad():
            missing_before = model.global_feature_aggregation(missing_jet)
            present_before = model.global_feature_aggregation(present_jet)
            model.jet0_embed.bias.add_(torch.arange(8) * 1000.0)
            missing_after = model.global_feature_aggregation(missing_jet)
            present_after = model.global_feature_aggregation(present_jet)

        torch.testing.assert_close(missing_after, missing_before)
        self.assertFalse(torch.allclose(present_after, present_before))

    def test_aux_condition_uses_normalized_periodic_features(self):
        mean = np.array([10.0, 1.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        scale = np.array([2.0, 4.0, 0.5, 0.5, 0.25, 0.25], dtype=np.float32)
        model = self.make_model(22, mean, scale).eval()
        x = torch.zeros(2, 22)
        x[:, 18:22] = torch.tensor([
            [12.0, 5.0, 0.0, torch.pi / 2.0],
            [8.0, -3.0, torch.pi, -torch.pi / 2.0],
        ])

        with torch.no_grad():
            _, aux = model(x, return_aux=True)

        raw_condition = torch.stack([
            x[:, 18],
            x[:, 19],
            torch.sin(x[:, 20]),
            torch.cos(x[:, 20]),
            torch.sin(x[:, 21]),
            torch.cos(x[:, 21]),
        ], dim=-1)
        expected = (raw_condition - torch.from_numpy(mean)) / torch.from_numpy(scale)
        torch.testing.assert_close(aux["cond"], expected)
        self.assertEqual(aux["cond"].shape, (2, 6))

    def test_periodic_condition_is_continuous_across_pi_boundary(self):
        model = self.make_model(
            22,
            np.zeros(6, dtype=np.float32),
            np.ones(6, dtype=np.float32),
        )
        x = torch.zeros(2, 22)
        x[:, 20] = torch.tensor([-torch.pi + 1.0e-4, torch.pi - 1.0e-4])

        condition = model._mmd_condition(x)

        self.assertLess(torch.linalg.vector_norm(condition[0] - condition[1]), 5.0e-4)

    def test_model_without_high_level_features_returns_empty_condition(self):
        model = self.make_model(18).eval()

        with torch.no_grad():
            _, aux = model(torch.randn(2, 18), return_aux=True)

        self.assertEqual(aux["cond"].shape, (2, 0))
