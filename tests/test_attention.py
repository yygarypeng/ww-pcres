import unittest
from unittest.mock import patch

import numpy as np
import torch

from data.preprocessing import neural_input_features_torch
from model.layers import WBosonFourVectorLayer
from model.model import WBosonRegressor


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
    def make_model(input_dim=21, mean=None, scale=None):
        return WBosonRegressor(
            input_dim=input_dim,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(21, dtype=np.float32) if mean is None else mean,
            std_scale_train=np.ones(21, dtype=np.float32) if scale is None else scale,
            attention_blocks=1,
            attention_dropout=0.0,
            decoder_dropout=0.0,
        )

    @staticmethod
    def make_inputs(batch_size=2):
        inputs = torch.randn(batch_size, 21)
        for start in (0, 4, 8, 12):
            inputs[:, start + 3] = (
                torch.linalg.vector_norm(inputs[:, start : start + 3], dim=1)
                + torch.rand(batch_size)
                + 0.1
            )
        return inputs

    def test_model_contract_uses_three_hl_features_and_six_tokens(self):
        model = self.make_model()

        self.assertEqual(model.norm.mean.numel(), 21)
        self.assertEqual(model.norm.std.numel(), 21)
        self.assertEqual(model.hl_embed.in_features, 3)
        self.assertEqual(model.num_tokens, 6)
        self.assertEqual(model(self.make_inputs()).shape, (2, 8))

    def test_model_contract_rejects_wrong_raw_or_neural_widths(self):
        with self.assertRaisesRegex(ValueError, "input_dim"):
            self.make_model(input_dim=20)
        with self.assertRaisesRegex(ValueError, "statistics"):
            self.make_model(mean=np.zeros(23), scale=np.ones(23))

    def test_aggregation_flattens_all_refined_tokens(self):
        model = self.make_model()
        aggregated = model.global_feature_aggregation(self.make_inputs(3))

        self.assertFalse(hasattr(model, "pooling_mode"))
        self.assertFalse(hasattr(model, "event_pool"))
        self.assertFalse(hasattr(model, "pool_norm"))
        self.assertEqual(aggregated.shape, (3, model.num_tokens * 8))

    def test_both_lepton_embedding_paths_receive_gradients(self):
        model = self.make_model()

        model(self.make_inputs(4)).square().mean().backward()

        gradients = [
            model.lep0_embed.weight.grad,
            model.lep1_embed.weight.grad,
        ]
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_embedding_inputs_use_shared_transform_then_normalization(self):
        mean = np.linspace(-1.0, 1.0, 21, dtype=np.float32)
        scale = np.linspace(1.0, 2.0, 21, dtype=np.float32)
        model = self.make_model(mean=mean, scale=scale).eval()
        inputs = self.make_inputs()
        captured = []
        hooks = [
            layer.register_forward_pre_hook(
                lambda _layer, args: captured.append(args[0].detach().clone())
            )
            for layer in (
                model.lep0_embed,
                model.lep1_embed,
                model.jet0_embed,
                model.jet1_embed,
                model.met_embed,
                model.hl_embed,
            )
        ]

        with torch.no_grad():
            model.global_feature_aggregation(inputs)
        for hook in hooks:
            hook.remove()

        transformed = neural_input_features_torch(inputs)
        normalized = (transformed - torch.from_numpy(mean)) / torch.from_numpy(scale)
        expected = [
            normalized[:, 0:4],
            normalized[:, 4:8],
            normalized[:, 8:12],
            normalized[:, 12:16],
            normalized[:, 16:18],
            normalized[:, 18:21],
        ]
        for actual, wanted in zip(captured, expected):
            torch.testing.assert_close(actual, wanted)

    def test_physics_layer_receives_raw_leptons_and_met(self):
        model = self.make_model().eval()
        inputs = self.make_inputs()

        with patch.object(model.w_layer, "forward", wraps=model.w_layer.forward) as w_layer:
            model(inputs)

        lep0, lep1, _, met = w_layer.call_args.args
        torch.testing.assert_close(lep0, inputs[:, :4])
        torch.testing.assert_close(lep1, inputs[:, 4:8])
        torch.testing.assert_close(met, inputs[:, 16:18])

    def test_empty_jet_embedding_values_are_masked_before_flattening(self):
        torch.manual_seed(3)
        mean = np.full(21, 5.0, dtype=np.float32)
        model = self.make_model(mean=mean).eval()
        missing_jet = self.make_inputs()
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
