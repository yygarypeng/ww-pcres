import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn.functional as F

from model import LightningWBoson
from model import losses as loss_module
from model.losses import dmet_loss, standardized_fourvec_huber_loss
from train import train as train_module
from train.train import compute_w_fourvec_scales


class StandardizedFourVectorHuberTest(unittest.TestCase):
    def test_compute_scales_pools_w_slots_by_component(self):
        targets = np.array(
            [
                [1.0, 10.0, 100.0, 1000.0, 3.0, 30.0, 300.0, 3000.0, 0.0, 0.0],
                [5.0, 50.0, 500.0, 5000.0, 7.0, 70.0, 700.0, 7000.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        scales = compute_w_fourvec_scales(targets)

        transformed = targets[:, :8].reshape(-1, 4).copy()
        transformed[:, 3] = np.log1p(transformed[:, 3])
        expected = np.std(transformed, axis=0)
        np.testing.assert_allclose(scales, expected)

    def test_compute_scales_clamps_constant_components(self):
        targets = np.ones((3, 10), dtype=np.float32)

        scales = compute_w_fourvec_scales(targets)

        self.assertTrue(np.all(scales > 0.0))

    def test_loss_applies_log1p_to_both_energy_slots(self):
        truth = torch.tensor([[0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0]])
        prediction = torch.tensor([[1.0, 2.0, 3.0, 7.0, 1.0, 2.0, 3.0, 15.0]])
        scales = torch.ones(4)

        loss = standardized_fourvec_huber_loss(truth, prediction, scales)

        log_two = torch.log(torch.tensor(2.0))
        standardized_residual = torch.tensor(
            [[[1.0, 2.0, 3.0, log_two], [1.0, 2.0, 3.0, log_two]]]
        )
        expected = F.huber_loss(standardized_residual, torch.zeros_like(standardized_residual))
        torch.testing.assert_close(loss, expected)

    def test_loss_is_finite_for_zero_energy(self):
        truth = torch.zeros((1, 10))
        prediction = torch.zeros((1, 8))

        loss = standardized_fourvec_huber_loss(truth, prediction, torch.ones(4))

        self.assertTrue(torch.isfinite(loss))

    def test_scales_are_saved_in_model_state(self):
        scales = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        model = LightningWBoson(
            input_dim=18,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(18, dtype=np.float32),
            std_scale_train=np.ones(18, dtype=np.float32),
            w_fourvec_scales=scales,
        )

        torch.testing.assert_close(model.w_fourvec_scales, torch.from_numpy(scales))
        torch.testing.assert_close(model.state_dict()["w_fourvec_scales"], torch.from_numpy(scales))


class StandardizedDmetHuberTest(unittest.TestCase):
    def test_compute_scales_uses_training_dmet_components(self):
        features = np.zeros((3, 18), dtype=np.float32)
        targets = np.zeros((3, 10), dtype=np.float32)
        nu0_t = np.array([[2.0, -1.0], [4.0, -2.0], [6.0, -3.0]], dtype=np.float32)
        nu1_t = np.array([[-3.0, 5.0], [-6.0, 10.0], [-9.0, 15.0]], dtype=np.float32)
        expected_dmet = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]], dtype=np.float32)
        features[:, :2] = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
        features[:, 4:6] = [[-1.0, 3.0], [-2.0, 4.0], [-3.0, 5.0]]
        targets[:, :2] = features[:, :2] + nu0_t
        targets[:, 4:6] = features[:, 4:6] + nu1_t
        features[:, 16:18] = expected_dmet + nu0_t + nu1_t

        scales = train_module.compute_dmet_scales(features, targets)

        np.testing.assert_allclose(scales, np.std(expected_dmet, axis=0))

    def test_compute_scales_clamps_constant_components(self):
        features = np.zeros((3, 18), dtype=np.float32)
        targets = np.zeros((3, 10), dtype=np.float32)

        scales = train_module.compute_dmet_scales(features, targets)

        self.assertTrue(np.all(scales > 0.0))

    def test_loss_standardizes_each_dmet_component(self):
        features = torch.zeros((1, 18))
        features[:, :2] = torch.tensor([[1.0, 2.0]])
        features[:, 4:6] = torch.tensor([[-3.0, 4.0]])
        features[:, 16:18] = torch.tensor([[20.0, 30.0]])
        targets = torch.zeros((1, 10))
        targets[:, :2] = torch.tensor([[6.0, 9.0]])
        targets[:, 4:6] = torch.tensor([[8.0, 2.0]])
        prediction = torch.tensor([[6.0, 31.0]])
        scales = torch.tensor([2.0, 3.0])

        loss = dmet_loss(features, targets, prediction, scales)

        residual = torch.tensor([[1.0, 2.0]])
        expected = F.huber_loss(residual, torch.zeros_like(residual))
        torch.testing.assert_close(loss, expected)

    def test_model_loss_uses_registered_scales(self):
        model = LightningWBoson(
            input_dim=18,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(18, dtype=np.float32),
            std_scale_train=np.ones(18, dtype=np.float32),
            dmet_scales=np.array([2.0, 3.0], dtype=np.float32),
            loss_weights={"huber": 0.0, "dmet": 1.0},
        )
        features = torch.zeros((1, 18))
        features[:, 16:18] = torch.tensor([[4.0, 5.0]])
        targets = torch.zeros((1, 10))
        prediction = torch.tensor([[6.0, 11.0]])

        total, losses = model._compute_losses(
            features,
            targets,
            torch.zeros((1, 8)),
            torch.empty((1, 0)),
            {"dmet": prediction},
        )

        residual = torch.tensor([[1.0, 2.0]])
        expected = F.huber_loss(residual, torch.zeros_like(residual))
        torch.testing.assert_close(losses["dmet"], expected)
        torch.testing.assert_close(total, expected)

    def test_scales_are_saved_in_model_state(self):
        scales = np.array([2.0, 3.0], dtype=np.float32)
        model = LightningWBoson(
            input_dim=18,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(18, dtype=np.float32),
            std_scale_train=np.ones(18, dtype=np.float32),
            dmet_scales=scales,
        )

        torch.testing.assert_close(model.dmet_scales, torch.from_numpy(scales))
        torch.testing.assert_close(model.state_dict()["dmet_scales"], torch.from_numpy(scales))


class LightningModelLossTest(unittest.TestCase):
    def test_weight_decay_is_forwarded_to_optimizer(self):
        model = LightningWBoson(
            input_dim=18,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(18, dtype=np.float32),
            std_scale_train=np.ones(18, dtype=np.float32),
            weight_decay=0.0123,
        )

        optimizer = model.configure_optimizers()

        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0123)

    def test_rejects_unsupported_loss_weight_keys(self):
        for key in ("w_mass_mmd", "kinematic_loss_mmd_typo"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, f"unsupported loss_weights key.*{key}"):
                    LightningWBoson(
                        input_dim=18,
                        d_model=8,
                        num_heads=2,
                        std_mean_train=np.zeros(18, dtype=np.float32),
                        std_scale_train=np.ones(18, dtype=np.float32),
                        loss_weights={key: 1.0},
                    )

    def test_conditional_mmd_requires_high_level_condition_features(self):
        model = LightningWBoson(
            input_dim=18,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(18, dtype=np.float32),
            std_scale_train=np.ones(18, dtype=np.float32),
            loss_weights={
                "huber": 0.0,
                "angular_loss_mmd": 1.0,
                "kinematic_loss_mmd": 1.0,
            },
        )

        with self.assertRaisesRegex(ValueError, "requires the four high-level"):
            model._compute_batch_losses(torch.randn(2, 18), torch.randn(2, 10))

    def test_normalized_periodic_condition_reaches_both_local_mmd_losses(self):
        mean = np.arange(6, dtype=np.float32)
        scale = np.arange(1, 7, dtype=np.float32)
        model = LightningWBoson(
            input_dim=22,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(22, dtype=np.float32),
            std_scale_train=np.ones(22, dtype=np.float32),
            mmd_cond_mean_train=mean,
            mmd_cond_scale_train=scale,
            loss_weights={
                "huber": 0.0,
                "angular_loss_mmd": 1.0,
                "kinematic_loss_mmd": 1.0,
            },
            attention_blocks=1,
            attention_dropout=0.0,
            decoder_dropout=0.0,
        ).eval()
        x = torch.zeros(4, 22)
        x[:, 18:22] = torch.tensor([
            [10.0, -1.0, -0.4, -0.7],
            [20.0, -0.5, -0.2, -0.3],
            [30.0, 0.5, 0.2, 0.3],
            [40.0, 1.0, 0.4, 0.7],
        ])
        w0 = torch.tensor([30.0, 5.0, 40.0, 100.0])
        w1 = torch.tensor([-20.0, 15.0, -30.0, 90.0])
        prediction = torch.cat([w0, w1]).repeat(4, 1)
        target_w = torch.cat([w0 + torch.tensor([1.0, 0.0, 0.0, 0.0]), w1]).repeat(4, 1)
        target = torch.cat([target_w, torch.zeros(4, 2)], dim=-1)
        condition = torch.stack([
            x[:, 18],
            x[:, 19],
            torch.sin(x[:, 20]),
            torch.cos(x[:, 20]),
            torch.sin(x[:, 21]),
            torch.cos(x[:, 21]),
        ], dim=-1)
        expected = (condition - torch.from_numpy(mean)) / torch.from_numpy(scale)
        captured = []

        def capture_local_mmd(pred_features, true_features, cond, **kwargs):
            captured.append((pred_features.shape[-1], cond.detach().clone()))
            return pred_features.sum() * 0.0

        with patch.object(model.model.w_layer, "forward", return_value=prediction) as w_layer:
            with patch("model.losses.compute_local_mmd", side_effect=capture_local_mmd) as local_mmd:
                model._compute_batch_losses(x, target)

        self.assertEqual(w_layer.call_count, 1)
        self.assertEqual(local_mmd.call_count, 2)
        self.assertEqual([feature_count for feature_count, _ in captured], [3, 6])
        for _, captured_condition in captured:
            torch.testing.assert_close(captured_condition, expected)
            self.assertEqual(captured_condition.shape, (4, 6))


class KinematicMMDTest(unittest.TestCase):
    def test_uses_charge_associated_neutrinos_and_ordered_scaled_features(self):
        x = torch.zeros((1, 18))
        x[0, :4] = torch.tensor([1.0, 0.0, 0.0, 1.0])
        x[0, 4:8] = torch.tensor([0.0, 2.0, 0.0, 2.0])
        y_true = torch.tensor([[4.0, 0.0, 0.0, 10.0, 0.0, 3.0, 0.0, 8.0, 0.0, 0.0]])
        y_pred = torch.tensor([[2.0, 0.0, 0.0, 6.0, 0.0, 5.0, 0.0, 9.0]])
        condition = torch.randn(1, 6)
        captured = {}

        def capture_local_mmd(pred_features, true_features, cond, **kwargs):
            captured["pred"] = pred_features
            captured["true"] = true_features
            captured["cond"] = cond
            return pred_features.sum() * 0.0

        with patch("model.losses.compute_local_mmd", side_effect=capture_local_mmd):
            loss_module.kinematic_loss_mmd(x, y_true, y_pred, condition)

        scale2 = loss_module.W_MASS_SCALE**2
        expected_true = torch.tensor([[0.5, 84.0 / scale2, 55.0 / scale2]])
        expected_pred = torch.tensor([[-0.5, 32.0 / scale2, 56.0 / scale2]])
        torch.testing.assert_close(captured["true"], expected_true)
        torch.testing.assert_close(captured["pred"], expected_pred)
        torch.testing.assert_close(captured["cond"], condition)

    def test_zero_total_neutrino_momentum_centers_alpha(self):
        x = torch.zeros((1, 18))
        x[0, :4] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        x[0, 4:8] = torch.tensor([-1.0, -2.0, -3.0, 4.0])
        y_true = torch.cat([x[:, :8], torch.zeros((1, 2))], dim=-1)
        y_pred = x[:, :8].clone()
        captured = {}

        def capture_local_mmd(pred_features, true_features, cond, **kwargs):
            captured["pred"] = pred_features
            captured["true"] = true_features
            return pred_features.sum() * 0.0

        with patch("model.losses.compute_local_mmd", side_effect=capture_local_mmd):
            loss_module.kinematic_loss_mmd(x, y_true, y_pred, torch.zeros((1, 6)))

        torch.testing.assert_close(captured["true"][:, 0], torch.zeros(1))
        torch.testing.assert_close(captured["pred"][:, 0], torch.zeros(1))

    def test_positive_sub_epsilon_total_preserves_alpha_ratio(self):
        tiny = torch.finfo(torch.float32).eps / 16.0
        x = torch.zeros((1, 18))
        y_true = torch.zeros((1, 10))
        y_true[0, 0] = tiny
        y_true[0, 4] = 3.0 * tiny
        captured = {}

        def capture_local_mmd(pred_features, true_features, cond, **kwargs):
            captured["true"] = true_features
            return pred_features.sum() * 0.0

        with patch("model.losses.compute_local_mmd", side_effect=capture_local_mmd):
            loss_module.kinematic_loss_mmd(x, y_true, y_true[:, :8], torch.zeros((1, 6)))

        torch.testing.assert_close(captured["true"][:, 0], torch.tensor([-0.5]))

    def test_mixed_nonfinite_rows_keep_condition_aligned(self):
        x = torch.zeros((3, 18))
        y_true = torch.zeros((3, 10))
        y_pred = torch.zeros((3, 8))
        y_pred[1, 0] = float("nan")
        condition = torch.arange(18, dtype=torch.float32).reshape(3, 6)
        condition[2, 0] = float("nan")
        captured = {}

        def capture_local_mmd(pred_features, true_features, cond, **kwargs):
            captured["rows"] = pred_features.shape[0]
            captured["cond"] = cond
            return pred_features.sum() * 0.0

        with patch("model.losses.compute_local_mmd", side_effect=capture_local_mmd):
            loss_module.kinematic_loss_mmd(x, y_true, y_pred, condition)

        self.assertEqual(captured["rows"], 1)
        torch.testing.assert_close(captured["cond"], condition[:1])

    def test_valid_and_exact_zero_totals_have_finite_gradients(self):
        x = torch.zeros((2, 18))
        y_true = torch.zeros((2, 10))
        y_pred = torch.zeros((2, 8))
        y_pred[0, 0] = 1.0
        y_pred[0, 4] = 3.0
        y_pred.requires_grad_()

        with patch("model.losses.compute_local_mmd", side_effect=lambda pred, true, cond: pred.sum()):
            loss = loss_module.kinematic_loss_mmd(x, y_true, y_pred, torch.zeros((2, 6)))
        loss.backward()

        self.assertTrue(torch.isfinite(y_pred.grad).all())

    def test_requires_condition_features(self):
        with self.assertRaisesRegex(ValueError, "requires the four high-level"):
            loss_module.kinematic_loss_mmd(
                torch.zeros((1, 18)),
                torch.zeros((1, 10)),
                torch.zeros((1, 8)),
                torch.empty((1, 0)),
            )

    def test_all_nonfinite_rows_return_differentiable_zero(self):
        y_pred = torch.full((2, 8), float("nan"), requires_grad=True)

        loss = loss_module.kinematic_loss_mmd(
            torch.zeros((2, 18)),
            torch.zeros((2, 10)),
            y_pred,
            torch.zeros((2, 6)),
        )

        torch.testing.assert_close(loss, torch.tensor(0.0))
        loss.backward()
        self.assertTrue(torch.isfinite(y_pred.grad).all())


class AngularMMDTest(unittest.TestCase):
    def test_uses_common_truth_prediction_validity_mask(self):
        true_booster = Mock()
        pred_booster = Mock()
        true_booster.valid_rest_frame_mask.return_value = torch.tensor([True, True, False])
        pred_booster.valid_rest_frame_mask.return_value = torch.tensor([True, False, True])
        angles = tuple(torch.arange(3, dtype=torch.float32) for _ in range(4))
        true_booster.lep_theta_phi_in_w_rest.return_value = angles
        pred_booster.lep_theta_phi_in_w_rest.return_value = angles
        condition = torch.arange(18, dtype=torch.float32).reshape(3, 6)
        captured = {}

        def capture_local_mmd(pred_features, true_features, cond, **kwargs):
            captured["rows"] = pred_features.shape[0]
            captured["cond"] = cond
            return pred_features.sum() * 0.0

        with patch("model.losses.Booster", side_effect=[true_booster, pred_booster]):
            with patch("model.losses.compute_local_mmd", side_effect=capture_local_mmd):
                loss_module.angular_loss_mmd(
                    torch.zeros((3, 18)),
                    torch.zeros((3, 10)),
                    torch.zeros((3, 8)),
                    condition,
                )

        self.assertEqual(captured["rows"], 1)
        torch.testing.assert_close(captured["cond"], condition[:1])


class WMassHuberTest(unittest.TestCase):
    def test_compares_normalized_predicted_mass_squared_to_target_masses(self):
        y_true = torch.zeros((1, 10))
        y_true[0, 8:] = torch.tensor([40.2, 80.4])
        y_pred = torch.tensor([[0.0, 0.0, 0.0, 80.4, 0.0, 0.0, 0.0, 40.2]])

        loss = loss_module.w_mass_huber_loss(y_true, y_pred)

        expected_pred = torch.tensor([[1.0, 0.25]])
        expected_true = torch.tensor([[0.25, 1.0]])
        torch.testing.assert_close(loss, F.huber_loss(expected_pred, expected_true))


class GradientCosineLoggingTest(unittest.TestCase):
    def test_logging_only_mode_emits_cosines_without_changing_weights(self):
        model = LightningWBoson(
            input_dim=18,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(18, dtype=np.float32),
            std_scale_train=np.ones(18, dtype=np.float32),
            loss_weights={"huber": 1.0, "higgs_mass": 2.0},
            adaptive_loss_weights=False,
            log_loss_gradient_cosines=True,
            attention_blocks=1,
            attention_dropout=0.0,
            decoder_dropout=0.0,
        )
        parameter = next(model.parameters())
        losses = {
            "huber": parameter.square().sum(),
            "higgs_mass": parameter.sum(),
        }
        total = losses["huber"] + 2.0 * losses["higgs_mass"]
        batch = (torch.randn(2, 18), torch.randn(2, 10))
        original_weights = dict(model.loss_weights)

        with patch.object(model, "_compute_batch_losses", return_value=(total, losses)):
            with patch.object(model, "_log_losses"), patch.object(model, "_log_loss_weights"):
                with patch.object(model, "_log_grad_cosines") as log_cosines:
                    model.training_step(batch, 0)
                    model.on_train_epoch_end()

        log_cosines.assert_called_once()
        self.assertEqual(set(log_cosines.call_args.args[0]), {"higgs_mass"})
        self.assertEqual(model.loss_weights, original_weights)


if __name__ == "__main__":
    unittest.main()
