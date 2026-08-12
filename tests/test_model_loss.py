import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn.functional as F

from model import LightningWBoson
from model import losses as loss_module
from model.losses import dmet_loss, standardized_fourvec_huber_loss
from train import train as train_module
from train.train import compute_mass_mmd_standardization, compute_w_fourvec_scales


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
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(22, dtype=np.float32),
            std_scale_train=np.ones(22, dtype=np.float32),
            w_fourvec_scales=scales,
        )

        torch.testing.assert_close(model.w_fourvec_scales, torch.from_numpy(scales))
        torch.testing.assert_close(model.state_dict()["w_fourvec_scales"], torch.from_numpy(scales))


class StandardizedDmetHuberTest(unittest.TestCase):
    def test_compute_scales_uses_training_dmet_components(self):
        features = np.zeros((3, 21), dtype=np.float32)
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
        features = np.zeros((3, 21), dtype=np.float32)
        targets = np.zeros((3, 10), dtype=np.float32)

        scales = train_module.compute_dmet_scales(features, targets)

        self.assertTrue(np.all(scales > 0.0))

    def test_loss_standardizes_each_dmet_component(self):
        features = torch.zeros((1, 21))
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
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(22, dtype=np.float32),
            std_scale_train=np.ones(22, dtype=np.float32),
            dmet_scales=np.array([2.0, 3.0], dtype=np.float32),
            loss_weights={"huber": 0.0, "dmet": 1.0},
        )
        features = torch.zeros((1, 21))
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
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(22, dtype=np.float32),
            std_scale_train=np.ones(22, dtype=np.float32),
            dmet_scales=scales,
        )

        torch.testing.assert_close(model.dmet_scales, torch.from_numpy(scales))
        torch.testing.assert_close(model.state_dict()["dmet_scales"], torch.from_numpy(scales))


class HiggsMassLossTest(unittest.TestCase):
    def _predictions(self, m_h):
        # Two zero-pz W four-vectors whose sum is a rest-frame Higgs of mass m_h.
        m_h = np.asarray(m_h, dtype=np.float32)
        pred = np.zeros((len(m_h), 8), dtype=np.float32)
        pred[:, 3] = pred[:, 7] = 0.5 * m_h
        return torch.from_numpy(pred)

    def test_loss_matches_direct_huber_on_higgs_mass(self):
        pred = self._predictions([145.0])
        h_mass = torch.tensor([145.0])

        loss = loss_module.higgs_mass_loss(pred)
        expected = F.huber_loss(h_mass, torch.full_like(h_mass, 125.0), delta=2)

        torch.testing.assert_close(loss, expected)

    def test_larger_delta_punishes_off_peak_more(self):
        pred = self._predictions([155.0])

        loss_small_delta = loss_module.higgs_mass_loss(pred, delta=2)
        loss_large_delta = loss_module.higgs_mass_loss(pred, delta=20)

        self.assertGreater(loss_large_delta.item(), loss_small_delta.item())

    def test_gradient_magnitude_grows_with_delta_off_peak(self):
        pred = self._predictions([155.0]).requires_grad_(True)
        grads = {}
        for delta in (2, 20):
            pred.grad = None
            loss = loss_module.higgs_mass_loss(pred, delta=delta)
            loss.backward()
            grads[delta] = pred.grad.abs().sum().item()

        self.assertGreater(grads[20], grads[2])

    def test_extreme_spacelike_sum_stays_finite(self):
        # E^2 - p^2 hugely negative: mass clamps to TOR, loss must stay finite.
        pred = torch.tensor([[0.0, 0.0, 1e6, 1.0, 0.0, 0.0, 1e6, 1.0]])

        loss = loss_module.higgs_mass_loss(pred, delta=8)

        self.assertTrue(torch.isfinite(loss))


class LightningModelLossTest(unittest.TestCase):
    def _warmup_model(
        self,
        mmd_start_epoch=100,
        adaptive_loss_weights=False,
        angular_mmd_schedule=None,
    ):
        kwargs = {}
        if mmd_start_epoch is not None:
            kwargs["mmd_start_epoch"] = mmd_start_epoch
        return LightningWBoson(
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(22, dtype=np.float32),
            std_scale_train=np.ones(22, dtype=np.float32),
            loss_weights={
                "huber": 2.0,
                "alpha_mmd": 3.0,
                "mass_mmd": 4.0,
                "angular_mmd": 5.0,
            },
            adaptive_loss_weights=adaptive_loss_weights,
            angular_mmd_schedule=angular_mmd_schedule,
            **kwargs,
        )

    def _computed_loss_names(self, model):
        inputs = torch.zeros((2, 21))
        targets = torch.zeros((2, 10))
        predictions = torch.zeros((2, 8))
        condition = torch.zeros((2, 4))
        loss_value = torch.tensor(1.0)
        with (
            patch("model.model.standardized_fourvec_huber_loss", return_value=loss_value),
            patch("model.model.alpha_mmd", return_value=loss_value),
            patch("model.model.mass_mmd", return_value=loss_value),
            patch("model.model.angular_mmd", return_value=loss_value),
        ):
            _, losses = model._compute_losses(inputs, targets, predictions, condition)
        return set(losses)

    def test_training_warmup_disables_mmd_but_keeps_non_mmd_losses(self):
        model = self._warmup_model()
        model.trainer = SimpleNamespace(current_epoch=99)

        self.assertEqual(self._computed_loss_names(model), {"huber"})

    def test_evaluation_uses_same_mmd_warmup_as_training(self):
        model = self._warmup_model().eval()
        model.trainer = SimpleNamespace(current_epoch=99)

        self.assertEqual(self._computed_loss_names(model), {"huber"})

    def test_training_enables_mmd_at_start_epoch(self):
        model = self._warmup_model()
        model.trainer = SimpleNamespace(current_epoch=100)

        self.assertEqual(
            self._computed_loss_names(model),
            {"huber", "alpha_mmd", "mass_mmd", "angular_mmd"},
        )

    def test_default_mmd_start_epoch_enables_mmd_immediately(self):
        model = self._warmup_model(mmd_start_epoch=None)

        self.assertEqual(
            self._computed_loss_names(model),
            {"huber", "alpha_mmd", "mass_mmd", "angular_mmd"},
        )

    def test_rejects_negative_mmd_start_epoch(self):
        with self.assertRaisesRegex(ValueError, "mmd_start_epoch"):
            self._warmup_model(mmd_start_epoch=-1)

    def test_angular_mmd_schedule_uses_cosine_ramp(self):
        model = self._warmup_model(
            mmd_start_epoch=0,
            angular_mmd_schedule={
                "initial_multiplier": 0.1,
                "hold_epochs": 10,
                "full_weight_epoch": 80,
            },
        )

        weights = {}
        for epoch in (0, 10, 45, 80, 100):
            model.trainer = SimpleNamespace(current_epoch=epoch)
            weights[epoch] = model._effective_loss_weights()

        self.assertEqual(weights[0]["angular_mmd"], 0.5)
        self.assertEqual(weights[10]["angular_mmd"], 0.5)
        self.assertAlmostEqual(weights[45]["angular_mmd"], 2.75)
        self.assertEqual(weights[80]["angular_mmd"], 5.0)
        self.assertEqual(weights[100]["angular_mmd"], 5.0)
        for epoch_weights in weights.values():
            self.assertEqual(epoch_weights["alpha_mmd"], 3.0)
            self.assertEqual(epoch_weights["mass_mmd"], 4.0)

    def test_angular_mmd_schedule_is_optional(self):
        model = self._warmup_model(mmd_start_epoch=0)
        model.trainer = SimpleNamespace(current_epoch=0)

        self.assertEqual(model._effective_loss_weights()["angular_mmd"], 5.0)

    def test_angular_mmd_schedule_respects_mmd_warmup(self):
        model = self._warmup_model(
            mmd_start_epoch=20,
            angular_mmd_schedule={
                "initial_multiplier": 0.1,
                "hold_epochs": 10,
                "full_weight_epoch": 80,
            },
        )
        model.trainer = SimpleNamespace(current_epoch=10)

        self.assertEqual(model._effective_loss_weights()["angular_mmd"], 0.0)

    def test_logs_scheduled_angular_mmd_weight(self):
        model = self._warmup_model(
            mmd_start_epoch=0,
            angular_mmd_schedule={
                "initial_multiplier": 0.1,
                "hold_epochs": 10,
                "full_weight_epoch": 80,
            },
        )
        model.trainer = SimpleNamespace(current_epoch=0)

        with patch.object(model, "log") as log:
            model._log_loss_weights()
        logged = {call.args[0]: call.args[1] for call in log.call_args_list}

        self.assertEqual(logged["loss_weight/alpha_mmd"], 3.0)
        self.assertEqual(logged["loss_weight/mass_mmd"], 4.0)
        self.assertEqual(logged["loss_weight/angular_mmd"], 0.5)

    def test_gradient_cosine_rest_uses_scheduled_angular_mmd_weight(self):
        model = self._warmup_model(
            mmd_start_epoch=0,
            angular_mmd_schedule={
                "initial_multiplier": 0.1,
                "hold_epochs": 10,
                "full_weight_epoch": 80,
            },
        )
        model.trainer = SimpleNamespace(current_epoch=0)
        parameter = next(model.parameters())
        losses = {"angular_mmd": parameter.reshape(-1)[0]}
        total = 0.5 * losses["angular_mmd"]

        cosines = model._compute_loss_gradient_cosines(losses, total)

        torch.testing.assert_close(cosines["angular_mmd"]["rest"], torch.tensor(0.0))

    def test_rejects_invalid_angular_mmd_schedule(self):
        schedules = (
            {"initial_multiplier": -0.1, "hold_epochs": 10, "full_weight_epoch": 80},
            {"initial_multiplier": 1.1, "hold_epochs": 10, "full_weight_epoch": 80},
            {"initial_multiplier": float("nan"), "hold_epochs": 10, "full_weight_epoch": 80},
            {"initial_multiplier": 0.1, "hold_epochs": -1, "full_weight_epoch": 80},
            {"initial_multiplier": 0.1, "hold_epochs": 10, "full_weight_epoch": 10},
            {
                "initial_multiplier": 0.1,
                "hold_epochs": 10,
                "full_weight_epoch": 80,
                "extra": 1,
            },
        )
        for schedule in schedules:
            with self.subTest(schedule=schedule):
                with self.assertRaisesRegex(ValueError, "angular_mmd_schedule"):
                    self._warmup_model(angular_mmd_schedule=schedule)

    def test_logs_effective_mmd_weights_without_mutating_configured_weights(self):
        model = self._warmup_model()
        configured_weights = dict(model.loss_weights)
        model.trainer = SimpleNamespace(current_epoch=99)

        with patch.object(model, "log") as log:
            model._log_loss_weights()
        warmup_logs = {call.args[0]: call.args[1] for call in log.call_args_list}

        self.assertEqual(warmup_logs["loss_weight/huber"], 2.0)
        self.assertEqual(warmup_logs["loss_weight/alpha_mmd"], 0.0)
        self.assertEqual(warmup_logs["loss_weight/mass_mmd"], 0.0)
        self.assertEqual(warmup_logs["loss_weight/angular_mmd"], 0.0)

        model.trainer = SimpleNamespace(current_epoch=100)
        with patch.object(model, "log") as log:
            model._log_loss_weights()
        active_logs = {call.args[0]: call.args[1] for call in log.call_args_list}

        self.assertEqual(active_logs["loss_weight/alpha_mmd"], 3.0)
        self.assertEqual(active_logs["loss_weight/mass_mmd"], 4.0)
        self.assertEqual(active_logs["loss_weight/angular_mmd"], 5.0)
        self.assertEqual(model.loss_weights, configured_weights)

    def test_adaptive_warmup_preserves_budget_and_mmd_weights(self):
        model = self._warmup_model(adaptive_loss_weights=True)
        model.trainer = SimpleNamespace(current_epoch=99)
        configured_weights = dict(model.loss_weights)
        cosines = {
            "huber": {"total": torch.tensor(0.0)},
            "higgs_mass": {"total": torch.tensor(0.5)},
        }

        model._update_adaptive_loss_weights(cosines)

        self.assertAlmostEqual(
            sum(model.loss_weights.values()),
            sum(configured_weights.values()),
        )
        for name in ("alpha_mmd", "mass_mmd", "angular_mmd"):
            self.assertEqual(model.loss_weights[name], configured_weights[name])
        self.assertNotEqual(model.loss_weights["huber"], configured_weights["huber"])
        self.assertNotEqual(model.loss_weights["higgs_mass"], configured_weights["higgs_mass"])

    def test_weight_decay_is_forwarded_to_optimizer(self):
        model = LightningWBoson(
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(22, dtype=np.float32),
            std_scale_train=np.ones(22, dtype=np.float32),
            weight_decay=0.0123,
        )

        optimizer = model.configure_optimizers()

        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0123)

    def test_feature_and_condition_mmd_config_are_routed_independently(self):
        model = LightningWBoson(
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(22, dtype=np.float32),
            std_scale_train=np.ones(22, dtype=np.float32),
            mmd_config={
                "condition": {"kernel": "imq", "bandwidth_multipliers": [3.0]},
                "alpha": {"kernel": "rbf", "bandwidth_multipliers": [0.2, 0.4]},
            },
        )

        self.assertEqual(
            model._mmd_kwargs("alpha"),
            {
                "local": True,
                "feature_kernel": "rbf",
                "condition_kernel": "imq",
                "feature_bandwidth_multipliers": [0.2, 0.4],
                "condition_bandwidth_multipliers": [3.0],
            },
        )

    def test_global_mmd_config_is_routed_to_estimator(self):
        model = LightningWBoson(
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(22, dtype=np.float32),
            std_scale_train=np.ones(22, dtype=np.float32),
            mmd_config={"local": False},
        )

        self.assertIs(model._mmd_kwargs("angular")["local"], False)

    def test_rejects_non_boolean_local_mmd_config(self):
        for value in (0, "false", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "mmd.local must be a boolean"):
                    LightningWBoson(
                        input_dim=21,
                        d_model=8,
                        num_heads=2,
                        std_mean_train=np.zeros(22, dtype=np.float32),
                        std_scale_train=np.ones(22, dtype=np.float32),
                        mmd_config={"local": value},
                    )

    def test_rejects_invalid_mmd_config_at_model_construction(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            LightningWBoson(
                input_dim=21,
                d_model=8,
                num_heads=2,
                std_mean_train=np.zeros(22, dtype=np.float32),
                std_scale_train=np.ones(22, dtype=np.float32),
                mmd_config={
                    "condition": {"bandwidth_multipliers": [0.0]},
                },
            )

    def test_rejects_unsupported_loss_weight_keys(self):
        for key in ("w_mass_mmd", "kinematic_loss_mmd_typo"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, f"unsupported loss_weights key.*{key}"):
                    LightningWBoson(
                        input_dim=21,
                        d_model=8,
                        num_heads=2,
                        std_mean_train=np.zeros(22, dtype=np.float32),
                        std_scale_train=np.ones(22, dtype=np.float32),
                        loss_weights={key: 1.0},
                    )

    def test_rejects_deprecated_mmd_loss_names_with_migration(self):
        for key, replacement in (
            ("kinematic_loss_mmd", "alpha_mmd and mass_mmd"),
            ("angular_loss_mmd", "angular_mmd"),
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, replacement):
                    LightningWBoson(
                        input_dim=21,
                        d_model=8,
                        num_heads=2,
                        std_mean_train=np.zeros(22, dtype=np.float32),
                        std_scale_train=np.ones(22, dtype=np.float32),
                        loss_weights={key: 1.0},
                    )

    def test_normalized_periodic_condition_reaches_both_local_mmd_losses(self):
        mean = np.array([3.0, -2.0, 0.0, 0.0], dtype=np.float32)
        scale = np.array([2.0, 4.0, 1.0, 1.0], dtype=np.float32)
        model = LightningWBoson(
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(22, dtype=np.float32),
            std_scale_train=np.ones(22, dtype=np.float32),
            mmd_cond_mean_train=mean,
            mmd_cond_scale_train=scale,
            loss_weights={
                "huber": 0.0,
                "alpha_mmd": 1.0,
                "mass_mmd": 1.0,
                "angular_mmd": 1.0,
            },
            attention_blocks=1,
            attention_dropout=0.0,
            decoder_dropout=0.0,
        ).eval()
        x = torch.zeros(4, 21)
        x[:, 18:21] = torch.tensor([
            [10.0, -1.0, -0.4],
            [20.0, -0.5, -0.2],
            [30.0, 0.5, 0.2],
            [40.0, 1.0, 0.4],
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
        self.assertEqual(local_mmd.call_count, 3)
        self.assertEqual([feature_count for feature_count, _ in captured], [1, 2, 6])
        for _, captured_condition in captured:
            torch.testing.assert_close(captured_condition, expected)
            self.assertEqual(captured_condition.shape, (4, 4))


class LocalMMDTest(unittest.TestCase):
    def setUp(self):
        self.pred = torch.tensor([[0.0], [0.5], [1.0]])
        self.truth = torch.tensor([[0.0], [0.25], [1.0]])
        self.condition = torch.tensor([[0.0], [1.0], [2.0]])

    def test_accepts_independent_feature_and_condition_bandwidth_lists(self):
        loss = loss_module.compute_local_mmd(
            self.pred,
            self.truth,
            self.condition,
            feature_kernel="imq",
            condition_kernel="rbf",
            feature_bandwidth_multipliers=[0.25, 0.5, 1.0, 2.0],
            condition_bandwidth_multipliers=[0.5, 1.0],
        )

        self.assertTrue(torch.isfinite(loss))

    def test_duplicate_bandwidths_do_not_rescale_loss(self):
        baseline = loss_module.compute_local_mmd(
            self.pred,
            self.truth,
            self.condition,
            feature_bandwidth_multipliers=[0.5],
            condition_bandwidth_multipliers=[1.0],
        )
        duplicated = loss_module.compute_local_mmd(
            self.pred,
            self.truth,
            self.condition,
            feature_bandwidth_multipliers=[0.5, 0.5],
            condition_bandwidth_multipliers=[1.0, 1.0, 1.0],
        )

        torch.testing.assert_close(duplicated, baseline)

    def test_global_mode_ignores_finite_condition_values(self):
        torch.manual_seed(17)
        pred = torch.randn(8, 2, requires_grad=True)
        truth = torch.randn(8, 2)
        condition_a = torch.randn(8, 4)
        condition_b = torch.randn(8, 4) * 100.0 + 50.0

        loss_a = loss_module.compute_local_mmd(
            pred,
            truth,
            condition_a,
            local=False,
        )
        loss_b = loss_module.compute_local_mmd(
            pred,
            truth,
            condition_b,
            local=False,
        )

        torch.testing.assert_close(loss_a, loss_b)
        loss_a.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_global_mode_ignores_nonfinite_condition_values(self):
        pred = torch.tensor([[0.0], [0.5], [1.0]])
        truth = torch.tensor([[0.0], [0.25], [1.0]])
        finite_condition = torch.zeros((3, 4))
        nonfinite_condition = finite_condition.clone()
        nonfinite_condition[0, 0] = float("nan")

        finite_loss = loss_module.compute_local_mmd(
            pred,
            truth,
            finite_condition,
            local=False,
        )
        nonfinite_loss = loss_module.compute_local_mmd(
            pred,
            truth,
            nonfinite_condition,
            local=False,
        )

        torch.testing.assert_close(nonfinite_loss, finite_loss)

    def test_equal_inputs_have_zero_biased_mmd(self):
        loss = loss_module.compute_local_mmd(
            self.truth,
            self.truth,
            self.condition,
        )

        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_rejects_nonpositive_bandwidth_multiplier(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            loss_module.compute_local_mmd(
                self.pred,
                self.truth,
                self.condition,
                condition_bandwidth_multipliers=[0.0],
            )


class AlphaMMDTest(unittest.TestCase):
    def test_matches_visualization_alpha_with_truth_on_shell_ordering(self):
        x = torch.zeros((2, 21))
        x[:, 3] = 10.0
        x[:, 7] = 20.0

        y_true = torch.zeros((2, 10))
        y_true[0, :4] = torch.tensor([3.0, 0.0, 0.0, 15.0])
        y_true[0, 4:8] = torch.tensor([0.0, 1.0, 0.0, 22.0])
        y_true[0, 8:10] = torch.tensor([80.379, 40.0])
        y_true[1, :4] = torch.tensor([1.0, 0.0, 0.0, 15.0])
        y_true[1, 4:8] = torch.tensor([0.0, 3.0, 0.0, 22.0])
        y_true[1, 8:10] = torch.tensor([40.0, 80.379])

        y_pred = torch.zeros((2, 8))
        y_pred[:, :4] = torch.tensor([1.0, 0.0, 0.0, 15.0])
        y_pred[:, 4:8] = torch.tensor([0.0, 3.0, 0.0, 22.0])
        condition = torch.randn(2, 4)
        captured = {}

        def capture_local_mmd(pred_features, true_features, cond, **kwargs):
            captured["pred"] = pred_features
            captured["true"] = true_features
            captured["cond"] = cond
            return pred_features.sum() * 0.0

        with patch("model.losses.compute_local_mmd", side_effect=capture_local_mmd):
            loss_module.alpha_mmd(x, y_true, y_pred, condition)

        # Event 0 is slot-0-on, but the larger composite mass uses the
        # off-shell neutrino. Event 1 is slot-1-on and uses its on-shell one.
        expected_true = torch.tensor([[-0.5], [0.5]])
        expected_pred = torch.tensor([[0.5], [0.5]])
        torch.testing.assert_close(captured["true"], expected_true)
        torch.testing.assert_close(captured["pred"], expected_pred)
        torch.testing.assert_close(captured["cond"], condition)

    def test_zero_total_neutrino_momentum_is_excluded(self):
        x = torch.zeros((2, 21))
        x[:, 3] = 10.0
        x[:, 7] = 20.0
        y_true = torch.cat(
            [x[:, :8], torch.tensor([[80.379, 40.0], [80.379, 40.0]])],
            dim=-1,
        )
        y_pred = x[:, :8].clone()
        y_true[1, 0] = 1.0
        y_pred[1, 0] = 1.0
        condition = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        captured = {}

        def capture_local_mmd(pred_features, true_features, cond, **kwargs):
            captured["pred"] = pred_features
            captured["true"] = true_features
            captured["cond"] = cond
            return pred_features.sum() * 0.0

        with patch("model.losses.compute_local_mmd", side_effect=capture_local_mmd):
            loss_module.alpha_mmd(x, y_true, y_pred, condition)

        self.assertEqual(captured["true"].shape[0], 1)
        self.assertEqual(captured["pred"].shape[0], 1)
        torch.testing.assert_close(captured["cond"], condition[1:])

    def test_invalid_composite_mass_squared_is_excluded(self):
        x = torch.zeros((2, 21))
        x[:, 3] = 10.0
        x[:, 7] = 20.0
        y_true = torch.cat(
            [x[:, :8], torch.tensor([[80.379, 40.0], [80.379, 40.0]])],
            dim=-1,
        )
        y_pred = x[:, :8].clone()
        y_true[:, 0] = 1.0
        y_pred[:, 0] = 1.0
        y_pred[0, 0] = 100.0
        condition = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        captured = {}

        def capture_local_mmd(pred_features, true_features, cond, **kwargs):
            captured["cond"] = cond
            return pred_features.sum() * 0.0

        with patch("model.losses.compute_local_mmd", side_effect=capture_local_mmd):
            loss_module.alpha_mmd(x, y_true, y_pred, condition)

        torch.testing.assert_close(captured["cond"], condition[1:])

    def test_positive_sub_epsilon_total_preserves_alpha_ratio(self):
        tiny = torch.finfo(torch.float32).eps / 16.0
        x = torch.zeros((1, 21))
        y_true = torch.zeros((1, 10))
        y_true[0, 0] = tiny
        y_true[0, 4] = 3.0 * tiny
        y_true[0, 3] = 1.0
        y_true[0, 7] = 1.0
        captured = {}

        def capture_local_mmd(pred_features, true_features, cond, **kwargs):
            captured["true"] = true_features
            return pred_features.sum() * 0.0

        with patch("model.losses.compute_local_mmd", side_effect=capture_local_mmd):
            loss_module.alpha_mmd(x, y_true, y_true[:, :8], torch.zeros((1, 4)))

        torch.testing.assert_close(captured["true"][:, 0], torch.tensor([-0.5]))

    def test_mixed_nonfinite_rows_keep_condition_aligned(self):
        x = torch.zeros((3, 21))
        x[:, 3] = 10.0
        x[:, 7] = 20.0
        y_true = torch.cat(
            [x[:, :8].clone(), torch.tensor([[80.379, 40.0]]).repeat(3, 1)],
            dim=-1,
        )
        y_true[:, 0] = 1.0
        y_pred = y_true[:, :8].clone()
        y_pred[1, 0] = float("nan")
        condition = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        condition[2, 0] = float("nan")
        captured = {}

        def capture_local_mmd(pred_features, true_features, cond, **kwargs):
            captured["rows"] = pred_features.shape[0]
            captured["cond"] = cond
            return pred_features.sum() * 0.0

        with patch("model.losses.compute_local_mmd", side_effect=capture_local_mmd):
            loss_module.alpha_mmd(x, y_true, y_pred, condition)

        self.assertEqual(captured["rows"], 1)
        torch.testing.assert_close(captured["cond"], condition[:1])

    def test_valid_and_exact_zero_totals_have_finite_gradients(self):
        x = torch.zeros((2, 21))
        y_true = torch.zeros((2, 10))
        y_pred = torch.zeros((2, 8))
        y_pred[0, 0] = 1.0
        y_pred[0, 4] = 3.0
        y_pred.requires_grad_()

        with patch("model.losses.compute_local_mmd", side_effect=lambda pred, true, cond, **kwargs: pred.sum()):
            loss = loss_module.alpha_mmd(x, y_true, y_pred, torch.zeros((2, 4)))
        loss.backward()

        self.assertTrue(torch.isfinite(y_pred.grad).all())

    def test_requires_condition_features(self):
        with self.assertRaisesRegex(ValueError, "requires the four high-level"):
            loss_module.alpha_mmd(
                torch.zeros((1, 21)),
                torch.zeros((1, 10)),
                torch.zeros((1, 8)),
                torch.empty((1, 0)),
            )

    def test_all_nonfinite_rows_return_differentiable_zero(self):
        y_pred = torch.full((2, 8), float("nan"), requires_grad=True)

        loss = loss_module.alpha_mmd(
            torch.zeros((2, 21)),
            torch.zeros((2, 10)),
            y_pred,
            torch.zeros((2, 4)),
        )

        torch.testing.assert_close(loss, torch.tensor(0.0))
        loss.backward()
        self.assertTrue(torch.isfinite(y_pred.grad).all())


class MassMMDTest(unittest.TestCase):
    def test_uses_charge_ordered_asinh_mass2_with_shared_training_scale(self):
        x = torch.zeros((1, 21))
        y_true = torch.tensor([[0.0, 0.0, 0.0, 80.4, 0.0, 0.0, 0.0, 40.2, 80.4, 40.2]])
        y_pred = torch.tensor([[0.0, 0.0, 0.0, 40.2, 0.0, 0.0, 0.0, 80.4]])
        captured = {}

        def capture_local_mmd(pred_features, true_features, cond, **kwargs):
            captured["pred"] = pred_features
            captured["true"] = true_features
            return pred_features.sum() * 0.0

        center = torch.tensor(0.2)
        scale = torch.tensor(0.5)
        with patch("model.losses.compute_local_mmd", side_effect=capture_local_mmd):
            loss_module.mass_mmd(
                x,
                y_true,
                y_pred,
                torch.zeros((1, 4)),
                center,
                scale,
            )

        expected_true = (torch.asinh(torch.tensor([[1.0, 0.25]])) - center) / scale
        expected_pred = (torch.asinh(torch.tensor([[0.25, 1.0]])) - center) / scale
        torch.testing.assert_close(captured["true"], expected_true)
        torch.testing.assert_close(captured["pred"], expected_pred)

    def test_training_standardization_pools_both_mass_slots(self):
        targets = np.zeros((2, 10), dtype=np.float32)
        targets[:, 8:10] = np.array([[20.0, 40.0], [60.0, 80.0]], dtype=np.float32)

        center, scale = compute_mass_mmd_standardization(targets)

        transformed = np.arcsinh((targets[:, 8:10].reshape(-1) / loss_module.W_MASS_SCALE) ** 2)
        expected_center = np.median(transformed)
        q25, q75 = np.percentile(transformed, [25.0, 75.0])
        np.testing.assert_allclose(center, expected_center)
        np.testing.assert_allclose(scale, (q75 - q25) / 1.349)

    def test_standardization_is_saved_in_model_state(self):
        model = LightningWBoson(
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(22, dtype=np.float32),
            std_scale_train=np.ones(22, dtype=np.float32),
            mass_mmd_center=0.25,
            mass_mmd_scale=0.75,
        )

        torch.testing.assert_close(model.state_dict()["mass_mmd_center"], torch.tensor(0.25))
        torch.testing.assert_close(model.state_dict()["mass_mmd_scale"], torch.tensor(0.75))


class AngularMMDTest(unittest.TestCase):
    def test_uses_common_truth_prediction_validity_mask(self):
        true_booster = Mock()
        pred_booster = Mock()
        true_booster.valid_rest_frame_mask.return_value = torch.tensor([True, True, False])
        pred_booster.valid_rest_frame_mask.return_value = torch.tensor([True, False, True])
        angles = tuple(torch.arange(3, dtype=torch.float32) for _ in range(4))
        true_booster.lep_theta_phi_in_w_rest.return_value = angles
        pred_booster.lep_theta_phi_in_w_rest.return_value = angles
        condition = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        captured = {}

        def capture_local_mmd(pred_features, true_features, cond, **kwargs):
            captured["rows"] = pred_features.shape[0]
            captured["cond"] = cond
            return pred_features.sum() * 0.0

        with patch("model.losses.Booster", side_effect=[true_booster, pred_booster]):
            with patch("model.losses.compute_local_mmd", side_effect=capture_local_mmd):
                loss_module.angular_mmd(
                    torch.zeros((3, 21)),
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
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(22, dtype=np.float32),
            std_scale_train=np.ones(22, dtype=np.float32),
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
        inputs = torch.randn(2, 21)
        for start in (0, 4, 8, 12):
            inputs[:, start + 3] = torch.linalg.vector_norm(
                inputs[:, start:start + 3], dim=1
            ) + torch.rand(2) + 0.1
        batch = (inputs, torch.randn(2, 10))
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
