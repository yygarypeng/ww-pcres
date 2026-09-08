import math
import unittest
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from model import LightningWBoson
from model import losses as loss_module
from model.losses import dmet_loss, fourvec_huber_loss
from physics.torchBoost import Booster, _mock_inputs, safe_acos, safe_atan2


class FixedVMMDTest(unittest.TestCase):
    def test_singleton_includes_diagonal_terms_with_absolute_bandwidth(self):
        prediction = torch.tensor([[0.0]], requires_grad=True)
        truth = torch.tensor([[1.0]])

        loss = loss_module.compute_mmd(
            prediction,
            truth,
            kernel="imq",
            bandwidths=[1.0],
        )

        torch.testing.assert_close(loss, torch.tensor(1.0))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())


class BoosterSharedAngularStateTest(unittest.TestCase):
    def test_combined_angles_and_validity_match_existing_physics(self):
        leptons = torch.tensor(
            [
                [10.0, 4.0, 8.0, 14.0, -8.0, 5.0, -4.0, 12.0],
                [7.0, -3.0, 2.0, 10.0, -5.0, -4.0, 6.0, 11.0],
            ]
        )
        w_bosons = torch.tensor(
            [
                [30.0, 10.0, 20.0, 90.0, -20.0, 15.0, -10.0, 88.0],
                [25.0, -12.0, 18.0, 87.0, -17.0, -9.0, 14.0, 86.0],
            ],
            requires_grad=True,
        )
        booster = Booster(leptons, w_bosons)
        expected_valid = booster.valid_rest_frame_mask()
        expected_angles = torch.stack(booster.lep_theta_phi_in_w_rest()[:4], dim=-1)

        actual_valid, actual_angles = booster.lep_theta_phi_with_validity()

        torch.testing.assert_close(actual_valid, expected_valid)
        torch.testing.assert_close(actual_angles, expected_angles)
        actual_angles[actual_valid].sum().backward()
        self.assertTrue(torch.isfinite(w_bosons.grad).all())


def _reference_angles(lep, wboson):
    """Frozen copy of the Booster angle pipeline used as a regression reference."""

    def eps_of(x):
        return max(1e-12, torch.finfo(x.dtype).eps)

    def norm(x, keepdim=True):
        return torch.linalg.vector_norm(x, dim=-1, keepdim=keepdim).clamp_min(eps_of(x))

    def mass2(p4):
        return p4[..., 3] ** 2 - torch.sum(p4[..., 0:3] ** 2, dim=-1)

    def has_rest_frame(p4):
        return torch.isfinite(p4).all(dim=-1) & (p4[..., 3] > 0.0) & (mass2(p4) > eps_of(p4))

    def boost(p4, beta):
        p3 = p4[..., 0:3]
        e = p4[..., 3:4]
        eps = eps_of(p4)
        beta = torch.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
        beta2 = torch.sum(beta * beta, dim=-1, keepdim=True)
        valid_beta = beta2 < 1.0
        beta = torch.where(valid_beta, beta, torch.zeros_like(beta))
        beta2 = torch.where(valid_beta, beta2, torch.zeros_like(beta2))
        gamma = torch.rsqrt((1.0 - beta2).clamp_min(eps))
        beta_dot_p = torch.sum(beta * p3, dim=-1, keepdim=True)
        gamma2 = torch.where(beta2 > eps, (gamma - 1.0) / beta2, 0.5 * torch.ones_like(beta2))
        boosted_p3 = p3 + gamma2 * beta_dot_p * beta + gamma * e * beta
        boosted_e = gamma * (e + beta_dot_p)
        return torch.cat([boosted_p3, boosted_e], dim=-1)

    def boost_to_rest(p4, reference):
        valid = has_rest_frame(reference).unsqueeze(-1)
        energy = torch.where(valid, reference[..., 3:4], torch.ones_like(reference[..., 3:4]))
        beta = torch.where(
            valid, reference[..., 0:3] / energy, torch.zeros_like(reference[..., 0:3])
        )
        return boost(p4, -beta)

    def basis(w_axis):
        k = w_axis[..., 0:3] / norm(w_axis[..., 0:3])
        beam = torch.zeros_like(k)
        beam[..., 2] = 1.0
        y = torch.sum(beam * k, dim=-1, keepdim=True)
        transverse = torch.sqrt((1.0 - y * y).clamp_min(0.0) + eps_of(w_axis))
        r = (beam - y * k) / transverse
        n = torch.cross(beam, k, dim=-1) / transverse
        return n, r, k

    def project(p4, n, r, k):
        p3 = p4[..., 0:3]
        p3_projected = torch.stack(
            [
                torch.sum(p3 * n, dim=-1),
                torch.sum(p3 * r, dim=-1),
                torch.sum(p3 * k, dim=-1),
            ],
            dim=-1,
        )
        return torch.cat([p3_projected, p4[..., 3:4]], dim=-1)

    def theta(p4):
        p = norm(p4[..., 0:3], keepdim=False)
        cos_theta = torch.clamp(p4[..., 2] / p, -1.0, 1.0)
        return safe_acos(cos_theta)

    def phi(p4):
        return safe_atan2(p4[..., 1], p4[..., 0])

    w0 = wboson[..., :4]
    lep0 = lep[..., :4]
    w1 = wboson[..., 4:8]
    lep1 = lep[..., 4:8]
    higgs = w0 + w1
    w0_h = boost_to_rest(w0, higgs)
    lep0_h = boost_to_rest(lep0, higgs)
    w1_h = boost_to_rest(w1, higgs)
    lep1_h = boost_to_rest(lep1, higgs)

    w1_axis = w1_h[..., 0:3]
    eps = eps_of(w1_h)
    axis_norm = torch.linalg.vector_norm(w1_axis, dim=-1)
    transverse_fraction = torch.linalg.vector_norm(w1_axis[..., 0:2], dim=-1) / axis_norm.clamp_min(
        eps
    )

    valid = (
        torch.isfinite(lep0).all(dim=-1)
        & torch.isfinite(lep1).all(dim=-1)
        & has_rest_frame(higgs)
        & has_rest_frame(w0)
        & has_rest_frame(w1)
        & torch.isfinite(w1_axis).all(dim=-1)
        & (axis_norm > eps)
        & (transverse_fraction > eps**0.5)
    )

    n, r, k = basis(w1_h)
    lep0_w = boost_to_rest(lep0_h, w0_h)
    lep1_w = boost_to_rest(lep1_h, w1_h)
    lep0_rest = project(lep0_w, n, r, k)
    lep1_rest = project(lep1_w, n, r, k)
    angles = torch.stack(
        [theta(lep0_rest), phi(lep0_rest), theta(lep1_rest), phi(lep1_rest)], dim=-1
    )
    return valid, angles


def _pathological_rows():
    """Deterministic rows that the validity mask must reject."""

    def w_row(w0_px, w0_py, w0_pz, w0_e):
        w1_e = math.sqrt(80.379**2 + 100.0)
        return torch.tensor([[w0_px, w0_py, w0_pz, w0_e, 0.0, 0.0, 0.0, w1_e]])

    def lep_row(lep0_e, lep1_e=5.0):
        return torch.tensor([[0.0, 0.0, 0.0, lep0_e, 0.0, 0.0, 0.0, lep1_e]])

    extra_lep = torch.cat(
        [
            lep_row(float("nan")),  # NaN lepton energy
            lep_row(5.0),
            lep_row(5.0),
        ]
    )
    extra_wboson = torch.cat(
        [
            w_row(0.0, 0.0, 10.0, math.sqrt(80.379**2 + 100.0)),  # massive W's
            w_row(0.0, 0.0, 10000.0, 10000.0),  # lightlike W: mass2 == 0 exactly
            w_row(0.0, 0.0, 0.0, -90.379),  # negative W energy
        ]
    )
    return extra_lep, extra_wboson


class BoosterAngleRegressionTest(unittest.TestCase):
    def test_booster_angles_unchanged_after_dedupe(self):
        torch.manual_seed(0)
        lep, wboson = _mock_inputs(batch=256)  # massive, finite W's by construction
        extra_lep, extra_wboson = _pathological_rows()
        lep = torch.cat([lep, extra_lep], dim=0)
        wboson = torch.cat([wboson, extra_wboson], dim=0)

        b = Booster(lep, wboson)
        valid, angles = b.lep_theta_phi_with_validity()
        ref_valid, ref_angles = _reference_angles(lep, wboson)

        assert valid.shape == (259,)
        assert valid[:256].all(), "mock rows should remain valid"
        assert not valid[256:].any(), "pathological rows must be flagged invalid"
        assert not valid.all()
        assert torch.equal(valid, ref_valid)
        assert torch.allclose(angles[ref_valid], ref_angles[ref_valid], atol=1e-6)


class FourVectorHuberTest(unittest.TestCase):
    def test_loss_uses_raw_energy_for_both_slots(self):
        truth = torch.tensor([[0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0]])
        prediction = torch.tensor([[1.0, 2.0, 3.0, 7.0, 1.0, 2.0, 3.0, 15.0]])
        loss = fourvec_huber_loss(truth, prediction)

        expected = torch.tensor(20.0 / 8.0)
        torch.testing.assert_close(loss, expected)

    def test_loss_is_finite_for_zero_energy(self):
        truth = torch.zeros((1, 10))
        prediction = torch.zeros((1, 8))

        loss = fourvec_huber_loss(truth, prediction)

        self.assertTrue(torch.isfinite(loss))


class DmetHuberTest(unittest.TestCase):
    def test_loss_uses_raw_dmet_components(self):
        features = torch.zeros((1, 21))
        features[:, :2] = torch.tensor([[1.0, 2.0]])
        features[:, 4:6] = torch.tensor([[-3.0, 4.0]])
        features[:, 16:18] = torch.tensor([[20.0, 30.0]])
        targets = torch.zeros((1, 10))
        targets[:, :2] = torch.tensor([[6.0, 9.0]])
        targets[:, 4:6] = torch.tensor([[8.0, 2.0]])
        prediction = torch.tensor([[6.0, 31.0]])
        loss = dmet_loss(features, targets, prediction)

        expected = torch.tensor(3.5)
        torch.testing.assert_close(loss, expected)


class HiggsMassLossTest(unittest.TestCase):
    def _predictions(self, m_h):
        # Two zero-pz W four-vectors whose sum is a rest-frame Higgs of mass m_h.
        m_h = np.asarray(m_h, dtype=np.float32)
        pred = np.zeros((len(m_h), 8), dtype=np.float32)
        pred[:, 3] = pred[:, 7] = 0.5 * m_h
        return torch.from_numpy(pred)

    def test_zero_loss_on_target_mass(self):
        pred = self._predictions([125.0])

        loss = loss_module.higgs_mass_loss(pred)

        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_is_quadratic_near_target(self):
        pred = self._predictions([125.5]).requires_grad_(True)

        loss = loss_module.higgs_mass_loss(pred)
        loss.backward()

        torch.testing.assert_close(loss, torch.tensor(0.125))
        torch.testing.assert_close(pred.grad[0, [3, 7]], torch.tensor([0.5, 0.5]))

    def test_large_mass_errors_have_linear_penalties_and_bounded_mass_gradients(self):
        pred = self._predictions([130.0, 165.0]).requires_grad_(True)

        loss = loss_module.higgs_mass_loss(pred)
        loss.backward()

        torch.testing.assert_close(loss, torch.tensor((4.5 + 39.5) / 2.0))
        # Both rest-frame masses have the same gradient despite different errors.
        torch.testing.assert_close(pred.grad[:, [3, 7]], torch.full((2, 2), 0.5))

    def test_gradient_pushes_mass_toward_target_from_both_sides(self):
        pred = self._predictions([120.0, 130.0]).requires_grad_(True)

        loss_module.higgs_mass_loss(pred).backward()

        torch.testing.assert_close(
            pred.grad[:, [3, 7]], torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        )

    def test_spacelike_mass_stays_below_target(self):
        """A spacelike pair must not be folded onto the timelike side."""
        spacelike = torch.tensor([[0.0, 0.0, 100.0, 1.0, 0.0, 0.0, 100.0, 1.0]])
        target = 125.0

        loss = loss_module.higgs_mass_loss(spacelike, target_mass=target)

        mass2 = loss_module.invariant_mass2(spacelike[..., :4] + spacelike[..., 4:8])
        self.assertLess(mass2.item(), 0.0)
        signed_mass = -np.sqrt(abs(mass2.item()))
        expected = abs(signed_mass - target) - 0.5
        torch.testing.assert_close(loss, torch.tensor(expected, dtype=torch.float32))

    def test_mass_floor_bounds_gradient_near_the_light_cone(self):
        """m^2 -> 0 must not blow d|m|/dm^2 up the way a bare sqrt would."""
        near_cone = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.001]])
        near_cone.requires_grad_(True)

        loss = loss_module.higgs_mass_loss(near_cone)
        loss.backward()

        mass2 = loss_module.invariant_mass2(near_cone[..., :4] + near_cone[..., 4:8])
        self.assertLess(abs(mass2.item()), loss_module.H_MASS2_FLOOR)
        self.assertTrue(torch.isfinite(near_cone.grad).all())

    def test_spacelike_sum_has_finite_nonzero_corrective_gradients(self):
        pred = torch.tensor(
            [[0.0, 0.0, 100.0, 1.0, 0.0, 0.0, 100.0, 1.0]],
            requires_grad=True,
        )

        loss = loss_module.higgs_mass_loss(pred)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad.abs().sum().item(), 0.0)


class LightningModelLossTest(unittest.TestCase):
    def _basic_model(self, **kwargs):
        return LightningWBoson(
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(21, dtype=np.float32),
            std_scale_train=np.ones(21, dtype=np.float32),
            **kwargs,
        )

    def test_higgs_mass_target_is_fixed(self):
        unsupported_values = (0.0, 124.0, 126.0, float("nan"), float("inf"))
        for value in unsupported_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "fixed at 125"):
                    self._basic_model(higgs_mass_target=value)

    def _ramp_model(self, angular_mmd_ramp_epochs=80):
        return LightningWBoson(
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(21, dtype=np.float32),
            std_scale_train=np.ones(21, dtype=np.float32),
            loss_weights={
                "huber": 2.0,
                "higgs_mass": 6.0,
                "alpha_mmd": 3.0,
                "mass_mmd": 4.0,
                "w_mass_huber": 7.0,
                "angular_mmd": 5.0,
                "dmet": 8.0,
            },
            angular_mmd_ramp_epochs=angular_mmd_ramp_epochs,
        )

    def test_angular_mmd_weight_uses_cosine_ramp(self):
        model = self._ramp_model(angular_mmd_ramp_epochs=80)
        expected = {
            0: 0.0,
            20: 5.0 * (1.0 - math.cos(math.pi * 0.25)) / 2.0,
            40: 2.5,
            60: 5.0 * (1.0 - math.cos(math.pi * 0.75)) / 2.0,
            80: 5.0,
            100: 5.0,
        }

        for epoch, expected_angular_weight in expected.items():
            with self.subTest(epoch=epoch):
                model.trainer = SimpleNamespace(current_epoch=epoch)
                weights = model._effective_loss_weights()

                self.assertAlmostEqual(weights["angular_mmd"], expected_angular_weight)
                self.assertEqual(weights["alpha_mmd"], 3.0)
                self.assertEqual(weights["mass_mmd"], 4.0)

    def test_logged_loss_weights_use_effective_angular_mmd_ramp_weight(self):
        model = self._ramp_model(angular_mmd_ramp_epochs=80)
        model.trainer = SimpleNamespace(current_epoch=40)

        with patch.object(model, "log") as log:
            model._log_loss_weights()

        logged_weights = {call.args[0]: call.args[1] for call in log.call_args_list}
        self.assertAlmostEqual(logged_weights["loss_weight/angular_mmd"], 2.5)
        self.assertEqual(logged_weights["loss_weight/alpha_mmd"], 3.0)
        self.assertEqual(logged_weights["loss_weight/mass_mmd"], 4.0)

    def test_gradient_rest_uses_effective_angular_mmd_ramp_weight(self):
        model = self._ramp_model(angular_mmd_ramp_epochs=80)
        model.trainer = SimpleNamespace(current_epoch=40)
        parameter = next(model.parameters())
        angular_loss = parameter.sum()
        total = 2.5 * angular_loss

        cosines = model._compute_loss_gradient_cosines(
            {"angular_mmd": angular_loss},
            total,
        )

        torch.testing.assert_close(cosines["angular_mmd"]["rest"], torch.tensor(0.0))

    def test_zero_angular_mmd_ramp_applies_full_weight_at_epoch_zero(self):
        model = self._ramp_model(angular_mmd_ramp_epochs=0)
        model.trainer = SimpleNamespace(current_epoch=0)

        self.assertEqual(model._effective_loss_weights()["angular_mmd"], 5.0)

    def test_angular_mmd_ramp_defaults_to_zero(self):
        model = self._basic_model(loss_weights={"angular_mmd": 5.0})
        model.trainer = SimpleNamespace(current_epoch=0)

        self.assertEqual(model.angular_mmd_ramp_epochs, 0)
        self.assertEqual(dict(model.hparams)["angular_mmd_ramp_epochs"], 0)
        self.assertEqual(model._effective_loss_weights()["angular_mmd"], 5.0)

    def test_rejects_invalid_angular_mmd_ramp_epochs(self):
        for value in (
            True,
            -1,
            1.5,
            float("inf"),
            float("nan"),
            Fraction(10**10000, 3),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "angular_mmd_ramp_epochs"):
                    self._ramp_model(angular_mmd_ramp_epochs=value)

    def test_preserves_exact_huge_integral_angular_mmd_ramp_epochs(self):
        ramp_epochs = 10**10000

        model = self._ramp_model(angular_mmd_ramp_epochs=ramp_epochs)

        self.assertEqual(model.angular_mmd_ramp_epochs, ramp_epochs)
        self.assertEqual(dict(model.hparams)["angular_mmd_ramp_epochs"], ramp_epochs)

    def test_weight_decay_is_forwarded_to_optimizer(self):
        model = LightningWBoson(
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(21, dtype=np.float32),
            std_scale_train=np.ones(21, dtype=np.float32),
            weight_decay=0.0123,
        )

        optimizer = model.configure_optimizers()

        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0123)

    def test_absolute_mmd_bandwidths_are_routed_uniformly(self):
        model = self._basic_model(
            mmd_config={
                "alpha": {"kernel": "rbf", "bandwidths": [0.2, 0.4]},
                "mass": {"kernel": "imq", "bandwidths": [0.5]},
                "angular": {"kernel": "imq", "bandwidths": [0.05, 0.5, 5.0]},
            }
        )

        self.assertEqual(
            model._mmd_kwargs("alpha"),
            {"kernel": "rbf", "bandwidths": [0.2, 0.4]},
        )
        self.assertEqual(
            model._mmd_kwargs("mass"),
            {"kernel": "imq", "bandwidths": [0.5]},
        )
        self.assertEqual(
            model._mmd_kwargs("angular"),
            {"kernel": "imq", "bandwidths": [0.05, 0.5, 5.0]},
        )

    def test_rejects_invalid_mmd_config_at_model_construction(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            LightningWBoson(
                input_dim=21,
                d_model=8,
                num_heads=2,
                std_mean_train=np.zeros(21, dtype=np.float32),
                std_scale_train=np.ones(21, dtype=np.float32),
                mmd_config={"alpha": {"bandwidths": [0.0]}},
            )

    def test_rejects_unknown_mmd_config_section(self):
        with self.assertRaisesRegex(ValueError, "unsupported MMD config section.*unknown_section"):
            self._basic_model(mmd_config={"unknown_section": {}})

    def test_public_mmd_results_are_used_without_transformation(self):
        model = self._basic_model(
            loss_weights={
                "huber": 0.0,
                "alpha_mmd": 1.0,
                "mass_mmd": 1.0,
                "angular_mmd": 1.0,
            }
        )
        raw_values = {
            "alpha_mmd": torch.tensor(0.0),
            "mass_mmd": torch.tensor(0.75),
            "angular_mmd": torch.tensor(3.75),
        }

        with (
            patch("model.model.alpha_mmd", return_value=raw_values["alpha_mmd"]),
            patch("model.model.mass_mmd", return_value=raw_values["mass_mmd"]),
            patch("model.model.angular_mmd", return_value=raw_values["angular_mmd"]),
        ):
            total, losses = model._compute_losses(
                torch.zeros((1, 21)),
                torch.zeros((1, 10)),
                torch.zeros((1, 8)),
            )

        self.assertIs(losses["alpha_mmd"], raw_values["alpha_mmd"])
        self.assertIs(losses["mass_mmd"], raw_values["mass_mmd"])
        self.assertIs(losses["angular_mmd"], raw_values["angular_mmd"])
        torch.testing.assert_close(total, sum(raw_values.values()))

    def test_rejects_unsupported_loss_weight_keys(self):
        for key in ("kinematic_loss_mmd_typo",):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, f"unsupported loss_weights key.*{key}"):
                    LightningWBoson(
                        input_dim=21,
                        d_model=8,
                        num_heads=2,
                        std_mean_train=np.zeros(21, dtype=np.float32),
                        std_scale_train=np.ones(21, dtype=np.float32),
                        loss_weights={key: 1.0},
                    )


class SharedValidMaskTest(unittest.TestCase):
    @staticmethod
    def _small_batch_with_invalid_rows():
        batch = 6
        x = torch.zeros(batch, 21)
        x[:, 3] = 10.0
        x[:, 7] = 20.0
        w0 = torch.tensor([30.0, 5.0, 40.0, 100.0])
        w1 = torch.tensor([-20.0, 15.0, -30.0, 90.0])
        y_pred = torch.cat([w0, w1]).repeat(batch, 1)
        y_true = torch.cat(
            [torch.cat([w0 + 1.0, w1]).repeat(batch, 1), torch.full((batch, 2), 80.4)], dim=-1
        )
        x[0, 0] = float("nan")
        y_true[1, 4] = float("inf")
        y_pred[2, 0] = float("nan")
        y_true[3, 9] = float("nan")  # only alpha_mmd's extra finiteness check drops this row
        return x, y_true, y_pred

    def _model(self):
        return LightningWBoson(
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(21, dtype=np.float32),
            std_scale_train=np.ones(21, dtype=np.float32),
            loss_weights={
                "huber": 0.0,
                "higgs_mass": 0.0,
                "w_mass_huber": 0.0,
                "alpha_mmd": 1.0,
                "mass_mmd": 1.0,
                "angular_mmd": 0.0,
                "dmet": 0.0,
            },
        )

    def test_valid_mask_is_passed_outside_mmd_kwargs(self):
        model = self._model()
        x, y_true, y_pred = self._small_batch_with_invalid_rows()
        calls = {}

        def capture(name):
            def wrapper(*args, **kwargs):
                calls[name] = kwargs
                return torch.tensor(0.5)

            return wrapper

        with (
            patch("model.model.alpha_mmd", side_effect=capture("alpha_mmd")),
            patch("model.model.mass_mmd", side_effect=capture("mass_mmd")),
        ):
            model._compute_losses(x, y_true, y_pred)

        for name in ("alpha_mmd", "mass_mmd"):
            self.assertIn("valid_mask", calls[name])
            self.assertIsInstance(calls[name]["valid_mask"], torch.Tensor)
            self.assertEqual(set(calls[name]) - {"valid_mask"}, {"kernel", "bandwidths"})


class NoHighLevelFeaturesTest(unittest.TestCase):
    def _make_model(self, **kwargs):
        return LightningWBoson(
            input_dim=18,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(18, dtype=np.float32),
            std_scale_train=np.ones(18, dtype=np.float32),
            attention_blocks=1,
            attention_dropout=0.0,
            decoder_dropout=0.0,
            **kwargs,
        )

    @staticmethod
    def _valid_inputs(batch_size=4):
        inputs = torch.randn(batch_size, 18)
        for start in (0, 4, 8, 12):
            inputs[:, start + 3] = (
                torch.linalg.vector_norm(inputs[:, start : start + 3], dim=1)
                + torch.rand(batch_size)
                + 0.1
            )
        return inputs

    def test_forward_without_high_level_features(self):
        model = self._make_model().eval()

        predictions = model(self._valid_inputs())

        self.assertEqual(predictions.shape, (4, 8))
        self.assertIsNone(model.model.hl_embed)
        self.assertEqual(model.model.hl_input_dim, 0)
        self.assertEqual(model.model.num_tokens, 5)

    def test_rejects_unsupported_input_width(self):
        with self.assertRaisesRegex(ValueError, "input_dim"):
            LightningWBoson(
                input_dim=20,
                d_model=8,
                num_heads=2,
                std_mean_train=np.zeros(20, dtype=np.float32),
                std_scale_train=np.ones(20, dtype=np.float32),
            )


class AngularMMDTest(unittest.TestCase):
    def test_feature_order_is_w_plus_then_w_minus(self):
        angles = torch.tensor(
            [[0.0, -torch.pi / 2.0, torch.pi, torch.pi]],
        )

        features = loss_module.angular_mmd_features(angles)

        expected = torch.tensor([[-1.0, -1.0, 0.0, 1.0, 0.0, -1.0]])
        torch.testing.assert_close(features, expected, atol=1.0e-6, rtol=0.0)


class WMassHuberTest(unittest.TestCase):
    def test_compares_normalized_predicted_mass_squared_to_target_masses(self):
        y_true = torch.zeros((1, 10))
        y_true[0, 8:] = torch.tensor([40.2, 80.4])
        y_pred = torch.tensor([[0.0, 0.0, 0.0, 80.4, 0.0, 0.0, 0.0, 40.2]])

        loss = loss_module.w_mass_huber_loss(y_true, y_pred)

        expected = torch.tensor(0.5 * 0.75**2)
        torch.testing.assert_close(loss, expected)


class GradientCosineLoggingTest(unittest.TestCase):
    def test_logging_only_mode_emits_cosines_without_changing_weights(self):
        model = LightningWBoson(
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(21, dtype=np.float32),
            std_scale_train=np.ones(21, dtype=np.float32),
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
            inputs[:, start + 3] = (
                torch.linalg.vector_norm(inputs[:, start : start + 3], dim=1) + torch.rand(2) + 0.1
            )
        batch = (inputs, torch.randn(2, 10))
        original_weights = dict(model.loss_weights)

        with patch.object(model, "_compute_batch_losses", return_value=(total, losses)):
            with patch.object(model, "_log_losses"), patch.object(model, "_log_loss_weights"):
                with patch.object(model, "_log_grad_cosines") as log_cosines:
                    model.training_step(batch, 0)
                    model.on_train_epoch_end()

        log_cosines.assert_called_once()
        self.assertEqual(set(log_cosines.call_args.args[0]), {"huber", "higgs_mass"})
        self.assertEqual(model.loss_weights, original_weights)


if __name__ == "__main__":
    unittest.main()
