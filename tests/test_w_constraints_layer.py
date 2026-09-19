import unittest

import torch

from model.layers import WBosonFourVectorLayer, WConstraintsLayer
from model.losses import H_MASS_SCALE, invariant_mass2


def massless(momentum):
    return torch.cat([momentum, torch.linalg.vector_norm(momentum, dim=-1, keepdim=True)], dim=-1)


def mass(fourvec):
    mass2 = invariant_mass2(fourvec)
    return torch.sign(mass2) * mass2.abs().sqrt()


def unconstrained_params(nu_params, met):
    nu0_3 = nu_params[..., 0:3]
    nu1_3 = nu_params[..., 3:6]
    return torch.cat(
        [
            nu0_3[..., :2] - nu1_3[..., :2],
            nu0_3[..., 2:3],
            nu1_3[..., 2:3],
            met - (nu0_3[..., :2] + nu1_3[..., :2]),
        ],
        dim=-1,
    )


def random_event(batch=512, dtype=torch.float64, seed=0):
    """Leptons and neutrino momenta at roughly the scales seen in data.

    Lepton pairs are kept below the Higgs mass, which is what the loader
    guarantees by dropping every event that reaches ``physics.HIGGS_MASS``.
    """
    generator = torch.Generator().manual_seed(seed)

    def sample(width, scale):
        return torch.randn(batch, width, generator=generator, dtype=dtype) * scale

    lep0 = massless(sample(3, 30.0))
    lep1 = massless(sample(3, 30.0))
    nu_params = torch.cat(
        [sample(2, 35.0), sample(1, 100.0), sample(2, 35.0), sample(1, 100.0)], dim=-1
    )

    selected = mass(lep0 + lep1) < H_MASS_SCALE
    return lep0[selected], lep1[selected], nu_params[selected]


class WConstraintsLayerTest(unittest.TestCase):
    def test_higgs_mass_is_exact(self):
        layer = WConstraintsLayer()
        lep0, lep1, nu_params = random_event()

        prediction = layer(lep0, lep1, nu_params)
        higgs_mass = mass(prediction[..., :4] + prediction[..., 4:8])

        self.assertGreater(higgs_mass.numel(), 100)
        torch.testing.assert_close(
            higgs_mass, torch.full_like(higgs_mass, H_MASS_SCALE), atol=1e-6, rtol=0.0
        )

    def test_higgs_mass_is_exact_in_single_precision(self):
        layer = WConstraintsLayer()
        lep0, lep1, nu_params = random_event(dtype=torch.float32)

        prediction = layer(lep0, lep1, nu_params)
        higgs_mass = mass(prediction[..., :4] + prediction[..., 4:8])

        self.assertLess(float((higgs_mass - H_MASS_SCALE).abs().max()), 1e-2)

    def test_neutrinos_stay_massless(self):
        layer = WConstraintsLayer()
        lep0, lep1, nu_params = random_event()

        prediction = layer(lep0, lep1, nu_params)

        for neutrino in (prediction[..., :4] - lep0, prediction[..., 4:8] - lep1):
            self.assertLess(float(invariant_mass2(neutrino).abs().max()), 1e-6)

    def test_only_the_neutrino_direction_reaches_the_output(self):
        layer = WConstraintsLayer()
        lep0, lep1, nu_params = random_event()

        torch.testing.assert_close(layer(lep0, lep1, 3.0 * nu_params), layer(lep0, lep1, nu_params))

    def test_constraint_only_rescales_the_unconstrained_solution(self):
        unconstrained = WBosonFourVectorLayer()
        layer = WConstraintsLayer()
        lep0, lep1, nu_params = random_event()
        generator = torch.Generator().manual_seed(1)
        met = torch.randn(lep0.shape[0], 2, generator=generator, dtype=lep0.dtype) * 40.0

        baseline = unconstrained(lep0, lep1, unconstrained_params(nu_params, met), met)
        prediction = layer(lep0, lep1, nu_params)
        scale = layer.higgs_scale(
            lep0 + lep1, massless(nu_params[..., 0:3]) + massless(nu_params[..., 3:6])
        )

        torch.testing.assert_close(prediction[..., :4] - lep0, scale * (baseline[..., :4] - lep0))
        torch.testing.assert_close(prediction[..., 4:8] - lep1, scale * (baseline[..., 4:8] - lep1))
        self.assertTrue(bool((scale > 0.0).all()))

    def test_gradients_are_finite(self):
        layer = WConstraintsLayer()
        lep0, lep1, nu_params = random_event(dtype=torch.float32)
        nu_params = nu_params.requires_grad_(True)

        layer(lep0, lep1, nu_params).sum().backward()

        self.assertTrue(bool(torch.isfinite(nu_params.grad).all()))
        self.assertGreater(float(nu_params.grad.abs().sum()), 0.0)
