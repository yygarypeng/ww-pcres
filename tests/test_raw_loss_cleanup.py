import inspect
from unittest.mock import patch

import numpy as np
import pytest
import torch

from model import LightningWBoson
from model import losses
from scripts import diagnose_angular_degradation as diagnostics
from train import train


def test_fourvec_raw_gev_values_and_gradients():
    assert hasattr(losses, "fourvec_huber_loss")
    prediction = torch.tensor([[0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -3.0, 4.0]], requires_grad=True)
    loss = losses.fourvec_huber_loss(torch.zeros(1, 10), prediction)
    torch.testing.assert_close(loss, torch.tensor(8.75 / 8.0))
    loss.backward()
    torch.testing.assert_close(
        prediction.grad, torch.tensor([[0.0, 0.5, -0.5, 1.0, -1.0, 1.0, -1.0, 1.0]]) / 8
    )


def test_dmet_raw_gev_values_and_gradients():
    assert "component_scales" not in inspect.signature(losses.dmet_loss).parameters
    features = torch.zeros(1, 21)
    features[:, :2] = torch.tensor([[1.0, 2.0]])
    features[:, 4:6] = torch.tensor([[-3.0, 4.0]])
    features[:, 16:18] = torch.tensor([[20.0, 30.0]])
    targets = torch.zeros(1, 10)
    targets[:, :2] = torch.tensor([[6.0, 9.0]])
    targets[:, 4:6] = torch.tensor([[8.0, 2.0]])
    prediction = torch.tensor([[4.5, 22.0]], requires_grad=True)
    loss = losses.dmet_loss(features, targets, prediction)
    torch.testing.assert_close(loss, torch.tensor(1.3125))
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[0.25, -0.5]]))


def test_mass_features_keep_signed_asinh_without_fitted_statistics():
    assert "center" not in inspect.signature(losses.mass_mmd).parameters
    truth = torch.tensor([[0.0, 0.0, 0.0, 80.4, 80.4, 0.0, 0.0, 0.0, 80.4, 0.0]])
    prediction = torch.tensor([[0.0, 0.0, 0.0, 40.2, 0.0, 0.0, 0.0, 80.4]], requires_grad=True)
    expected_truth = torch.tensor([[0.881373587, -0.881373587]])
    expected_prediction = torch.tensor([[0.247466462, 0.881373587]])
    torch.testing.assert_close(losses._mass_features(truth[:, :8]), expected_truth)
    torch.testing.assert_close(losses._mass_features(prediction), expected_prediction)
    actual = losses.mass_mmd(torch.zeros(1, 21), truth, prediction, bandwidths=(0.5,))
    expected = losses.compute_mmd(expected_prediction, expected_truth, bandwidths=(0.5,))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_training_constructs_model_without_loss_statistics():
    assert "w_fourvec_scales" not in inspect.signature(train.run_training).parameters
    cfg = {
        "parameters": {
            "learning_rate": 1e-4,
            "loss_weights": {"huber": 2.0, "dmet": 3.0},
            "d_model": 8,
            "n_heads": 2,
            "epochs": 1,
        }
    }

    class DataModule:
        test_ds = None

        def train_dataloader(self):
            return [None]

    with (
        patch.object(train, "Trainer") as trainer,
        patch.object(train, "clean_training_output"),
        patch.object(train, "create_loggers", return_value=([], None)),
    ):
        train.run_training(cfg, DataModule(), 21, (np.zeros(21), np.ones(21)), "unused", False)
    model = trainer.return_value.fit.call_args.args[0]
    assert isinstance(model, LightningWBoson)
    prediction = torch.full((1, 8), 2.0, requires_grad=True)
    dmet = torch.tensor([[0.5, -3.0]], requires_grad=True)
    total, terms = model._compute_losses(
        torch.zeros(1, 21), torch.zeros(1, 10), prediction, {"dmet": dmet}
    )
    torch.testing.assert_close(terms["huber"], torch.tensor(1.5))
    torch.testing.assert_close(terms["dmet"], torch.tensor(1.3125))
    torch.testing.assert_close(total, torch.tensor(6.9375))
    total.backward()
    torch.testing.assert_close(prediction.grad, torch.full((1, 8), 0.25))
    torch.testing.assert_close(dmet.grad, torch.tensor([[0.75, -1.5]]))


def test_slot_diagnostic_uses_raw_huber_and_filters_invalid_rows():
    assert hasattr(diagnostics, "_slot_huber")
    prediction = torch.tensor([[0.5, -1.0, 2.0, -3.0, 0.0, 0.0, 0.0, 0.0], [float("nan")] * 8])
    assert diagnostics._slot_huber(prediction, torch.zeros(2, 8), slice(0, 4)) == pytest.approx(
        1.15625
    )
