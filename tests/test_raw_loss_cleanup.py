from unittest.mock import patch

import numpy as np
import torch

from model import LightningWBoson, losses
from train import train


def test_w_fourvec_raw_gev_values_and_gradients():
    prediction = torch.tensor([[0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -3.0, 4.0]], requires_grad=True)
    loss = losses.w_fourvec_loss(torch.zeros(1, 10), prediction)
    torch.testing.assert_close(loss, torch.tensor(1.5))
    loss.backward()
    torch.testing.assert_close(
        prediction.grad, torch.tensor([[0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]]) / 8
    )


def test_dmet_raw_gev_values_and_gradients():
    features = torch.zeros(1, 21)
    features[:, :2] = torch.tensor([[1.0, 2.0]])
    features[:, 4:6] = torch.tensor([[-3.0, 4.0]])
    features[:, 16:18] = torch.tensor([[20.0, 30.0]])
    targets = torch.zeros(1, 10)
    targets[:, :2] = torch.tensor([[6.0, 9.0]])
    targets[:, 4:6] = torch.tensor([[8.0, 2.0]])
    prediction = torch.tensor([[4.5, 22.0]], requires_grad=True)
    loss = losses.dmet_loss(features, targets, prediction)
    torch.testing.assert_close(loss, torch.tensor(1.75))
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[0.5, -0.5]]))


def test_w_mass_mmd_uses_bounded_mass_features():
    """Raw m^2 would sit far outside every configured bandwidth and collapse the kernel."""
    truth = torch.tensor([[0.0, 0.0, 0.0, 80.4, 80.4, 0.0, 0.0, 0.0, 80.4, 0.0]])
    prediction = torch.tensor([[0.0, 0.0, 0.0, 40.2, 0.0, 0.0, 0.0, 80.4]], requires_grad=True)
    expected_truth = torch.tensor([[0.881374, -0.881374]])
    expected_prediction = torch.tensor([[0.247466, 0.881374]])

    torch.testing.assert_close(
        losses.mass_mmd_features(truth[..., :8]), expected_truth, atol=1e-6, rtol=0.0
    )

    actual = losses.w_mass_mmd(torch.zeros(1, 21), truth, prediction, bandwidths=(1.0,))
    expected = losses.compute_mmd(expected_prediction, expected_truth, bandwidths=(1.0,))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_training_constructs_model_with_configured_loss_weights():
    cfg = {
        "parameters": {
            "learning_rate": 1e-4,
            "loss_weights": {"w_fourvec": 2.0, "dmet": 3.0},
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
    torch.testing.assert_close(terms["w_fourvec"], torch.tensor(2.0))
    torch.testing.assert_close(terms["dmet"], torch.tensor(1.75))
    torch.testing.assert_close(total, torch.tensor(9.25))
    total.backward()
    torch.testing.assert_close(prediction.grad, torch.full((1, 8), 0.25))
    torch.testing.assert_close(dmet.grad, torch.tensor([[1.5, -1.5]]))
