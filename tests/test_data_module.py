import re

import numpy as np
import pytest
import torch

from data.data_module import WBosonDataModule


X = np.zeros((4, 3), dtype=np.float64)
Y = np.zeros((4, 2), dtype=np.float64)


@pytest.mark.parametrize(
    ("split", "split_x", "split_y", "message"),
    [
        (
            "val",
            np.zeros((3, 3)),
            np.zeros((2, 2)),
            "X_val and Y_val must have the same number of samples, got 3 and 2",
        ),
        (
            "val",
            np.zeros((3, 4)),
            np.zeros((3, 2)),
            "X and X_val must have matching feature dimensions, got (3,) and (4,)",
        ),
        (
            "val",
            np.zeros((3, 3)),
            np.zeros((3, 1)),
            "Y and Y_val must have matching target dimensions, got (2,) and (1,)",
        ),
        (
            "test",
            np.zeros((3, 3)),
            np.zeros((2, 2)),
            "X_test and Y_test must have the same number of samples, got 3 and 2",
        ),
        (
            "test",
            np.zeros((3, 4)),
            np.zeros((3, 2)),
            "X and X_test must have matching feature dimensions, got (3,) and (4,)",
        ),
        (
            "test",
            np.zeros((3, 3)),
            np.zeros((3, 1)),
            "Y and Y_test must have matching target dimensions, got (2,) and (1,)",
        ),
    ],
)
def test_presplit_shape_validation(split, split_x, split_y, message):
    kwargs = {
        "X_val": np.zeros((3, 3)),
        "Y_val": np.zeros((3, 2)),
    }
    kwargs[f"X_{split}"] = split_x
    kwargs[f"Y_{split}"] = split_y

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        WBosonDataModule(X, Y, **kwargs)


def test_presplit_validation_is_checked_before_test_validation():
    with pytest.raises(
        ValueError,
        match=r"^Y and Y_val must have matching target dimensions, got \(2,\) and \(1,\)$",
    ):
        WBosonDataModule(
            X,
            Y,
            X_val=np.zeros((3, 3)),
            Y_val=np.zeros((3, 1)),
            X_test=np.zeros((3, 3)),
            Y_test=np.zeros((2, 2)),
        )


def test_training_shuffle_continues_from_restored_generator_state():
    features = np.arange(12, dtype=np.float32).reshape(-1, 1)
    targets = np.zeros((12, 1), dtype=np.float32)

    def make_datamodule():
        datamodule = WBosonDataModule(
            features,
            targets,
            X_val=features[:2],
            Y_val=targets[:2],
            batch_size=4,
            seed=73,
            num_workers=0,
            pin_memory=False,
        )
        datamodule.setup()
        return datamodule

    def epoch_order(datamodule):
        return torch.cat(
            [batch_features[:, 0] for batch_features, _ in datamodule.train_dataloader()]
        )

    uninterrupted = make_datamodule()
    initial_order = epoch_order(uninterrupted)
    saved_state = uninterrupted.state_dict()
    expected_next_order = epoch_order(uninterrupted)

    restored = make_datamodule()
    restored.load_state_dict(saved_state)
    restored_next_order = epoch_order(restored)

    torch.testing.assert_close(restored_next_order, expected_next_order, rtol=0, atol=0)
    assert not torch.equal(restored_next_order, initial_order)
