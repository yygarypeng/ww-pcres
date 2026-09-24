import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.data_module import WBosonDataModule

X = np.zeros((4, 3), dtype=np.float64)
Y = np.zeros((4, 2), dtype=np.float64)


def test_spawn_workers_do_not_serialize_the_attached_trainer():
    datamodule = WBosonDataModule(
        X, Y, X_val=X, Y_val=Y, num_workers=1, multiprocessing_context="spawn"
    )
    # Lightning attaches the live trainer before starting loaders. A local lambda
    # cannot be pickled and stands in for state that must stay in the parent.
    datamodule.trainer = SimpleNamespace(parent_only=lambda: None)
    features, targets = next(iter(datamodule.val_dataloader()))
    torch.testing.assert_close(features, torch.tensor(X, dtype=torch.float32))
    torch.testing.assert_close(targets, torch.tensor(Y, dtype=torch.float32))


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


@pytest.mark.parametrize("loader_name", ["val_dataloader", "test_dataloader"])
def test_evaluation_splits_preserve_source_order(loader_name):
    features = np.arange(12, dtype=np.float32).reshape(-1, 1)
    targets = features + 100.0
    datamodule = WBosonDataModule(
        features[:4],
        targets[:4],
        X_val=features[4:8],
        Y_val=targets[4:8],
        X_test=features[8:],
        Y_test=targets[8:],
        batch_size=2,
        seed=73,
        num_workers=0,
        pin_memory=False,
    )

    batches = list(getattr(datamodule, loader_name)())
    loaded_features = torch.cat([batch_features for batch_features, _ in batches])
    loaded_targets = torch.cat([batch_targets for _, batch_targets in batches])
    start = 4 if loader_name == "val_dataloader" else 8

    torch.testing.assert_close(
        loaded_features[:, 0], torch.arange(start, start + 4, dtype=torch.float32)
    )
    torch.testing.assert_close(
        loaded_targets[:, 0], torch.arange(start + 100, start + 104, dtype=torch.float32)
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
