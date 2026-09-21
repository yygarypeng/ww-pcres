import numpy as np
import pytest

from train import k_fold_train


def test_build_fold_datamodule_fits_statistics_on_selected_training_rows(monkeypatch):
    captured = {}
    expected_stats = (np.array([1.0]), np.array([2.0]))

    monkeypatch.setattr(
        k_fold_train.data,
        "load_presplit_data",
        lambda *_args, **_kwargs: _pooled_arrays(),
    )

    def capture_stats(features):
        captured["features"] = features.copy()
        return expected_stats

    class FakeDataModule:
        def __init__(self, X, Y, **kwargs):
            self.X = X
            self.Y = Y
            self.kwargs = kwargs

    monkeypatch.setattr(k_fold_train.base_training, "compute_neural_input_stats", capture_stats)
    monkeypatch.setattr(k_fold_train.base_training, "WBosonDataModule", FakeDataModule)

    cfg = {"parameters": {"batch_size": 2}, "data": {}}
    datamodule, input_dim, stats = k_fold_train.build_fold_datamodule(cfg, "unused.h5", fold=0)

    expected_training = [[2, 3], [6, 7], [102, 103], [106, 107]]
    np.testing.assert_array_equal(captured["features"], expected_training)
    np.testing.assert_array_equal(datamodule.X, expected_training)
    assert datamodule.kwargs["X_val"][:, 0].tolist() == [0, 4, 100, 104]
    assert datamodule.kwargs["X_test"][:, 0].tolist() == [200, 202, 204, 206]
    assert input_dim == 2
    assert stats is expected_stats


def test_parse_args_defaults_to_every_fold():
    args = k_fold_train.parse_args([])

    assert args.fold == "all"


def test_parse_args_rejects_a_fold_that_is_neither_an_index_nor_all():
    with pytest.raises(SystemExit):
        k_fold_train.parse_args(["--fold", "middle"])


def test_rejects_non_integer_configured_fold_count():
    with pytest.raises(ValueError, match="integer"):
        k_fold_train.resolve_folds({"parameters": {"folds": 2.9}})


def test_sequential_training_reconfigures_runtime_before_each_fold(monkeypatch):
    cfg = {"parameters": {"seed": 73}}
    events = []

    monkeypatch.setattr(k_fold_train.base_training, "load_config", lambda _path: cfg)
    monkeypatch.setattr(
        k_fold_train.base_training,
        "configure_runtime",
        lambda params, gpu: events.append(("configure", params["seed"], gpu)),
    )
    monkeypatch.setattr(
        k_fold_train,
        "run_fold",
        lambda received_cfg, fold, wandb, folds: events.append(
            ("run", received_cfg is cfg, fold, wandb, folds)
        ),
    )

    k_fold_train.main(["--config", "unused.yaml", "--fold", "all", "--gpu", "0"])

    assert events == [
        ("configure", 73, 0),
        ("run", True, 0, False, 2),
        ("configure", 73, 0),
        ("run", True, 1, False, 2),
    ]


def test_training_takes_the_fold_count_from_the_config(monkeypatch):
    cfg = {"parameters": {"seed": 73, "folds": 4}}
    events = []
    monkeypatch.setattr(k_fold_train.base_training, "load_config", lambda _path: cfg)
    monkeypatch.setattr(k_fold_train.base_training, "configure_runtime", lambda params, gpu: None)
    monkeypatch.setattr(
        k_fold_train,
        "run_fold",
        lambda _cfg, fold, _w, folds: events.append((fold, folds)),
    )

    k_fold_train.main(["--config", "unused.yaml"])

    assert events == [(0, 4), (1, 4), (2, 4), (3, 4)]


def test_training_a_single_fold_leaves_the_others_alone(monkeypatch):
    cfg = {"parameters": {"seed": 73, "folds": 8}}
    events = []
    monkeypatch.setattr(k_fold_train.base_training, "load_config", lambda _path: cfg)
    monkeypatch.setattr(k_fold_train.base_training, "configure_runtime", lambda params, gpu: None)
    monkeypatch.setattr(
        k_fold_train,
        "run_fold",
        lambda _cfg, fold, _w, folds: events.append((fold, folds)),
    )

    k_fold_train.main(["--config", "unused.yaml", "--fold", "5"])

    assert events == [(5, 8)]


def _pooled_arrays():
    """Nine split arrays whose pooled eventNumbers run 0..7 in row order."""
    return (
        np.arange(0, 8, dtype=np.float32).reshape(4, 2),
        np.arange(0, 4, dtype=np.float32).reshape(4, 1),
        np.arange(0, 4, dtype=np.uint64),
        np.arange(100, 108, dtype=np.float32).reshape(4, 2),
        np.arange(100, 104, dtype=np.float32).reshape(4, 1),
        np.arange(4, 8, dtype=np.uint64),
        np.arange(200, 208, dtype=np.float32).reshape(4, 2),
        np.arange(200, 204, dtype=np.float32).reshape(4, 1),
        np.arange(200, 204, dtype=np.uint64),
    )


def _with_pooled_event_numbers(event_numbers):
    """The pooled arrays, relabelled with the given train+validation eventNumbers."""
    splits = list(_pooled_arrays())
    splits[2], splits[5] = np.split(np.asarray(event_numbers, dtype=np.uint64), 2)
    return tuple(splits)


def test_folds_rotate_validation_through_train_and_val():
    splits = _pooled_arrays()

    seen = []
    for fold in range(2):
        _, _, X_val, _, _, _ = k_fold_train.select_fold_splits(splits, fold, folds=2)
        seen.append(X_val[:, 0].tolist())

    # Every pooled row is validated exactly once, and the pool spans both groups.
    assert sorted(seen[0] + seen[1]) == [0.0, 2.0, 4.0, 6.0, 100.0, 102.0, 104.0, 106.0]
    assert set(seen[0]).isdisjoint(seen[1])


def test_folds_follow_event_numbers_rather_than_row_order():
    # Row order says 0, 1, 2, ...; these eventNumbers put the last row of each group in fold 0.
    splits = _with_pooled_event_numbers([1, 3, 5, 8, 9, 11, 13, 16])

    X_fit, _, X_check, _, _, _ = k_fold_train.select_fold_splits(splits, 0, folds=2)

    assert X_check[:, 0].tolist() == [6.0, 106.0]
    assert X_fit[:, 0].tolist() == [0.0, 2.0, 4.0, 100.0, 102.0, 104.0]


def test_rejects_negative_event_numbers():
    with pytest.raises(ValueError, match="non-negative"):
        k_fold_train.fold_of_event(np.array([1, -3], dtype=np.int64), 2)


def test_rejects_splits_without_event_numbers():
    with pytest.raises(ValueError, match="nine"):
        k_fold_train.select_fold_splits(_pooled_arrays()[:6], 0, folds=2)


def test_every_fold_keeps_the_whole_test_group():
    splits = _pooled_arrays()

    for fold in range(4):
        *_, X_test, _ = k_fold_train.select_fold_splits(splits, fold, folds=4)
        assert X_test[:, 0].tolist() == [200.0, 202.0, 204.0, 206.0]


def test_each_fold_trains_on_the_rest_of_the_pool():
    splits = _pooled_arrays()

    X_fit, _, X_check, _, _, _ = k_fold_train.select_fold_splits(splits, 0, folds=4)

    assert X_fit.shape[0] == 6
    assert X_check.shape[0] == 2


@pytest.mark.parametrize("folds", [1, 0, -1])
def test_rejects_fewer_than_two_folds(folds):
    with pytest.raises(ValueError, match="at least 2"):
        k_fold_train.select_fold_splits(_pooled_arrays(), 0, folds=folds)
