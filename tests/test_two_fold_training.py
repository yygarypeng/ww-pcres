from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from train import two_fold_train


def _split_arrays():
    return (
        np.arange(12, dtype=np.float32).reshape(6, 2),
        np.arange(6, dtype=np.float32).reshape(6, 1),
        np.arange(8, dtype=np.float32).reshape(4, 2),
        np.arange(4, dtype=np.float32).reshape(4, 1),
        np.arange(8, dtype=np.float32).reshape(4, 2),
        np.arange(4, dtype=np.float32).reshape(4, 1),
    )


@pytest.mark.parametrize(
    ("fold", "expected_train", "expected_val", "expected_test"),
    [
        (0, [2, 6, 10], [2, 6], [0, 4]),
        (1, [0, 4, 8], [0, 4], [2, 6]),
    ],
)
def test_select_fold_splits_uses_training_parity_opposite_to_inference(
    fold, expected_train, expected_val, expected_test
):
    X_train, _, X_val, _, X_test, _ = two_fold_train.select_fold_splits(
        _split_arrays(), fold
    )

    assert X_train[:, 0].tolist() == expected_train
    assert X_val[:, 0].tolist() == expected_val
    assert X_test[:, 0].tolist() == expected_test


def test_select_fold_splits_rejects_an_empty_parity_subset():
    splits = list(_split_arrays())
    splits[2] = splits[2][:1]
    splits[3] = splits[3][:1]

    with pytest.raises(ValueError, match="Fold 0 validation split is empty"):
        two_fold_train.select_fold_splits(tuple(splits), fold=0)


def test_build_fold_datamodule_fits_statistics_on_selected_training_rows(monkeypatch):
    captured = {}
    expected_stats = (np.array([1.0]), np.array([2.0]))

    monkeypatch.setattr(
        two_fold_train.data,
        "load_presplit_data",
        lambda *_args, **_kwargs: _split_arrays(),
    )

    def capture_stats(features):
        captured["features"] = features.copy()
        return expected_stats

    class FakeDataModule:
        def __init__(self, X, Y, **kwargs):
            self.X = X
            self.Y = Y
            self.kwargs = kwargs
            self.was_setup = False

        def setup(self):
            self.was_setup = True

    monkeypatch.setattr(two_fold_train, "compute_neural_input_stats", capture_stats)
    monkeypatch.setattr(two_fold_train, "WBosonDataModule", FakeDataModule)

    cfg = {"parameters": {"batch_size": 2}, "data": {}}
    datamodule, input_dim, stats = two_fold_train.build_fold_datamodule(
        cfg, "unused.h5", fold=0
    )

    np.testing.assert_array_equal(captured["features"], [[2, 3], [6, 7], [10, 11]])
    assert datamodule.was_setup
    assert datamodule.kwargs["X_val"][:, 0].tolist() == [2, 6]
    assert datamodule.kwargs["X_test"][:, 0].tolist() == [0, 4]
    assert input_dim == 2
    assert stats is expected_stats


def test_fold_output_path_keeps_checkpoints_separate():
    assert two_fold_train.fold_output_path("outputs/run", 0) == Path("outputs/run/fold0")
    assert two_fold_train.fold_output_path("outputs/run", 1) == Path("outputs/run/fold1")


def test_parse_args_defaults_to_both_folds_sequentially():
    args = two_fold_train.parse_args([])

    assert args.fold == "both"
    assert not args.parallel


def test_parse_args_rejects_parallel_single_fold():
    with pytest.raises(SystemExit):
        two_fold_train.parse_args(["--fold", "0", "--parallel"])


def test_parallel_commands_relaunch_one_process_per_fold_on_same_gpu():
    args = SimpleNamespace(config="custom.yaml", wandb=True, gpu=0)

    commands = two_fold_train.parallel_commands(args, script_path=Path("runner.py"))

    assert commands == [
        [
            two_fold_train.sys.executable,
            "runner.py",
            "--config",
            "custom.yaml",
            "--fold",
            "0",
            "--wandb",
            "--gpu",
            "0",
        ],
        [
            two_fold_train.sys.executable,
            "runner.py",
            "--config",
            "custom.yaml",
            "--fold",
            "1",
            "--wandb",
            "--gpu",
            "0",
        ],
    ]


def test_wait_for_folds_reports_every_failed_process():
    processes = [(0, SimpleNamespace(wait=lambda: 2)), (1, SimpleNamespace(wait=lambda: 3))]

    with pytest.raises(RuntimeError, match=r"fold 0 \(exit 2\), fold 1 \(exit 3\)"):
        two_fold_train.wait_for_folds(processes)


def test_parallel_launch_failure_stops_an_already_started_fold(monkeypatch):
    class RunningProcess:
        def __init__(self):
            self.terminated = False
            self.waited = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self):
            self.waited = True
            return 0

    first_process = RunningProcess()
    launches = iter((first_process, OSError("could not launch fold 1")))

    def launch(_command):
        result = next(launches)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(two_fold_train.subprocess, "Popen", launch)
    args = SimpleNamespace(config="custom.yaml", wandb=False, gpu=0)

    with pytest.raises(OSError, match="could not launch fold 1"):
        two_fold_train.run_parallel(args)

    assert first_process.terminated
    assert first_process.waited


def test_sequential_training_reconfigures_runtime_before_each_fold(monkeypatch):
    cfg = {"parameters": {"seed": 73}}
    events = []

    monkeypatch.setattr(two_fold_train.base_training, "load_config", lambda _path: cfg)
    monkeypatch.setattr(
        two_fold_train,
        "configure_runtime",
        lambda params, gpu: events.append(("configure", params["seed"], gpu)),
    )
    monkeypatch.setattr(
        two_fold_train,
        "run_fold",
        lambda received_cfg, fold, wandb: events.append(
            ("run", received_cfg is cfg, fold, wandb)
        ),
    )

    two_fold_train.main(["--config", "unused.yaml", "--fold", "both", "--gpu", "0"])

    assert events == [
        ("configure", 73, 0),
        ("run", True, 0, False),
        ("configure", 73, 0),
        ("run", True, 1, False),
    ]
