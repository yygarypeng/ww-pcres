"""Exercise real CUDA training across the prune-to-next-trial boundary."""

import multiprocessing
import weakref
from pathlib import Path

import numpy as np
import optuna
import pytest
import torch

from physics.torchBoost import _mock_inputs
from sweep import runner, workflow


class AlternatingPruner(optuna.pruners.BasePruner):
    def prune(self, study, trial):
        return trial.number % 2 == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA training")
@pytest.mark.parametrize(("num_workers", "full_epochs"), [(0, False), (1, False), (0, True)])
def test_pruned_gpu_trials_release_resources_before_the_next_trial(
    tmp_path, monkeypatch, num_workers, full_epochs
):
    torch.manual_seed(2330)
    leptons, bosons = _mock_inputs(batch=96)
    x = torch.cat([leptons, torch.zeros(96, 10)], dim=1).numpy()
    y = torch.cat([bosons, torch.full((96, 2), 80.4)], dim=1).numpy()
    splits = (x[:64], y[:64], x[64:], y[64:])
    stats = (np.zeros(18, dtype=np.float32), np.ones(18, dtype=np.float32))
    monkeypatch.setattr(runner, "fold_inputs", lambda cfg, fold: (splits, stats))
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path)
    cfg = {
        "parameters": {
            "batch_size": 16,
            "seed": 2330,
            "num_workers": num_workers,
            "persistent_workers": True,
            "pin_memory": True,
            "learning_rate": 1.0e-4,
            "d_model": 8,
            "n_heads": 2,
            "attention_blocks": 1,
            "loss_weights": {"w_fourvec": 1.0},
            "early_stopping_patience": 0,
        }
    }
    references = []
    trainer_class = runner.Trainer

    def track_trainer(*args, **kwargs):
        trainer = trainer_class(*args, **kwargs)
        references.append(weakref.ref(trainer))
        return trainer

    monkeypatch.setattr(runner, "Trainer", track_trainer)
    children = {child.pid for child in multiprocessing.active_children()}
    descriptors = []

    def check_cleanup(study, trial):
        assert all(reference() is None for reference in references)
        assert {child.pid for child in multiprocessing.active_children()} == children
        descriptors.append(len(list(Path("/proc/self/fd").iterdir())))

    def objective(trial):
        report = runner.run_trial(
            cfg,
            tmp_path / f"trial_{trial.number}",
            epochs=3,
            full_epochs=full_epochs,
            callback_factory=lambda fidelity: (
                [] if full_epochs else [workflow.OptunaPruning(trial, fidelity, warmup_epochs=0)]
            ),
        )
        assert np.isfinite(report["composite"])
        assert Path(report["checkpoint"]).exists()
        if full_epochs:
            assert report["epochs_trained"] == 3
        return report["composite"]

    study = optuna.create_study(pruner=AlternatingPruner())
    study.optimize(objective, n_trials=4, gc_after_trial=True, callbacks=[check_cleanup])
    expected = (
        ["COMPLETE"] * 4
        if full_epochs
        else [
            "PRUNED",
            "COMPLETE",
            "PRUNED",
            "COMPLETE",
        ]
    )
    assert [trial.state.name for trial in study.trials] == expected
    # After the first complete run, repeated pruning/completion must not leak pipes.
    assert max(descriptors[1:]) - min(descriptors[1:]) <= 4, descriptors
