import gc
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sweep import resources, runner


def test_worker_candidates_use_unique_bounded_counts():
    assert resources.worker_candidates(30) == (0, 2, 4, 8, 16)
    assert resources.worker_candidates(3) == (0, 2)


def test_worker_selection_breaks_throughput_ties_toward_fewer_workers():
    assert resources.choose_fastest_worker_count({0: 100.0, 4: 140.0, 8: 140.0}) == 4


def test_spawn_worker_memory_check_includes_datasets_and_prefetch(monkeypatch):
    splits = tuple(np.zeros((32, width), dtype=np.float32) for width in (18, 10, 18, 10))
    cfg = {"parameters": {"batch_size": 8192, "prefetch_factor": 2}}
    monkeypatch.setattr(
        resources.shutil, "disk_usage", lambda path: SimpleNamespace(free=32 * 1024**2)
    )
    assert resources.safe_worker_count(cfg, splits, 2) == 2
    assert resources.safe_worker_count(cfg, splits, 16) == 0


def test_sweep_and_benchmark_fall_back_when_shared_memory_is_full(monkeypatch):
    splits = tuple(np.zeros((32, width), dtype=np.float32) for width in (18, 10, 18, 10))
    cfg = {"parameters": {"batch_size": 8, "num_workers": 2, "pin_memory": False}}
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    assert runner.build_datamodule(cfg, splits).num_workers == 0
    assert resources.benchmark_dataloader_workers(cfg, splits, candidates=(2, 4), steps=2) == 0


def test_worker_benchmark_moves_batches_without_building_a_model(tmp_path):
    splits = (
        np.zeros((64, 18), dtype=np.float32),
        np.zeros((64, 10), dtype=np.float32),
        np.zeros((16, 18), dtype=np.float32),
        np.zeros((16, 10), dtype=np.float32),
    )
    cfg = {"parameters": {"batch_size": 16, "pin_memory": False}}
    output = tmp_path / "resources.json"

    selected = resources.benchmark_dataloader_workers(
        cfg, splits, candidates=(0,), steps=2, device="cpu", output_path=output
    )

    assert selected == 0
    payload = json.loads(output.read_text())
    assert payload["selected_num_workers"] == 0
    assert float(payload["rows_per_second"]["0"]) > 0.0


@pytest.mark.parametrize("prefetch", [None, 0, 2])
def test_benchmark_accepts_the_same_prefetch_settings_as_training(prefetch):
    splits = tuple(np.zeros((32, width), dtype=np.float32) for width in (18, 10, 18, 10))
    cfg = {"parameters": {"batch_size": 8, "prefetch_factor": prefetch, "pin_memory": False}}
    datamodule = runner.build_datamodule(cfg, splits, num_workers=1)
    assert len(list(datamodule.train_dataloader())) == 4
    assert resources.benchmark_dataloader_workers(cfg, splits, (1,), steps=2, device="cpu") == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires pinned CUDA loaders")
@pytest.mark.parametrize("operation", ["trial", "interrupted_trial", "benchmark"])
def test_repeated_sweep_loaders_release_file_descriptors(operation):
    """Repeated trials must not retain worker pipes until the driver exits."""
    splits = tuple(np.zeros((32, width), dtype=np.float32) for width in (18, 10, 18, 10))
    cfg = {
        "parameters": {
            "batch_size": 8,
            "num_workers": 2,
            "persistent_workers": True,
            "pin_memory": True,
        }
    }

    def run_loaders():
        if operation == "benchmark":
            resources.benchmark_dataloader_workers(
                cfg, splits, candidates=(2,), steps=2, device="cuda"
            )
        else:
            datamodule = runner.build_datamodule(cfg, splits)
            # Training, validation, and checkpoint scoring create separate loaders.
            for loader in (
                datamodule.train_dataloader(),
                datamodule.val_dataloader(),
                datamodule.val_dataloader(),
            ):
                for features, targets in loader:
                    assert features.is_pinned() and targets.is_pinned()
                    if operation == "interrupted_trial":
                        break

    descriptor_counts = []
    original_children = {child.pid for child in multiprocessing.active_children()}
    for _ in range(6):
        run_loaders()
        gc.collect()
        assert {child.pid for child in multiprocessing.active_children()} == original_children
        descriptor_counts.append(len(list(Path("/proc/self/fd").iterdir())))

    # Allow one-time runtime bookkeeping, but no per-trial growth in worker pipes.
    assert max(descriptor_counts) - min(descriptor_counts) <= 4, descriptor_counts
