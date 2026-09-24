"""Runtime resource selection for the sweep."""

import json
import math
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


def worker_candidates(cpu_count):
    """Small bounded worker ladder for an in-memory dataset."""
    available = max(1, int(cpu_count))
    return tuple(count for count in (0, 2, 4, 8, 16) if count <= available)


def choose_fastest_worker_count(samples):
    """Choose maximum throughput, preferring fewer workers on ties."""
    if not samples:
        raise ValueError("at least one worker benchmark is required")
    return max(samples, key=lambda count: (samples[count], -count))


def safe_worker_count(cfg, splits, workers):
    """Use the main process if spawned tensor storage will not fit in shared memory."""
    workers = max(0, int(workers or 0))
    shared_memory = Path("/dev/shm")
    if not workers or not shared_memory.exists():
        return workers
    params = cfg["parameters"]
    # Budget float32 datasets, prefetched batches, and 1 MiB for queues.
    dataset_bytes = sum(array.size * 4 for array in splits)
    row_bytes = sum(math.prod(array.shape[1:]) * 4 for array in splits)
    prefetch = params.get("prefetch_factor")
    prefetch = 2 if prefetch is None else max(1, int(prefetch))
    required = dataset_bytes + row_bytes * int(params["batch_size"]) * workers * prefetch
    available = shutil.disk_usage(shared_memory).free
    if required + 1024**2 > available:
        print(
            f"Using 0 loader workers: {workers} spawned workers need about "
            f"{required / 1024**2:.1f} MiB shared memory; "
            f"/dev/shm has {available / 1024**2:.1f} MiB free.",
            flush=True,
        )
        return 0
    return workers


def benchmark_dataloader_workers(
    cfg,
    splits,
    candidates,
    *,
    steps=32,
    device=None,
    output_path=None,
):
    """Measure host-to-device batch throughput without constructing or training a model."""
    params = cfg["parameters"]
    prefetch = params.get("prefetch_factor")
    prefetch = 2 if prefetch is None else max(1, int(prefetch))
    x_fit, y_fit = (torch.as_tensor(array, dtype=torch.float32) for array in splits[:2])
    dataset = TensorDataset(x_fit.contiguous(), y_fit.contiguous())
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    throughputs = {}

    candidates = dict.fromkeys(safe_worker_count(cfg, splits, count) for count in candidates)
    for workers in candidates:
        loader = DataLoader(
            dataset,
            batch_size=int(params["batch_size"]),
            shuffle=False,
            num_workers=workers,
            pin_memory=bool(params.get("pin_memory", True) and device.type == "cuda"),
            # Match trial loaders: pinned persistent workers retain pipes until exit.
            persistent_workers=False,
            prefetch_factor=prefetch if workers else None,
            multiprocessing_context="spawn" if workers else None,
        )
        iterator = iter(loader)
        try:
            rows = 0
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            for _ in range(max(1, int(steps))):
                try:
                    x_batch, y_batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    x_batch, y_batch = next(iterator)
                x_batch.to(device, non_blocking=True)
                y_batch.to(device, non_blocking=True)
                rows += len(x_batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = max(time.perf_counter() - started, 1.0e-12)
            throughputs[workers] = rows / elapsed
        finally:
            # Stop workers even when the benchmark ends before exhausting the loader.
            del iterator, loader

    selected = choose_fastest_worker_count(throughputs)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "rows_per_second": {str(key): value for key, value in throughputs.items()},
                    "selected_num_workers": selected,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return selected
