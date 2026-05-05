import argparse
import copy
import itertools
import os
import random
import signal
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent
RUN_NAME_ALIASES = {
    "parameters.loss_weights.dinu_pt": "pt",
    "parameters.loss_weights.angular_loss_mmd": "ang",
    "parameters.loss_weights.higgs_mass": "higgs",
}
TOKEN_REPLACEMENTS = {
    ".": "p",
    "-": "m",
    "+": "",
    "/": "_",
    " ": "",
    ":": "_",
}


@dataclass
class Job:
    name: str
    config: Path
    log: Path
    run_dir: Path
    parameters: dict


@dataclass
class RunningJob:
    job: Job
    gpu: str | None
    attempt: int
    process: subprocess.Popen
    log_handle: object


class TerminationRequested(Exception):
    pass


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def resolve_path(raw_path):
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def set_nested(config, dotted_key, value):
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def parse_override(raw):
    if "=" not in raw:
        raise ValueError(f"Override must use dotted.key=value format, got: {raw}")

    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override has an empty key: {raw}")
    return key, yaml.safe_load(value)


def scan_parameters(config):
    params = config.get("scan", {}).get("parameters")
    if not isinstance(params, dict) or not params:
        raise ValueError("Config must contain scan.parameters")

    normalized = {}
    for key, values in params.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"scan parameter '{key}' must be a non-empty list")
        normalized[str(key)] = values
    return normalized


def safe_token(value):
    text = f"{value:g}" if isinstance(value, float) else str(value)
    for old, new in TOKEN_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def build_run_name(index, values_by_key):
    tokens = [
        f"{RUN_NAME_ALIASES.get(key, key.split('.')[-1])}-{safe_token(value)}"
        for key, value in values_by_key.items()
    ]
    return f"scan_{index:04d}_" + "_".join(tokens)


def parse_gpus(raw):
    if raw is None or str(raw).lower() == "cpu":
        return [None]
    if isinstance(raw, list):
        return [str(item) for item in raw] or [None]

    gpus = [item.strip() for item in str(raw).split(",") if item.strip()]
    return gpus or [None]


def choose_combinations(params, strategy, num_samples, seed):
    keys = list(params)
    combinations = [
        dict(zip(keys, values))
        for values in itertools.product(*(params[key] for key in keys))
    ]

    if strategy == "grid":
        return combinations
    if strategy != "random":
        raise ValueError("scan.strategy must be 'grid' or 'random'")
    if num_samples is None:
        raise ValueError("scan.num_samples is required when scan.strategy is random")

    rng = random.Random(seed)
    if num_samples >= len(combinations):
        rng.shuffle(combinations)
        return combinations
    return rng.sample(combinations, num_samples)


def scan_root_for(args):
    scan_root = resolve_path(args.scan_root)
    if args.no_timestamp:
        return scan_root
    return scan_root / datetime.now().strftime("%Y%m%d_%H%M%S")


def generated_config(
    base_config,
    config_path,
    index,
    run_name,
    run_dir,
    values_by_key,
    overrides,
):
    config = copy.deepcopy(base_config)
    config.pop("scan", None)

    for key, value in values_by_key.items():
        set_nested(config, key, value)
    for key, value in overrides:
        set_nested(config, key, value)

    config.setdefault("paths", {})["saved_path"] = str(run_dir)
    config["scan_metadata"] = {
        "base_config": str(config_path.resolve()),
        "run_index": index,
        "run_name": run_name,
        "scanned_parameters": values_by_key,
    }
    return config


def write_manifest(scan_root, config_path, strategy, wandb_project, jobs):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_config": str(config_path.resolve()),
        "strategy": strategy,
        "wandb_project": wandb_project,
        "jobs": [
            {
                "name": job.name,
                "config": str(job.config),
                "log": str(job.log),
                "run_dir": str(job.run_dir),
                "parameters": job.parameters,
            }
            for job in jobs
        ],
    }
    with open(scan_root / "manifest.yaml", "w") as file:
        yaml.safe_dump(manifest, file, sort_keys=False)


def build_jobs(base_config, config_path, args):
    scan_cfg = base_config.get("scan", {})
    strategy = scan_cfg.get("strategy", "grid")
    combinations = choose_combinations(
        scan_parameters(base_config),
        strategy,
        scan_cfg.get("num_samples"),
        int(scan_cfg.get("seed", 114)),
    )
    overrides = [parse_override(raw) for raw in args.override]

    scan_root = scan_root_for(args)
    configs_dir = scan_root / "configs"
    logs_dir = scan_root / "logs"
    runs_dir = scan_root / "runs"
    configs_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for index, values_by_key in enumerate(combinations, start=1):
        run_name = build_run_name(index, values_by_key)
        run_dir = runs_dir / run_name
        config_path_out = configs_dir / f"{run_name}.yaml"

        config = generated_config(
            base_config,
            config_path,
            index,
            run_name,
            run_dir,
            values_by_key,
            overrides,
        )
        with open(config_path_out, "w") as file:
            yaml.safe_dump(config, file, sort_keys=False)

        jobs.append(
            Job(
                name=run_name,
                config=config_path_out,
                log=logs_dir / f"{run_name}.log",
                run_dir=run_dir,
                parameters=values_by_key,
            )
        )

    write_manifest(scan_root, config_path, strategy, args.wandb_project, jobs)
    return scan_root, jobs


def remove_job_run_dir(job):
    if not job.run_dir.exists():
        return
    if not job.run_dir.is_dir():
        raise ValueError(f"run_dir exists but is not a directory: {job.run_dir}")
    shutil.rmtree(job.run_dir)


def train_command(job, args):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "train.py"),
        "--config",
        str(job.config),
        "--run-name",
        job.name,
        "--wandb-project",
        args.wandb_project,
    ]
    if args.wandb:
        cmd.append("--wandb")
    return cmd


def train_env(gpu):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("WANDB_START_METHOD", "thread")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def start_job(job, gpu, args, attempt, max_attempts):
    if attempt > 1:
        remove_job_run_dir(job)

    cmd = train_command(job, args)
    env = train_env(gpu)
    log_handle = open(job.log, "w", buffering=1)
    log_handle.write(f"Command: {' '.join(cmd)}\n")
    log_handle.write(f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', '<unset>')}\n\n")
    log_handle.write(f"Attempt: {attempt}/{max_attempts}\n\n")

    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    return RunningJob(
        job=job,
        gpu=gpu,
        attempt=attempt,
        process=process,
        log_handle=log_handle,
    )


def stop_active(active):
    for running in active:
        if running.process.poll() is None:
            os.killpg(running.process.pid, signal.SIGTERM)

    deadline = time.time() + 20
    for running in active:
        while running.process.poll() is None and time.time() < deadline:
            time.sleep(0.5)
        if running.process.poll() is None:
            os.killpg(running.process.pid, signal.SIGKILL)
        running.log_handle.close()


def request_termination(signum, _frame):
    raise TerminationRequested(f"received signal {signum}")


def launch_pending(pending, active, args, gpus, next_gpu, max_parallel, max_attempts):
    while pending and len(active) < max_parallel:
        job, attempt = pending.popleft()
        gpu = gpus[next_gpu % len(gpus)]
        next_gpu += 1

        running = start_job(job, gpu, args, attempt, max_attempts)
        active.append(running)
        print(
            f"Started {job.name} attempt={attempt}/{max_attempts} "
            f"on gpu={gpu}; log={job.log}"
        )
    return next_gpu


def poll_active(active, pending, finished, max_attempts):
    still_active = []
    for running in active:
        return_code = running.process.poll()
        if return_code is None:
            still_active.append(running)
            continue

        running.log_handle.close()
        job = running.job
        attempt = running.attempt
        if return_code != 0 and attempt < max_attempts:
            next_attempt = attempt + 1
            pending.append((job, next_attempt))
            print(
                f"Finished {job.name} attempt={attempt}/{max_attempts}: "
                f"failed ({return_code}); retrying attempt={next_attempt}/{max_attempts}"
            )
            continue

        status = "ok" if return_code == 0 else f"failed ({return_code})"
        print(f"Finished {job.name} attempt={attempt}/{max_attempts}: {status}")
        finished.append((job, return_code, attempt))
    return still_active


def scan_option(args, scan_cfg, name, default):
    value = getattr(args, name)
    return value if value is not None else scan_cfg.get(name, default)


def run_jobs(jobs, args, scan_cfg):
    gpus = parse_gpus(args.gpus if args.gpus is not None else scan_cfg.get("gpus"))
    max_parallel = max(1, int(scan_option(args, scan_cfg, "max_parallel", 3)))
    max_retries = max(0, int(scan_option(args, scan_cfg, "max_retries", 1)))
    max_attempts = max_retries + 1

    print(
        f"Launching {len(jobs)} jobs with max_parallel={max_parallel}, "
        f"max_retries={max_retries}, gpus={gpus}"
    )

    pending = deque((job, 1) for job in jobs)
    active = []
    finished = []
    next_gpu = 0

    previous_sigterm = signal.signal(signal.SIGTERM, request_termination)
    try:
        while pending or active:
            next_gpu = launch_pending(
                pending,
                active,
                args,
                gpus,
                next_gpu,
                max_parallel,
                max_attempts,
            )
            if active:
                time.sleep(10)
                active = poll_active(active, pending, finished, max_attempts)
    except (KeyboardInterrupt, TerminationRequested):
        print("Interrupted; terminating active jobs...")
        stop_active(active)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)

    failures = [(job, code, attempt) for job, code, attempt in finished if code != 0]
    if failures:
        print("Failed jobs:")
        for job, code, attempt in failures:
            print(
                f"  {job.name} attempts={attempt}/{max_attempts} "
                f"return_code={code} log={job.log}"
            )
        return 1
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Concurrent YAML parameter scan launcher")
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Base YAML config containing a scan section",
    )
    parser.add_argument("--scan-root", default="scans", help="Directory for generated configs, logs, and runs")
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Use scan-root directly instead of scan-root/<timestamp>",
    )
    parser.add_argument("--max-parallel", type=int, default=None, help="Maximum concurrent training jobs")
    parser.add_argument("--max-retries", type=int, default=None, help="Retries per failed training job")
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids, or 'cpu'. One id can be reused by many jobs",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Apply dotted.key=value to every generated config",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate configs and manifest without launching jobs")
    parser.add_argument("--wandb-project", default="scan-pcres", help="Weights & Biases project name for scan runs")
    parser.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        default=True,
        help="Disable W&B logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    base_config = load_config(config_path)
    scan_root, jobs = build_jobs(base_config, config_path, args)

    print(f"Scan workspace: {scan_root}")
    print(f"Generated {len(jobs)} configs")
    if args.dry_run:
        for job in jobs:
            print(f"  {job.name}: {job.parameters}")
        return 0

    return run_jobs(jobs, args, base_config.get("scan", {}))


if __name__ == "__main__":
    raise SystemExit(main())
