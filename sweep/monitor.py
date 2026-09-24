"""Watch the sweep driver and resume it if it dies before ``selection.json`` exists.

    python -m sweep.monitor --interval 60

Each poll appends a JSON line to ``monitor.log`` and rewrites ``monitor.json``.
"""

import argparse
import json
import os
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "sweep" / "run_sweep.sh"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "sweep" / "outputs"

# A driver that dies this fast has a fault that a resume will hit again.
FAST_FAILURE_SECONDS = 120.0


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def driver_pid(output_dir):
    """The recorded driver pid, or ``None`` when no pid file has been written."""
    path = Path(output_dir) / "sweep.pid"
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def is_alive(pid):
    """Whether the pid is live, checked via /proc so the driver is never signalled."""
    return pid is not None and Path(f"/proc/{pid}").exists()


def trial_counts(output_dir):
    """Trial states from the study database, without taking a write lock on it."""
    path = Path(output_dir) / "study.db"
    if not path.exists():
        return {}
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            rows = connection.execute("SELECT state, COUNT(*) FROM trials GROUP BY state")
            return {str(state): int(count) for state, count in rows}
    except sqlite3.Error:
        # A poll that lands mid-write retries on the next one.
        return {}


def last_activity_seconds(output_dir, now=None):
    """Seconds since the last write to a metrics CSV or the sweep log, or ``None``."""
    output_dir = Path(output_dir)
    written = [path.stat().st_mtime for path in output_dir.rglob("metrics.csv")]
    log = output_dir / "sweep.log"
    if log.exists():
        written.append(log.stat().st_mtime)
    if not written:
        return None
    return max(0.0, (time.time() if now is None else now) - max(written))


def snapshot(output_dir, *, stall_seconds, restarts):
    """One poll: what the driver is, what the study holds, and what looks wrong."""
    output_dir = Path(output_dir)
    pid = driver_pid(output_dir)
    alive = is_alive(pid)
    idle = last_activity_seconds(output_dir)
    record = {
        "checked_at_utc": _utc_now(),
        "pid": pid,
        "driver_alive": alive,
        "trial_counts": trial_counts(output_dir),
        "seconds_since_last_write": None if idle is None else round(idle, 1),
        "restarts": restarts,
        "finished": (output_dir / "selection.json").exists(),
    }
    if alive and idle is not None and idle > stall_seconds:
        record["alert"] = (
            f"driver {pid} has written nothing for {idle / 60.0:.0f} min; "
            "check sweep.log before assuming it is still training"
        )
    return record


def write_record(output_dir, record):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "monitor.log", "a") as handle:
        handle.write(json.dumps(record) + "\n")
    path = output_dir / "monitor.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(path)


def launch_driver(output_dir, args=()):
    """Restart the driver through the launcher, with its pid and log in ``output_dir``."""
    env = dict(os.environ, SWEEP_OUTPUT_DIR=str(Path(output_dir).resolve()))
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=1800,
    )


def watch(
    output_dir=DEFAULT_OUTPUT_DIR,
    *,
    interval=60.0,
    max_restarts=5,
    stall_seconds=3600.0,
    launcher_args=(),
    launch=launch_driver,
    sleep=time.sleep,
    now=time.monotonic,
):
    """Poll until the workflow finishes, restarting a dead driver; return why it stopped.

    Gives up after ``max_restarts``, or after two restarts in a row die within
    ``FAST_FAILURE_SECONDS``.
    """
    output_dir = Path(output_dir)
    restarts = 0
    fast_failures = 0
    started_at = now()
    while True:
        record = snapshot(output_dir, stall_seconds=stall_seconds, restarts=restarts)
        if record["finished"]:
            record["status"] = "finished"
            write_record(output_dir, record)
            return "finished"

        if not record["driver_alive"]:
            if now() - started_at < FAST_FAILURE_SECONDS and restarts:
                fast_failures += 1
            else:
                fast_failures = 0
            if restarts >= max_restarts:
                record["status"] = "gave up"
                record["alert"] = (
                    f"driver is gone and {restarts} restarts have been used; "
                    "read sweep.log and restart by hand once the cause is fixed"
                )
                write_record(output_dir, record)
                return "restart limit reached"
            if fast_failures >= 2:
                record["status"] = "gave up"
                record["alert"] = (
                    "two restarts died within "
                    f"{FAST_FAILURE_SECONDS:.0f} s; a resume cannot fix this. "
                    "Read the tracebacks in sweep.log"
                )
                write_record(output_dir, record)
                return "restarts keep failing"

            result = launch(output_dir, launcher_args)
            restarts += 1
            started_at = now()
            record["restarts"] = restarts
            record["status"] = "restarted" if result.returncode == 0 else "restart failed"
            record["restart_output"] = (result.stdout or "") + (result.stderr or "")
            write_record(output_dir, record)
        else:
            record["status"] = "watching"
            write_record(output_dir, record)
        sleep(interval)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--max-restarts", type=int, default=5)
    parser.add_argument(
        "--stall-minutes",
        type=float,
        default=60.0,
        help="how long a live driver may write nothing before the log says so",
    )
    parser.add_argument(
        "--launcher-arg",
        action="append",
        default=[],
        metavar="ARG",
        help=(
            "repeatable argument passed to run_sweep.sh on a restart, so the same study "
            "resumes; write it attached, as --launcher-arg=--epochs"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    reason = watch(
        args.output_dir,
        interval=args.interval,
        max_restarts=args.max_restarts,
        stall_seconds=args.stall_minutes * 60.0,
        launcher_args=tuple(args.launcher_arg),
    )
    print(f"monitor stopped: {reason}")
    return reason


if __name__ == "__main__":
    main()
