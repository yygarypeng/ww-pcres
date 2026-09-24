import itertools
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from sweep import monitor


def _launcher(calls, *, returncode=0, on_call=None):
    def launch(output_dir, args=()):
        calls.append((Path(output_dir), tuple(args)))
        if on_call is not None:
            on_call(Path(output_dir))
        return SimpleNamespace(returncode=returncode, stdout="started", stderr="")

    return launch


def _bounded_sleep(limit=8):
    """A sleep that fails the test rather than letting a broken loop run forever."""
    calls = []

    def sleep(_seconds):
        calls.append(_seconds)
        if len(calls) > limit:
            pytest.fail(f"watch() polled {len(calls)} times without reaching a decision")

    return sleep


def _records(output_dir):
    lines = (Path(output_dir) / "monitor.log").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_finished_study_is_left_alone(tmp_path):
    (tmp_path / "selection.json").write_text("{}")
    calls = []

    reason = monitor.watch(tmp_path, launch=_launcher(calls), sleep=_bounded_sleep())

    assert reason == "finished"
    assert calls == []
    assert json.loads((tmp_path / "monitor.json").read_text())["status"] == "finished"


def test_a_dead_driver_is_restarted_and_the_resume_finishes(tmp_path):
    (tmp_path / "sweep.pid").write_text("99999999")
    calls = []
    launch = _launcher(
        calls, on_call=lambda directory: (directory / "selection.json").write_text("{}")
    )

    reason = monitor.watch(
        tmp_path, launch=launch, sleep=_bounded_sleep(), launcher_args=("--epochs", "60")
    )

    assert reason == "finished"
    assert calls == [(tmp_path, ("--epochs", "60"))]
    statuses = [record["status"] for record in _records(tmp_path)]
    assert statuses == ["restarted", "finished"]


def test_a_live_driver_is_never_restarted(tmp_path, monkeypatch):
    (tmp_path / "sweep.pid").write_text("4242")
    monkeypatch.setattr(monitor, "is_alive", lambda pid: pid == 4242)
    calls = []
    sleep = _bounded_sleep(limit=2)

    with pytest.raises(pytest.fail.Exception):
        monitor.watch(tmp_path, launch=_launcher(calls), sleep=sleep)

    assert calls == []
    assert {record["status"] for record in _records(tmp_path)} == {"watching"}


def test_restarts_stop_at_the_cap(tmp_path):
    (tmp_path / "sweep.pid").write_text("99999999")
    calls = []
    clock = itertools.count(0.0, 1000.0)  # every restart is a slow one

    reason = monitor.watch(
        tmp_path,
        max_restarts=2,
        launch=_launcher(calls),
        sleep=_bounded_sleep(),
        now=lambda: next(clock),
    )

    assert reason == "restart limit reached"
    assert len(calls) == 2
    assert "restarts have been used" in json.loads((tmp_path / "monitor.json").read_text())["alert"]


def test_a_restart_that_dies_immediately_is_not_retried_forever(tmp_path):
    (tmp_path / "sweep.pid").write_text("99999999")
    calls = []

    reason = monitor.watch(
        tmp_path,
        max_restarts=10,
        launch=_launcher(calls),
        sleep=_bounded_sleep(),
        now=lambda: 0.0,
    )

    assert reason == "restarts keep failing"
    assert len(calls) == 2, "a resume that dies at once should not be repeated ten times"


def test_a_silent_live_driver_is_flagged(tmp_path):
    (tmp_path / "sweep.log").write_text("training")
    record = monitor.snapshot(tmp_path, stall_seconds=0.0, restarts=0)
    assert "alert" not in record, "a dead driver is reported as dead, not as stalled"

    (tmp_path / "sweep.pid").write_text("1")
    record = monitor.snapshot(tmp_path, stall_seconds=0.0, restarts=0)
    assert record["driver_alive"] is True
    assert "has written nothing" in record["alert"]


def test_trial_counts_read_the_study_database(tmp_path):
    database = tmp_path / "study.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE trials (trial_id INTEGER PRIMARY KEY, state TEXT)")
        connection.executemany(
            "INSERT INTO trials (state) VALUES (?)",
            [("COMPLETE",), ("COMPLETE",), ("PRUNED",), ("FAIL",)],
        )

    assert monitor.trial_counts(tmp_path) == {"COMPLETE": 2, "PRUNED": 1, "FAIL": 1}
    assert monitor.trial_counts(tmp_path / "missing") == {}


def test_launcher_arguments_survive_a_restart():
    """A study on a non-default space must come back as the same study."""
    args = monitor.parse_args(
        ["--output-dir", "sweep/outputs_v2", "--launcher-arg=--space", "--launcher-arg=x.yaml"]
    )
    assert args.launcher_arg == ["--space", "x.yaml"]


def test_the_launcher_is_told_which_output_directory_to_use(tmp_path, monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen.update(command=command, env=kwargs["env"])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(monitor.subprocess, "run", fake_run)
    monitor.launch_driver(tmp_path, ("--study-name", "v2"))

    assert seen["env"]["SWEEP_OUTPUT_DIR"] == str(tmp_path.resolve())
    assert seen["command"][-2:] == ["--study-name", "v2"]
