import json
import sqlite3

import pytest

from sweep import progress

SPACE = {
    "axes": {
        "parameters.learning_rate": {"type": "float", "low": 1.0e-4, "high": 1.0e-2, "log": True},
        "parameters.loss_weights.dmet": {"type": "float", "low": 0.0, "high": 1.0},
    }
}


def _study(tmp_path, rows):
    """A minimal optuna-shaped database: the reader must not need optuna itself."""
    database = tmp_path / "study.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE studies (study_id INTEGER PRIMARY KEY, study_name TEXT)")
        connection.execute(
            "CREATE TABLE trials (trial_id INTEGER PRIMARY KEY, study_id INTEGER,"
            " number INTEGER, state TEXT)"
        )
        connection.execute("CREATE TABLE trial_values (trial_id INTEGER, value REAL)")
        connection.execute(
            "CREATE TABLE trial_params (trial_id INTEGER, param_name TEXT, param_value REAL)"
        )
        connection.execute(
            "CREATE TABLE trial_user_attributes (trial_id INTEGER, key TEXT, value_json TEXT)"
        )
        connection.execute("INSERT INTO studies VALUES (1, 's')")
        for index, row in enumerate(rows, start=1):
            connection.execute(
                "INSERT INTO trials VALUES (?, 1, ?, ?)", (index, index - 1, row["state"])
            )
            if row.get("value") is not None:
                connection.execute("INSERT INTO trial_values VALUES (?, ?)", (index, row["value"]))
            for name, value in row.get("params", {}).items():
                connection.execute(
                    "INSERT INTO trial_params VALUES (?, ?, ?)", (index, name, value)
                )
            for key in ("violations", "objectives"):
                if key in row:
                    connection.execute(
                        "INSERT INTO trial_user_attributes VALUES (?, ?, ?)",
                        (index, key, json.dumps(row[key])),
                    )
    return tmp_path


def _row(number_state, value, violations, lr=1.0e-3, dmet=0.5):
    return {
        "state": number_state,
        "value": value,
        "params": {"parameters.learning_rate": lr, "parameters.loss_weights.dmet": dmet},
        "violations": violations,
        "objectives": {"bias": value / 2, "angular_1d": 0.01, "interangular_1d": 0.01},
    }


def test_reads_a_live_database_without_optuna(tmp_path):
    _study(tmp_path, [_row("COMPLETE", 0.2, {"angular_1d": -0.1}), {"state": "RUNNING"}])
    trials = progress.load_trials(tmp_path, "s")
    assert [t["state"] for t in trials] == ["COMPLETE", "RUNNING"]
    assert trials[0]["params"]["parameters.learning_rate"] == pytest.approx(1.0e-3)


def test_feasibility_needs_every_constraint_satisfied(tmp_path):
    _study(
        tmp_path,
        [
            _row("COMPLETE", 0.2, {"angular_1d": -0.1, "resolution.px_sum": -0.2}),
            _row("COMPLETE", 0.1, {"angular_1d": -0.1, "resolution.px_sum": 0.3}),
        ],
    )
    trials = progress.load_trials(tmp_path, "s")
    assert [progress.is_feasible(t) for t in trials] == [True, False]


def test_binding_constraints_are_counted(tmp_path):
    _study(
        tmp_path,
        [
            _row("COMPLETE", 0.2, {"angular_1d": 0.5, "resolution.px_sum": -0.2}),
            _row("COMPLETE", 0.3, {"angular_1d": 0.4, "resolution.px_sum": 0.1}),
        ],
    )
    counts, total = progress.binding_constraints(progress.load_trials(tmp_path, "s"))
    assert total == 2
    assert counts == {"angular_1d": 2, "resolution.px_sum": 1}


def test_a_railed_axis_is_named(tmp_path):
    rows = [
        _row("COMPLETE", 0.10, {"a": -1.0}, lr=9.9e-3),
        _row("COMPLETE", 0.11, {"a": -1.0}, lr=9.8e-3),
    ]
    _study(tmp_path, rows)
    rails = progress.rail_report(progress.load_trials(tmp_path, "s"), SPACE)
    assert rails["parameters.learning_rate"]["rail"] == "high"
    assert rails["parameters.loss_weights.dmet"]["rail"] is None


def test_report_and_figure_survive_a_study_with_nothing_feasible(tmp_path):
    _study(tmp_path, [_row("COMPLETE", 0.2, {"angular_1d": 0.5})])
    trials = progress.load_trials(tmp_path, "s")
    assert "no feasible trial yet" in progress.text_report(trials, SPACE)
    written = progress.figure(trials, SPACE, tmp_path / "figure" / "progress.png")
    assert written.exists()


def test_figure_is_skipped_before_anything_is_scored(tmp_path):
    _study(tmp_path, [{"state": "RUNNING"}])
    trials = progress.load_trials(tmp_path, "s")
    assert progress.figure(trials, SPACE, tmp_path / "figure" / "progress.png") is None


def test_a_missing_study_is_named(tmp_path):
    _study(tmp_path, [_row("COMPLETE", 0.2, {"a": -1.0})])
    with pytest.raises(ValueError, match="no study named"):
        progress.load_trials(tmp_path, "absent")
