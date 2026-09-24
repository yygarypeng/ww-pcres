import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
import yaml

from sweep import metrics, optimize, space, workflow


def test_full_epoch_retries_are_deduplicated_and_completed_configs_are_preserved():
    study = optuna.create_study()
    distribution = {"x": optuna.distributions.IntDistribution(1, 3)}
    for value, state in [(1, "FAIL"), (1, "PRUNED"), (2, "COMPLETE"), (2, "FAIL")]:
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": value},
                distributions=distribution,
                state=getattr(optuna.trial.TrialState, state),
                value=0.1 if state == "COMPLETE" else None,
            )
        )
    workflow.enqueue_unfinished_trials(study)
    workflow.enqueue_unfinished_trials(study)
    waiting = [trial for trial in study.trials if trial.state.name == "WAITING"]
    assert len(waiting) == 1
    assert waiting[0].system_attrs["fixed_params"] == {"x": 1}
    assert waiting[0].user_attrs["retry_of"] == 0


def test_full_epoch_search_does_not_install_a_pruning_callback(tmp_path):
    def run_trial(config, directory, **kwargs):
        assert kwargs["full_epochs"] is True
        assert kwargs["callback_factory"](object()) == []
        return _report()

    settings = workflow.WorkflowSettings(
        base_config={"parameters": {}},
        sweep_space={
            "axes": {"parameters.learning_rate": {"type": "categorical", "choices": [0.001]}}
        },
        output_dir=tmp_path,
        storage="sqlite://",
        full_epochs=True,
        run_trial=run_trial,
    )
    study = optuna.create_study()
    thresholds = {
        "angular_1d": 1.0,
        "interangular_1d": 1.0,
        "bias_gev": {name: 1.0 for name in metrics.SCORED_DIRECTIONS},
    }
    study.optimize(workflow._objective(settings, thresholds), n_trials=1)
    assert study.trials[0].state.name == "COMPLETE"


def test_convergence_does_not_stop_before_queued_retries():
    trial = SimpleNamespace(state=optuna.trial.TrialState.COMPLETE, user_attrs={})
    study = SimpleNamespace(
        trials=[trial, SimpleNamespace(state=optuna.trial.TrialState.WAITING)],
        stop=lambda: pytest.fail("stopped with a retry still waiting"),
    )
    workflow.ConvergenceStopper(no_feasible_limit=1)(study, trial)


def _write_space(tmp_path, body):
    path = tmp_path / "space.yaml"
    path.write_text(yaml.safe_dump(body, sort_keys=False))
    return path


def _report(score=0.1, bias_gev=0.0):
    return {
        "objectives": {
            "bias": score,
            "angular_1d": score,
            "interangular_1d": score,
        },
        "bias_directions": {
            name: {"bias_gev": bias_gev, "bias_over_resolution": score, "resolution_gev": 20.0}
            for name in metrics.SCORED_DIRECTIONS
        },
    }


def _settings(tmp_path, run_trial):
    return SimpleNamespace(
        output_dir=tmp_path,
        base_config={"parameters": {"seed": 2330}},
        seeds=(2330, 2331, 2332),
        search_fold=0,
        epochs=5,
        num_workers=0,
        threshold_sigma=2.0,
        run_trial=run_trial,
    )


def test_baselines_resume_and_run_only_missing_seeds(tmp_path):
    calls = []

    def fake_run(cfg, trial_dir, **kwargs):
        calls.append((cfg["parameters"]["seed"], Path(trial_dir), kwargs["fold"]))
        return _report(score=0.1 + len(calls) * 0.01)

    settings = _settings(tmp_path, fake_run)
    first = workflow.load_or_run_baselines(settings)
    second = workflow.load_or_run_baselines(settings)
    missing = tmp_path / "baseline" / "fold_0" / "seed_2331" / "metrics.json"
    missing.unlink()
    third = workflow.load_or_run_baselines(settings)

    assert len(first) == len(second) == len(third) == 3
    assert [seed for seed, _, _ in calls] == [2330, 2331, 2332, 2331]
    assert all(fold == 0 for _, _, fold in calls)


def test_baseline_resume_names_a_malformed_report(tmp_path):
    settings = _settings(tmp_path, lambda *args, **kwargs: _report())
    path = tmp_path / "baseline" / "fold_0" / "seed_2330" / "metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text("not-json")

    with pytest.raises(RuntimeError, match=str(path)):
        workflow.load_or_run_baselines(settings)


def test_baseline_resume_reruns_a_report_that_was_never_stamped(tmp_path):
    calls = []

    def fake_run(cfg, trial_dir, **kwargs):
        calls.append(cfg["parameters"]["seed"])
        return _report()

    settings = _settings(tmp_path, fake_run)
    workflow.load_or_run_baselines(settings)
    path = tmp_path / "baseline" / "fold_0" / "seed_2331" / "metrics.json"
    unstamped = json.loads(path.read_text())
    unstamped.pop("config_fingerprint")
    path.write_text(json.dumps(unstamped))

    assert len(workflow.load_or_run_baselines(settings)) == 3
    assert calls == [2330, 2331, 2332, 2331]


def test_baseline_resume_rejects_a_different_config_fingerprint(tmp_path):
    settings = _settings(tmp_path, lambda *args, **kwargs: _report())
    workflow.load_or_run_baselines(settings)
    settings.base_config["parameters"]["new_setting"] = 4

    with pytest.raises(RuntimeError, match="fingerprint"):
        workflow.load_or_run_baselines(settings)


class _FakeStudy:
    def __init__(self, trials):
        self.trials = trials
        self.stopped = False

    def stop(self):
        self.stopped = True


def _trial(number, value, feasible):
    constraints = [-0.1] if feasible else [0.1]
    return SimpleNamespace(
        number=number,
        value=value,
        state=optuna.trial.TrialState.COMPLETE,
        user_attrs={"constraints": constraints},
    )


def test_convergence_stops_after_feasible_non_improvement_patience():
    stopper = workflow.ConvergenceStopper(min_trials=3, patience=2, no_feasible_limit=10)
    study = _FakeStudy([_trial(0, 0.5, True), _trial(1, 0.4, True), _trial(2, 0.45, True)])
    stopper(study, study.trials[-1])
    assert not study.stopped

    study.trials.append(_trial(3, 0.46, True))
    stopper(study, study.trials[-1])
    assert study.stopped


def test_convergence_stops_an_all_infeasible_study_at_its_ceiling():
    stopper = workflow.ConvergenceStopper(min_trials=2, patience=2, no_feasible_limit=3)
    study = _FakeStudy([_trial(index, 1.0, False) for index in range(3)])
    stopper(study, study.trials[-1])
    assert study.stopped


def test_constrained_study_uses_optuna_constraint_attributes(tmp_path):
    settings = SimpleNamespace(
        study_name="test-study",
        storage=f"sqlite:///{tmp_path / 'study.db'}",
        sampler_seed=2330,
        startup_trials=5,
        prune_warmup=20,
    )
    study = workflow.create_constrained_study(settings, {"angular_1d": 1.0})
    trial = study.ask()
    trial.set_constraint("angular_1d", -0.1)
    trial.set_constraint("interangular_1d", 0.0)
    trial.set_user_attr("constraints", [-0.1, 0.0])
    trial.set_user_attr("objectives", _report()["objectives"])
    study.tell(trial, 0.3)

    frozen = study.trials[0]
    assert frozen.constraints == {"angular_1d": -0.1, "interangular_1d": 0.0}
    assert frozen.user_attrs["constraints"] == [-0.1, 0.0]
    assert frozen.user_attrs["objectives"]["angular_1d"] == 0.1


def test_trial_pruner_reports_the_constrained_checkpoint_score():
    class FakeTrial:
        def __init__(self):
            self.reports = []

        def report(self, value, step):
            self.reports.append((value, step))

        def should_prune(self):
            return True

    trial = FakeTrial()
    callback = workflow.OptunaPruning(trial, SimpleNamespace(last_composite=0.42), warmup_epochs=60)
    trainer = SimpleNamespace(sanity_checking=False, current_epoch=60)

    with pytest.raises(optuna.TrialPruned):
        callback.on_validation_epoch_end(trainer, None)
    assert trial.reports == [(0.42, 60)]


def test_confirmation_scores_each_candidate_against_its_own_fold_baselines(tmp_path):
    def fake_run(cfg, trial_dir, **kwargs):
        tuned = cfg["parameters"].get("tuned", False)
        value = {1: 0.30, 2: 0.60}[kwargs["fold"]] - (0.03 if tuned else 0.0)
        return _report(score=value / 3)

    settings = _settings(tmp_path, fake_run)
    settings.confirm_folds = (1, 2)
    candidates = [{"trial_number": 7, "config": {"parameters": {"seed": 2330, "tuned": True}}}]

    (result,) = workflow.confirm_candidates(settings, candidates)

    assert [entry["fold"] for entry in result["folds"]] == [1, 2]
    assert [entry["baseline_score"] for entry in result["folds"]] == pytest.approx([0.30, 0.60])
    assert [entry["score"] for entry in result["folds"]] == pytest.approx([0.27, 0.57])
    assert result["baseline_score"] == pytest.approx(0.45)
    assert result["score"] == pytest.approx(0.42)
    assert result["feasible"]


def test_selection_requires_every_confirmation_fold_and_beats_baseline():
    base = {"parameters": {"fourvec_loss": "l1", "loss_weights": {"fourvec_bias": 0.0}}}
    winner = deepcopy(base)
    winner["parameters"]["fourvec_loss"] = "rmse"
    winner["parameters"]["loss_weights"]["fourvec_bias"] = 5.0
    winner["sweep"] = {"legacy": True}
    confirmations = [
        {
            "trial_number": 4,
            "config": winner,
            "score": 0.3,
            "baseline_score": 0.4,
            "feasible": True,
        },
        {
            "trial_number": 5,
            "config": base,
            "score": 0.2,
            "baseline_score": 0.4,
            "feasible": False,
        },
    ]

    status, selected = workflow.select_configuration(base, confirmations)

    assert status == "tuned"
    assert selected["parameters"]["fourvec_loss"] == "rmse"
    assert selected["parameters"]["loss_weights"]["fourvec_bias"] == 5.0
    assert "sweep" not in selected


def test_selection_ignores_a_candidate_that_only_beats_another_folds_baseline():
    base = {"parameters": {"fourvec_loss": "l1"}}
    original = deepcopy(base)
    # 0.55 would clear a 0.6 fold-0 search baseline, but loses to the untuned
    # baselines trained on this candidate's own confirmation folds.
    confirmations = [
        {
            "trial_number": 4,
            "config": {"parameters": {"fourvec_loss": "rmse"}},
            "score": 0.55,
            "baseline_score": 0.50,
            "feasible": True,
        }
    ]

    status, selected = workflow.select_configuration(base, confirmations)

    assert status == "untuned"
    assert selected == original


def test_selection_falls_back_to_an_unchanged_untuned_config():
    base = {"parameters": {"fourvec_loss": "l1"}}
    original = deepcopy(base)
    status, selected = workflow.select_configuration(
        base,
        [
            {
                "trial_number": 1,
                "config": {"parameters": {}},
                "score": 0.1,
                "baseline_score": 0.6,
                "feasible": False,
            }
        ],
    )
    assert status == "untuned"
    assert selected == original
    assert base == original


def _end_to_end_settings(tmp_path):
    """Workflow settings with training stubbed out; fold 1 scores worse than fold 0."""

    def fake_run(cfg, trial_dir, **kwargs):
        value = cfg["parameters"]["learning_rate"] * 1000.0 + 0.1 * kwargs["fold"]
        report = _report(score=value / 3)
        report["wall_seconds"] = 60.0
        return report

    return workflow.WorkflowSettings(
        base_config={"parameters": {"seed": 2330, "learning_rate": 0.0005}},
        sweep_space=space.load_space(
            _write_space(
                tmp_path,
                {
                    "axes": {
                        "parameters.learning_rate": {
                            "type": "float",
                            "low": 0.0001,
                            "high": 0.001,
                            "log": True,
                        }
                    }
                },
            )
        ),
        output_dir=tmp_path,
        storage=f"sqlite:///{tmp_path / 'study.db'}",
        epochs=2,
        num_workers=0,
        seeds=(2330, 2331, 2332),
        confirm_folds=(1,),
        top_candidates=1,
        min_trials=8,
        patience=4,
        no_feasible_limit=12,
        budget_hours=16.0,
        run_trial=fake_run,
    )


def test_run_workflow_end_to_end_selects_a_winner_that_beats_its_own_folds(tmp_path):
    """A winner picked against the wrong fold's baseline would show a wrong score."""
    selected_path = workflow.run_workflow(_end_to_end_settings(tmp_path))

    selected = yaml.safe_load(selected_path.read_text())
    report = json.loads((tmp_path / "selection.json").read_text())
    assert report["status"] == "tuned"
    assert selected["parameters"]["learning_rate"] < 0.0005
    # fold 0 scores 0.5, fold 1 scores 0.6: the winner was judged against fold 1.
    assert report["search_baseline_score"] == pytest.approx(0.5)
    assert report["confirmation_baseline_score"] == pytest.approx(0.6)
    assert report["confirmations"][0]["score"] < report["confirmations"][0]["baseline_score"]


def _budget_settings(**overrides):
    settings = SimpleNamespace(
        confirm_folds=(1, 2),
        seeds=(2330, 2331, 2332),
        top_candidates=2,
        budget_hours=16.0,
        budget_margin=1.0,
    )
    for name, value in overrides.items():
        setattr(settings, name, value)
    return settings


def test_search_budget_reserves_every_confirmation_run():
    settings = _budget_settings()
    assert workflow.confirmation_run_count(settings) == 18

    remaining = workflow.search_budget_seconds(settings, [{"wall_seconds": 1465.0}] * 3, 4395.0)

    assert remaining == pytest.approx(16 * 3600 - 4395.0 - 18 * 1465.0)


def test_search_budget_refuses_a_budget_that_confirmation_would_exhaust():
    settings = _budget_settings(budget_hours=6.0)

    with pytest.raises(RuntimeError, match="cannot cover"):
        workflow.search_budget_seconds(settings, [{"wall_seconds": 1465.0}] * 3, 4395.0)


def test_search_budget_is_unlimited_when_the_cap_is_removed():
    settings = _budget_settings(budget_hours=0.0)
    assert workflow.search_budget_seconds(settings, [{"wall_seconds": 1465.0}], 0.0) is None


def test_search_budget_needs_a_measured_run_to_reserve_from():
    settings = _budget_settings()
    assert workflow.search_budget_seconds(settings, [{}], 0.0) is None


def test_prune_warmup_stays_proportional_to_the_epoch_budget():
    assert optimize.default_prune_warmup(SimpleNamespace(prune_warmup=None, epochs=40)) == 13
    assert optimize.default_prune_warmup(SimpleNamespace(prune_warmup=None, epochs=200)) == 66
    assert optimize.default_prune_warmup(SimpleNamespace(prune_warmup=None, epochs=6)) == 5
    assert optimize.default_prune_warmup(SimpleNamespace(prune_warmup=25, epochs=40)) == 25


def test_the_shipped_space_can_reach_the_shipped_base_config():
    """The study has to be able to evaluate the configuration it is trying to beat."""
    base = yaml.safe_load(optimize.DEFAULT_BASE_CONFIG.read_text())
    sweep_space = space.load_space(optimize.DEFAULT_SPACE)

    anchor = space.baseline_point(sweep_space, base)

    assert anchor, "the untuned config is unreachable inside the search space"
    assert anchor["sweep.bias_penalty"] == "off"
    assert anchor["parameters.learning_rate"] == base["parameters"]["learning_rate"]


def test_validate_and_dry_run_never_construct_a_trainer(tmp_path, monkeypatch):
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "parameters": {
                    "seed": 2330,
                    "folds": 8,
                    "fourvec_loss": "l1",
                    "loss_weights": {"angular_mmd": 20.0},
                },
                "paths": {"data_path": str(tmp_path / "data.h5")},
            }
        )
    )
    (tmp_path / "data.h5").touch()
    space_path = tmp_path / "space.yaml"
    space_path.write_text(
        yaml.safe_dump(
            {
                "sampler": "tpe",
                "axes": {
                    "parameters.loss_weights.angular_mmd": {
                        "type": "float",
                        "low": 5.0,
                        "high": 40.0,
                        "log": True,
                    }
                },
            }
        )
    )

    def fail_trainer(*args, **kwargs):
        raise AssertionError("Trainer constructed")

    monkeypatch.setattr("sweep.runner.Trainer", fail_trainer)
    common = ["--base-config", str(base_path), "--space", str(space_path)]
    assert optimize.main(["--validate", *common]) is None
    assert optimize.main(["--dry-run", *common]) is None


def _validate_common(tmp_path):
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "parameters": {"seed": 2330, "folds": 8},
                "paths": {"data_path": str(tmp_path / "data.h5")},
            }
        )
    )
    (tmp_path / "data.h5").touch()
    space_path = tmp_path / "space.yaml"
    space_path.write_text(
        yaml.safe_dump({"axes": {"parameters.seed": {"type": "int", "low": 1, "high": 2}}})
    )
    return [
        "--base-config",
        str(base_path),
        "--space",
        str(space_path),
        "--output-dir",
        str(tmp_path / "outputs"),
    ]


def test_validate_rejects_a_corrupt_resume_database(tmp_path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "study.db").write_text("not-sqlite")

    with pytest.raises(RuntimeError, match="study.db"):
        optimize.main(["--validate", *_validate_common(tmp_path)])


def test_validate_rejects_a_study_with_the_wrong_direction(tmp_path):
    storage = f"sqlite:///{tmp_path / 'outputs' / 'study.db'}"
    (tmp_path / "outputs").mkdir()
    optuna.create_study(study_name="constrained-sweep", storage=storage, direction="maximize")

    with pytest.raises(RuntimeError, match="constrained-sweep"):
        optimize.main(["--validate", *_validate_common(tmp_path)])


def test_validate_passes_without_a_resume_database(tmp_path):
    assert optimize.main(["--validate", *_validate_common(tmp_path)]) is None


def test_a_finished_trial_resets_the_failure_streak():
    stopped = []
    study = SimpleNamespace(stop=lambda: stopped.append(True))
    stopper = workflow.RepeatedFailureStopper(limit=2)
    failed = SimpleNamespace(state=optuna.trial.TrialState.FAIL)
    finished = SimpleNamespace(state=optuna.trial.TrialState.COMPLETE)

    stopper(study, failed)
    stopper(study, finished)
    stopper(study, failed)
    assert stopped == [], "two failures either side of a finished trial are not a streak"

    stopper(study, failed)
    assert stopped == [True]


def _failing_search(tmp_path, fake_run, **overrides):
    settings = workflow.WorkflowSettings(
        base_config={"parameters": {"seed": 2330, "learning_rate": 0.0005}},
        sweep_space=space.load_space(
            _write_space(
                tmp_path,
                {
                    "axes": {
                        "parameters.learning_rate": {
                            "type": "float",
                            "low": 0.0001,
                            "high": 0.001,
                            "log": True,
                        }
                    }
                },
            )
        ),
        output_dir=tmp_path,
        storage=f"sqlite:///{tmp_path / 'study.db'}",
        epochs=2,
        num_workers=0,
        confirm_folds=(1,),
        top_candidates=1,
        min_trials=6,
        patience=3,
        no_feasible_limit=10,
        budget_hours=16.0,
        run_trial=fake_run,
        **overrides,
    )
    workflow.run_workflow(settings)
    study = optuna.load_study(study_name=settings.study_name, storage=settings.storage)
    return json.loads((tmp_path / "selection.json").read_text()), study


def _stub_report(cfg, fold=0):
    report = _report(score=(cfg["parameters"]["learning_rate"] * 1000.0 + 0.1 * fold) / 3)
    report["wall_seconds"] = 60.0
    return report


def test_a_failed_search_trial_leaves_the_rest_of_the_study_running(tmp_path):
    """A dead trial is one FAIL in the study, not the end of a run with hours in it."""
    failed = []

    def fake_run(cfg, trial_dir, **kwargs):
        if Path(trial_dir).parent.name == "trials" and not failed:
            failed.append(Path(trial_dir).name)
            raise RuntimeError("DataLoader worker (pid 4242) is killed by signal: Aborted.")
        return _stub_report(cfg, kwargs["fold"])

    report, study = _failing_search(tmp_path, fake_run)

    states = [trial.state.name for trial in study.trials]
    assert failed == ["trial_0000"]
    assert states.count("FAIL") == 1
    assert states.count("COMPLETE") >= 6, "the search stopped at the first failure"
    assert report["status"] == "tuned"


def test_a_search_that_only_fails_stops_instead_of_spending_the_budget(tmp_path):
    """Tolerating failures still needs a floor, or a broken space would run all night."""

    def fake_run(cfg, trial_dir, **kwargs):
        if Path(trial_dir).parent.name == "trials":
            raise RuntimeError("every configuration fails")
        return _stub_report(cfg, kwargs["fold"])

    report, study = _failing_search(tmp_path, fake_run, failure_limit=3)

    assert [trial.state.name for trial in study.trials] == ["FAIL"] * 3
    assert report["status"] == "untuned", "the untuned config stays selected when nothing wins"


def test_resuming_a_converged_study_runs_no_new_search_trial(tmp_path):
    settings = _end_to_end_settings(tmp_path)
    workflow.run_workflow(settings)
    study = optuna.load_study(study_name=settings.study_name, storage=settings.storage)
    trials_before = len(study.trials)

    workflow.run_workflow(_end_to_end_settings(tmp_path))

    study = optuna.load_study(study_name=settings.study_name, storage=settings.storage)
    assert len(study.trials) == trials_before
