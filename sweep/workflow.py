"""Resumable constrained sweep workflow."""

import copy
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import optuna
import pytorch_lightning as L
import yaml

from sweep import resources, runner, selection, space

DEFAULT_SEEDS = (2330, 2331, 2332)
DEFAULT_CONFIRM_FOLDS = (1, 2)


@dataclass
class WorkflowSettings:
    base_config: dict
    sweep_space: dict
    output_dir: Path
    storage: str
    study_name: str = "constrained-sweep"
    sampler_seed: int = 2330
    seeds: tuple = DEFAULT_SEEDS
    search_fold: int = 0
    confirm_folds: tuple = DEFAULT_CONFIRM_FOLDS
    epochs: int = runner.DEFAULT_EPOCHS
    num_workers: int | None = None
    min_trials: int = 32
    patience: int = 20
    no_feasible_limit: int = 80
    top_candidates: int = 3
    threshold_sigma: float = 2.0
    startup_trials: int = 5
    prune_warmup: int = 60
    full_epochs: bool = False
    budget_hours: float | None = None
    budget_margin: float = 1.10
    failure_limit: int = 3
    run_trial: object = runner.run_trial


class OptunaPruning(L.Callback):
    """Report the epoch composite to Optuna and prune once past the warmup."""

    def __init__(self, trial, fidelity, warmup_epochs=60):
        super().__init__()
        self.trial = trial
        self.fidelity = fidelity
        self.warmup_epochs = int(warmup_epochs)

    def on_validation_epoch_end(self, trainer, _pl_module):
        score = self.fidelity.last_composite
        if trainer.sanity_checking or score is None or trainer.current_epoch < self.warmup_epochs:
            return
        self.trial.report(score, trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"pruned at epoch {trainer.current_epoch}")


def config_fingerprint(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _atomic_yaml(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False))
    temporary.replace(path)


def _read_report(path, fingerprint):
    """A finished report for ``fingerprint``, or ``None`` when it was never stamped."""
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot resume malformed report {path}: {error}") from error
    recorded = report.get("config_fingerprint")
    if recorded is None:
        return None
    if recorded != fingerprint:
        raise RuntimeError(f"report {path} has a different base-config fingerprint")
    return report


def _seeded_config(base_config, seed):
    config = copy.deepcopy(base_config)
    config["parameters"]["seed"] = int(seed)
    return config


def _run_or_load_report(settings, config, fold, seed, directory, fingerprint, thresholds=None):
    path = Path(directory) / f"seed_{seed}" / "metrics.json"
    if path.exists():
        report = _read_report(path, fingerprint)
        if report is not None:
            return report
    trial_dir = path.parent
    report = settings.run_trial(
        _seeded_config(config, seed),
        trial_dir,
        fold=fold,
        epochs=settings.epochs,
        thresholds=thresholds,
        num_workers=settings.num_workers,
        output_root=settings.output_dir,
    )
    report["config_fingerprint"] = fingerprint
    report["fold"] = int(fold)
    report["seed"] = int(seed)
    _atomic_json(path, report)
    return report


def load_or_run_baselines(settings):
    """Load finished search-fold baselines and run only the missing seeds."""
    fingerprint = config_fingerprint(settings.base_config)
    directory = Path(settings.output_dir) / "baseline" / f"fold_{settings.search_fold}"
    reports = [
        _run_or_load_report(
            settings,
            settings.base_config,
            settings.search_fold,
            seed,
            directory,
            fingerprint,
        )
        for seed in settings.seeds
    ]
    if len(reports) < 3:
        raise RuntimeError("three complete baseline reports are required before optimization")
    return reports


def create_constrained_study(settings, thresholds):
    """Create or resume the study; refuse to resume it against different thresholds."""
    sampler = optuna.samplers.TPESampler(
        seed=settings.sampler_seed,
        multivariate=True,
        n_startup_trials=settings.startup_trials,
    )
    study = optuna.create_study(
        study_name=settings.study_name,
        storage=settings.storage,
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=settings.startup_trials,
            n_warmup_steps=settings.prune_warmup,
        ),
        load_if_exists=True,
    )
    recorded = study.user_attrs.get("thresholds")
    if recorded is not None and not _thresholds_match(recorded, thresholds):
        raise RuntimeError(
            f"study {settings.study_name!r} was built against different feasibility "
            "thresholds, and its recorded trial constraints would not mean the same "
            "thing as new ones. Resume with the thresholds it was created with, or "
            "start a separate --study-name."
        )
    study.set_user_attr("thresholds", thresholds)
    return study


def _flatten_thresholds(thresholds):
    flat = {}
    for name, value in thresholds.items():
        if isinstance(value, dict):
            flat.update({f"{name}.{key}": float(item) for key, item in value.items()})
        else:
            flat[name] = float(value)
    return flat


def _thresholds_match(recorded, thresholds):
    left, right = _flatten_thresholds(recorded), _flatten_thresholds(thresholds)
    if left.keys() != right.keys():
        return False
    return all(math.isclose(left[name], right[name], rel_tol=1e-9, abs_tol=1e-12) for name in left)


def _is_feasible_trial(trial):
    named = getattr(trial, "constraints", None)
    constraints = list(named.values()) if named else trial.user_attrs.get("constraints")
    return constraints is not None and all(value <= 0.0 for value in constraints)


class ConvergenceStopper:
    def __init__(self, min_trials=32, patience=20, no_feasible_limit=80):
        self.min_trials = int(min_trials)
        self.patience = int(patience)
        self.no_feasible_limit = int(no_feasible_limit)

    def __call__(self, study, _trial):
        if self.converged(study):
            study.stop()

    def converged(self, study):
        """No feasible gain for ``patience`` trials, or no feasible trial at all."""
        if any(trial.state == optuna.trial.TrialState.WAITING for trial in study.trials):
            return False
        completed = [
            trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        feasible = [trial for trial in completed if _is_feasible_trial(trial)]
        if not feasible:
            return len(completed) >= self.no_feasible_limit
        if len(completed) < self.min_trials:
            return False

        best = math.inf
        last_improvement = -1
        for index, trial in enumerate(completed):
            if _is_feasible_trial(trial) and trial.value < best:
                best = trial.value
                last_improvement = index
        return len(completed) - 1 - last_improvement >= self.patience


class RepeatedFailureStopper:
    """Stop the search after ``limit`` consecutive failed trials."""

    def __init__(self, limit=3):
        self.limit = int(limit)
        self.streak = 0

    def __call__(self, study, trial):
        if trial.state == optuna.trial.TrialState.FAIL:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= self.limit:
            print(
                f"Stopping the search: {self.streak} trials failed in a row. "
                "Their tracebacks are above, in this log.",
                flush=True,
            )
            study.stop()


def _finite_constraints(violations):
    return [float(value) if math.isfinite(value) else 1.0e6 for value in violations.values()]


def enqueue_unfinished_trials(study):
    """Retry each failed/pruned parameter set once, preserving completed work."""
    trials = study.get_trials()
    covered = {
        json.dumps(trial.params or trial.system_attrs.get("fixed_params", {}), sort_keys=True)
        for trial in trials
        if trial.state.name in ("COMPLETE", "RUNNING", "WAITING")
    }
    for trial in trials:
        key = json.dumps(trial.params, sort_keys=True)
        if trial.state.name in ("FAIL", "PRUNED") and trial.params and key not in covered:
            study.enqueue_trial(trial.params, user_attrs={"retry_of": trial.number})
            covered.add(key)


def _objective(settings, thresholds):
    def objective(trial):
        chosen = space.suggest(settings.sweep_space, trial)
        config = space.apply(settings.base_config, chosen)
        trial_dir = Path(settings.output_dir) / "trials" / f"trial_{trial.number:04d}"
        report = settings.run_trial(
            config,
            trial_dir,
            fold=settings.search_fold,
            epochs=settings.epochs,
            thresholds=thresholds,
            num_workers=settings.num_workers,
            full_epochs=settings.full_epochs,
            output_root=settings.output_dir,
            callback_factory=lambda fidelity: (
                []
                if settings.full_epochs
                else [OptunaPruning(trial, fidelity, warmup_epochs=settings.prune_warmup)]
            ),
        )
        violations = selection.constraint_violations(report, thresholds)
        score = selection.selection_score(report)
        finite_constraints = _finite_constraints(violations)
        for name, value in zip(violations, finite_constraints):
            trial.set_constraint(name, value)
        trial.set_user_attr("constraints", finite_constraints)
        trial.set_user_attr("violations", violations)
        trial.set_user_attr("objectives", report["objectives"])
        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("config", config)
        return score

    return objective


def _mean(values):
    return sum(values) / len(values) if values else float("inf")


def _mean_selection_score(reports):
    return float(_mean([selection.selection_score(report) for report in reports]))


def confirmation_run_count(settings):
    """Runs confirmation will cost: paired baselines plus every candidate, on every fold."""
    return len(settings.confirm_folds) * len(settings.seeds) * (1 + settings.top_candidates)


def search_budget_seconds(settings, baseline_reports, elapsed_seconds):
    """Search seconds left in this launch after reserving confirmation; ``None`` if unbounded."""
    if not settings.budget_hours or float(settings.budget_hours) <= 0.0:
        return None
    measured = [
        float(report["wall_seconds"]) for report in baseline_reports if report.get("wall_seconds")
    ]
    if not measured:
        return None
    runs = confirmation_run_count(settings)
    reserved = runs * _mean(measured) * float(settings.budget_margin)
    remaining = float(settings.budget_hours) * 3600.0 - elapsed_seconds - reserved
    if remaining <= 0.0:
        raise RuntimeError(
            f"a {float(settings.budget_hours):g} h budget cannot cover this workflow: baselines "
            f"took {elapsed_seconds / 3600.0:.1f} h and {runs} confirmation runs need about "
            f"{reserved / 3600.0:.1f} h, leaving nothing for the search. Raise --budget-hours, or "
            "lower --epochs, --top-candidates, or --confirm-folds. Finished runs are kept and "
            "will be reused on the next launch."
        )
    return remaining


def _confirmation_reports(settings, config, fold, label, thresholds=None):
    fingerprint = config_fingerprint(config)
    directory = Path(settings.output_dir) / "confirmation" / label / f"fold_{fold}"
    return [
        _run_or_load_report(
            settings, config, fold, seed, directory, fingerprint, thresholds=thresholds
        )
        for seed in settings.seeds
    ]


def confirm_candidates(settings, candidates):
    """Confirm each candidate against untuned baselines paired on the same folds and seeds."""
    configs = {
        candidate["trial_number"]: copy.deepcopy(candidate["config"]) for candidate in candidates
    }
    fold_results = {number: [] for number in configs}
    baselines = []

    # Fold-major, because runner.fold_inputs caches one fold at a time.
    for fold in settings.confirm_folds:
        baseline_reports = _confirmation_reports(settings, settings.base_config, fold, "baseline")
        thresholds = selection.derive_thresholds(baseline_reports, sigma=settings.threshold_sigma)
        baseline_score = _mean_selection_score(baseline_reports)
        baselines.extend(baseline_reports)
        for number, config in configs.items():
            reports = _confirmation_reports(
                settings, config, fold, f"trial_{number:04d}", thresholds=thresholds
            )
            fold_results[number].append(
                {
                    "fold": fold,
                    "feasible": all(
                        selection.is_feasible(report, thresholds) for report in reports
                    ),
                    "score": _mean_selection_score(reports),
                    "baseline_score": baseline_score,
                    "reports": reports,
                    "thresholds": thresholds,
                }
            )

    baseline_score = _mean_selection_score(baselines)
    results = []
    for number, config in configs.items():
        folds = fold_results[number]
        all_reports = [report for result in folds for report in result["reports"]]
        results.append(
            {
                "trial_number": number,
                "config": config,
                "score": _mean_selection_score(all_reports),
                "baseline_score": baseline_score,
                "feasible": all(result["feasible"] for result in folds),
                "folds": folds,
            }
        )
    return results


def select_configuration(base_config, confirmations):
    """The best feasible candidate that beats its paired baselines, else the base config."""
    eligible = [
        result
        for result in confirmations
        if result.get("feasible") and float(result["score"]) < float(result["baseline_score"])
    ]
    if not eligible:
        return "untuned", copy.deepcopy(base_config)
    winner = min(eligible, key=lambda result: (result["score"], result["trial_number"]))
    selected = copy.deepcopy(winner["config"])
    selected.pop("sweep", None)
    return "tuned", selected


def _candidate_records(study, count):
    feasible = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and _is_feasible_trial(trial)
    ]
    feasible.sort(key=lambda trial: (trial.value, trial.number))
    return [
        {
            "trial_number": trial.number,
            "score": trial.value,
            "config": trial.user_attrs["config"],
        }
        for trial in feasible[:count]
    ]


def run_workflow(settings):
    """Run baseline calibration, constrained search, confirmation, and selection."""
    started = time.monotonic()
    settings.output_dir = Path(settings.output_dir)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    if settings.num_workers is None:
        splits, _ = runner.fold_inputs(settings.base_config, settings.search_fold)
        settings.num_workers = resources.benchmark_dataloader_workers(
            settings.base_config,
            splits,
            resources.worker_candidates(len(os.sched_getaffinity(0))),
            output_path=settings.output_dir / "resources.json",
        )

    baseline_reports = load_or_run_baselines(settings)
    thresholds = selection.derive_thresholds(baseline_reports, sigma=settings.threshold_sigma)
    _atomic_json(settings.output_dir / "thresholds.json", thresholds)

    study = create_constrained_study(settings, thresholds)
    settings.full_epochs = settings.full_epochs or study.user_attrs.get("full_epochs", False)
    if settings.full_epochs:
        study.set_user_attr("full_epochs", True)
        enqueue_unfinished_trials(study)
        print(
            f"Full-epoch mode: queued unfinished configurations; new trials run {settings.epochs} epochs."
        )
    anchor = space.baseline_point(settings.sweep_space, settings.base_config)
    if anchor:
        study.enqueue_trial(anchor, skip_if_exists=True)
    else:
        print(
            "No anchor trial enqueued: the base config sits outside the search space, so "
            "the study has no point directly comparable to the untuned baselines."
        )
    stopper = ConvergenceStopper(settings.min_trials, settings.patience, settings.no_feasible_limit)
    failure_stopper = RepeatedFailureStopper(settings.failure_limit)
    search_seconds = search_budget_seconds(settings, baseline_reports, time.monotonic() - started)
    if search_seconds is not None:
        print(
            f"Budget {float(settings.budget_hours):g} h: {confirmation_run_count(settings)} "
            f"confirmation runs reserved, {search_seconds / 3600.0:.1f} h left for the search. "
            "Optuna finishes the trial that is running when the clock runs out."
        )
    # Optuna forgets a previous stop on resume, so a converged study must not search again.
    if stopper.converged(study):
        print("The search has already converged; going straight to confirmation.")
    else:
        study.optimize(
            _objective(settings, thresholds),
            n_trials=None,
            timeout=search_seconds,
            callbacks=[stopper, failure_stopper],
            # A crashed trial is recorded as FAIL; KeyboardInterrupt still stops the driver.
            catch=(Exception,),
            # Collect after Optuna releases pruning/error tracebacks that retain the trainer.
            gc_after_trial=True,
        )
    candidates = _candidate_records(study, settings.top_candidates)
    confirmations = confirm_candidates(settings, candidates)
    status, selected = select_configuration(settings.base_config, confirmations)

    selected_path = settings.output_dir / "selected_config.yaml"
    _atomic_yaml(selected_path, selected)
    _atomic_json(
        settings.output_dir / "selection.json",
        {
            "status": status,
            "base_config_fingerprint": config_fingerprint(settings.base_config),
            "search_baseline_score": _mean_selection_score(baseline_reports),
            "confirmation_baseline_score": (
                confirmations[0]["baseline_score"] if confirmations else None
            ),
            "confirmations": confirmations,
        },
    )
    return selected_path
