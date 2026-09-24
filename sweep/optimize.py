"""Command-line entry point for the constrained hyperparameter sweep."""

import argparse
from pathlib import Path

import optuna

from sweep import runner, space, workflow
from train import train as base_training

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "untuned_kfold_config.yaml"
DEFAULT_SPACE = REPO_ROOT / "sweep" / "spaces" / "constrained.yaml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--space", type=Path, default=DEFAULT_SPACE)
    parser.add_argument("--output-dir", type=Path, default=runner.OUTPUT_ROOT)
    parser.add_argument("--study-name", default="constrained-sweep")
    # Defaults fit --budget-hours 20 at 36.5 s/epoch; see sweep/README.md.
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--full-epochs",
        action="store_true",
        help="retry failed/pruned trials without pruning or early stopping; persists on resume",
    )
    parser.add_argument("--search-fold", type=int, default=0)
    parser.add_argument("--confirm-folds", type=int, nargs="+", default=[1])
    parser.add_argument("--min-trials", type=int, default=12)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--no-feasible-limit", type=int, default=40)
    parser.add_argument("--top-candidates", type=int, default=2)
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=20.0,
        help="wall-clock budget for one launch; 0 removes the cap",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=2.0,
        help="how many baseline seed sigmas a bias may sit from zero and still pass",
    )
    parser.add_argument(
        "--startup-trials",
        type=int,
        default=5,
        help="trials drawn before TPE models the space and the pruner starts cutting",
    )
    parser.add_argument(
        "--prune-warmup",
        type=int,
        default=None,
        help="epochs before a trial may be pruned (default: a third of --epochs)",
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", action="store_true")
    return parser.parse_args(argv)


def default_prune_warmup(args):
    """Epochs before pruning may fire, proportional so short runs still prune."""
    if args.prune_warmup is not None:
        return int(args.prune_warmup)
    return max(5, int(args.epochs) // 3)


def _load_inputs(args):
    config = base_training.load_config(args.base_config)
    sweep_space = space.load_space(args.space)
    data_path = base_training.resolve_repo_path(config["paths"]["data_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"data file not found: {data_path}")
    return config, sweep_space


def _validate_storage(args):
    """Fail in the foreground on an unreadable or incompatible resume database."""
    db_path = args.output_dir / "study.db"
    storage = f"sqlite:///{db_path}"
    if not db_path.exists() or db_path.stat().st_size == 0:
        return storage
    try:
        names = optuna.get_all_study_names(storage)
    except Exception as error:
        raise RuntimeError(f"resume database {db_path} is unreadable: {error}") from error
    if args.study_name in names:
        try:
            study = optuna.load_study(study_name=args.study_name, storage=storage)
        except Exception as error:
            raise RuntimeError(
                f"resume database {db_path} cannot load study {args.study_name!r}: {error}"
            ) from error
        if study.direction != optuna.study.StudyDirection.MINIMIZE:
            raise RuntimeError(
                f"study {args.study_name!r} in {db_path} does not minimize "
                "the constrained score and cannot be resumed"
            )
    return storage


def _dry_run(sweep_space, seed=2330, count=3):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    for index in range(count):
        trial = study.ask()
        print(f"trial {index}: {space.suggest(sweep_space, trial)}")


def main(argv=None):
    args = parse_args(argv)
    config, sweep_space = _load_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.validate:
        _validate_storage(args)
        print("Configuration, data path, constrained search space, and resume database are valid.")
        return None
    if args.dry_run:
        _dry_run(sweep_space)
        return None
    if args.report:
        path = args.output_dir / "selection.json"
        if not path.exists():
            raise FileNotFoundError(f"selection report not found: {path}")
        print(path.read_text())
        return None

    settings = workflow.WorkflowSettings(
        base_config=config,
        sweep_space=sweep_space,
        output_dir=args.output_dir,
        storage=_validate_storage(args),
        study_name=args.study_name,
        search_fold=args.search_fold,
        confirm_folds=tuple(args.confirm_folds),
        epochs=args.epochs,
        full_epochs=args.full_epochs,
        min_trials=args.min_trials,
        patience=args.patience,
        no_feasible_limit=args.no_feasible_limit,
        top_candidates=args.top_candidates,
        threshold_sigma=args.threshold_sigma,
        startup_trials=args.startup_trials,
        prune_warmup=default_prune_warmup(args),
        budget_hours=args.budget_hours,
    )
    selected = workflow.run_workflow(settings)
    print(f"Selected configuration: {selected}")
    return selected


if __name__ == "__main__":
    main()
