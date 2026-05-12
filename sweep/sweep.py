import argparse
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from train.main import main as train_main

DEFAULT_SWEEP_ROOT = "wandb_sweep_runs"


def resolve_path(raw_path):
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def set_nested(config, dotted_key, value):
    target = config
    parts = str(dotted_key).split(".")
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def flatten_sweep_config(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value} if prefix else {}

    items = {}
    for key, nested_value in value.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(nested_value, dict):
            items.update(flatten_sweep_config(nested_value, full_key))
        else:
            items[full_key] = nested_value
    return items


def sweep_values(wandb_config):
    return {
        key: value
        for key, value in flatten_sweep_config(dict(wandb_config)).items()
        if key.startswith("parameters.")
    }


def write_sweep_config(base_config, base_config_path, run, sweep_root, values):
    config = copy.deepcopy(base_config)
    config.pop("scan", None)

    for key, value in values.items():
        set_nested(config, key, value)

    run_name = run.name or run.id
    sweep_root_path = resolve_path(sweep_root)
    run_dir = sweep_root_path / "runs" / run.id
    configs_dir = sweep_root_path / "configs"
    config.setdefault("paths", {})["saved_path"] = str(run_dir)
    config["sweep_metadata"] = {
        "base_config": str(base_config_path.resolve()),
        "wandb_project": run.project,
        "wandb_run_id": run.id,
        "wandb_run_name": run_name,
        "swept_parameters": values,
    }

    configs_dir.mkdir(parents=True, exist_ok=True)
    config_path = configs_dir / f"{run.id}.yaml"
    with open(config_path, "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return config_path, run_name, run.project


def parse_args():
    parser = argparse.ArgumentParser(description="Run one W&B sweep trial using the project training code")
    parser.add_argument("--config", "-c", default="configs/config.yaml", help="Base YAML config file")
    parser.add_argument(
        "--sweep-root",
        default=DEFAULT_SWEEP_ROOT,
        help="Directory for generated per-run configs and outputs",
    )
    parser.add_argument("--watch-model", action="store_true", help="Log model gradients/parameters to W&B")
    return parser.parse_args()


def main():
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is required for sweeps. Install it with `pip install wandb`.") from exc

    args = parse_args()
    config_path = resolve_path(args.config)
    base_config = load_config(config_path)

    run = wandb.init()
    values = sweep_values(wandb.config)
    generated_config, run_name, project = write_sweep_config(
        base_config,
        config_path,
        run,
        args.sweep_root,
        values,
    )
    wandb.config.update({"generated_config": str(generated_config)}, allow_val_change=True)

    train_args = SimpleNamespace(
        config=str(generated_config),
        wandb=True,
        run_name=run_name,
        wandb_project=project,
        watch_model=args.watch_model,
        swept_config_keys=list(values),
    )
    train_main(train=True, arg=train_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
