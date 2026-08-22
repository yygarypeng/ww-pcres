#!/usr/bin/env python3
import argparse
import copy
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = Path("configs/config.yaml")
DEFAULT_MANIFEST = Path("artifacts/angular_diagnosis/baseline_manifest.json")
DEFAULT_CHECKPOINT = Path(
    "outputs/angular_diagnosis/baseline-resume-2/logs/version_0/checkpoints/"
    "reg-epoch=134-val_loss=12.49.ckpt"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/angular_diagnosis/continuations")


@dataclass(frozen=True)
class ContinuationSpec:
    name: str
    config: dict


def build_continuation_specs(base_config, angular_bandwidths, *, output_root):
    bandwidths = [float(value) for value in angular_bandwidths]
    if not bandwidths or any(value <= 0.0 for value in bandwidths):
        raise ValueError("joint angular feature bandwidths must be positive")

    treatments = {
        "A": None,
        "B": {"learning_rate": 5.0e-5},
        "C": {"angular_mmd_weight": 0.0},
        "D": {
            "angular_mmd_weight": 2000.0,
            "angular_mmd_bandwidths": bandwidths,
        },
    }
    specs = []
    for name, treatment in treatments.items():
        config = copy.deepcopy(base_config)
        params = config.setdefault("parameters", {})
        params.update(
            {
                "epochs": 137,
                "save_every_epoch": True,
                "disable_early_stopping": True,
            }
        )
        config.setdefault("paths", {})["saved_path"] = str(Path(output_root) / name)
        if treatment is not None:
            config["continuation_treatment"] = copy.deepcopy(treatment)
        else:
            config.pop("continuation_treatment", None)
        specs.append(ContinuationSpec(name=name, config=config))
    return specs


def build_train_command(*, config_path, checkpoint, gpu=None):
    command = [
        sys.executable,
        "train/train.py",
        "--config",
        str(config_path),
        "--resume-from",
        str(checkpoint),
    ]
    if gpu is not None:
        command.extend(["--gpu", str(gpu)])
    return command


def run_sequentially(named_commands, *, run_command=subprocess.run):
    for name, command in named_commands:
        print(f"[{name}] starting: {' '.join(command)}", flush=True)
        try:
            run_command(command, check=True, cwd=REPO_ROOT)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"continuation {name} failed with exit code {error.returncode}"
            ) from error
        print(f"[{name}] completed", flush=True)


def _repo_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def prepare_commands(*, base_config, manifest, checkpoint, output_root, gpu=None):
    base_path = _repo_path(base_config)
    manifest_path = _repo_path(manifest)
    checkpoint_path = _repo_path(checkpoint)
    missing = [path for path in (base_path, manifest_path, checkpoint_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "required continuation input missing: " + ", ".join(map(str, missing))
        )

    with base_path.open() as stream:
        base = yaml.safe_load(stream)
    with manifest_path.open() as stream:
        manifest_data = json.load(stream)
    try:
        bandwidths = manifest_data["feature_bandwidths"]["joint"]
    except (KeyError, TypeError) as error:
        raise ValueError("manifest must contain feature_bandwidths.joint") from error

    output_path = _repo_path(output_root)
    config_dir = output_path / "run-configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    specs = build_continuation_specs(base, bandwidths, output_root=output_path)
    commands = []
    for spec in specs:
        config_path = config_dir / f"{spec.name}.yaml"
        with config_path.open("w") as stream:
            yaml.safe_dump(spec.config, stream, sort_keys=False)
        commands.append(
            (
                spec.name,
                build_train_command(config_path=config_path, checkpoint=checkpoint_path, gpu=gpu),
            )
        )
    return commands


def parse_args():
    parser = argparse.ArgumentParser(description="Run controlled angular-loss continuations")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--gpu", type=int, choices=(0, 1))
    return parser.parse_args()


def main():
    args = parse_args()
    commands = prepare_commands(
        base_config=args.base_config,
        manifest=args.manifest,
        checkpoint=args.checkpoint,
        output_root=args.output_root,
        gpu=args.gpu,
    )
    run_sequentially(commands)


if __name__ == "__main__":
    main()
