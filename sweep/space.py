"""Search spaces: YAML axes keyed by dotted config paths.

``parameters.*`` and ``mmd.*`` axes land in the training config; ``sweep.*`` axes are
sweep-only switches, such as the bias-penalty gate.
"""

import copy
import json
from pathlib import Path

import yaml

SUPPORTED_TYPES = ("float", "int", "categorical")

# Implicit base-config values, so spaces with sweep-only axes still get a baseline point.
SWEEP_DEFAULTS = {
    "parameters.fourvec_loss": "l1",
    "parameters.huber_delta": 10.0,
    "parameters.loss_weights.fourvec_bias": 0.0,
    "sweep.bias_penalty": "off",
}


def load_space(path):
    """Read a space file and check it before any GPU time is spent on it."""
    with open(Path(path).expanduser(), "r") as handle:
        space = yaml.safe_load(handle)

    axes = space.get("axes")
    if not axes:
        raise ValueError(f"space {path} defines no axes")

    for dotted, axis in axes.items():
        kind = axis.get("type")
        if kind not in SUPPORTED_TYPES:
            raise ValueError(f"axis {dotted} has unsupported type {kind!r}")
        if kind == "categorical":
            if not axis.get("choices"):
                raise ValueError(f"categorical axis {dotted} needs a non-empty choices list")
        elif "low" not in axis or "high" not in axis:
            raise ValueError(f"{kind} axis {dotted} needs both low and high")
        for dependency in axis.get("when", {}):
            if dependency not in axes:
                raise ValueError(f"axis {dotted} depends on unknown axis {dependency}")
            if list(axes).index(dependency) >= list(axes).index(dotted):
                raise ValueError(f"axis {dotted} must come after its dependency {dependency}")
    return space


def _encode(choice):
    """Optuna categoricals must be hashable, so list choices travel as JSON text."""
    return json.dumps(choice) if isinstance(choice, (list, dict)) else choice


def _decode(value):
    if isinstance(value, str) and value[:1] in "[{":
        return json.loads(value)
    return value


def _is_active(axis, chosen):
    """Whether a conditional axis applies, given the values already drawn this trial."""
    return all(chosen.get(name) == expected for name, expected in axis.get("when", {}).items())


def suggest(space, trial):
    """Draw one point from the space, honouring ``when:`` conditions between axes."""
    chosen = {}
    for dotted, axis in space["axes"].items():
        if not _is_active(axis, chosen):
            continue
        if axis["type"] == "categorical":
            raw = trial.suggest_categorical(dotted, [_encode(item) for item in axis["choices"]])
            chosen[dotted] = _decode(raw)
        elif axis["type"] == "int":
            chosen[dotted] = trial.suggest_int(
                dotted, int(axis["low"]), int(axis["high"]), log=bool(axis.get("log", False))
            )
        else:
            chosen[dotted] = trial.suggest_float(
                dotted, float(axis["low"]), float(axis["high"]), log=bool(axis.get("log", False))
            )
    return chosen


def get_by_path(config, dotted, default=None):
    node = config
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def set_by_path(config, dotted, value):
    node = config
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def apply(base_config, chosen):
    """A deep copy of the base config with every drawn value written into place."""
    config = copy.deepcopy(base_config)
    for dotted, value in chosen.items():
        set_by_path(config, dotted, value)
    return config


def baseline_point(space, base_config):
    """The base config as a point in the space, or ``{}`` when it lies outside."""
    point = {}
    for dotted, axis in space["axes"].items():
        if not _is_active(axis, {name: _decode(value) for name, value in point.items()}):
            continue
        current = get_by_path(base_config, dotted, SWEEP_DEFAULTS.get(dotted))
        if current is None:
            return {}
        if axis["type"] == "categorical":
            encoded = _encode(current)
            if encoded not in [_encode(item) for item in axis["choices"]]:
                return {}
            point[dotted] = encoded
        else:
            if not float(axis["low"]) <= float(current) <= float(axis["high"]):
                return {}
            point[dotted] = int(current) if axis["type"] == "int" else float(current)
    return point
