"""Read-only progress report on a running sweep: feasibility, binding constraints, railed axes.

    python -m sweep.progress --output-dir sweep/outputs_v2 \\
        --study-name constrained-sweep-v2 --space sweep/spaces/constrained_v2.yaml
"""

import argparse
import json
import math
import sqlite3
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "sweep" / "outputs"

FEASIBLE_COLOR = "#2a78d6"
INFEASIBLE_COLOR = "#eb6834"
MUTED = "#8a8a83"

# Best trials this close to an axis end are pressed against it.
RAIL_FRACTION = 0.1


def load_trials(output_dir, study_name):
    """Every trial with its params, objectives and constraint violations."""
    path = Path(output_dir) / "study.db"
    if not path.exists():
        raise FileNotFoundError(f"no study database at {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    study_id = connection.execute(
        "SELECT study_id FROM studies WHERE study_name = ?", (study_name,)
    ).fetchone()
    if study_id is None:
        raise ValueError(f"no study named {study_name!r} in {path}")

    trials = {}
    for trial_id, number, state in connection.execute(
        "SELECT trial_id, number, state FROM trials WHERE study_id = ? ORDER BY number",
        (study_id[0],),
    ):
        trials[trial_id] = {"number": number, "state": state, "params": {}, "value": None}
    if not trials:
        return []

    for trial_id, value in connection.execute("SELECT trial_id, value FROM trial_values"):
        if trial_id in trials:
            trials[trial_id]["value"] = value
    for trial_id, name, value in connection.execute(
        "SELECT trial_id, param_name, param_value FROM trial_params"
    ):
        if trial_id in trials:
            trials[trial_id]["params"][name] = value
    for trial_id, key, payload in connection.execute(
        "SELECT trial_id, key, value_json FROM trial_user_attributes"
    ):
        if trial_id in trials and key in ("violations", "objectives"):
            trials[trial_id][key] = json.loads(payload)
    connection.close()
    return sorted(trials.values(), key=lambda trial: trial["number"])


def is_feasible(trial):
    violations = trial.get("violations")
    return bool(violations) and all(value <= 0.0 for value in violations.values())


def binding_constraints(trials):
    """How often each constraint is the one rejecting a finished trial."""
    counts, total = {}, 0
    for trial in trials:
        violations = trial.get("violations")
        if not violations:
            continue
        total += 1
        for name, value in violations.items():
            if value > 0.0:
                counts[name] = counts.get(name, 0) + 1
    return counts, total


def axis_position(value, axis):
    """Where a value sits in its axis, 0 at the low end and 1 at the high end."""
    low, high = float(axis["low"]), float(axis["high"])
    if axis.get("log"):
        return math.log(value / low) / math.log(high / low)
    return (value - low) / (high - low)


def rail_report(trials, space, top=8):
    """Axes whose best trials are pressed against an end of their own range."""
    ranked = sorted((t for t in trials if is_feasible(t)), key=lambda t: t["value"])[:top]
    if not ranked:
        return {}
    rails = {}
    for name, axis in space["axes"].items():
        if axis["type"] != "float":
            continue
        positions = [axis_position(t["params"][name], axis) for t in ranked if name in t["params"]]
        if not positions:
            continue
        mean = sum(positions) / len(positions)
        at_low = all(p < RAIL_FRACTION for p in positions)
        at_high = all(p > 1.0 - RAIL_FRACTION for p in positions)
        rails[name] = {"mean": mean, "rail": "low" if at_low else ("high" if at_high else None)}
    return rails


def text_report(trials, space):
    states = {}
    for trial in trials:
        states[trial["state"]] = states.get(trial["state"], 0) + 1
    feasible = [t for t in trials if is_feasible(t)]
    lines = [
        "trial states: " + ", ".join(f"{k} {v}" for k, v in sorted(states.items())),
        f"feasible: {len(feasible)}",
    ]

    if feasible:
        best = min(feasible, key=lambda t: t["value"])
        lines.append(f"\nbest feasible: trial {best['number']}, score {best['value']:.4f}")
        for name, value in sorted(best.get("objectives", {}).items()):
            lines.append(f"    {name:18s} {value:.4f}")
        lines.append("  parameters:")
        for name, value in sorted(best["params"].items()):
            lines.append(f"    {name:40s} {value:.6g}")
    else:
        lines.append("\nno feasible trial yet")

    counts, total = binding_constraints(trials)
    if total:
        lines.append(f"\nconstraints rejecting trials ({total} scored):")
        for name, count in sorted(counts.items(), key=lambda item: -item[1]):
            lines.append(f"    {name:28s} {count:3d}  ({count / total * 100:.0f}%)")
        if not counts:
            lines.append("    none")

    rails = rail_report(trials, space)
    pressed = {name: entry for name, entry in rails.items() if entry["rail"]}
    if pressed:
        lines.append("\naxes pressed against their range (every top trial within 10% of an end):")
        for name, entry in pressed.items():
            lines.append(f"    {name:40s} at the {entry['rail']} end (mean {entry['mean']:.2f})")
    elif rails:
        lines.append("\nno axis is railed: the best trials sit inside their ranges")
    return "\n".join(lines)


def figure(trials, space, path):
    """Four panels: progress, what rejects trials, the trade, and axis pressure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scored = [t for t in trials if t.get("violations")]
    if not scored:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for cell in axes.flat:
        cell.grid(True, color="#e6e6e1", linewidth=0.8)
        cell.set_axisbelow(True)
        for side in ("top", "right"):
            cell.spines[side].set_visible(False)

    # 1. score against trial order, with the running best
    cell = axes[0][0]
    feas = [(t["number"], t["value"]) for t in scored if is_feasible(t)]
    infeas = [(t["number"], t["value"]) for t in scored if not is_feasible(t)]
    if infeas:
        cell.scatter(
            *zip(*infeas),
            s=42,
            facecolors="none",
            edgecolors=INFEASIBLE_COLOR,
            linewidths=1.6,
            label="infeasible",
        )
    if feas:
        cell.scatter(*zip(*feas), s=42, color=FEASIBLE_COLOR, label="feasible")
        best, best_line = float("inf"), []
        for number, value in feas:
            best = min(best, value)
            best_line.append((number, best))
        cell.step(
            *zip(*best_line), where="post", color=FEASIBLE_COLOR, linewidth=2, label="best so far"
        )
    cell.set_yscale("log")
    cell.set_xlabel("trial")
    cell.set_ylabel("selection score (lower is better)")
    cell.set_title("Search progress", loc="left", fontsize=11)
    cell.legend(frameon=False, fontsize=9)

    # 2. which constraint does the rejecting
    cell = axes[0][1]
    counts, total = binding_constraints(trials)
    if counts:
        names = sorted(counts, key=lambda n: counts[n])
        cell.barh(names, [counts[n] / total * 100 for n in names], color=FEASIBLE_COLOR, height=0.7)
        cell.set_xlabel(f"% of {total} scored trials rejected by this limit")
    else:
        cell.text(
            0.5,
            0.5,
            "no trial violated any constraint",
            ha="center",
            va="center",
            color=MUTED,
            transform=cell.transAxes,
        )
    cell.set_title("What rejects a trial", loc="left", fontsize=11)

    # 3. the trade the score cannot see: bias against the worst resolution margin
    cell = axes[1][0]
    for subset, color, face, label in (
        ([t for t in scored if is_feasible(t)], FEASIBLE_COLOR, FEASIBLE_COLOR, "feasible"),
        ([t for t in scored if not is_feasible(t)], INFEASIBLE_COLOR, "none", "infeasible"),
    ):
        points = [
            (
                t["objectives"]["bias"],
                max(
                    (v for k, v in t["violations"].items() if k.startswith("resolution.")),
                    default=0.0,
                ),
            )
            for t in subset
            if t.get("objectives")
        ]
        if points:
            cell.scatter(
                *zip(*points),
                s=42,
                color=color,
                facecolors=face,
                edgecolors=color,
                linewidths=1.6,
                label=label,
            )
    cell.axhline(0.0, color=MUTED, linestyle="--", linewidth=1.5)
    cell.set_xlabel("bias objective (lower is better)")
    cell.set_ylabel("worst resolution margin (0 = at the limit)")
    cell.set_title("Bias against resolution", loc="left", fontsize=11)
    cell.legend(frameon=False, fontsize=9)

    # 4. where the best trials sit inside each axis
    cell = axes[1][1]
    rails = rail_report(trials, space)
    if rails:
        names = list(rails)
        cell.barh(names, [rails[n]["mean"] for n in names], color=FEASIBLE_COLOR, height=0.7)
        cell.axvspan(0.0, RAIL_FRACTION, color=INFEASIBLE_COLOR, alpha=0.15)
        cell.axvspan(1.0 - RAIL_FRACTION, 1.0, color=INFEASIBLE_COLOR, alpha=0.15)
        cell.set_xlim(0, 1)
        cell.set_xlabel("mean position of the best trials in the axis (shaded = railed)")
        cell.set_yticklabels([n.split(".")[-1] for n in names], fontsize=9)
    else:
        cell.text(
            0.5,
            0.5,
            "no feasible trial yet",
            ha="center",
            va="center",
            color=MUTED,
            transform=cell.transAxes,
        )
    cell.set_title("Is the space wide enough", loc="left", fontsize=11)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--study-name", default="constrained-sweep")
    parser.add_argument("--space", type=Path, default=REPO_ROOT / "sweep/spaces/constrained.yaml")
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    trials = load_trials(args.output_dir, args.study_name)
    space = yaml.safe_load(Path(args.space).read_text())
    print(text_report(trials, space))
    if not args.no_figure:
        written = figure(trials, space, Path(args.output_dir) / "figure" / "progress.png")
        print(f"\nfigure: {written}" if written else "\nfigure: nothing scored yet")
    return trials


if __name__ == "__main__":
    main()
