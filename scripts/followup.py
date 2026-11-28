"""Recompute the follow-up comparisons from the published CSVs or a replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from scaling.config import PRESETS


RESULTS = Path(__file__).resolve().parents[1] / "results"
EXPERIMENT = RESULTS / "followup" / "experiment.json"


def validation_rows(path: Path, validation_set: str) -> list[dict]:
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["val_set"] == validation_set]


def endpoint(directory: Path, run: dict, experiment: dict) -> dict:
    path = directory / f"{run['id']}.csv"
    rows = validation_rows(path, experiment["validation_set"])
    steps = run["steps"]
    if [int(row["step"]) for row in rows] != [round(steps / 2), steps]:
        raise ValueError(f"{path}: expected one midpoint and one terminal checkpoint")
    for row in rows:
        if (
            row["run_id"] != run["id"]
            or row["model_size"] != run["model"]
            or int(row["seed"]) != run["seed"]
            or int(row["tokens_seen"]) != int(row["step"]) * experiment["tokens_per_step"]
            or int(row["n_doc"]) != experiment["validation_documents"]
            or not math.isfinite(float(row["val_loss"]))
        ):
            raise ValueError(f"{path}: CSV row does not match the experiment")
    if not math.isclose(float(rows[-1]["lr"]), 5e-5, rel_tol=0.001):
        raise ValueError(f"{path}: unexpected terminal learning rate")
    return rows[-1]


def baseline_checkpoint(run: dict, terminal: dict, validation_set: str) -> dict:
    path = RESULTS / "baseline" / "runs" / f"{run['reference']}.csv"
    matches = [
        row for row in validation_rows(path, validation_set)
        if row["tokens_seen"] == terminal["tokens_seen"]
        and row["model_size"] == terminal["model_size"]
        and row["seed"] == terminal["seed"]
        and row["n_doc"] == terminal["n_doc"]
    ]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected exactly one matching checkpoint")
    return matches[0]


def predict(surface: dict, model: str, tokens: int) -> float:
    e, a, b, alpha, beta = surface["theta_scaled"]
    units = surface["scaling_units"]
    n = PRESETS[model].n_params_nonemb
    return e + a * (n / units["N0"]) ** -alpha + b * (tokens / units["D0"]) ** -beta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=EXPERIMENT.parent / "runs")
    args = parser.parse_args()
    experiment = json.loads(EXPERIMENT.read_text())
    runs = experiment["runs"]
    endpoints = {run["id"]: endpoint(args.runs_dir, run, experiment) for run in runs}

    print("Schedule controls (short terminal minus long-run checkpoint)")
    print(f"{'Model':<6} {'Seed':>4} {'Tokens':>12} {'Long loss':>10} {'Short loss':>11} {'Delta':>10}")
    for run in runs:
        if "reference" not in run:
            continue
        new = endpoints[run["id"]]
        old = baseline_checkpoint(run, new, experiment["validation_set"])
        long_loss, short_loss = float(old["val_loss"]), float(new["val_loss"])
        print(f"{run['model']:<6} {run['seed']:>4} {int(new['tokens_seen']):>12,} "
              f"{long_loss:>10.6f} {short_loss:>11.6f} {short_loss - long_loss:>+10.6f}")

    print()
    print("IsoFLOP comparison (25M minus 20M)")
    print(f"{'Seed':>4} {'20M loss':>10} {'25M loss':>10} {'Delta':>10}")
    pairs = {
        (run["seed"], run["model"]): endpoints[run["id"]]
        for run in runs if "reference" not in run
    }
    deltas = []
    for seed in sorted({seed for seed, _ in pairs}):
        loss20 = float(pairs[seed, "20M"]["val_loss"])
        loss25 = float(pairs[seed, "25M"]["val_loss"])
        deltas.append(loss25 - loss20)
        print(f"{seed:>4} {loss20:>10.6f} {loss25:>10.6f} {deltas[-1]:>+10.6f}")

    tokens = {model: int(row["tokens_seen"]) for (_, model), row in pairs.items()}
    surface = experiment["frozen_surface"]
    frozen_delta = predict(surface, "25M", tokens["25M"]) - predict(surface, "20M", tokens["20M"])
    print(f"Mean observed gap  {mean(deltas):+.6f} nats")
    print(f"Frozen prediction  {frozen_delta:+.6f} nats")
    print(f"Observed minus predicted  {mean(deltas) - frozen_delta:+.5f} nats")
    print(f"Near-optimal tolerance    {experiment['near_optimal_tolerance_nats']:.6f} nats")


if __name__ == "__main__":
    main()
