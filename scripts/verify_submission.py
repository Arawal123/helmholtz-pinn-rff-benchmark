"""Read-only verification of the distributed evidence; never trains a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHA = "0677b40869fabf50035075e240aea3b98a0c7e05"
METRICS = ("relative_l2", "linf", "normalized_residual_rms", "boundary_rmse")


def rooted(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Missing or invalid submission file: {name}")
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    return json.loads(rooted(name).read_text(encoding="utf-8"))


def rows(name):
    with rooted(name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def equal(a, b, atol=1e-12, rtol=1e-10):
    return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)


def verify(cpu=False):
    count = 0
    def check(condition, label):
        nonlocal count
        if not condition:
            raise ValueError(label)
        count += 1

    manifest = rooted("submission/checksums.sha256")
    names = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        check(name not in names and digest(rooted(name)) == expected, f"Checksum mismatch: {name}")
        names.add(name)
    mapping = load("submission/provenance/original_file_map.json")
    check(len(mapping["files"]) == 149, "Original-file mapping is incomplete")
    for entry in mapping["files"]:
        path = rooted(entry["submission_path"])
        check(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"],
              f"Original evidence changed: {entry['original_path']}")
    states, metrics = {}, {}
    folders = sorted((ROOT / "results/raw").glob("n*/**/attempt_*"))
    check(len(folders) == 12, "Expected exactly 12 submitted attempts")
    config_hash = digest(rooted("configs/frozen.yaml"))
    for n in (6, 12):
        for method in ("vanilla", "rff"):
            for seed in (0, 1, 2):
                slot = (n, method, seed)
                prefix = f"results/raw/n{n:02d}/{method}/seed_{seed}/attempt_000"
                state, metric = load(prefix + "/run.json"), load(prefix + "/metrics.json")
                states[slot], metrics[slot] = state, metric
                check(state["status"] == state["evaluation_status"] == "completed"
                      and state["updates_completed"] == state["updates_planned"] == 8000,
                      f"Incomplete slot: {slot}")
                p = state["provenance"]
                check(p["git_sha"] == SHA and p["git_status"] == ""
                      and p["config_sha256"] == config_hash, f"Wrong provenance: {slot}")
                config = load(prefix + "/resolved_config.json")
                for record in config["data"]["frozen_sobol_assets"].values():
                    check(digest(rooted(record["path"])) == record["sha256"], "Sobol asset mismatch")
                rff = config["model"]["rff"]
                check(digest(rooted(rff["matrix_path"])) == rff["matrix_sha256"] == p["rff_sha256"],
                      "Fourier asset mismatch")
                check(p["sobol_sha256"] == {str(s): r["sha256"] for s, r in
                      config["data"]["frozen_sobol_assets"].items()}, "Sobol provenance mismatch")
                log = rows(prefix + "/training_log.csv")
                check([int(r["update"]) for r in log] == [1] + list(range(100, 8001, 100)),
                      f"Incomplete training log: {slot}")
                check(all(math.isfinite(float(r[k])) for r in log for k in
                      ("lr", "total_loss", "residual_mse", "boundary_mse")), "Nonfinite log")
                check(all(equal(r["lr"], .0001 + .0009 * (1 + math.cos(math.pi *
                      (int(r["update"])-1)/7999))/2) and equal(r["total_loss"],
                      float(r["residual_mse"])+float(r["boundary_mse"]), 1e-6, 1e-6)
                      for r in log), "Schedule or loss decomposition mismatch")
                check(all(math.isfinite(metric[k]) for k in METRICS), "Nonfinite metric")
    date = datetime.fromisoformat
    for n in (6, 12):
        check(max(date(s["ended_at_utc"]) for k, s in states.items() if k[0] == n)
              < min(date(m["evaluated_at_utc"]) for k, m in metrics.items() if k[0] == n),
              f"Condition evaluation gate failed: {n}")
    check(max(date(m["evaluated_at_utc"]) for k, m in metrics.items() if k[0] == 6)
          < min(date(s["started_at_utc"]) for k, s in states.items() if k[0] == 12), "Stress gate failed")
    tables = {name: rows(f"results/processed/{name}.csv") for name in
              ("per_seed_results", "paired_effects", "summary", "compute_record", "run_manifest")}
    check(tuple(len(t) for t in tables.values()) == (12, 24, 32, 12, 12), "Table dimensions mismatch")
    slot_for = lambda r: (int(r["n"]), r["method"], int(r["seed"]))
    for r in tables["per_seed_results"]:
        check(all(equal(r[k], metrics[slot_for(r)][k]) for k in METRICS), "Per-seed table mismatch")
    for r in tables["paired_effects"]:
        v = metrics[(int(r["n"]), "vanilla", int(r["seed"]))][r["metric"]]
        f = metrics[(int(r["n"]), "rff", int(r["seed"]))][r["metric"]]
        check(equal(r["vanilla"], v) and equal(r["rff"], f)
              and equal(r["absolute_difference_rff_minus_vanilla"], f-v)
              and equal(r["percentage_improvement"], 100*(v-f)/v), "Paired-effect mismatch")
    for r in tables["summary"]:
        n, method, metric = int(r["n"]), r["method"], r["metric"]
        if method in ("vanilla", "rff"):
            values = [metrics[(n, method, s)][metric] for s in (0, 1, 2)]
        else:
            field = "absolute_difference_rff_minus_vanilla" if method == "paired_absolute_difference" else "percentage_improvement"
            values = [float(e[field]) for e in tables["paired_effects"] if int(e["n"]) == n and e["metric"] == metric]
        check(equal(r["mean"], statistics.mean(values)) and equal(r["sd"], statistics.stdev(values))
              and r["n_valid"] == "3", "Summary mismatch")
    for r in tables["compute_record"]:
        check(equal(r["wall_seconds_training"], states[slot_for(r)]["wall_seconds_training"])
              and equal(r["wall_seconds_evaluation"], metrics[slot_for(r)]["evaluation_wall_seconds"])
              and equal(r["wall_seconds_total"], float(r["wall_seconds_training"])+float(r["wall_seconds_evaluation"])),
              "Compute record mismatch")
    check(all(r["selected"] == "True" and r["status"] == r["evaluation_status"] == "completed"
              and r["git_sha"] == SHA for r in tables["run_manifest"]), "Run manifest mismatch")
    print(f"PASS: {count} integrity and numerical checks; 149 original files preserved; 12 completed models.")
    if cpu:
        cpu_check(metrics)


def cpu_check(metrics):
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT))
    from src.data import boundary_points, heldout_grid
    from src.models import build_model
    from src.pde import exact_solution, normalized_residual
    torch.set_num_threads(2)
    config = load("results/raw/n06/vanilla/seed_0/attempt_000/resolved_config.json")
    matrix = np.load(rooted(config["model"]["rff"]["matrix_path"]), allow_pickle=False)
    _, grid, _ = heldout_grid(257)
    for (n, method, seed), official in metrics.items():
        prefix = f"results/raw/n{n:02d}/{method}/seed_{seed}/attempt_000"
        model = build_model(method, matrix if method == "rff" else None)
        model.load_state_dict(torch.load(rooted(prefix + "/final_model.pt"), map_location="cpu", weights_only=True))
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        predictions, residual_sum = [], 0.0
        for start in range(0, len(grid), 512):
            points = torch.tensor(grid[start:start+512], requires_grad=True)
            with torch.no_grad():
                predictions.append(model(points).flatten().numpy())
            residual_sum += float(normalized_residual(model, points, n, 20.).detach().square().sum())
        prediction = np.concatenate(predictions)
        exact = exact_solution(torch.tensor(grid), n).flatten().numpy().astype(np.float64)
        diff = prediction.astype(np.float64) - exact
        with torch.no_grad():
            boundary = math.sqrt(float(model(torch.tensor(boundary_points())).square().mean()))
        actual = dict(relative_l2=float(np.linalg.norm(diff)/np.linalg.norm(exact)), linf=float(np.abs(diff).max()),
                      normalized_residual_rms=math.sqrt(residual_sum/len(grid)), boundary_rmse=boundary)
        stored = np.load(rooted(prefix + "/grid_predictions.npy"), allow_pickle=False)
        if stored.shape != (257, 257) or not np.isfinite(stored).all() or np.abs(prediction-stored.ravel()).max() >= 1e-5:
            raise ValueError(f"Prediction reproduction failed: {(n, method, seed)}")
        if not all(equal(actual[k], official[k], 2e-6, 1e-6) for k in METRICS):
            raise ValueError(f"CPU metric reproduction failed: {(n, method, seed)}")
        print(f"PASS CPU: n={n}, {method}, seed={seed}; all four metrics reproduced.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-model-check", action="store_true", help="Re-evaluate saved weights without optimization")
    args = parser.parse_args()
    try:
        verify(args.cpu_model_check)
    except (ValueError, KeyError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
