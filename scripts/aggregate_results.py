"""Create all processed tables, figures, and the scientific note from protected raw evidence."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import load_yaml

METRICS = ("relative_l2", "linf", "normalized_residual_rms", "boundary_rmse")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect() -> tuple[list[dict], list[dict], list[dict]]:
    per_seed, manifest, compute = [], [], []
    for n in (6, 12):
        for method in ("vanilla", "rff"):
            for seed in (0, 1, 2):
                root = ROOT / f"results/raw/n{n:02d}/{method}/seed_{seed}"
                attempts = sorted(root.glob("attempt_*"))
                if not attempts:
                    per_seed.append(dict(n=n, condition="primary" if n == 6 else "preregistered_stress_failure",
                                         method=method, seed=seed, status="PENDING PROTECTED RUN"))
                    continue
                for attempt in attempts:
                    file = attempt / "run.json"
                    state = json.loads(file.read_text(encoding="utf-8")) if file.exists() else {"status": "infrastructure_interrupted"}
                    source = state.get("provenance", {})
                    metric_file = attempt / "metrics.json"
                    raw_metrics = json.loads(metric_file.read_text(encoding="utf-8")) if metric_file.exists() else {}
                    manifest.append(dict(n=n, method=method, seed=seed, attempt=attempt.name,
                                         selected=attempt == attempts[-1], status=state.get("status"),
                                         evaluation_status=state.get("evaluation_status"),
                                         path=attempt.relative_to(ROOT).as_posix(), git_sha=source.get("git_sha"),
                                         config_sha256=source.get("config_sha256"),
                                         sobol_sha256=source.get("sobol_sha256", {}).get(str(seed)),
                                         rff_sha256=source.get("rff_sha256") if method == "rff" else "",
                                         failure_reason=state.get("failure_reason", state.get("interruption_reason", ""))))
                    compute.append(dict(n=n, method=method, seed=seed, attempt=attempt.name,
                                        status=state.get("status"), accelerator=source.get("accelerator"),
                                        parameter_count=state.get("parameter_count"),
                                        wall_seconds_training=state.get("wall_seconds_training"),
                                        wall_seconds_evaluation=raw_metrics.get("evaluation_wall_seconds"),
                                        wall_seconds_total=(state.get("wall_seconds_training", 0) + raw_metrics.get("evaluation_wall_seconds", 0))
                                        if state.get("wall_seconds_training") is not None else "",
                                        started_at_utc=state.get("started_at_utc"),
                                        ended_at_utc=state.get("ended_at_utc"), git_sha=source.get("git_sha")))
                chosen = attempts[-1]
                state = json.loads((chosen / "run.json").read_text(encoding="utf-8"))
                row = dict(n=n, condition="primary" if n == 6 else "preregistered_stress_failure",
                           method=method, seed=seed, status=state["status"],
                           evaluation_status=state.get("evaluation_status"), attempt=chosen.name,
                           git_sha=state.get("provenance", {}).get("git_sha"),
                           accelerator=state.get("provenance", {}).get("accelerator"),
                           parameter_count=state.get("parameter_count"),
                           wall_seconds_training=state.get("wall_seconds_training"),
                           failure_reason=state.get("failure_reason", ""))
                metric_path = chosen / "metrics.json"
                if metric_path.exists():
                    row.update(json.loads(metric_path.read_text(encoding="utf-8")))
                per_seed.append(row)
    return per_seed, manifest, compute


def effects_and_summary(per_seed: list[dict]) -> tuple[list[dict], list[dict]]:
    effects, summary = [], []
    for n in (6, 12):
        for seed in (0, 1, 2):
            vanilla = next(row for row in per_seed if (row["n"], row["method"], row["seed"]) == (n, "vanilla", seed))
            rff = next(row for row in per_seed if (row["n"], row["method"], row["seed"]) == (n, "rff", seed))
            for metric in METRICS:
                a, b = vanilla.get(metric), rff.get(metric)
                effect = dict(n=n, seed=seed, metric=metric, vanilla=a, rff=b,
                              absolute_difference_rff_minus_vanilla="", percentage_improvement="",
                              status="unavailable_failure_or_pending")
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    effect.update(absolute_difference_rff_minus_vanilla=b-a,
                                  percentage_improvement=100*(a-b)/a if a != 0 else "undefined_zero_baseline",
                                  status="computed")
                effects.append(effect)
        for method in ("vanilla", "rff"):
            rows = [row for row in per_seed if row["n"] == n and row["method"] == method]
            for metric in METRICS:
                values = [row[metric] for row in rows if isinstance(row.get(metric), (int, float))]
                summary.append(dict(n=n, condition="primary" if n == 6 else "preregistered_stress_failure",
                                    method=method, metric=metric, n_valid=len(values), n_planned=3,
                                    mean=statistics.mean(values) if values else "",
                                    sd=statistics.stdev(values) if len(values)>1 else ""))
        for metric in METRICS:
            paired = [row for row in effects if row["n"] == n and row["metric"] == metric and row["status"] == "computed"]
            for field, label in (("absolute_difference_rff_minus_vanilla", "paired_absolute_difference"),
                                 ("percentage_improvement", "paired_percentage_improvement")):
                values = [row[field] for row in paired if isinstance(row[field], (int, float))]
                summary.append(dict(n=n, condition="primary" if n == 6 else "preregistered_stress_failure",
                                    method=label, metric=metric, n_valid=len(values), n_planned=3,
                                    mean=statistics.mean(values) if values else "",
                                    sd=statistics.stdev(values) if len(values)>1 else ""))
    return effects, summary


def make_figures(per_seed: list[dict]) -> None:
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    for n in (6, 12):
        fig, ax = plt.subplots(figsize=(8, 5))
        for row in per_seed:
            if row["n"] != n or not row.get("attempt"):
                continue
            path = ROOT / f"results/raw/n{n:02d}/{row['method']}/seed_{row['seed']}/{row['attempt']}"
            log = path / "training_log.csv"
            if not log.exists():
                continue
            with log.open(newline="", encoding="utf-8") as handle:
                data = list(csv.DictReader(handle))
            if data:
                ax.semilogy([int(x["update"]) for x in data], [float(x["total_loss"]) for x in data],
                            label=f"{row['method']} s{row['seed']}")
        ax.set(xlabel="Adam update", ylabel="Training loss", title=f"n={n} training convergence")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"n{n:02d}_training_curves.png", dpi=170)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        labels, values, colors = [], [], []
        for row in per_seed:
            if row["n"] == n and isinstance(row.get("relative_l2"), (int, float)):
                labels.append(f"{row['method']} s{row['seed']}")
                values.append(row["relative_l2"])
                colors.append("#345ca8" if row["method"] == "vanilla" else "#b55435")
        ax.bar(labels, values, color=colors)
        ax.set(ylabel="Held-out relative L2", title=f"n={n}: all available seeds")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(figures / f"n{n:02d}_comparison.png", dpi=170)
        plt.close(fig)
        axis = np.linspace(-1, 1, 257)
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        exact = np.sin(n*np.pi*xx)*np.sin(n*np.pi*yy)
        for row in per_seed:
            if row["n"] != n or not row.get("attempt"):
                continue
            path = ROOT / f"results/raw/n{n:02d}/{row['method']}/seed_{row['seed']}/{row['attempt']}/grid_predictions.npy"
            if not path.exists():
                continue
            pred = np.load(path, allow_pickle=False)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, image, title in zip(axes, (exact, pred, np.abs(pred-exact)),
                                        ("Exact", "Prediction", "Absolute error")):
                handle = ax.imshow(image, origin="lower", extent=(-1,1,-1,1), cmap="coolwarm" if title != "Absolute error" else "magma")
                ax.set_title(title)
                fig.colorbar(handle, ax=ax, shrink=.75)
            fig.suptitle(f"n={n} {row['method']} seed {row['seed']}")
            fig.tight_layout()
            fig.savefig(figures / f"n{n:02d}_{row['method']}_seed_{row['seed']}_error.png", dpi=140)
            plt.close(fig)


def make_report(per_seed: list[dict], effects: list[dict], summary: list[dict], provenance: dict | None = None) -> None:
    lines = ["# Methods and results — Ryan PINN failure-and-repair trial", "",
             "## Status", ""]
    complete = len(per_seed) == 12 and all(row.get("status") in ("completed", "failed") and
                                            (row.get("status") == "failed" or all(metric in row for metric in METRICS))
                                            for row in per_seed)
    lines.append("PROTECTED RUN COMPLETE" if complete else "PENDING PROTECTED RUN")
    lines += ["", "## Problem and preregistered hypothesis", "",
              "On Ω=[-1,1]², solve Δu+20²u=f with zero Dirichlet data and u*=sin(nπx)sin(nπy). "
              "The hypothesis is that fixed random Fourier features mitigate spectral bias of the tanh PINN "
              "on primary n=6. n=12 remains the preregistered stress/failure condition.", "",
              "## Frozen methods", "",
              "For each seed 0,1,2, both methods and both n values reuse the same committed 2,048-point scrambled Sobol set. "
              "The 512 boundary points are fixed. Both models use the normalized PDE residual plus boundary MSE, "
              "full-batch float32 Adam for exactly 8,000 updates, and cosine LR 1e-3 to 1e-4. "
              "Vanilla is 2→64→64→64→64→1 tanh (12,737 parameters); repair is a fixed seed-2026 "
              "32×2 Gaussian B (σ=3) with sin/cos features then 64→55→55→55→55→1 tanh (12,871 parameters). "
              "There is no adaptive weighting, resampling, early stopping, best-checkpoint selection, or post-hoc tuning.", "",
              "Held-out 257×257 grid metrics are evaluated only after all six training runs at that n are terminal. "
              "Scientific failures are retained without restart; infrastructure interruptions have separate attempts.", "",
              "## Per-seed results", "",
              "| n / role | method | seed | status | rel L2 | L∞ | residual RMS | boundary RMSE | train s |", "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in per_seed:
        fmt = lambda key: f"{row[key]:.6g}" if isinstance(row.get(key), (int,float)) else "—"
        lines.append(f"| {row['n']} / {row['condition']} | {row['method']} | {row['seed']} | {row['status']} | "
                     f"{fmt('relative_l2')} | {fmt('linf')} | {fmt('normalized_residual_rms')} | "
                     f"{fmt('boundary_rmse')} | {fmt('wall_seconds_training')} |")
    lines += ["", "## Paired descriptive effects", "",
              "Difference is RFF − vanilla; improvement is 100×(vanilla − RFF)/vanilla when vanilla is nonzero. "
              "Negative improvement means repair lost. These are descriptive comparisons across three seeds, not significance tests.", "",
              "| n | seed | metric | difference | improvement % |", "|---:|---:|---|---:|---:|"]
    for row in effects:
        a = row["absolute_difference_rff_minus_vanilla"]
        b = row["percentage_improvement"]
        lines.append(f"| {row['n']} | {row['seed']} | {row['metric']} | {a if a != '' else '—'} | {b if b != '' else '—'} |")
    lines += ["", "## Mean and sample SD by condition/method", "",
              "| n | method | metric | valid/planned | mean | SD |", "|---:|---|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['n']} | {row['method']} | {row['metric']} | {row['n_valid']}/{row['n_planned']} | "
                     f"{row['mean'] if row['mean'] != '' else '—'} | {row['sd'] if row['sd'] != '' else '—'} |")
    lines += ["", "## Convergence, compute, failures, and seed sensitivity", "",
              "| n | method | seed | first logged loss | last logged loss | last logged update |",
              "|---:|---|---:|---:|---:|---:|"]
    for row in per_seed:
        if not row.get("attempt"):
            continue
        log_path = ROOT / f"results/raw/n{row['n']:02d}/{row['method']}/seed_{row['seed']}/{row['attempt']}/training_log.csv"
        if not log_path.exists():
            continue
        with log_path.open(newline="", encoding="utf-8") as handle:
            logs = list(csv.DictReader(handle))
        if logs:
            lines.append(f"| {row['n']} | {row['method']} | {row['seed']} | {logs[0]['total_loss']} | "
                         f"{logs[-1]['total_loss']} | {logs[-1]['update']} |")
    measured = [row["wall_seconds_training"] for row in per_seed if isinstance(row.get("wall_seconds_training"), (int,float))]
    lines += ["", f"Measured training wall time across selected attempts: {sum(measured):.3f} s "
              f"for {len(measured)}/12 slots. Per-attempt compute and actual accelerators are in `results/processed/compute_record.csv`.", ""]
    failed = [row for row in per_seed if row.get("status") == "failed"]
    if failed:
        lines.append("Scientific failures (retained without retry):")
        lines.append("")
        for row in failed:
            lines.append(f"- n={row['n']} {row['method']} seed={row['seed']}: {row.get('failure_reason') or 'see raw run.json'}")
    else:
        lines.append("Scientific failures: none recorded" if complete else "Scientific failures: PENDING PROTECTED RUN")
    lines.append("")
    for n in (6, 12):
        paired = [row for row in effects if row["n"] == n and row["metric"] == "relative_l2" and row["status"] == "computed"]
        if paired:
            wins = sum(row["absolute_difference_rff_minus_vanilla"] < 0 for row in paired)
            losses = sum(row["absolute_difference_rff_minus_vanilla"] > 0 for row in paired)
            lines.append(f"n={n} paired primary-metric comparison: RFF lower error in {wins}/{len(paired)}, "
                         f"higher error in {losses}/{len(paired)} available pairs; see every seed above.")
        else:
            lines.append(f"n={n} paired primary-metric comparison: PENDING PROTECTED RUN or unavailable because of failures.")
    lines.append("")
    lines.append("Seed sensitivity is the per-seed spread and sample SD in the tables above; three seeds support descriptive interpretation only.")
    lines += ["", "## Interpretation and scope", "",
              "Training convergence and error fields are in figures/. Compute, hardware, and failure records are in results/processed/ "
              "and the per-attempt raw directories. Per-seed variability is the approved seed sensitivity evidence. "
              "An independent hyperparameter sweep was not part of the approved protected protocol and was not added post hoc. "
              "This limits claims about hyperparameter sensitivity. Preserve negative, divergent, or inconclusive outcomes.", "",
              "## Reproducibility", "",
              "Exact command: `python scripts/run_frozen.py --config configs/frozen.yaml`. "
              "Frozen asset hashes and resolved configs appear in raw provenance. GPU floating-point execution may vary "
              "across accelerator and library versions despite deterministic seeds and algorithm settings.", ""]
    if provenance:
        lines += [f"Git SHA: `{provenance['git_sha']}`. Accelerator: `{provenance['accelerator']}`. "
                  f"PyTorch: `{provenance['torch']}`. CUDA: `{provenance['cuda_runtime']}`.", ""]
    (ROOT / "report").mkdir(exist_ok=True)
    (ROOT / "report/methods_results.md").write_text("\n".join(lines), encoding="utf-8")


def aggregate(config: dict | None = None, provenance: dict | None = None) -> None:
    per_seed, manifest, compute = collect()
    effects, summary = effects_and_summary(per_seed)
    output = ROOT / "results/processed"
    write_csv(output / "per_seed_results.csv", per_seed, ["n","condition","method","seed","status","evaluation_status","attempt","git_sha","accelerator","parameter_count","wall_seconds_training","failure_reason",*METRICS,"evaluation_wall_seconds","evaluated_at_utc","grid_size"])
    write_csv(output / "paired_effects.csv", effects, ["n","seed","metric","vanilla","rff","absolute_difference_rff_minus_vanilla","percentage_improvement","status"])
    write_csv(output / "summary.csv", summary, ["n","condition","method","metric","n_valid","n_planned","mean","sd"])
    write_csv(output / "compute_record.csv", compute, ["n","method","seed","attempt","status","accelerator","parameter_count","wall_seconds_training","wall_seconds_evaluation","wall_seconds_total","started_at_utc","ended_at_utc","git_sha"])
    write_csv(output / "run_manifest.csv", manifest, ["n","method","seed","attempt","selected","status","evaluation_status","path","git_sha","config_sha256","sobol_sha256","rff_sha256","failure_reason"])
    if all(row["status"] in ("completed", "failed") for row in per_seed):
        make_figures(per_seed)
    make_report(per_seed, effects, summary, provenance)


if __name__ == "__main__":
    aggregate(load_yaml("configs/frozen.yaml"))
