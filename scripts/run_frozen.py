"""Protected orchestration: whole pipeline or individual frozen run slots."""
from __future__ import annotations

import argparse
import csv
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_protocol import validate
from src.evaluate import evaluate_run
from src.provenance import runtime_record
from src.train import train_run
from src.utils import atomic_json_dump, utc_now


def attempt_dirs(run_root: Path) -> list[Path]:
    return sorted(path for path in run_root.glob("attempt_*")) if run_root.exists() else []


def latest_state(run_root: Path) -> tuple[Path | None, dict | None]:
    attempts = attempt_dirs(run_root)
    if not attempts:
        return None, None
    path = attempts[-1]
    record = path / "run.json"
    return path, json.loads(record.read_text(encoding="utf-8")) if record.exists() else None


def reserve_attempt(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    index = len(attempt_dirs(run_root))
    directory = run_root / f"attempt_{index:03d}"
    directory.mkdir(exist_ok=False)
    return directory


def terminal(state: dict | None) -> bool:
    return state is not None and state["status"] in ("completed", "failed")


def mark_interrupted(directory: Path, state: dict | None, reason: str) -> None:
    record = dict(state or {})
    record.update(status="infrastructure_interrupted", interruption_reason=reason,
                  interrupted_at_utc=utc_now())
    atomic_json_dump(directory / "run.json", record)


def require_condition_evaluated(n: int, provenance: dict) -> None:
    slots = []
    for method in ("vanilla", "rff"):
        for seed in (0, 1, 2):
            directory, state = latest_state(ROOT / f"results/raw/n{n:02d}/{method}/seed_{seed}")
            if not terminal(state):
                raise RuntimeError(f"Finish condition n={n} before proceeding: {method} seed={seed}")
            if any(state.get("provenance", {}).get(key) != provenance[key] for key in
                   ("git_sha", "config_sha256", "sobol_sha256", "rff_sha256")):
                raise RuntimeError(f"Existing attempt requires its original revision/assets: {directory}")
            slots.append((method, seed, directory, state))
    for method, seed, directory, state in slots:
        if state["status"] == "completed" and (
                state.get("evaluation_status") != "completed" or not (directory / "metrics.json").exists()):
            raise RuntimeError(f"Evaluate condition n={n} before proceeding: {method} seed={seed}")


def print_status() -> None:
    """Read training/status metadata only, without CUDA or held-out metrics."""
    finished = 0
    print("n   method   seed  status                       logged_update")
    for n in (6, 12):
        for method in ("vanilla", "rff"):
            for seed in (0, 1, 2):
                directory, state = latest_state(ROOT / f"results/raw/n{n:02d}/{method}/seed_{seed}")
                status = state.get("status", "infrastructure_interrupted") if state else (
                    "infrastructure_interrupted" if directory else "not_started")
                update = state.get("updates_completed", 0) if state else 0
                if directory and (directory / "training_log.csv").exists():
                    with (directory / "training_log.csv").open(newline="", encoding="utf-8") as handle:
                        for row in csv.DictReader(handle):
                            if row.get("update", "").isdigit():
                                update = max(update, int(row["update"]))
                finished += terminal(state)
                print(f"{n:<3} {method:<8} {seed:<5} {status:<28} {update}/8000")
    print(f"Terminal training slots: {finished}/12")


def execute(config_path: str, recover_infrastructure: bool = False, *, action: str = "all",
            n: int | None = None, method: str | None = None, seed: int | None = None) -> None:
    if action not in ("all", "train", "evaluate", "finalize", "status"):
        raise ValueError("Unknown action")
    if action == "train":
        if n not in (6, 12) or method not in ("vanilla", "rff") or seed not in (0, 1, 2):
            raise ValueError("train requires --n {6,12}, --method {vanilla,rff}, --seed {0,1,2}")
    elif action == "evaluate":
        if n not in (6, 12) or method is not None or seed is not None:
            raise ValueError("evaluate requires only --n {6,12}")
    elif any(value is not None for value in (n, method, seed)):
        raise ValueError(f"{action} does not accept run selectors")
    config = validate(config_path)
    if action == "status":
        print_status()
        return
    provenance = runtime_record(ROOT / config_path, config)
    if action == "finalize":
        for condition in (6, 12):
            require_condition_evaluated(condition, provenance)
        from scripts.aggregate_results import aggregate
        aggregate(config, provenance)
        print("All 12 protected run slots terminal; processed evidence generated.", flush=True)
        return
    if action in ("train", "evaluate") and n == 12:
        require_condition_evaluated(6, provenance)
    raw = ROOT / "results/raw"
    conditions = (n,) if action in ("train", "evaluate") else (6, 12)
    methods = (method,) if action == "train" else ("vanilla", "rff")
    seeds = (seed,) if action == "train" else (0, 1, 2)
    # Condition barrier: no held-out data or metrics are touched during training.
    for n in conditions:
        condition = []
        for method in methods:
            for seed in seeds:
                run_root = raw / f"n{n:02d}" / method / f"seed_{seed}"
                directory, state = latest_state(run_root)
                if terminal(state):
                    prior = state["provenance"]
                    if any(prior.get(key) != provenance[key] for key in
                           ("git_sha", "config_sha256", "sobol_sha256", "rff_sha256")):
                        raise RuntimeError(f"Existing protected run uses a different code/config revision: {directory}")
                    condition.append((method, seed, directory, state))
                    print(f"SKIP n={n} {method} seed={seed}: already {state['status']}", flush=True)
                    continue
                if action == "evaluate":
                    raise RuntimeError(f"Condition n={n} is not terminal: {method} seed={seed}; evaluation/finalization blocked")
                if directory is not None:
                    if not recover_infrastructure:
                        raise RuntimeError(f"Interrupted attempt requires explicit --recover-infrastructure: {directory}")
                    prior = (state or {}).get("provenance", {})
                    if prior and (prior.get("git_sha") != provenance["git_sha"] or
                                  prior.get("config_sha256") != provenance["config_sha256"] or
                                  prior.get("sobol_sha256") != provenance["sobol_sha256"] or
                                  prior.get("rff_sha256") != provenance["rff_sha256"]):
                        raise RuntimeError(f"Infrastructure recovery requires identical frozen revision/assets: {directory}")
                    if state is None or state.get("status") == "running":
                        mark_interrupted(directory, state, "process ended without a terminal record")
                    elif state.get("status") != "infrastructure_interrupted":
                        raise RuntimeError(f"Unknown attempt status: {directory}")
                directory = reserve_attempt(run_root)
                print(f"TRAIN n={n} method={method} seed={seed} -> {directory}", flush=True)
                try:
                    state = train_run(config, n, method, seed, directory, provenance)
                except BaseException as error:
                    existing = directory / "run.json"
                    record = json.loads(existing.read_text(encoding="utf-8")) if existing.exists() else None
                    mark_interrupted(directory, record, f"{type(error).__name__}: {error}\n{traceback.format_exc()}")
                    raise
                condition.append((method, seed, directory, state))
                print(f"TRAINING TERMINAL n={n} {method} seed={seed}: {state['status']}", flush=True)
        if action == "train":
            return
        if len(condition) != 6 or not all(terminal(state) for _, _, _, state in condition):
            raise RuntimeError(f"Condition n={n} has not reached six terminal training states")
        print(f"CONDITION GATE PASSED n={n}; starting held-out evaluation", flush=True)
        for method, seed, directory, state in condition:
            if state["status"] == "failed":
                continue
            if (directory / "metrics.json").exists():
                if state.get("evaluation_status") != "completed":
                    state["evaluation_status"] = "completed"
                    atomic_json_dump(directory / "run.json", state)
                continue
            if state.get("evaluation_status") == "infrastructure_interrupted" and not recover_infrastructure:
                raise RuntimeError(f"Evaluation interruption requires explicit --recover-infrastructure: {directory}")
            try:
                print(f"EVALUATE n={n} {method} seed={seed}", flush=True)
                evaluate_run(config, n, method, directory)
                state["evaluation_status"] = "completed"
            except FloatingPointError as error:
                state.update(status="failed", evaluation_status="failed_nonfinite",
                             failure_reason=str(error), failure_step="heldout_evaluation")
            except BaseException:
                state["evaluation_status"] = "infrastructure_interrupted"
                atomic_json_dump(directory / "run.json", state)
                raise
            atomic_json_dump(directory / "run.json", state)
        print(f"CONDITION EVALUATION COMPLETE n={n}", flush=True)
    if action == "evaluate":
        return
    for n in (6, 12):
        require_condition_evaluated(n, provenance)
    from scripts.aggregate_results import aggregate
    aggregate(config, provenance)
    print("All 12 protected run slots terminal; processed evidence generated.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/frozen.yaml", choices=["configs/frozen.yaml"])
    parser.add_argument("--action", default="all", choices=["all", "train", "evaluate", "finalize", "status"])
    parser.add_argument("--n", type=int, choices=[6, 12])
    parser.add_argument("--method", choices=["vanilla", "rff"])
    parser.add_argument("--seed", type=int, choices=[0, 1, 2])
    parser.add_argument("--recover-infrastructure", action="store_true",
                        help="Explicitly re-execute only infrastructure-interrupted attempts from the same frozen Git/config revision")
    args = parser.parse_args()
    try:
        execute(args.config, args.recover_infrastructure, action=args.action,
                n=args.n, method=args.method, seed=args.seed)
    except ValueError as error:
        parser.error(str(error))
