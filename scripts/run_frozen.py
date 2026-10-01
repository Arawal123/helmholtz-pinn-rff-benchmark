"""The sole official entrypoint; no scientific CLI overrides."""
from __future__ import annotations

import argparse
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


def execute(config_path: str, recover_infrastructure: bool = False) -> None:
    config = validate(config_path)
    provenance = runtime_record(ROOT / config_path, config)
    raw = ROOT / "results/raw"
    # Condition barrier: no held-out data or metrics are touched during training.
    for n in (6, 12):
        condition = []
        for method in ("vanilla", "rff"):
            for seed in (0, 1, 2):
                run_root = raw / f"n{n:02d}" / method / f"seed_{seed}"
                directory, state = latest_state(run_root)
                if terminal(state):
                    prior = state["provenance"]
                    if any(prior.get(key) != provenance[key] for key in
                           ("git_sha", "config_sha256", "sobol_sha256", "rff_sha256")):
                        raise RuntimeError(f"Existing protected run uses a different code/config revision: {directory}")
                    condition.append((method, seed, directory, state))
                    continue
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
    from scripts.aggregate_results import aggregate
    aggregate(config, provenance)
    print("All 12 protected run slots terminal; processed evidence generated.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/frozen.yaml", choices=["configs/frozen.yaml"])
    parser.add_argument("--recover-infrastructure", action="store_true",
                        help="Explicitly re-execute only infrastructure-interrupted attempts from the same frozen Git/config revision")
    args = parser.parse_args()
    execute(args.config, args.recover_infrastructure)
