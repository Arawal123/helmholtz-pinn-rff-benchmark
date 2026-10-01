"""Tiny CPU wiring check; not protected research evidence."""
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import train_run
from src.evaluate import evaluate_run
from src.utils import load_yaml


def main() -> None:
    smoke = load_yaml("configs/smoke.yaml")
    frozen = load_yaml("configs/frozen.yaml")
    config = {**frozen, "protocol": {**frozen["protocol"], "k": smoke["k"]},
              "data": {**frozen["data"], "interior_points": smoke["interior_points"],
                       "boundary_points_per_side": smoke["boundary_points_per_side"],
                       "heldout_grid_size": smoke["grid_size"]},
              "training": {**frozen["training"], "updates": smoke["updates"], "log_every": 1}}
    root = ROOT / smoke["output_root"] / f"smoke_{uuid4().hex[:12]}"
    root.mkdir(parents=True, exist_ok=False)
    for method in ("vanilla", "rff"):
        directory = root / method
        if directory.exists():
            raise FileExistsError(f"Smoke output exists; remove manually before rerun: {directory}")
        directory.mkdir()
        status = train_run(config, smoke["n"], method, smoke["seed"], directory,
                           {"purpose": smoke["label"]}, smoke=True)
        if status["status"] != "completed" or status["updates_completed"] != smoke["updates"]:
            raise RuntimeError(f"Smoke failed: {method}: {status}")
    # The non-official smoke gate follows both tiny training runs.
    for method in ("vanilla", "rff"):
        metrics = evaluate_run(config, smoke["n"], method, root / method, smoke=True)
        if metrics["grid_size"] != smoke["grid_size"]:
            raise RuntimeError("Smoke evaluation grid mismatch")
    print(smoke["label"] + ": PASS")


if __name__ == "__main__":
    main()
