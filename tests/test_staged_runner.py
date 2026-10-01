"""Synthetic orchestration fixtures; no protected models are executed."""
import json
import ast
from pathlib import Path

import pytest

from scripts.validate_protocol import validate
from src.utils import atomic_json_dump


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    from scripts import run_frozen, aggregate_results
    config = validate()
    provenance = {"git_sha": "a"*40, "config_sha256": "b"*64,
                  "sobol_sha256": {str(s): config["data"]["frozen_sobol_assets"][s]["sha256"] for s in (0,1,2)},
                  "rff_sha256": config["model"]["rff"]["matrix_sha256"]}
    events = []
    monkeypatch.setattr(run_frozen, "ROOT", tmp_path)
    monkeypatch.setattr(run_frozen, "validate", lambda _: config)
    monkeypatch.setattr(run_frozen, "runtime_record", lambda *args: provenance)
    def train(config, n, method, seed, directory, provenance):
        events.append(("train", n, method, seed))
        state = {"status": "completed", "evaluation_status": "pending_condition_gate",
                 "updates_completed": 8000, "provenance": provenance}
        atomic_json_dump(directory / "run.json", state)
        return state
    def evaluate(config, n, method, directory):
        events.append(("evaluate", n, method))
        atomic_json_dump(directory / "metrics.json", {})
    monkeypatch.setattr(run_frozen, "train_run", train)
    monkeypatch.setattr(run_frozen, "evaluate_run", evaluate)
    monkeypatch.setattr(aggregate_results, "aggregate", lambda *args: events.append(("finalize",)))
    return run_frozen, events, provenance


def test_individual_training_delayed_evaluation_and_finalization(pipeline):
    runner, events, _ = pipeline
    runner.execute("configs/frozen.yaml", action="train", n=6, method="vanilla", seed=0)
    assert events == [("train", 6, "vanilla", 0)]
    runner.execute("configs/frozen.yaml", action="train", n=6, method="vanilla", seed=0)
    assert len(events) == 1  # No completed rerun.
    with pytest.raises(RuntimeError, match="not terminal"):
        runner.execute("configs/frozen.yaml", action="evaluate", n=6)
    with pytest.raises(RuntimeError, match="Finish condition"):
        runner.execute("configs/frozen.yaml", action="train", n=12, method="rff", seed=0)
    for method in ("vanilla", "rff"):
        for seed in (0, 1, 2):
            runner.execute("configs/frozen.yaml", action="train", n=6, method=method, seed=seed)
    assert all(event[0] == "train" for event in events)
    with pytest.raises(RuntimeError, match="Evaluate condition"):
        runner.execute("configs/frozen.yaml", action="train", n=12, method="rff", seed=0)
    runner.execute("configs/frozen.yaml", action="evaluate", n=6)
    assert sum(event[0] == "evaluate" for event in events) == 6
    with pytest.raises(RuntimeError, match="Finish condition"):
        runner.execute("configs/frozen.yaml", action="finalize")
    for method in ("vanilla", "rff"):
        for seed in (0, 1, 2):
            runner.execute("configs/frozen.yaml", action="train", n=12, method=method, seed=seed)
    runner.execute("configs/frozen.yaml", action="evaluate", n=12)
    runner.execute("configs/frozen.yaml", action="finalize")
    assert sum(event[0] == "train" for event in events) == 12
    assert sum(event[0] == "evaluate" for event in events) == 12
    assert events[-1] == ("finalize",)


def test_failed_slots_never_retrained_and_revision_mismatch_refused(pipeline):
    runner, events, provenance = pipeline
    directory = runner.reserve_attempt(runner.ROOT / "results/raw/n06/vanilla/seed_0")
    state = {"status": "failed", "provenance": provenance}
    atomic_json_dump(directory / "run.json", state)
    runner.execute("configs/frozen.yaml", True, action="train", n=6, method="vanilla", seed=0)
    assert not events
    state["provenance"] = {**provenance, "git_sha": "f"*40}
    atomic_json_dump(directory / "run.json", state)
    with pytest.raises(RuntimeError, match="different code/config"):
        runner.execute("configs/frozen.yaml", action="train", n=6, method="vanilla", seed=0)


def test_status_uses_no_cuda_or_heldout_metrics(pipeline, monkeypatch, capsys):
    runner, events, provenance = pipeline
    directory = runner.reserve_attempt(runner.ROOT / "results/raw/n06/rff/seed_2")
    atomic_json_dump(directory / "run.json", {"status": "running", "updates_completed": 0, "provenance": provenance})
    (directory / "training_log.csv").write_text("update,total_loss\n100,1.0\n200,0.9\n", encoding="utf-8")
    (directory / "metrics.json").write_text("invalid held-out JSON must not be read", encoding="utf-8")
    monkeypatch.setattr(runner, "runtime_record", lambda *args: pytest.fail("status requested CUDA/provenance"))
    runner.execute("configs/frozen.yaml", action="status")
    output = capsys.readouterr().out
    assert "200/8000" in output and "Terminal training slots: 0/12" in output
    assert not events


def test_selectors_cannot_change_frozen_workload(pipeline):
    runner, _, _ = pipeline
    for args in (dict(action="train", n=6, method="rff", seed=3),
                 dict(action="train", n=2, method="vanilla", seed=0),
                 dict(action="all", n=6), dict(action="evaluate", n=6, seed=0)):
        with pytest.raises(ValueError):
            runner.execute("configs/frozen.yaml", **args)


def test_interrupted_slot_requires_explicit_recovery_and_preserves_attempt(pipeline):
    runner, events, provenance = pipeline
    root = runner.ROOT / "results/raw/n06/rff/seed_1"
    first = runner.reserve_attempt(root)
    atomic_json_dump(first / "run.json", {"status": "infrastructure_interrupted", "provenance": provenance})
    original = (first / "run.json").read_bytes()
    with pytest.raises(RuntimeError, match="explicit --recover"):
        runner.execute("configs/frozen.yaml", action="train", n=6, method="rff", seed=1)
    runner.execute("configs/frozen.yaml", True, action="train", n=6, method="rff", seed=1)
    assert events == [("train", 6, "rff", 1)]
    assert (first / "run.json").read_bytes() == original
    assert (root / "attempt_001/run.json").exists()


def test_live_progress_tracks_tiny_nonprotected_updates(tmp_path, capsys):
    from src.train import train_run
    config = validate()
    config["protocol"]["k"] = 3.0
    config["data"]["interior_points"] = 24
    config["data"]["boundary_points_per_side"] = 8
    config["training"]["updates"] = 3
    config["training"]["log_every"] = 1
    state = train_run(config, 2, "vanilla", 99173, tmp_path, {}, smoke=True)
    output = capsys.readouterr().out
    assert state["status"] == "completed" and state["updates_completed"] == 3
    assert "100.0%" in output and "ETA=" in output and "loss=" in output
    assert (tmp_path / "final_model.pt").exists()
    assert not (tmp_path / "metrics.json").exists()


def test_thin_notebook_has_all_frozen_slots_in_order_and_no_saved_outputs():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "colab/run_protected.ipynb").read_text(encoding="utf-8"))
    actions = []
    for cell in notebook["cells"]:
        assert cell.get("id")
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None and not cell["outputs"]
        source = "".join(cell["source"])
        tree = ast.parse(source)
        compile(source, "launcher_cell", "exec")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_stage":
                actions.append(tuple(ast.literal_eval(arg) for arg in node.args))
    expected = [("--action", "status")]
    for n in (6, 12):
        for method in ("vanilla", "rff"):
            for seed in (0,1,2):
                expected.append(("--action", "train", "--n", str(n), "--method", method, "--seed", str(seed)))
        expected.append(("--action", "evaluate", "--n", str(n)))
    expected += [("--action", "finalize"), ("--action", "status")]
    assert actions == expected
