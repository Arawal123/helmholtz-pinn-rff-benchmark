"""Read-only monitoring fixtures; the original protected pipeline is never run."""
import ast
import json
from pathlib import Path


def notebook():
    path = Path(__file__).resolve().parents[1] / "colab/resume_original.ipynb"
    return json.loads(path.read_text(encoding="utf-8"))


def function_scope():
    tree = ast.parse("".join(notebook()["cells"][2]["source"]))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "training_snapshot")
    import csv
    scope = {"json": json, "csv": csv}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "snapshot", "exec"), scope)
    return scope


def test_legacy_progress_uses_csv_when_running_status_update_is_stale(tmp_path):
    scope = function_scope()
    attempt = tmp_path / "results/raw/n12/vanilla/seed_0/attempt_000"
    attempt.mkdir(parents=True)
    (attempt / "run.json").write_text(json.dumps({"status": "running", "updates_completed": 0}))
    (attempt / "training_log.csv").write_text("update,total_loss\n1,2.5\n100,1.5\n200,0.5\n")
    # These files must never be opened by the training monitor.
    (attempt / "metrics.json").write_text("invalid JSON")
    (attempt / "predictions.npy").write_bytes(b"invalid numpy")
    rows = scope["training_snapshot"](tmp_path)
    row = next(row for row in rows if row[:3] == (12, "vanilla", 0))
    assert row[3:6] == ("running", 200, "0.5")
    assert len(rows) == 12


def test_legacy_monitor_uses_new_attempt_and_keeps_old_bytes(tmp_path):
    scope = function_scope()
    root = tmp_path / "results/raw/n12/vanilla/seed_0"
    for index, update, status in ((0, 4000, "infrastructure_interrupted"), (1, 1, "running")):
        attempt = root / f"attempt_{index:03d}"
        attempt.mkdir(parents=True)
        (attempt / "run.json").write_text(json.dumps({"status": status, "updates_completed": 0}))
        (attempt / "training_log.csv").write_text(f"update,total_loss\n{update},1.0\n")
    original = (root / "attempt_000/training_log.csv").read_bytes()
    row = next(row for row in scope["training_snapshot"](tmp_path) if row[:3] == (12, "vanilla", 0))
    assert row[4] == 1 and row[6].endswith("attempt_001")
    assert (root / "attempt_000/training_log.csv").read_bytes() == original


def test_legacy_launcher_pins_original_and_invokes_only_original_entrypoint():
    n = notebook()
    for cell in n["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
            compile("".join(cell["source"]), "legacy_cell", "exec")
    first = "".join(n["cells"][1]["source"])
    assert 'COMMIT_SHA = "3258b1d47baa734fea102d7f8ea38fdd626d8174"' in first
    assert "RECOVER_INFRASTRUCTURE = False" in first
    resume = "".join(n["cells"][6]["source"])
    tree = ast.parse(resume)
    command = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "command" for t in node.targets))
    assert [ast.literal_eval(arg) for arg in command.elts[1:]] == [
        "-u", "scripts/run_frozen.py", "--config", "configs/frozen.yaml"]
    assert 'command.append("--recover-infrastructure")' in resume
    assert "--action" not in resume and "train_run" not in resume
    assert 'assert not live_runner_processes()' in resume


def test_live_process_check_detects_relative_and_absolute_runner_paths():
    import re
    from types import SimpleNamespace
    tree = ast.parse("".join(notebook()["cells"][2]["source"]))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "live_runner_processes")
    ps = ("PID COMMAND\n101 python -u scripts/run_frozen.py --config configs/frozen.yaml\n"
          "102 python /content/drive/MyDrive/repository/scripts/run_frozen.py\n"
          "103 python -m ipykernel_launcher\n104 python unrelated_script.py\n")
    scope = {"re": re, "subprocess": SimpleNamespace(check_output=lambda *args, **kwargs: ps)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "process_check", "exec"), scope)
    processes = scope["live_runner_processes"]()
    assert len(processes) == 2
    assert processes[0].startswith("101 ") and processes[1].startswith("102 ")
