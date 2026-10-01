"""Fresh launcher starts in a separate persistent folder with every frozen stage."""
import ast
import json
from pathlib import Path


def test_fresh_launcher_has_fixed_sha_separate_folder_and_complete_frozen_sequence():
    root = Path(__file__).resolve().parents[1]
    n = json.loads((root / "colab/start_from_beginning.ipynb").read_text(encoding="utf-8"))
    actions = []
    first = "".join(n["cells"][1]["source"])
    assert 'COMMIT_SHA = "0677b40869fabf50035075e240aea3b98a0c7e05"' in first
    assert 'BASE = Path("/content/drive/MyDrive/ryan_pinn_trial_from_start_0677b40")' in first
    for cell in n["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None and not cell["outputs"]
        text = "".join(cell["source"])
        assert "PASTE_REVIEWED" not in text
        compile(text, "fresh_cell", "exec")
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_stage":
                actions.append(tuple(ast.literal_eval(arg) for arg in node.args))
    expected = [("--action", "status")]
    for n_value in (6, 12):
        for method in ("vanilla", "rff"):
            for seed in (0, 1, 2):
                expected.append(("--action", "train", "--n", str(n_value), "--method", method, "--seed", str(seed)))
        expected.append(("--action", "evaluate", "--n", str(n_value)))
    expected += [("--action", "finalize"), ("--action", "status")]
    assert actions == expected


def test_fresh_setup_records_restart_and_does_not_move_or_delete_prior_evidence():
    root = Path(__file__).resolve().parents[1]
    n = json.loads((root / "colab/start_from_beginning.ipynb").read_text(encoding="utf-8"))
    source = "".join(n["cells"][2]["source"])
    assert "User requested a fresh run from the beginning" in source
    tree = ast.parse(source)
    forbidden = {"move", "rmtree", "unlink", "remove", "rename", "replace"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden
