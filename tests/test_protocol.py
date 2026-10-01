import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.aggregate_results import collect, effects_and_summary, make_report
from scripts.run_frozen import reserve_attempt, terminal
from scripts.validate_protocol import validate
from src.data import boundary_points, generate_sobol_points, heldout_grid, load_frozen_sobol, load_rff_matrix
from src.models import build_model, count_trainable_parameters
from src.pde import exact_solution, forcing, normalized_residual, pinn_loss, residual_scale
from src.train import learning_rate, train_run
from src.utils import sha256_file


def test_frozen_protocol_assets_models():
    c = validate()
    assert c["protocol"]["seeds"] == [0, 1, 2]
    metadata = json.loads((Path(__file__).resolve().parents[1] / "data/frozen/generation.json").read_text(encoding="utf-8"))
    for seed in (0, 1, 2):
        points = load_frozen_sobol(c, seed)
        if metadata["torch_version"] == torch.__version__:
            assert np.array_equal(points, generate_sobol_points(seed))
        assert sha256_file(c["data"]["frozen_sobol_assets"][seed]["path"]) == c["data"]["frozen_sobol_assets"][seed]["sha256"]
    matrix = load_rff_matrix(c)
    if metadata["torch_version"] == torch.__version__:
        regenerated = (3 * torch.randn((32, 2), generator=torch.Generator(device="cpu").manual_seed(2026), dtype=torch.float32)).numpy()
        assert np.array_equal(matrix, regenerated)
    assert count_trainable_parameters(build_model("vanilla")) == 12737
    rff = build_model("rff", matrix)
    assert count_trainable_parameters(rff) == 12871
    assert rff.features(torch.zeros((2, 2))).shape == (2, 64)
    assert not any(name == "B" for name, _ in rff.named_parameters())


def test_geometry_and_pde():
    boundary = boundary_points()
    assert boundary.shape == (512, 2)
    assert np.unique(boundary, axis=0).shape[0] == 512
    assert heldout_grid()[1].shape == (257**2, 2)
    xy = torch.tensor(boundary, dtype=torch.float64)
    for n in (6, 12):
        assert exact_solution(xy, n).abs().max().item() < 1e-13
        interior = torch.tensor([[.13, -.27], [.22, .36]], dtype=torch.float64, requires_grad=True)
        exact = exact_solution(interior, n)
        grad = torch.autograd.grad(exact.sum(), interior, create_graph=True)[0]
        lap = sum(torch.autograd.grad(grad[:, i].sum(), interior, retain_graph=True)[0][:, i:i+1] for i in (0, 1))
        assert torch.allclose(lap + 400*exact, forcing(interior, n), atol=1e-10)
        assert residual_scale(n) == max(1, abs(400 - 2*(n*math.pi)**2))


def test_loss_and_schedule():
    class Quadratic(torch.nn.Module):
        def forward(self, x):
            return x[:, :1].square() + x[:, 1:2].square()
    interior = torch.tensor([[.17, .23]], requires_grad=True)
    boundary = torch.tensor([[1., 0.]])
    loss, parts = pinn_loss(Quadratic(), interior, boundary, 6)
    prediction = interior.square().sum(dim=1, keepdim=True)
    expected = ((4 + 400*prediction - forcing(interior, 6))/residual_scale(6)).square().mean() + 1
    assert torch.allclose(loss, expected) and torch.allclose(parts["boundary_mse"], torch.tensor(1.))
    assert learning_rate(1) == pytest.approx(.001)
    assert learning_rate(8000) == pytest.approx(.0001)
    assert learning_rate(4000) > learning_rate(4001)


def test_failure_record_and_no_retry(tmp_path, monkeypatch):
    from src import train as module
    c = validate()
    c["data"]["interior_points"] = 8
    c["data"]["boundary_points_per_side"] = 2
    c["training"]["updates"] = 3
    def bad_loss(*args, **kwargs):
        return torch.tensor(float("nan")), {"residual_mse": torch.tensor(float("nan")), "boundary_mse": torch.tensor(0.)}
    monkeypatch.setattr(module, "pinn_loss", bad_loss)
    state = train_run(c, 2, "vanilla", 5, tmp_path, {}, smoke=True)
    assert state["status"] == "failed" and state["failure_step"] == 1
    assert not (tmp_path / "final_model.pt").exists()
    assert (tmp_path / "training_log.csv").exists()


def test_attempts_overwrite_safe_and_pairing(tmp_path):
    first = reserve_attempt(tmp_path)
    assert first.name == "attempt_000"
    second = reserve_attempt(tmp_path)
    assert second.name == "attempt_001" and first.exists()
    assert not terminal({"status": "infrastructure_interrupted"})
    assert terminal({"status": "failed"})
    rows = [dict(n=6, method=m, seed=s, status="completed", relative_l2=v)
            for s in (0, 1, 2) for m, v in (("vanilla", 2.), ("rff", 1.))]
    rows += [dict(n=12, method=m, seed=s, status="failed") for s in (0,1,2) for m in ("vanilla", "rff")]
    effects, summary = effects_and_summary(rows)
    assert next(x for x in effects if x["n"] == 6 and x["seed"] == 0 and x["metric"] == "relative_l2")["percentage_improvement"] == 50
    assert next(x for x in effects if x["n"] == 12)["status"] == "unavailable_failure_or_pending"


def test_pending_report_when_no_results():
    from scripts.aggregate_results import make_report
    make_report([], [], [])
    report = Path(__file__).resolve().parents[1] / "report/methods_results.md"
    assert "PENDING PROTECTED RUN" in report.read_text(encoding="utf-8")


def test_git_provenance_capture(tmp_path, monkeypatch):
    from src import provenance
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "evidence.txt").write_text("test", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "evidence.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "test"], check=True)
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    state = provenance.git_state()
    assert len(state["git_sha"]) == 40 and state["git_status"] == ""
    (tmp_path / "untracked.txt").write_text("x", encoding="utf-8")
    assert "untracked.txt" in provenance.git_state()["git_status"]


def test_condition_gate_before_heldout(tmp_path, monkeypatch):
    from scripts import run_frozen, aggregate_results
    c = validate()
    monkeypatch.setattr(run_frozen, "ROOT", tmp_path)
    monkeypatch.setattr(run_frozen, "validate", lambda _: c)
    provenance = {"git_sha": "a"*40, "config_sha256": "b"*64,
                  "sobol_sha256": {str(s): c["data"]["frozen_sobol_assets"][s]["sha256"] for s in (0,1,2)},
                  "rff_sha256": c["model"]["rff"]["matrix_sha256"]}
    monkeypatch.setattr(run_frozen, "runtime_record", lambda *args: provenance)
    def fake_train(config, n, method, seed, directory, provenance):
        state = {"status": "completed", "provenance": provenance}
        (directory / "run.json").write_text(json.dumps(state), encoding="utf-8")
        return state
    gates = []
    def fake_evaluate(config, n, method, directory):
        states = list((tmp_path / f"results/raw/n{n:02d}").glob("*/seed_*/attempt_*/run.json"))
        assert len(states) == 6
        assert all(json.loads(path.read_text(encoding="utf-8"))["status"] == "completed" for path in states)
        gates.append(n)
        (directory / "metrics.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run_frozen, "train_run", fake_train)
    monkeypatch.setattr(run_frozen, "evaluate_run", fake_evaluate)
    monkeypatch.setattr(aggregate_results, "aggregate", lambda *args: None)
    run_frozen.execute("configs/frozen.yaml")
    assert gates == [6]*6 + [12]*6
    run_frozen.execute("configs/frozen.yaml")
    assert len(list((tmp_path / "results/raw").glob("n*/**/attempt_*"))) == 12


def test_processed_pipeline_retains_failures(tmp_path, monkeypatch):
    from scripts import aggregate_results
    monkeypatch.setattr(aggregate_results, "ROOT", tmp_path)
    for n in (6, 12):
        for method in ("vanilla", "rff"):
            for seed in (0, 1, 2):
                path = tmp_path / f"results/raw/n{n:02d}/{method}/seed_{seed}/attempt_000"
                path.mkdir(parents=True)
                success = n == 6 and seed == 0
                state = {"status": "completed" if success else "failed", "evaluation_status": "completed" if success else "not_applicable_failed_training",
                         "provenance": {"git_sha": "a"*40, "accelerator": "test GPU", "config_sha256": "b"*64,
                                        "sobol_sha256": {str(seed): "c"*64}, "rff_sha256": "d"*64},
                         "parameter_count": 12737 if method == "vanilla" else 12871,
                         "wall_seconds_training": 1., "failure_reason": "nonfinite_loss" if not success else ""}
                (path / "run.json").write_text(json.dumps(state), encoding="utf-8")
                (path / "training_log.csv").write_text("update,lr,total_loss,residual_mse,boundary_mse\n1,0.001,1,0.8,0.2\n", encoding="utf-8")
                if success:
                    (path / "metrics.json").write_text(json.dumps({"relative_l2": .5 if method == "vanilla" else .4,
                        "linf": 1., "normalized_residual_rms": .2, "boundary_rmse": .1, "evaluation_wall_seconds": .5}), encoding="utf-8")
                    np.save(path / "grid_predictions.npy", np.zeros((257,257), dtype=np.float32))
    aggregate_results.aggregate(provenance={"git_sha": "a"*40, "accelerator": "test GPU", "torch": "test", "cuda_runtime": "test"})
    processed = tmp_path / "results/processed"
    for name in ("per_seed_results.csv", "paired_effects.csv", "summary.csv", "compute_record.csv", "run_manifest.csv"):
        assert (processed / name).exists()
    with (processed / "per_seed_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(__import__("csv").DictReader(handle))
    assert len(rows) == 12 and sum(row["status"] == "failed" for row in rows) == 10
    assert (tmp_path / "figures/n06_rff_seed_0_error.png").exists()
    report = (tmp_path / "report/methods_results.md").read_text(encoding="utf-8")
    assert "PROTECTED RUN COMPLETE" in report and "Scientific failures" in report
