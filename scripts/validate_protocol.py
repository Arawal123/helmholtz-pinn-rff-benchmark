"""Assert the approved protocol and verify committed frozen assets."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.data import boundary_points, heldout_grid, load_frozen_sobol, load_rff_matrix
from src.models import build_model, count_trainable_parameters
from src.utils import load_yaml

EXPECTED_SOBOL_SHA256 = {
    0: "f5fc079758da5519dbbfc48084a7f85de8db052a05c64ef250f76bffb4665c48",
    1: "e7332b2a7bb1da1cf6e78ab82ebae5990938322016388d8894bd7d1f2ec43c56",
    2: "9ddf3c59bd5ae415262906a8868540782908fdc88d1b0501684bd84ff78eee7b",
}
EXPECTED_RFF_SHA256 = "76130b8c82524338e339d86ec525d3476ffcd3ea889ca93dc7db51c578d2ed63"


def validate(config_path: str = "configs/frozen.yaml") -> dict:
    if not __debug__:
        raise RuntimeError("Protocol validation cannot run with Python optimizations (-O)")
    path = (ROOT / config_path).resolve()
    if path != (ROOT / "configs/frozen.yaml").resolve():
        raise ValueError("Protected validation accepts only configs/frozen.yaml")
    c = load_yaml(path)
    p, d, m, t, e, x = (c[key] for key in ("protocol", "data", "model", "training", "evaluation", "execution"))
    assert p == dict(name="ryan_pinn_failure_repair_v1", status="FROZEN_APPROVED", primary_n=6,
                     stress_n=12, stress_label="preregistered_stress_failure_condition", k=20.0,
                     domain=[-1.0, 1.0], seeds=[0, 1, 2], precision="float32")
    assert d["interior_points"] == 2048 and d["interior_sampler"] == "scrambled_sobol"
    assert d["boundary_points"] == 512 and d["boundary_points_per_side"] == 128
    assert d["heldout_grid_size"] == 257 and set(d["frozen_sobol_assets"]) == {0, 1, 2}
    assert m["vanilla"] == dict(layers=[2,64,64,64,64,1], activation="tanh",
                                weight_initialization="xavier_uniform", bias_initialization="zeros")
    assert m["rff"] == dict(layers=[64,55,55,55,55,1], activation="tanh",
                            weight_initialization="xavier_uniform", bias_initialization="zeros",
                            feature_formula="[sin(2*pi*B*x), cos(2*pi*B*x)]", frequencies=32,
                            bandwidth=3.0, seed=2026, matrix_path="artifacts/rff/B_seed_2026.npy",
                            matrix_sha256=EXPECTED_RFF_SHA256)
    assert t == dict(optimizer="Adam", full_batch=True, updates=8000, initial_lr=0.001,
                     final_lr=0.0001, scheduler="cosine", adaptive_loss_weights=False,
                     resampling=False, curriculum=False, early_stopping=False,
                     best_checkpoint=False, retries_on_numerical_failure=0, log_every=100)
    assert e == dict(metrics=["relative_l2", "linf", "normalized_residual_rms", "boundary_rmse"],
                     delayed_until_condition_terminal=True, selection_use="forbidden")
    assert x == dict(protected_runs=12, require_cuda=True, require_clean_git=True,
                     refuse_raw_overwrite=True, expected_accelerator="T4-class or better",
                     official_output_root="results/raw")
    assert len(boundary_points()) == 512 and len(heldout_grid()[1]) == 257**2
    for seed in p["seeds"]:
        assert d["frozen_sobol_assets"][seed]["path"] == f"data/frozen/sobol_seed_{seed}.npy"
        assert d["frozen_sobol_assets"][seed]["sha256"] == EXPECTED_SOBOL_SHA256[seed]
        points = load_frozen_sobol(c, seed)
        assert points.shape == (2048, 2) and points.dtype == np.float32
    matrix = load_rff_matrix(c)
    assert matrix.shape == (32, 2)
    vanilla = count_trainable_parameters(build_model("vanilla"))
    rff = count_trainable_parameters(build_model("rff", matrix))
    assert vanilla == 12737 and rff == 12871 and abs(rff / vanilla - 1) < .02
    return c


if __name__ == "__main__":
    validate()
    print("Frozen protocol and asset hashes: VALID")
