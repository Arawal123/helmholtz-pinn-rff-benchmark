"""Held-out evaluation, invoked only after a whole condition is terminal."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import as_tensor, boundary_points, heldout_grid, load_rff_matrix
from .models import build_model
from .pde import exact_solution, normalized_residual
from .utils import atomic_json_dump, utc_now


def evaluate_run(config: dict, n: int, method: str, directory: Path, *, smoke: bool = False) -> dict:
    device = torch.device("cpu" if smoke else "cuda:0")
    model = build_model(method, load_rff_matrix(config) if method == "rff" else None).to(device)
    model.load_state_dict(torch.load(directory / "final_model.pt", map_location=device, weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    size = config["data"]["heldout_grid_size"]
    if not smoke and size != 257:
        raise ValueError("Protected held-out grid must be 257x257")
    axis, grid, _ = heldout_grid(size)
    pred = np.empty((len(grid),), dtype=np.float32)
    sum_res_sq = 0.0
    started = time.perf_counter()
    for start in range(0, len(grid), 512):
        xy = as_tensor(grid[start:start + 512], device, requires_grad=True)
        with torch.no_grad():
            pred[start:start + len(xy)] = model(xy).flatten().cpu().numpy()
        residual = normalized_residual(model, xy, n, config["protocol"]["k"])
        sum_res_sq += float(residual.detach().square().sum().item())
    with torch.no_grad():
        exact = exact_solution(as_tensor(grid, device), n).flatten().cpu().numpy()
        boundary = model(as_tensor(boundary_points(), device)).flatten()
        boundary_rmse = math.sqrt(float(boundary.square().mean().item()))
    diff = pred.astype(np.float64) - exact.astype(np.float64)
    metrics = {"relative_l2": float(np.linalg.norm(diff) / np.linalg.norm(exact)),
               "linf": float(np.max(np.abs(diff))),
               "normalized_residual_rms": math.sqrt(sum_res_sq / len(grid)),
               "boundary_rmse": boundary_rmse,
               "evaluation_wall_seconds": time.perf_counter() - started,
               "evaluated_at_utc": utc_now(), "grid_size": size,
               "label": "SMOKE / NON-PROTECTED / NOT RESEARCH EVIDENCE" if smoke else "PROTECTED"}
    if not all(math.isfinite(value) for key, value in metrics.items() if isinstance(value, float)):
        raise FloatingPointError("Nonfinite held-out metric")
    prediction_path = directory / "grid_predictions.npy"
    shaped = pred.reshape(size, size)
    if prediction_path.exists():
        if not np.array_equal(np.load(prediction_path, allow_pickle=False), shaped):
            raise RuntimeError("Interrupted evaluation predictions differ; preserving prior raw file")
    else:
        with prediction_path.open("xb") as handle:
            np.save(handle, shaped)
    atomic_json_dump(directory / "metrics.json", metrics)
    return metrics
