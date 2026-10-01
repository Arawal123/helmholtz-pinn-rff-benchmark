from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .utils import resolve_repo_path, sha256_file


def generate_sobol_points(seed: int, count: int = 2048) -> np.ndarray:
    engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=int(seed))
    unit = engine.draw(count).cpu().numpy()
    points = (2.0 * unit - 1.0).astype(np.float32, copy=False)
    if points.shape != (count, 2) or not np.all((points > -1.0) & (points < 1.0)):
        raise RuntimeError("Scrambled Sobol points must lie strictly inside (-1,1)^2")
    return points


def load_frozen_sobol(config: dict, seed: int, verify_hash: bool = True) -> np.ndarray:
    record = config["data"]["frozen_sobol_assets"][int(seed)]
    path = resolve_repo_path(record["path"])
    if verify_hash and sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"Frozen Sobol hash mismatch for seed {seed}: {path}")
    points = np.load(path, allow_pickle=False)
    expected = int(config["data"]["interior_points"])
    if points.dtype != np.float32 or points.shape != (expected, 2):
        raise RuntimeError(f"Invalid frozen Sobol asset for seed {seed}")
    if not np.all((points > -1.0) & (points < 1.0)):
        raise RuntimeError(f"Sobol asset contains boundary/out-of-domain values for seed {seed}")
    return points


def boundary_points(points_per_side: int = 128) -> np.ndarray:
    if points_per_side < 2:
        raise ValueError("points_per_side must be at least 2")
    t = np.linspace(-1.0, 1.0, points_per_side, endpoint=False, dtype=np.float32)
    bottom = np.column_stack((t, np.full_like(t, -1.0)))
    right = np.column_stack((np.full_like(t, 1.0), t))
    top = np.column_stack((-t, np.full_like(t, 1.0)))
    left = np.column_stack((np.full_like(t, -1.0), -t))
    points = np.concatenate((bottom, right, top, left), axis=0).astype(np.float32)
    if np.unique(points, axis=0).shape[0] != 4 * points_per_side:
        raise RuntimeError("Boundary construction duplicated a corner")
    return points


def heldout_grid(size: int = 257) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    return axis, points, (xx, yy)


def load_rff_matrix(config: dict, verify_hash: bool = True) -> np.ndarray:
    record = config["model"]["rff"]
    path = resolve_repo_path(record["matrix_path"])
    if verify_hash and sha256_file(path) != record["matrix_sha256"]:
        raise RuntimeError(f"Frozen RFF matrix hash mismatch: {path}")
    matrix = np.load(path, allow_pickle=False)
    if matrix.dtype != np.float32 or matrix.shape != (32, 2):
        raise RuntimeError("Frozen RFF matrix must be float32 with shape (32,2)")
    return matrix


def as_tensor(array: np.ndarray, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, device=device, requires_grad=requires_grad)

