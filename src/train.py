"""Frozen full-batch Adam training. Only the final update is retained."""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import as_tensor, boundary_points, generate_sobol_points, load_frozen_sobol, load_rff_matrix
from .models import build_model, count_trainable_parameters
from .pde import pinn_loss
from .utils import atomic_json_dump, set_determinism, utc_now


def learning_rate(update: int, updates: int = 8000) -> float:
    if not 1 <= update <= updates:
        raise ValueError("Update outside frozen schedule")
    return .0001 + (.001 - .0001) * (1 + math.cos(math.pi * (update - 1) / (updates - 1))) / 2


def train_run(config: dict, n: int, method: str, seed: int, directory: Path, provenance: dict,
              *, smoke: bool = False) -> dict:
    if not smoke and (n not in (6, 12) or method not in ("vanilla", "rff") or seed not in (0, 1, 2)):
        raise ValueError("Run outside frozen workload")
    set_determinism(seed)
    device = torch.device("cpu" if smoke else "cuda:0")
    if not smoke and not torch.cuda.is_available():
        raise RuntimeError("Protected runs require CUDA")
    count = config["data"]["interior_points"]
    points = load_frozen_sobol(config, seed) if not smoke else generate_sobol_points(seed, count)
    interior = as_tensor(points, device, requires_grad=True)
    boundary = as_tensor(boundary_points(config["data"]["boundary_points_per_side"]), device)
    matrix = load_rff_matrix(config) if method == "rff" else None
    model = build_model(method, matrix).to(device)
    parameters = count_trainable_parameters(model)
    assert parameters == (12737 if method == "vanilla" else 12871)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    updates = config["training"]["updates"]
    status = {"n": n, "condition": "primary" if n == 6 else "preregistered_stress_failure",
              "method": method, "seed": seed, "status": "running", "evaluation_status": "pending_condition_gate",
              "parameter_count": parameters, "updates_planned": updates, "updates_completed": 0,
              "started_at_utc": utc_now(), "provenance": provenance,
              "label": "SMOKE / NON-PROTECTED / NOT RESEARCH EVIDENCE" if smoke else "PROTECTED"}
    atomic_json_dump(directory / "run.json", status)
    atomic_json_dump(directory / "resolved_config.json", config)
    if not smoke:
        assert count == 2048 and len(boundary) == 512 and updates == 8000
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    log_path = directory / "training_log.csv"
    with log_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["update", "lr", "total_loss", "residual_mse", "boundary_mse"])
        writer.writeheader()
        for update in range(1, updates + 1):
            lr = learning_rate(update, updates)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            interior.grad = None
            loss, parts = pinn_loss(model, interior, boundary, n, config["protocol"]["k"])
            if not bool(torch.isfinite(loss).item()):
                status.update(status="failed", failure_step=update, failure_reason="nonfinite_loss")
                break
            loss.backward()
            if not all(bool(torch.isfinite(p.grad).all().item()) for p in model.parameters() if p.grad is not None):
                status.update(status="failed", failure_step=update, failure_reason="nonfinite_gradient")
                break
            optimizer.step()
            if not all(bool(torch.isfinite(p).all().item()) for p in model.parameters()):
                status.update(status="failed", failure_step=update, failure_reason="nonfinite_parameter")
                break
            status["updates_completed"] = update
            if update == 1 or update % config["training"]["log_every"] == 0 or update == updates:
                writer.writerow({"update": update, "lr": lr, "total_loss": loss.item(),
                                 "residual_mse": parts["residual_mse"].item(),
                                 "boundary_mse": parts["boundary_mse"].item()})
                handle.flush()
                elapsed = time.perf_counter() - started
                eta = elapsed * (updates - update) / update
                print(f"[n={n} {method} seed={seed}] {update:4d}/{updates} "
                      f"({100*update/updates:5.1f}%) loss={loss.item():.6g} lr={lr:.6g} "
                      f"elapsed={elapsed/60:.1f}min ETA={eta/60:.1f}min", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    status["wall_seconds_training"] = time.perf_counter() - started
    status["ended_at_utc"] = utc_now()
    if status["status"] == "running":
        status["status"] = "completed"
        torch.save(model.state_dict(), directory / "final_model.pt")
    else:
        status["evaluation_status"] = "not_applicable_failed_training"
    atomic_json_dump(directory / "run.json", status)
    return status
