from __future__ import annotations

import math

import torch


def exact_solution(xy: torch.Tensor, n: int) -> torch.Tensor:
    return torch.sin(n * math.pi * xy[..., 0:1]) * torch.sin(n * math.pi * xy[..., 1:2])


def forcing(xy: torch.Tensor, n: int, k: float = 20.0) -> torch.Tensor:
    coefficient = k**2 - 2.0 * (n * math.pi) ** 2
    return coefficient * exact_solution(xy, n)


def residual_scale(n: int, k: float = 20.0) -> float:
    return max(1.0, abs(k**2 - 2.0 * (n * math.pi) ** 2))


def normalized_residual(model: torch.nn.Module, xy: torch.Tensor, n: int, k: float = 20.0) -> torch.Tensor:
    if not xy.requires_grad:
        raise ValueError("xy must require gradients for PDE residual evaluation")
    prediction = model(xy)
    first = torch.autograd.grad(
        prediction, xy, grad_outputs=torch.ones_like(prediction), create_graph=True, retain_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        first[:, 0:1], xy, grad_outputs=torch.ones_like(first[:, 0:1]), create_graph=True, retain_graph=True
    )[0][:, 0:1]
    u_yy = torch.autograd.grad(
        first[:, 1:2], xy, grad_outputs=torch.ones_like(first[:, 1:2]), create_graph=True, retain_graph=True
    )[0][:, 1:2]
    raw = u_xx + u_yy + k**2 * prediction - forcing(xy, n, k)
    return raw / residual_scale(n, k)


def pinn_loss(
    model: torch.nn.Module, interior: torch.Tensor, boundary: torch.Tensor, n: int, k: float = 20.0
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    residual = normalized_residual(model, interior, n=n, k=k)
    boundary_value = model(boundary)
    residual_mse = torch.mean(residual.square())
    boundary_mse = torch.mean(boundary_value.square())
    total = residual_mse + boundary_mse
    return total, {"residual_mse": residual_mse, "boundary_mse": boundary_mse}

