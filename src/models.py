from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def _initialize_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class MLP(nn.Module):
    def __init__(self, layers: list[int]):
        super().__init__()
        blocks: list[nn.Module] = []
        for input_width, output_width in zip(layers[:-2], layers[1:-1]):
            blocks.extend((nn.Linear(input_width, output_width), nn.Tanh()))
        blocks.append(nn.Linear(layers[-2], layers[-1]))
        self.network = nn.Sequential(*blocks)
        self.apply(_initialize_linear)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class VanillaPINN(MLP):
    def __init__(self):
        super().__init__([2, 64, 64, 64, 64, 1])


class RFFPINN(nn.Module):
    def __init__(self, matrix: np.ndarray | torch.Tensor):
        super().__init__()
        frozen = torch.as_tensor(matrix, dtype=torch.float32).clone().detach()
        if frozen.shape != (32, 2):
            raise ValueError("RFF matrix must have shape (32,2)")
        self.register_buffer("B", frozen, persistent=True)
        self.mlp = MLP([64, 55, 55, 55, 55, 1])

    def features(self, xy: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * math.pi * (xy @ self.B.T)
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.features(xy))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_model(method: str, rff_matrix: np.ndarray | None = None) -> nn.Module:
    if method == "vanilla":
        return VanillaPINN()
    if method == "rff":
        if rff_matrix is None:
            raise ValueError("rff_matrix is required for the RFF model")
        return RFFPINN(rff_matrix)
    raise ValueError(f"Unknown method: {method}")
