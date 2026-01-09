"""Example custom model builder for ChaosML."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


class CustomLSTM(nn.Module):
    def __init__(self, input_dim: int, horizon: int, units: int, depth: int, dropout: float):
        super().__init__()
        self.horizon = horizon
        self.input_dim = input_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=units,
            num_layers=depth,
            batch_first=True,
            dropout=dropout if depth > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Linear(units, horizon * input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        flat = self.head(last)
        return flat.view(x.size(0), self.horizon, self.input_dim)


def build_model(model_cfg: Dict, input_shape: Tuple[int, int], output_dim: int, horizon: int) -> nn.Module:
    units = int(model_cfg.get("units", 128))
    depth = int(model_cfg.get("depth", 2))
    dropout = float(model_cfg.get("dropout", 0.0))
    return CustomLSTM(output_dim, horizon, units, depth, dropout)
