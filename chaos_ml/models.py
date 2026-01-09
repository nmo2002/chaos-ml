"""PyTorch model builders for chaotic system forecasting."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import nn


def positional_encoding(length: int, d_model: int) -> torch.Tensor:
    positions = np.arange(length)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
    angles = positions * angle_rates
    pe = np.zeros_like(angles, dtype=np.float32)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return torch.tensor(pe)


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, units: int, depth: int, dropout: float, bidirectional: bool):
        super().__init__()
        self.horizon = horizon
        self.input_dim = input_dim
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=units,
            num_layers=depth,
            batch_first=True,
            dropout=dropout if depth > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = units * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, horizon * input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        flat = self.head(last)
        return flat.view(x.size(0), self.horizon, self.input_dim)


class TransformerForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, d_model: int, num_layers: int, num_heads: int, dff: int, dropout: float, window: int):
        super().__init__()
        self.horizon = horizon
        self.input_dim = input_dim
        if num_heads < 1:
            num_heads = 1
        if d_model % num_heads != 0:
            num_heads = 1
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.register_buffer("pe", positional_encoding(window, d_model))
        self.head = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x + self.pe[: x.size(1)].unsqueeze(0)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        flat = self.head(pooled)
        return flat.view(x.size(0), self.horizon, self.input_dim)


class EncoderDecoderForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, units: int, depth: int, dropout: float, attention: bool):
        super().__init__()
        self.horizon = horizon
        self.input_dim = input_dim
        self.attention = attention
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=units,
            num_layers=depth,
            batch_first=True,
            dropout=dropout if depth > 1 else 0.0,
            bidirectional=False,
        )
        self.decoder = nn.LSTM(
            input_size=units,
            hidden_size=units,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(units * (2 if attention else 1), input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out, (h, c) = self.encoder(x)
        repeated = h[-1].unsqueeze(1).repeat(1, self.horizon, 1)
        dec_out, _ = self.decoder(repeated, (h[-1:].contiguous(), c[-1:].contiguous()))
        if self.attention:
            scores = torch.matmul(dec_out, enc_out.transpose(1, 2))
            weights = torch.softmax(scores, dim=-1)
            context = torch.matmul(weights, enc_out)
            dec_out = torch.cat([dec_out, context], dim=-1)
        return self.out(dec_out)


def build_model(model_name: str, input_shape, output_dim: int, horizon: int, config: Dict):
    model_name = model_name.lower()
    input_dim = output_dim
    window = input_shape[0]
    if model_name == "lstm":
        return LSTMForecaster(
            input_dim=input_dim,
            horizon=horizon,
            units=config.get("units", 256),
            depth=config.get("depth", 3),
            dropout=config.get("dropout", 0.0),
            bidirectional=False,
        )
    if model_name == "bilstm":
        return LSTMForecaster(
            input_dim=input_dim,
            horizon=horizon,
            units=config.get("units", 256),
            depth=config.get("depth", 3),
            dropout=config.get("dropout", 0.0),
            bidirectional=True,
        )
    if model_name == "transformer":
        return TransformerForecaster(
            input_dim=input_dim,
            horizon=horizon,
            d_model=config.get("d_model", 64),
            num_layers=config.get("num_layers", 4),
            num_heads=config.get("num_heads", 4),
            dff=config.get("dff", 128),
            dropout=config.get("dropout", 0.1),
            window=window,
        )
    if model_name == "encoder_decoder":
        return EncoderDecoderForecaster(
            input_dim=input_dim,
            horizon=horizon,
            units=config.get("units", 128),
            depth=config.get("depth", 2),
            dropout=config.get("dropout", 0.0),
            attention=config.get("attention", True),
        )
    raise ValueError(f"Unknown model '{model_name}'.")
