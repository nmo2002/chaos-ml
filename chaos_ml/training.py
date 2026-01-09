"""Training and evaluation utilities (PyTorch)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .datasets import inverse_scale
from .metrics import summarize


@dataclass
class TrainResult:
    history: Dict
    metrics: Dict
    predictions: np.ndarray


def _to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=shuffle)


def fit_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    patience: int,
    learning_rate: float,
    out_dir: Path,
    device: torch.device,
    save_best: bool = True,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_loader = _to_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    has_val = len(x_val) and len(y_val)
    val_loader = _to_loader(x_val, y_val, batch_size=batch_size, shuffle=False) if has_val else None

    history = {"train_loss": [], "val_loss": []}
    best_loss = float("inf")
    bad_epochs = 0
    best_path = out_dir / "model.pt"

    for _ in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))
        history["train_loss"].append(train_loss)

        if has_val:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    val_losses.append(loss_fn(pred, yb).item())
            val_loss = float(np.mean(val_losses))
            history["val_loss"].append(val_loss)
            monitor = val_loss
        else:
            history["val_loss"].append(None)
            monitor = train_loss

        if monitor < best_loss:
            best_loss = monitor
            bad_epochs = 0
            if save_best:
                torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    history["best_loss"] = best_loss

    if save_best and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    return history


def predict_direct(model: nn.Module, x_test: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    x_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(x_t).cpu().numpy()
    return pred


def evaluate_and_save(
    out_dir: Path,
    scaler,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    history: Dict,
) -> TrainResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true_inv = inverse_scale(scaler, y_true)
    y_pred_inv = inverse_scale(scaler, y_pred)
    metrics = summarize(y_true_inv, y_pred_inv)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.save(out_dir / "y_true.npy", y_true_inv)
    np.save(out_dir / "y_pred.npy", y_pred_inv)

    return TrainResult(history=history, metrics=metrics, predictions=y_pred_inv)
