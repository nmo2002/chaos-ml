"""Dataset preparation: scaling, windowing, and splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class DatasetSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    scaler: object
    raw_train: np.ndarray
    raw_test: np.ndarray


def _get_scaler(name: str, feature_range: Tuple[float, float]):
    if name == "minmax":
        return MinMaxScaler(feature_range=feature_range)
    if name == "standard":
        return StandardScaler()
    raise ValueError("Scaler must be 'minmax' or 'standard'.")


def make_windows(
    series: np.ndarray,
    window: int,
    horizon: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if window < 1 or horizon < 1:
        raise ValueError("window and horizon must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    x, y = [], []
    end = len(series) - window - horizon + 1
    for i in range(0, end, stride):
        x.append(series[i : i + window])
        y.append(series[i + window : i + window + horizon])
    return np.asarray(x), np.asarray(y)


def train_val_test_split(
    series: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0 < train_ratio < 1) or not (0 <= val_ratio < 1):
        raise ValueError("train_ratio must be in (0,1), val_ratio in [0,1).")
    n = len(series)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return series[:train_end], series[train_end:val_end], series[val_end:]


def prepare_dataset(
    series: np.ndarray,
    window: int,
    horizon: int,
    stride: int,
    train_ratio: float,
    val_ratio: float,
    scaler_name: str,
    feature_range: Tuple[float, float],
) -> DatasetSplits:
    train_raw, val_raw, test_raw = train_val_test_split(series, train_ratio, val_ratio)

    scaler = _get_scaler(scaler_name, feature_range)
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw) if len(val_raw) else np.empty((0, train_raw.shape[1]))
    test_scaled = scaler.transform(test_raw) if len(test_raw) else np.empty((0, train_raw.shape[1]))

    x_train, y_train = make_windows(train_scaled, window, horizon, stride)
    x_val, y_val = make_windows(val_scaled, window, horizon, stride) if len(val_scaled) else (np.empty((0,)), np.empty((0,)))
    x_test, y_test = make_windows(test_scaled, window, horizon, stride) if len(test_scaled) else (np.empty((0,)), np.empty((0,)))

    if len(x_train) == 0:
        raise ValueError("Training split is too small for the chosen window/horizon.")

    return DatasetSplits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        scaler=scaler,
        raw_train=train_raw,
        raw_test=test_raw,
    )


def inverse_scale(scaler, data: np.ndarray) -> np.ndarray:
    shape = data.shape
    flat = data.reshape(-1, shape[-1])
    inv = scaler.inverse_transform(flat)
    return inv.reshape(shape)
