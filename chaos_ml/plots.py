"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_timeseries(t: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dims = y_true.shape[-1]
    fig, axes = plt.subplots(dims, 1, figsize=(10, 3 * dims), sharex=True)
    if dims == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, y_true[:, i], label="ground truth", color="tab:orange")
        ax.plot(t, y_pred[:, i], label="prediction", color="tab:blue", alpha=0.8)
        ax.set_ylabel(f"dim {i}")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
