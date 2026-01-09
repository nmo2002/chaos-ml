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
        ax.set_xlabel("time")
        ax.tick_params(labelbottom=True)
        ax.legend(loc="upper right")
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.995])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_3d_trajectory(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(y_true[:, 0], y_true[:, 1], y_true[:, 2], label="ground truth", color="tab:orange")
    ax.plot(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], label="prediction", color="tab:blue", alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right")
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.985])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap_pair(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    im0 = axes[0].imshow(y_true.T, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("ground truth")
    axes[0].set_xlabel("time index")
    axes[0].set_ylabel("dimension")

    im1 = axes[1].imshow(y_pred.T, aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_title("prediction")
    axes[1].set_xlabel("time index")

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.985])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap(data: np.ndarray, out_path: Path, title: str, xlabel: str, ylabel: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(data.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, label="value")
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.985])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
