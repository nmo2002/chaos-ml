"""CLI entrypoint for unified experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .datasets import prepare_dataset
from .factory import build_model_from_config
from .plots import plot_timeseries, plot_3d_trajectory, plot_heatmap_pair, plot_heatmap
from .systems import generate_timeseries, select_observable
from .training import evaluate_and_save, fit_model, predict_direct
from .tuner import run_tuning


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Unified chaotic system ML experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    out_dir = Path(config.get("output_dir", "runs/default"))
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config.get("seed", 1)))

    system_cfg = config["system"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    plot_opts = config.get("plot_options", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tuning_cfg = config.get("tuning", {})
    if tuning_cfg.get("enabled"):
        best_params = run_tuning(config, device=device)
        for key, value in best_params.items():
            if key in {"window", "horizon", "stride"}:
                data_cfg[key] = value
            elif key in {"learning_rate", "batch_size", "patience", "epochs"}:
                train_cfg[key] = value
            else:
                model_cfg[key] = value
        (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    ic = system_cfg.get("initial_conditions")
    ic = np.array(ic, dtype=float) if ic is not None else None

    t, series = generate_timeseries(
        system=system_cfg["name"],
        t_end=system_cfg.get("t_end", 100.0),
        t_points=system_cfg.get("t_points", 1000),
        lyapunov_exponent=system_cfg.get("lyapunov_exponent", 1.0),
        initial_conditions=ic,
        params=system_cfg.get("params", {}),
    )
    series = select_observable(series, system_cfg["name"], system_cfg.get("observable", "full"))

    splits = prepare_dataset(
        series=series,
        window=data_cfg.get("window", 1),
        horizon=data_cfg.get("horizon", 1),
        stride=data_cfg.get("stride", 1),
        train_ratio=data_cfg.get("train_ratio", 0.2),
        val_ratio=data_cfg.get("val_ratio", 0.1),
        scaler_name=data_cfg.get("scaler", "minmax"),
        feature_range=tuple(data_cfg.get("feature_range", [0.0, 1.0])),
    )

    model = build_model_from_config(config)

    history = fit_model(
        model,
        splits.x_train,
        splits.y_train,
        splits.x_val,
        splits.y_val,
        epochs=train_cfg.get("epochs", 200),
        batch_size=train_cfg.get("batch_size", 32),
        patience=train_cfg.get("patience", 20),
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        out_dir=out_dir,
        device=device,
        save_best=True,
    )

    y_pred = predict_direct(model, splits.x_test, device=device)
    result = evaluate_and_save(out_dir, splits.scaler, splits.y_test, y_pred, history)

    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if config.get("plot", True) and len(splits.y_test):
        test_len = len(splits.raw_test)
        window = data_cfg.get("window", 1)
        t_test = t[-test_len:]
        t_plot = t_test[window : window + len(result.predictions)]
        y_true_first = splits.y_test[:, 0, :]
        y_pred_first = y_pred[:, 0, :]
        y_true_inv = splits.scaler.inverse_transform(y_true_first)
        y_pred_inv = splits.scaler.inverse_transform(y_pred_first)

        if system_cfg["name"] == "lorenz96":
            heatmap_mode = plot_opts.get("lorenz96_heatmap_mode", "pair")
            include_lines = plot_opts.get("lorenz96_lines", plot_opts.get("lorenz96_view") == "lines")

            if heatmap_mode == "error":
                error = abs(y_true_inv - y_pred_inv)
                plot_heatmap(
                    error,
                    out_dir / "heatmap.png",
                    title=f"{system_cfg['name']} - {model_cfg['name']} (abs error)",
                    xlabel="time index",
                    ylabel="dimension",
                )
            else:
                plot_heatmap_pair(
                    y_true_inv,
                    y_pred_inv,
                    out_dir / "heatmap.png",
                    title=f"{system_cfg['name']} - {model_cfg['name']}",
                )

            if include_lines:
                plot_timeseries(
                    t_plot,
                    y_true_inv,
                    y_pred_inv,
                    out_dir / "forecast.png",
                    title=f"{system_cfg['name']} - {model_cfg['name']} (horizon=1)",
                )
        else:
            plot_timeseries(
                t_plot,
                y_true_inv,
                y_pred_inv,
                out_dir / "forecast.png",
                title=f"{system_cfg['name']} - {model_cfg['name']} (horizon=1)",
            )

        if system_cfg["name"] == "lorenz63" and y_true_inv.shape[1] == 3:
            plot_3d_trajectory(
                y_true_inv,
                y_pred_inv,
                out_dir / "trajectory3d.png",
                title="Lorenz-63 Trajectory (3D)",
            )

    print(f"Done. Metrics: {result.metrics}")


if __name__ == "__main__":
    main()
