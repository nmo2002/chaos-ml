"""Hyperparameter tuning with Optuna."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import optuna
import torch

from .datasets import prepare_dataset
from .factory import build_model_from_config
from .systems import generate_timeseries, select_observable
from .training import fit_model


DEFAULT_SPACE = {
    "window": {"low": 5, "high": 30, "step": 5},
    "learning_rate": {"low": 1e-4, "high": 1e-2, "log": True},
    "units": {"low": 64, "high": 512, "step": 64},
    "depth": {"low": 2, "high": 4, "step": 1},
    "dropout": {"low": 0.0, "high": 0.3, "step": 0.05},
    "num_layers": {"low": 2, "high": 4, "step": 1},
    "d_model": {"low": 32, "high": 128, "step": 32},
    "num_heads": {"low": 2, "high": 8, "step": 2},
    "dff": {"low": 64, "high": 256, "step": 64},
}


def _suggest(trial: optuna.Trial, name: str, spec: Dict[str, Any]):
    if spec.get("log"):
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if "step" in spec:
        step = spec.get("step")
        if isinstance(step, int) and step > 0:
            return trial.suggest_int(name, spec["low"], spec["high"], step=step)
        return trial.suggest_int(name, spec["low"], spec["high"])
    return trial.suggest_float(name, spec["low"], spec["high"])


def run_tuning(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    tuning_cfg = config.get("tuning", {})
    n_trials = int(tuning_cfg.get("n_trials", 20))
    space = tuning_cfg.get("search_space", {}) or DEFAULT_SPACE
    epochs = int(tuning_cfg.get("epochs", min(50, config.get("training", {}).get("epochs", 200))))

    base_model_cfg = dict(config["model"])
    base_data_cfg = dict(config["data"])
    base_train_cfg = dict(config["training"])

    out_dir = Path(config.get("output_dir", "runs/default")) / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        model_cfg = dict(base_model_cfg)
        data_cfg = dict(base_data_cfg)
        train_cfg = dict(base_train_cfg)

        if "window" in space:
            data_cfg["window"] = _suggest(trial, "window", space["window"])
        if "learning_rate" in space:
            train_cfg["learning_rate"] = _suggest(trial, "learning_rate", space["learning_rate"])

        for key in ("units", "depth", "dropout", "num_layers", "d_model", "num_heads", "dff"):
            if key in space:
                model_cfg[key] = _suggest(trial, key, space[key])


        if "d_model" in model_cfg and "num_heads" in model_cfg:
            d_model = int(model_cfg["d_model"])
            num_heads = int(model_cfg["num_heads"])
            if num_heads < 1:
                model_cfg["num_heads"] = 1
            elif d_model % num_heads != 0:
                divisors = [h for h in range(1, num_heads + 1) if d_model % h == 0]
                model_cfg["num_heads"] = max(divisors) if divisors else 1

        system_cfg = config["system"]
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

        trial_cfg = {"system": system_cfg, "data": data_cfg, "model": model_cfg, "training": train_cfg}
        model = build_model_from_config(trial_cfg)

        history = fit_model(
            model,
            splits.x_train,
            splits.y_train,
            splits.x_val,
            splits.y_val,
            epochs=epochs,
            batch_size=train_cfg.get("batch_size", 32),
            patience=min(10, train_cfg.get("patience", 20)),
            learning_rate=train_cfg.get("learning_rate", 1e-3),
            out_dir=out_dir / f"trial_{trial.number}",
            device=device,
            save_best=False,
        )

        val_loss = [v for v in history.get("val_loss", []) if v is not None]
        if val_loss:
            return float(min(val_loss))
        return float(history.get("best_loss", 1e9))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    (out_dir / "study.json").write_text(json.dumps({"value": study.best_value, "params": best_params}, indent=2), encoding="utf-8")

    return best_params
