"""Model factory and custom model loader."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Tuple

from .models import build_model
from .systems import SYSTEMS


class CustomModelError(RuntimeError):
    pass


def _infer_output_dim(system_cfg: Dict[str, Any]) -> int:
    name = system_cfg["name"]
    if name == "duffing":
        observable = system_cfg.get("observable", "full")
        if observable in {"x", "x_dot"}:
            return 1
        return 2
    return SYSTEMS[name].dim


def _load_custom_builder(path: Path, func_name: str):
    if not path.exists():
        raise CustomModelError(f"Custom model file not found: {path}")
    spec = importlib.util.spec_from_file_location("custom_model", str(path))
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise CustomModelError(f"Failed to load custom model module: {path}")
    spec.loader.exec_module(module)
    builder = getattr(module, func_name, None)
    if builder is None:
        raise CustomModelError(f"Missing function '{func_name}' in {path}")
    return builder


def build_model_from_config(config: Dict[str, Any]):
    system_cfg = config["system"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    output_dim = _infer_output_dim(system_cfg)
    input_shape = (data_cfg.get("window", 1), output_dim)
    horizon = data_cfg.get("horizon", 1)

    if model_cfg.get("name", "").lower() == "custom" or model_cfg.get("custom_path"):
        custom_path = Path(model_cfg["custom_path"])
        func_name = model_cfg.get("custom_builder", "build_model")
        builder = _load_custom_builder(custom_path, func_name)
        return builder(model_cfg, input_shape, output_dim, horizon)

    return build_model(
        model_name=model_cfg["name"],
        input_shape=input_shape,
        output_dim=output_dim,
        horizon=horizon,
        config=model_cfg,
    )


def infer_shapes_from_config(config: Dict[str, Any]) -> Tuple[Tuple[int, int], int, int]:
    system_cfg = config["system"]
    data_cfg = config["data"]
    output_dim = _infer_output_dim(system_cfg)
    input_shape = (data_cfg.get("window", 1), output_dim)
    horizon = data_cfg.get("horizon", 1)
    return input_shape, output_dim, horizon
