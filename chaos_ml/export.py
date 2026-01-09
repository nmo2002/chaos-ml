"""Model export utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch

from .factory import build_model_from_config, infer_shapes_from_config


ExportFormat = Literal["pt", "torchscript"]


def export_model(run_dir: Path, output_path: Path, fmt: ExportFormat = "pt") -> Path:
    run_dir = Path(run_dir)
    output_path = Path(output_path)

    config_path = run_dir / "config.json"
    model_path = run_dir / "model.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.pt in {run_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = build_model_from_config(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    if fmt == "pt":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        return output_path

    if fmt == "torchscript":
        input_shape, output_dim, _ = infer_shapes_from_config(config)
        dummy = torch.zeros(1, input_shape[0], output_dim)
        traced = torch.jit.trace(model, dummy)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        traced.save(str(output_path))
        return output_path

    raise ValueError(f"Unknown export format: {fmt}")
