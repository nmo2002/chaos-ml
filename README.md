# ChaosML

ChaosML is a unified toolkit for forecasting classic chaotic systems (Duffing, Lorenz-63, Lorenz-96) using PyTorch.
It replaces notebook duplication with a config-driven CLI, reusable modules, and an interactive UI.

Highlights
- Shared data generation + scaling + windowing pipeline
- LSTM, BiLSTM, Transformer, and Encoder-Decoder models
- Reproducible runs (seeded), checkpointing, metrics, and plots
- Optional hyperparameter tuning (Optuna)
- Custom model plug-ins
- UI for running, comparing, and exporting results

Use cases
- Benchmarking sequence models on chaotic dynamics
- Testing forecasting stability and multi-step error growth
- Prototyping data-driven surrogates for complex dynamical systems
- Exploring education and research demos in nonlinear dynamics
- Real-world analogs: weather/climate prototypes, mechanical oscillators, power/grid dynamics, and stress-testing forecasting workflows in finance

## Quick start (Windows)

Recommended Python: 3.11

Create a venv and install deps:

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Run a single experiment:

```bash
.\.venv\Scripts\python -m chaos_ml.cli --config configs/lorenz63_lstm.json
```

Run all example configs:

```bash
.
un_all.ps1
```

Outputs are saved to `runs/...` (metrics, predictions, plots, model checkpoint).

If PyTorch fails to load on Windows, install the Microsoft Visual C++ 2015-2022 Redistributable (x64) and retry.

## Install as a package (editable)

```bash
.\.venv\Scripts\python -m pip install -e .
```

## Streamlit UI

```bash
.\.venv\Scripts\python -m streamlit run app.py
```

UI features:
- Select and edit configs
- Save presets
- Queue multiple runs
- Inspect run history, metrics, and plots
- Compare runs across metrics
- Top-N run spotlight and CSV export
- Export trained models

## Streamlit Community Cloud

This repo includes `runtime.txt` for Python 3.11 compatibility.
For faster cloud demos, use the `*_demo.json` presets in `configs/` to keep runs short.

## Config structure

Each config is a JSON file with:
- `system`: which system, params, and time span
- `data`: window, horizon, splits, scaler
- `model`: architecture and hyperparameters
- `training`: epochs, batch size, learning rate, patience
- `tuning` (optional): Optuna search config

Example (minimal):

```json
{
  "system": {"name": "lorenz63", "t_end": 100, "t_points": 1000},
  "data": {"window": 10, "horizon": 1, "stride": 1, "train_ratio": 0.2, "val_ratio": 0.1, "scaler": "minmax"},
  "model": {"name": "lstm", "units": 256, "depth": 3, "dropout": 0.1},
  "training": {"epochs": 200, "batch_size": 64, "patience": 20, "learning_rate": 0.001}
}
```

## Hyperparameter tuning (Optuna)

Add a `tuning` section to your config:

```json
"tuning": {
  "enabled": true,
  "n_trials": 25,
  "epochs": 50,
  "search_space": {
    "window": {"low": 5, "high": 30, "step": 5},
    "learning_rate": {"low": 1e-4, "high": 1e-2, "log": true},
    "units": {"low": 64, "high": 512, "step": 64},
    "depth": {"low": 2, "high": 4, "step": 1},
    "dropout": {"low": 0.0, "high": 0.3, "step": 0.05}
  }
}
```

The best params are saved to `runs/<output>/tuning/best_params.json` and applied to the final training run.

## Custom models

Set `model.name` to `custom` and point to your builder:

```json
"model": {
  "name": "custom",
  "custom_path": "custom_model.py",
  "custom_builder": "build_model"
}
```

See `custom_model.py` for a working template.

Your `custom_model.py` should define:

```python
import torch
from torch import nn

def build_model(model_cfg, input_shape, output_dim, horizon):
    # input_shape is (window, features)
    # return a torch.nn.Module
    return nn.Sequential(...)
```

## Plot options (Lorenz-96)
- `plot_options.lorenz96_lines`: include line plots alongside heatmap
- `plot_options.lorenz96_heatmap_mode`: `pair` or `error`

## Model export

Every run saves a best checkpoint to `runs/<output>/model.pt`.
You can export from the UI (History tab) in either:
- `pt` (state_dict)
- `torchscript`

## Enterprise usage notes

- Reproducibility: set `seed` in the config for deterministic runs.
- Separation of concerns: configs are versionable artifacts; UI edits write to JSON.
- Extensibility: custom models can be checked into repo or loaded from external paths.
- Auditability: each run writes `config.json`, `history.json`, and `metrics.json`.

## Troubleshooting

- `No module named streamlit`: install in venv using `python -m pip install streamlit`.
- `DLL load failed` (PyTorch): install the VC++ 2015-2022 Redistributable (x64).
- Empty validation set: reduce `window` or increase `train_ratio`/`val_ratio`.

## FAQ

Q: Can I run tuning for only one model?
A: Yes. Select the model in the config and enable `tuning.enabled`.

Q: Where are results stored?
A: `runs/<output>/` contains the metrics, history, plots, and model checkpoint.

Q: Can I share a trained model?
A: Yes. Export from the UI or copy `model.pt` from the run directory.

## Performance tips

- Start with small `t_points` or fewer epochs while iterating.
- Use GPU if available; the CLI auto-detects CUDA.
- For transformers, increase `window` cautiously to control memory use.
