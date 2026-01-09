"""Streamlit UI for running chaos_ml experiments."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import altair as alt
import streamlit as st

from chaos_ml.export import export_model


ROOT = Path(__file__).resolve().parent
DEFAULT_PY = sys.executable
RUNS_DIR = ROOT / "runs"
CONFIGS_DIR = ROOT / "configs"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_config(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_cli(python_exe: str, config_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [python_exe, "-m", "chaos_ml.cli", "--config", str(config_path)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


@st.cache_data(show_spinner=False)
def list_configs() -> list[Path]:
    return sorted(p for p in CONFIGS_DIR.glob("*.json"))


@st.cache_data(show_spinner=False)
def list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted((p for p in RUNS_DIR.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)



@st.cache_data(show_spinner=False)
def list_run_summaries() -> list[dict]:
    rows = []
    for run_dir in list_runs():
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": run_dir.name,
                "system": config.get("system", {}).get("name"),
                "model": config.get("model", {}).get("name"),
                "mse": metrics.get("mse"),
                "mae": metrics.get("mae"),
                "rmse": metrics.get("rmse"),
                "output_dir": str(config.get("output_dir", "")),
            }
        )
    return rows


def tail_text(text: str, limit: int = 2000) -> str:
    return text[-limit:] if len(text) > limit else text




def apply_plot_options(config_obj: dict, enabled: bool, lorenz96_lines: bool, heatmap_mode: str) -> dict:
    if not enabled:
        config_obj.pop("plot_options", None)
        return config_obj
    plot_opts = config_obj.get("plot_options", {})
    plot_opts["lorenz96_view"] = "lines" if lorenz96_lines else "heatmap"
    plot_opts["lorenz96_heatmap_mode"] = heatmap_mode
    config_obj["plot_options"] = plot_opts
    return config_obj


def apply_tuning(config_obj: dict, enabled: bool, trials: int, epochs: int) -> dict:
    if not enabled:
        config_obj.pop("tuning", None)
        return config_obj

    tuning = config_obj.get("tuning", {})
    tuning["enabled"] = True
    tuning["n_trials"] = int(trials)
    tuning["epochs"] = int(epochs)
    config_obj["tuning"] = tuning
    return config_obj


st.set_page_config(page_title="ChaosML", page_icon="CM", layout="wide")

st.title("ChaosML")
st.caption("Unified UI for Duffing, Lorenz-63, and Lorenz-96 experiments")

if "queue" not in st.session_state:
    st.session_state.queue = []
if "last_run_output" not in st.session_state:
    st.session_state.last_run_output = {"stdout": "", "stderr": ""}

python_exe = DEFAULT_PY

tab_config, tab_queue, tab_history, tab_compare = st.tabs(["Config", "Queue", "History", "Compare"])

with tab_config:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Experiment Config")
        config_files = list_configs()
        config_names = [p.name for p in config_files]
        selected = st.selectbox("Choose a preset", config_names, index=0 if config_names else None)
        config_path = CONFIGS_DIR / selected if selected else None
        config_data = load_config(config_path) if config_path else {}
        raw_json = st.text_area("Edit JSON config", value=json.dumps(config_data, indent=2), height=420)

        current_system = config_data.get("system", {}).get("name")
        try:
            current_system = json.loads(raw_json).get("system", {}).get("name")
        except Exception:
            pass
        is_lorenz96 = current_system == "lorenz96"

        save_name = st.text_input("Save as preset (filename)", value="new_experiment.json")
        save_btn = st.button("Save Preset", use_container_width=True)
        if save_btn:
            try:
                config_obj = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
            else:
                save_config(config_obj, CONFIGS_DIR / save_name)
                st.cache_data.clear()
                st.success(f"Saved preset: {save_name}")

        st.subheader("Tuning")
        tuning_enabled = st.checkbox("Enable Optuna tuning", value=False)
        if tuning_enabled:
            tuning_trials = st.number_input("Trials", min_value=1, max_value=200, value=20, step=1)
            tuning_epochs = st.number_input("Tuning epochs", min_value=1, max_value=200, value=50, step=1)
        else:
            tuning_trials = 20
            tuning_epochs = 50

        if is_lorenz96:
            st.subheader("Plot Options")
            lorenz96_lines = st.checkbox("Include Lorenz-96 line plots", value=False)
            heatmap_mode = st.selectbox("Heatmap mode", ["pair", "error"], index=0)
        else:
            lorenz96_lines = False
            heatmap_mode = "pair"

    with col_b:
        st.subheader("Run")
        st.caption(f"Using Python: {python_exe}")
        run_btn = st.button("Run Now", type="primary", use_container_width=True)
        queue_btn = st.button("Add to Queue", use_container_width=True)
        if queue_btn:
            st.session_state.queue.append({
                "raw": raw_json,
                "tuning": {"enabled": tuning_enabled, "trials": tuning_trials, "epochs": tuning_epochs},
                "plot": {"enabled": is_lorenz96, "lorenz96_lines": lorenz96_lines, "heatmap_mode": heatmap_mode},
            })
            st.success("Config added to queue.")

    if run_btn:
        try:
            config_obj = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
            st.stop()

        config_obj = apply_tuning(config_obj, tuning_enabled, tuning_trials, tuning_epochs)
        config_obj = apply_plot_options(config_obj, is_lorenz96, lorenz96_lines, heatmap_mode)
        temp_config = RUNS_DIR / "ui_config.json"
        save_config(config_obj, temp_config)

        with st.spinner("Running experiment..."):
            result = run_cli(python_exe, temp_config)

        st.session_state.last_run_output = {"stdout": result.stdout, "stderr": result.stderr}

        output_dir = Path(config_obj.get("output_dir", "runs/default"))
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            st.subheader("Metrics")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.json(metrics)

        plot_path = output_dir / "forecast.png"
        heatmap_path = output_dir / "heatmap.png"
        traj3d_path = output_dir / "trajectory3d.png"
        if heatmap_path.exists():
            st.subheader("Lorenz-96 Heatmap")
            st.image(str(heatmap_path), use_container_width=True)
            if lorenz96_lines and plot_path.exists():
                st.subheader("Forecast Plot")
                st.image(str(plot_path), use_container_width=True)
        elif plot_path.exists():
            st.subheader("Forecast Plot")
            st.image(str(plot_path), use_container_width=True)

        if traj3d_path.exists():
            st.subheader("3D Trajectory")
            st.image(str(traj3d_path), use_container_width=True)

    if st.session_state.last_run_output["stdout"] or st.session_state.last_run_output["stderr"]:
        st.subheader("Latest CLI Output")
        if st.session_state.last_run_output["stdout"]:
            st.code(tail_text(st.session_state.last_run_output["stdout"]), language="text")
        if st.session_state.last_run_output["stderr"]:
            st.code(tail_text(st.session_state.last_run_output["stderr"]), language="text")

with tab_queue:
    st.subheader("Run Queue")
    if not st.session_state.queue:
        st.info("Queue is empty.")
    else:
        st.write(f"Queued runs: {len(st.session_state.queue)}")
        if st.button("Clear Queue"):
            st.session_state.queue = []
            st.success("Queue cleared.")
        run_all = st.button("Run Queue", type="primary", use_container_width=True)
        if run_all:
            for idx, item in enumerate(list(st.session_state.queue), start=1):
                st.write(f"Running {idx}/{len(st.session_state.queue)}")
                try:
                    config_obj = json.loads(item["raw"])
                except json.JSONDecodeError as exc:
                    st.error(f"Invalid JSON in queue: {exc}")
                    continue
                config_obj = apply_tuning(
                    config_obj,
                    item["tuning"]["enabled"],
                    item["tuning"]["trials"],
                    item["tuning"]["epochs"],
                )
                config_obj = apply_plot_options(
                    config_obj,
                    item.get("plot", {}).get("enabled", False),
                    item.get("plot", {}).get("lorenz96_lines", False),
                    item.get("plot", {}).get("heatmap_mode", "pair"),
                )
                temp_config = RUNS_DIR / f"ui_queue_{idx}.json"
                save_config(config_obj, temp_config)
                with st.spinner(f"Executing run {idx}..."):
                    result = run_cli(python_exe, temp_config)
                st.code(tail_text(result.stdout), language="text")
                if result.stderr:
                    st.code(tail_text(result.stderr), language="text")
            st.session_state.queue = []
            st.success("Queue completed.")

with tab_history:
    st.subheader("Experiment History")
    if st.button("Refresh history"):
        st.cache_data.clear()
    runs = list_runs()
    if not runs:
        st.info("No runs found.")
    else:
        run_names = [p.name for p in runs]
        selected_run = st.selectbox("Select run", run_names, index=0)
        run_dir = RUNS_DIR / selected_run

        metrics_path = run_dir / "metrics.json"
        history_path = run_dir / "history.json"
        plot_path = run_dir / "forecast.png"
        heatmap_path = run_dir / "heatmap.png"
        traj3d_path = run_dir / "trajectory3d.png"
        config_path = run_dir / "config.json"

        if config_path.exists():
            st.subheader("Config")
            st.code(config_path.read_text(encoding="utf-8"), language="json")

        if metrics_path.exists():
            st.subheader("Metrics")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.json(metrics)

        if history_path.exists():
            st.subheader("Loss Curves")
            history = json.loads(history_path.read_text(encoding="utf-8"))
            train_loss = [x for x in history.get("train_loss", []) if x is not None]
            val_loss = [x for x in history.get("val_loss", []) if x is not None]
            if train_loss:
                st.line_chart({"train_loss": train_loss})
            if val_loss:
                st.line_chart({"val_loss": val_loss})

        if heatmap_path.exists():
            st.subheader("Lorenz-96 Heatmap")
            st.image(str(heatmap_path), use_container_width=True)
            if lorenz96_lines and plot_path.exists():
                st.subheader("Forecast Plot")
                st.image(str(plot_path), use_container_width=True)
        elif plot_path.exists():
            st.subheader("Forecast Plot")
            st.image(str(plot_path), use_container_width=True)

        if traj3d_path.exists():
            st.subheader("3D Trajectory")
            st.image(str(traj3d_path), use_container_width=True)

        st.subheader("Export Model")
        export_format = st.selectbox("Format", ["pt", "torchscript"], index=0)
        default_name = "exported_model.pt" if export_format == "pt" else "exported_model.ts"
        export_name = st.text_input("Output filename", value=default_name)
        export_btn = st.button("Export", use_container_width=True)
        if export_btn:
            out_path = run_dir / export_name
            try:
                export_model(run_dir, out_path, fmt=export_format)
            except Exception as exc:
                st.error(f"Export failed: {exc}")
            else:
                st.success(f"Saved: {out_path}")




with tab_compare:
    st.subheader("Compare Runs")
    if st.button("Refresh comparisons"):
        st.cache_data.clear()
    rows = list_run_summaries()
    if not rows:
        st.info("No comparable runs found.")
    else:
        systems = sorted({r["system"] for r in rows if r["system"]})
        models = sorted({r["model"] for r in rows if r["model"]})
        sel_system = st.selectbox("Filter system", ["All"] + systems, index=0)
        sel_model = st.selectbox("Filter model", ["All"] + models, index=0)
        metric = st.selectbox("Metric", ["rmse", "mse", "mae"], index=0)
        order = st.radio("Sort", ["Best to worst", "Worst to best"], horizontal=True)
        top_n = st.slider("Top-N spotlight", min_value=3, max_value=20, value=5, step=1)

        filtered = [
            r
            for r in rows
            if (sel_system == "All" or r["system"] == sel_system)
            and (sel_model == "All" or r["model"] == sel_model)
            and r.get(metric) is not None
        ]

        if not filtered:
            st.info("No runs match the selected filters.")
        else:
            reverse = order == "Worst to best"
            filtered = sorted(filtered, key=lambda r: r.get(metric, float("inf")), reverse=reverse)
            best = min(filtered, key=lambda r: r.get(metric, float("inf")))
            st.success(f"Best {metric}: {best['run']} ({best[metric]:.6g})")

            st.subheader("Top Runs")
            spotlight = filtered[:top_n]
            for i, r in enumerate(spotlight, start=1):
                st.write(f"{i}. {r['run']} | {r['system']} | {r['model']} | {metric}={r[metric]:.6g}")

            st.subheader("Metric Chart")
            labels = [r["run"] for r in filtered]
            values = [r[metric] for r in filtered]
            data = [{"run": labels[i], metric: values[i], "system": filtered[i]["system"], "model": filtered[i]["model"]} for i in range(len(filtered))]
            chart = (
                alt.Chart(alt.Data(values=data))
                .mark_bar()
                .encode(
                    x=alt.X("run:N", sort=None, axis=alt.Axis(labelAngle=-45, labelLimit=300)),
                    y=alt.Y(f"{metric}:Q"),
                    tooltip=["run:N", "system:N", "model:N", alt.Tooltip(f"{metric}:Q", format=".6g")],
                )
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("Chart order follows the table below.")

            st.subheader("Runs Table")
            st.dataframe(filtered, use_container_width=True)

            st.subheader("Export Comparison CSV")
            csv_name = st.text_input("CSV filename", value="run_comparison.csv")
            lines = ["run,system,model,mse,mae,rmse,output_dir"]
            for r in filtered:
                line = ",".join([
                    str(r.get("run", "")),
                    str(r.get("system", "")),
                    str(r.get("model", "")),
                    str(r.get("mse", "")),
                    str(r.get("mae", "")),
                    str(r.get("rmse", "")),
                    str(r.get("output_dir", "")),
                ])
                lines.append(line)
            csv_text = '\n'.join(lines)

            st.download_button(
                label="Download CSV",
                data=csv_text,
                file_name=csv_name,
                mime="text/csv",
                use_container_width=True,
            )
