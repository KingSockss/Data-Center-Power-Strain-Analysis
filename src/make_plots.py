from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import PLOTS_DIR


DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _has_any(df: pd.DataFrame, columns: list[str]) -> bool:
    return all(col in df.columns for col in columns) and not df[columns].dropna(how="all").empty


def _write(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")


def make_plots(df: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> list[Path]:
    if df.empty:
        return []
    out: list[Path] = []
    plot_df = df.copy()
    plot_df["timestamp_et"] = pd.to_datetime(plot_df["timestamp_et"])

    if _has_any(plot_df, ["pjm_actual_load_mw", "pjm_forecast_load_mw"]):
        fig = px.line(
            plot_df,
            x="timestamp_et",
            y=["pjm_actual_load_mw", "pjm_forecast_load_mw"],
            title="Actual Load vs Forecast Load",
            labels={"value": "MW", "timestamp_et": "Timestamp (ET)", "variable": "Series"},
        )
        path = output_dir / "01_actual_vs_forecast_load.html"
        _write(fig, path)
        out.append(path)

    if _has_any(plot_df, ["pjm_load_forecast_error_mw"]):
        fig = px.line(plot_df, x="timestamp_et", y="pjm_load_forecast_error_mw", title="Load Forecast Error")
        path = output_dir / "02_load_forecast_error.html"
        _write(fig, path)
        out.append(path)

    if _has_any(plot_df, ["dom_rt_lmp", "dom_da_lmp"]):
        fig = px.line(
            plot_df,
            x="timestamp_et",
            y=["dom_rt_lmp", "dom_da_lmp"],
            title="DOM Real-Time vs Day-Ahead LMP",
            labels={"value": "$/MWh", "timestamp_et": "Timestamp (ET)", "variable": "Series"},
        )
        path = output_dir / "03_rt_vs_da_lmp.html"
        _write(fig, path)
        out.append(path)

    if _has_any(plot_df, ["dom_rt_da_spread"]):
        fig = px.line(plot_df, x="timestamp_et", y="dom_rt_da_spread", title="DOM RT - DA LMP Spread")
        path = output_dir / "04_rt_da_lmp_spread.html"
        _write(fig, path)
        out.append(path)

    if _has_any(plot_df, ["dom_congestion_component"]):
        fig = px.line(plot_df, x="timestamp_et", y="dom_congestion_component", title="DOM Congestion Component")
        path = output_dir / "05_congestion_component.html"
        _write(fig, path)
        out.append(path)

    if _has_any(plot_df, ["iad_temperature_f", "iad_cooling_degree_hour"]):
        fig = px.line(
            plot_df,
            x="timestamp_et",
            y=["iad_temperature_f", "iad_cooling_degree_hour"],
            title="Dulles Temperature and Cooling Degree Hours",
            labels={"value": "Value", "timestamp_et": "Timestamp (ET)", "variable": "Series"},
        )
        path = output_dir / "06_temperature_cooling_degree_hours.html"
        _write(fig, path)
        out.append(path)

    if _has_any(plot_df, ["iad_temperature_f", "pjm_actual_load_mw"]):
        fig = px.scatter(
            plot_df,
            x="iad_temperature_f",
            y="pjm_actual_load_mw",
            title="Temperature vs PJM Actual Load",
            labels={"iad_temperature_f": "Temperature (F)", "pjm_actual_load_mw": "Load (MW)"},
        )
        path = output_dir / "07_temperature_vs_load.html"
        _write(fig, path)
        out.append(path)

    if _has_any(plot_df, ["iad_temperature_f", "dom_rt_lmp"]):
        fig = px.scatter(
            plot_df,
            x="iad_temperature_f",
            y="dom_rt_lmp",
            title="Temperature vs DOM Real-Time LMP",
            labels={"iad_temperature_f": "Temperature (F)", "dom_rt_lmp": "$/MWh"},
        )
        path = output_dir / "08_temperature_vs_rt_lmp.html"
        _write(fig, path)
        out.append(path)

    if _has_any(plot_df, ["pjm_actual_load_mw"]):
        path = output_dir / "09_avg_load_heatmap.html"
        _heatmap(plot_df, "pjm_actual_load_mw", "Average Load by Hour and Day", "MW", path)
        out.append(path)

    if _has_any(plot_df, ["dom_rt_lmp"]):
        path = output_dir / "10_avg_rt_lmp_heatmap.html"
        _heatmap(plot_df, "dom_rt_lmp", "Average DOM Real-Time LMP by Hour and Day", "$/MWh", path)
        out.append(path)

    return out


def _heatmap(df: pd.DataFrame, value_col: str, title: str, color_label: str, path: Path) -> None:
    pivot = (
        df.pivot_table(index="day_of_week", columns="hour", values=value_col, aggfunc="mean")
        .reindex(DAY_ORDER)
    )
    fig = px.imshow(
        pivot,
        aspect="auto",
        title=title,
        labels={"x": "Hour of Day", "y": "Day of Week", "color": color_label},
    )
    _write(fig, path)
