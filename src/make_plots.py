from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import PLOTS_DIR


DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PLOT_TEMPLATE = "plotly_dark"
SERIES_COLORS = {
    "load": "#60a5fa",
    "temperature": "#f97316",
    "congestion": "#a78bfa",
    "price": "#22c55e",
}


def _has_any(df: pd.DataFrame, columns: list[str]) -> bool:
    return all(col in df.columns for col in columns) and not df[columns].dropna(how="all").empty


def _series_has_data(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and df[column].notna().any()


def _has_each_series(df: pd.DataFrame, columns: list[str]) -> bool:
    return all(_series_has_data(df, col) for col in columns)


def _preferred_load_column(df: pd.DataFrame) -> tuple[str | None, str]:
    if _series_has_data(df, "dom_load_mw"):
        return "dom_load_mw", "DOM Load (MW)"
    if _series_has_data(df, "pjm_actual_load_mw"):
        return "pjm_actual_load_mw", "PJM Actual Load (MW)"
    return None, "Load (MW)"


def _write(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font={"color": "#e5e7eb"},
        legend={"bgcolor": "rgba(17, 24, 39, 0.65)"},
    )
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

    load_col, load_label = _preferred_load_column(plot_df)
    if load_col and _has_each_series(plot_df, [load_col, "iad_temperature_f", "dom_congestion_component"]):
        path = output_dir / "11_load_temperature_congestion_overlay.html"
        _three_axis_overlay(
            plot_df,
            title="Load, Temperature, and DOM Congestion Over Time",
            path=path,
            y1_col=load_col,
            y1_name=load_label,
            y1_title=load_label,
            y1_color=SERIES_COLORS["load"],
            y2_col="iad_temperature_f",
            y2_name="Dulles Temperature (F)",
            y2_title="Temperature (F)",
            y2_color=SERIES_COLORS["temperature"],
            y3_col="dom_congestion_component",
            y3_name="DOM Congestion ($/MWh)",
            y3_title="Congestion ($/MWh)",
            y3_color=SERIES_COLORS["congestion"],
        )
        out.append(path)

    if _has_each_series(plot_df, ["dom_rt_lmp", "iad_temperature_f", "dom_congestion_component"]):
        path = output_dir / "12_price_temperature_congestion_overlay.html"
        _three_axis_overlay(
            plot_df,
            title="DOM Real-Time LMP, Temperature, and Congestion Over Time",
            path=path,
            y1_col="dom_rt_lmp",
            y1_name="DOM Real-Time LMP ($/MWh)",
            y1_title="Price ($/MWh)",
            y1_color=SERIES_COLORS["price"],
            y2_col="iad_temperature_f",
            y2_name="Dulles Temperature (F)",
            y2_title="Temperature (F)",
            y2_color=SERIES_COLORS["temperature"],
            y3_col="dom_congestion_component",
            y3_name="DOM Congestion ($/MWh)",
            y3_title="Congestion ($/MWh)",
            y3_color=SERIES_COLORS["congestion"],
        )
        out.append(path)

    if load_col and _has_each_series(plot_df, [load_col, "dom_rt_lmp"]):
        path = output_dir / "13_load_price_overlay.html"
        _two_axis_overlay(
            plot_df,
            title="Load and DOM Real-Time LMP Over Time",
            path=path,
            y1_col=load_col,
            y1_name=load_label,
            y1_title=load_label,
            y1_color=SERIES_COLORS["load"],
            y2_col="dom_rt_lmp",
            y2_name="DOM Real-Time LMP ($/MWh)",
            y2_title="Price ($/MWh)",
            y2_color=SERIES_COLORS["price"],
        )
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


def _two_axis_overlay(
    df: pd.DataFrame,
    title: str,
    path: Path,
    y1_col: str,
    y1_name: str,
    y1_title: str,
    y1_color: str,
    y2_col: str,
    y2_name: str,
    y2_title: str,
    y2_color: str,
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp_et"],
            y=df[y1_col],
            name=y1_name,
            mode="lines",
            line={"color": y1_color, "width": 2},
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp_et"],
            y=df[y2_col],
            name=y2_name,
            mode="lines",
            line={"color": y2_color, "width": 2},
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=title,
        xaxis={"title": "Timestamp (ET)"},
        yaxis={"title": {"text": y1_title, "font": {"color": y1_color}}, "tickfont": {"color": y1_color}},
        yaxis2={
            "title": {"text": y2_title, "font": {"color": y2_color}},
            "tickfont": {"color": y2_color},
            "overlaying": "y",
            "side": "right",
        },
        hovermode="x unified",
    )
    _write(fig, path)


def _three_axis_overlay(
    df: pd.DataFrame,
    title: str,
    path: Path,
    y1_col: str,
    y1_name: str,
    y1_title: str,
    y1_color: str,
    y2_col: str,
    y2_name: str,
    y2_title: str,
    y2_color: str,
    y3_col: str,
    y3_name: str,
    y3_title: str,
    y3_color: str,
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp_et"],
            y=df[y1_col],
            name=y1_name,
            mode="lines",
            line={"color": y1_color, "width": 2},
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp_et"],
            y=df[y2_col],
            name=y2_name,
            mode="lines",
            line={"color": y2_color, "width": 2},
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp_et"],
            y=df[y3_col],
            name=y3_name,
            mode="lines",
            line={"color": y3_color, "width": 2},
            yaxis="y3",
        )
    )
    fig.update_layout(
        title=title,
        xaxis={"title": "Timestamp (ET)", "domain": [0.0, 0.88]},
        yaxis={"title": {"text": y1_title, "font": {"color": y1_color}}, "tickfont": {"color": y1_color}},
        yaxis2={
            "title": {"text": y2_title, "font": {"color": y2_color}},
            "tickfont": {"color": y2_color},
            "overlaying": "y",
            "side": "right",
        },
        yaxis3={
            "title": {"text": y3_title, "font": {"color": y3_color}},
            "tickfont": {"color": y3_color},
            "anchor": "free",
            "overlaying": "y",
            "side": "right",
            "position": 0.96,
        },
        hovermode="x unified",
    )
    _write(fig, path)
