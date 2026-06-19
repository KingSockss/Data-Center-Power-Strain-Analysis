from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


TARGET_COL = "dom_rt_lmp"
DAY_AHEAD_COL = "dom_da_lmp"
CONGESTION_COL = "dom_congestion_component"
TEMP_COL = "iad_temperature_f"
DEFAULT_DATASET_PATH = Path("data/processed/merged/hourly_pjm_dom_dataset.parquet")
SEASONAL_PERIOD_HOURS = 24


def load_hourly_dataset(path: Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Merged hourly dataset not found: {path}")
    df = pd.read_parquet(path)
    if "timestamp_et" not in df.columns:
        raise ValueError("Merged dataset must include timestamp_et.")
    df["timestamp_et"] = pd.to_datetime(df["timestamp_et"])
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    return df.sort_values("timestamp_et").reset_index(drop=True)


def choose_load_column(df: pd.DataFrame) -> tuple[str, str]:
    if "dom_load_mw" in df.columns and df["dom_load_mw"].notna().any():
        return "dom_load_mw", "DOM Load MW"
    if "pjm_actual_load_mw" in df.columns and df["pjm_actual_load_mw"].notna().any():
        return "pjm_actual_load_mw", "PJM Actual Load MW"
    raise ValueError("No usable load column found. Expected dom_load_mw or pjm_actual_load_mw.")


def add_train_test_split(df: pd.DataFrame, train_months: int = 9) -> pd.DataFrame:
    out = df.copy()
    start = out["timestamp_et"].min()
    split_at = start + pd.DateOffset(months=train_months)
    if split_at >= out["timestamp_et"].max():
        split_at = out["timestamp_et"].quantile(0.75)
    out["split"] = np.where(out["timestamp_et"] < split_at, "train", "test")
    return out


def make_baseline_predictions(df: pd.DataFrame, train_months: int = 9) -> pd.DataFrame:
    required = [TARGET_COL, DAY_AHEAD_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Merged dataset missing required baseline columns: {missing}")

    out = df[["timestamp_utc", "timestamp_et", TARGET_COL, DAY_AHEAD_COL]].copy()
    out = add_train_test_split(out, train_months=train_months)
    out = out.rename(columns={TARGET_COL: "actual_lmp", DAY_AHEAD_COL: "pjm_day_ahead_lmp"})
    out["reference_lmp_t_minus_1"] = out["actual_lmp"].shift(1)
    out["persistence"] = out["actual_lmp"].shift(1)
    out["daily_persistence"] = out["actual_lmp"].shift(SEASONAL_PERIOD_HOURS)
    return out


def mase_denominator(train_actual: pd.Series, seasonal_period: int = SEASONAL_PERIOD_HOURS) -> float:
    diffs = (train_actual - train_actual.shift(seasonal_period)).abs().dropna()
    if diffs.empty:
        return np.nan
    denom = float(diffs.mean())
    return denom if denom != 0 else np.nan


def compute_metrics(
    frame: pd.DataFrame,
    model_name: str,
    prediction_col: str,
    actual_col: str = "actual_lmp",
    reference_col: str = "reference_lmp_t_minus_1",
    mase_denom: float | None = None,
    split: str = "test",
) -> dict[str, float | int | str]:
    eval_df = frame[frame["split"].eq(split)].copy()
    eval_df = eval_df[[actual_col, prediction_col, reference_col]].dropna()
    if eval_df.empty:
        return {
            "model": model_name,
            "split": split,
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias_mean_error": np.nan,
            "wape_pct": np.nan,
            "mase": np.nan,
            "directional_accuracy_pct": np.nan,
        }

    actual = eval_df[actual_col].astype(float)
    pred = eval_df[prediction_col].astype(float)
    reference = eval_df[reference_col].astype(float)
    error = pred - actual
    mae = float(error.abs().mean())
    rmse = float(np.sqrt(np.mean(np.square(error))))
    bias = float(error.mean())
    denom = float(actual.abs().sum())
    wape = float(error.abs().sum() / denom * 100) if denom else np.nan
    mase = float(mae / mase_denom) if mase_denom and not np.isnan(mase_denom) else np.nan

    actual_direction = np.sign(actual - reference)
    pred_direction = np.sign(pred - reference)
    direction_mask = actual_direction.ne(0)
    if direction_mask.any():
        directional_accuracy = float((actual_direction[direction_mask] == pred_direction[direction_mask]).mean() * 100)
    else:
        directional_accuracy = np.nan

    return {
        "model": model_name,
        "split": split,
        "n": int(len(eval_df)),
        "mae": mae,
        "rmse": rmse,
        "bias_mean_error": bias,
        "wape_pct": wape,
        "mase": mase,
        "directional_accuracy_pct": directional_accuracy,
    }


def compute_many_metrics(
    frame: pd.DataFrame,
    model_columns: dict[str, str],
    train_actual: pd.Series,
    split: str = "test",
) -> pd.DataFrame:
    denom = mase_denominator(train_actual)
    rows = [
        compute_metrics(frame, model_name, prediction_col, mase_denom=denom, split=split)
        for model_name, prediction_col in model_columns.items()
    ]
    return pd.DataFrame(rows)


def write_dark_line_plot(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font={"color": "#e5e7eb"},
        legend={"bgcolor": "rgba(17, 24, 39, 0.65)"},
        hovermode="x unified",
    )
    fig.write_html(path, include_plotlyjs="cdn")
