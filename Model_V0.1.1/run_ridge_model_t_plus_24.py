from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling_utils import (  # noqa: E402
    CONGESTION_COL,
    DAY_AHEAD_COL,
    DEFAULT_DATASET_PATH,
    TARGET_COL,
    TEMP_COL,
    choose_load_column,
    compute_many_metrics,
    load_hourly_dataset,
    write_dark_line_plot,
)


HORIZON_HOURS = 24
BASELINE_MODEL_COLUMNS = {
    "persistence_origin_lmp": "persistence_origin_lmp",
    "daily_persistence_origin_minus_24": "daily_persistence_origin_minus_24",
    "pjm_day_ahead_lmp_for_target": "pjm_day_ahead_lmp_for_target",
}
TEMP_FORECAST_CANDIDATES = [
    "iad_temperature_forecast_f",
    "temperature_forecast_f",
    "forecast_temperature_f",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model_V0.1.1 Ridge regression for t+24 DOM RT LMP.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET_PATH, help="Merged hourly parquet input.")
    parser.add_argument("--output-dir", type=Path, default=Path("Model_V0.1.1/outputs"), help="Output directory.")
    parser.add_argument("--alpha", type=float, default=10.0, help="Ridge regularization strength.")
    parser.add_argument("--train-months", type=int, default=9, help="Number of first target months treated as train period.")
    return parser.parse_args()


def build_feature_frame(df: pd.DataFrame, train_months: int = 9) -> tuple[pd.DataFrame, list[str], str]:
    load_col, load_label = choose_load_column(df)
    required = [TARGET_COL, DAY_AHEAD_COL, load_col, CONGESTION_COL, TEMP_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Merged dataset missing required Model_V0.1.1 columns: {missing}")

    out = df.copy().sort_values("timestamp_et").reset_index(drop=True)
    out["forecast_origin_utc"] = out["timestamp_utc"]
    out["forecast_origin_et"] = out["timestamp_et"]
    out["target_timestamp_utc"] = out["timestamp_utc"].shift(-HORIZON_HOURS)
    out["target_timestamp_et"] = out["timestamp_et"].shift(-HORIZON_HOURS)
    out["actual_lmp"] = out[TARGET_COL].shift(-HORIZON_HOURS)
    out["pjm_day_ahead_lmp_for_target"] = out[DAY_AHEAD_COL].shift(-HORIZON_HOURS)

    # Baselines for the t+24 target from information anchored at forecast origin t.
    out["reference_lmp_t_minus_1"] = out[TARGET_COL].shift(1)
    out["persistence_origin_lmp"] = out[TARGET_COL]
    out["daily_persistence_origin_minus_24"] = out[TARGET_COL].shift(24)

    out["lmp_t_minus_1"] = out[TARGET_COL].shift(1)
    out["lmp_t_minus_24"] = out[TARGET_COL].shift(24)
    out["load_t_minus_1"] = out[load_col].shift(1)
    out["load_t_minus_24"] = out[load_col].shift(24)
    out["congestion_t_minus_1"] = out[CONGESTION_COL].shift(1)
    out["congestion_t_minus_24"] = out[CONGESTION_COL].shift(24)
    out["temp_t_minus_1"] = out[TEMP_COL].shift(1)
    out["temp_t_minus_24"] = out[TEMP_COL].shift(24)

    temp_forecast_col = next((col for col in TEMP_FORECAST_CANDIDATES if col in out.columns and out[col].notna().any()), None)
    if temp_forecast_col:
        out["temp_forecast"] = out[temp_forecast_col].shift(-HORIZON_HOURS)

    origin = out["forecast_origin_et"]
    hour_angle = 2 * np.pi * origin.dt.hour / 24
    day_angle = 2 * np.pi * origin.dt.dayofweek / 7
    month_angle = 2 * np.pi * (origin.dt.month - 1) / 12
    out["hour_sin"] = np.sin(hour_angle)
    out["hour_cos"] = np.cos(hour_angle)
    out["day_sin"] = np.sin(day_angle)
    out["day_cos"] = np.cos(day_angle)
    out["month_sin"] = np.sin(month_angle)
    out["month_cos"] = np.cos(month_angle)
    out["is_weekend_binary"] = (origin.dt.dayofweek >= 5).astype(int)

    out["lmp_sma_24"] = out[TARGET_COL].shift(1).rolling(24, min_periods=6).mean()
    out["lmp_sma_6"] = out[TARGET_COL].shift(1).rolling(6, min_periods=3).mean()
    out["load_sma_24"] = out[load_col].shift(1).rolling(24, min_periods=6).mean()
    out["load_sma_6"] = out[load_col].shift(1).rolling(6, min_periods=3).mean()
    out["congestion_sma_24"] = out[CONGESTION_COL].shift(1).rolling(24, min_periods=6).mean()
    out["congestion_sma_6"] = out[CONGESTION_COL].shift(1).rolling(6, min_periods=3).mean()

    target_start = out["target_timestamp_et"].dropna().min()
    split_at = target_start + pd.DateOffset(months=train_months)
    if split_at >= out["target_timestamp_et"].dropna().max():
        split_at = out["target_timestamp_et"].dropna().quantile(0.75)
    out["split"] = np.where(out["target_timestamp_et"] < split_at, "train", "test")

    features = [
        "lmp_t_minus_1",
        "lmp_t_minus_24",
        "load_t_minus_1",
        "load_t_minus_24",
        "congestion_t_minus_1",
        "congestion_t_minus_24",
        "temp_t_minus_1",
        "temp_t_minus_24",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "month_sin",
        "month_cos",
        "is_weekend_binary",
        "lmp_sma_24",
        "lmp_sma_6",
        "load_sma_24",
        "load_sma_6",
        "congestion_sma_24",
        "congestion_sma_6",
    ]
    if temp_forecast_col:
        features.append("temp_forecast")

    keep_required = [
        "actual_lmp",
        "reference_lmp_t_minus_1",
        "target_timestamp_et",
        "target_timestamp_utc",
        *features,
    ]
    out = out.dropna(subset=keep_required).copy()
    return out, features, load_label


def fit_ridge(train: pd.DataFrame, features: list[str], alpha: float) -> dict[str, np.ndarray | pd.Series | float]:
    x = train[features].astype(float)
    y = train["actual_lmp"].astype(float).to_numpy()
    means = x.mean()
    stds = x.std(ddof=0).replace(0, 1.0)
    x_scaled = ((x - means) / stds).to_numpy()
    design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    penalty = alpha * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {"beta": beta, "means": means, "stds": stds, "alpha": alpha}


def predict_ridge(model: dict[str, np.ndarray | pd.Series | float], frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    x = frame[features].astype(float)
    means = model["means"]
    stds = model["stds"]
    beta = model["beta"]
    x_scaled = ((x - means) / stds).to_numpy()
    design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    return design @ beta


def coefficient_table(model: dict[str, np.ndarray | pd.Series | float], features: list[str]) -> pd.DataFrame:
    beta = model["beta"]
    means = model["means"]
    stds = model["stds"]
    standardized = beta[1:]
    original_scale = standardized / stds.to_numpy()
    intercept_original = beta[0] - np.sum(standardized * means.to_numpy() / stds.to_numpy())
    rows = [
        {
            "feature": "intercept",
            "coefficient_standardized_space": beta[0],
            "coefficient_original_scale": intercept_original,
        }
    ]
    rows.extend(
        {
            "feature": feature,
            "coefficient_standardized_space": coef,
            "coefficient_original_scale": orig,
        }
        for feature, coef, orig in zip(features, standardized, original_scale, strict=False)
    )
    return pd.DataFrame(rows)


def plot_true_vs_ridge(predictions: pd.DataFrame, output_dir: Path) -> None:
    test = predictions[predictions["split"].eq("test")]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test["target_timestamp_et"], y=test["actual_lmp"], name="Actual DOM RT LMP", mode="lines"))
    fig.add_trace(
        go.Scatter(x=test["target_timestamp_et"], y=test["ridge_v0_1_1"], name="Ridge V0.1.1 t+24 Forecast", mode="lines")
    )
    fig.update_layout(
        title="True DOM Real-Time LMP vs Ridge V0.1.1 t+24 Forecast",
        xaxis_title="Target Timestamp (ET)",
        yaxis_title="$/MWh",
    )
    write_dark_line_plot(fig, output_dir / "true_vs_ridge_t_plus_24_forecast.html")


def plot_true_vs_all(predictions: pd.DataFrame, output_dir: Path) -> None:
    test = predictions[predictions["split"].eq("test")]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test["target_timestamp_et"], y=test["actual_lmp"], name="Actual DOM RT LMP", mode="lines"))
    fig.add_trace(go.Scatter(x=test["target_timestamp_et"], y=test["ridge_v0_1_1"], name="Ridge V0.1.1", mode="lines"))
    fig.add_trace(go.Scatter(x=test["target_timestamp_et"], y=test["persistence_origin_lmp"], name="Persistence Origin LMP", mode="lines"))
    fig.add_trace(
        go.Scatter(
            x=test["target_timestamp_et"],
            y=test["daily_persistence_origin_minus_24"],
            name="Daily Persistence Origin-24",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test["target_timestamp_et"],
            y=test["pjm_day_ahead_lmp_for_target"],
            name="PJM Day-Ahead LMP for Target",
            mode="lines",
        )
    )
    fig.update_layout(
        title="True DOM Real-Time LMP vs Ridge V0.1.1 and t+24 Baselines",
        xaxis_title="Target Timestamp (ET)",
        yaxis_title="$/MWh",
    )
    write_dark_line_plot(fig, output_dir / "true_vs_all_t_plus_24_models.html")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_hourly_dataset(args.input)
    features, feature_cols, load_label = build_feature_frame(raw, train_months=args.train_months)
    train = features[features["split"].eq("train")]
    test = features[features["split"].eq("test")]
    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty train or test set.")

    model = fit_ridge(train, feature_cols, alpha=args.alpha)
    features["ridge_v0_1_1"] = predict_ridge(model, features, feature_cols)

    prediction_cols = [
        "forecast_origin_utc",
        "forecast_origin_et",
        "target_timestamp_utc",
        "target_timestamp_et",
        "split",
        "actual_lmp",
        "reference_lmp_t_minus_1",
        "ridge_v0_1_1",
        "persistence_origin_lmp",
        "daily_persistence_origin_minus_24",
        "pjm_day_ahead_lmp_for_target",
    ]
    predictions = features[prediction_cols].copy()

    model_columns = {"ridge_v0_1_1": "ridge_v0_1_1", **BASELINE_MODEL_COLUMNS}
    metrics = compute_many_metrics(
        predictions,
        model_columns,
        predictions.loc[predictions["split"].eq("train"), "actual_lmp"],
        split="test",
    )

    features.to_parquet(args.output_dir / "model_dataset.parquet", index=False)
    features.to_csv(args.output_dir / "model_dataset.csv", index=False)
    predictions.to_csv(args.output_dir / "ridge_t_plus_24_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "model_comparison_metrics.csv", index=False)
    coefficient_table(model, feature_cols).to_csv(args.output_dir / "ridge_coefficients.csv", index=False)
    pd.DataFrame(
        {
            "setting": ["horizon_hours", "alpha", "train_rows", "test_rows", "load_source", "features"],
            "value": [HORIZON_HOURS, args.alpha, len(train), len(test), load_label, ", ".join(feature_cols)],
        }
    ).to_csv(args.output_dir / "run_summary.csv", index=False)

    plot_true_vs_ridge(predictions, args.output_dir)
    plot_true_vs_all(predictions, args.output_dir)

    print(f"Forecast horizon: t+{HORIZON_HOURS}")
    print(f"Train rows: {len(train):,}; test rows: {len(test):,}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Load source: {load_label}")
    print(f"Wrote outputs under {args.output_dir}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
