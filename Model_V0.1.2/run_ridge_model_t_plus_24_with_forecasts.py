from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling_utils import (  # noqa: E402
    DAY_AHEAD_COL,
    DEFAULT_DATASET_PATH,
    compute_many_metrics,
    load_hourly_dataset,
    write_dark_line_plot,
)


HORIZON_HOURS = 24
TARGET_LOAD_FORECAST_COL = "pjm_forecast_load_mw"
MODEL_NAME = "ridge_v0_1_2"
BASELINE_MODEL_COLUMNS = {
    "persistence_origin_lmp": "persistence_origin_lmp",
    "daily_persistence_origin_minus_24": "daily_persistence_origin_minus_24",
    "pjm_day_ahead_lmp_for_target": "pjm_day_ahead_lmp_for_target",
}


def load_v011_module():
    path = PROJECT_ROOT / "Model_V0.1.1" / "run_ridge_model_t_plus_24.py"
    spec = importlib.util.spec_from_file_location("model_v0_1_1", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Model V0.1.1 module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model_V0.1.2 Ridge regression for t+24 DOM RT LMP with forecast inputs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET_PATH, help="Merged hourly parquet input.")
    parser.add_argument("--output-dir", type=Path, default=Path("Model_V0.1.2/outputs"), help="Output directory.")
    parser.add_argument("--alpha", type=float, default=10.0, help="Ridge regularization strength.")
    parser.add_argument("--train-months", type=int, default=9, help="Number of first target months treated as train period.")
    return parser.parse_args()


def add_target_forecast_features(frame: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    required = [TARGET_LOAD_FORECAST_COL, DAY_AHEAD_COL, "timestamp_et"]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"Merged dataset missing required Model_V0.1.2 forecast feature columns: {missing}")

    raw_sorted = raw.sort_values("timestamp_et").reset_index(drop=True)
    forecast_features = pd.DataFrame(
        {
            "forecast_origin_et": raw_sorted["timestamp_et"],
            "target_pjm_forecast_load_mw": raw_sorted[TARGET_LOAD_FORECAST_COL].shift(-HORIZON_HOURS),
            "target_dom_da_lmp": raw_sorted[DAY_AHEAD_COL].shift(-HORIZON_HOURS),
        }
    )
    return frame.merge(forecast_features, on="forecast_origin_et", how="left")


def build_feature_frame(df: pd.DataFrame, train_months: int = 9) -> tuple[pd.DataFrame, list[str], str]:
    v011 = load_v011_module()
    frame, features, load_label = v011.build_feature_frame(df, train_months=train_months)
    frame = add_target_forecast_features(frame, df)
    added_features = ["target_pjm_forecast_load_mw", "target_dom_da_lmp"]
    frame = frame.dropna(subset=added_features).copy()
    return frame, [*features, *added_features], load_label


def plot_true_vs_ridge(predictions: pd.DataFrame, output_dir: Path) -> None:
    test = predictions[predictions["split"].eq("test")]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test["target_timestamp_et"], y=test["actual_lmp"], name="Actual DOM RT LMP", mode="lines"))
    fig.add_trace(
        go.Scatter(x=test["target_timestamp_et"], y=test[MODEL_NAME], name="Ridge V0.1.2 t+24 Forecast", mode="lines")
    )
    fig.update_layout(
        title="True DOM Real-Time LMP vs Ridge V0.1.2 t+24 Forecast",
        xaxis_title="Target Timestamp (ET)",
        yaxis_title="$/MWh",
    )
    write_dark_line_plot(fig, output_dir / "true_vs_ridge_t_plus_24_forecast.html")


def plot_true_vs_all(predictions: pd.DataFrame, output_dir: Path) -> None:
    test = predictions[predictions["split"].eq("test")]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test["target_timestamp_et"], y=test["actual_lmp"], name="Actual DOM RT LMP", mode="lines"))
    fig.add_trace(go.Scatter(x=test["target_timestamp_et"], y=test[MODEL_NAME], name="Ridge V0.1.2", mode="lines"))
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
        title="True DOM Real-Time LMP vs Ridge V0.1.2 and t+24 Baselines",
        xaxis_title="Target Timestamp (ET)",
        yaxis_title="$/MWh",
    )
    write_dark_line_plot(fig, output_dir / "true_vs_all_t_plus_24_models.html")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v011 = load_v011_module()

    raw = load_hourly_dataset(args.input)
    features, feature_cols, load_label = build_feature_frame(raw, train_months=args.train_months)
    train = features[features["split"].eq("train")]
    test = features[features["split"].eq("test")]
    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty train or test set.")

    model = v011.fit_ridge(train, feature_cols, alpha=args.alpha)
    features[MODEL_NAME] = v011.predict_ridge(model, features, feature_cols)

    prediction_cols = [
        "forecast_origin_utc",
        "forecast_origin_et",
        "target_timestamp_utc",
        "target_timestamp_et",
        "split",
        "actual_lmp",
        "reference_lmp_t_minus_1",
        MODEL_NAME,
        "persistence_origin_lmp",
        "daily_persistence_origin_minus_24",
        "pjm_day_ahead_lmp_for_target",
        "target_pjm_forecast_load_mw",
        "target_dom_da_lmp",
    ]
    predictions = features[prediction_cols].copy()

    model_columns = {MODEL_NAME: MODEL_NAME, **BASELINE_MODEL_COLUMNS}
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
    v011.coefficient_table(model, feature_cols).to_csv(args.output_dir / "ridge_coefficients.csv", index=False)
    pd.DataFrame(
        {
            "setting": ["horizon_hours", "alpha", "train_rows", "test_rows", "load_source", "features"],
            "value": [HORIZON_HOURS, args.alpha, len(train), len(test), load_label, ", ".join(feature_cols)],
        }
    ).to_csv(args.output_dir / "run_summary.csv", index=False)

    plot_true_vs_ridge(predictions, args.output_dir)
    plot_true_vs_all(predictions, args.output_dir)

    print(f"Forecast horizon: t+{HORIZON_HOURS}")
    print(f"Added target forecast features: target_pjm_forecast_load_mw, target_dom_da_lmp")
    print(f"Train rows: {len(train):,}; test rows: {len(test):,}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Load source: {load_label}")
    print(f"Wrote outputs under {args.output_dir}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
