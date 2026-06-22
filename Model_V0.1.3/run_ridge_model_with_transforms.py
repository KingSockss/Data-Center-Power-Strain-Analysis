from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling_utils import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    compute_many_metrics,
    load_hourly_dataset,
    write_dark_line_plot,
)


HORIZON_HOURS = 24
MODEL_NAME = "ridge_v0_1_3"
BASE_MODEL_NAME = "ridge_v0_1_2_same_rows"
BASELINE_MODEL_COLUMNS = {
    "persistence_origin_lmp": "persistence_origin_lmp",
    "daily_persistence_origin_minus_24": "daily_persistence_origin_minus_24",
    "pjm_day_ahead_lmp_for_target": "pjm_day_ahead_lmp_for_target",
}
TRANSFORM_FEATURES = [
    "target_load_x_congestion_t_minus_1",
    "target_load_x_hour_sin",
    "target_load_x_hour_cos",
    "temp_t_minus_1_squared",
    "congestion_t_minus_1_squared",
    "target_forecast_load_squared",
    "ln_target_dom_da_lmp",
    "load_congestion_squared_over_lmp_forecast",
]


def load_v012_module():
    path = PROJECT_ROOT / "Model_V0.1.2" / "run_ridge_model_t_plus_24_with_forecasts.py"
    spec = importlib.util.spec_from_file_location("model_v0_1_2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Model V0.1.2 module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model_V0.1.3 t+24 Ridge regression with nonlinear transforms.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET_PATH, help="Merged hourly parquet input.")
    parser.add_argument("--output-dir", type=Path, default=Path("Model_V0.1.3/outputs"), help="Output directory.")
    parser.add_argument("--alpha", type=float, default=10.0, help="Ridge regularization strength.")
    parser.add_argument("--train-months", type=int, default=9, help="Number of first target months treated as train period.")
    return parser.parse_args()


def add_transform_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    load = out["target_pjm_forecast_load_mw"].astype(float)
    congestion = out["congestion_t_minus_1"].astype(float)
    temperature = out["temp_t_minus_1"].astype(float)
    lmp_forecast = out["target_dom_da_lmp"].astype(float)

    load_x_congestion = load * congestion
    out["target_load_x_congestion_t_minus_1"] = load_x_congestion
    out["target_load_x_hour_sin"] = load * out["hour_sin"]
    out["target_load_x_hour_cos"] = load * out["hour_cos"]
    out["temp_t_minus_1_squared"] = np.square(temperature)
    out["congestion_t_minus_1_squared"] = np.square(congestion)
    out["target_forecast_load_squared"] = np.square(load)

    positive_lmp = lmp_forecast.gt(0)
    out["ln_target_dom_da_lmp"] = np.where(positive_lmp, np.log(lmp_forecast), np.nan)
    out["load_congestion_squared_over_lmp_forecast"] = np.where(
        positive_lmp,
        np.square(load_x_congestion) / lmp_forecast,
        np.nan,
    )
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=TRANSFORM_FEATURES).copy()


def transform_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"feature": "target_load_x_congestion_t_minus_1", "formula": "target_pjm_forecast_load_mw * congestion_t_minus_1"},
            {"feature": "target_load_x_hour_sin", "formula": "target_pjm_forecast_load_mw * hour_sin"},
            {"feature": "target_load_x_hour_cos", "formula": "target_pjm_forecast_load_mw * hour_cos"},
            {"feature": "temp_t_minus_1_squared", "formula": "temp_t_minus_1 ** 2"},
            {"feature": "congestion_t_minus_1_squared", "formula": "congestion_t_minus_1 ** 2"},
            {"feature": "target_forecast_load_squared", "formula": "target_pjm_forecast_load_mw ** 2"},
            {"feature": "ln_target_dom_da_lmp", "formula": "ln(target_dom_da_lmp), positive prices only"},
            {
                "feature": "load_congestion_squared_over_lmp_forecast",
                "formula": "(target_pjm_forecast_load_mw * congestion_t_minus_1) ** 2 / target_dom_da_lmp",
            },
        ]
    )


def plot_true_vs_ridge(predictions: pd.DataFrame, output_dir: Path) -> None:
    test = predictions[predictions["split"].eq("test")]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test["target_timestamp_et"], y=test["actual_lmp"], name="Actual DOM RT LMP", mode="lines"))
    fig.add_trace(go.Scatter(x=test["target_timestamp_et"], y=test[MODEL_NAME], name="Ridge V0.1.3", mode="lines"))
    fig.update_layout(
        title="True DOM Real-Time LMP vs Ridge V0.1.3 t+24 Forecast",
        xaxis_title="Target Timestamp (ET)",
        yaxis_title="$/MWh",
    )
    write_dark_line_plot(fig, output_dir / "true_vs_ridge_t_plus_24_forecast.html")


def plot_true_vs_all(predictions: pd.DataFrame, output_dir: Path) -> None:
    test = predictions[predictions["split"].eq("test")]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test["target_timestamp_et"], y=test["actual_lmp"], name="Actual DOM RT LMP", mode="lines"))
    fig.add_trace(go.Scatter(x=test["target_timestamp_et"], y=test[MODEL_NAME], name="Ridge V0.1.3", mode="lines"))
    fig.add_trace(go.Scatter(x=test["target_timestamp_et"], y=test[BASE_MODEL_NAME], name="Ridge V0.1.2 Same Rows", mode="lines"))
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
        title="True DOM Real-Time LMP vs Ridge V0.1.3, V0.1.2, and Baselines",
        xaxis_title="Target Timestamp (ET)",
        yaxis_title="$/MWh",
    )
    write_dark_line_plot(fig, output_dir / "true_vs_all_t_plus_24_models.html")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v012 = load_v012_module()
    v011 = v012.load_v011_module()

    raw = load_hourly_dataset(args.input)
    features, base_feature_cols, load_label = v012.build_feature_frame(raw, train_months=args.train_months)
    features = add_transform_features(features)
    feature_cols = [*base_feature_cols, *TRANSFORM_FEATURES]

    train = features[features["split"].eq("train")]
    test = features[features["split"].eq("test")]
    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty train or test set.")

    base_model = v011.fit_ridge(train, base_feature_cols, alpha=args.alpha)
    transformed_model = v011.fit_ridge(train, feature_cols, alpha=args.alpha)
    features[BASE_MODEL_NAME] = v011.predict_ridge(base_model, features, base_feature_cols)
    features[MODEL_NAME] = v011.predict_ridge(transformed_model, features, feature_cols)

    prediction_cols = [
        "forecast_origin_utc",
        "forecast_origin_et",
        "target_timestamp_utc",
        "target_timestamp_et",
        "split",
        "actual_lmp",
        "reference_lmp_t_minus_1",
        MODEL_NAME,
        BASE_MODEL_NAME,
        "persistence_origin_lmp",
        "daily_persistence_origin_minus_24",
        "pjm_day_ahead_lmp_for_target",
        "target_pjm_forecast_load_mw",
        "target_dom_da_lmp",
        *TRANSFORM_FEATURES,
    ]
    predictions = features[prediction_cols].copy()

    model_columns = {
        MODEL_NAME: MODEL_NAME,
        BASE_MODEL_NAME: BASE_MODEL_NAME,
        **BASELINE_MODEL_COLUMNS,
    }
    metrics = compute_many_metrics(
        predictions,
        model_columns,
        predictions.loc[predictions["split"].eq("train"), "actual_lmp"],
        split="test",
    )

    coefficients = v011.coefficient_table(transformed_model, feature_cols)
    features.to_parquet(args.output_dir / "model_dataset.parquet", index=False)
    features.to_csv(args.output_dir / "model_dataset.csv", index=False)
    predictions.to_csv(args.output_dir / "ridge_t_plus_24_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "model_comparison_metrics.csv", index=False)
    coefficients.to_csv(args.output_dir / "ridge_coefficients.csv", index=False)
    coefficients[coefficients["feature"].isin(TRANSFORM_FEATURES)].to_csv(
        args.output_dir / "transform_coefficients.csv",
        index=False,
    )
    transform_definitions().to_csv(args.output_dir / "transform_definitions.csv", index=False)
    pd.DataFrame(
        {
            "setting": ["horizon_hours", "alpha", "train_rows", "test_rows", "load_source", "base_features", "transform_features"],
            "value": [
                HORIZON_HOURS,
                args.alpha,
                len(train),
                len(test),
                load_label,
                ", ".join(base_feature_cols),
                ", ".join(TRANSFORM_FEATURES),
            ],
        }
    ).to_csv(args.output_dir / "run_summary.csv", index=False)

    plot_true_vs_ridge(predictions, args.output_dir)
    plot_true_vs_all(predictions, args.output_dir)

    print(f"Forecast horizon: t+{HORIZON_HOURS}")
    print(f"Train rows: {len(train):,}; test rows: {len(test):,}")
    print(f"Base feature count: {len(base_feature_cols)}; transform count: {len(TRANSFORM_FEATURES)}")
    print(f"Load source: {load_label}")
    print(f"Wrote outputs under {args.output_dir}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
