from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling_utils import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    compute_many_metrics,
    load_hourly_dataset,
    make_baseline_predictions,
)


BASELINE_MODEL_COLUMNS = {
    "persistence": "persistence",
    "daily_persistence": "daily_persistence",
    "pjm_day_ahead_lmp": "pjm_day_ahead_lmp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute LMP baseline forecasts and metrics.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET_PATH, help="Merged hourly parquet input.")
    parser.add_argument("--output-dir", type=Path, default=Path("Baselines/outputs"), help="Output directory.")
    parser.add_argument("--train-months", type=int, default=9, help="Number of first months treated as train period.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_hourly_dataset(args.input)
    predictions = make_baseline_predictions(df, train_months=args.train_months)
    train_actual = predictions.loc[predictions["split"].eq("train"), "actual_lmp"]
    metrics = compute_many_metrics(predictions, BASELINE_MODEL_COLUMNS, train_actual, split="test")

    predictions.to_csv(args.output_dir / "baseline_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "baseline_metrics.csv", index=False)

    print(f"Wrote {len(predictions):,} baseline prediction rows to {args.output_dir / 'baseline_predictions.csv'}")
    print(f"Wrote baseline metrics to {args.output_dir / 'baseline_metrics.csv'}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
