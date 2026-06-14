from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DIR, RegionConfig
from src.utils import (
    choose_first_existing,
    normalize_columns,
    numeric_series,
    rolling_percentile,
    save_dataframe,
    timestamp_pair,
)


EIA_TYPE_MAP = {
    "D": "pjm_actual_load_mw",
    "DF": "pjm_forecast_load_mw",
    "NG": "pjm_net_generation_mw",
    "TI": "pjm_interchange_mw",
}


def transform_eia930(raw_df: pd.DataFrame, region: RegionConfig) -> pd.DataFrame:
    if raw_df.empty:
        return _empty_grid_frame(region)

    df = normalize_columns(raw_df)
    before = len(df)
    timestamp_col = choose_first_existing(df, ["period", "timestamp", "datetime"])
    if timestamp_col is None:
        raise ValueError("EIA-930 data is missing a timestamp column such as period.")
    if "type" not in df.columns:
        raise ValueError("EIA-930 data is missing the type column needed to identify demand/forecast/generation/interchange.")

    df["timestamp_utc"], df["timestamp_et"] = timestamp_pair(df[timestamp_col], source_tz=None, project_tz=region.timezone)
    df["value"] = numeric_series(df, ["value"])
    df["metric"] = df["type"].map(EIA_TYPE_MAP)
    df = df[df["metric"].notna()].copy()

    pivot = (
        df.pivot_table(index=["timestamp_utc", "timestamp_et"], columns="metric", values="value", aggfunc="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in EIA_TYPE_MAP.values():
        if col not in pivot.columns:
            pivot[col] = pd.NA

    pivot["pjm_load_forecast_error_mw"] = pivot["pjm_actual_load_mw"] - pivot["pjm_forecast_load_mw"]
    pivot["pjm_load_forecast_error_pct"] = pivot["pjm_load_forecast_error_mw"] / pivot["pjm_forecast_load_mw"] * 100
    pivot["load_peak_percentile_30d"] = rolling_percentile(pivot["pjm_actual_load_mw"], 24 * 30)
    pivot["load_peak_percentile_90d"] = rolling_percentile(pivot["pjm_actual_load_mw"], 24 * 90)
    pivot["is_top_5pct_load_90d"] = pivot["load_peak_percentile_90d"] >= 0.95
    pivot["is_top_1pct_load_90d"] = pivot["load_peak_percentile_90d"] >= 0.99
    pivot["reserve_margin_pct"] = pd.NA
    pivot["emergency_alert"] = pd.NA
    pivot["data_quality_flags"] = pd.NA
    pivot["source"] = "EIA-930"
    pivot["region_id"] = region.region_id
    pivot["rto"] = region.rto
    pivot["zone"] = region.zone
    pivot = pivot.sort_values("timestamp_utc").drop_duplicates(["timestamp_utc", "region_id"])

    print(f"EIA-930 transform rows before={before} after={len(pivot)}")
    save_dataframe(pivot, PROCESSED_DIR / "grid" / f"{region.region_id.lower()}_eia930_grid.parquet")
    save_dataframe(pivot, PROCESSED_DIR / "grid" / f"{region.region_id.lower()}_eia930_grid.csv")
    return pivot


def transform_pjm_load(raw_df: pd.DataFrame, region: RegionConfig) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(
            columns=["timestamp_utc", "timestamp_et", "region_id", "source", "dom_load_mw"]
        )

    df = normalize_columns(raw_df)
    before = len(df)
    timestamp_utc_col = choose_first_existing(df, ["datetime_beginning_utc", "datetime_utc", "timestamp_utc"])
    timestamp_et_col = choose_first_existing(df, ["datetime_beginning_ept", "datetime_beginning_ep", "datetime_ept"])
    if timestamp_utc_col:
        df["timestamp_utc"], df["timestamp_et"] = timestamp_pair(df[timestamp_utc_col], source_tz=None, project_tz=region.timezone)
    elif timestamp_et_col:
        df["timestamp_utc"], df["timestamp_et"] = timestamp_pair(df[timestamp_et_col], source_tz=region.timezone, project_tz=region.timezone)
    else:
        raise ValueError("PJM load data is missing datetime_beginning_utc/ept.")

    df["dom_load_mw"] = numeric_series(df, ["mw", "load_mw", "metered_load_mw", "load"])
    out = (
        df.groupby(["timestamp_utc", "timestamp_et"], as_index=False)["dom_load_mw"]
        .mean()
        .sort_values("timestamp_utc")
    )
    out["source"] = "PJM Data Miner hrl_load_metered"
    out["region_id"] = region.region_id
    print(f"PJM load transform rows before={before} after={len(out)}")
    save_dataframe(out, PROCESSED_DIR / "grid" / f"{region.region_id.lower()}_pjm_load.parquet")
    save_dataframe(out, PROCESSED_DIR / "grid" / f"{region.region_id.lower()}_pjm_load.csv")
    return out


def _empty_grid_frame(region: RegionConfig) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp_utc",
            "timestamp_et",
            "region_id",
            "rto",
            "zone",
            "source",
            *EIA_TYPE_MAP.values(),
            "pjm_load_forecast_error_mw",
            "pjm_load_forecast_error_pct",
            "load_peak_percentile_30d",
            "load_peak_percentile_90d",
            "is_top_5pct_load_90d",
            "is_top_1pct_load_90d",
        ]
    ).assign(region_id=region.region_id, rto=region.rto, zone=region.zone, source="EIA-930")
