from __future__ import annotations

import pandas as pd

from src.config import METADATA_DIR, PROCESSED_DIR, RegionConfig
from src.utils import append_source_log, inclusive_hour_range, save_dataframe


FINAL_COLUMNS = [
    "timestamp_utc",
    "timestamp_et",
    "region_id",
    "source",
    "rto",
    "zone",
    "pjm_actual_load_mw",
    "pjm_forecast_load_mw",
    "pjm_load_forecast_error_mw",
    "pjm_load_forecast_error_pct",
    "pjm_net_generation_mw",
    "pjm_interchange_mw",
    "dom_load_mw",
    "load_peak_percentile_30d",
    "load_peak_percentile_90d",
    "is_top_5pct_load_90d",
    "is_top_1pct_load_90d",
    "reserve_margin_pct",
    "emergency_alert",
    "dom_rt_lmp",
    "dom_da_lmp",
    "dom_rt_da_spread",
    "dom_energy_component",
    "dom_congestion_component",
    "dom_loss_component",
    "western_hub_rt_lmp",
    "western_hub_da_lmp",
    "dom_basis_to_western_hub",
    "is_negative_rt_lmp",
    "is_negative_da_lmp",
    "is_top_5pct_rt_lmp_90d",
    "is_top_1pct_rt_lmp_90d",
    "rolling_rt_lmp_percentile_30d",
    "rolling_rt_lmp_percentile_90d",
    "rolling_congestion_percentile_90d",
    "is_top_5pct_congestion_90d",
    "iad_temperature_f",
    "iad_dewpoint_f",
    "iad_relative_humidity",
    "iad_wind_speed",
    "iad_cloud_cover",
    "iad_precipitation",
    "iad_apparent_temperature_f",
    "iad_cooling_degree_hour",
    "iad_heating_degree_hour",
    "iad_temp_rolling_3h",
    "iad_temp_rolling_24h",
    "iad_cooling_degree_rolling_24h",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "date",
]


def _dedupe_hourly(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.empty:
        return df
    before = len(df)
    out = df.sort_values("timestamp_utc").drop_duplicates(["timestamp_utc", "region_id"], keep="last")
    print(f"{name} merge prep rows before={before} after={len(out)}")
    return out


def _price_wide(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=["timestamp_utc"])

    dom = prices[prices["price_location"].eq("dom")].copy()
    western = prices[prices["price_location"].eq("western_hub")].copy()

    dom_cols = {
        "rt_lmp": "dom_rt_lmp",
        "da_lmp": "dom_da_lmp",
        "rt_da_spread": "dom_rt_da_spread",
        "energy_component": "dom_energy_component",
        "congestion_component": "dom_congestion_component",
        "loss_component": "dom_loss_component",
    }
    dom_keep = [
        "timestamp_utc",
        *dom_cols,
        "is_negative_rt_lmp",
        "is_negative_da_lmp",
        "is_top_5pct_rt_lmp_90d",
        "is_top_1pct_rt_lmp_90d",
        "rolling_rt_lmp_percentile_30d",
        "rolling_rt_lmp_percentile_90d",
        "rolling_congestion_percentile_90d",
        "is_top_5pct_congestion_90d",
    ]
    dom = dom[[c for c in dom_keep if c in dom.columns]].rename(columns=dom_cols)

    western = western[["timestamp_utc", "rt_lmp", "da_lmp"]].rename(
        columns={"rt_lmp": "western_hub_rt_lmp", "da_lmp": "western_hub_da_lmp"}
    )
    out = dom.merge(western, on="timestamp_utc", how="outer")
    out["dom_basis_to_western_hub"] = out["dom_rt_lmp"] - out["western_hub_rt_lmp"]
    return out.drop_duplicates("timestamp_utc")


def _weather_wide(weather: pd.DataFrame) -> pd.DataFrame:
    if weather.empty:
        return pd.DataFrame(columns=["timestamp_utc"])

    point = weather[weather["weather_point_key"].eq("dulles_airport")].copy()
    if point.empty:
        point = weather.sort_values(["weather_point_key", "timestamp_utc"]).drop_duplicates("timestamp_utc")
    rename = {
        "temperature_f": "iad_temperature_f",
        "dewpoint_f": "iad_dewpoint_f",
        "relative_humidity": "iad_relative_humidity",
        "wind_speed": "iad_wind_speed",
        "cloud_cover": "iad_cloud_cover",
        "precipitation": "iad_precipitation",
        "apparent_temperature_f": "iad_apparent_temperature_f",
        "cooling_degree_hour": "iad_cooling_degree_hour",
        "heating_degree_hour": "iad_heating_degree_hour",
        "temp_rolling_3h": "iad_temp_rolling_3h",
        "temp_rolling_24h": "iad_temp_rolling_24h",
        "cooling_degree_rolling_24h": "iad_cooling_degree_rolling_24h",
    }
    keep = ["timestamp_utc", *rename]
    return point[[c for c in keep if c in point.columns]].rename(columns=rename).drop_duplicates("timestamp_utc")


def merge_hourly(
    grid: pd.DataFrame,
    pjm_load: pd.DataFrame,
    prices: pd.DataFrame,
    weather: pd.DataFrame,
    start_date: str,
    end_date: str,
    region: RegionConfig,
) -> pd.DataFrame:
    base = inclusive_hour_range(start_date, end_date, region.timezone)
    base["region_id"] = region.region_id
    base["source"] = "merged"
    base["rto"] = region.rto
    base["zone"] = region.zone

    grid = _dedupe_hourly(grid, "Grid")
    if not grid.empty:
        drop = [c for c in ["timestamp_et", "source", "rto", "zone"] if c in grid.columns]
        base = base.merge(grid.drop(columns=drop), on=["timestamp_utc", "region_id"], how="left")

    pjm_load = _dedupe_hourly(pjm_load, "PJM load")
    if not pjm_load.empty:
        drop = [c for c in ["timestamp_et", "source"] if c in pjm_load.columns]
        base = base.merge(pjm_load.drop(columns=drop), on=["timestamp_utc", "region_id"], how="left")

    price_wide = _price_wide(prices)
    base = base.merge(price_wide, on="timestamp_utc", how="left")

    weather_wide = _weather_wide(weather)
    base = base.merge(weather_wide, on="timestamp_utc", how="left")

    base["timestamp_et"] = base["timestamp_utc"].dt.tz_convert(region.timezone)
    base["hour"] = base["timestamp_et"].dt.hour
    base["day_of_week"] = base["timestamp_et"].dt.day_name()
    base["month"] = base["timestamp_et"].dt.month
    base["is_weekend"] = base["timestamp_et"].dt.dayofweek >= 5
    base["date"] = base["timestamp_et"].dt.date.astype(str)

    for col in FINAL_COLUMNS:
        if col not in base.columns:
            base[col] = pd.NA
    out = base[FINAL_COLUMNS].sort_values("timestamp_utc").drop_duplicates(["timestamp_utc", "region_id"])
    print(f"Merged hourly rows={len(out)}")

    parquet_path = PROCESSED_DIR / "merged" / "hourly_pjm_dom_dataset.parquet"
    csv_path = PROCESSED_DIR / "merged" / "hourly_pjm_dom_dataset.csv"
    save_dataframe(out, parquet_path)
    save_dataframe(out, csv_path)
    write_data_dictionary(out)
    append_source_log(
        METADATA_DIR / "source_log.csv",
        "project_pipeline",
        "merge_hourly",
        start_date,
        end_date,
        len(out),
        "success",
        "Final hourly merged dataset created with one row per timestamp/region.",
    )
    return out


def write_data_dictionary(df: pd.DataFrame) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    descriptions = {
        "timestamp_utc": "Timezone-aware hourly timestamp in UTC.",
        "timestamp_et": "Timezone-aware hourly timestamp in US/Eastern.",
        "region_id": "Configured project region identifier.",
        "source": "Dataset construction source marker.",
        "rto": "Regional transmission organization.",
        "zone": "PJM zone/service-territory code.",
        "pjm_actual_load_mw": "PJM balancing-authority actual demand/load from EIA-930.",
        "pjm_forecast_load_mw": "PJM balancing-authority forecast demand/load from EIA-930.",
        "pjm_load_forecast_error_mw": "Actual load minus forecast load.",
        "pjm_load_forecast_error_pct": "Load forecast error divided by forecast load.",
        "pjm_net_generation_mw": "PJM balancing-authority net generation from EIA-930.",
        "pjm_interchange_mw": "PJM balancing-authority interchange from EIA-930.",
        "dom_load_mw": "DOM hourly metered/service-territory load from PJM if available.",
        "dom_rt_lmp": "DOM real-time hourly LMP.",
        "dom_da_lmp": "DOM day-ahead hourly LMP.",
        "dom_rt_da_spread": "DOM real-time LMP minus day-ahead LMP.",
        "dom_basis_to_western_hub": "DOM real-time LMP minus PJM Western Hub real-time LMP.",
        "iad_temperature_f": "Hourly temperature at Dulles/Ashburn proxy in Fahrenheit.",
        "iad_cooling_degree_hour": "max(temperature_f - 65, 0).",
        "iad_heating_degree_hour": "max(65 - temperature_f, 0).",
    }
    rows = [
        {
            "column_name": col,
            "dtype": str(df[col].dtype),
            "description": descriptions.get(col, col.replace("_", " ")),
        }
        for col in df.columns
    ]
    pd.DataFrame(rows).to_csv(METADATA_DIR / "data_dictionary.csv", index=False)
