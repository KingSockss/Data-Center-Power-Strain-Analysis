from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DIR, RegionConfig
from src.utils import choose_first_existing, normalize_columns, numeric_series, save_dataframe, slugify, timestamp_pair


def transform_weather(raw_df: pd.DataFrame, region: RegionConfig) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    df = normalize_columns(raw_df)
    before = len(df)
    timestamp_col = choose_first_existing(df, ["time", "timestamp", "datetime"])
    if timestamp_col is None:
        raise ValueError("Weather data is missing time/timestamp column.")

    df["timestamp_utc"], df["timestamp_et"] = timestamp_pair(df[timestamp_col], source_tz=None, project_tz=region.timezone)
    df["temperature_f"] = numeric_series(df, ["temperature_2m", "temperature_f"])
    df["dewpoint_f"] = numeric_series(df, ["dew_point_2m", "dewpoint_f", "dew_point_f"])
    df["relative_humidity"] = numeric_series(df, ["relative_humidity_2m", "relative_humidity"])
    df["apparent_temperature_f"] = numeric_series(df, ["apparent_temperature", "apparent_temperature_f"])
    df["wind_speed"] = numeric_series(df, ["wind_speed_10m", "wind_speed"])
    df["cloud_cover"] = numeric_series(df, ["cloud_cover"])
    df["precipitation"] = numeric_series(df, ["precipitation"])
    df["cooling_degree_hour"] = (df["temperature_f"] - 65).clip(lower=0)
    df["heating_degree_hour"] = (65 - df["temperature_f"]).clip(lower=0)

    point_col = choose_first_existing(df, ["weather_point"])
    df["weather_point_key"] = df[point_col].map(slugify) if point_col else "weather_point"
    out_frames: list[pd.DataFrame] = []
    for _, point_df in df.sort_values("timestamp_utc").groupby("weather_point_key"):
        point_df = point_df.copy()
        point_df["temp_rolling_3h"] = point_df["temperature_f"].rolling(3, min_periods=1).mean()
        point_df["temp_rolling_24h"] = point_df["temperature_f"].rolling(24, min_periods=1).mean()
        point_df["cooling_degree_rolling_24h"] = point_df["cooling_degree_hour"].rolling(24, min_periods=1).sum()
        out_frames.append(point_df)

    out = pd.concat(out_frames, ignore_index=True)
    keep = [
        "timestamp_utc",
        "timestamp_et",
        "weather_point_key",
        "temperature_f",
        "dewpoint_f",
        "relative_humidity",
        "apparent_temperature_f",
        "wind_speed",
        "cloud_cover",
        "precipitation",
        "cooling_degree_hour",
        "heating_degree_hour",
        "temp_rolling_3h",
        "temp_rolling_24h",
        "cooling_degree_rolling_24h",
    ]
    out = out[keep].sort_values(["weather_point_key", "timestamp_utc"])
    out["source"] = "Open-Meteo"
    out["region_id"] = region.region_id

    print(f"Weather transform rows before={before} after={len(out)}")
    save_dataframe(out, PROCESSED_DIR / "weather" / f"{region.region_id.lower()}_weather.parquet")
    save_dataframe(out, PROCESSED_DIR / "weather" / f"{region.region_id.lower()}_weather.csv")
    return out
