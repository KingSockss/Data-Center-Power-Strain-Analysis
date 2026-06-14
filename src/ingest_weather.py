from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import METADATA_DIR, RAW_DIR, RegionConfig, WeatherPoint
from src.utils import append_source_log, normalize_columns, request_json, save_json, slugify


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
]


def _point_to_frame(payload: dict[str, Any], point: WeatherPoint) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise ValueError("Open-Meteo response did not include hourly.time")
    df = pd.DataFrame(hourly)
    df["weather_point"] = point.name
    df["station_hint"] = point.station_hint
    df["latitude"] = point.latitude
    df["longitude"] = point.longitude
    df["point_weight"] = point.weight
    return normalize_columns(df)


def fetch_weather(start_date: str, end_date: str, region: RegionConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for point in region.weather_points:
        params: dict[str, Any] = {
            "latitude": point.latitude,
            "longitude": point.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(OPEN_METEO_HOURLY),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "UTC",
        }
        payload = request_json(OPEN_METEO_ARCHIVE_URL, params=params)
        save_json(payload, RAW_DIR / "weather" / f"{region.region_id}_{slugify(point.name)}_{start_date}_{end_date}.json")
        df = _point_to_frame(payload, point)
        frames.append(df)
        append_source_log(
            METADATA_DIR / "source_log.csv",
            "Open-Meteo",
            "historical-weather-api/archive",
            start_date,
            end_date,
            len(df),
            "success",
            f"Weather proxy point: {point.name} ({point.station_hint or 'no station hint'})",
        )

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(RAW_DIR / "weather" / f"{slugify(region.region_id)}_{start_date}_{end_date}_records.csv", index=False)
    return combined
