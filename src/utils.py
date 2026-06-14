from __future__ import annotations

import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def snake_case(value: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(value))
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value)
    return value.strip("_").lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [snake_case(c) for c in df.columns]
    return df


def parse_date_arg(value: str) -> str:
    return pd.Timestamp(value).date().isoformat()


def request_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
    retries: int = 3,
    backoff_seconds: float = 1.5,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else None
            if status in {401, 403} or attempt == retries:
                break
            time.sleep(backoff_seconds * attempt)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retries:
                break
            time.sleep(backoff_seconds * attempt)
    raise RuntimeError(f"GET failed after {retries} attempts: {url}") from last_error


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported dataframe output format: {path}")


def timestamp_pair(
    series: pd.Series,
    source_tz: str | None = None,
    project_tz: str = "America/New_York",
) -> tuple[pd.Series, pd.Series]:
    ts = pd.to_datetime(series, errors="coerce")
    if ts.dt.tz is None:
        if source_tz:
            ts = ts.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="shift_forward")
        else:
            ts = ts.dt.tz_localize("UTC")
    timestamp_utc = ts.dt.tz_convert("UTC")
    timestamp_et = timestamp_utc.dt.tz_convert(project_tz)
    return timestamp_utc, timestamp_et


def choose_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    columns = set(df.columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def numeric_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    col = choose_first_existing(df, candidates)
    if col is None:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    return pd.to_numeric(df[col], errors="coerce")


def append_source_log(
    path: Path,
    source_name: str,
    endpoint_or_feed_name: str,
    start_date: str,
    end_date: str,
    row_count: int,
    status: str,
    notes: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "source_name": source_name,
        "endpoint_or_feed_name": endpoint_or_feed_name,
        "pull_timestamp_utc": utc_now_iso(),
        "start_date": start_date,
        "end_date": end_date,
        "row_count": row_count,
        "status": status,
        "notes": notes,
    }
    log_df = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_df.to_csv(path, index=False)


def pjm_headers() -> dict[str, str]:
    api_key = os.getenv("PJM_API_KEY", "").strip()
    if not api_key:
        return {}
    return {
        "Ocp-Apim-Subscription-Key": api_key,
        "subscription-key": api_key,
    }


def inclusive_hour_range(start: str, end: str, timezone: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start).tz_localize(timezone)
    end_ts = pd.Timestamp(end).tz_localize(timezone) + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    idx = pd.date_range(start_ts, end_ts, freq="h")
    return pd.DataFrame(
        {
            "timestamp_et": idx,
            "timestamp_utc": idx.tz_convert("UTC"),
        }
    )


def rolling_percentile(series: pd.Series, window_hours: int) -> pd.Series:
    return series.rolling(window_hours, min_periods=24).rank(pct=True).astype("Float64")


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
