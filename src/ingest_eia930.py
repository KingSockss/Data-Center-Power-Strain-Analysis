from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import METADATA_DIR, RAW_DIR, RegionConfig
from src.utils import append_source_log, normalize_columns, request_json, save_json, slugify


EIA_RTO_REGION_DATA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
EIA_RTO_TYPES = ["D", "DF", "NG", "TI"]


def _extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    response = payload.get("response", {})
    data = response.get("data", [])
    if not isinstance(data, list):
        raise ValueError("EIA response did not include response.data as a list")
    return data


def fetch_eia930(start_date: str, end_date: str, region: RegionConfig) -> pd.DataFrame:
    api_key = os.getenv("EIA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("EIA_API_KEY is required for EIA Open Data API calls.")

    raw_records: list[dict[str, Any]] = []
    offset = 0
    length = 5000
    page = 1

    while True:
        params: dict[str, Any] = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": region.rto,
            "facets[type][]": EIA_RTO_TYPES,
            "start": start_date,
            "end": end_date,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": length,
        }
        payload = request_json(EIA_RTO_REGION_DATA_URL, params=params)
        save_json(
            payload,
            RAW_DIR / "eia930" / f"{region.region_id}_{start_date}_{end_date}_page_{page}.json",
        )

        rows = _extract_rows(payload)
        raw_records.extend(rows)
        total = int(payload.get("response", {}).get("total", len(raw_records)) or len(raw_records))
        if offset + length >= total or not rows:
            break
        offset += length
        page += 1

    df = normalize_columns(pd.DataFrame(raw_records))
    append_source_log(
        METADATA_DIR / "source_log.csv",
        "EIA-930",
        "electricity/rto/region-data",
        start_date,
        end_date,
        len(df),
        "success",
        "PJM balancing-authority hourly operating data; may not be DOM-zone-level.",
    )
    df.to_csv(RAW_DIR / "eia930" / f"{slugify(region.region_id)}_{start_date}_{end_date}_records.csv", index=False)
    return df


def safe_fetch_eia930(start_date: str, end_date: str, region: RegionConfig) -> pd.DataFrame:
    try:
        return fetch_eia930(start_date, end_date, region)
    except Exception as exc:  # noqa: BLE001
        append_source_log(
            METADATA_DIR / "source_log.csv",
            "EIA-930",
            "electricity/rto/region-data",
            start_date,
            end_date,
            0,
            "failed",
            str(exc),
        )
        raise
