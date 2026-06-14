from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from src.config import METADATA_DIR, RAW_DIR, RegionConfig
from src.ingest_pjm_lmp import PJM_API_BASE
from src.utils import append_source_log, normalize_columns, pjm_headers, request_json, save_json, slugify


PJM_LOAD_FEED = "hrl_load_metered"


def _records_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("items", "data", "rows", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    raise ValueError("PJM load response did not include a recognized row list.")


def fetch_pjm_load(start_date: str, end_date: str, region: RegionConfig) -> pd.DataFrame:
    params: dict[str, Any] = {
        "rowCount": 50000,
        "startRow": 1,
        "datetime_beginning_ept": f"{start_date} 00:00 to {end_date} 23:59",
        "load_area": region.zone,
    }
    try:
        payload = request_json(f"{PJM_API_BASE}/{PJM_LOAD_FEED}", params=params, headers=pjm_headers())
    except RuntimeError as exc:
        cause = exc.__cause__
        if isinstance(cause, requests.HTTPError) and cause.response is not None and cause.response.status_code == 401:
            raise RuntimeError(
                "PJM Data Miner returned 401 Unauthorized. Check that PJM_API_KEY is set in .env, "
                "that the key is copied without quotes/spaces, and that it is subscribed/authorized "
                f"for the {PJM_LOAD_FEED} feed."
            ) from exc
        raise
    save_json(payload, RAW_DIR / "pjm_load" / f"{region.region_id}_{PJM_LOAD_FEED}_{start_date}_{end_date}.json")
    rows = _records_from_payload(payload)
    df = normalize_columns(pd.DataFrame(rows))

    if df.empty:
        # Try a common alternate field name before giving up.
        params.pop("load_area", None)
        params["area"] = region.zone
        payload = request_json(f"{PJM_API_BASE}/{PJM_LOAD_FEED}", params=params, headers=pjm_headers())
        save_json(payload, RAW_DIR / "pjm_load" / f"{region.region_id}_{PJM_LOAD_FEED}_{start_date}_{end_date}_area_retry.json")
        df = normalize_columns(pd.DataFrame(_records_from_payload(payload)))

    append_source_log(
        METADATA_DIR / "source_log.csv",
        "PJM Data Miner",
        PJM_LOAD_FEED,
        start_date,
        end_date,
        len(df),
        "success",
        "DOM hourly metered/service-territory load if available from public feed.",
    )
    df.to_csv(RAW_DIR / "pjm_load" / f"{slugify(region.region_id)}_{start_date}_{end_date}_records.csv", index=False)
    return df


def empty_pjm_load_frame(start_date: str, end_date: str, notes: str) -> pd.DataFrame:
    append_source_log(
        METADATA_DIR / "source_log.csv",
        "PJM Data Miner",
        PJM_LOAD_FEED,
        start_date,
        end_date,
        0,
        "not_available",
        notes,
    )
    return pd.DataFrame()
