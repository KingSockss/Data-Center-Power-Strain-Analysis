from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from src.config import METADATA_DIR, RAW_DIR, RegionConfig
from src.utils import append_source_log, normalize_columns, pjm_headers, request_json, save_json, slugify


PJM_API_BASE = "https://api.pjm.com/api/v1"
RT_LMP_FEED = "rt_hrl_lmps"
DA_LMP_FEED = "da_hrl_lmps"

PRICE_LOCATION_ALIASES = {
    "DOM": ["DOM", "DOMINION"],
    "PJM_WESTERN_HUB": ["PJM WESTERN HUB", "WESTERN HUB"],
}

PRICE_LOCATION_TYPES = {
    "DOM": "ZONE",
    "PJM_WESTERN_HUB": "HUB",
}


def _records_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("items", "data", "rows"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    if isinstance(payload.get("results"), list):
        return payload["results"]
    raise ValueError("PJM Data Miner response did not include a recognized row list.")


def _total_rows(payload: dict[str, Any], current_count: int) -> int:
    for key in ("totalRows", "total_rows", "totalRowCount", "total_count"):
        if key in payload:
            return int(payload[key])
    return current_count


def _has_next_link(payload: dict[str, Any]) -> bool:
    return any(link.get("rel") == "next" for link in payload.get("links", []))


def _matches_location(row: dict[str, Any], aliases: list[str]) -> bool:
    pnode_name = str(row.get("pnode_name", "")).upper().replace("-", " ").replace("_", " ")
    normalized_aliases = {alias.upper().replace("-", " ").replace("_", " ") for alias in aliases}
    return pnode_name in normalized_aliases


def _fetch_feed(
    feed_name: str,
    start_date: str,
    end_date: str,
    region: RegionConfig,
    location: str,
) -> pd.DataFrame:
    aliases = PRICE_LOCATION_ALIASES.get(location, [location])
    all_rows: list[dict[str, Any]] = []

    start_row = 1
    row_count = 50000
    location_type = PRICE_LOCATION_TYPES.get(location)
    while True:
        params: dict[str, Any] = {
            "rowCount": row_count,
            "startRow": start_row,
            "datetime_beginning_ept": f"{start_date} 00:00 to {end_date} 23:59",
            "row_is_current": 1,
        }
        if location_type:
            # PJM archived LMP data rejects pnode_name/pnode_id filters. Type is archive-filterable
            # and keeps zone/hub aggregate requests small; pnode_name is filtered locally below.
            params["type"] = location_type
        else:
            params["pnode_name"] = aliases[0]

        try:
            payload = request_json(f"{PJM_API_BASE}/{feed_name}", params=params, headers=pjm_headers())
        except RuntimeError as exc:
            cause = exc.__cause__
            if isinstance(cause, requests.HTTPError) and cause.response is not None and cause.response.status_code == 401:
                raise RuntimeError(
                    "PJM Data Miner returned 401 Unauthorized. Check that PJM_API_KEY is set in .env, "
                    "that the key is copied without quotes/spaces, and that it is subscribed/authorized "
                    f"for the {feed_name} feed."
                ) from exc
            raise
        save_json(
            payload,
            RAW_DIR / "pjm_lmp" / f"{region.region_id}_{feed_name}_{slugify(location)}_{start_row}.json",
        )
        rows = _records_from_payload(payload)
        if location_type:
            rows = [row for row in rows if _matches_location(row, aliases)]

        for row in rows:
            row["_requested_location"] = location
            row["_matched_alias"] = str(row.get("pnode_name", aliases[0]))
            row["_feed_name"] = feed_name
        all_rows.extend(rows)

        total = _total_rows(payload, len(all_rows))
        if not _records_from_payload(payload) or (not _has_next_link(payload) and start_row + row_count > total):
            break
        start_row += row_count

    return normalize_columns(pd.DataFrame(all_rows))


def fetch_pjm_lmp(start_date: str, end_date: str, region: RegionConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for feed_name in (RT_LMP_FEED, DA_LMP_FEED):
        for location in region.price_locations:
            try:
                df = _fetch_feed(feed_name, start_date, end_date, region, location)
            except Exception as exc:  # noqa: BLE001
                append_source_log(
                    METADATA_DIR / "source_log.csv",
                    "PJM Data Miner",
                    feed_name,
                    start_date,
                    end_date,
                    0,
                    "failed",
                    f"Requested price location: {location}; {exc}",
                )
                raise
            frames.append(df)
            append_source_log(
                METADATA_DIR / "source_log.csv",
                "PJM Data Miner",
                feed_name,
                start_date,
                end_date,
                len(df),
                "success",
                f"Requested price location: {location}",
            )

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(RAW_DIR / "pjm_lmp" / f"{slugify(region.region_id)}_{start_date}_{end_date}_records.csv", index=False)
    return combined
