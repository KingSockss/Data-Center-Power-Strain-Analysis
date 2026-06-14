from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DIR, RegionConfig
from src.utils import choose_first_existing, normalize_columns, numeric_series, rolling_percentile, save_dataframe, slugify, timestamp_pair


def _standard_location(value: object, requested: object | None = None) -> str:
    text = str(requested or value or "").upper().replace("-", "_").replace(" ", "_")
    if "WESTERN" in text and "HUB" in text:
        return "western_hub"
    if "DOM" in text or "DOMINION" in text:
        return "dom"
    return slugify(text) or "unknown"


def transform_prices(raw_df: pd.DataFrame, region: RegionConfig) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    df = normalize_columns(raw_df)
    before = len(df)
    feed_col = choose_first_existing(df, ["feed_name", "_feed_name"])
    if feed_col is None:
        raise ValueError("PJM LMP data is missing feed_name metadata.")

    timestamp_utc_col = choose_first_existing(df, ["datetime_beginning_utc", "datetime_utc", "timestamp_utc"])
    timestamp_et_col = choose_first_existing(df, ["datetime_beginning_ept", "datetime_beginning_ep", "datetime_ept"])
    if timestamp_utc_col:
        df["timestamp_utc"], df["timestamp_et"] = timestamp_pair(df[timestamp_utc_col], source_tz=None, project_tz=region.timezone)
    elif timestamp_et_col:
        df["timestamp_utc"], df["timestamp_et"] = timestamp_pair(df[timestamp_et_col], source_tz=region.timezone, project_tz=region.timezone)
    else:
        raise ValueError("PJM LMP data is missing datetime_beginning_utc/ept.")

    pnode_col = choose_first_existing(df, ["pnode_name", "pricing_node_name", "node_name", "location"])
    requested_col = choose_first_existing(df, ["requested_location"])
    requested_values = df[requested_col] if requested_col else pd.Series([None] * len(df), index=df.index)
    pnode_values = df[pnode_col] if pnode_col else pd.Series([None] * len(df), index=df.index)
    df["price_location"] = [
        _standard_location(pnode, requested)
        for pnode, requested in zip(pnode_values, requested_values, strict=False)
    ]

    rt = df[df[feed_col].astype(str).str.contains("rt", case=False, na=False)].copy()
    da = df[df[feed_col].astype(str).str.contains("da", case=False, na=False)].copy()

    rt_out = pd.DataFrame(
        {
            "timestamp_utc": rt["timestamp_utc"],
            "timestamp_et": rt["timestamp_et"],
            "price_location": rt["price_location"],
            "rt_lmp": numeric_series(rt, ["total_lmp_rt", "lmp", "total_lmp", "rt_lmp"]),
            "energy_component": numeric_series(rt, ["system_energy_price_rt", "energy_component", "energy_price", "energy"]),
            "congestion_component": numeric_series(rt, ["congestion_price_rt", "congestion_component", "congestion_price", "congestion"]),
            "loss_component": numeric_series(rt, ["marginal_loss_price_rt", "loss_component", "loss_price", "marginal_loss"]),
        }
    )
    da_out = pd.DataFrame(
        {
            "timestamp_utc": da["timestamp_utc"],
            "timestamp_et": da["timestamp_et"],
            "price_location": da["price_location"],
            "da_lmp": numeric_series(da, ["total_lmp_da", "lmp", "total_lmp", "da_lmp"]),
            "da_energy_component": numeric_series(da, ["system_energy_price_da", "energy_component", "energy_price", "energy"]),
            "da_congestion_component": numeric_series(da, ["congestion_price_da", "congestion_component", "congestion_price", "congestion"]),
            "da_loss_component": numeric_series(da, ["marginal_loss_price_da", "loss_component", "loss_price", "marginal_loss"]),
        }
    )

    keys = ["timestamp_utc", "timestamp_et", "price_location"]
    rt_out = rt_out.groupby(keys, as_index=False).mean(numeric_only=True)
    da_out = da_out.groupby(keys, as_index=False).mean(numeric_only=True)
    out = rt_out.merge(da_out, on=keys, how="outer").sort_values(["price_location", "timestamp_utc"])
    out["rt_da_spread"] = out["rt_lmp"] - out["da_lmp"]
    out["is_negative_rt_lmp"] = out["rt_lmp"] < 0
    out["is_negative_da_lmp"] = out["da_lmp"] < 0
    out["rolling_rt_lmp_percentile_30d"] = out.groupby("price_location")["rt_lmp"].transform(lambda s: rolling_percentile(s, 24 * 30))
    out["rolling_rt_lmp_percentile_90d"] = out.groupby("price_location")["rt_lmp"].transform(lambda s: rolling_percentile(s, 24 * 90))
    out["is_top_5pct_rt_lmp_90d"] = out["rolling_rt_lmp_percentile_90d"] >= 0.95
    out["is_top_1pct_rt_lmp_90d"] = out["rolling_rt_lmp_percentile_90d"] >= 0.99
    out["rolling_congestion_percentile_90d"] = out.groupby("price_location")["congestion_component"].transform(
        lambda s: rolling_percentile(s, 24 * 90)
    )
    out["is_top_5pct_congestion_90d"] = out["rolling_congestion_percentile_90d"] >= 0.95
    out["source"] = "PJM Data Miner LMP"
    out["region_id"] = region.region_id

    print(f"PJM LMP transform rows before={before} after={len(out)}")
    save_dataframe(out, PROCESSED_DIR / "prices" / f"{region.region_id.lower()}_prices.parquet")
    save_dataframe(out, PROCESSED_DIR / "prices" / f"{region.region_id.lower()}_prices.csv")
    return out
