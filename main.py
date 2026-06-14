from __future__ import annotations

import argparse
import os
from typing import Any


def load_pipeline_modules() -> dict[str, Any]:
    try:
        import pandas as pd

        from src.config import METADATA_DIR, ensure_project_dirs, get_region, load_environment
        from src.ingest_eia930 import safe_fetch_eia930
        from src.ingest_pjm_lmp import fetch_pjm_lmp
        from src.ingest_pjm_load import empty_pjm_load_frame, fetch_pjm_load
        from src.ingest_weather import fetch_weather
        from src.make_plots import make_plots
        from src.merge_hourly import merge_hourly
        from src.transform_grid import transform_eia930, transform_pjm_load
        from src.transform_prices import transform_prices
        from src.transform_weather import transform_weather
        from src.utils import append_source_log, parse_date_arg
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        raise SystemExit(
            f"Missing Python dependency: {missing}\n\n"
            "Install project dependencies and run with the project environment:\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  pip install -r requirements.txt\n"
            "  python main.py --start 2024-01-01 --end 2024-01-31 --region PJM_DOM\n"
        ) from exc

    return locals()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and lightly process PJM/DOM grid, price, and weather data.")
    parser.add_argument("--start", required=True, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="End date, YYYY-MM-DD.")
    parser.add_argument("--region", default="PJM_DOM", help="Region id from config/regions.yaml.")
    parser.add_argument(
        "--enable-pjm",
        action="store_true",
        help="Enable PJM Data Miner LMP/load pulls. PJM is skipped by default until API access is ready.",
    )
    parser.add_argument(
        "--continue-without-pjm-load",
        action="store_true",
        help="With --enable-pjm, continue with nullable DOM load if the PJM load feed is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modules = load_pipeline_modules()
    METADATA_DIR = modules["METADATA_DIR"]
    append_source_log = modules["append_source_log"]
    empty_pjm_load_frame = modules["empty_pjm_load_frame"]
    ensure_project_dirs = modules["ensure_project_dirs"]
    fetch_pjm_lmp = modules["fetch_pjm_lmp"]
    fetch_pjm_load = modules["fetch_pjm_load"]
    fetch_weather = modules["fetch_weather"]
    get_region = modules["get_region"]
    load_environment = modules["load_environment"]
    make_plots = modules["make_plots"]
    merge_hourly = modules["merge_hourly"]
    pd = modules["pd"]
    parse_date_arg = modules["parse_date_arg"]
    safe_fetch_eia930 = modules["safe_fetch_eia930"]
    transform_eia930 = modules["transform_eia930"]
    transform_pjm_load = modules["transform_pjm_load"]
    transform_prices = modules["transform_prices"]
    transform_weather = modules["transform_weather"]

    start_date = parse_date_arg(args.start)
    end_date = parse_date_arg(args.end)
    if start_date > end_date:
        raise ValueError("--start must be on or before --end")

    load_environment()
    ensure_project_dirs()
    validate_required_environment(enable_pjm=args.enable_pjm)
    region = get_region(args.region)
    write_locations_metadata(region, METADATA_DIR)

    print(f"Running pipeline for {region.region_id} from {start_date} to {end_date}")

    raw_eia = safe_fetch_eia930(start_date, end_date, region)
    grid = transform_eia930(raw_eia, region)

    if args.enable_pjm:
        raw_prices = fetch_pjm_lmp(start_date, end_date, region)
        prices = transform_prices(raw_prices, region)

        try:
            raw_pjm_load = fetch_pjm_load(start_date, end_date, region)
        except Exception as exc:  # noqa: BLE001
            if not args.continue_without_pjm_load:
                raise RuntimeError(
                    "PJM DOM load feed was unavailable or did not match the expected schema. "
                    "Re-run with --enable-pjm --continue-without-pjm-load to keep nullable dom_load_mw."
                ) from exc
            raw_pjm_load = empty_pjm_load_frame(start_date, end_date, str(exc))
        pjm_load = transform_pjm_load(raw_pjm_load, region)
    else:
        print("Skipping PJM Data Miner LMP/load pulls. Re-enable later with --enable-pjm.")
        append_source_log(
            METADATA_DIR / "source_log.csv",
            "PJM Data Miner",
            "rt_hrl_lmps, da_hrl_lmps, hrl_load_metered",
            start_date,
            end_date,
            0,
            "skipped",
            "PJM temporarily disabled until API access is ready; use --enable-pjm to run these feeds.",
        )
        prices = transform_prices(pd.DataFrame(), region)
        pjm_load = transform_pjm_load(pd.DataFrame(), region)

    raw_weather = fetch_weather(start_date, end_date, region)
    weather = transform_weather(raw_weather, region)

    merged = merge_hourly(grid, pjm_load, prices, weather, start_date, end_date, region)
    plot_paths = make_plots(merged)

    append_source_log(
        METADATA_DIR / "source_log.csv",
        "project_pipeline",
        "make_plots",
        start_date,
        end_date,
        len(plot_paths),
        "success",
        "Plotly sanity-check HTML charts written to outputs/plots.",
    )
    print(f"Pipeline complete. Merged rows={len(merged)} plots={len(plot_paths)}")


def validate_required_environment(enable_pjm: bool = False) -> None:
    if not os.getenv("EIA_API_KEY", "").strip():
        raise SystemExit(
            "Missing required EIA_API_KEY.\n\n"
            "Create or update .env in the project root:\n"
            "  EIA_API_KEY=your_eia_api_key\n"
            "  PJM_API_KEY=your_pjm_data_miner_subscription_key_optional_until_pjm_is_enabled\n\n"
            "Then rerun:\n"
            "  python main.py --start 2024-01-01 --end 2024-01-31 --region PJM_DOM\n"
        )
    if enable_pjm and not os.getenv("PJM_API_KEY", "").strip():
        raise SystemExit(
            "Missing required PJM_API_KEY.\n\n"
            "The PJM Data Miner API returned 401 Unauthorized without a subscription key. "
            "Create or update .env in the project root:\n"
            "  EIA_API_KEY=your_eia_api_key\n"
            "  PJM_API_KEY=your_pjm_data_miner_subscription_key\n\n"
            "Then rerun:\n"
            "  python main.py --start 2024-01-01 --end 2024-01-31 --region PJM_DOM --enable-pjm\n"
        )


def write_locations_metadata(region, metadata_dir) -> None:
    rows = []
    for point in region.weather_points:
        rows.append(
            {
                "region_id": region.region_id,
                "location_name": point.name,
                "location_type": "weather_proxy",
                "station_hint": point.station_hint,
                "latitude": point.latitude,
                "longitude": point.longitude,
                "weight": point.weight,
                "notes": "Dulles/Ashburn proxy for Northern Virginia DOM data-center concentration.",
            }
        )
    for location in region.price_locations:
        rows.append(
            {
                "region_id": region.region_id,
                "location_name": location,
                "location_type": "price_location",
                "station_hint": "",
                "latitude": "",
                "longitude": "",
                "weight": "",
                "notes": "Configured PJM LMP location or hub.",
            }
        )

    import pandas as pd

    pd.DataFrame(rows).to_csv(metadata_dir / "locations.csv", index=False)


if __name__ == "__main__":
    main()
