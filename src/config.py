from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"


class WeatherPoint(BaseModel):
    name: str
    station_hint: str | None = None
    latitude: float
    longitude: float
    weight: float = 1.0


class RegionConfig(BaseModel):
    region_id: str
    rto: str
    zone: str
    zone_name: str
    timezone: str = "America/New_York"
    weather_points: list[WeatherPoint] = Field(default_factory=list)
    price_locations: list[str] = Field(default_factory=list)


def load_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def load_regions(config_path: Path | None = None) -> dict[str, RegionConfig]:
    path = config_path or CONFIG_DIR / "regions.yaml"
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    regions: dict[str, RegionConfig] = {}
    for region_id, payload in raw.items():
        regions[region_id] = RegionConfig(region_id=region_id, **payload)
    return regions


def get_region(region_id: str) -> RegionConfig:
    regions = load_regions()
    if region_id not in regions:
        valid = ", ".join(sorted(regions))
        raise ValueError(f"Unknown region '{region_id}'. Available regions: {valid}")
    return regions[region_id]


def ensure_project_dirs() -> None:
    for path in [
        RAW_DIR / "eia930",
        RAW_DIR / "pjm_lmp",
        RAW_DIR / "pjm_load",
        RAW_DIR / "weather",
        PROCESSED_DIR / "grid",
        PROCESSED_DIR / "prices",
        PROCESSED_DIR / "weather",
        PROCESSED_DIR / "merged",
        METADATA_DIR,
        PLOTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
