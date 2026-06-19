# PJM DOM Power Grid Data Collection

This project is the first data-collection layer for a PJM / DOM power-grid and data-center infrastructure research workflow. It collects, standardizes, stores, lightly merges, and visualizes raw operating, price, and weather data for later analysis.

It does not build forecasting models, machine-learning pipelines, or causal claims about data-center electricity use.

## Focus

- Geography: PJM, DOM / Dominion zone.
- Practical interpretation: Dominion territory with emphasis on Northern Virginia, Ashburn, and Loudoun County data-center concentration.
- Weather proxy: Dulles Airport / Ashburn-area coordinates.

## Data Sources

- EIA Open Data API / EIA-930-style RTO data: PJM hourly actual demand, forecast demand, net generation, and interchange where available.
- PJM Data Miner: real-time hourly LMP and day-ahead hourly LMP for configured price locations.
- PJM Data Miner load feed: attempts DOM hourly metered/service-territory load if available.
- Open-Meteo archive API: hourly historical weather for configured weather points.

## Required API Keys

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Then set:

```text
EIA_API_KEY=your_eia_key
PJM_API_KEY=your_pjm_data_miner_subscription_key
```

Open-Meteo does not require an API key for the usage pattern in this project. PJM Data Miner API calls use `PJM_API_KEY`; if PJM returns `401 Unauthorized`, the key is missing, invalid, or not subscribed/authorized for the requested Data Miner feed.

The pipeline requires `EIA_API_KEY` before it can pull EIA-930 / EIA Open Data API data, and `PJM_API_KEY` before it can pull PJM LMP/load feeds.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Run the full EIA, PJM, weather, merge, and plotting pipeline:

```bash
python main.py --start 2024-01-01 --end 2024-12-31 --region PJM_DOM
```

If your shell or editor runs `/usr/local/bin/python3` directly, use the virtual environment path explicitly:

```bash
.venv/bin/python main.py --start 2024-01-01 --end 2024-12-31 --region PJM_DOM
```

If the PJM DOM load feed is unavailable but you still want the rest of the dataset:

```bash
python main.py --start 2024-01-01 --end 2024-12-31 --region PJM_DOM --continue-without-pjm-load
```

## Configuration

The first region is configured in `config/regions.yaml`:

- `PJM_DOM`
- RTO: `PJM`
- Zone: `DOM`
- Weather point: Dulles Airport / KIAD proxy
- Price locations: `DOM` and `PJM_WESTERN_HUB`

The pipeline is config-driven so additional PJM zones can be added later without rewriting the source modules.

## Outputs

Raw responses and records:

- `data/raw/eia930/`
- `data/raw/pjm_lmp/`
- `data/raw/pjm_load/`
- `data/raw/weather/`

Cleaned source-specific datasets:

- `data/processed/grid/`
- `data/processed/prices/`
- `data/processed/weather/`

Final merged hourly dataset:

- `data/processed/merged/hourly_pjm_dom_dataset.parquet`
- `data/processed/merged/hourly_pjm_dom_dataset.csv`

These merged files are the canonical latest run output and are overwritten by each run. Raw source pulls include the requested date range in their filenames for provenance.

Metadata:

- `data/metadata/source_log.csv`
- `data/metadata/data_dictionary.csv`
- `data/metadata/locations.csv`

Plotly sanity-check charts:

- `outputs/plots/*.html`

## Plot Outputs

The pipeline writes simple exploratory HTML charts for:

1. Actual load vs forecast load
2. Load forecast error
3. Real-time LMP vs day-ahead LMP
4. RT - DA LMP spread
5. Congestion component
6. Temperature and cooling degree hours
7. Temperature vs load
8. Temperature vs real-time LMP
9. Average load heatmap by hour and weekday
10. Average real-time LMP heatmap by hour and weekday
11. Load, temperature, and DOM congestion synchronized time overlay
12. DOM real-time LMP, temperature, and DOM congestion synchronized time overlay
13. Load and DOM real-time LMP synchronized time overlay

## First Models

Baseline forecasts live in `Baselines/`:

```bash
python Baselines/run_baselines.py
```

Model V0.1 lives in `Model_V0.1/`:

```bash
python Model_V0.1/run_ridge_model.py
```

Both scripts use the canonical merged dataset at `data/processed/merged/hourly_pjm_dom_dataset.parquet`. They train/reference on the first nine months and report test metrics on the final three months.

## Known Limitations

- EIA-930 / EIA RTO data is often balancing-authority-level. It is useful for PJM-wide operating context but is not a perfect DOM-zone load source.
- Data-center-specific electricity usage is usually not directly public. This project does not infer or claim direct data-center load.
- Weather is proxied by Dulles/Ashburn and should be expanded later if the analysis requires broader Dominion weather coverage.
- PJM reserve, emergency, and reliability-alert data can be limited or not exposed through the same public feeds. The merged dataset keeps nullable fields rather than inventing values.
- PJM Data Miner feed schemas and access requirements can differ. The code is schema-tolerant for common column names, but feed-level changes should fail clearly.
