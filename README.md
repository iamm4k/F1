# F1 Performance & Strategy Analytics

Local FastF1 + Kaggle pipeline for race engineering analyses (descriptive → diagnostic → predictive → prescriptive), scaled across 2020–2025 with resumable checkpoints.

Requirements

- **Python 3.10–3.13** (3.11 or 3.12 recommended)
- Internet on first FastF1 download (later runs use `./fastf1_cache`)
- ~2 GB free disk for cache + figures if regenerating from scratch
- Windows, macOS, or Linux

This folder already includes:

- Source: `f1_analytics/`, `main.py`
- Kaggle CSVs: `Kaggel Data/` (2024 bundle; folder name is intentional)
- Checkpoints: `data/processed/` (all races 2020–2025)
- Figures: `output/figures/`
- Audits: `data/quality/`


## Setup

From this project root:

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/verify_install.py
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/verify_install.py
```

**Cache:** FastF1 writes to `./fastf1_cache` on **local disk**. Avoid OneDrive/network/synced folders for the project or cache path.

**Windows note:** If `xgboost` fails to import, install the latest [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

## Quick start

```powershell
# Smoke test — 2024 Bahrain (uses checkpoint if present)
python main.py --year 2024 --round 1

# Full season (resumable; skips races already in data/processed)
python main.py --season --year 2024

# Multi-season
python main.py --years 2020-2025

# Season slice (smoke)
python main.py --season --year 2023 --max-rounds 3

# Cross-season roll-ups from checkpoints only (no FastF1 needed if processed exists)
python main.py --cross-season --years 2020-2025

# Checkpoints only
python main.py --season --year 2024 --no-figures
```

On Windows you can also use `.\.venv\Scripts\python.exe main.py ...` without activating the venv.

Telemetry overlays are **off** in season mode by default. Use `--with-telemetry` for a single race when needed. Use `--force` to rebuild checkpoints/figures.

## Output layout

```
data/processed/{year}/r{RR}_{race,stints,laps,merged}.parquet
data/processed/{year}/r{RR}_meta.json
data/quality/…
output/figures/{year}/r{RR}/*.png|*.svg
output/figures/{year}/season/*.png|*.svg
output/figures/cross_season/*.png|*.svg
output/manifest.csv
```

## Resumable design

After each race, parquet + meta are written under `data/processed/`. Reruns **skip** FastF1 reloads for checkpointed races. Figure regen is skipped when the pace-vs-lap PNG already exists (unless `--force`).

## Rate limits & runtime

FastF1 ≈ **500 API calls/hour**. First uncached pass over ~120 races can take **2–3 hours**. Prefer one season at a time. With `data/processed/` already present, figure regeneration and cross-season roll-ups are much faster and mostly offline.

## Wet / mixed races

- **wet** — majority of laps on INTERMEDIATE/WET  
- **mixed** — any wet-compound laps, or rainfall while still on dry compounds  
- **dry** — dry compounds only, no rainfall  

Degradation slopes use **dry compounds only**. Fully wet races are excluded from dry slope aggregates.

## Kaggle coverage

Only the **2024** Kaggle CSV bundle ships in `Kaggel Data/`. Other seasons run FastF1-only until you add matching `f1_{year}_race_results.csv` and `f1_qualifying_results_{year}.csv`. Round counts always come from `fastf1.get_event_schedule(year)`.

## Known failure modes

1. Rate-limit / schedule errors → retry+backoff + local cache; use `--debug`.
2. Missing session → logged in `data/quality/skipped_races.csv`.
3. Name mismatches → `data/quality/unmatched.csv` (never silent drop).

## Project layout

```
main.py                 CLI entry
requirements.txt
f1_analytics/           Package (analyses, loaders, viz)
Kaggel Data/            Kaggle CSVs (2024)
data/processed/         Race parquet checkpoints
data/quality/           Audits + schema notes
output/figures/         PNG + SVG figures
scripts/verify_install.py
Project.md              Original product brief
```
