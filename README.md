# F1 Performance & Strategy Analytics

Local FastF1 + Kaggle pipeline for race engineering analyses (descriptive → diagnostic → predictive → prescriptive), scaled across 2020–2025 with resumable checkpoints.

This folder already includes:

- Source: `f1_analytics/`, `main.py`
- Kaggle CSVs: `Kaggel Data/` (2024 bundle; folder name is intentional)
- Checkpoints: `data/processed/` (all races 2020–2025)

```

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


## Wet / mixed races

- **wet** — majority of laps on INTERMEDIATE/WET  
- **mixed** — any wet-compound laps, or rainfall while still on dry compounds  
- **dry** — dry compounds only, no rainfall  

Degradation slopes use **dry compounds only**. Fully wet races are excluded from dry slope aggregates.

## Kaggle coverage

Only the **2024** Kaggle CSV bundle ships in `Kaggel Data/`. Other seasons run FastF1-only until you add matching `f1_{year}_race_results.csv` and `f1_qualifying_results_{year}.csv`. Round counts always come from `fastf1.get_event_schedule(year)`.


## Project layout

```
main.py                 CLI entry
requirements.txt
f1_analytics/           Analysis 
Kaggel Data/            Kaggle CSVs (2024)
data/processed/         Race parquet checkpoints

