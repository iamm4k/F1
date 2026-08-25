# Kaggle schema audit (2020–2025)

## Local inventory (this machine)

Only the **2024** Kaggle bundle is present under `Kaggel Data/`:

| File | Present |
|---|---|
| `f1_2024_race_results.csv` | yes |
| `f1_2024_driver_standings.csv` | yes |
| `f1_2024_constructor_standings.csv` | yes |
| `f1_qualifying_results_2024.csv` | yes |
| `f1_circuits_metadata.csv` | yes |
| `f1_historical_drivers.csv` | yes |
| `f1_20{20–23,25}_*.csv` | **no** |

## 2024 column check (confirmed)

- `f1_2024_race_results.csv`: `race_id, race_name, circuit, city, country, circuit_length_km, total_laps, position, driver_name, team, nationality, car_number, race_time, points, fastest_lap, pole_position`
- `f1_qualifying_results_2024.csv`: includes `gap_to_pole`, `best_time`, `qualifying_position`, Q1–Q3
- `race_id` == championship round number

## Adaptation

- Loader resolves year-specific files via `kaggle_files_for_year()`.
- If a season has no Kaggle CSVs, the pipeline continues **FastF1-only**: race/stint/lap tables still build; Kaggle merge columns are left empty; quali-gap figures may be sparse.
- When additional season CSVs are dropped into `Kaggel Data/` using `f1_{year}_race_results.csv` + `f1_qualifying_results_{year}.csv`, they are picked up automatically.

## FastF1 schedule

Round counts come from `fastf1.get_event_schedule(year, include_testing=False)` — never hardcoded. 2025 is truncated to events whose `EventDate` is not in the future.
