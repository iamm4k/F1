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
