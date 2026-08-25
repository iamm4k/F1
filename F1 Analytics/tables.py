"""Build race / stint / lap tables and merge Kaggle summaries."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_analytics.config import FUEL_CORRECTION_S_PER_LAP, KAGGLE_DIR
from f1_analytics.name_map import attach_driver_codes

logger = logging.getLogger(__name__)

TIMEDELTA_COLS = [
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "PitInTime",
    "PitOutTime",
]


def _td_to_seconds(series: pd.Series) -> pd.Series:
    return series.apply(
        lambda x: x.total_seconds() if pd.notna(x) and hasattr(x, "total_seconds") else np.nan
    )


def weather_summary(session: Any) -> dict[str, float | bool | None]:
    """Aggregate session weather into race-level scalars."""
    weather = getattr(session, "weather_data", None)
    if weather is None or len(weather) == 0:
        return {
            "AirTempMean": None,
            "TrackTempMean": None,
            "RainfallAny": None,
            "HumidityMean": None,
        }
    return {
        "AirTempMean": float(weather["AirTemp"].mean()),
        "TrackTempMean": float(weather["TrackTemp"].mean()),
        "RainfallAny": bool(weather["Rainfall"].fillna(False).any()),
        "HumidityMean": float(weather["Humidity"].mean()),
    }


def build_race_table(session: Any, year: int, round_number: int) -> pd.DataFrame:
    """One row per driver from session.results + weather summary."""
    results = session.results.copy()
    wx = weather_summary(session)
    cols = [
        "Abbreviation",
        "FullName",
        "TeamName",
        "GridPosition",
        "Position",
        "Points",
        "Status",
    ]
    race = results[cols].rename(
        columns={
            "Abbreviation": "DriverCode",
            "FullName": "DriverName",
            "TeamName": "Team",
        }
    )
    race["Year"] = year
    race["Round"] = round_number
    race["EventName"] = session.event["EventName"]
    for key, value in wx.items():
        race[key] = value
    return race.reset_index(drop=True)


def build_lap_table(
    session: Any,
    year: int,
    round_number: int,
    *,
    quicklaps_only: bool = True,
) -> pd.DataFrame:
    """Lap-level table with timedelta columns in seconds + fuel correction."""
    laps = session.laps.copy()
    if quicklaps_only:
        try:
            laps = laps.pick_quicklaps()
        except Exception as exc:  # noqa: BLE001
            logger.warning("pick_quicklaps failed (%s); using all laps", exc)

    lap = laps.copy()
    for col in TIMEDELTA_COLS:
        if col in lap.columns:
            lap[f"{col}Seconds"] = _td_to_seconds(lap[col])

    if "LapTimeSeconds" in lap.columns and "LapNumber" in lap.columns:
        lap["FuelCorrectedLapTime"] = (
            lap["LapTimeSeconds"] + FUEL_CORRECTION_S_PER_LAP * lap["LapNumber"].astype(float)
        )

    keep = [
        "Driver",
        "Team",
        "LapNumber",
        "Stint",
        "Compound",
        "TyreLife",
        "LapTimeSeconds",
        "FuelCorrectedLapTime",
        "Sector1TimeSeconds",
        "Sector2TimeSeconds",
        "Sector3TimeSeconds",
        "TrackStatus",
        "PitInTimeSeconds",
        "PitOutTimeSeconds",
        "IsAccurate",
    ]
    existing = [c for c in keep if c in lap.columns]
    out = lap[existing].copy()
    out = out.rename(columns={"Driver": "DriverCode"})
    out["Year"] = year
    out["Round"] = round_number
    return out.reset_index(drop=True)


def build_stint_table(lap_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate laps by Driver + Stint."""
    if lap_table.empty:
        return pd.DataFrame()

    grouped = lap_table.groupby(["Year", "Round", "DriverCode", "Stint"], dropna=False)
    rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        year, rnd, driver, stint = keys
        compound = group["Compound"].mode(dropna=True)
        compound_val = compound.iloc[0] if len(compound) else "UNKNOWN"
        rows.append(
            {
                "Year": year,
                "Round": rnd,
                "DriverCode": driver,
                "Stint": stint,
                "Compound": compound_val,
                "StintLength": int(len(group)),
                "TyreLifeStart": float(group["TyreLife"].min()) if "TyreLife" in group else np.nan,
                "TyreLifeEnd": float(group["TyreLife"].max()) if "TyreLife" in group else np.nan,
                "MeanLapTime": float(group["LapTimeSeconds"].mean()),
                "MedianLapTime": float(group["LapTimeSeconds"].median()),
                "MeanFuelCorrectedLapTime": float(group["FuelCorrectedLapTime"].mean())
                if "FuelCorrectedLapTime" in group
                else np.nan,
                "LapStart": float(group["LapNumber"].min()),
                "LapEnd": float(group["LapNumber"].max()),
            }
        )
    return pd.DataFrame(rows)


def kaggle_files_for_year(year: int, root: Path) -> dict[str, Path] | None:
    """
    Resolve Kaggle CSV paths for a season.

    Currently only the 2024 bundle is present locally. Other years return None
    so callers can fall back to FastF1-only tables.
    """
    if year == 2024:
        mapping = {
            "race_results": root / "f1_2024_race_results.csv",
            "driver_standings": root / "f1_2024_driver_standings.csv",
            "constructor_standings": root / "f1_2024_constructor_standings.csv",
            "qualifying": root / "f1_qualifying_results_2024.csv",
            "circuits": root / "f1_circuits_metadata.csv",
            "historical_drivers": root / "f1_historical_drivers.csv",
        }
        if all(path.exists() for path in mapping.values()):
            return mapping
    # Future: support f1_{year}_race_results.csv naming if added
    generic = {
        "race_results": root / f"f1_{year}_race_results.csv",
        "qualifying": root / f"f1_qualifying_results_{year}.csv",
    }
    if generic["race_results"].exists() and generic["qualifying"].exists():
        mapping = dict(generic)
        for key, name in {
            "driver_standings": f"f1_{year}_driver_standings.csv",
            "constructor_standings": f"f1_{year}_constructor_standings.csv",
            "circuits": "f1_circuits_metadata.csv",
            "historical_drivers": "f1_historical_drivers.csv",
        }.items():
            path = root / name
            if path.exists():
                mapping[key] = path
        return mapping
    return None


def load_kaggle(
    kaggle_dir: Path | None = None,
    *,
    year: int = 2024,
) -> dict[str, pd.DataFrame] | None:
    """Load the local Kaggle CSV bundle for a year, or None if unavailable."""
    root = Path(kaggle_dir or KAGGLE_DIR)
    files = kaggle_files_for_year(year, root)
    if files is None:
        logger.warning("No Kaggle CSV bundle found for %s under %s", year, root)
        return None
    return {key: pd.read_csv(path) for key, path in files.items()}


def merge_kaggle_race(
    race_table: pd.DataFrame,
    kaggle: dict[str, pd.DataFrame] | None,
    round_number: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Join FastF1 race-level to Kaggle race + qualifying on driver code."""
    if kaggle is None or "race_results" not in kaggle:
        merged = race_table.copy()
        for col in (
            "KaggleDriverName",
            "KaggleTeam",
            "KagglePosition",
            "KagglePoints",
            "KaggleRaceTime",
            "KaggleFastestLap",
            "KagglePole",
            "Circuit",
            "City",
            "Country",
            "QualiPosition",
            "QualiBestTime",
            "GapToPole",
            "EliminatedIn",
            "Q1",
            "Q2",
            "Q3",
        ):
            if col not in merged.columns:
                merged[col] = pd.NA
        return merged, []

    rr = kaggle["race_results"].copy()
    rr = rr[rr["race_id"].astype(int) == int(round_number)].copy()
    rr, unmatched_rr = attach_driver_codes(rr, "driver_name")

    quali = kaggle.get("qualifying", pd.DataFrame()).copy()
    unmatched_quali: list[str] = []
    if not quali.empty and "race_id" in quali.columns:
        quali = quali[quali["race_id"].astype(int) == int(round_number)].copy()
        quali, unmatched_quali = attach_driver_codes(quali, "driver_name")

    kaggle_keep = rr[
        [
            c
            for c in [
                "DriverCode",
                "driver_name",
                "team",
                "position",
                "points",
                "race_time",
                "fastest_lap",
                "pole_position",
                "circuit",
                "city",
                "country",
            ]
            if c in rr.columns
        ]
    ].rename(
        columns={
            "driver_name": "KaggleDriverName",
            "team": "KaggleTeam",
            "position": "KagglePosition",
            "points": "KagglePoints",
            "race_time": "KaggleRaceTime",
            "fastest_lap": "KaggleFastestLap",
            "pole_position": "KagglePole",
            "circuit": "Circuit",
            "city": "City",
            "country": "Country",
        }
    )

    if not quali.empty:
        quali_keep = quali[
            [
                c
                for c in [
                    "DriverCode",
                    "qualifying_position",
                    "best_time",
                    "gap_to_pole",
                    "eliminated_in",
                    "q1_time",
                    "q2_time",
                    "q3_time",
                ]
                if c in quali.columns
            ]
        ].rename(
            columns={
                "qualifying_position": "QualiPosition",
                "best_time": "QualiBestTime",
                "gap_to_pole": "GapToPole",
                "eliminated_in": "EliminatedIn",
                "q1_time": "Q1",
                "q2_time": "Q2",
                "q3_time": "Q3",
            }
        )
    else:
        quali_keep = pd.DataFrame({"DriverCode": []})

    merged = race_table.merge(kaggle_keep, on="DriverCode", how="left")
    if not quali_keep.empty:
        merged = merged.merge(quali_keep, on="DriverCode", how="left")

    unmatched = sorted(set(unmatched_rr + unmatched_quali))
    if "KaggleDriverName" in merged.columns:
        ff_only = merged[merged["KaggleDriverName"].isna()]["DriverName"].tolist()
        if ff_only:
            logger.warning("FastF1 drivers with no Kaggle match: %s", ff_only)
            unmatched = sorted(set(unmatched + [str(x) for x in ff_only]))

    return merged, unmatched


def build_all_tables(
    session: Any,
    year: int,
    round_number: int,
    kaggle: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame | list[str]]:
    """Build race, stint, lap tables and Kaggle merge for one race."""
    if kaggle is None:
        kaggle = load_kaggle(year=year)
    race = build_race_table(session, year, round_number)
    laps = build_lap_table(session, year, round_number)
    stints = build_stint_table(laps)
    merged, unmatched = merge_kaggle_race(race, kaggle, round_number)
    return {
        "race": race,
        "stints": stints,
        "laps": laps,
        "merged": merged,
        "unmatched_names": unmatched,
    }
