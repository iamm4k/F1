"""Race condition tagging (dry / mixed / wet) and dry-only degradation filters."""

from __future__ import annotations

from typing import Any

import pandas as pd

from f1_analytics.config import DRY_COMPOUNDS, WET_COMPOUNDS


def tag_race_condition(laps: pd.DataFrame, race: pd.DataFrame | None = None) -> str:
    """
    Classify a race as dry / mixed / wet from compound usage + rainfall.

    - wet: majority of laps on INTER/WET (true wet races)
    - mixed: any INTER/WET laps, or rainfall recorded while still on dry compounds
    - dry: only SOFT/MEDIUM/HARD and no rainfall flag

    Brief weather RainfallAny without wet compounds must not exclude a race from
    dry-compound degradation — those races are tagged mixed, and slopes still use
    dry_laps_only().
    """
    rainfall = False
    if race is not None and not race.empty and "RainfallAny" in race.columns:
        rainfall = bool(race["RainfallAny"].fillna(False).astype(bool).any())

    if laps.empty or "Compound" not in laps.columns:
        if rainfall:
            return "mixed"
        return "dry"

    compounds = laps["Compound"].dropna().astype(str).str.upper()
    # FastF1 sometimes labels intermediates as INTERMEDIATE
    compounds = compounds.replace({"INTER": "INTERMEDIATE"})
    n = len(compounds)
    wet_n = int(compounds.isin(WET_COMPOUNDS).sum())
    if n > 0 and wet_n / n >= 0.5:
        return "wet"
    if wet_n > 0 or rainfall:
        return "mixed"
    return "dry"


def dry_laps_only(laps: pd.DataFrame) -> pd.DataFrame:
    """Laps on dry compounds only — never pool wet into dry degradation."""
    if laps.empty or "Compound" not in laps.columns:
        return laps.copy()
    mask = laps["Compound"].astype(str).str.upper().isin(DRY_COMPOUNDS)
    return laps.loc[mask].copy()


def condition_meta(laps: pd.DataFrame, race: pd.DataFrame | None = None) -> dict[str, Any]:
    tag = tag_race_condition(laps, race)
    compounds = (
        sorted(laps["Compound"].dropna().astype(str).str.upper().unique())
        if not laps.empty and "Compound" in laps.columns
        else []
    )
    return {
        "Condition": tag,
        "CompoundsPresent": ",".join(compounds),
        "DryLaps": int(len(dry_laps_only(laps))),
        "TotalLaps": int(len(laps)),
    }
