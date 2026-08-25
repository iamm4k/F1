"""Season schedule helpers — never hardcode round counts."""

from __future__ import annotations

import logging
from typing import Any

import fastf1
import pandas as pd

logger = logging.getLogger(__name__)


def race_rounds(year: int) -> list[int]:
    """Return championship round numbers for a year (excludes testing)."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    rounds = (
        schedule.loc[schedule["RoundNumber"] > 0, "RoundNumber"]
        .astype(int)
        .drop_duplicates()
        .tolist()
    )
    # 2025 (and in-progress seasons): only keep rounds that already happened
    # or have API support — skip future events without data.
    if "EventDate" in schedule.columns:
        today = pd.Timestamp.now(tz=None).normalize()
        usable: list[int] = []
        for rnd in rounds:
            row = schedule.loc[schedule["RoundNumber"] == rnd].iloc[0]
            event_date = pd.Timestamp(row["EventDate"]).normalize()
            api_ok = bool(row.get("F1ApiSupport", True))
            if event_date <= today + pd.Timedelta(days=1) and api_ok:
                usable.append(int(rnd))
            elif event_date <= today + pd.Timedelta(days=1):
                usable.append(int(rnd))
        if usable:
            rounds = usable
    logger.info("%s schedule: %s race rounds", year, len(rounds))
    return rounds


def event_label(year: int, round_number: int, session: Any | None = None) -> str:
    if session is not None:
        try:
            return f"{year} {session.event['EventName']} · Race"
        except Exception:  # noqa: BLE001
            pass
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        row = schedule.loc[schedule["RoundNumber"] == round_number].iloc[0]
        return f"{year} {row['EventName']} · Race"
    except Exception:  # noqa: BLE001
        return f"{year} Round {round_number} · Race"
