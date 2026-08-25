"""Data-quality audit writers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from f1_analytics.config import QUALITY_DIR, ensure_dirs


def append_csv(path: Path, rows: list[dict] | pd.DataFrame) -> None:
    ensure_dirs()
    frame = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    if frame.empty:
        return
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing.dropna(axis=1, how="all")
        frame = frame.dropna(axis=1, how="all")
        if existing.empty:
            combined = frame
        else:
            combined = pd.concat([existing, frame], ignore_index=True)
        frame = combined.drop_duplicates()
    frame.to_csv(path, index=False)


def ensure_audit_files() -> None:
    """Create empty audit CSVs with headers so paths always exist."""
    ensure_dirs()
    defaults = {
        QUALITY_DIR / "skipped_races.csv": ["Year", "Round", "Reason"],
        QUALITY_DIR / "unmatched.csv": ["Year", "Round", "Name", "Status"],
        QUALITY_DIR / "race_conditions.csv": [
            "Year",
            "Round",
            "Condition",
            "CompoundsPresent",
            "DryLaps",
            "TotalLaps",
            "EventName",
        ],
    }
    for path, cols in defaults.items():
        if not path.exists():
            pd.DataFrame(columns=cols).to_csv(path, index=False)


def log_unmatched(year: int, round_number: int, names: list[str]) -> None:
    if not names:
        return
    append_csv(
        QUALITY_DIR / "unmatched.csv",
        [
            {"Year": year, "Round": round_number, "Name": name, "Status": "unmatched"}
            for name in names
        ],
    )


def log_skip(year: int, round_number: int, reason: str) -> None:
    append_csv(
        QUALITY_DIR / "skipped_races.csv",
        [{"Year": year, "Round": round_number, "Reason": reason}],
    )


def log_condition(year: int, round_number: int, meta: dict) -> None:
    append_csv(
        QUALITY_DIR / "race_conditions.csv",
        [
            {
                "Year": year,
                "Round": round_number,
                "Condition": meta.get("Condition"),
                "CompoundsPresent": meta.get("CompoundsPresent"),
                "DryLaps": meta.get("DryLaps"),
                "TotalLaps": meta.get("TotalLaps"),
                "EventName": meta.get("EventName"),
            }
        ],
    )


def retag_conditions_from_checkpoints(years: list[int] | None = None) -> pd.DataFrame:
    """
    Recompute dry/mixed/wet from existing parquet checkpoints (no FastF1 calls).
    Rewrites race_conditions.csv and updates each race meta.json Condition fields.
    """
    import json

    from f1_analytics.checkpoints import checkpoint_paths, load_checkpoint
    from f1_analytics.condition import condition_meta
    from f1_analytics.config import PROCESSED_DIR, SEASON_YEARS

    ensure_audit_files()
    years = years or SEASON_YEARS
    rows: list[dict] = []
    for year in years:
        root = PROCESSED_DIR / str(year)
        if not root.exists():
            continue
        for meta_path in sorted(root.glob("r*_meta.json")):
            rnd = int(meta_path.stem.split("_")[0][1:])
            cached = load_checkpoint(year, rnd)
            cond = condition_meta(cached["tables"]["laps"], cached["tables"]["race"])
            meta = {**cached["meta"], **cond}
            paths = checkpoint_paths(year, rnd)
            paths["meta"].write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
            rows.append(
                {
                    "Year": year,
                    "Round": rnd,
                    "Condition": meta.get("Condition"),
                    "CompoundsPresent": meta.get("CompoundsPresent"),
                    "DryLaps": meta.get("DryLaps"),
                    "TotalLaps": meta.get("TotalLaps"),
                    "EventName": meta.get("EventName"),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(QUALITY_DIR / "race_conditions.csv", index=False)
    return frame
