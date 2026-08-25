"""Manifest of generated figures for later report assembly."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from f1_analytics.config import MANIFEST_PATH, ensure_dirs


def append_manifest(rows: list[dict]) -> None:
    if not rows:
        return
    ensure_dirs()
    frame = pd.DataFrame(rows)
    if MANIFEST_PATH.exists():
        existing = pd.read_csv(MANIFEST_PATH)
        # Avoid FutureWarning on empty/all-NA concat columns
        existing = existing.dropna(axis=1, how="all")
        frame = frame.dropna(axis=1, how="all")
        if existing.empty:
            combined = frame
        elif frame.empty:
            combined = existing
        else:
            combined = pd.concat([existing, frame], ignore_index=True)
        frame = combined.drop_duplicates(subset=["path"], keep="last")
    frame.to_csv(MANIFEST_PATH, index=False)


def manifest_rows_for_dir(
    directory: Path,
    *,
    year: int | None,
    round_number: int | None,
    analysis: str,
    condition: str | None = None,
) -> list[dict]:
    rows: list[dict] = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.png")):
        rows.append(
            {
                "year": year,
                "round": round_number,
                "analysis": analysis,
                "figure_name": path.stem,
                "path": str(path.as_posix()),
                "svg_path": str(path.with_suffix(".svg").as_posix())
                if path.with_suffix(".svg").exists()
                else "",
                "condition": condition,
            }
        )
    return rows
