"""Resumable parquet checkpoints per race."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from f1_analytics.config import processed_race_dir


TABLE_NAMES = ("race", "stints", "laps", "merged")


def checkpoint_paths(year: int, round_number: int) -> dict[str, Path]:
    root = processed_race_dir(year)
    paths = {name: root / f"r{round_number:02d}_{name}.parquet" for name in TABLE_NAMES}
    paths["meta"] = root / f"r{round_number:02d}_meta.json"
    return paths


def checkpoint_exists(year: int, round_number: int) -> bool:
    paths = checkpoint_paths(year, round_number)
    return all(paths[name].exists() for name in TABLE_NAMES) and paths["meta"].exists()


def save_checkpoint(
    year: int,
    round_number: int,
    tables: dict[str, pd.DataFrame],
    meta: dict[str, Any],
) -> dict[str, Path]:
    paths = checkpoint_paths(year, round_number)
    for name in TABLE_NAMES:
        frame = tables[name]
        frame.to_parquet(paths[name], index=False)
    paths["meta"].write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return paths


def load_checkpoint(year: int, round_number: int) -> dict[str, Any]:
    paths = checkpoint_paths(year, round_number)
    tables = {name: pd.read_parquet(paths[name]) for name in TABLE_NAMES}
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    return {"tables": tables, "meta": meta, "paths": paths}
