"""Project paths and default run parameters."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "fastf1_cache"
KAGGLE_DIR = ROOT / "Kaggel Data"
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
QUALITY_DIR = DATA_DIR / "quality"
OUTPUT_DIR = ROOT / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"

# First-run defaults (Bahrain GP 2024)
DEFAULT_YEAR = 2024
DEFAULT_ROUND = 1
DEFAULT_SESSION = "R"
SEASON_YEARS = list(range(2020, 2026))

# Fuel correction: approximate lap-time gain per lap as fuel burns (seconds).
FUEL_CORRECTION_S_PER_LAP = 0.035

# Kept in sync with viz_style.COMPOUND_COLORS (canonical source is viz_style).
COMPOUND_COLORS = {
    "SOFT": "#DA291C",
    "MEDIUM": "#FFD12E",
    "HARD": "#F0F0EC",
    "INTERMEDIATE": "#43B02A",
    "WET": "#0067AD",
    "UNKNOWN": "#888888",
}

DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}


def ensure_dirs() -> None:
    """Create cache, data, and output directories if missing."""
    for path in (CACHE_DIR, TABLES_DIR, FIGURES_DIR, PROCESSED_DIR, QUALITY_DIR):
        path.mkdir(parents=True, exist_ok=True)


def race_figure_dir(year: int, round_number: int) -> Path:
    path = FIGURES_DIR / str(year) / f"r{round_number:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def season_figure_dir(year: int) -> Path:
    path = FIGURES_DIR / str(year) / "season"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cross_season_figure_dir() -> Path:
    path = FIGURES_DIR / "cross_season"
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_race_dir(year: int) -> Path:
    path = PROCESSED_DIR / str(year)
    path.mkdir(parents=True, exist_ok=True)
    return path
