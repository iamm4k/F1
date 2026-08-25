"""Verify the client environment can import deps and resolve project paths."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print(f"Python {sys.version}")
    print(f"Project root: {ROOT}")

    missing = []
    for name in (
        "fastf1",
        "pandas",
        "numpy",
        "matplotlib",
        "sklearn",
        "xgboost",
        "scipy",
        "seaborn",
        "pyarrow",
    ):
        try:
            __import__(name)
            print(f"  OK  {name}")
        except Exception as exc:  # noqa: BLE001
            missing.append(name)
            print(f"  FAIL {name}: {exc}")

    sys.path.insert(0, str(ROOT))
    try:
        from f1_analytics.config import KAGGLE_DIR, PROCESSED_DIR, ensure_dirs

        ensure_dirs()
        kaggle_ok = KAGGLE_DIR.is_dir() and any(KAGGLE_DIR.glob("*.csv"))
        processed_ok = PROCESSED_DIR.is_dir() and any(PROCESSED_DIR.glob("*/r*_meta.json"))
        print(f"  Kaggle dir: {'OK' if kaggle_ok else 'MISSING'} ({KAGGLE_DIR})")
        print(
            f"  Checkpoints: {'OK' if processed_ok else 'empty (will download on first run)'} "
            f"({PROCESSED_DIR})"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL f1_analytics import: {exc}")
        return 1

    if missing:
        print("\nInstall failed packages with: pip install -r requirements.txt")
        return 1

    print("\nEnvironment OK. Try: python main.py --year 2024 --round 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
