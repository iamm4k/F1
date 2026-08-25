"""CLI entrypoint for F1 Performance & Strategy Analytics."""

from __future__ import annotations

import argparse
import logging
import sys

from f1_analytics.config import DEFAULT_ROUND, DEFAULT_YEAR, SEASON_YEARS, ensure_dirs
from f1_analytics.cross_season import build_cross_season
from f1_analytics.process_race import process_race
from f1_analytics.quality import ensure_audit_files
from f1_analytics.season_runner import process_season

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("f1_analytics")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1 Performance & Strategy Analytics")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--round", type=int, default=DEFAULT_ROUND)
    parser.add_argument(
        "--season",
        action="store_true",
        help="Process all rounds for --year (resumable checkpoints).",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="",
        help="Comma range or list, e.g. 2020-2025 or 2023,2024. Implies season mode.",
    )
    parser.add_argument(
        "--cross-season",
        action="store_true",
        help="Build cross-season roll-ups from existing checkpoints.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild checkpoints/figures")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--skip-telemetry",
        action="store_true",
        default=True,
        help="Skip telemetry overlay (default on for season loops).",
    )
    parser.add_argument(
        "--with-telemetry",
        action="store_true",
        help="Enable telemetry overlay (isolated load; slow).",
    )
    parser.add_argument("--n-sims", type=int, default=800)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Limit rounds per season (batching / smoke tests).",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Only build parquet checkpoints (no figure export).",
    )
    return parser.parse_args(argv)


def _parse_years(spec: str) -> list[int]:
    """Parse '2020-2025' or '2020,2022,2024' or '2020-2023,2025'."""
    spec = spec.strip()
    if not spec:
        return []
    years: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_dirs()
    ensure_audit_files()
    try:
        import fastf1  # noqa: F401

        print("fastf1 import OK; cache dir will be ./fastf1_cache")
    except ImportError as exc:
        print(f"fastf1 not installed: {exc}", file=sys.stderr)
        return 1

    skip_telemetry = not args.with_telemetry
    make_figures = not args.no_figures

    if args.cross_season:
        years = _parse_years(args.years) or SEASON_YEARS
        build_cross_season(years)
        print("Cross-season roll-ups done.")
        return 0

    years = _parse_years(args.years)
    if years or args.season:
        target_years = years or [args.year]
        # Estimate before full multi-year loop
        if len(target_years) > 1:
            print(
                "Note: ~6-10 FastF1 API calls per uncached race; ~500 calls/h limit. "
                "Checkpoints make reruns skip completed races. Prefer one season at a time "
                "or use --max-rounds for batches."
            )
        for year in target_years:
            summary = process_season(
                year,
                force=args.force,
                skip_telemetry=skip_telemetry,
                n_sims=args.n_sims,
                make_figures=make_figures,
                max_rounds=args.max_rounds,
                debug=args.debug,
            )
            print(
                f"Season {year}: ok={summary['ok']} skipped={summary['skipped']}"
            )
        if len(target_years) > 1:
            build_cross_season(target_years)
    else:
        result = process_race(
            args.year,
            args.round,
            force=args.force,
            skip_telemetry=skip_telemetry,
            n_sims=args.n_sims,
            make_figures=make_figures,
            debug=args.debug,
        )
        print(f"Race {args.year} R{args.round}: {result.get('status')}")

    print("Done. Checkpoints -> data/processed/  Figures -> output/figures/")
    print("Quality audits -> data/quality/  Manifest -> output/manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
