"""Single-race processing with resumable parquet checkpoints."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from f1_analytics.analyses.descriptive import run_descriptive
from f1_analytics.analyses.diagnostic import run_diagnostic
from f1_analytics.analyses.predictive import run_predictive
from f1_analytics.analyses.prescriptive import run_prescriptive
from f1_analytics.checkpoints import checkpoint_exists, load_checkpoint, save_checkpoint
from f1_analytics.condition import condition_meta
from f1_analytics.config import (
    TABLES_DIR,
    ensure_dirs,
    race_figure_dir,
)
from f1_analytics.manifest import append_manifest, manifest_rows_for_dir
from f1_analytics.quality import log_condition, log_skip, log_unmatched
from f1_analytics.schedule import event_label
from f1_analytics.session_loader import enable_cache, load_session
from f1_analytics.tables import build_all_tables, load_kaggle
from f1_analytics.viz_style import clear_event_subtitle, figure_output_dir, set_event_subtitle

logger = logging.getLogger(__name__)


class table_output_dir:
    """Temporarily redirect analysis CSV exports."""

    def __init__(self, path) -> None:
        from f1_analytics import config

        self.path = path
        self._config = config
        self._prev = None

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self._prev = self._config.TABLES_DIR
        self._config.TABLES_DIR = self.path
        return self.path

    def __exit__(self, *args: object) -> None:
        self._config.TABLES_DIR = self._prev


def process_race(
    year: int,
    round_number: int,
    *,
    force: bool = False,
    skip_telemetry: bool = True,
    n_sims: int = 800,
    make_figures: bool = True,
    kaggle: dict | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Load / checkpoint / analyse one race.

    Returns tables + meta. Skips FastF1 reload when parquet checkpoint exists
    unless force=True.
    """
    ensure_dirs()
    enable_cache(debug=debug)
    tag = f"{year}_r{round_number:02d}"

    session = None
    if checkpoint_exists(year, round_number) and not force:
        cached = load_checkpoint(year, round_number)
        tables = cached["tables"]
        meta = cached["meta"]
        logger.info(
            "Checkpoint hit %s R%s (%s) — skipping FastF1 reload",
            year,
            round_number,
            meta.get("Condition"),
        )
        # Figures that need session timing (pit loss) still require a cached reload.
        if make_figures:
            try:
                session = load_session(year, round_number, "R", telemetry=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Checkpoint figures without session %s R%s: %s",
                    year,
                    round_number,
                    exc,
                )
    else:
        try:
            session = load_session(year, round_number, "R", telemetry=False)
        except Exception as exc:  # noqa: BLE001
            reason = f"session_load_failed: {exc}"
            logger.error("%s R%s %s", year, round_number, reason)
            log_skip(year, round_number, reason)
            return {"status": "skipped", "reason": reason}

        season_kaggle = kaggle if kaggle is not None else load_kaggle(year=year)
        built = build_all_tables(session, year, round_number, kaggle=season_kaggle)
        tables = {
            "race": built["race"],
            "stints": built["stints"],
            "laps": built["laps"],
            "merged": built["merged"],
        }
        unmatched = list(built["unmatched_names"])
        cond = condition_meta(tables["laps"], tables["race"])
        meta = {
            "Year": year,
            "Round": round_number,
            "EventName": str(session.event.get("EventName", "")),
            "Tag": tag,
            "UnmatchedNames": unmatched,
            **cond,
        }
        if tables["laps"].empty or len(tables["race"]) == 0:
            reason = "insufficient_data: empty laps or race results"
            log_skip(year, round_number, reason)
            return {"status": "skipped", "reason": reason, "meta": meta}

        save_checkpoint(year, round_number, tables, meta)
        log_unmatched(year, round_number, unmatched)
        log_condition(year, round_number, meta)
        logger.info(
            "Processed %s R%s %s — laps=%s stints=%s condition=%s unmatched=%s",
            year,
            round_number,
            meta["EventName"],
            len(tables["laps"]),
            len(tables["stints"]),
            meta["Condition"],
            unmatched or "none",
        )

    if make_figures:
        fig_dir = race_figure_dir(year, round_number)
        table_dir = fig_dir / "tables"
        _patch_tables_dir(table_dir)
        label = event_label(year, round_number, session)
        if not label or "Round" in label:
            name = meta.get("EventName") or f"Round {round_number}"
            label = f"{year} {name} · Race"
        set_event_subtitle(label)
        try:
            with figure_output_dir(fig_dir):
                _run_analyses(
                    tables,
                    session=session,
                    tag=tag,
                    skip_telemetry=skip_telemetry,
                    n_sims=n_sims,
                    year=year,
                    round_number=round_number,
                )
        finally:
            clear_event_subtitle()
            _patch_tables_dir(TABLES_DIR)

        append_manifest(
            manifest_rows_for_dir(
                fig_dir,
                year=year,
                round_number=round_number,
                analysis="per_race",
                condition=meta.get("Condition"),
            )
        )

    return {
        "status": "ok",
        "tables": tables,
        "meta": meta,
        "event_label": event_label(year, round_number),
    }


def _patch_tables_dir(path) -> None:
    """Point analysis modules at the race-specific tables directory."""
    import f1_analytics.analyses.descriptive as descriptive
    import f1_analytics.analyses.diagnostic as diagnostic
    import f1_analytics.analyses.predictive as predictive
    import f1_analytics.analyses.prescriptive as prescriptive
    import f1_analytics.config as config

    config.TABLES_DIR = path
    path.mkdir(parents=True, exist_ok=True)
    for mod in (descriptive, diagnostic, predictive, prescriptive):
        if hasattr(mod, "TABLES_DIR"):
            mod.TABLES_DIR = path


def _run_analyses(
    tables: dict[str, pd.DataFrame],
    *,
    session: Any | None,
    tag: str,
    skip_telemetry: bool,
    n_sims: int,
    year: int,
    round_number: int,
) -> None:
    laps = tables["laps"]
    stints = tables["stints"]
    merged = tables["merged"]

    # Telemetry only for selected showcase races (lazy) — default skip in season loop
    run_descriptive(
        laps,
        stints,
        merged,
        session=session,
        tag=tag,
        skip_telemetry=skip_telemetry,
    )
    run_diagnostic(laps, stints, merged, session=session, tag=tag)
    run_predictive(merged, stints, laps, tag=tag)
    run_prescriptive(laps, stints, merged, session=session, tag=tag, n_sims=n_sims)
