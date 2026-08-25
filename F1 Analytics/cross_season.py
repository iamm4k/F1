"""Cross-season roll-up figures (2020–2025)."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd

from f1_analytics.config import (
    FIGURES_DIR,
    QUALITY_DIR,
    TABLES_DIR,
    cross_season_figure_dir,
    ensure_dirs,
)
from f1_analytics.degradation import fit_degradation
from f1_analytics.manifest import append_manifest, manifest_rows_for_dir
from f1_analytics.season_runner import _dry_eligible_rounds, _load_season_frames
from f1_analytics.viz_style import (
    annotate_figure,
    apply_theme,
    compound_color,
    figure_output_dir,
    present_compounds,
    save_fig,
)

logger = logging.getLogger(__name__)


def build_cross_season(years: list[int]) -> None:
    ensure_dirs()
    apply_theme()
    out = cross_season_figure_dir()

    condition_frames = []
    slope_frames = []
    pit_frames = []
    winner_frames = []

    for year in years:
        data = _load_season_frames(year)
        if not data or data["meta"].empty:
            continue
        meta = data["meta"].copy()
        meta["Year"] = year
        condition_frames.append(meta)

        dry_rounds = _dry_eligible_rounds(meta)
        for rnd in dry_rounds:
            deg = fit_degradation(data["laps"][data["laps"]["Round"] == rnd])
            if deg.empty:
                continue
            deg = deg.copy()
            deg["Year"] = year
            deg["Round"] = rnd
            slope_frames.append(deg)

        for rnd in meta["Round"].astype(int).unique():
            pit_path = (
                FIGURES_DIR
                / str(year)
                / f"r{int(rnd):02d}"
                / "tables"
                / f"{year}_r{int(rnd):02d}_a1_pit_loss.csv"
            )
            if not pit_path.exists() or pit_path.stat().st_size < 10:
                continue
            try:
                pits = pd.read_csv(pit_path)
            except (pd.errors.EmptyDataError, ValueError, OSError):
                continue
            if pits.empty or "PitLossSeconds" not in pits.columns:
                continue
            pit_frames.append(
                {
                    "Year": year,
                    "Round": int(rnd),
                    "MedianPitLoss": float(pits["PitLossSeconds"].median()),
                }
            )

        merged = data["merged"]
        if not merged.empty and "Position" in merged.columns:
            winners = (
                merged.dropna(subset=["Position"])
                .sort_values(["Round", "Position"])
                .groupby("Round", as_index=False)
                .first()
            )
            keep = [
                c
                for c in ("Round", "DriverCode", "TeamName", "Team", "Position")
                if c in winners.columns
            ]
            w = winners[keep].copy()
            w["Year"] = year
            winner_frames.append(w)

    if not condition_frames:
        logger.warning("No seasons available for cross-season roll-up")
        return

    meta_all = pd.concat(condition_frames, ignore_index=True)

    with figure_output_dir(out):
        # Condition mix by year
        fig, ax = plt.subplots(figsize=(10, 5.5))
        pivot = (
            meta_all.groupby(["Year", "Condition"]).size().unstack(fill_value=0)
            if "Condition" in meta_all.columns
            else pd.DataFrame()
        )
        if not pivot.empty:
            for col in ("dry", "mixed", "wet"):
                if col not in pivot.columns:
                    pivot[col] = 0
            pivot = pivot[["dry", "mixed", "wet"]]
            pivot.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                edgecolor="#333333",
                color=["#C4A000", "#43B02A", "#0067AD"],
            )
            ax.set_ylabel("Races")
            ax.set_xlabel("Year")
            ax.legend(title="Condition")
            annotate_figure(fig, ax, "Race conditions across seasons", "2020–2025")
            fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.86))
            save_fig(fig, "cross_season_condition_mix.png")

        # Races processed per year
        fig, ax = plt.subplots(figsize=(9, 5))
        counts = meta_all.groupby("Year").size()
        ax.bar(counts.index.astype(str), counts.values, color="#1f4e79", edgecolor="#333")
        ax.set_ylabel("Processed races")
        annotate_figure(fig, ax, "Processed races by season", "checkpoint coverage")
        fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.86))
        save_fig(fig, "cross_season_race_counts.png")

        # Degradation evolution by compound (median slope per year)
        if slope_frames:
            slopes = pd.concat(slope_frames, ignore_index=True)
            fig, ax = plt.subplots(figsize=(10, 5.5))
            for compound in present_compounds(slopes["Compound"]):
                yearly = (
                    slopes.loc[slopes["Compound"] == compound]
                    .groupby("Year")["Slope"]
                    .median()
                    .sort_index()
                )
                if yearly.empty:
                    continue
                ax.plot(
                    yearly.index.astype(int),
                    yearly.values,
                    marker="o",
                    linewidth=2.2,
                    color=compound_color(compound),
                    label=compound,
                )
            ax.set_xlabel("Year")
            ax.set_ylabel("Median degradation slope (s/lap)")
            ax.legend()
            annotate_figure(
                fig,
                ax,
                "Dry-compound degradation across seasons",
                "median slope · wet races excluded",
            )
            fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.86))
            save_fig(fig, "cross_season_degradation_trends.png")
            slopes.to_csv(TABLES_DIR / "cross_season_degradation_slopes.csv", index=False)

        # Pit-loss trend
        if pit_frames:
            pits = pd.DataFrame(pit_frames)
            yearly = pits.groupby("Year")["MedianPitLoss"].median().sort_index()
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(
                yearly.index.astype(int),
                yearly.values,
                marker="o",
                color="#1f4e79",
                linewidth=2.2,
            )
            ax.set_xlabel("Year")
            ax.set_ylabel("Median of race-median pit loss (s)")
            annotate_figure(fig, ax, "Pit-loss trend across seasons", "2020–2025")
            fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.86))
            save_fig(fig, "cross_season_pit_loss_trend.png")
            pits.to_csv(TABLES_DIR / "cross_season_pit_loss.csv", index=False)

        # Competitive order: wins by driver code
        if winner_frames:
            winners = pd.concat(winner_frames, ignore_index=True)
            if "DriverCode" in winners.columns:
                fig, ax = plt.subplots(figsize=(11, 6))
                pivot = winners.groupby(["Year", "DriverCode"]).size().unstack(fill_value=0)
                top = pivot.sum().sort_values(ascending=False).head(8).index.tolist()
                pivot[top].plot(kind="bar", stacked=True, ax=ax, edgecolor="#333333")
                ax.set_ylabel("Race wins (P1 finishes)")
                ax.set_xlabel("Year")
                ax.legend(title="Driver", ncol=2, fontsize=8)
                annotate_figure(fig, ax, "Competitive order shifts", "race winners 2020–2025")
                fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.86))
                save_fig(fig, "cross_season_winners.png")

    unmatched = QUALITY_DIR / "unmatched.csv"
    if unmatched.exists():
        logger.info("Unmatched-name audit available at %s", unmatched)

    append_manifest(
        manifest_rows_for_dir(
            out,
            year=None,
            round_number=None,
            analysis="cross_season",
            condition="aggregate",
        )
    )
    logger.info("Cross-season figures -> %s", out)
