"""Season loop and season-level aggregate figures."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1_analytics.analyses.diagnostic import track_status_windows
from f1_analytics.analyses.predictive import run_predictive
from f1_analytics.checkpoints import load_checkpoint
from f1_analytics.config import FIGURES_DIR, PROCESSED_DIR, TABLES_DIR, ensure_dirs, season_figure_dir
from f1_analytics.degradation import fit_degradation
from f1_analytics.manifest import append_manifest, manifest_rows_for_dir
from f1_analytics.process_race import process_race
from f1_analytics.schedule import race_rounds
from f1_analytics.tables import load_kaggle
from f1_analytics.viz_style import (
    annotate_figure,
    apply_theme,
    compound_color,
    figure_output_dir,
    present_compounds,
    save_fig,
    set_event_subtitle,
    clear_event_subtitle,
)

logger = logging.getLogger(__name__)


def process_season(
    year: int,
    *,
    force: bool = False,
    skip_telemetry: bool = True,
    n_sims: int = 600,
    make_figures: bool = True,
    max_rounds: int | None = None,
    debug: bool = False,
) -> dict:
    """Process all real rounds for a season (resumable)."""
    ensure_dirs()
    rounds = race_rounds(year)
    if max_rounds is not None:
        rounds = rounds[:max_rounds]
    kaggle = load_kaggle(year=year)
    results = []
    for rnd in rounds:
        fig_dir = season_figure_dir(year).parent / f"r{rnd:02d}"
        skip_figs = (
            make_figures
            and (fig_dir / f"{year}_r{rnd:02d}_a1_pace_vs_lap.png").exists()
            and not force
        )
        try:
            result = process_race(
                year,
                rnd,
                force=force,
                skip_telemetry=skip_telemetry,
                n_sims=n_sims,
                make_figures=make_figures and not skip_figs,
                kaggle=kaggle,
                debug=debug,
            )
        except Exception as exc:  # noqa: BLE001
            from f1_analytics.quality import log_skip

            reason = f"unhandled: {exc}"
            logger.error("Season %s round %s failed: %s", year, rnd, exc)
            log_skip(year, rnd, reason)
            result = {"status": "skipped", "reason": reason}
        results.append(result)
        logger.info("Season %s progress: round %s -> %s", year, rnd, result.get("status"))

    if make_figures:
        build_season_aggregates(year)

    ok = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    return {"year": year, "ok": ok, "skipped": skipped, "results": results}


def _load_season_frames(year: int) -> dict[str, pd.DataFrame]:
    root = PROCESSED_DIR / str(year)
    if not root.exists():
        return {}
    races, stints, laps, merged, metas = [], [], [], [], []
    for meta_path in sorted(root.glob("r*_meta.json")):
        rnd = int(meta_path.stem.split("_")[0][1:])
        cached = load_checkpoint(year, rnd)
        for frame, bucket in (
            (cached["tables"]["race"], races),
            (cached["tables"]["stints"], stints),
            (cached["tables"]["laps"], laps),
            (cached["tables"]["merged"], merged),
        ):
            f = frame.copy()
            f["Year"] = year
            f["Round"] = rnd
            bucket.append(f)
        meta = cached["meta"]
        metas.append(meta)
    return {
        "race": pd.concat(races, ignore_index=True) if races else pd.DataFrame(),
        "stints": pd.concat(stints, ignore_index=True) if stints else pd.DataFrame(),
        "laps": pd.concat(laps, ignore_index=True) if laps else pd.DataFrame(),
        "merged": pd.concat(merged, ignore_index=True) if merged else pd.DataFrame(),
        "meta": pd.DataFrame(metas),
    }


def _event_label(name: str) -> str:
    text = str(name or "").replace(" Grand Prix", "").strip()
    return text or "Unknown"


def _dry_eligible_rounds(meta: pd.DataFrame) -> set[int]:
    """Rounds that may contribute dry-compound slopes (exclude full wet)."""
    if meta.empty or "Condition" not in meta.columns:
        return set(meta["Round"].astype(int)) if not meta.empty else set()
    return set(meta.loc[meta["Condition"] != "wet", "Round"].astype(int))


def build_season_aggregates(year: int) -> None:
    """Season roll-up figures for one year."""
    apply_theme()
    data = _load_season_frames(year)
    if not data or data["laps"].empty:
        logger.warning("No processed data for season aggregates %s", year)
        return

    out = season_figure_dir(year)
    meta = data["meta"].copy()
    laps = data["laps"]
    merged = data["merged"]
    if "EventName" not in meta.columns:
        meta["EventName"] = meta.get("EventName", "Unknown")
    event_map = {
        int(r["Round"]): _event_label(r.get("EventName", f"R{int(r['Round'])}"))
        for _, r in meta.iterrows()
    }

    with figure_output_dir(out):
        dry_rounds = _dry_eligible_rounds(meta)
        slope_rows: list[pd.DataFrame] = []
        for rnd in sorted(dry_rounds):
            sub = laps[laps["Round"] == rnd]
            deg = fit_degradation(sub)
            if deg.empty:
                continue
            deg = deg.copy()
            deg["Round"] = rnd
            deg["Event"] = event_map.get(int(rnd), f"R{rnd}")
            slope_rows.append(deg)

        slopes = pd.concat(slope_rows, ignore_index=True) if slope_rows else pd.DataFrame()

        # 1) Degradation slope distribution (dry compounds; wet races excluded)
        if not slopes.empty:
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for compound in present_compounds(slopes["Compound"]):
                vals = slopes.loc[slopes["Compound"] == compound, "Slope"].dropna()
                if vals.empty:
                    continue
                ax.hist(
                    vals,
                    bins=12,
                    alpha=0.45,
                    color=compound_color(compound),
                    edgecolor="#333333",
                    label=f"{compound} (n={len(vals)})",
                )
            ax.set_xlabel("Degradation slope (s/lap)")
            ax.set_ylabel("Race count")
            ax.legend()
            annotate_figure(
                fig,
                ax,
                f"{year} dry-compound degradation slopes",
                f"{year} season · wet races excluded · dry stints only",
            )
            fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.86))
            save_fig(fig, f"{year}_season_degradation_slopes.png")

            # Diagnostic heatmap: circuit × compound
            heat = (
                slopes.groupby(["Event", "Compound"])["Slope"]
                .median()
                .unstack(fill_value=np.nan)
            )
            compounds = [c for c in ("SOFT", "MEDIUM", "HARD") if c in heat.columns]
            if compounds and not heat.empty:
                heat = heat[compounds]
                fig, ax = plt.subplots(figsize=(10, max(5.5, 0.35 * len(heat) + 2)))
                im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="coolwarm")
                ax.set_xticks(range(len(compounds)))
                ax.set_xticklabels(compounds)
                ax.set_yticks(range(len(heat.index)))
                ax.set_yticklabels(heat.index.tolist(), fontsize=8)
                fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Median slope (s/lap)")
                annotate_figure(
                    fig,
                    ax,
                    f"{year} degradation by circuit × compound",
                    "median dry-stint slope · wet races excluded",
                )
                fig.tight_layout(rect=(0.02, 0.04, 0.92, 0.86))
                save_fig(fig, f"{year}_season_degradation_heatmap.png")

        # 2) Median pit-loss by circuit (from per-race session-based tables)
        pit_rows: list[dict] = []
        for rnd in sorted(meta["Round"].astype(int).unique()):
            pit_path = (
                FIGURES_DIR
                / str(year)
                / f"r{rnd:02d}"
                / "tables"
                / f"{year}_r{rnd:02d}_a1_pit_loss.csv"
            )
            if not pit_path.exists() or pit_path.stat().st_size < 10:
                continue
            try:
                pits = pd.read_csv(pit_path)
            except (pd.errors.EmptyDataError, ValueError, OSError):
                continue
            if pits.empty or "PitLossSeconds" not in pits.columns:
                continue
            pit_rows.append(
                {
                    "Round": int(rnd),
                    "Event": event_map.get(int(rnd), f"R{rnd}"),
                    "MedianPitLoss": float(pits["PitLossSeconds"].median()),
                    "N": int(len(pits)),
                }
            )
        if pit_rows:
            pit_df = pd.DataFrame(pit_rows).sort_values("MedianPitLoss")
            fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(pit_df) + 1.5)))
            ax.barh(
                pit_df["Event"],
                pit_df["MedianPitLoss"],
                color="#1f4e79",
                edgecolor="#333333",
            )
            ax.set_xlabel("Median pit loss (s)")
            annotate_figure(fig, ax, f"{year} median pit loss by circuit", f"{year} season")
            fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.86))
            save_fig(fig, f"{year}_season_pit_loss_by_circuit.png")
            pit_df.to_csv(TABLES_DIR / f"{year}_season_a1_pit_loss_by_circuit.csv", index=False)

        # 3) Qualifying gap trends (if available)
        if "GapToPole" in merged.columns and merged["GapToPole"].notna().any():
            from f1_analytics.analyses.descriptive import _parse_gap_seconds

            q = merged.dropna(subset=["GapToPole"]).copy()
            q["GapSeconds"] = q["GapToPole"].map(_parse_gap_seconds)
            trend = q.groupby("Round")["GapSeconds"].median().reset_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(trend["Round"], trend["GapSeconds"], marker="o", color="#1f4e79")
            ax.set_xlabel("Round")
            ax.set_ylabel("Median gap to pole (s)")
            annotate_figure(fig, ax, f"{year} qualifying gap-to-pole trend", f"{year} season")
            fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.86))
            save_fig(fig, f"{year}_season_quali_gap_trend.png")

        # 4) Condition mix
        if not meta.empty and "Condition" in meta.columns:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            order = [c for c in ("dry", "mixed", "wet") if c in set(meta["Condition"])]
            counts = meta["Condition"].value_counts().reindex(order)
            colors = {"dry": "#C4A000", "mixed": "#43B02A", "wet": "#0067AD"}
            ax.bar(
                counts.index.astype(str),
                counts.values,
                color=[colors.get(c, "#1f4e79") for c in counts.index],
                edgecolor="#333",
            )
            ax.set_ylabel("Races")
            annotate_figure(fig, ax, f"{year} race condition mix", f"{year} season")
            fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.86))
            save_fig(fig, f"{year}_season_condition_mix.png")

        # 5) SC/VSC frequency by circuit
        sc_rows: list[dict] = []
        for rnd, group in laps.groupby("Round"):
            windows = track_status_windows(group)
            sc_rows.append(
                {
                    "Event": event_map.get(int(rnd), f"R{rnd}"),
                    "SC": int((windows["Label"] == "SC").sum()) if not windows.empty else 0,
                    "VSC": int((windows["Label"] == "VSC").sum()) if not windows.empty else 0,
                }
            )
        if sc_rows:
            sc_df = pd.DataFrame(sc_rows)
            sc_df["Total"] = sc_df["SC"] + sc_df["VSC"]
            sc_df = sc_df.sort_values("Total", ascending=False)
            fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(sc_df) + 1.5)))
            y = np.arange(len(sc_df))
            ax.barh(y, sc_df["SC"], color="#DA291C", edgecolor="#333", label="SC laps")
            ax.barh(
                y,
                sc_df["VSC"],
                left=sc_df["SC"],
                color="#FFD12E",
                edgecolor="#333",
                label="VSC laps",
            )
            ax.set_yticks(y)
            ax.set_yticklabels(sc_df["Event"], fontsize=8)
            ax.set_xlabel("Laps under SC / VSC")
            ax.legend()
            annotate_figure(fig, ax, f"{year} SC/VSC exposure by circuit", f"{year} season")
            fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.86))
            save_fig(fig, f"{year}_season_sc_vsc_by_circuit.png")

        # 6) Prescriptive: recommended strategy per circuit (from per-race sims)
        rec_rows: list[dict] = []
        for rnd in sorted(meta["Round"].astype(int).unique()):
            rec_path = (
                FIGURES_DIR
                / str(year)
                / f"r{rnd:02d}"
                / "tables"
                / f"{year}_r{rnd:02d}_a4_recommendation.csv"
            )
            if not rec_path.exists():
                alt = TABLES_DIR / f"{year}_r{rnd:02d}_a4_recommendation.csv"
                rec_path = alt if alt.exists() else rec_path
            if not rec_path.exists():
                continue
            rec = pd.read_csv(rec_path)
            if rec.empty:
                continue
            row = rec.iloc[0]
            rec_rows.append(
                {
                    "Round": rnd,
                    "Event": event_map.get(rnd, f"R{rnd}"),
                    "Strategy": str(row.get("RecommendedStrategy", "")),
                    "Compounds": str(row.get("Compounds", "")),
                }
            )
        if rec_rows:
            rec_df = pd.DataFrame(rec_rows)
            counts = rec_df["Strategy"].value_counts()
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.barh(counts.index.astype(str), counts.values, color="#1f4e79", edgecolor="#333")
            ax.set_xlabel("Races where strategy wins MC")
            annotate_figure(
                fig,
                ax,
                f"{year} optimal strategy summary",
                "most frequent MC-winning compound sequences",
            )
            fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.86))
            save_fig(fig, f"{year}_season_strategy_summary.png")
            rec_df.to_csv(TABLES_DIR / f"{year}_season_a4_strategy_by_circuit.csv", index=False)

        # 7) Season-scale predictive (leave-one-race-out)
        from f1_analytics.analyses.predictive import _engineer_features

        feats = []
        for rnd, group in merged.groupby("Round"):
            st = data["stints"][data["stints"]["Round"] == rnd]
            lp = data["laps"][data["laps"]["Round"] == rnd]
            if group.empty or st.empty or lp.empty:
                continue
            feat = _engineer_features(group, st, lp)
            feat["Round"] = int(rnd)
            feats.append(feat)
        if feats:
            all_feat = pd.concat(feats, ignore_index=True)
            set_event_subtitle(f"{year} season")
            try:
                run_predictive(
                    all_feat,
                    pd.DataFrame(),
                    pd.DataFrame(),
                    tag=f"{year}_season",
                    multi_race=all_feat,
                )
            finally:
                clear_event_subtitle()

    append_manifest(
        manifest_rows_for_dir(
            out,
            year=year,
            round_number=None,
            analysis="season",
            condition="aggregate",
        )
    )
    logger.info("Season aggregates written for %s -> %s", year, out)
