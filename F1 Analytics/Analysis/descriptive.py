"""Analysis 1 — Descriptive race engineering metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1_analytics.config import DEFAULT_ROUND, DEFAULT_YEAR, TABLES_DIR
from f1_analytics.degradation import demeaned_stint_laps, fit_degradation
from f1_analytics.plotting_utils import finish_figure, setup_plotting
from f1_analytics.session_loader import enable_cache, load_session
from f1_analytics.viz_style import (
    annotate_figure,
    compound_color,
    compound_edge,
    compound_legend_handles,
    compound_line_color,
    compound_line_legend_handles,
    event_subtitle,
    present_compounds,
    save_fig,
    team_color,
)

logger = logging.getLogger(__name__)

LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]


def pit_loss_table(laps: pd.DataFrame) -> pd.DataFrame:
    """Estimate pit-stop time loss from PitIn/PitOut deltas where available."""
    if laps.empty:
        return pd.DataFrame(columns=["DriverCode", "PitLossSeconds", "Method"])

    rows: list[dict[str, Any]] = []
    for driver, group in laps.groupby("DriverCode"):
        g = group.sort_values("LapNumber")
        in_laps = g[g["PitInTimeSeconds"].notna()] if "PitInTimeSeconds" in g else g.iloc[0:0]
        for _, in_row in in_laps.iterrows():
            out_candidates = g[g["LapNumber"] == in_row["LapNumber"] + 1]
            if out_candidates.empty:
                continue
            out_row = out_candidates.iloc[0]
            if pd.notna(out_row.get("PitOutTimeSeconds")) and pd.notna(
                in_row.get("PitInTimeSeconds")
            ):
                loss = float(out_row["PitOutTimeSeconds"] - in_row["PitInTimeSeconds"])
                if 10 < loss < 60:
                    rows.append(
                        {
                            "DriverCode": driver,
                            "PitLossSeconds": loss,
                            "Method": "PitInOutDelta",
                            "InLap": in_row["LapNumber"],
                        }
                    )

        if not any(r["DriverCode"] == driver for r in rows) and "Stint" in g.columns:
            for stint_id, stint in g.groupby("Stint"):
                if stint_id == g["Stint"].min():
                    continue
                first = stint.sort_values("LapNumber").iloc[0]
                median = stint["LapTimeSeconds"].median()
                if pd.notna(first["LapTimeSeconds"]) and pd.notna(median):
                    loss = float(first["LapTimeSeconds"] - median)
                    if 5 < loss < 45:
                        rows.append(
                            {
                                "DriverCode": driver,
                                "PitLossSeconds": loss,
                                "Method": "OutLapDelta",
                                "InLap": first["LapNumber"] - 1,
                            }
                        )

    return pd.DataFrame(rows)


def pit_loss_from_session(session: Any) -> pd.DataFrame:
    """Build pit-loss rows from full session laps (not quicklaps-filtered)."""
    if session is None:
        return pd.DataFrame()
    laps = session.laps.copy()
    frame = pd.DataFrame(
        {
            "DriverCode": laps["Driver"],
            "LapNumber": laps["LapNumber"],
            "Stint": laps["Stint"],
            "LapTimeSeconds": laps["LapTime"].dt.total_seconds(),
            "PitInTimeSeconds": laps["PitInTime"].dt.total_seconds()
            if "PitInTime" in laps
            else np.nan,
            "PitOutTimeSeconds": laps["PitOutTime"].dt.total_seconds()
            if "PitOutTime" in laps
            else np.nan,
        }
    )
    return pit_loss_table(frame)


def run_descriptive(
    laps: pd.DataFrame,
    stints: pd.DataFrame,
    merged: pd.DataFrame,
    session: Any | None = None,
    *,
    tag: str = "2024_r01",
    top_n_pace: int = 6,
    skip_telemetry: bool = False,
) -> dict[str, Path | pd.DataFrame]:
    """Produce descriptive tables and figures."""
    setup_plotting()
    outputs: dict[str, Path | pd.DataFrame] = {}

    pace = (
        laps.groupby(["DriverCode", "Compound"], dropna=False)
        .agg(
            MeanLapTime=("LapTimeSeconds", "mean"),
            MedianLapTime=("LapTimeSeconds", "median"),
            MeanFuelCorrected=("FuelCorrectedLapTime", "mean"),
            Laps=("LapTimeSeconds", "count"),
        )
        .reset_index()
    )
    stint_dist = (
        stints.groupby("Compound", dropna=False)
        .agg(
            MeanStintLength=("StintLength", "mean"),
            MedianStintLength=("StintLength", "median"),
            StintCount=("StintLength", "count"),
            MinLength=("StintLength", "min"),
            MaxLength=("StintLength", "max"),
        )
        .reset_index()
    )
    pits = pit_loss_from_session(session)
    if pits.empty:
        pits = pit_loss_table(laps)

    quali = merged[
        ["DriverCode", "DriverName", "QualiPosition", "QualiBestTime", "GapToPole"]
    ].copy()

    sector = (
        laps.groupby("DriverCode")
        .agg(
            Sector1=("Sector1TimeSeconds", "median"),
            Sector2=("Sector2TimeSeconds", "median"),
            Sector3=("Sector3TimeSeconds", "median"),
        )
        .reset_index()
    )
    for col in ("Sector1", "Sector2", "Sector3"):
        best = sector[col].min()
        sector[f"{col}Delta"] = sector[col] - best

    ranking = (
        laps.groupby("DriverCode")["FuelCorrectedLapTime"]
        .median()
        .reset_index()
        .rename(columns={"FuelCorrectedLapTime": "MedianFuelCorrectedPace"})
        .sort_values("MedianFuelCorrectedPace")
    )

    for name, frame in {
        "a1_pace_by_driver_compound": pace,
        "a1_stint_length_by_compound": stint_dist,
        "a1_pit_loss": pits,
        "a1_quali_gaps": quali,
        "a1_sector_deltas": sector,
        "a1_pace_ranking": ranking,
    }.items():
        path = TABLES_DIR / f"{tag}_{name}.csv"
        frame.to_csv(path, index=False)
        outputs[name] = frame
        outputs[f"{name}_path"] = path

    outputs["fig_pace_trace"] = _fig_pace_vs_lap(
        laps, merged, session, tag, top_n=top_n_pace
    )
    outputs["fig_degradation"] = _fig_tyre_degradation(laps, session, tag)
    outputs["fig_strategy"] = _fig_strategy_gantt(stints, merged, session, tag)
    outputs["fig_pit_loss"] = _fig_pit_loss(pits, session, tag)
    outputs["fig_quali_gaps"] = _fig_quali_gaps(quali, session, tag)
    outputs["fig_sector_deltas"] = _fig_sector_deltas(sector, merged, session, tag)
    outputs["fig_pace_ranking"] = _fig_pace_ranking(ranking, merged, session, tag)

    if skip_telemetry:
        outputs["fig_telemetry"] = None
    else:
        outputs["fig_telemetry"] = plot_telemetry_overlay(merged, session, tag)

    logger.info("Analysis 1 complete — %s", tag)
    return outputs


def _top_drivers(merged: pd.DataFrame, laps: pd.DataFrame, n: int) -> list[str]:
    if "Position" in merged.columns and merged["Position"].notna().any():
        return (
            merged.sort_values("Position")
            .dropna(subset=["Position"])["DriverCode"]
            .head(n)
            .tolist()
        )
    return list(laps["DriverCode"].unique()[:n])


def _fig_pace_vs_lap(
    laps: pd.DataFrame,
    merged: pd.DataFrame,
    session: Any | None,
    tag: str,
    *,
    top_n: int = 6,
) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    drivers = _top_drivers(merged, laps, top_n)
    subset = laps[laps["DriverCode"].isin(drivers)]
    compounds = present_compounds(subset["Compound"])

    for i, driver in enumerate(drivers):
        style = LINE_STYLES[i % len(LINE_STYLES)]
        d = subset[subset["DriverCode"] == driver].sort_values("LapNumber")
        for _, stint in d.groupby("Stint"):
            compound = stint["Compound"].iloc[0]
            is_hard = str(compound).upper() == "HARD"
            ax.plot(
                stint["LapNumber"],
                stint["FuelCorrectedLapTime"],
                color=compound_line_color(compound),
                linestyle=style,
                linewidth=3.0 if is_hard else 1.9,
                alpha=1.0 if is_hard else 0.95,
                solid_capstyle="round",
                zorder=3 if is_hard else 2,
            )

    # Right-margin driver legend (avoids overlapping end-of-line labels)
    driver_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            linewidth=1.8,
            label=driver,
        )
        for i, driver in enumerate(drivers)
    ]
    compound_leg = ax.legend(
        handles=compound_line_legend_handles(compounds),
        title="Compound",
        loc="upper left",
    )
    ax.add_artist(compound_leg)
    ax.legend(
        handles=driver_handles,
        title="Driver",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
    )
    ax.set_xlabel("Lap")
    ax.set_ylabel("Fuel-corrected lap time (s)")
    return finish_figure(
        fig,
        ax,
        f"{tag}_a1_pace_vs_lap.png",
        f"Fuel-corrected pace by stint (top {top_n})",
        session,
        top=0.86,
    )


def _fig_tyre_degradation(laps: pd.DataFrame, session: Any | None, tag: str) -> Path:
    deg = fit_degradation(laps)
    demeaned = demeaned_stint_laps(laps)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    compounds = present_compounds(demeaned["Compound"]) if not demeaned.empty else []

    for compound in compounds:
        group = demeaned[demeaned["Compound"] == compound]
        color = compound_color(compound)
        edge = compound_edge(compound)
        ax.scatter(
            group["TyreLife"],
            group["StintDemeanedLapTime"],
            s=16,
            alpha=0.4,
            color=color,
            edgecolors=edge,
            linewidths=0.4,
            label=str(compound),
        )

    for _, row in deg.iterrows():
        if pd.isna(row["Slope"]):
            continue
        compound = row["Compound"]
        color = compound_color(compound)
        subset = demeaned[demeaned["Compound"] == compound]
        if subset.empty:
            continue
        x_max = float(subset["TyreLife"].max())
        if not np.isfinite(x_max):
            continue
        x = np.linspace(1, x_max, 60)
        x_c = x - x.mean()
        y = row["Slope"] * x_c
        label = f"{compound} slope={row['Slope']:.3f} s/lap (mean R²={row['R2']:.2f})"
        ax.plot(x, y, color=color, linewidth=2.4, label=label)

    honesty = getattr(deg, "attrs", {}).get("honesty_note")
    ax.set_xlabel("Tyre life (laps)")
    ax.set_ylabel("Stint-demeaned fuel-corrected lap time (s)")
    ax.legend(fontsize=8, loc="best")
    title = "Tyre degradation by compound"
    path = finish_figure(fig, ax, f"{tag}_a1_tyre_degradation.png", title, session)
    if honesty:
        logger.info("Degradation honesty note: %s", honesty)
        note_path = TABLES_DIR / f"{tag}_a1_degradation_note.txt"
        note_path.write_text(honesty + "\n" + deg.to_string(index=False), encoding="utf-8")
        deg.to_csv(TABLES_DIR / f"{tag}_a1_degradation_slopes.csv", index=False)
    return path


def _fig_strategy_gantt(
    stints: pd.DataFrame,
    merged: pd.DataFrame,
    session: Any | None,
    tag: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 8))
    order = (
        merged.sort_values("Position")["DriverCode"].tolist()
        if "Position" in merged.columns
        else sorted(stints["DriverCode"].unique())
    )
    y_map = {d: i for i, d in enumerate(order)}
    for _, row in stints.iterrows():
        if row["DriverCode"] not in y_map:
            continue
        compound = row["Compound"]
        ax.barh(
            y=y_map[row["DriverCode"]],
            width=row["StintLength"],
            left=row["LapStart"] - 1,
            color=compound_color(compound),
            edgecolor=compound_edge(compound),
            linewidth=0.6,
            height=0.7,
        )
    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(list(y_map.keys()))
    ax.set_xlabel("Lap")
    ax.invert_yaxis()
    compounds = present_compounds(stints["Compound"])
    ax.legend(
        handles=compound_legend_handles(compounds),
        title="Compound",
        loc="lower right",
    )
    return finish_figure(
        fig, ax, f"{tag}_a1_strategy_gantt.png", "Stint / strategy timeline", session
    )


def _fig_pit_loss(pits: pd.DataFrame, session: Any | None, tag: str) -> Path | None:
    if pits.empty:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.text(0.5, 0.5, "No pit-loss samples available", ha="center", va="center")
        return finish_figure(
            fig, ax, f"{tag}_a1_pit_loss.png", "Pit-stop time loss", session
        )

    plot = pits.copy()
    # One bar per driver (mean loss) — per-stop labels were unreadable at field size
    grouped = (
        plot.groupby("DriverCode", as_index=False)
        .agg(
            PitLossSeconds=("PitLossSeconds", "mean"),
            NStops=("PitLossSeconds", "count"),
        )
        .sort_values("PitLossSeconds")
        .reset_index(drop=True)
    )
    grouped["StopLabel"] = [
        f"{row.DriverCode}  ×{int(row.NStops)}" for row in grouped.itertuples()
    ]

    n = len(grouped)
    fig_h = max(7.0, 0.52 * n + 2.2)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y = np.arange(n)
    colors = [team_color(d, session) or "#1f4e79" for d in grouped["DriverCode"]]
    ax.barh(
        y,
        grouped["PitLossSeconds"],
        color=colors,
        edgecolor="#333333",
        linewidth=0.4,
        height=0.68,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(grouped["StopLabel"].tolist(), fontsize=11)
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(axis="y", pad=8, length=0)
    median = float(plot["PitLossSeconds"].median())
    ax.axvline(median, color="#DA291C", linestyle="--", linewidth=1.5, label=f"Median {median:.1f}s")
    ax.set_xlabel("Mean pit loss (s)")
    ax.legend(loc="lower right")
    annotate_figure(fig, ax, "Pit-stop time loss by driver", event_subtitle(session))
    fig.subplots_adjust(left=0.18, right=0.98, top=0.88, bottom=0.08)
    return save_fig(fig, f"{tag}_a1_pit_loss.png")


def _parse_gap_seconds(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if text in {"", "0", "0.0", "nan", "NaN", "+0.000", "0.000"}:
        return 0.0
    text = text.replace("+", "")
    try:
        return float(text)
    except ValueError:
        # handle mm:ss.sss unlikely for gap
        return np.nan


def _fig_quali_gaps(quali: pd.DataFrame, session: Any | None, tag: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    q = quali.copy()
    q["GapSeconds"] = q["GapToPole"].map(_parse_gap_seconds)
    q = q.dropna(subset=["GapSeconds"]).sort_values("GapSeconds", ascending=True)
    colors = [team_color(d, session) or "#444444" for d in q["DriverCode"]]
    ax.barh(
        q["DriverCode"],
        q["GapSeconds"],
        color=colors,
        edgecolor="#333333",
        linewidth=0.4,
    )
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Gap to pole (s)")
    return finish_figure(
        fig, ax, f"{tag}_a1_quali_gaps.png", "Qualifying gaps to pole", session
    )


def _fig_sector_deltas(
    sector: pd.DataFrame,
    merged: pd.DataFrame,
    session: Any | None,
    tag: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    top = _top_drivers(merged, sector.rename(columns={"DriverCode": "DriverCode"}), 8)
    sub = sector[sector["DriverCode"].isin(top)].set_index("DriverCode").loc[
        [d for d in top if d in sector["DriverCode"].values]
    ]
    cols = ["Sector1Delta", "Sector2Delta", "Sector3Delta"]
    x = np.arange(len(sub))
    width = 0.25
    palette = ["#DA291C", "#FFD12E", "#4A4A4A"]
    for i, col in enumerate(cols):
        vals = sub[col].to_numpy()
        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            color=palette[i],
            edgecolor="#333333",
            linewidth=0.4,
            label=col.replace("Delta", ""),
        )
        # Label Sector 2 (usually the biggest spread) and any large deltas
        if col == "Sector2Delta":
            for bar, val in zip(bars, vals, strict=False):
                if pd.isna(val):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.01 if val >= 0 else -0.04),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=7,
                    color="#333333",
                )
    ax.set_xticks(x)
    ax.set_xticklabels(sub.index)
    ax.set_ylabel("Delta vs session-best sector (s)")
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.legend(loc="upper right")
    return finish_figure(
        fig,
        ax,
        f"{tag}_a1_sector_deltas.png",
        "Sector-time deltas vs session best",
        session,
    )


def _fig_pace_ranking(
    ranking: pd.DataFrame,
    merged: pd.DataFrame,
    session: Any | None,
    tag: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    r = ranking.sort_values("MedianFuelCorrectedPace", ascending=True).copy()
    fastest = float(r["MedianFuelCorrectedPace"].min())
    r["DeltaToFastest"] = r["MedianFuelCorrectedPace"] - fastest
    colors = [team_color(d, session) or "#1f4e79" for d in r["DriverCode"]]
    y = np.arange(len(r))
    ax.barh(
        y,
        r["DeltaToFastest"],
        color=colors,
        edgecolor="#333333",
        linewidth=0.4,
        height=0.72,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(r["DriverCode"].tolist())
    ax.set_xlabel("Delta to fastest median (s)")
    xmax = float(r["DeltaToFastest"].max())
    ax.set_xlim(0, xmax * 1.18 if xmax > 0 else 1)
    for yi, val in zip(y, r["DeltaToFastest"], strict=False):
        ax.text(
            val + xmax * 0.015,
            yi,
            f"+{val:.2f}",
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )
    return finish_figure(
        fig,
        ax,
        f"{tag}_a1_pace_ranking.png",
        "Fuel-corrected pace ranking",
        session,
    )


def _pick_telemetry_drivers(merged: pd.DataFrame) -> list[str]:
    """Pick two top finishers from different teams so traces are distinguishable."""
    ordered = merged.sort_values("Position").dropna(subset=["Position"])
    if ordered.empty:
        return []
    first = ordered.iloc[0]
    d1 = str(first["DriverCode"])
    team1 = str(first.get("Team", ""))
    for _, row in ordered.iloc[1:].iterrows():
        team2 = str(row.get("Team", ""))
        if team2 and team1 and team2 != team1:
            return [d1, str(row["DriverCode"])]
        if not team1 or not team2:
            # Fall back: first vs third finisher if teams unknown
            continue
    # Last resort: first vs second even if teammates
    codes = ordered["DriverCode"].astype(str).tolist()
    return codes[:2]


def plot_telemetry_overlay(
    merged: pd.DataFrame,
    session: Any | None,
    tag: str,
) -> Path | None:
    """
    Isolated telemetry load for two drivers' fastest laps.

    Loads telemetry only inside this function so the main pipeline can stay light.
    Picks drivers from different teams so colours differ.
    """
    try:
        year = DEFAULT_YEAR
        rnd = DEFAULT_ROUND
        if session is not None:
            try:
                year = int(session.event.year)
                rnd = int(session.event["RoundNumber"])
            except Exception:  # noqa: BLE001
                pass

        enable_cache()
        tel_session = load_session(year, rnd, "R", telemetry=True)

        drivers = _pick_telemetry_drivers(merged)
        if len(drivers) < 2:
            return None

        setup_plotting()
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)
        styles = ["-", "--"]
        for i, driver in enumerate(drivers):
            lap = tel_session.laps.pick_drivers(driver).pick_fastest()
            tel = lap.get_car_data().add_distance()
            color = team_color(driver, tel_session) or team_color(str(lap["Team"]), tel_session)
            kw = {
                "label": driver,
                "color": color,
                "linewidth": 1.6,
                "linestyle": styles[i % 2],
            }
            axes[0].plot(tel["Distance"], tel["Speed"], **kw)
            axes[1].plot(tel["Distance"], tel["Throttle"], **kw)
            axes[2].plot(tel["Distance"], tel["Brake"], **kw)

        axes[0].set_ylabel("Speed (km/h)")
        axes[1].set_ylabel("Throttle (%)")
        axes[2].set_ylabel("Brake")
        axes[2].set_xlabel("Distance (m)")
        axes[0].legend(loc="upper right", title="Driver")
        for ax in axes:
            ax.grid(True, axis="y", alpha=0.25)

        annotate_figure(
            fig,
            axes[0],
            "Fastest-lap telemetry overlay",
            event_subtitle(tel_session),
        )
        fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.86))
        return save_fig(fig, f"{tag}_a1_telemetry_overlay.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telemetry overlay skipped: %s", exc)
        return None
