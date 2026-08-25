"""Analysis 2 — Diagnostic: degradation, SC/VSC, undercut attribution."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1_analytics.config import TABLES_DIR
from f1_analytics.degradation import demeaned_stint_laps, fit_degradation
from f1_analytics.plotting_utils import finish_figure, setup_plotting
from f1_analytics.viz_style import compound_color, compound_edge, team_color

logger = logging.getLogger(__name__)


def track_status_windows(laps: pd.DataFrame) -> pd.DataFrame:
    """Extract SC/VSC windows from TrackStatus codes."""
    if "TrackStatus" not in laps.columns or laps.empty:
        return pd.DataFrame(columns=["LapNumber", "Status", "Label"])

    sample = laps.sort_values("LapNumber")
    by_lap = sample.groupby("LapNumber")["TrackStatus"].agg(
        lambda s: str(s.mode().iloc[0]) if len(s.mode()) else ""
    )
    rows = []
    for lap, status in by_lap.items():
        label = None
        if "4" in status:
            label = "SC"
        elif "6" in status or "5" in status:
            label = "VSC"
        elif "2" in status:
            label = "Yellow"
        if label:
            rows.append({"LapNumber": lap, "Status": status, "Label": label})
    return pd.DataFrame(rows)


def weather_outcome_corr(merged: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """Simple correlations between weather / SC exposure and finishing position."""
    sc_laps = track_status_windows(laps)
    sc_count = int((sc_laps["Label"] == "SC").sum()) if not sc_laps.empty else 0
    vsc_count = int((sc_laps["Label"] == "VSC").sum()) if not sc_laps.empty else 0

    rows = [
        {
            "Metric": "SC_lap_count",
            "Value": sc_count,
            "Note": "laps under Safety Car (status contains 4)",
        },
        {
            "Metric": "VSC_lap_count",
            "Value": vsc_count,
            "Note": "laps under VSC",
        },
    ]
    if "AirTempMean" in merged.columns and merged["AirTempMean"].notna().any():
        rows.append(
            {
                "Metric": "AirTempMean",
                "Value": float(merged["AirTempMean"].iloc[0]),
                "Note": "race-level mean",
            }
        )
        rows.append(
            {
                "Metric": "RainfallAny",
                "Value": float(bool(merged["RainfallAny"].iloc[0])),
                "Note": "any rainfall in session weather",
            }
        )

    if {"GridPosition", "Position"}.issubset(merged.columns):
        delta = merged.dropna(subset=["GridPosition", "Position"]).copy()
        delta["PosChange"] = delta["GridPosition"] - delta["Position"]
        rows.append(
            {
                "Metric": "MeanPositionsGained",
                "Value": float(delta["PosChange"].mean()),
                "Note": "grid - finish (positive = gained)",
            }
        )
    return pd.DataFrame(rows)


def undercut_attribution(stints: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """Heuristic undercut/overcut flags from early pit relative to median pit lap."""
    if stints.empty:
        return pd.DataFrame()

    first_stops = (
        stints[stints["Stint"] == 2]
        .groupby("DriverCode")["LapStart"]
        .min()
        .rename("FirstStopLap")
        .reset_index()
    )
    if first_stops.empty:
        return pd.DataFrame(columns=["DriverCode", "FirstStopLap", "StrategyNote"])

    median_stop = float(first_stops["FirstStopLap"].median())
    first_stops["StrategyNote"] = first_stops["FirstStopLap"].apply(
        lambda x: "undercut_attempt"
        if x < median_stop - 1
        else ("overcut_attempt" if x > median_stop + 1 else "nominal")
    )
    out = first_stops.merge(
        merged[["DriverCode", "GridPosition", "Position"]],
        on="DriverCode",
        how="left",
    )
    out["PosChange"] = out["GridPosition"] - out["Position"]
    out["MedianStopLap"] = median_stop
    return out


def gap_to_leader(session: Any) -> pd.DataFrame:
    """Build gap-to-leader time series from session laps if possible."""
    try:
        laps = session.laps.copy()
        leader_code = session.results.sort_values("Position").iloc[0]["Abbreviation"]
        leader = laps[laps["Driver"] == leader_code][["LapNumber", "Time"]].rename(
            columns={"Time": "LeaderTime"}
        )
        frames = []
        for driver, group in laps.groupby("Driver"):
            g = group[["Driver", "LapNumber", "Time"]].merge(leader, on="LapNumber", how="inner")
            g["GapToLeaderSeconds"] = (g["Time"] - g["LeaderTime"]).dt.total_seconds()
            frames.append(g[["Driver", "LapNumber", "GapToLeaderSeconds"]])
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gap-to-leader build failed: %s", exc)
        return pd.DataFrame()


def run_diagnostic(
    laps: pd.DataFrame,
    stints: pd.DataFrame,
    merged: pd.DataFrame,
    session: Any | None = None,
    *,
    tag: str = "2024_r01",
) -> dict[str, Path | pd.DataFrame]:
    setup_plotting()
    outputs: dict[str, Path | pd.DataFrame] = {}

    deg = fit_degradation(laps)
    sc = track_status_windows(laps)
    wx = weather_outcome_corr(merged, laps)
    under = undercut_attribution(stints, merged)
    stint_fits = getattr(deg, "attrs", {}).get("stint_fits", pd.DataFrame())

    for name, frame in {
        "a2_degradation_slopes": deg,
        "a2_degradation_stint_fits": stint_fits if isinstance(stint_fits, pd.DataFrame) else pd.DataFrame(),
        "a2_track_status_windows": sc,
        "a2_weather_sc_summary": wx,
        "a2_undercut_attribution": under,
    }.items():
        path = TABLES_DIR / f"{tag}_{name}.csv"
        frame.to_csv(path, index=False)
        outputs[name] = frame
        outputs[f"{name}_path"] = path

    honesty = getattr(deg, "attrs", {}).get("honesty_note", "")
    if honesty:
        note_path = TABLES_DIR / f"{tag}_a2_degradation_note.txt"
        note_path.write_text(honesty + "\n", encoding="utf-8")
        outputs["degradation_note"] = note_path

    # Degradation fits on stint-demeaned space
    demeaned = demeaned_stint_laps(laps)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for compound, group in demeaned.groupby("Compound"):
        ax.scatter(
            group["TyreLife"],
            group["StintDemeanedLapTime"],
            s=12,
            alpha=0.35,
            color=compound_color(compound),
            edgecolors=compound_edge(compound),
            linewidths=0.35,
        )
    for _, row in deg.iterrows():
        compound = row["Compound"]
        if pd.isna(row["Slope"]):
            continue
        color = compound_color(compound)
        subset = demeaned[demeaned["Compound"] == compound]
        if subset.empty:
            continue
        x = np.linspace(float(subset["TyreLife"].min()), float(subset["TyreLife"].max()), 80)
        x_c = x - x.mean()
        y = row["Slope"] * x_c
        ax.plot(
            x,
            y,
            color=color,
            linewidth=2.4,
            label=f"{compound} ({row['Slope']:.3f} s/lap, R²={row['R2']:.2f})",
        )
        if pd.notna(row.get("SlopeCILow")):
            ax.fill_between(
                x,
                row["SlopeCILow"] * x_c,
                row["SlopeCIHigh"] * x_c,
                color=color,
                alpha=0.15,
            )
    ax.set_xlabel("Tyre life")
    ax.set_ylabel("Stint-demeaned fuel-corrected lap time (s)")
    ax.legend(fontsize=8)
    subtitle_extra = honesty[:90] + ("…" if len(honesty) > 90 else "") if honesty else None
    from f1_analytics.viz_style import event_subtitle

    sub = event_subtitle(session)
    if subtitle_extra:
        sub = f"{sub}\n{subtitle_extra}"
    outputs["fig_deg_fits"] = finish_figure(
        fig,
        ax,
        f"{tag}_a2_degradation_fits.png",
        "Degradation slope fits by compound",
        session,
        subtitle=sub,
    )

    gaps = gap_to_leader(session) if session is not None else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    if not gaps.empty:
        top = (
            merged.sort_values("Position").dropna(subset=["Position"])["DriverCode"].head(5).tolist()
        )
        for driver in top:
            g = gaps[gaps["Driver"] == driver]
            color = team_color(driver, session)
            ax.plot(
                g["LapNumber"],
                g["GapToLeaderSeconds"],
                label=driver,
                linewidth=1.6,
                color=color,
            )
        for _, win in sc.iterrows():
            band = "#ffcccc" if win["Label"] == "SC" else "#ffe6cc"
            ax.axvspan(win["LapNumber"] - 0.5, win["LapNumber"] + 0.5, color=band, alpha=0.45)
        ax.set_ylabel("Gap to leader (s)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Gap-to-leader unavailable", ha="center", va="center")
    ax.set_xlabel("Lap")
    outputs["fig_gap"] = finish_figure(
        fig, ax, f"{tag}_a2_gap_to_leader.png", "Gap to leader with SC/VSC windows", session
    )

    logger.info("Analysis 2 complete — %s", tag)
    return outputs
