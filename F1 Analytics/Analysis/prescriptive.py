"""Analysis 4 — Prescriptive Monte Carlo pit-strategy simulation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from f1_analytics.analyses.descriptive import pit_loss_table
from f1_analytics.analyses.diagnostic import track_status_windows
from f1_analytics.config import TABLES_DIR
from f1_analytics.degradation import fit_degradation
from f1_analytics.plotting_utils import finish_figure, setup_plotting
from f1_analytics.viz_style import (
    annotate_figure,
    compound_color,
    compound_edge,
    compound_legend_handles,
    event_subtitle,
    present_compounds,
    save_fig,
)

logger = logging.getLogger(__name__)


def _base_pace(laps: pd.DataFrame, compound: str) -> float:
    subset = laps[laps["Compound"] == compound]["FuelCorrectedLapTime"].dropna()
    if subset.empty:
        return float(laps["FuelCorrectedLapTime"].median())
    # Use early-life pace as undegraded baseline
    early = laps[(laps["Compound"] == compound) & (laps["TyreLife"] <= 3)][
        "FuelCorrectedLapTime"
    ].dropna()
    return float(early.median() if len(early) else subset.median())


def simulate_strategies(
    laps: pd.DataFrame,
    stints: pd.DataFrame,
    total_laps: int,
    *,
    n_sims: int = 1500,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Monte Carlo total-race-time for compound sequences.

    Lap model: base_pace(compound) + slope(compound) * tyre_life + noise,
    plus pit loss per stop. SC probability shortens effective green-flag
    degradation / adds fixed SC time loss.
    """
    rng = rng or np.random.default_rng(42)
    deg = fit_degradation(laps).set_index("Compound")
    pits = pit_loss_table(laps)
    pit_loss = float(pits["PitLossSeconds"].median()) if not pits.empty else 22.0
    if not np.isfinite(pit_loss):
        pit_loss = 22.0

    sc_windows = track_status_windows(laps)
    sc_prior = min(0.35, max(0.05, len(sc_windows) / max(total_laps, 1)))

    # Precompute per-compound base pace and slope (avoid DataFrame work in the inner loop)
    compounds_seen = [c for c in laps["Compound"].dropna().unique()]
    base_pace = {c: _base_pace(laps, c) for c in compounds_seen}
    default_base = float(laps["FuelCorrectedLapTime"].median())
    slopes = {
        c: (
            float(deg.loc[c, "Slope"])
            if c in deg.index and pd.notna(deg.loc[c, "Slope"])
            else 0.05
        )
        for c in compounds_seen
    }

    strategies: list[dict[str, Any]] = [
        {"name": "1stop_SM", "compounds": ["SOFT", "MEDIUM"], "windows": [(18, 28)]},
        {"name": "1stop_MH", "compounds": ["MEDIUM", "HARD"], "windows": [(22, 34)]},
        {"name": "1stop_MS", "compounds": ["MEDIUM", "SOFT"], "windows": [(20, 30)]},
        {"name": "2stop_SMH", "compounds": ["SOFT", "MEDIUM", "HARD"], "windows": [(12, 18), (32, 42)]},
        {"name": "2stop_SMS", "compounds": ["SOFT", "MEDIUM", "SOFT"], "windows": [(14, 20), (34, 44)]},
    ]

    rows: list[dict[str, Any]] = []
    for strat in strategies:
        times: list[float] = []
        for _ in range(n_sims):
            pit_laps = []
            for lo, hi in strat["windows"]:
                lo_i = max(2, min(lo, total_laps - 2))
                hi_i = max(lo_i + 1, min(hi, total_laps - 1))
                pit_laps.append(int(rng.integers(lo_i, hi_i + 1)))
            pit_set = set(pit_laps)

            compound_idx = 0
            tyre_life = 1
            total = 0.0
            sc_happens = rng.random() < sc_prior
            sc_lap = int(rng.integers(8, max(9, total_laps - 5))) if sc_happens else None

            for lap in range(1, total_laps + 1):
                compound = strat["compounds"][compound_idx]
                slope = slopes.get(compound, 0.05)
                base = base_pace.get(compound, default_base)
                lap_time = base + slope * (tyre_life - 1) + float(rng.normal(0, 0.25))
                if sc_lap is not None and abs(lap - sc_lap) <= 2:
                    lap_time += 8.0
                total += lap_time
                tyre_life += 1

                if lap in pit_set and compound_idx < len(strat["compounds"]) - 1:
                    total += pit_loss + float(rng.normal(0, 0.8))
                    compound_idx += 1
                    tyre_life = 1

            times.append(total)

        arr = np.asarray(times)
        rows.append(
            {
                "Strategy": strat["name"],
                "Compounds": ">".join(strat["compounds"]),
                "MeanRaceTime": float(arr.mean()),
                "MedianRaceTime": float(np.median(arr)),
                "P10": float(np.percentile(arr, 10)),
                "P90": float(np.percentile(arr, 90)),
                "Std": float(arr.std()),
                "SCPrior": sc_prior,
                "PitLossUsed": pit_loss,
                "NSims": n_sims,
                "Samples": arr,
            }
        )

    result = pd.DataFrame(rows).sort_values("MeanRaceTime").reset_index(drop=True)
    return result


def actual_strategy_timeline(stints: pd.DataFrame, driver: str | None = None) -> pd.DataFrame:
    if stints.empty:
        return stints
    if driver is None:
        # pick most common winner-like: shortest mean lap among finishers — use first by lap start coverage
        driver = stints.groupby("DriverCode")["StintLength"].sum().idxmax()
    return stints[stints["DriverCode"] == driver].copy()


def run_prescriptive(
    laps: pd.DataFrame,
    stints: pd.DataFrame,
    merged: pd.DataFrame,
    session: Any | None = None,
    *,
    tag: str = "2024_r01",
    n_sims: int = 1500,
) -> dict[str, Path | pd.DataFrame]:
    setup_plotting()
    outputs: dict[str, Path | pd.DataFrame] = {}

    total_laps = int(laps["LapNumber"].max()) if not laps.empty else 57
    sim = simulate_strategies(laps, stints, total_laps, n_sims=n_sims)
    samples = {row["Strategy"]: row["Samples"] for _, row in sim.iterrows()}
    table = sim.drop(columns=["Samples"])
    best = table.iloc[0]

    recommendation = pd.DataFrame(
        [
            {
                "RecommendedStrategy": best["Strategy"],
                "Compounds": best["Compounds"],
                "ExpectedRaceTime": best["MeanRaceTime"],
                "P10": best["P10"],
                "P90": best["P90"],
                "Note": "Minimises expected total race time under MC tyre+SC model",
            }
        ]
    )

    # Actual race winner strategy for comparison
    winner = None
    if "Position" in merged.columns and merged["Position"].notna().any():
        winner = merged.sort_values("Position").iloc[0]["DriverCode"]
    actual = actual_strategy_timeline(stints, winner)

    path = TABLES_DIR / f"{tag}_a4_strategy_simulation.csv"
    rec_path = TABLES_DIR / f"{tag}_a4_recommendation.csv"
    act_path = TABLES_DIR / f"{tag}_a4_actual_winner_strategy.csv"
    table.to_csv(path, index=False)
    recommendation.to_csv(rec_path, index=False)
    actual.to_csv(act_path, index=False)
    outputs.update(
        {
            "simulation": table,
            "recommendation": recommendation,
            "actual": actual,
            "simulation_path": path,
            "recommendation_path": rec_path,
            "actual_path": act_path,
        }
    )

    # Distribution per strategy — KDE outlines (filled hist was muddy)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    palette = ["#1f4e79", "#DA291C", "#43B02A", "#C4A000", "#0067AD"]
    all_times = np.concatenate([np.asarray(v) for v in samples.values()])
    xs = np.linspace(float(all_times.min()), float(all_times.max()), 240)
    for i, (name, arr) in enumerate(samples.items()):
        color = palette[i % len(palette)]
        data = np.asarray(arr, dtype=float)
        kde = gaussian_kde(data)
        dens = kde(xs)
        ax.plot(xs, dens, color=color, linewidth=2.2, label=name)
        ax.fill_between(xs, dens, color=color, alpha=0.12)
    ax.set_xlabel("Simulated total race time (s)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, loc="upper right")
    outputs["fig_distributions"] = finish_figure(
        fig,
        ax,
        f"{tag}_a4_strategy_distributions.png",
        "Monte Carlo race-time distributions by strategy",
        session,
    )

    # Recommended vs actual timeline — legend outside, only compounds present
    fig, ax = plt.subplots(figsize=(11, 4.4))
    comps = str(best["Compounds"]).split(">")
    stint_len = total_laps // len(comps)
    left = 0
    used_compounds: list[str] = []
    for i, compound in enumerate(comps):
        width = stint_len if i < len(comps) - 1 else total_laps - left
        ax.barh(
            y=0,
            width=width,
            left=left,
            color=compound_color(compound),
            edgecolor=compound_edge(compound),
            height=0.55,
        )
        used_compounds.append(compound)
        left += width

    if not actual.empty:
        for _, row in actual.iterrows():
            ax.barh(
                y=1,
                width=row["StintLength"],
                left=row["LapStart"] - 1,
                color=compound_color(row["Compound"]),
                edgecolor=compound_edge(row["Compound"]),
                height=0.55,
            )
            used_compounds.append(row["Compound"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Recommended", f"Actual ({winner or 'n/a'})"])
    ax.set_xlabel("Lap")
    compounds = present_compounds(used_compounds)
    ax.legend(
        handles=compound_legend_handles(compounds),
        title="Compound",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=max(len(compounds), 1),
        frameon=True,
    )
    annotate_figure(
        fig,
        ax,
        f"Recommended ({best['Strategy']}) vs actual strategy",
        event_subtitle(session),
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.78, bottom=0.28)
    outputs["fig_timeline"] = save_fig(fig, f"{tag}_a4_recommended_vs_actual.png")

    logger.info(
        "Analysis 4 complete — recommend %s (E[T]=%.1fs)",
        best["Strategy"],
        best["MeanRaceTime"],
    )
    return outputs
