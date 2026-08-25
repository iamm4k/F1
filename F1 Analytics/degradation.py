"""Tyre degradation fitting with green-flag / per-stint isolation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from f1_analytics.config import DRY_COMPOUNDS

logger = logging.getLogger(__name__)


def _dry_only(laps: pd.DataFrame) -> pd.DataFrame:
    if laps.empty or "Compound" not in laps.columns:
        return laps.copy()
    return laps.loc[laps["Compound"].astype(str).str.upper().isin(DRY_COMPOUNDS)].copy()


def is_green_flag(status: Any) -> bool:
    """True when TrackStatus is clear green (code 1 only)."""
    if status is None or (isinstance(status, float) and np.isnan(status)):
        return True
    text = str(status).strip()
    if not text or text.lower() == "nan":
        return True
    # Any SC (4), VSC (5/6), yellow (2), red (5 variants) → exclude
    return text == "1"


def filter_green_flag_laps(laps: pd.DataFrame) -> pd.DataFrame:
    if laps.empty:
        return laps.copy()
    out = laps.copy()
    if "TrackStatus" in out.columns:
        out = out[out["TrackStatus"].map(is_green_flag)]
    return out.reset_index(drop=True)


def stint_clean_laps(laps: pd.DataFrame, *, min_tyre_life: int = 2) -> pd.DataFrame:
    """
    Green-flag flying laps suitable for degradation fits.

    Dry compounds only (SOFT/MEDIUM/HARD). Wet/INTER laps are excluded so
    they never pollute dry degradation slopes.
    """
    clean = _dry_only(filter_green_flag_laps(laps))
    clean = clean.dropna(subset=["TyreLife", "FuelCorrectedLapTime", "DriverCode", "Stint"])
    clean = clean[clean["TyreLife"].astype(float) >= float(min_tyre_life)]
    return clean.reset_index(drop=True)


def _fit_one(x: np.ndarray, y: np.ndarray) -> dict[str, float] | None:
    if len(x) < 5:
        return None
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    pred = model.predict(x.reshape(-1, 1))
    resid = y - pred
    dof = max(len(y) - 2, 1)
    mse = float(np.sum(resid**2) / dof)
    x_var = float(np.sum((x - x.mean()) ** 2))
    slope_se = float(np.sqrt(mse / x_var)) if x_var > 0 else np.nan
    tcrit = float(stats.t.ppf(0.975, dof))
    slope = float(model.coef_[0])
    return {
        "Slope": slope,
        "Intercept": float(model.intercept_),
        "R2": float(r2_score(y, pred)),
        "SlopeCILow": slope - tcrit * slope_se,
        "SlopeCIHigh": slope + tcrit * slope_se,
        "N": float(len(y)),
    }


def fit_degradation(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Fit LapTime ~ TyreLife per driver-stint, then aggregate by compound.

    Pooling raw laps across drivers mixes pace offsets into the slope and
    previously made SOFT look flatter than HARD. Per-stint slopes isolate
    degradation; the compound slope is the sample-size-weighted mean.
    """
    clean = stint_clean_laps(laps)
    stint_rows: list[dict[str, Any]] = []

    for (driver, stint, compound), group in clean.groupby(
        ["DriverCode", "Stint", "Compound"], dropna=False
    ):
        x = group["TyreLife"].astype(float).to_numpy()
        y = group["FuelCorrectedLapTime"].astype(float).to_numpy()
        fit = _fit_one(x, y)
        if fit is None:
            continue
        stint_rows.append(
            {
                "DriverCode": driver,
                "Stint": stint,
                "Compound": compound,
                **fit,
            }
        )

    stint_fits = pd.DataFrame(stint_rows)
    compound_rows: list[dict[str, Any]] = []

    if stint_fits.empty:
        for compound in sorted(laps["Compound"].dropna().unique()):
            compound_rows.append(
                {
                    "Compound": compound,
                    "Slope": np.nan,
                    "Intercept": np.nan,
                    "R2": np.nan,
                    "SlopeCILow": np.nan,
                    "SlopeCIHigh": np.nan,
                    "N": 0,
                    "NStints": 0,
                    "Method": "per_stint_aggregate",
                    "Note": "insufficient_stint_samples",
                }
            )
        return pd.DataFrame(compound_rows)

    for compound, group in stint_fits.groupby("Compound"):
        weights = group["N"].to_numpy()
        slopes = group["Slope"].to_numpy()
        w_mean = float(np.average(slopes, weights=weights))
        # Pooled residual-style CI: SE of weighted mean across stints
        if len(slopes) > 1:
            w_var = float(np.average((slopes - w_mean) ** 2, weights=weights))
            se = float(np.sqrt(w_var / len(slopes)))
            tcrit = float(stats.t.ppf(0.975, max(len(slopes) - 1, 1)))
            ci_low, ci_high = w_mean - tcrit * se, w_mean + tcrit * se
        else:
            ci_low = float(group["SlopeCILow"].iloc[0])
            ci_high = float(group["SlopeCIHigh"].iloc[0])

        # Representative intercept: median of stint intercepts (for plotting only)
        intercept = float(group["Intercept"].median())
        # Mean of per-stint R² (descriptive, not a pooled R²)
        mean_r2 = float(group["R2"].mean())

        compound_rows.append(
            {
                "Compound": compound,
                "Slope": w_mean,
                "Intercept": intercept,
                "R2": mean_r2,
                "SlopeCILow": ci_low,
                "SlopeCIHigh": ci_high,
                "N": int(group["N"].sum()),
                "NStints": int(len(group)),
                "Method": "per_stint_aggregate",
                "Note": "ok",
            }
        )

    result = pd.DataFrame(compound_rows).sort_values("Compound").reset_index(drop=True)

    # Honest reporting
    soft = result.loc[result["Compound"] == "SOFT", "Slope"]
    hard = result.loc[result["Compound"] == "HARD", "Slope"]
    if len(soft) and len(hard) and pd.notna(soft.iloc[0]) and pd.notna(hard.iloc[0]):
        if soft.iloc[0] <= hard.iloc[0]:
            note = (
                f"SOFT slope ({soft.iloc[0]:.4f}) is not steeper than HARD "
                f"({hard.iloc[0]:.4f}) after per-stint green-flag fits — "
                "short Soft stints / limited tyre-life span may under-identify Soft wear."
            )
            logger.warning(note)
            result.loc[result["Compound"] == "SOFT", "Note"] = "soft_not_steeper_than_hard"
            result.attrs["honesty_note"] = note
        else:
            result.attrs["honesty_note"] = (
                f"SOFT slope {soft.iloc[0]:.4f} > HARD {hard.iloc[0]:.4f} (expected)."
            )

    for _, row in result.iterrows():
        logger.info(
            "Degradation %s: slope=%.4f intercept=%.3f mean_stint_R2=%.3f n=%s stints=%s (%s)",
            row["Compound"],
            row["Slope"] if pd.notna(row["Slope"]) else float("nan"),
            row["Intercept"] if pd.notna(row["Intercept"]) else float("nan"),
            row["R2"] if pd.notna(row["R2"]) else float("nan"),
            row["N"],
            row["NStints"],
            row["Note"],
        )

    # Attach per-stint detail for optional export
    result.attrs["stint_fits"] = stint_fits
    return result


def demeaned_stint_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Green-flag laps with stint-mean removed (pace-offset free for scatter)."""
    clean = stint_clean_laps(laps)
    if clean.empty:
        return clean
    clean = clean.copy()
    clean["StintDemeanedLapTime"] = clean["FuelCorrectedLapTime"] - clean.groupby(
        ["DriverCode", "Stint"]
    )["FuelCorrectedLapTime"].transform("mean")
    return clean
