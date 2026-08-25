"""Analysis 3 — Predictive finishing position (illustrative on small-n)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

from f1_analytics.config import TABLES_DIR
from f1_analytics.plotting_utils import finish_figure, setup_plotting
from f1_analytics.viz_style import event_subtitle

logger = logging.getLogger(__name__)


def _engineer_features(merged: pd.DataFrame, stints: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    feat = merged[
        [
            "DriverCode",
            "GridPosition",
            "Position",
            "AirTempMean",
            "TrackTempMean",
            "RainfallAny",
            "QualiPosition",
        ]
    ].copy()

    stint_feat = (
        stints.groupby("DriverCode")
        .agg(
            NStints=("Stint", "nunique"),
            MeanStintLen=("StintLength", "mean"),
            SoftStints=("Compound", lambda s: int((s == "SOFT").sum())),
            MediumStints=("Compound", lambda s: int((s == "MEDIUM").sum())),
            HardStints=("Compound", lambda s: int((s == "HARD").sum())),
        )
        .reset_index()
    )
    pace = (
        laps.groupby("DriverCode")
        .agg(
            MedianPace=("FuelCorrectedLapTime", "median"),
            PaceStd=("FuelCorrectedLapTime", "std"),
        )
        .reset_index()
    )
    feat = feat.merge(stint_feat, on="DriverCode", how="left")
    feat = feat.merge(pace, on="DriverCode", how="left")
    feat["RainfallAny"] = feat["RainfallAny"].fillna(False).astype(int)
    return feat


def run_predictive(
    merged: pd.DataFrame,
    stints: pd.DataFrame,
    laps: pd.DataFrame,
    *,
    tag: str = "2024_r01",
    multi_race: pd.DataFrame | None = None,
) -> dict[str, Path | pd.DataFrame | str]:
    """
    Predict finish position from grid + stint/tyre/weather features.

    With a single race, CV is leave-one-driver-out style k-fold on ~20 rows —
    illustrative only, not production-grade. Prefer leave-one-race-out when
    multi_race feature frame is supplied.
    """
    setup_plotting()
    outputs: dict[str, Path | pd.DataFrame | str] = {}

    data = multi_race if multi_race is not None else _engineer_features(merged, stints, laps)
    feature_cols = [
        c
        for c in [
            "GridPosition",
            "QualiPosition",
            "AirTempMean",
            "TrackTempMean",
            "RainfallAny",
            "NStints",
            "MeanStintLen",
            "SoftStints",
            "MediumStints",
            "HardStints",
            "MedianPace",
            "PaceStd",
        ]
        if c in data.columns and data[c].notna().any()
    ]

    clean = data.dropna(subset=["Position"] + feature_cols).copy()
    # Without Kaggle, QualiPosition may be sparse — don't require it if mostly missing
    if len(clean) < 8 and "QualiPosition" in feature_cols:
        feature_cols = [c for c in feature_cols if c != "QualiPosition"]
        clean = data.dropna(subset=["Position"] + feature_cols).copy()

    n_races = int(clean["Round"].nunique()) if "Round" in clean.columns else 1
    caveat = (
        "ILLUSTRATIVE: single-race / small-n model. Metrics are unstable; "
        "do not treat as a production forecast."
        if multi_race is None or n_races <= 1
        else f"Season-scale leave-one-race-out across {n_races} races."
    )
    outputs["caveat"] = caveat

    if len(clean) < 8:
        note = pd.DataFrame([{"Status": "insufficient_rows", "N": len(clean), "Caveat": caveat}])
        path = TABLES_DIR / f"{tag}_a3_predictions.csv"
        note.to_csv(path, index=False)
        outputs["predictions"] = note
        outputs["predictions_path"] = path
        logger.warning("Analysis 3 skipped — only %s usable rows", len(clean))
        return outputs

    x = clean[feature_cols]
    y = clean["Position"].astype(float)

    model = XGBRegressor(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )

    n_splits = min(5, len(clean))
    if "Round" in clean.columns and clean["Round"].nunique() > 1:
        # Leave-one-race-out style: train on other rounds
        preds = np.full(len(clean), np.nan)
        for rnd in sorted(clean["Round"].unique()):
            train_idx = clean["Round"] != rnd
            test_idx = clean["Round"] == rnd
            if train_idx.sum() < 5:
                continue
            m = XGBRegressor(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
            )
            m.fit(x.loc[train_idx], y.loc[train_idx])
            preds[test_idx.to_numpy()] = m.predict(x.loc[test_idx])
        cv_method = "leave_one_race_out"
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        preds = cross_val_predict(model, x, y, cv=cv)
        cv_method = f"{n_splits}_fold_driver"

    clean = clean.copy()
    clean["PredictedPosition"] = preds
    clean["AbsError"] = (clean["PredictedPosition"] - clean["Position"]).abs()
    mae = float(mean_absolute_error(clean["Position"], clean["PredictedPosition"]))

    metrics = pd.DataFrame(
        [
            {"Metric": "MAE_position", "Value": mae, "CV": cv_method, "N": len(clean)},
            {"Metric": "caveat", "Value": np.nan, "CV": caveat, "N": len(clean)},
        ]
    )

    # Fit on all for importances
    model.fit(x, y)
    importance = pd.DataFrame(
        {"Feature": feature_cols, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    pred_path = TABLES_DIR / f"{tag}_a3_predictions.csv"
    met_path = TABLES_DIR / f"{tag}_a3_metrics.csv"
    imp_path = TABLES_DIR / f"{tag}_a3_feature_importance.csv"
    clean.to_csv(pred_path, index=False)
    metrics.to_csv(met_path, index=False)
    importance.to_csv(imp_path, index=False)
    outputs.update(
        {
            "predictions": clean,
            "metrics": metrics,
            "importance": importance,
            "predictions_path": pred_path,
            "metrics_path": met_path,
            "importance_path": imp_path,
            "mae": mae,
        }
    )

    # Predicted vs actual — caveat on its own subtitle line (never truncate title)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.scatter(clean["Position"], clean["PredictedPosition"], s=45, alpha=0.85, color="#1f4e79")
    lims = [0.5, max(clean["Position"].max(), clean["PredictedPosition"].max()) + 0.5]
    ax.plot(lims, lims, "--", color="grey", linewidth=1)
    for _, row in clean.iterrows():
        ax.annotate(row["DriverCode"], (row["Position"], row["PredictedPosition"]), fontsize=7)
    ax.set_xlabel("Actual finish position")
    ax.set_ylabel("Predicted finish position")
    ax.set_aspect("equal", adjustable="box")
    outputs["fig_pred_vs_actual"] = finish_figure(
        fig,
        ax,
        f"{tag}_a3_pred_vs_actual.png",
        f"Predicted vs actual position (MAE={mae:.2f})",
        subtitle=f"{event_subtitle()}\n{caveat}".strip(),
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.barh(importance["Feature"], importance["Importance"], color="#1f4e79", edgecolor="#333333")
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    outputs["fig_importance"] = finish_figure(
        fig,
        ax,
        f"{tag}_a3_feature_importance.png",
        "XGBoost feature importance",
    )

    logger.info("Analysis 3 complete — MAE=%.3f (%s)", mae, cv_method)
    return outputs
