"""Canonical race-engineering visual theme for all F1 figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from f1_analytics import config
from f1_analytics.config import FIGURES_DIR

# Canonical compound colours — used in every figure, no overrides.
COMPOUND_COLORS: dict[str, str] = {
    "SOFT": "#DA291C",
    "MEDIUM": "#FFD12E",
    "HARD": "#F0F0EC",
    "INTERMEDIATE": "#43B02A",
    "WET": "#0067AD",
    "UNKNOWN": "#888888",
}

HARD_EDGE = "#333333"
# Line plots can't rely on a light fill + edge the way bars can — use a
# mid-dark grey stroke so HARD stays readable on white.
HARD_LINE = "#5C5C58"
SOURCE_CREDIT = "Source: FastF1"
SUBTITLE_COLOR = "#666666"

# Set by process_race so figures get the correct event label even when
# session is None (checkpoint-only regenerations).
_CURRENT_EVENT_SUBTITLE: str | None = None


def set_event_subtitle(text: str | None) -> None:
    """Bind the active race/season subtitle for annotate/finish helpers."""
    global _CURRENT_EVENT_SUBTITLE
    _CURRENT_EVENT_SUBTITLE = text


def clear_event_subtitle() -> None:
    set_event_subtitle(None)


def apply_theme() -> None:
    """Set global rcParams once. Call at the start of every analysis run."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlepad": 12,
        }
    )
    try:
        import fastf1.plotting

        fastf1.plotting.setup_mpl(
            mpl_timedelta_support=False,
            color_scheme=None,
            misc_mpl_mods=False,
        )
    except Exception:  # noqa: BLE001
        pass


def compound_color(compound: str | None) -> str:
    """Return the canonical fill colour for a tyre compound."""
    key = str(compound).upper().strip() if compound is not None else "UNKNOWN"
    return COMPOUND_COLORS.get(key, COMPOUND_COLORS["UNKNOWN"])


def compound_line_color(compound: str | None) -> str:
    """Stroke colour for line charts (HARD slightly darker so it reads on white)."""
    key = str(compound).upper().strip() if compound is not None else "UNKNOWN"
    if key == "HARD":
        return HARD_LINE
    return compound_color(key)


def compound_edge(compound: str | None) -> str:
    """Edge colour — darker for HARD/off-white so it reads on white backgrounds."""
    key = str(compound).upper().strip() if compound is not None else "UNKNOWN"
    if key in {"HARD", "UNKNOWN"}:
        return HARD_EDGE
    return HARD_EDGE if compound_color(key).upper() in {"#F0F0EC", "#FFFFFF", "#F0F0F0"} else "#222222"


def present_compounds(values: Any) -> list[str]:
    """Ordered unique compounds actually present in a series/list."""
    order = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"]
    seen = {
        str(v).upper().strip()
        for v in values
        if v is not None and str(v).strip() and str(v).lower() != "nan"
    }
    return [c for c in order if c in seen]


def compound_legend_handles(compounds: list[str] | None = None):
    """Legend patches with correct compound colours (empty barh proxies break)."""
    compounds = compounds or ["SOFT", "MEDIUM", "HARD"]
    return [
        Patch(
            facecolor=compound_color(c),
            edgecolor=compound_edge(c),
            linewidth=1.0,
            label=c,
        )
        for c in compounds
    ]


def compound_line_legend_handles(compounds: list[str] | None = None):
    """Legend line proxies matching compound_line_color."""
    compounds = compounds or ["SOFT", "MEDIUM", "HARD"]
    return [
        Line2D([0], [0], color=compound_line_color(c), linewidth=3, label=c)
        for c in compounds
    ]


def team_color(identifier: str, session: Any | None) -> str | None:
    """FastF1 team/driver colour; None if unavailable."""
    if session is None or not identifier:
        return None
    try:
        import fastf1.plotting

        try:
            return fastf1.plotting.get_driver_color(identifier, session=session)
        except Exception:  # noqa: BLE001
            return fastf1.plotting.get_team_color(identifier, session=session)
    except Exception:  # noqa: BLE001
        return None


def event_subtitle(
    session: Any | None = None,
    *,
    fallback: str | None = None,
) -> str:
    """
    Event line under the chart title.

    Prefer the live session, then the process_race context subtitle.
    Never default to a hardcoded race (that stamped every chart as
    '2024 Bahrain GP · Race' when session was missing).
    """
    if session is not None:
        try:
            year_str = str(session_year(session))
            name = str(session.event["EventName"])
            return f"{year_str} {name} · Race"
        except Exception:  # noqa: BLE001
            pass
    if _CURRENT_EVENT_SUBTITLE:
        return _CURRENT_EVENT_SUBTITLE
    if fallback is not None:
        return fallback
    return ""


def session_year(session: Any) -> int:
    try:
        return int(session.event.year)
    except Exception:  # noqa: BLE001
        return int(str(session.date)[:4])


def annotate_figure(
    fig: Figure,
    ax: Axes | None,
    title: str,
    subtitle: str,
    *,
    wrap_subtitle: bool = False,
) -> None:
    """Short title + optional grey subtitle + FastF1 credit — never overlapping."""
    del wrap_subtitle  # reserved
    target = ax if ax is not None else fig.axes[0]
    sub = (subtitle or "").strip()
    n_sub = max(sub.count("\n") + 1, 1) if sub else 0
    fig.suptitle(title, fontsize=15, fontweight="semibold", x=0.02, ha="left", y=0.995)
    if sub:
        fig.text(
            0.02,
            0.995 - 0.045,
            sub,
            fontsize=10,
            color=SUBTITLE_COLOR,
            ha="left",
            va="top",
            linespacing=1.4,
        )
    fig.text(
        0.99,
        0.01,
        SOURCE_CREDIT,
        transform=fig.transFigure,
        fontsize=8,
        color="#999999",
        ha="right",
        va="bottom",
    )
    target.set_title("")
    fig._f1_subtitle_lines = max(n_sub, 1)  # type: ignore[attr-defined]


def save_fig(fig: Figure, name: str, *, directory: Path | None = None) -> Path:
    """Save PNG + SVG (print/zoom quality for multi-season report)."""
    out_dir = directory if directory is not None else config.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    png_path = out_dir / f"{stem}.png"
    svg_path = out_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path


class figure_output_dir:
    """Temporarily redirect figure exports to a race/season subdirectory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._prev: Path | None = None

    def __enter__(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        self._prev = config.FIGURES_DIR
        config.FIGURES_DIR = self.path
        return self.path

    def __exit__(self, *args: object) -> None:
        assert self._prev is not None
        config.FIGURES_DIR = self._prev
