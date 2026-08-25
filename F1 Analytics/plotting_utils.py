"""Shared plotting helpers — thin wrappers over viz_style."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from f1_analytics.viz_style import (
    annotate_figure,
    apply_theme,
    compound_color,
    compound_edge,
    event_subtitle,
    save_fig,
    team_color,
)

# Back-compat alias used by older call sites
setup_plotting = apply_theme


def finish_figure(
    fig: Figure,
    ax: Any,
    name: str,
    title: str,
    session: Any | None = None,
    *,
    subtitle: str | None = None,
    top: float | None = None,
    directory: Path | None = None,
) -> Path:
    text = subtitle or event_subtitle(session)
    annotate_figure(fig, ax, title, text)
    n_sub = getattr(fig, "_f1_subtitle_lines", text.count("\n") + 1)
    top_rect = top if top is not None else max(0.72, 0.90 - 0.06 * max(n_sub - 1, 0))
    fig.tight_layout(rect=(0.02, 0.04, 0.98, top_rect))
    return save_fig(fig, name, directory=directory)
