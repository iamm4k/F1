"""FastF1 session loading with local cache and retry/backoff."""

from __future__ import annotations

import logging
import time
from typing import Any

import fastf1

from f1_analytics.config import CACHE_DIR, ensure_dirs

logger = logging.getLogger(__name__)


def enable_cache(debug: bool = False) -> None:
    """Enable FastF1 SQLite cache on a local disk path."""
    ensure_dirs()
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    if debug:
        fastf1.set_log_level("DEBUG")
    logger.info("FastF1 cache enabled at %s", CACHE_DIR)


def _is_rate_limit(exc: BaseException) -> bool:
    name = type(exc).__name__
    text = str(exc).lower()
    return (
        "ratelimit" in name.lower()
        or "rate limit" in text
        or "500 calls" in text
        or "ratelimitexceeded" in name.lower()
    )


def load_session(
    year: int,
    round_number: int,
    session_type: str = "R",
    *,
    laps: bool = True,
    weather: bool = True,
    telemetry: bool = False,
    max_retries: int = 6,
    base_delay: float = 2.0,
) -> Any:
    """Load a FastF1 session with exponential backoff on transient failures."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            session = fastf1.get_session(year, round_number, session_type)
            session.load(laps=laps, weather=weather, telemetry=telemetry)
            # Force materialisation — soft FastF1 failures can leave empties
            if laps and (session.laps is None or len(session.laps) == 0):
                raise RuntimeError("Session loaded but laps are empty")
            _ = session.results
            return session
        except Exception as exc:  # noqa: BLE001 — FastF1 raises varied types
            last_error = exc
            if _is_rate_limit(exc):
                # Stay under 500 calls/h: wait several minutes between attempts
                delay = 90.0 * attempt
            else:
                delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Session load failed (attempt %s/%s): %s — retrying in %.1fs",
                attempt,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)

    assert last_error is not None
    raise RuntimeError(
        f"Failed to load {year} round {round_number} {session_type} "
        f"after {max_retries} attempts"
    ) from last_error
