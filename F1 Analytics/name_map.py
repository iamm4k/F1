"""Explicit FastF1 ↔ Kaggle driver name normalisation."""

from __future__ import annotations

import logging
import re
import unicodedata

import pandas as pd

logger = logging.getLogger(__name__)

# Explicit overrides for known mismatches / aliases.
NAME_TO_CODE: dict[str, str] = {
    "max verstappen": "VER",
    "lewis hamilton": "HAM",
    "charles leclerc": "LEC",
    "carlos sainz": "SAI",
    "carlos sainz jr": "SAI",
    "sergio perez": "PER",
    "sergio pérez": "PER",
    "george russell": "RUS",
    "lando norris": "NOR",
    "oscar piastri": "PIA",
    "fernando alonso": "ALO",
    "lance stroll": "STR",
    "pierre gasly": "GAS",
    "esteban ocon": "OCO",
    "alexander albon": "ALB",
    "alex albon": "ALB",
    "yuki tsunoda": "TSU",
    "daniel ricciardo": "RIC",
    "nico hulkenberg": "HUL",
    "nico hülkenberg": "HUL",
    "kevin magnussen": "MAG",
    "valtteri bottas": "BOT",
    "zhou guanyu": "ZHO",
    "guanyu zhou": "ZHO",
    "logan sargeant": "SAR",
    "oliver bearman": "BEA",
    "ollie bearman": "BEA",
    "franco colapinto": "COL",
    "liam lawson": "LAW",
    "jack doohan": "DOO",
    "nyck de vries": "DEV",
    "mick schumacher": "MSC",
    "nicholas latifi": "LAT",
    "nikita mazepin": "MAZ",
    "antonio giovinazzi": "GIO",
    "kimi raikkonen": "RAI",
    "kimi räikkönen": "RAI",
    "sebastian vettel": "VET",
    "romain grosjean": "GRO",
    "daniil kvyat": "KVY",
    "isack hadjar": "HAD",
    "andrea kimi antonelli": "ANT",
    "kimi antonelli": "ANT",
    "gabriel bortoleto": "BOR",
}


def _strip_accents(value: str) -> str:
    normalised = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalised if not unicodedata.combining(ch))


def normalise_name(name: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = _strip_accents(str(name)).lower().strip()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Drop common suffixes
    text = re.sub(r"\bjr\b$", "", text).strip()
    return text


def name_to_code(name: str, fallback_code: str | None = None) -> str | None:
    """Map a full name to a 3-letter driver code."""
    key = normalise_name(name)
    if key in NAME_TO_CODE:
        return NAME_TO_CODE[key]
    if fallback_code and isinstance(fallback_code, str) and len(fallback_code) == 3:
        return fallback_code.upper()
    return None


def attach_driver_codes(
    frame: pd.DataFrame,
    name_col: str,
    *,
    code_col: str = "DriverCode",
    fallback_code_col: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add DriverCode column; return frame and list of unmatched names."""
    out = frame.copy()
    codes: list[str | None] = []
    unmatched: list[str] = []
    for _, row in out.iterrows():
        fallback = row[fallback_code_col] if fallback_code_col else None
        code = name_to_code(row[name_col], fallback_code=fallback)
        codes.append(code)
        if code is None:
            unmatched.append(str(row[name_col]))
    out[code_col] = codes
    if unmatched:
        unique = sorted(set(unmatched))
        logger.warning("Unmatched driver names (%s): %s", len(unique), unique)
    return out, sorted(set(unmatched))
