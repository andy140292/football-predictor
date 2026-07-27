from __future__ import annotations

from typing import Iterable

import pandas as pd

try:
    from backend.supabase_client import get_supabase_client
except ImportError:  # pragma: no cover - package-style import fallback
    from src.backend.supabase_client import get_supabase_client


SUPABASE_COLUMNS = (
    "match_date,home_team,away_team,home_score,away_score,"
    "tournament,season,round,neutral,source,source_file"
)
CANONICAL_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "result",
    "competition",
    "country",
    "season",
    "round",
    "neutral",
    "source",
    "source_file",
    "div_code",
]


def normalize_libertadores_matches(rows: Iterable[dict]) -> pd.DataFrame:
    """Convert Supabase rows to the canonical schema used by model training."""
    matches = pd.DataFrame(list(rows))
    if matches.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    matches = matches.rename(
        columns={
            "match_date": "date",
            "tournament": "competition",
        }
    )

    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = sorted(required.difference(matches.columns))
    if missing:
        raise ValueError(
            "libertadores_matches is missing required columns: "
            + ", ".join(missing)
        )

    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
    matches = matches.dropna(
        subset=["date", "home_team", "away_team", "home_score", "away_score"]
    ).copy()

    matches["result"] = "D"
    matches.loc[matches["home_score"] > matches["away_score"], "result"] = "H"
    matches.loc[matches["home_score"] < matches["away_score"], "result"] = "A"

    defaults = {
        "competition": "",
        "country": "South America",
        "season": "",
        "round": "",
        "neutral": False,
        "source": "",
        "source_file": "supabase:libertadores_matches",
        "div_code": "",
    }
    for column, default in defaults.items():
        if column not in matches.columns:
            matches[column] = default
        else:
            matches[column] = matches[column].fillna(default)

    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches.dropna(subset=["date"])
    return matches[CANONICAL_COLUMNS].sort_values("date").reset_index(drop=True)


def fetch_libertadores_matches(*, client=None, page_size: int = 1000) -> pd.DataFrame:
    """Fetch all completed Libertadores matches from Supabase with pagination."""
    if page_size <= 0:
        raise ValueError("page_size must be greater than zero")

    if client is None:
        client = get_supabase_client(use_service_role=True)

    rows = []
    offset = 0
    while True:
        response = (
            client.table("libertadores_matches")
            .select(SUPABASE_COLUMNS)
            .not_.is_("home_score", "null")
            .not_.is_("away_score", "null")
            .order("match_date", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    matches = normalize_libertadores_matches(rows)
    if matches.empty:
        raise RuntimeError(
            "No completed matches were loaded from Supabase table "
            "'libertadores_matches'."
        )
    return matches
