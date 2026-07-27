from __future__ import annotations

from typing import Iterable

import pandas as pd

try:
    from backend.supabase_client import get_supabase_client
except ImportError:  # pragma: no cover - package-style import fallback
    from src.backend.supabase_client import get_supabase_client


SUPABASE_COLUMNS = (
    "match_date,home_team,away_team,home_score,away_score,tournament,"
    "city,country,neutral,home_team_confederation,away_team_confederation,"
    "home_team_fifa_rank,home_team_fifa_points,"
    "away_team_fifa_rank,away_team_fifa_points"
)
CANONICAL_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
    "home_team_confederation",
    "away_team_confederation",
    "home_team_fifa_rank",
    "home_team_fifa_points",
    "away_team_fifa_rank",
    "away_team_fifa_points",
]


def normalize_national_matches(rows: Iterable[dict]) -> pd.DataFrame:
    """Convert Supabase rows to the canonical schema used by national training."""
    matches = pd.DataFrame(list(rows))
    if matches.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    matches = matches.rename(columns={"match_date": "date"})
    required = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
    }
    missing = sorted(required.difference(matches.columns))
    if missing:
        raise ValueError(
            "matches is missing required columns: " + ", ".join(missing)
        )

    optional_defaults = {
        "city": None,
        "country": None,
        "neutral": False,
        "home_team_confederation": None,
        "away_team_confederation": None,
        "home_team_fifa_rank": None,
        "home_team_fifa_points": None,
        "away_team_fifa_rank": None,
        "away_team_fifa_points": None,
    }
    for column, default in optional_defaults.items():
        if column not in matches.columns:
            matches[column] = default

    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    numeric_columns = [
        "home_score",
        "away_score",
        "home_team_fifa_rank",
        "home_team_fifa_points",
        "away_team_fifa_rank",
        "away_team_fifa_points",
    ]
    for column in numeric_columns:
        matches[column] = pd.to_numeric(matches[column], errors="coerce")

    matches = matches.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
        ]
    )
    return matches[CANONICAL_COLUMNS].sort_values("date").reset_index(drop=True)


def fetch_national_matches(*, client=None, page_size: int = 1000) -> pd.DataFrame:
    """Fetch all completed national-team matches from Supabase with pagination."""
    if page_size <= 0:
        raise ValueError("page_size must be greater than zero")

    if client is None:
        client = get_supabase_client(use_service_role=True)

    rows = []
    offset = 0
    while True:
        response = (
            client.table("matches")
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

    matches = normalize_national_matches(rows)
    if matches.empty:
        raise RuntimeError(
            "No completed matches were loaded from Supabase table 'matches'."
        )
    return matches


def fetch_national_team_names(*, client=None, page_size: int = 1000) -> list[str]:
    """Fetch the distinct national-team names represented in the matches table."""
    if page_size <= 0:
        raise ValueError("page_size must be greater than zero")

    if client is None:
        client = get_supabase_client(use_service_role=True)

    names = set()
    offset = 0
    while True:
        response = (
            client.table("matches")
            .select("home_team,away_team")
            .order("match_date", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = response.data or []
        for row in batch:
            for column in ("home_team", "away_team"):
                name = str(row.get(column) or "").strip()
                if name:
                    names.add(name)
        if len(batch) < page_size:
            break
        offset += page_size

    return sorted(names, key=str.casefold)
