from functools import lru_cache
from typing import List, Dict
import logging
from time import perf_counter

import pandas as pd

try:
    from .paths import MATCHES_PATH
except ImportError:  # pragma: no cover - fallback for direct module execution
    from src.backend.paths import MATCHES_PATH

logger = logging.getLogger("futbolconu.match_service")


@lru_cache(maxsize=1)
def load_matches() -> pd.DataFrame:
    start_time = perf_counter()
    df = pd.read_csv(MATCHES_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    logger.info(
        "load_matches_completed rows=%s cols=%s elapsed_ms=%.2f",
        len(df),
        len(df.columns),
        (perf_counter() - start_time) * 1000.0,
    )
    return df


def _format_matches(df: pd.DataFrame) -> List[Dict[str, int]]:
    formatted = df.copy()
    if "date" in formatted.columns:
        formatted["date"] = pd.to_datetime(formatted["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        formatted["date"] = formatted["date"].where(formatted["date"].notna(), None)
    return formatted[["date", "home_team", "away_team", "home_score", "away_score"]].to_dict(orient="records")


def get_recent_matches(
    home_team: str,
    away_team: str,
    last_matches: int,
    request_id: str = "-",
) -> List[Dict[str, int]]:
    start_time = perf_counter()
    cache_before = load_matches.cache_info()
    df = load_matches()
    cache_after = load_matches.cache_info()
    cache_event = "hit" if cache_after.hits > cache_before.hits else "miss"
    logger.info(
        "recent_form_service_started request_id=%s home_team=%s away_team=%s last_matches=%s rows=%s cache=%s",
        request_id,
        home_team,
        away_team,
        last_matches,
        len(df),
        cache_event,
    )
    recent_home = (
        df[(df["home_team"] == home_team) | (df["away_team"] == home_team)]
        .sort_values("date", ascending=False)
        .head(last_matches)
    )
    recent_away = (
        df[(df["home_team"] == away_team) | (df["away_team"] == away_team)]
        .sort_values("date", ascending=False)
        .head(last_matches)
    )

    combined = pd.concat([recent_home, recent_away], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["date", "home_team", "away_team", "home_score", "away_score"]
    ).sort_values("date", ascending=False)

    result = _format_matches(combined)
    logger.info(
        "recent_form_service_completed request_id=%s returned_matches=%s elapsed_ms=%.2f",
        request_id,
        len(result),
        (perf_counter() - start_time) * 1000.0,
    )
    return result


_FORM_TOURNAMENTS = {
    "AFC Asian Cup",
    "AFC Asian Cup qualification",
    "African Cup of Nations",
    "African Cup of Nations qualification",
    "Arab Cup",
    "CONCACAF Championship",
    "CONCACAF Championship qualification",
    "CONCACAF Nations League",
    "CONCACAF Nations League qualification",
    "Copa América",
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "Friendly",
    "Gold Cup",
    "Gold Cup qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "UEFA Nations League",
    "Confederations Cup",
}


def _team_form(team: str, matches: pd.DataFrame) -> Dict[str, int]:
    wins = draws = losses = goals = 0
    filtered = matches[matches["tournament"].isin(_FORM_TOURNAMENTS)]
    for _, row in filtered.iterrows():
        if row["home_team"] == team:
            team_goals = row["home_score"]
            opp_goals = row["away_score"]
        else:
            team_goals = row["away_score"]
            opp_goals = row["home_score"]

        goals += int(team_goals)
        if team_goals > opp_goals:
            wins += 1
        elif team_goals == opp_goals:
            draws += 1
        else:
            losses += 1

    return {"team": team, "wins": wins, "draws": draws, "losses": losses, "goals": goals}


def get_head_to_head(
    home_team: str,
    away_team: str,
    tournaments: List[str],
    request_id: str = "-",
) -> Dict[str, object]:
    start_time = perf_counter()
    cache_before = load_matches.cache_info()
    df = load_matches()
    cache_after = load_matches.cache_info()
    cache_event = "hit" if cache_after.hits > cache_before.hits else "miss"
    logger.info(
        "head_to_head_service_started request_id=%s home_team=%s away_team=%s tournaments=%s rows=%s cache=%s",
        request_id,
        home_team,
        away_team,
        len(tournaments or []),
        len(df),
        cache_event,
    )

    if tournaments:
        df = df[df["tournament"].isin(tournaments)]

    h2h = df[
        ((df["home_team"] == home_team) & (df["away_team"] == away_team))
        | ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    ].sort_values("date", ascending=False)

    matches = _format_matches(h2h)
    result = {
        "matches": matches,
        "home_form": _team_form(home_team, h2h),
        "away_form": _team_form(away_team, h2h),
    }
    logger.info(
        "head_to_head_service_completed request_id=%s returned_matches=%s elapsed_ms=%.2f",
        request_id,
        len(matches),
        (perf_counter() - start_time) * 1000.0,
    )
    return result


def _team_record(team: str, matches: pd.DataFrame) -> Dict[str, int]:
    wins = draws = losses = goals_for = goals_against = 0
    for _, row in matches.iterrows():
        if row["home_team"] == team:
            team_goals = row["home_score"]
            opp_goals = row["away_score"]
        else:
            team_goals = row["away_score"]
            opp_goals = row["home_score"]

        goals_for += int(team_goals)
        goals_against += int(opp_goals)
        if team_goals > opp_goals:
            wins += 1
        elif team_goals == opp_goals:
            draws += 1
        else:
            losses += 1

    return {
        "matches_count": int(len(matches)),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
    }


def get_team_vs_confed(
    team: str,
    opponent_confederation: str,
    request_id: str = "-",
) -> Dict[str, object]:
    start_time = perf_counter()
    cache_before = load_matches.cache_info()
    df = load_matches()
    cache_after = load_matches.cache_info()
    cache_event = "hit" if cache_after.hits > cache_before.hits else "miss"
    logger.info(
        "team_vs_confed_service_started request_id=%s team=%s opponent_confed=%s rows=%s cache=%s",
        request_id,
        team,
        opponent_confederation,
        len(df),
        cache_event,
    )
    df = df[df["tournament"].isin(_FORM_TOURNAMENTS)]
    confed = (opponent_confederation or "").strip().upper()

    home_conf = df["home_team_confederation"].astype(str).str.upper()
    away_conf = df["away_team_confederation"].astype(str).str.upper()

    filtered = df[
        ((df["home_team"] == team) & (away_conf == confed))
        | ((df["away_team"] == team) & (home_conf == confed))
    ]

    record = _team_record(team, filtered)
    result = {
        "team": team,
        "opponent_confederation": confed,
        **record,
    }
    logger.info(
        "team_vs_confed_service_completed request_id=%s matches_count=%s elapsed_ms=%.2f",
        request_id,
        result.get("matches_count"),
        (perf_counter() - start_time) * 1000.0,
    )
    return result
