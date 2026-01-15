from functools import lru_cache
from typing import List, Dict

import pandas as pd

from src.utils.paths import MATCHES_PATH


@lru_cache(maxsize=1)
def load_matches() -> pd.DataFrame:
    df = pd.read_csv(MATCHES_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _format_matches(df: pd.DataFrame) -> List[Dict[str, int]]:
    return df[["home_team", "away_team", "home_score", "away_score"]].to_dict(orient="records")


def get_recent_matches(home_team: str, away_team: str, last_matches: int) -> List[Dict[str, int]]:
    df = load_matches()
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

    return _format_matches(combined)


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


def get_head_to_head(home_team: str, away_team: str, tournaments: List[str]) -> Dict[str, object]:
    df = load_matches()

    if tournaments:
        df = df[df["tournament"].isin(tournaments)]

    h2h = df[
        ((df["home_team"] == home_team) & (df["away_team"] == away_team))
        | ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    ].sort_values("date", ascending=False)

    matches = _format_matches(h2h)
    return {
        "matches": matches,
        "home_form": _team_form(home_team, h2h),
        "away_form": _team_form(away_team, h2h),
    }
