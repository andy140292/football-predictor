from types import SimpleNamespace

import pandas as pd

from src.backend.prediction.match_data_preprocessor import MatchDataPreprocessor
from src.backend.prediction.national_match_source import (
    fetch_national_matches,
    fetch_national_team_names,
    normalize_national_matches,
)


def _supabase_row(match_date, home_team, away_team, home_score, away_score):
    return {
        "match_date": match_date,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "tournament": "Friendly",
        "city": "Los Angeles",
        "country": "United States",
        "neutral": True,
        "home_team_confederation": "CONMEBOL",
        "away_team_confederation": "UEFA",
        "home_team_fifa_rank": 1,
        "home_team_fifa_points": 1900,
        "away_team_fifa_rank": 2,
        "away_team_fifa_points": 1850,
    }


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows
        self.ranges = []
        self.current_range = (0, len(rows) - 1)
        self.not_ = self

    def select(self, _columns):
        return self

    def is_(self, _column, _value):
        return self

    def order(self, _column, desc=False):
        return self

    def range(self, start, end):
        self.current_range = (start, end)
        self.ranges.append(self.current_range)
        return self

    def execute(self):
        start, end = self.current_range
        return SimpleNamespace(data=self.rows[start : end + 1])


class FakeClient:
    def __init__(self, rows):
        self.query = FakeQuery(rows)
        self.table_name = None

    def table(self, table_name):
        self.table_name = table_name
        return self.query


def test_normalize_national_matches_renames_date_only_in_memory():
    row = _supabase_row("2026-06-01", "Argentina", "Spain", 2, 1)

    matches = normalize_national_matches([row])

    assert "match_date" in row
    assert "date" not in row
    assert list(matches["home_team"]) == ["Argentina"]
    assert list(matches["tournament"]) == ["Friendly"]
    assert pd.api.types.is_datetime64_any_dtype(matches["date"])


def test_fetch_national_matches_paginates_all_rows():
    rows = [
        _supabase_row("2026-06-01", "A", "B", 1, 0),
        _supabase_row("2026-06-02", "C", "D", 0, 0),
        _supabase_row("2026-06-03", "E", "F", 0, 2),
    ]
    client = FakeClient(rows)

    matches = fetch_national_matches(client=client, page_size=2)

    assert client.table_name == "matches"
    assert client.query.ranges == [(0, 1), (2, 3)]
    assert len(matches) == 3


def test_fetch_national_team_names_paginates_and_deduplicates():
    rows = [
        _supabase_row("2026-06-01", "Argentina", "Brazil", 1, 0),
        _supabase_row("2026-06-02", "Brazil", "Chile", 0, 0),
        _supabase_row("2026-06-03", "Argentina", "Uruguay", 0, 2),
    ]
    client = FakeClient(rows)

    names = fetch_national_team_names(client=client, page_size=2)

    assert client.table_name == "matches"
    assert client.query.ranges == [(0, 1), (2, 3)]
    assert names == ["Argentina", "Brazil", "Chile", "Uruguay"]


def test_match_preprocessor_accepts_matches_dataframe():
    matches_df = pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "home_team": "Argentina",
                "away_team": "Spain",
                "home_score": 2,
                "away_score": 1,
                "tournament": "Friendly",
            }
        ]
    )

    preprocessor = MatchDataPreprocessor(matches_df=matches_df)
    preprocessor.load_and_filter_data()

    assert preprocessor.file_path == "supabase:matches"
    assert len(preprocessor.matches) == 1
