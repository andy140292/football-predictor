from types import SimpleNamespace

import pandas as pd

from src.backend.prediction.club_match_data_preprocessor import (
    ClubMatchDataPreprocessor,
)
from src.backend.prediction.libertadores_match_source import (
    fetch_libertadores_matches,
    normalize_libertadores_matches,
)


def _supabase_row(match_date, home_team, away_team, home_score, away_score):
    return {
        "match_date": match_date,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "tournament": "Libertadores",
        "season": 2026,
        "round": "Group stage",
        "neutral": False,
        "source": "fbref.com",
        "source_file": "source-url",
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


def test_normalize_libertadores_matches_renames_only_in_memory():
    row = _supabase_row("2026-04-08", "Flamengo", "Palmeiras", 2, 1)

    matches = normalize_libertadores_matches([row])

    assert "match_date" in row
    assert "tournament" in row
    assert "date" not in row
    assert "competition" not in row
    assert list(matches["competition"]) == ["Libertadores"]
    assert list(matches["result"]) == ["H"]
    assert list(matches["country"]) == ["South America"]


def test_fetch_libertadores_matches_paginates_all_rows():
    rows = [
        _supabase_row("2026-04-01", "A", "B", 1, 0),
        _supabase_row("2026-04-02", "C", "D", 0, 0),
        _supabase_row("2026-04-03", "E", "F", 0, 2),
    ]
    client = FakeClient(rows)

    matches = fetch_libertadores_matches(client=client, page_size=2)

    assert client.table_name == "libertadores_matches"
    assert client.query.ranges == [(0, 1), (2, 3)]
    assert len(matches) == 3
    assert list(matches["result"]) == ["H", "D", "A"]


def test_club_preprocessor_accepts_matches_dataframe():
    matches_df = pd.DataFrame(
        [
            {
                "date": "2026-04-08",
                "home_team": "Flamengo",
                "away_team": "Palmeiras",
                "home_score": 2,
                "away_score": 1,
                "competition": "Libertadores",
            }
        ]
    )

    preprocessor = ClubMatchDataPreprocessor(matches_df=matches_df)
    preprocessor.load_and_transform_data()

    assert preprocessor.file_path == "supabase:libertadores_matches"
    assert len(preprocessor.matches) == 1
    assert preprocessor.matches.loc[0, "competition"] == "Libertadores"
