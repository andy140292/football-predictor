import csv
from pathlib import Path

from src.utils.extract_pre_world_cup_champion_matches import (
    CHAMPION_YEARS,
    DEFAULT_MATCHES_PATH,
    extract_matches,
    load_rankings_by_date,
    load_team_code_lookup,
    read_csv_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATCHES_PATH = DEFAULT_MATCHES_PATH
RANKINGS_PATH = Path("/Users/andresuribe/Downloads/archive (9)/fifa_ranking-2024-06-20.csv")
ALIAS_PATH = PROJECT_ROOT / "data" / "team_code_aliases.csv"
COUNTRY_CODES_PATH = PROJECT_ROOT / "data" / "fifa_country_codes.csv"


def build_extracted_rows():
    _, matches_rows = read_csv_rows(MATCHES_PATH)
    rankings_by_date = load_rankings_by_date(RANKINGS_PATH)
    team_code_lookup = load_team_code_lookup(ALIAS_PATH, COUNTRY_CODES_PATH)
    return extract_matches(matches_rows, rankings_by_date, team_code_lookup)


def test_extract_matches_returns_expected_row_count():
    extracted_rows = build_extracted_rows()

    assert len(extracted_rows) == 30
    for champion_team, world_cup_year in CHAMPION_YEARS:
        champion_rows = [
            row
            for row in extracted_rows
            if row["champion_team"] == champion_team and row["world_cup_year"] == str(world_cup_year)
        ]
        assert len(champion_rows) == 5


def test_selected_rank_dates_match_expected_values():
    extracted_rows = build_extracted_rows()
    expected_dates = {
        ("Brazil", "2002"): "2002-05-15",
        ("Italy", "2006"): "2006-05-17",
        ("Spain", "2010"): "2010-05-26",
        ("Germany", "2014"): "2014-06-05",
        ("France", "2018"): "2018-06-07",
        ("Argentina", "2022"): "2022-10-06",
    }

    for key, expected_date in expected_dates.items():
        champion_rows = [
            row for row in extracted_rows if (row["champion_team"], row["world_cup_year"]) == key
        ]
        assert {row["selected_rank_date"] for row in champion_rows} == {expected_date}


def test_selected_opponents_match_expected_sequence():
    extracted_rows = build_extracted_rows()
    expected_opponents = {
        ("Brazil", "2002"): ["Iceland", "Serbia", "Portugal", "Catalonia", "Malaysia"],
        ("Italy", "2006"): ["Netherlands", "Ivory Coast", "Germany", "Switzerland", "Ukraine"],
        ("Spain", "2010"): ["Austria", "France", "Saudi Arabia", "South Korea", "Poland"],
        ("Germany", "2014"): ["England", "Chile", "Poland", "Cameroon", "Armenia"],
        ("France", "2018"): ["Colombia", "Russian Federation", "Republic of Ireland", "Italy", "United States"],
        ("Argentina", "2022"): ["Italy", "Estonia", "Honduras", "Jamaica", "United Arab Emirates"],
    }

    for key, expected in expected_opponents.items():
        champion_rows = [
            row for row in extracted_rows if (row["champion_team"], row["world_cup_year"]) == key
        ]
        opponents = [
            row["away_team"] if row["home_team"] == row["champion_team"] else row["home_team"]
            for row in champion_rows
        ]
        assert opponents == expected


def test_missing_opponent_rank_is_left_blank_for_catalonia():
    extracted_rows = build_extracted_rows()
    brazil_rows = [
        row for row in extracted_rows if row["champion_team"] == "Brazil" and row["world_cup_year"] == "2002"
    ]
    catalonia_row = next(
        row
        for row in brazil_rows
        if (row["away_team"] if row["home_team"] == "Brazil" else row["home_team"]) == "Catalonia"
    )

    assert catalonia_row["opponent_rank_pre_wc"] == ""
    assert catalonia_row["champion_rank_pre_wc"] == "2"
