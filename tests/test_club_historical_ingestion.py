from pathlib import Path

import pandas as pd

from src.data.build_club_historical_dataset import (
    load_ucl_canonical_or_legacy,
    merge_club_datasets,
)
from src.data.ingest_football_data_leagues import ingest_football_data_leagues


def test_ingest_leagues_handles_aliases_and_bad_lines(tmp_path):
    root = tmp_path / "European_Leagues_Data_1993_2026"
    season_a = root / "data"
    season_b = root / "data (1)"
    season_a.mkdir(parents=True)
    season_b.mkdir(parents=True)

    # Canonical football-data style
    (season_a / "E0.csv").write_text(
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "E0,15/08/2025,Liverpool,Arsenal,2,1,H\n",
        encoding="utf-8",
    )

    # Legacy alias style + malformed trailing fields to trigger python fallback path.
    (season_b / "G1.csv").write_text(
        "Div,Date,HT,AT,HG,AG,Res\n"
        "G1,17/08/2024,PAOK,AEK,1,1,D\n"
        "G1,18/08/2024,Olympiakos,Panathinaikos,2,0,H,,,,\n",
        encoding="latin-1",
    )

    out_csv = tmp_path / "club_leagues_historical_1993_2026.csv"
    df = ingest_football_data_leagues(root, out_csv)

    assert out_csv.exists()
    assert len(df) >= 2
    assert {"home_team", "away_team", "home_score", "away_score", "competition", "season"}.issubset(df.columns)

    assert ((df["home_team"] == "Liverpool") & (df["away_team"] == "Arsenal")).any()
    assert ((df["home_team"] == "PAOK") & (df["away_team"] == "AEK")).any()


def test_load_legacy_ucl_team_rows_and_merge(tmp_path):
    legacy_ucl = tmp_path / "ucl_legacy.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-02-17",
                "team": "Benfica",
                "opponent": "Real Madrid",
                "venue": "Home",
                "gf": 1,
                "ga": 0,
                "competition": "Champions Lg",
                "round": "Knockout phase play-offs",
            },
            {
                "date": "2026-02-17",
                "team": "Real Madrid",
                "opponent": "Benfica",
                "venue": "Away",
                "gf": 0,
                "ga": 1,
                "competition": "Champions Lg",
                "round": "Knockout phase play-offs",
            },
        ]
    ).to_csv(legacy_ucl, index=False)

    ucl_df = load_ucl_canonical_or_legacy(legacy_ucl)
    assert len(ucl_df) == 1
    assert {"home_team", "away_team", "home_score", "away_score", "competition"}.issubset(ucl_df.columns)

    league_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-02-10"),
                "home_team": "Benfica",
                "away_team": "Porto",
                "home_score": 2,
                "away_score": 1,
                "result": "H",
                "competition": "Primeira Liga",
                "country": "Portugal",
                "season": "2025-2026",
                "round": "Regular season",
                "neutral": 0,
                "source": "football-data.co.uk",
                "source_file": "dummy",
                "div_code": "P1",
            }
        ]
    )

    merged = merge_club_datasets(league_df=league_df, ucl_df=ucl_df, alias_file=None)
    assert len(merged) == 2
    assert (merged["competition"] == "Champions Lg").any()
    assert (merged["competition"] == "Primeira Liga").any()
