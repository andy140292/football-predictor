import pandas as pd

from src.prediction.match_data_preprocessor import MatchDataPreprocessor


def test_preprocess_builds_features(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / "test_matches_unit.csv"

    pd.DataFrame(
        [
            {
                "date": "2022-01-10",
                "home_score": 2,
                "away_score": 1,
                "tournament": "Friendly",
                "home_team": "Brazil",
                "away_team": "Argentina",
                "home_team_confederation": "CONMEBOL",
                "away_team_confederation": "CONMEBOL",
                "home_team_fifa_rank": 1,
                "away_team_fifa_rank": 2,
                "home_team_fifa_points": 1800,
                "away_team_fifa_points": 1750,
            }
            ,
            {
                "date": "2022-02-10",
                "home_score": 1,
                "away_score": 3,
                "tournament": "Friendly",
                "home_team": "Brazil",
                "away_team": "Argentina",
                "home_team_confederation": "CONMEBOL",
                "away_team_confederation": "CONMEBOL",
                "home_team_fifa_rank": 1,
                "away_team_fifa_rank": 2,
                "home_team_fifa_points": 1800,
                "away_team_fifa_points": 1750,
            },
        ]
    ).to_csv(file_path, index=False)

    preprocessor = MatchDataPreprocessor(file_path=str(file_path), from_supabase=False)
    preprocessor.file_path = str(file_path)

    X, y = preprocessor.preprocess()

    assert len(X) == 1
    assert y.iloc[0] in [0, 1, 2]
    assert "fifa_rank_diff" in X.columns
