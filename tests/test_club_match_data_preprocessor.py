import pandas as pd

from src.prediction.club_match_data_preprocessor import ClubMatchDataPreprocessor


def test_club_preprocess_transforms_and_builds_features(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / "test_club_matches.csv"

    # Team-centric rows (like FBref Scores & Fixtures) with one mirrored duplicate.
    pd.DataFrame(
        [
            {
                "date": "2026-01-10",
                "team": "Benfica",
                "opponent": "Real Madrid",
                "venue": "Home",
                "competition": "Champions Lg",
                "round": "Knockout phase play-offs",
                "gf": 1,
                "ga": 0,
            },
            {
                "date": "2026-01-10",
                "team": "Real Madrid",
                "opponent": "Benfica",
                "venue": "Away",
                "competition": "Champions Lg",
                "round": "Knockout phase play-offs",
                "gf": 0,
                "ga": 1,
            },
            {
                "date": "2026-01-17",
                "team": "Benfica",
                "opponent": "Porto",
                "venue": "Away",
                "competition": "Primeira Liga",
                "round": "Regular season",
                "gf": 2,
                "ga": 2,
            },
            {
                "date": "2026-01-24",
                "team": "Benfica",
                "opponent": "Real Madrid",
                "venue": "Home",
                "competition": "Champions Lg",
                "round": "Knockout phase play-offs",
                "gf": None,
                "ga": None,
            },
        ]
    ).to_csv(file_path, index=False)

    preprocessor = ClubMatchDataPreprocessor(file_path=str(file_path))
    preprocessor.file_path = str(file_path)

    X, y = preprocessor.preprocess()

    # Completed unique matches: Benfica-Real Madrid and Porto-Benfica
    assert len(X) == 2
    assert len(y) == 2
    assert all(y.isin([0, 1, 2]))

    assert "head_to_head_goal_diff" in X.columns
    assert "home_team_avg_scored" in X.columns
    assert "away_team_avg_scored" in X.columns
    assert "is_ucl_match" in X.columns

    # Club pipeline should not require FIFA-based columns.
    assert "fifa_rank_diff" not in X.columns
