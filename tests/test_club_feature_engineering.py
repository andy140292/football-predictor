import pandas as pd

from src.prediction.club_feature_engineering import ClubFeatureEngineer


def test_generate_features_adds_expected_club_columns():
    matches = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "home_team": "Benfica",
                "away_team": "Real Madrid",
                "home_score": 1,
                "away_score": 0,
                "competition": "Champions Lg",
                "round": "Knockout phase play-offs",
                "neutral": 0,
            },
            {
                "date": "2026-01-08",
                "home_team": "Real Madrid",
                "away_team": "Benfica",
                "home_score": 2,
                "away_score": 2,
                "competition": "Champions Lg",
                "round": "Knockout phase play-offs",
                "neutral": 0,
            },
        ]
    )

    engineered = ClubFeatureEngineer(matches).generate_features()

    expected_columns = [
        "goal_diff",
        "matchup_id",
        "head_to_head_goal_diff",
        "h2h_available",
        "is_ucl_match",
        "is_knockout_playoff",
        "home_team_avg_scored",
        "away_team_avg_conceded",
    ]
    for col in expected_columns:
        assert col in engineered.columns

    assert "home_team_Benfica" in engineered.columns
    assert "away_team_Benfica" in engineered.columns
