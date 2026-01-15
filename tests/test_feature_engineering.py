import pandas as pd

from src.prediction.feature_engineering import FeatureEngineer


def test_generate_features_adds_expected_columns():
    matches = pd.DataFrame(
        [
            {
                "home_team": "Brazil",
                "away_team": "Argentina",
                "home_score": 2,
                "away_score": 1,
                "home_team_confederation": "CONMEBOL",
                "away_team_confederation": "CONMEBOL",
                "home_team_fifa_rank": 1,
                "away_team_fifa_rank": 2,
                "home_team_fifa_points": 1800,
                "away_team_fifa_points": 1750,
            },
            {
                "home_team": "Argentina",
                "away_team": "Brazil",
                "home_score": 0,
                "away_score": 0,
                "home_team_confederation": "CONMEBOL",
                "away_team_confederation": "CONMEBOL",
                "home_team_fifa_rank": 2,
                "away_team_fifa_rank": 1,
                "home_team_fifa_points": 1750,
                "away_team_fifa_points": 1800,
            },
        ]
    )

    engineered = FeatureEngineer(matches).generate_features()

    expected_columns = [
        "goal_diff",
        "matchup_id",
        "head_to_head_goal_diff",
        "same_confederation",
        "h2h_available",
        "home_team_avg_scored",
        "away_team_avg_conceded",
        "fifa_rank_diff",
        "fifa_points_diff",
    ]
    for col in expected_columns:
        assert col in engineered.columns

    assert "home_team_Brazil" in engineered.columns
    assert "away_team_Argentina" in engineered.columns
