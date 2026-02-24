import pandas as pd

from backend.club_feature_engineering import ClubFeatureEngineer


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
        "is_knockout_round",
        "home_pre_elo",
        "away_pre_elo",
        "elo_diff",
        "home_form_goals_for_5",
        "away_form_goals_against_5",
        "home_form_win_rate_5",
        "away_form_loss_rate_10",
        "home_form_btts_rate_10",
        "home_rest_days",
        "away_rest_days",
    ]
    for col in expected_columns:
        assert col in engineered.columns

    # Leakage-safe H2H should only be available from the second matchup onward.
    assert engineered.loc[0, "h2h_available"] == 0
    assert engineered.loc[1, "h2h_available"] == 1
