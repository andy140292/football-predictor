import pandas as pd

from src.prediction.club_match_data_preprocessor import ClubMatchDataPreprocessor
import src.prediction.club_match_data_preprocessor as club_preprocessor_module


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
    assert "home_form_goals_for_5" in X.columns
    assert "away_form_goals_for_5" in X.columns
    assert "home_pre_elo" in X.columns
    assert "is_ucl_match" in X.columns

    # Club pipeline should not require FIFA-based columns.
    assert "fifa_rank_diff" not in X.columns


def test_club_preprocess_merges_uefa_coefficients_with_aliases_and_country_diffs(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    matches_path = data_dir / "club_matches.csv"
    coeff_path = data_dir / "uefa_club_coeffs.csv"
    country_coeff_path = data_dir / "uefa_country_coeffs.csv"
    aliases_path = data_dir / "club_team_aliases.csv"

    pd.DataFrame(
        [
            {
                "date": "2026-02-10",
                "home_team": "Real Madrid CF",
                "away_team": "Bayern Munich",
                "home_score": 2,
                "away_score": 1,
                "competition": "Champions Lg",
                "round": "Round of 16",
                "neutral": 0,
            }
        ]
    ).to_csv(matches_path, index=False)

    pd.DataFrame(
        [
            {
                "team": "Real Madrid",
                "display_name": "Real Madrid",
                "official_name": "Real Madrid C.F.",
                "country": "Spain",
                "overall_club_coefficient": 140.0,
                "season_club_coefficient": 25.0,
                "overall_rank": 1,
                "season_rank": 2,
                "uefa_season_year": 2026,
            },
            {
                "team": "Bayern Munich",
                "display_name": "Bayern Munich",
                "official_name": "FC Bayern München",
                "country": "Germany",
                "overall_club_coefficient": 130.0,
                "season_club_coefficient": 21.0,
                "overall_rank": 3,
                "season_rank": 4,
                "uefa_season_year": 2026,
            },
        ]
    ).to_csv(coeff_path, index=False)

    pd.DataFrame(
        [
            {
                "country": "Spain",
                "overall_country_coefficient": 105.0,
                "season_country_coefficient": 17.0,
                "overall_rank": 1,
                "season_rank": 1,
                "uefa_season_year": 2026,
            },
            {
                "country": "Germany",
                "overall_country_coefficient": 96.0,
                "season_country_coefficient": 14.0,
                "overall_rank": 3,
                "season_rank": 3,
                "uefa_season_year": 2026,
            },
        ]
    ).to_csv(country_coeff_path, index=False)

    pd.DataFrame(
        [
            {"alias": "Real Madrid CF", "canonical": "Real Madrid"},
            {"alias": "Bayern Munich", "canonical": "FC Bayern München"},
        ]
    ).to_csv(aliases_path, index=False)

    monkeypatch.setattr(club_preprocessor_module, "CLUB_COEFFICIENTS_HISTORY_PATH", coeff_path)
    monkeypatch.setattr(club_preprocessor_module, "CLUB_COEFFICIENTS_PATH", coeff_path)
    monkeypatch.setattr(club_preprocessor_module, "COUNTRY_COEFFICIENTS_HISTORY_PATH", country_coeff_path)
    monkeypatch.setattr(club_preprocessor_module, "CLUB_TEAM_ALIASES_PATH", aliases_path)

    preprocessor = ClubMatchDataPreprocessor(
        file_path=str(matches_path),
        include_uefa_coefficients=True,
    )
    X, y = preprocessor.preprocess()

    assert len(X) == 1
    assert y.iloc[0] == 2
    assert float(X.loc[0, "home_team_uefa_overall_coefficient"]) == 140.0
    assert float(X.loc[0, "away_team_uefa_overall_coefficient"]) == 130.0
    assert float(X.loc[0, "home_overall_country_coefficient"]) == 105.0
    assert float(X.loc[0, "away_overall_country_coefficient"]) == 96.0
    assert float(X.loc[0, "country_overall_coefficient_diff"]) == 9.0
    assert float(X.loc[0, "overall_country_coefficient"]) == 105.0
