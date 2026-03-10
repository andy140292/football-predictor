import importlib
from datetime import datetime, timedelta

import pandas as pd


def _load_predict_match(monkeypatch):
    monkeypatch.setenv("API_ENV", "dev")
    import backend.predict_match as predict_match
    return importlib.reload(predict_match)


def test_build_feature_vector_sets_expected_values(monkeypatch):
    predict_match = _load_predict_match(monkeypatch)

    feature_template = pd.DataFrame(
        columns=[
            "home_team_fifa_rank",
            "away_team_fifa_rank",
            "home_team_fifa_points",
            "away_team_fifa_points",
            "fifa_rank_diff",
            "fifa_points_diff",
            "home_team_Brazil",
            "away_team_Argentina",
            "confed_pair_CONMEBOL_vs_CONMEBOL",
            "home_team_confederation_CONMEBOL",
            "away_team_confederation_CONMEBOL",
            "neutral",
        ]
    )

    ranking = pd.DataFrame(
        [
            {"team": "Brazil", "ranking": 1, "points": 1800, "confederation": "CONMEBOL"},
            {"team": "Argentina", "ranking": 2, "points": 1750, "confederation": "CONMEBOL"},
        ]
    )

    vector = predict_match.build_feature_vector("Brazil", "Argentina", feature_template, ranking)

    assert int(vector.loc[0, "home_team_fifa_rank"]) == 1
    assert int(vector.loc[0, "away_team_fifa_rank"]) == 2
    assert int(vector.loc[0, "fifa_rank_diff"]) == -1
    assert int(vector.loc[0, "home_team_Brazil"]) == 1
    assert int(vector.loc[0, "away_team_Argentina"]) == 1
    assert int(vector.loc[0, "neutral"]) == 0


def test_build_feature_vector_national_resolves_team_aliases(monkeypatch, tmp_path):
    predict_match = _load_predict_match(monkeypatch)

    fifa_path = tmp_path / "fifa_codes.csv"
    pd.DataFrame(
        [
            {"country_name": "DR Congo", "team_code": "COD"},
            {"country_name": "United Arab Emirates", "team_code": "UAE"},
            {"country_name": "Curaçao", "team_code": "CUW"},
            {"country_name": "Brazil", "team_code": "BRA"},
        ]
    ).to_csv(fifa_path, index=False)

    alias_path = tmp_path / "team_code_aliases.csv"
    pd.DataFrame(
        [
            {"alias_name": "Congo DR", "team_code": "COD"},
            {"alias_name": "United Arab Emirates UAE", "team_code": "UAE"},
            {"alias_name": "Curacao", "team_code": "CUW"},
        ]
    ).to_csv(alias_path, index=False)

    monkeypatch.setattr(predict_match, "FIFA_CODES_PATH", str(fifa_path))
    monkeypatch.setattr(predict_match, "TEAM_CODE_ALIASES_PATH", str(alias_path))
    monkeypatch.setattr(predict_match, "_team_name_to_code", None)
    monkeypatch.setattr(predict_match, "_team_alias_to_code", None)

    feature_template = pd.DataFrame(
        columns=[
            "home_team_fifa_rank",
            "away_team_fifa_rank",
            "home_team_fifa_points",
            "away_team_fifa_points",
            "fifa_rank_diff",
            "fifa_points_diff",
            "neutral",
        ]
    )
    ranking = pd.DataFrame(
        [
            {"team": "DR Congo", "ranking": 61, "points": 1392, "confederation": "CAF"},
            {"team": "United Arab Emirates\xa0UAE", "ranking": 63, "points": 1380, "confederation": "AFC"},
            {"team": "Curaçao", "ranking": 91, "points": 1235, "confederation": "CONCACAF"},
            {"team": "Brazil", "ranking": 4, "points": 1816, "confederation": "CONMEBOL"},
        ]
    )

    vector_congo = predict_match.build_feature_vector(
        "Congo DR", "Curacao", feature_template, ranking, mode="national"
    )
    assert int(vector_congo.loc[0, "home_team_fifa_rank"]) == 61
    assert int(vector_congo.loc[0, "away_team_fifa_rank"]) == 91

    vector_uae = predict_match.build_feature_vector(
        "United Arab Emirates", "Brazil", feature_template, ranking, mode="national"
    )
    assert int(vector_uae.loc[0, "home_team_fifa_rank"]) == 63
    assert int(vector_uae.loc[0, "away_team_fifa_rank"]) == 4


def test_lookup_national_ranking_row_resolves_hyphenated_bosnia(monkeypatch, tmp_path):
    predict_match = _load_predict_match(monkeypatch)

    fifa_path = tmp_path / "fifa_codes.csv"
    pd.DataFrame(
        [
            {"country_name": "Bosnia and Herzegovina", "team_code": "BIH"},
        ]
    ).to_csv(fifa_path, index=False)

    alias_path = tmp_path / "team_code_aliases.csv"
    pd.DataFrame(
        [
            {"alias_name": "Bosnia and Herzegovina", "team_code": "BIH"},
            {"alias_name": "Bosnia-Herzegovina", "team_code": "BIH"},
        ]
    ).to_csv(alias_path, index=False)

    monkeypatch.setattr(predict_match, "FIFA_CODES_PATH", str(fifa_path))
    monkeypatch.setattr(predict_match, "TEAM_CODE_ALIASES_PATH", str(alias_path))
    monkeypatch.setattr(predict_match, "_team_name_to_code", None)
    monkeypatch.setattr(predict_match, "_team_alias_to_code", None)

    ranking = pd.DataFrame(
        [
            {"team": "Bosnia-Herzegovina", "ranking": 70, "points": 1351, "confederation": "UEFA"},
        ]
    )

    row = predict_match._lookup_national_ranking_row("Bosnia and Herzegovina", ranking)

    assert row is not None
    assert row["team"] == "Bosnia-Herzegovina"


def test_predict_outcome_uses_models(monkeypatch):
    predict_match = _load_predict_match(monkeypatch)

    class DummyModel:
        def predict_proba(self, _):
            return [[0.1, 0.2, 0.7]]

    def fake_assets():
        X = pd.DataFrame(columns=["fifa_rank_diff", "fifa_points_diff"])
        ranking = pd.DataFrame(
            [
                {"team": "Brazil", "ranking": 1, "points": 1800, "confederation": "CONMEBOL"},
                {"team": "Argentina", "ranking": 2, "points": 1750, "confederation": "CONMEBOL"},
            ]
        )
        return {
            "X": X,
            "fifa_rank": ranking,
            "models": {
                "random_forest": DummyModel(),
                "logistic_regression": DummyModel(),
                "mlp": DummyModel(),
            },
        }

    monkeypatch.setattr(predict_match, "_load_assets", fake_assets)
    monkeypatch.setattr(predict_match, "get_remaining_predictions", lambda *_: 5)
    monkeypatch.setattr(predict_match, "register_prediction", lambda *args, **kwargs: None)
    monkeypatch.setattr(predict_match, "get_unpredicted_future_matches", lambda **_: [])

    results = predict_match.predict_outcome("Brazil", "Argentina", token="dummy")

    assert results["random_forest"]["home_win"] == 0.7
    assert results["logistic_regression"]["draw"] == 0.2
    assert results["mlp"]["away_win"] == 0.1


def test_predict_match_probabilities_offline_uses_models(monkeypatch):
    predict_match = _load_predict_match(monkeypatch)

    class DummyModel:
        def predict_proba(self, _):
            return [[0.2, 0.3, 0.5]]

    def fake_assets():
        X = pd.DataFrame(columns=["fifa_rank_diff", "fifa_points_diff"])
        ranking = pd.DataFrame(
            [
                {"team": "Brazil", "ranking": 1, "points": 1800, "confederation": "CONMEBOL"},
                {"team": "Argentina", "ranking": 2, "points": 1750, "confederation": "CONMEBOL"},
            ]
        )
        return {
            "X": X,
            "fifa_rank": ranking,
            "models": {
                "random_forest": DummyModel(),
                "logistic_regression": DummyModel(),
                "mlp": DummyModel(),
            },
        }

    monkeypatch.setattr(predict_match, "_load_assets", fake_assets)
    result = predict_match.predict_match_probabilities_offline("Brazil", "Argentina", mode="national")

    assert result["random_forest"]["home_win"] == 0.5
    assert result["logistic_regression"]["draw"] == 0.3
    assert result["mlp"]["away_win"] == 0.2


def test_predict_outcome_club_mode_uses_club_assets(monkeypatch):
    predict_match = _load_predict_match(monkeypatch)

    class DummyModel:
        def predict_proba(self, _):
            return [[0.25, 0.5, 0.25]]

    def fake_club_assets():
        X = pd.DataFrame(
            columns=[
                "home_team_Benfica",
                "away_team_Real Madrid",
                "competition_Champions Lg",
                "round_Knockout phase play-offs",
                "head_to_head_goal_diff",
                "h2h_available",
                "home_team_avg_scored",
                "away_team_avg_scored",
                "neutral",
            ]
        )
        X_full = pd.DataFrame(
            [
                {
                    "home_team_Benfica": 1,
                    "away_team_Real Madrid": 1,
                    "head_to_head_goal_diff": 0.3,
                    "h2h_available": 1,
                    "home_team_avg_scored": 1.8,
                    "away_team_avg_scored": 1.4,
                }
            ]
        )
        return {
            "X": X,
            "X_full": X_full,
            "models": {
                "random_forest": DummyModel(),
                "logistic_regression": DummyModel(),
                "mlp": DummyModel(),
            },
        }

    monkeypatch.setattr(predict_match, "_load_club_assets", fake_club_assets)
    monkeypatch.setattr(predict_match, "get_remaining_predictions", lambda *_: 5)
    monkeypatch.setattr(predict_match, "register_prediction", lambda *args, **kwargs: None)
    monkeypatch.setattr(predict_match, "get_unpredicted_future_matches", lambda **_: [])

    results = predict_match.predict_outcome("Benfica", "Real Madrid", token="dummy", mode="club")

    assert results["random_forest"]["home_win"] == 0.25
    assert results["logistic_regression"]["draw"] == 0.5
    assert results["mlp"]["away_win"] == 0.25


def test_predict_outcome_champions_mode_uses_champions_assets(monkeypatch):
    predict_match = _load_predict_match(monkeypatch)

    class DummyModel:
        def predict_proba(self, _):
            return [[0.2, 0.4, 0.4]]

    def fake_champions_assets():
        X = pd.DataFrame(
            columns=[
                "home_pre_elo",
                "away_pre_elo",
                "elo_diff",
                "is_ucl_match",
                "is_knockout_round",
                "neutral",
            ]
        )
        X_full = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "home_team": "Benfica",
                    "away_team": "Real Madrid",
                    "home_score": 1,
                    "away_score": 1,
                }
            ]
        )
        return {
            "X": X,
            "X_full": X_full,
            "club_coefficients": pd.DataFrame(
                [
                    {
                        "team": "Benfica",
                        "display_name": "Benfica",
                        "official_name": "SL Benfica",
                        "overall_club_coefficient": 90.0,
                        "season_club_coefficient": 12.25,
                        "overall_rank": 14,
                        "season_rank": 46,
                    },
                    {
                        "team": "Real Madrid",
                        "display_name": "Real Madrid",
                        "official_name": "Real Madrid C.F.",
                        "overall_club_coefficient": 137.5,
                        "season_club_coefficient": 20.0,
                        "overall_rank": 1,
                        "season_rank": 11,
                    },
                ]
            ),
            "team_states": {},
            "pair_states": {},
            "models": {
                "random_forest": DummyModel(),
                "logistic_regression": DummyModel(),
                "mlp": DummyModel(),
            },
        }

    monkeypatch.setattr(predict_match, "_load_champions_assets", fake_champions_assets)
    monkeypatch.setattr(predict_match, "get_remaining_predictions", lambda *_: 5)
    monkeypatch.setattr(predict_match, "register_prediction", lambda *args, **kwargs: None)
    monkeypatch.setattr(predict_match, "get_unpredicted_future_matches", lambda **_: [])

    results = predict_match.predict_outcome("Benfica", "Real Madrid", token="dummy", mode="champions")

    assert results["random_forest"]["home_win"] == 0.4
    assert results["logistic_regression"]["draw"] == 0.4
    assert results["mlp"]["away_win"] == 0.2


def test_predict_outcome_applies_probability_floor(monkeypatch):
    monkeypatch.setenv("PREDICTION_PROBA_FLOOR", "0.01")
    predict_match = _load_predict_match(monkeypatch)

    class ExtremeModel:
        def predict_proba(self, _):
            return [[0.0, 0.2, 0.8]]

    def fake_assets():
        X = pd.DataFrame(columns=["fifa_rank_diff", "fifa_points_diff"])
        ranking = pd.DataFrame(
            [
                {"team": "Brazil", "ranking": 1, "points": 1800, "confederation": "CONMEBOL"},
                {"team": "Argentina", "ranking": 2, "points": 1750, "confederation": "CONMEBOL"},
            ]
        )
        return {
            "X": X,
            "fifa_rank": ranking,
            "models": {
                "random_forest": ExtremeModel(),
                "logistic_regression": ExtremeModel(),
                "mlp": ExtremeModel(),
            },
        }

    monkeypatch.setattr(predict_match, "_load_assets", fake_assets)
    monkeypatch.setattr(predict_match, "get_remaining_predictions", lambda *_: 5)
    monkeypatch.setattr(predict_match, "register_prediction", lambda *args, **kwargs: None)
    monkeypatch.setattr(predict_match, "get_unpredicted_future_matches", lambda **_: [])

    results = predict_match.predict_outcome("Brazil", "Argentina", token="dummy")
    rf = results["random_forest"]
    total = rf["home_win"] + rf["draw"] + rf["away_win"]

    assert rf["away_win"] >= 0.009
    assert abs(total - 1.0) < 1e-9


def test_predict_outcome_includes_future_matches(monkeypatch):
    predict_match = _load_predict_match(monkeypatch)

    class DummyModel:
        def predict_proba(self, _):
            return [[0.1, 0.2, 0.7]]

    def fake_assets():
        X = pd.DataFrame(columns=["fifa_rank_diff", "fifa_points_diff"])
        ranking = pd.DataFrame(
            [
                {"team": "Brazil", "ranking": 1, "points": 1800, "confederation": "CONMEBOL"},
                {"team": "Argentina", "ranking": 2, "points": 1750, "confederation": "CONMEBOL"},
            ]
        )
        return {
            "X": X,
            "fifa_rank": ranking,
            "models": {
                "random_forest": DummyModel(),
                "logistic_regression": DummyModel(),
                "mlp": DummyModel(),
            },
        }

    monkeypatch.setattr(predict_match, "_load_assets", fake_assets)
    monkeypatch.setattr(predict_match, "get_remaining_predictions", lambda *_: 5)
    monkeypatch.setattr(predict_match, "register_prediction", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        predict_match,
        "get_unpredicted_future_matches",
        lambda **_: [
            {
                "match_id": "11111111-1111-1111-1111-111111111111",
                "home_team": "Brazil",
                "away_team": "Argentina",
                "match_date": "2026-03-01",
            }
        ],
    )

    results = predict_match.predict_outcome("Brazil", "Argentina", token="dummy")

    assert results["future_matches"][0]["match_id"] == "11111111-1111-1111-1111-111111111111"


def test_create_or_get_match_prediction_returns_existing(monkeypatch):
    predict_match = _load_predict_match(monkeypatch)

    future_date = (datetime.utcnow() + timedelta(days=5)).date().isoformat()
    monkeypatch.setattr(
        predict_match,
        "_get_calendar_match_by_id",
        lambda _match_id: {"match_id": _match_id, "match_date": future_date},
    )
    monkeypatch.setattr(
        predict_match,
        "_get_existing_match_prediction",
        lambda *_args, **_kwargs: {
            "prediction_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "match_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "predicted_outcome": "away_win",
            "created_at": "2026-02-18T12:00:00+00:00",
        },
    )

    response = predict_match.create_or_get_match_prediction(
        user_id="11111111-1111-1111-1111-111111111111",
        email="test@example.com",
        match_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        predicted_outcome="home_win",
    )

    assert response["status"] == "exists"
    assert response["prediction"]["predicted_outcome"] == "away_win"


def test_create_or_get_match_prediction_creates_new_row(monkeypatch):
    predict_match = _load_predict_match(monkeypatch)

    future_date = (datetime.utcnow() + timedelta(days=5)).date().isoformat()
    monkeypatch.setattr(
        predict_match,
        "_get_calendar_match_by_id",
        lambda _match_id: {"match_id": _match_id, "match_date": future_date},
    )
    monkeypatch.setattr(predict_match, "_get_existing_match_prediction", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        predict_match,
        "_insert_match_prediction",
        lambda *_args, **_kwargs: {
            "prediction_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "match_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "predicted_outcome": "draw",
            "created_at": "2026-02-18T12:00:00+00:00",
        },
    )

    response = predict_match.create_or_get_match_prediction(
        user_id="11111111-1111-1111-1111-111111111111",
        email="test@example.com",
        match_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        predicted_outcome="draw",
    )

    assert response["status"] == "created"
    assert response["prediction"]["predicted_outcome"] == "draw"


def test_upsert_matches_calendar_batch_validates_and_summarizes(monkeypatch):
    predict_match = _load_predict_match(monkeypatch)

    monkeypatch.setattr(
        predict_match,
        "_calendar_row_exists",
        lambda row: row.get("home_team") == "Brazil" and row.get("away_team") == "Argentina",
    )
    monkeypatch.setattr(predict_match, "_bulk_upsert_calendar_rows", lambda _rows: None)

    response = predict_match.upsert_matches_calendar_batch(
        [
            {"home_team": "Brazil", "away_team": "Argentina", "match_date": "2026-03-01"},
            {"home_team": "Argentina", "away_team": "Brazil", "match_date": "2026-06-01"},
            {"home_team": "", "away_team": "Brazil", "match_date": "2026-06-01"},
            {"home_team": "Argentina", "away_team": "Brazil", "match_date": "2026-06-01"},
        ]
    )

    assert response["received"] == 4
    assert response["updated"] == 1
    assert response["inserted"] == 1
    assert response["skipped"] == 2
    assert len(response["errors"]) == 2


def test_get_unpredicted_future_matches_filters_in_dev(monkeypatch):
    predict_match = _load_predict_match(monkeypatch)

    monkeypatch.setattr(
        predict_match,
        "_fetch_future_matches_for_pair",
        lambda *_args, **_kwargs: [
            {
                "match_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "home_team": "Brazil",
                "away_team": "Argentina",
                "match_date": "2030-03-01",
            },
            {
                "match_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "home_team": "Argentina",
                "away_team": "Brazil",
                "match_date": "2030-06-01",
            },
        ],
    )
    monkeypatch.setattr(
        predict_match,
        "_get_user_predicted_match_ids",
        lambda _user_id: {"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"},
    )

    matches = predict_match.get_unpredicted_future_matches(
        home_team="Brazil",
        away_team="Argentina",
        user_id="dev-user",
    )

    assert len(matches) == 1
    assert matches[0]["match_id"] == "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def test_get_unpredicted_future_matches_champions_uses_alias_variants(monkeypatch, tmp_path):
    predict_match = _load_predict_match(monkeypatch)

    alias_path = tmp_path / "club_team_aliases.csv"
    pd.DataFrame(
        [
            {"alias": "PSG", "canonical": "Paris"},
            {"alias": "Paris Saint-Germain", "canonical": "Paris"},
        ]
    ).to_csv(alias_path, index=False)

    monkeypatch.setattr(predict_match, "CLUB_TEAM_ALIASES_PATH", alias_path)
    monkeypatch.setattr(predict_match, "_club_team_aliases", None)
    monkeypatch.setattr(predict_match, "_club_canonical_to_aliases", None)
    monkeypatch.setattr(predict_match, "_resolve_team_code", lambda _name: "")

    query_calls = []

    def _fake_query(*, home_col, away_col, home_value, away_value, today_iso, token=None):
        query_calls.append((home_col, away_col, home_value, away_value))
        if home_value == "PSG" and away_value == "Chelsea":
            return [
                {
                    "match_id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
                    "home_team": "PSG",
                    "away_team": "Chelsea",
                    "match_date": "2030-03-11",
                }
            ]
        return []

    monkeypatch.setattr(predict_match, "_query_future_matches_pair", _fake_query)

    matches = predict_match.get_unpredicted_future_matches(
        home_team="Paris Saint-Germain",
        away_team="Chelsea",
        mode="champions",
    )

    assert len(matches) == 1
    assert matches[0]["home_team"] == "PSG"
    assert any(
        home_col == "home_team"
        and away_col == "away_team"
        and home_value == "PSG"
        and away_value == "Chelsea"
        for home_col, away_col, home_value, away_value in query_calls
    )


def test_build_feature_vector_champions_uses_alias_and_country_diffs(monkeypatch, tmp_path):
    predict_match = _load_predict_match(monkeypatch)

    alias_path = tmp_path / "club_team_aliases.csv"
    pd.DataFrame(
        [
            {"alias": "Bayern Munich", "canonical": "FC Bayern München"},
            {"alias": "Real Madrid CF", "canonical": "Real Madrid"},
        ]
    ).to_csv(alias_path, index=False)

    monkeypatch.setattr(predict_match, "CLUB_TEAM_ALIASES_PATH", alias_path)
    monkeypatch.setattr(predict_match, "_club_team_aliases", None)

    feature_template = pd.DataFrame(
        columns=[
            "home_team_uefa_overall_coefficient",
            "away_team_uefa_overall_coefficient",
            "uefa_overall_coefficient_diff",
            "home_overall_country_coefficient",
            "away_overall_country_coefficient",
            "country_overall_coefficient_diff",
            "overall_country_coefficient",
            "country_uefa_missing",
            "home_pre_elo",
            "away_pre_elo",
        ]
    )
    ranking = pd.DataFrame(
        [
            {
                "team": "Real Madrid",
                "display_name": "Real Madrid",
                "official_name": "Real Madrid C.F.",
                "overall_club_coefficient": 140.0,
                "season_club_coefficient": 20.0,
                "overall_rank": 1,
                "season_rank": 3,
                "overall_country_coefficient": 105.0,
                "season_country_coefficient": 17.0,
                "country_uefa_overall_rank": 1,
                "country_uefa_season_rank": 1,
            },
            {
                "team": "Bayern Munich",
                "display_name": "Bayern Munich",
                "official_name": "FC Bayern München",
                "overall_club_coefficient": 130.0,
                "season_club_coefficient": 18.0,
                "overall_rank": 2,
                "season_rank": 4,
                "overall_country_coefficient": 96.0,
                "season_country_coefficient": 14.0,
                "country_uefa_overall_rank": 3,
                "country_uefa_season_rank": 3,
            },
        ]
    )
    history = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "home_team": "Real Madrid",
                "away_team": "Bayern Munich",
                "home_score": 1,
                "away_score": 0,
            }
        ]
    )

    vector = predict_match.build_feature_vector(
        home_team="Real Madrid CF",
        away_team="Bayern Munich",
        feature_template_df=feature_template,
        ranking_df=ranking,
        mode="champions",
        history_df=history,
    )

    assert float(vector.loc[0, "home_team_uefa_overall_coefficient"]) == 140.0
    assert float(vector.loc[0, "away_team_uefa_overall_coefficient"]) == 130.0
    assert float(vector.loc[0, "uefa_overall_coefficient_diff"]) == 10.0
    assert float(vector.loc[0, "home_overall_country_coefficient"]) == 105.0
    assert float(vector.loc[0, "away_overall_country_coefficient"]) == 96.0
    assert float(vector.loc[0, "country_overall_coefficient_diff"]) == 9.0
    assert float(vector.loc[0, "overall_country_coefficient"]) == 105.0
    assert float(vector.loc[0, "country_uefa_missing"]) == 0.0
