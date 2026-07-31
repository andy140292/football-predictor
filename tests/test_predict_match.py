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
    assert int(vector.loc[0, "neutral"]) == 1

    neutral_vector = predict_match.build_feature_vector(
        "Brazil",
        "Argentina",
        feature_template,
        ranking,
        neutral=1,
    )

    assert int(neutral_vector.loc[0, "neutral"]) == 1

    non_neutral_vector = predict_match.build_feature_vector(
        "Brazil",
        "Argentina",
        feature_template,
        ranking,
        neutral=0,
    )

    assert int(non_neutral_vector.loc[0, "neutral"]) == 0


def test_build_feature_vector_defaults_world_cup_hosts_to_non_neutral(monkeypatch, tmp_path):
    predict_match = _load_predict_match(monkeypatch)

    fifa_path = tmp_path / "fifa_codes.csv"
    pd.DataFrame(
        [
            {"country_name": "United States", "team_code": "USA"},
            {"country_name": "Mexico", "team_code": "MEX"},
            {"country_name": "Canada", "team_code": "CAN"},
            {"country_name": "Brazil", "team_code": "BRA"},
        ]
    ).to_csv(fifa_path, index=False)
    alias_path = tmp_path / "team_code_aliases.csv"
    pd.DataFrame(
        [
            {"alias_name": "USA", "team_code": "USA"},
            {"alias_name": "USMNT", "team_code": "USA"},
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
            "fifa_rank_diff",
            "neutral",
        ]
    )
    ranking = pd.DataFrame(
        [
            {"team": "United States", "ranking": 15, "points": 1676, "confederation": "CONCACAF"},
            {"team": "Mexico", "ranking": 14, "points": 1680, "confederation": "CONCACAF"},
            {"team": "Canada", "ranking": 31, "points": 1520, "confederation": "CONCACAF"},
            {"team": "Brazil", "ranking": 4, "points": 1816, "confederation": "CONMEBOL"},
        ]
    )

    for host in ["United States", "Mexico", "Canada", "USA"]:
        vector = predict_match.build_feature_vector(host, "Brazil", feature_template, ranking)
        assert int(vector.loc[0, "neutral"]) == 0

    away_host_vector = predict_match.build_feature_vector("Brazil", "United States", feature_template, ranking)
    assert int(away_host_vector.loc[0, "neutral"]) == 1


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
    captured = {}

    monkeypatch.setattr(
        predict_match,
        "_calendar_row_match_state",
        lambda row: "exact" if row.get("home_team") == "Brazil" and row.get("away_team") == "Argentina" else None,
    )
    monkeypatch.setattr(
        predict_match,
        "_bulk_upsert_calendar_rows",
        lambda rows: captured.setdefault("rows", rows),
    )

    response = predict_match.upsert_matches_calendar_batch(
        [
            {"home_team": "Brazil", "away_team": "Argentina", "match_date": "2026-03-01"},
            {
                "home_team": "Argentina",
                "away_team": "Brazil",
                "match_date": "2026-06-01",
                "tournament": "FIFA World Cup",
            },
            {"home_team": "", "away_team": "Brazil", "match_date": "2026-06-01"},
            {"home_team": "Argentina", "away_team": "Brazil", "match_date": "2026-06-01"},
        ]
    )

    assert response["received"] == 4
    assert response["updated"] == 1
    assert response["inserted"] == 1
    assert response["skipped"] == 2
    assert len(response["errors"]) == 2
    assert captured["rows"] == [
        {"home_team": "Brazil", "away_team": "Argentina", "match_date": "2026-03-01"},
        {
            "home_team": "Argentina",
            "away_team": "Brazil",
            "match_date": "2026-06-01",
            "tournament": "FIFA World Cup",
        },
    ]


def test_upsert_matches_calendar_batch_skips_reversed_existing_fixture(monkeypatch):
    predict_match = _load_predict_match(monkeypatch)
    captured = {}

    monkeypatch.setattr(
        predict_match,
        "_calendar_row_match_state",
        lambda row: "reverse" if row.get("home_team") == "Jamaica" and row.get("away_team") == "DR Congo" else None,
    )
    monkeypatch.setattr(
        predict_match,
        "_bulk_upsert_calendar_rows",
        lambda rows: captured.setdefault("rows", rows),
    )

    response = predict_match.upsert_matches_calendar_batch(
        [
            {"home_team": "Jamaica", "away_team": "DR Congo", "match_date": "2026-03-31"},
            {
                "home_team": "Brazil",
                "away_team": "Argentina",
                "match_date": "2026-03-31",
                "tournament": "FIFA World Cup",
            },
        ]
    )

    assert response["received"] == 2
    assert response["inserted"] == 1
    assert response["updated"] == 0
    assert response["skipped"] == 1
    assert response["errors"] == [
        {"row_index": -1, "reason": "fixture already exists on this date with reversed home/away teams"}
    ]
    assert captured["rows"] == [
        {
            "home_team": "Brazil",
            "away_team": "Argentina",
            "match_date": "2026-03-31",
            "tournament": "FIFA World Cup",
        }
    ]


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


def test_list_matches_calendar_merges_user_predictions(monkeypatch):
    predict_match = _load_predict_match(monkeypatch)

    monkeypatch.setattr(
        predict_match,
        "_fetch_future_matches_for_mode",
        lambda *_args, **_kwargs: [
            {
                "match_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "home_team": "Ecuador",
                "away_team": "Germany",
                "match_date": "2026-06-25",
                "tournament": "FIFA World Cup",
            },
            {
                "match_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "home_team": "Curacao",
                "away_team": "Ivory Coast",
                "match_date": "2026-06-25",
                "tournament": "FIFA World Cup",
            },
        ],
    )
    monkeypatch.setattr(
        predict_match,
        "_get_user_match_predictions_by_match_id",
        lambda *_args, **_kwargs: {
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa": {
                "prediction_id": "11111111-1111-1111-1111-111111111111",
                "match_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "predicted_outcome": "home_win",
                "created_at": "2026-02-18T12:00:00+00:00",
            }
        },
    )

    response = predict_match.list_matches_calendar(mode="world_cup", user_id="dev-user")

    assert response["mode"] == "world_cup"
    assert len(response["matches"]) == 2
    assert response["matches"][0]["predicted_outcome"] == "home_win"
    assert response["matches"][0]["prediction_id"] == "11111111-1111-1111-1111-111111111111"
    assert response["matches"][1]["predicted_outcome"] is None


def test_list_matches_calendar_accepts_libertadores(monkeypatch):
    predict_match = _load_predict_match(monkeypatch)
    captured = {}

    def fake_matches(mode, token=None):
        captured.update({"mode": mode, "token": token})
        return [
            {
                "match_id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
                "home_team": "Cruzeiro",
                "away_team": "Flamengo",
                "match_date": "2026-08-12",
                "tournament": "Libertadores",
            }
        ]

    monkeypatch.setattr(predict_match, "_fetch_future_matches_for_mode", fake_matches)

    response = predict_match.list_matches_calendar(
        mode="libertadores",
        token="user-token",
    )

    assert response["mode"] == "libertadores"
    assert response["matches"][0]["tournament"] == "Libertadores"
    assert captured == {"mode": "libertadores", "token": "user-token"}


def test_list_matches_calendar_rejects_unknown_mode(monkeypatch):
    predict_match = _load_predict_match(monkeypatch)

    try:
        predict_match.list_matches_calendar(mode="champions")
    except ValueError as exc:
        assert str(exc) == "mode debe ser world_cup o libertadores"
    else:
        raise AssertionError("Expected an invalid mode error")


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


def test_get_unpredicted_future_matches_national_uses_alias_variants(monkeypatch, tmp_path):
    predict_match = _load_predict_match(monkeypatch)

    fifa_codes_path = tmp_path / "fifa_country_codes.csv"
    team_aliases_path = tmp_path / "team_code_aliases.csv"
    pd.DataFrame(
        [
            {"country_name": "DR Congo", "team_code": "COD"},
            {"country_name": "Uzbekistan", "team_code": "UZB"},
        ]
    ).to_csv(fifa_codes_path, index=False)
    pd.DataFrame(
        [
            {"alias_name": "RD del Congo", "team_code": "COD"},
            {"alias_name": "Congo DR", "team_code": "COD"},
            {"alias_name": "RD Congo", "team_code": "COD"},
            {"alias_name": "Uzbekistán", "team_code": "UZB"},
        ]
    ).to_csv(team_aliases_path, index=False)

    monkeypatch.setattr(predict_match, "FIFA_CODES_PATH", str(fifa_codes_path))
    monkeypatch.setattr(predict_match, "TEAM_CODE_ALIASES_PATH", str(team_aliases_path))
    monkeypatch.setattr(predict_match, "_team_name_to_code", None)
    monkeypatch.setattr(predict_match, "_team_alias_to_code", None)
    monkeypatch.setattr(predict_match, "_team_code_to_names", None)

    query_calls = []

    def _fake_query(*, home_col, away_col, home_value, away_value, today_iso, token=None):
        query_calls.append((home_col, away_col, home_value, away_value))
        if home_col == "home_team" and away_col == "away_team" and home_value == "Congo DR" and away_value == "Uzbekistan":
            return [
                {
                    "match_id": "ffffffff-ffff-ffff-ffff-ffffffffffff",
                    "home_team": "Congo DR",
                    "away_team": "Uzbekistan",
                    "match_date": "2030-06-27",
                }
            ]
        return []

    monkeypatch.setattr(predict_match, "_query_future_matches_pair", _fake_query)

    matches = predict_match.get_unpredicted_future_matches(
        home_team="RD del Congo",
        away_team="Uzbekistan",
        mode="national",
    )

    assert "RD Congo" in predict_match._calendar_team_query_values("RD del Congo", "national")
    assert matches == [
        {
            "match_id": "ffffffff-ffff-ffff-ffff-ffffffffffff",
            "home_team": "Congo DR",
            "away_team": "Uzbekistan",
            "match_date": "2030-06-27",
        }
    ]
    assert ("home_team", "away_team", "Congo DR", "Uzbekistan") in query_calls
    assert ("home_team_code", "away_team_code", "COD", "UZB") in query_calls


def test_get_unpredicted_future_matches_libertadores_uses_pair_feed(monkeypatch):
    predict_match = _load_predict_match(monkeypatch)

    called = {"mode_feed": 0, "pair_feed": 0}

    def _fake_mode_feed(mode, token=None):
        called["mode_feed"] += 1
        assert mode == "libertadores"
        return [
            {
                "match_id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
                "home_team": "Independiente Medellín",
                "away_team": "Estudiantes-LP",
                "match_date": "2030-04-08",
            },
            {
                "match_id": "dddddddd-dddd-dddd-dddd-dddddddddddd",
                "home_team": "Cusco",
                "away_team": "Flamengo",
                "match_date": "2030-04-08",
            },
        ]

    def _fake_pair_feed(*_args, **_kwargs):
        called["pair_feed"] += 1
        return [
            {
                "match_id": "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
                "home_team": "Boca Juniors",
                "away_team": "Flamengo",
                "match_date": "2030-04-09",
            }
        ]

    monkeypatch.setattr(predict_match, "_fetch_future_matches_for_mode", _fake_mode_feed)
    monkeypatch.setattr(predict_match, "_fetch_future_matches_for_pair", _fake_pair_feed)

    matches = predict_match.get_unpredicted_future_matches(
        home_team="Boca Juniors",
        away_team="Flamengo",
        mode="libertadores",
    )

    assert called == {"mode_feed": 0, "pair_feed": 1}
    assert len(matches) == 1
    assert matches[0]["match_id"] == "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"


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
