import importlib

import pandas as pd


def _load_predict_match(monkeypatch):
    monkeypatch.setenv("API_ENV", "dev")
    import src.predict_match as predict_match
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

    results = predict_match.predict_outcome("Brazil", "Argentina", token="dummy")

    assert results["random_forest"]["home_win"] == 0.7
    assert results["logistic_regression"]["draw"] == 0.2
    assert results["mlp"]["away_win"] == 0.1
