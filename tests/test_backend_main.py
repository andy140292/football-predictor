import importlib

from fastapi.testclient import TestClient


def _load_main(monkeypatch):
    monkeypatch.setenv("API_ENV", "dev")
    import backend.main as main
    return importlib.reload(main)


def test_healthz_returns_ok(monkeypatch):
    main = _load_main(monkeypatch)

    client = TestClient(main.app)
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_uses_predict_outcome(monkeypatch):
    main = _load_main(monkeypatch)

    monkeypatch.setattr(main, "predict_outcome", lambda **_: {"ok": True})

    client = TestClient(main.app)
    response = client.post("/predict", json={"home_team": "Brazil", "away_team": "Argentina"})

    assert response.status_code == 200
    assert response.json() == {"predicción": {"ok": True}}


def test_team_vs_confed_endpoint_uses_service(monkeypatch):
    main = _load_main(monkeypatch)

    payload = {
        "team": "Brazil",
        "opponent_confederation": "UEFA",
        "matches_count": 5,
        "wins": 3,
        "draws": 1,
        "losses": 1,
        "goals_for": 9,
        "goals_against": 5,
    }
    monkeypatch.setattr(main, "get_team_vs_confed", lambda **_: payload)

    client = TestClient(main.app)
    response = client.post(
        "/team-vs-confed",
        json={"team": "Brazil", "opponent_confederation": "UEFA"},
    )

    assert response.status_code == 200
    assert response.json() == payload


def test_match_predictions_endpoint_uses_service(monkeypatch):
    main = _load_main(monkeypatch)

    service_response = {
        "status": "created",
        "prediction": {
            "prediction_id": "11111111-1111-1111-1111-111111111111",
            "match_id": "22222222-2222-2222-2222-222222222222",
            "predicted_outcome": "home_win",
            "created_at": "2026-02-18T12:00:00+00:00",
        },
    }
    monkeypatch.setattr(main, "create_or_get_match_prediction", lambda **_: service_response)

    client = TestClient(main.app)
    response = client.post(
        "/match-predictions",
        json={
            "match_id": "22222222-2222-2222-2222-222222222222",
            "predicted_outcome": "home_win",
        },
    )

    assert response.status_code == 200
    assert response.json() == service_response


def test_admin_calendar_upsert_requires_valid_admin_key(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "super-secret")
    main = _load_main(monkeypatch)

    client = TestClient(main.app)
    response = client.post(
        "/admin/matches-calendar/upsert-batch",
        json={"matches": [{"home_team": "Brazil", "away_team": "Argentina", "match_date": "2026-03-01"}]},
    )

    assert response.status_code == 401


def test_admin_calendar_upsert_uses_service(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "super-secret")
    main = _load_main(monkeypatch)

    service_response = {
        "received": 2,
        "inserted": 1,
        "updated": 1,
        "skipped": 0,
        "errors": [],
    }
    monkeypatch.setattr(main, "upsert_matches_calendar_batch", lambda *_args, **_kwargs: service_response)

    client = TestClient(main.app)
    response = client.post(
        "/admin/matches-calendar/upsert-batch",
        headers={"X-Admin-Key": "super-secret"},
        json={
            "matches": [
                {"home_team": "Brazil", "away_team": "Argentina", "match_date": "2026-03-01"},
                {"home_team": "Argentina", "away_team": "Brazil", "match_date": "2026-06-01"},
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == service_response


def test_model_scorecard_endpoint_uses_service(monkeypatch):
    main = _load_main(monkeypatch)

    service_response = {
        "mode": "national",
        "model_version": "2026_01_national_v1",
        "from_date": "2025-09-01",
        "to_date": "2026-01-31",
        "correct_count": 210,
        "incorrect_count": 120,
        "total_scored": 330,
        "accuracy_pct": 63.6364,
    }
    monkeypatch.setattr(main, "get_model_scorecard", lambda **_: service_response)

    client = TestClient(main.app)
    response = client.get(
        "/model-scorecard",
        params={
            "mode": "national",
            "model_version": "2026_01_national_v1",
            "from_date": "2025-09-01",
            "to_date": "2026-01-31",
        },
    )

    assert response.status_code == 200
    assert response.json() == service_response


def test_model_scorecard_matches_endpoint_uses_service(monkeypatch):
    main = _load_main(monkeypatch)

    service_response = {
        "mode": "national",
        "model_version": "2026_01_national_v1",
        "from_date": "2025-09-01",
        "to_date": "2026-01-31",
        "verdict": "all",
        "page": 1,
        "page_size": 2,
        "total": 2,
        "matches": [
            {
                "match_id": "22222222-2222-2222-2222-222222222222",
                "match_date": "2026-01-10",
                "home_team": "Brazil",
                "away_team": "Argentina",
                "tournament": "FIFA World Cup qualification",
                "actual_outcome": "home_win",
                "consensus_predicted_outcome": "home_win",
                "consensus_prob_home_win": 0.6,
                "consensus_prob_draw": 0.2,
                "consensus_prob_away_win": 0.2,
                "is_correct": True,
            },
            {
                "match_id": "33333333-3333-3333-3333-333333333333",
                "match_date": "2026-01-11",
                "home_team": "Chile",
                "away_team": "Peru",
                "tournament": "Friendly",
                "actual_outcome": "draw",
                "consensus_predicted_outcome": "away_win",
                "consensus_prob_home_win": 0.3,
                "consensus_prob_draw": 0.29,
                "consensus_prob_away_win": 0.41,
                "is_correct": False,
            },
        ],
    }
    monkeypatch.setattr(main, "list_model_scorecard_matches", lambda **_: service_response)

    client = TestClient(main.app)
    response = client.get(
        "/model-scorecard/matches",
        params={
            "mode": "national",
            "model_version": "2026_01_national_v1",
            "from_date": "2025-09-01",
            "to_date": "2026-01-31",
            "page": 1,
            "page_size": 2,
            "verdict": "all",
        },
    )

    assert response.status_code == 200
    assert response.json() == service_response
