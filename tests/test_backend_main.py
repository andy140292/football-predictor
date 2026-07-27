import importlib
import os
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient


def test_backend_imports_with_src_as_application_root(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(project_root / "src"),
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_KEY": "test-key",
            "SUPABASE_SERVICE_ROLE_KEY": "test-service-key",
            "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
            "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache"),
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import backend.main; import backend.prediction.national_match_source",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def _load_main(monkeypatch):
    monkeypatch.setenv("API_ENV", "dev")
    import backend.main as main
    return importlib.reload(main)


def _load_main_prod(monkeypatch):
    monkeypatch.setenv("API_ENV", "prod")
    import backend.main as main
    return importlib.reload(main)


def test_application_route_inventory(monkeypatch):
    main = _load_main(monkeypatch)

    documented_routes = {
        (method, route.path)
        for route in main.app.routes
        if route.path not in {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
        for method in route.methods
        if method not in {"HEAD", "OPTIONS"}
    }

    assert documented_routes == {
        ("GET", "/healthz"),
        ("POST", "/predict"),
        ("POST", "/match-predictions"),
        ("GET", "/matches-calendar"),
        ("POST", "/admin/matches-calendar/upsert-batch"),
        ("POST", "/admin/matches/upsert-batch"),
        ("GET", "/model-scorecard"),
        ("GET", "/model-scorecard/matches"),
        ("GET", "/top-searched-teams"),
        ("GET", "/prediction-rankings"),
        ("POST", "/recent-form"),
        ("POST", "/head-to-head"),
        ("POST", "/team-vs-confed"),
    }


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


def test_head_to_head_endpoint_uses_service(monkeypatch):
    main = _load_main(monkeypatch)

    payload = {
        "matches": [
            {
                "date": "2026-03-01",
                "home_team": "Brazil",
                "away_team": "Argentina",
                "home_score": 1,
                "away_score": 0,
            }
        ],
        "home_form": {"team": "Brazil", "wins": 1, "draws": 0, "losses": 0, "goals": 1},
        "away_form": {"team": "Argentina", "wins": 0, "draws": 0, "losses": 1, "goals": 0},
    }
    monkeypatch.setattr(main, "get_head_to_head", lambda **_: payload)

    client = TestClient(main.app)
    response = client.post(
        "/head-to-head",
        json={
            "home_team": "Brazil",
            "away_team": "Argentina",
            "tournaments": ["Friendly"],
        },
    )

    assert response.status_code == 200
    assert response.json() == payload


def test_recent_form_endpoint_uses_service(monkeypatch):
    main = _load_main(monkeypatch)

    payload = {
        "home_team": "Bolivia",
        "away_team": "Jamaica",
        "home_matches": [
            {"date": "2026-03-26", "home_team": "Bolivia", "away_team": "Suriname", "home_score": 2, "away_score": 1},
            {"date": "2026-03-15", "home_team": "Bolivia", "away_team": "Trinidad and Tobago", "home_score": 3, "away_score": 0},
        ],
        "away_matches": [
            {"date": "2026-03-27", "home_team": "New Caledonia", "away_team": "Jamaica", "home_score": 0, "away_score": 1},
            {"date": "2026-01-18", "home_team": "Grenada", "away_team": "Jamaica", "home_score": 0, "away_score": 1},
        ],
    }
    monkeypatch.setattr(main, "get_recent_matches", lambda **_: payload)

    client = TestClient(main.app)
    response = client.post(
        "/recent-form",
        json={"home_team": "Bolivia", "away_team": "Jamaica", "last_matches": 5},
    )

    assert response.status_code == 200
    assert response.json() == payload


def test_recent_form_endpoint_passes_libertadores_mode(monkeypatch):
    main = _load_main(monkeypatch)

    captured = {}

    def fake_service(**kwargs):
        captured.update(kwargs)
        return {
            "home_team": "Palmeiras",
            "away_team": "Boca Juniors",
            "home_matches": [],
            "away_matches": [],
        }

    monkeypatch.setattr(main, "get_recent_matches", fake_service)

    client = TestClient(main.app)
    response = client.post(
        "/recent-form",
        json={"home_team": "Palmeiras", "away_team": "Boca Juniors", "last_matches": 5, "mode": "libertadores"},
    )

    assert response.status_code == 200
    assert captured["mode"] == "libertadores"


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


def test_matches_calendar_endpoint_uses_service(monkeypatch):
    main = _load_main(monkeypatch)

    service_response = {
        "mode": "world_cup",
        "matches": [
            {
                "match_id": "22222222-2222-2222-2222-222222222222",
                "home_team": "Ecuador",
                "away_team": "Germany",
                "match_date": "2026-06-25",
                "tournament": "FIFA World Cup",
                "predicted_outcome": "home_win",
                "prediction_id": "11111111-1111-1111-1111-111111111111",
                "prediction_created_at": "2026-02-18T12:00:00+00:00",
            }
        ],
    }
    captured = {}

    def fake_service(**kwargs):
        captured.update(kwargs)
        return service_response

    monkeypatch.setattr(main, "list_matches_calendar", fake_service)

    client = TestClient(main.app)
    response = client.get("/matches-calendar?mode=world_cup")

    assert response.status_code == 200
    assert response.json() == service_response
    assert captured["mode"] == "world_cup"
    assert captured["user_id"] == "dev-user"


def test_matches_calendar_endpoint_requires_token_in_prod(monkeypatch):
    main = _load_main_prod(monkeypatch)

    client = TestClient(main.app)
    response = client.get("/matches-calendar?mode=world_cup")

    assert response.status_code == 401


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


def test_admin_matches_upsert_uses_service(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "super-secret")
    main = _load_main(monkeypatch)

    service_response = {
        "received": 2,
        "inserted": 1,
        "updated": 1,
        "skipped": 0,
        "errors": [],
    }
    monkeypatch.setattr(main, "upsert_manual_matches_batch", lambda *_args, **_kwargs: service_response)

    client = TestClient(main.app)
    response = client.post(
        "/admin/matches/upsert-batch",
        headers={"X-Admin-Key": "super-secret"},
        json={
            "matches": [
                {
                    "date": "2026-03-01",
                    "home_team": "Brazil",
                    "away_team": "Argentina",
                    "home_score": 1,
                    "away_score": 0,
                    "tournament": "Friendly",
                    "city": "Rio de Janeiro",
                    "country": "Brazil",
                    "neutral": False,
                },
                {
                    "date": "2026-03-05",
                    "home_team": "USA",
                    "away_team": "Mexico",
                    "home_score": 2,
                    "away_score": 1,
                    "tournament": "Friendly",
                    "city": "Houston",
                    "country": "United States",
                    "neutral": False,
                },
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


def test_model_scorecard_endpoint_allows_missing_model_version(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "2026_01_national_v1")
    main = _load_main(monkeypatch)

    captured = {}

    def fake_service(**kwargs):
        captured.update(kwargs)
        return {
            "mode": "national",
            "model_version": "2026_01_national_v1",
            "from_date": "2025-09-01",
            "to_date": "2026-01-31",
            "correct_count": 210,
            "incorrect_count": 120,
            "total_scored": 330,
            "accuracy_pct": 63.6364,
        }

    monkeypatch.setattr(main, "get_model_scorecard", fake_service)

    client = TestClient(main.app)
    response = client.get(
        "/model-scorecard",
        params={
            "mode": "national",
            "from_date": "2025-09-01",
            "to_date": "2026-01-31",
        },
    )

    assert response.status_code == 200
    assert captured["model_version"] == ""


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


def test_model_scorecard_matches_endpoint_allows_missing_model_version(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "2026_01_national_v1")
    main = _load_main(monkeypatch)

    captured = {}

    def fake_service(**kwargs):
        captured.update(kwargs)
        return {
            "mode": "national",
            "model_version": "2026_01_national_v1",
            "from_date": "2025-09-01",
            "to_date": "2026-01-31",
            "verdict": "all",
            "page": 1,
            "page_size": 2,
            "total": 0,
            "matches": [],
        }

    monkeypatch.setattr(main, "list_model_scorecard_matches", fake_service)

    client = TestClient(main.app)
    response = client.get(
        "/model-scorecard/matches",
        params={
            "mode": "national",
            "from_date": "2025-09-01",
            "to_date": "2026-01-31",
            "page": 1,
            "page_size": 2,
            "verdict": "all",
        },
    )

    assert response.status_code == 200
    assert captured["model_version"] == ""


def test_top_searched_teams_endpoint_uses_service(monkeypatch):
    main = _load_main(monkeypatch)

    service_response = {
        "mode": "national",
        "snapshot_date": "2026-03-12",
        "lookback_days_used": 7,
        "calculated_at": "2026-03-12T18:00:00Z",
        "teams": [
            {"rank": 1, "team": "Argentina", "searches": 128},
            {"rank": 2, "team": "Brazil", "searches": 121},
        ],
    }
    monkeypatch.setattr(main, "get_top_searched_teams_snapshot", lambda **_: service_response)

    client = TestClient(main.app)
    response = client.get("/top-searched-teams", params={"mode": "national"})

    assert response.status_code == 200
    assert response.json() == service_response


def test_top_searched_teams_endpoint_rejects_invalid_mode(monkeypatch):
    main = _load_main(monkeypatch)

    def raise_invalid_mode(**_kwargs):
        raise ValueError("mode must be one of: national, champions")

    monkeypatch.setattr(main, "get_top_searched_teams_snapshot", raise_invalid_mode)

    client = TestClient(main.app)
    response = client.get("/top-searched-teams", params={"mode": "invalid"})

    assert response.status_code == 400
    assert response.json()["detail"] == "mode must be one of: national, champions"


def test_top_searched_teams_endpoint_requires_auth(monkeypatch):
    main = _load_main_prod(monkeypatch)

    client = TestClient(main.app)
    response = client.get("/top-searched-teams")

    assert response.status_code == 401


def test_prediction_rankings_endpoint_uses_service(monkeypatch):
    main = _load_main(monkeypatch)

    service_response = {
        "mode": "national",
        "sort_by": "correct_count",
        "sort_order": "desc",
        "page": 1,
        "page_size": 5,
        "total_users": 2,
        "rankings": [
            {
                "rank": 1,
                "user_id": "user-1",
                "display_name": "Ana",
                "correct_count": 4,
                "incorrect_count": 1,
                "total_resolved_predictions": 5,
                "accuracy_pct": 80.0,
            },
            {
                "rank": 2,
                "user_id": "user-2",
                "display_name": "Ben",
                "correct_count": 3,
                "incorrect_count": 2,
                "total_resolved_predictions": 5,
                "accuracy_pct": 60.0,
            },
        ],
    }
    captured = {}

    def fake_service(**kwargs):
        captured.update(kwargs)
        return service_response

    monkeypatch.setattr(main, "list_prediction_rankings", fake_service)

    client = TestClient(main.app)
    response = client.get(
        "/prediction-rankings",
        params={
            "mode": "national",
            "sort_by": "correct_count",
            "sort_order": "desc",
            "page": 1,
            "page_size": 5,
        },
    )

    assert response.status_code == 200
    assert response.json() == service_response
    assert captured == {
        "mode": "national",
        "sort_by": "correct_count",
        "sort_order": "desc",
        "page": 1,
        "page_size": 5,
    }


def test_prediction_rankings_endpoint_supports_world_cup_mode(monkeypatch):
    main = _load_main(monkeypatch)

    service_response = {
        "mode": "world_cup",
        "sort_by": "correct_count",
        "sort_order": "desc",
        "page": 1,
        "page_size": 5,
        "total_users": 0,
        "rankings": [],
    }
    captured = {}

    def fake_service(**kwargs):
        captured.update(kwargs)
        return service_response

    monkeypatch.setattr(main, "list_prediction_rankings", fake_service)

    client = TestClient(main.app)
    response = client.get(
        "/prediction-rankings",
        params={
            "mode": "world_cup",
            "sort_by": "correct_count",
            "sort_order": "desc",
            "page": 1,
            "page_size": 5,
        },
    )

    assert response.status_code == 200
    assert response.json() == service_response
    assert captured["mode"] == "world_cup"


def test_prediction_rankings_endpoint_rejects_invalid_input(monkeypatch):
    main = _load_main(monkeypatch)

    def raise_invalid(**_kwargs):
        raise ValueError("mode must be one of: all, national, champions")

    monkeypatch.setattr(main, "list_prediction_rankings", raise_invalid)

    client = TestClient(main.app)
    response = client.get("/prediction-rankings", params={"mode": "invalid"})

    assert response.status_code == 400
    assert response.json()["detail"] == "mode must be one of: all, national, champions"


def test_prediction_rankings_endpoint_supports_missing_mode(monkeypatch):
    main = _load_main(monkeypatch)

    captured = {}

    def fake_service(**kwargs):
        captured.update(kwargs)
        return {
            "mode": "all",
            "sort_by": "correct_count",
            "sort_order": "desc",
            "page": 1,
            "page_size": 50,
            "total_users": 0,
            "rankings": [],
        }

    monkeypatch.setattr(main, "list_prediction_rankings", fake_service)

    client = TestClient(main.app)
    response = client.get("/prediction-rankings")

    assert response.status_code == 200
    assert captured["mode"] is None


def test_prediction_rankings_endpoint_requires_auth(monkeypatch):
    main = _load_main_prod(monkeypatch)

    client = TestClient(main.app)
    response = client.get("/prediction-rankings", params={"mode": "national"})

    assert response.status_code == 401
