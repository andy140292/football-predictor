import importlib

from fastapi.testclient import TestClient


def test_healthz_returns_ok(monkeypatch):
    monkeypatch.setenv("API_ENV", "dev")
    import src.backend.main as main
    importlib.reload(main)

    client = TestClient(main.app)
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_uses_predict_outcome(monkeypatch):
    monkeypatch.setenv("API_ENV", "dev")
    import src.backend.main as main
    importlib.reload(main)

    monkeypatch.setattr(main, "predict_outcome", lambda **_: {"ok": True})

    client = TestClient(main.app)
    response = client.post("/predict", json={"home_team": "Brazil", "away_team": "Argentina"})

    assert response.status_code == 200
    assert response.json() == {"predicción": {"ok": True}}
