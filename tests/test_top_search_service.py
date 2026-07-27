import importlib
from datetime import datetime


def _load_service():
    import backend.top_search_service as top_search_service
    return importlib.reload(top_search_service)


def test_load_national_team_keys_uses_supabase_names(monkeypatch):
    service = _load_service()
    monkeypatch.setattr(
        service,
        "fetch_national_team_names",
        lambda: ["Argentina", "Brazil", "Curaçao"],
    )
    service._national_team_keys = None

    keys = service._load_national_team_keys()

    assert service._normalized_text("Argentina") in keys
    assert service._normalized_text("Curaçao") in keys


def test_build_daily_top_searched_snapshot_uses_7_day_window_for_national(monkeypatch):
    service = _load_service()
    now = datetime(2026, 3, 12, 18, 0, 0)
    expected_start_iso = datetime(2026, 3, 5, 18, 0, 0).isoformat()
    calls = []

    def fake_fetch(start_iso, **_kwargs):
        calls.append(start_iso)
        assert start_iso == expected_start_iso
        return [
            {"home_team": "Argentina", "away_team": "Brazil", "mode": "national"},
            {"home_team": "Brazil", "away_team": "Chile", "mode": "national"},
            {"home_team": "Argentina", "away_team": "Brazil", "mode": "national"},
            {"home_team": "Chile", "away_team": "Brazil", "mode": "national"},
        ]

    monkeypatch.setattr(service, "_fetch_recent_user_predictions", fake_fetch)

    snapshot = service.build_daily_top_searched_snapshot("national", now=now)

    assert calls == [expected_start_iso]
    assert snapshot["snapshot_date"] == "2026-03-12"
    assert snapshot["lookback_days_used"] == 7
    assert snapshot["teams"] == [
        {"rank": 1, "team": "Brazil", "searches": 4},
        {"rank": 2, "team": "Argentina", "searches": 2},
        {"rank": 3, "team": "Chile", "searches": 2},
    ]


def test_build_daily_top_searched_snapshot_falls_back_to_30_days_for_national(monkeypatch):
    service = _load_service()
    now = datetime(2026, 3, 12, 18, 0, 0)
    seven_day_start_iso = datetime(2026, 3, 5, 18, 0, 0).isoformat()
    thirty_day_start_iso = datetime(2026, 2, 10, 18, 0, 0).isoformat()
    calls = []

    def fake_fetch(start_iso, **_kwargs):
        calls.append(start_iso)
        if start_iso == seven_day_start_iso:
            return []
        if start_iso == thirty_day_start_iso:
            return [
                {"home_team": "Argentina", "away_team": "Brazil", "mode": "national"},
                {"home_team": "Brazil", "away_team": "Chile", "mode": "national"},
            ]
        raise AssertionError(f"unexpected start_iso: {start_iso}")

    monkeypatch.setattr(service, "_fetch_recent_user_predictions", fake_fetch)

    snapshot = service.build_daily_top_searched_snapshot("national", now=now)

    assert calls == [seven_day_start_iso, thirty_day_start_iso]
    assert snapshot["lookback_days_used"] == 30
    assert snapshot["teams"] == [
        {"rank": 1, "team": "Brazil", "searches": 2},
        {"rank": 2, "team": "Argentina", "searches": 1},
        {"rank": 3, "team": "Chile", "searches": 1},
    ]


def test_build_daily_top_searched_snapshot_uses_30_day_window_for_champions(monkeypatch):
    service = _load_service()
    now = datetime(2026, 3, 12, 18, 0, 0)
    expected_start_iso = datetime(2026, 2, 10, 18, 0, 0).isoformat()
    calls = []

    def fake_fetch(start_iso, **_kwargs):
        calls.append(start_iso)
        assert start_iso == expected_start_iso
        return [
            {"home_team": "River Plate", "away_team": "Boca Juniors", "mode": "club"},
        ]

    monkeypatch.setattr(service, "_fetch_recent_user_predictions", fake_fetch)

    snapshot = service.build_daily_top_searched_snapshot("champions", now=now)

    assert calls == [expected_start_iso]
    assert snapshot["lookback_days_used"] == 30
    assert snapshot["teams"] == [
        {"rank": 1, "team": "Boca Juniors", "searches": 1},
        {"rank": 2, "team": "River Plate", "searches": 1},
    ]


def test_get_top_searched_teams_snapshot_uses_existing_snapshot(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(
        service,
        "_fetch_daily_snapshot",
        lambda mode, snapshot_date: {
            "mode": mode,
            "snapshot_date": snapshot_date,
            "lookback_days_used": 7,
            "calculated_at": "2026-03-12T18:00:00Z",
            "teams": [{"rank": 1, "team": "Argentina", "searches": 10}],
        },
    )
    monkeypatch.setattr(
        service,
        "build_daily_top_searched_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("snapshot should not be rebuilt")),
    )

    result = service.get_top_searched_teams_snapshot("national", now=datetime(2026, 3, 12, 18, 0, 0))

    assert result["mode"] == "national"
    assert result["snapshot_date"] == "2026-03-12"
    assert result["lookback_days_used"] == 7
    assert result["teams"] == [{"rank": 1, "team": "Argentina", "searches": 10}]


def test_get_top_searched_teams_snapshot_builds_and_upserts_when_missing(monkeypatch):
    service = _load_service()
    captured = {}

    monkeypatch.setattr(service, "_fetch_daily_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        service,
        "build_daily_top_searched_snapshot",
        lambda mode, now=None: {
            "mode": mode,
            "snapshot_date": "2026-03-13",
            "lookback_days_used": 30,
            "calculated_at": "2026-03-13T00:00:01Z",
            "teams": [{"rank": 1, "team": "Brazil", "searches": 12}],
        },
    )

    def fake_upsert(snapshot):
        captured["snapshot"] = snapshot
        return snapshot

    monkeypatch.setattr(service, "upsert_daily_snapshot", fake_upsert)

    result = service.get_top_searched_teams_snapshot("champions", now=datetime(2026, 3, 13, 0, 0, 1))

    assert captured["snapshot"]["mode"] == "champions"
    assert result["snapshot_date"] == "2026-03-13"
    assert result["lookback_days_used"] == 30
    assert result["teams"] == [{"rank": 1, "team": "Brazil", "searches": 12}]


def test_get_top_searched_teams_snapshot_rebuilds_existing_snapshot_without_lookback(monkeypatch):
    service = _load_service()
    captured = {}

    monkeypatch.setattr(
        service,
        "_fetch_daily_snapshot",
        lambda mode, snapshot_date: {
            "mode": mode,
            "snapshot_date": snapshot_date,
            "calculated_at": "2026-03-12T18:00:00Z",
            "teams": [{"rank": 1, "team": "Argentina", "searches": 10}],
        },
    )
    monkeypatch.setattr(
        service,
        "build_daily_top_searched_snapshot",
        lambda mode, now=None: {
            "mode": mode,
            "snapshot_date": "2026-03-12",
            "lookback_days_used": 7,
            "calculated_at": "2026-03-12T19:00:00Z",
            "teams": [{"rank": 1, "team": "Brazil", "searches": 11}],
        },
    )

    def fake_upsert(snapshot):
        captured["snapshot"] = snapshot
        return snapshot

    monkeypatch.setattr(service, "upsert_daily_snapshot", fake_upsert)

    result = service.get_top_searched_teams_snapshot("national", now=datetime(2026, 3, 12, 19, 0, 0))

    assert captured["snapshot"]["lookback_days_used"] == 7
    assert result["teams"] == [{"rank": 1, "team": "Brazil", "searches": 11}]


def test_normalize_search_bucket_mode_maps_club_to_champions(monkeypatch):
    service = _load_service()

    assert service.normalize_search_bucket_mode("club") == "champions"
    assert service.normalize_search_bucket_mode("champions") == "champions"
    assert service.normalize_search_bucket_mode(None) == "national"
