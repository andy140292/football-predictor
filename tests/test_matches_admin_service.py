import importlib

import pandas as pd


def _load_service():
    import backend.matches_admin_service as matches_admin_service
    return importlib.reload(matches_admin_service)


def test_upsert_manual_matches_batch_enriches_and_summarizes(monkeypatch):
    service = _load_service()
    captured = {}

    monkeypatch.setattr(
        service,
        "add_confederation_to_matches",
        lambda df, confed_path=None: df.assign(
            home_team_confederation="CONMEBOL",
            away_team_confederation="CONMEBOL",
        ),
    )
    monkeypatch.setattr(
        service,
        "_load_ranking_df",
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        service,
        "add_ranking_fifa_to_matches",
        lambda df, _ranking: df.assign(
            home_team=df["home_team"].replace({"USA": "United States"}),
            away_team=df["away_team"].replace({"USA": "United States"}),
            home_team_fifa_rank=1.0,
            home_team_fifa_points=1800.0,
            away_team_fifa_rank=2.0,
            away_team_fifa_points=1750.0,
        ),
    )
    monkeypatch.setattr(
        service,
        "_reconcile_match_records",
        lambda records: (
            [records[1]],
            [{"match_id": 999, "payload": records[0]}],
            [],
            {records[0]["match_key"]},
        ),
    )
    monkeypatch.setattr(service, "_insert_match_rows", lambda rows: captured.setdefault("insert_rows", rows))
    monkeypatch.setattr(service, "_update_match_row", lambda match_id, payload: captured.setdefault("update_rows", []).append({"match_id": match_id, "payload": payload}))
    monkeypatch.setattr(service, "_delete_matches_rows", lambda match_ids: 0)

    result = service.upsert_manual_matches_batch(
        [
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
            {
                "date": "",
                "home_team": "Brazil",
                "away_team": "Chile",
                "home_score": 1,
                "away_score": 1,
                "tournament": "Friendly",
                "city": "Sao Paulo",
                "country": "Brazil",
                "neutral": False,
            },
        ]
    )

    assert result["received"] == 4
    assert result["updated"] == 1
    assert result["inserted"] == 1
    assert result["skipped"] == 2
    assert len(result["errors"]) == 2
    assert captured["update_rows"][0]["payload"]["source"] == "manual"
    assert captured["update_rows"][0]["payload"]["home_team_confederation"] == "CONMEBOL"
    assert captured["insert_rows"][0]["home_team"] == "United States"


def test_upsert_manual_matches_batch_skips_missing_confederation(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(
        service,
        "add_confederation_to_matches",
        lambda df, confed_path=None: df.assign(
            home_team_confederation=pd.NA,
            away_team_confederation="CONMEBOL",
        ),
    )
    monkeypatch.setattr(service, "_load_ranking_df", lambda: pd.DataFrame())
    monkeypatch.setattr(
        service,
        "add_ranking_fifa_to_matches",
        lambda df, _ranking: df.assign(
            home_team_fifa_rank=1.0,
            home_team_fifa_points=1800.0,
            away_team_fifa_rank=2.0,
            away_team_fifa_points=1750.0,
        ),
    )
    monkeypatch.setattr(service, "_reconcile_match_records", lambda records: (_ for _ in ()).throw(AssertionError("should not reconcile")))

    result = service.upsert_manual_matches_batch(
        [
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
        ]
    )

    assert result["inserted"] == 0
    assert result["updated"] == 0
    assert result["skipped"] == 1
    assert result["errors"][0]["reason"] == "missing confederation mapping"


def test_upsert_manual_matches_batch_updates_reversed_existing_fixture(monkeypatch):
    service = _load_service()
    captured = {"updates": [], "inserts": [], "deletes": []}

    monkeypatch.setattr(
        service,
        "add_confederation_to_matches",
        lambda df, confed_path=None: df.assign(
            home_team_confederation="CONMEBOL",
            away_team_confederation="UEFA",
        ),
    )
    monkeypatch.setattr(service, "_load_ranking_df", lambda: pd.DataFrame())
    monkeypatch.setattr(
        service,
        "add_ranking_fifa_to_matches",
        lambda df, _ranking: df.assign(
            home_team_fifa_rank=10.0,
            home_team_fifa_points=1700.0,
            away_team_fifa_rank=11.0,
            away_team_fifa_points=1680.0,
        ),
    )
    monkeypatch.setattr(
        service,
        "_fetch_existing_matches_for_range",
        lambda from_date, to_date, chunk_size=1000: [
            {
                "match_id": 102711,
                "match_key": "2026-03-26|colombia|croatia|friendly|||0",
                "match_date": "2026-03-26",
                "home_team": "Colombia",
                "away_team": "Croatia",
                "home_score": 1,
                "away_score": 2,
                "tournament": "Friendly",
                "city": "",
                "country": "",
                "neutral": False,
                "source": "soccerway_feed",
                "source_file": "feed",
                "created_at": "2026-03-31T03:09:14.731942+00:00",
            }
        ],
    )
    monkeypatch.setattr(service, "_insert_match_rows", lambda rows: captured["inserts"].extend(rows))
    monkeypatch.setattr(service, "_update_match_row", lambda match_id, payload: captured["updates"].append({"match_id": match_id, "payload": payload}))
    monkeypatch.setattr(service, "_delete_matches_rows", lambda match_ids: captured["deletes"].extend(match_ids) or len(match_ids))

    result = service.upsert_manual_matches_batch(
        [
            {
                "date": "2026-03-26",
                "home_team": "Croatia",
                "away_team": "Colombia",
                "home_score": 2,
                "away_score": 1,
                "tournament": "Friendly",
                "city": "Orlando",
                "country": "United States",
                "neutral": True,
            },
        ]
    )

    assert result["inserted"] == 0
    assert result["updated"] == 1
    assert captured["inserts"] == []
    assert captured["deletes"] == []
    assert captured["updates"][0]["match_id"] == 102711
    assert captured["updates"][0]["payload"]["home_team"] == "Croatia"
    assert captured["updates"][0]["payload"]["away_team"] == "Colombia"


def test_cleanup_matches_duplicates_prefers_richer_exact_row(monkeypatch):
    service = _load_service()
    deleted = []

    monkeypatch.setattr(
        service,
        "_fetch_matches_rows_for_cleanup",
        lambda from_date, to_date, chunk_size=1000: [
            {
                "match_id": 102712,
                "match_date": "2026-03-26",
                "home_team": "Brazil",
                "away_team": "France",
                "city": "",
                "country": "",
                "neutral": False,
                "source": "soccerway_feed",
                "source_file": "feed",
                "created_at": "2026-03-31T03:09:14.731942+00:00",
            },
            {
                "match_id": 102654,
                "match_date": "2026-03-26",
                "home_team": "Brazil",
                "away_team": "France",
                "city": "Boston",
                "country": "United States",
                "neutral": True,
                "source": "csv_import",
                "source_file": "matches_2026_03_26_uefa.csv",
                "created_at": "2026-03-27T06:55:16.494994+00:00",
            },
        ],
    )
    monkeypatch.setattr(service, "_delete_matches_rows", lambda match_ids, chunk_size=200: deleted.extend(match_ids) or len(match_ids))

    result = service.cleanup_matches_duplicates("2026-03-26", "2026-03-31")

    assert result["duplicate_groups"] == 1
    assert result["exact_duplicate_groups"] == 1
    assert result["reversed_duplicate_groups"] == 0
    assert result["deleted_rows"] == 1
    assert deleted == [102712]


def test_cleanup_matches_duplicates_collapses_reversed_fixture(monkeypatch):
    service = _load_service()
    deleted = []

    monkeypatch.setattr(
        service,
        "_fetch_matches_rows_for_cleanup",
        lambda from_date, to_date, chunk_size=1000: [
            {
                "match_id": 102711,
                "match_date": "2026-03-26",
                "home_team": "Colombia",
                "away_team": "Croatia",
                "city": "",
                "country": "",
                "neutral": False,
                "source": "soccerway_feed",
                "source_file": "feed",
                "created_at": "2026-03-31T03:09:14.731942+00:00",
            },
            {
                "match_id": 102655,
                "match_date": "2026-03-26",
                "home_team": "Croatia",
                "away_team": "Colombia",
                "city": "Orlando",
                "country": "United States",
                "neutral": True,
                "source": "csv_import",
                "source_file": "matches_2026_03_26_uefa.csv",
                "created_at": "2026-03-27T06:55:16.494994+00:00",
            },
        ],
    )
    monkeypatch.setattr(service, "_delete_matches_rows", lambda match_ids, chunk_size=200: deleted.extend(match_ids) or len(match_ids))

    result = service.cleanup_matches_duplicates("2026-03-26", "2026-03-31")

    assert result["duplicate_groups"] == 1
    assert result["exact_duplicate_groups"] == 0
    assert result["reversed_duplicate_groups"] == 1
    assert result["deleted_rows"] == 1
    assert deleted == [102711]
