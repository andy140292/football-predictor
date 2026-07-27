import importlib

import pytest

from src.utils.matches_common import resolve_latest_matches_csv


def _load_module():
    import utils.upload_matches_to_supabase as upload_matches_to_supabase

    return importlib.reload(upload_matches_to_supabase)


def test_resolve_latest_matches_csv_ignores_calendar_files(tmp_path):
    (tmp_path / "matches_2026_05.csv").write_text("date\\n", encoding="utf-8")
    (tmp_path / "matches_2026_06.csv").write_text("date\\n", encoding="utf-8")
    (tmp_path / "matches_calendar_uefa_2026_07_01.csv").write_text(
        "match_date\\n",
        encoding="utf-8",
    )

    assert resolve_latest_matches_csv(tmp_path) == tmp_path / "matches_2026_06.csv"


def test_load_records_reports_incompatible_csv_schema(tmp_path):
    module = _load_module()
    csv_path = tmp_path / "matches_calendar.csv"
    csv_path.write_text(
        "home_team,away_team,match_date\\nArgentina,Brazil,2026-06-01\\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns:.*date"):
        module.load_records(csv_path)


def test_reconcile_skips_cosmetic_source_file_only_updates(monkeypatch):
    module = _load_module()

    monkeypatch.setattr(
        module,
        "_fetch_existing_rows_for_date_range",
        lambda client, table_name, from_date, to_date: [
            {
                "match_id": 1,
                "match_date": "2026-03-01",
                "home_team": "Brazil",
                "away_team": "Argentina",
                "home_score": 1,
                "away_score": 0,
                "tournament": "Friendly",
                "city": "Rio de Janeiro",
                "country": "Brazil",
                "neutral": False,
                "source": "csv_import",
                "source_file": "matches_2026_02.csv",
                "created_at": "2026-03-01T00:00:00+00:00",
            }
        ],
    )

    inserts, updates, deletes, conflicts, resolved_conflicts = module._reconcile_records_with_existing(
        client=None,
        table_name="matches",
        records=[
            {
                "match_date": "2026-03-01",
                "home_team": "Brazil",
                "away_team": "Argentina",
                "home_score": 1,
                "away_score": 0,
                "tournament": "Friendly",
                "city": "Rio de Janeiro",
                "country": "Brazil",
                "neutral": False,
                "source": "csv_import",
                "source_file": "matches_2026_03.csv",
            }
        ],
    )

    assert inserts == []
    assert updates == []
    assert deletes == []
    assert conflicts == []
    assert resolved_conflicts == []


def test_reconcile_allows_safe_enrichment_updates(monkeypatch):
    module = _load_module()

    monkeypatch.setattr(
        module,
        "_fetch_existing_rows_for_date_range",
        lambda client, table_name, from_date, to_date: [
            {
                "match_id": 1,
                "match_date": "2026-03-27",
                "home_team": "Argentina",
                "away_team": "Mauritania",
                "home_score": 2,
                "away_score": 0,
                "tournament": "Friendly",
                "city": None,
                "country": None,
                "neutral": False,
                "source": "soccerway_feed",
                "source_file": "feed",
                "created_at": "2026-03-31T00:00:00+00:00",
            }
        ],
    )

    inserts, updates, deletes, conflicts, resolved_conflicts = module._reconcile_records_with_existing(
        client=None,
        table_name="matches",
        records=[
            {
                "match_date": "2026-03-27",
                "home_team": "Argentina",
                "away_team": "Mauritania",
                "home_score": 2,
                "away_score": 0,
                "tournament": "Friendly",
                "city": "Buenos Aires",
                "country": "Argentina",
                "neutral": False,
                "source": "csv_import",
                "source_file": "matches_2026_03.csv",
            }
        ],
    )

    assert inserts == []
    assert len(updates) == 1
    assert updates[0]["match_id"] == 1
    assert updates[0]["payload"]["city"] == "Buenos Aires"
    assert deletes == []
    assert conflicts == []
    assert resolved_conflicts == []


def test_reconcile_reports_conflicts_without_updating(monkeypatch):
    module = _load_module()

    monkeypatch.setattr(
        module,
        "_fetch_existing_rows_for_date_range",
        lambda client, table_name, from_date, to_date: [
            {
                "match_id": 1,
                "match_date": "2026-03-26",
                "home_team": "Croatia",
                "away_team": "Colombia",
                "home_score": 2,
                "away_score": 1,
                "tournament": "Friendly",
                "city": "Orlando",
                "country": "United States",
                "neutral": True,
                "source": "csv_import",
                "source_file": "matches_2026_03_26_uefa.csv",
                "created_at": "2026-03-27T00:00:00+00:00",
            }
        ],
    )

    inserts, updates, deletes, conflicts, resolved_conflicts = module._reconcile_records_with_existing(
        client=None,
        table_name="matches",
        records=[
            {
                "match_date": "2026-03-26",
                "home_team": "Colombia",
                "away_team": "Croatia",
                "home_score": 1,
                "away_score": 2,
                "tournament": "Friendly",
                "city": "Orlando",
                "country": "United States",
                "neutral": True,
                "source": "csv_import",
                "source_file": "matches_2026_03.csv",
            }
        ],
    )

    assert inserts == []
    assert updates == []
    assert deletes == []
    assert len(conflicts) == 1
    assert conflicts[0]["match_id"] == 1
    assert "home_team" in conflicts[0]["diffs"]
    assert "away_team" in conflicts[0]["diffs"]
    assert resolved_conflicts == []


def test_reconcile_can_resolve_conflicts_with_incoming(monkeypatch):
    module = _load_module()

    monkeypatch.setattr(
        module,
        "_fetch_existing_rows_for_date_range",
        lambda client, table_name, from_date, to_date: [
            {
                "match_id": 1,
                "match_date": "2026-03-26",
                "home_team": "Croatia",
                "away_team": "Colombia",
                "home_score": 2,
                "away_score": 1,
                "tournament": "Friendly",
                "city": "Orlando",
                "country": "United States",
                "neutral": True,
                "source": "csv_import",
                "source_file": "matches_2026_03_26_uefa.csv",
                "created_at": "2026-03-27T00:00:00+00:00",
            }
        ],
    )

    inserts, updates, deletes, conflicts, resolved_conflicts = module._reconcile_records_with_existing(
        client=None,
        table_name="matches",
        records=[
            {
                "match_date": "2026-03-26",
                "home_team": "Colombia",
                "away_team": "Croatia",
                "home_score": 1,
                "away_score": 2,
                "tournament": "Friendly",
                "city": "Orlando",
                "country": "United States",
                "neutral": True,
                "source": "csv_import",
                "source_file": "matches_2026_03.csv",
            }
        ],
        resolve_conflicts="incoming",
    )

    assert inserts == []
    assert len(updates) == 1
    assert updates[0]["match_id"] == 1
    assert updates[0]["payload"]["home_team"] == "Colombia"
    assert deletes == []
    assert conflicts == []
    assert len(resolved_conflicts) == 1
    assert resolved_conflicts[0]["match_id"] == 1


def test_split_completed_vs_calendar_routes_missing_scores_to_calendar():
    module = _load_module()

    completed, calendar_rows = module._split_completed_vs_calendar(
        [
            {
                "match_date": "2026-03-31",
                "home_team": "Brazil",
                "away_team": "Croatia",
                "home_score": 3,
                "away_score": 1,
                "tournament": "Friendly",
            },
            {
                "match_date": "2026-06-13",
                "home_team": "Brazil",
                "away_team": "Morocco",
                "home_score": None,
                "away_score": None,
                "tournament": "FIFA World Cup",
            },
        ]
    )

    assert len(completed) == 1
    assert completed[0]["away_team"] == "Croatia"
    assert calendar_rows == [
        {
            "match_date": "2026-06-13",
            "home_team": "Brazil",
            "away_team": "Morocco",
            "tournament": "FIFA World Cup",
        }
    ]
