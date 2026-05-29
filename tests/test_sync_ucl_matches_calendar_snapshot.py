from pathlib import Path

from src.backend.jobs.sync_ucl_matches_calendar_snapshot import (
    SnapshotFixture,
    build_sync_plan,
    load_alias_maps,
)


def test_build_sync_plan_reuses_existing_ucl_aliases(tmp_path):
    alias_path = tmp_path / "club_team_aliases.csv"
    alias_path.write_text(
        "alias,canonical\n"
        "Paris Saint-Germain,Paris\n"
        "PSG,Paris\n"
        "Bayern Munich,Bayern München\n"
        "Bayern,Bayern München\n"
        "Sporting Lisboa,Sporting CP\n"
    )
    alias_to_canonical, _ = load_alias_maps(alias_path)

    existing_rows = [
        {
            "match_id": "1",
            "match_date": "2026-04-14",
            "home_team": "Liverpool",
            "away_team": "PSG",
            "home_score": None,
            "away_score": None,
            "tournament": "UEFA Champions League",
            "result_source": None,
        },
        {
            "match_id": "2",
            "match_date": "2026-04-15",
            "home_team": "Arsenal",
            "away_team": "Sporting Lisboa",
            "home_score": None,
            "away_score": None,
            "tournament": "UEFA Champions League",
            "result_source": None,
        },
    ]
    fixtures = [
        SnapshotFixture("2026-04-14", "Liverpool", "Paris Saint-Germain", 0, 2, "Quarter-finals"),
        SnapshotFixture("2026-04-15", "Arsenal", "Sporting CP", 0, 0, "Quarter-finals"),
    ]

    plan = build_sync_plan(fixtures, existing_rows, alias_to_canonical)

    assert [item.action for item in plan] == ["update", "update"]
    assert plan[0].payload["away_team"] == "PSG"
    assert plan[1].payload["away_team"] == "Sporting Lisboa"


def test_build_sync_plan_inserts_future_rows_with_existing_notation(tmp_path):
    alias_path = tmp_path / "club_team_aliases.csv"
    alias_path.write_text(
        "alias,canonical\n"
        "Paris Saint-Germain,Paris\n"
        "PSG,Paris\n"
        "Bayern Munich,Bayern München\n"
        "Bayern,Bayern München\n"
    )
    alias_to_canonical, _ = load_alias_maps(alias_path)

    existing_rows = [
        {
            "match_id": "1",
            "match_date": "2026-04-08",
            "home_team": "PSG",
            "away_team": "Liverpool",
            "home_score": 2,
            "away_score": 0,
            "tournament": "UEFA Champions League",
            "result_source": "fbref",
        },
        {
            "match_id": "2",
            "match_date": "2026-04-07",
            "home_team": "Real Madrid",
            "away_team": "Bayern",
            "home_score": 1,
            "away_score": 2,
            "tournament": "UEFA Champions League",
            "result_source": "fbref",
        },
    ]
    fixtures = [
        SnapshotFixture("2026-04-28", "Paris Saint-Germain", "Bayern Munich", None, None, "Semi-finals"),
    ]

    plan = build_sync_plan(fixtures, existing_rows, alias_to_canonical)

    assert len(plan) == 1
    assert plan[0].action == "insert"
    assert plan[0].payload == {
        "match_date": "2026-04-28",
        "home_team": "PSG",
        "away_team": "Bayern",
        "tournament": "UEFA Champions League",
    }
