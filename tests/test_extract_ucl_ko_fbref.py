import pandas as pd

from src.data.extract_ucl_ko_fbref import (
    TEAM_FIXTURE_COLUMNS,
    FIRST_LEG_COLUMNS,
    TeamSource,
    build_matchlog_url,
    dedupe_team_sources,
    ensure_columns,
    filter_first_legs,
    is_ko_playoff_round,
    normalize_team_display,
)


def test_is_ko_playoff_round():
    assert is_ko_playoff_round("Knockout phase play-offs")
    assert is_ko_playoff_round(" knockout phase play-offs ")
    assert not is_ko_playoff_round("League phase")


def test_normalize_team_display_removes_country_codes():
    assert normalize_team_display("Benfica pt") == "Benfica"
    assert normalize_team_display("es Real Madrid") == "Real Madrid"
    assert normalize_team_display("Bodø/Glimt no") == "Bodø/Glimt"


def test_build_matchlog_url_from_squad_url():
    url = build_matchlog_url("https://fbref.com/en/squads/a77c513e/Benfica-Stats")
    assert url == (
        "https://fbref.com/en/squads/a77c513e/2025-2026/matchlogs/all_comps/schedule/"
        "Benfica-Scores-and-Fixtures-All-Competitions"
    )


def test_dedupe_team_sources_by_team_name():
    raw = [
        TeamSource(team="Benfica pt", team_url="/en/squads/a77c513e/Benfica-Stats"),
        TeamSource(team="pt Benfica", team_url="https://fbref.com/en/squads/a77c513e/Benfica-Stats"),
        TeamSource(team="Real Madrid es", team_url="/en/squads/53a2f082/Real-Madrid-Stats"),
    ]
    deduped = dedupe_team_sources(raw)
    assert len(deduped) == 2
    assert deduped[0].team == "Benfica"
    assert deduped[0].team_url == "https://fbref.com/en/squads/a77c513e/Benfica-Stats"


def test_filter_first_legs_only_returns_feb_17_18():
    rows = [
        {
            "date": "2026-02-17",
            "time": "20:00",
            "round": "Knockout phase play-offs",
            "home_team": "Benfica pt",
            "away_team": "es Real Madrid",
            "home_team_url": "/en/squads/a77c513e/Benfica-Stats",
            "away_team_url": "/en/squads/53a2f082/Real-Madrid-Stats",
        },
        {
            "date": "2026-02-24",
            "time": "20:00",
            "round": "Knockout phase play-offs",
            "home_team": "Real Madrid es",
            "away_team": "pt Benfica",
            "home_team_url": "/en/squads/53a2f082/Real-Madrid-Stats",
            "away_team_url": "/en/squads/a77c513e/Benfica-Stats",
        },
    ]

    first_legs = filter_first_legs(rows)
    assert len(first_legs) == 1
    assert first_legs[0]["home_team"] == "Benfica"
    assert first_legs[0]["away_team"] == "Real Madrid"


def test_first_leg_regression_count_is_eight():
    rows = []
    for i in range(8):
        rows.append(
            {
                "date": "2026-02-17",
                "time": "20:00",
                "round": "Knockout phase play-offs",
                "home_team": f"Home {i} pt",
                "away_team": f"es Away {i}",
                "home_team_url": f"/en/squads/home{i}/Home-{i}-Stats",
                "away_team_url": f"/en/squads/away{i}/Away-{i}-Stats",
            }
        )
    for i in range(8):
        rows.append(
            {
                "date": "2026-02-24",
                "time": "20:00",
                "round": "Knockout phase play-offs",
                "home_team": f"Return Home {i}",
                "away_team": f"Return Away {i}",
                "home_team_url": f"/en/squads/rhome{i}/RHome-{i}-Stats",
                "away_team_url": f"/en/squads/raway{i}/RAway-{i}-Stats",
            }
        )

    first_legs = filter_first_legs(rows)
    assert len(first_legs) == 8


def test_ensure_columns_for_team_fixture_schema():
    df = pd.DataFrame([{"date": "2026-02-17", "team": "Benfica"}])
    aligned = ensure_columns(df, TEAM_FIXTURE_COLUMNS)
    assert list(aligned.columns) == TEAM_FIXTURE_COLUMNS
    assert aligned.loc[0, "date"] == "2026-02-17"


def test_ensure_columns_for_first_leg_schema():
    df = pd.DataFrame([{"date": "2026-02-17", "home_team": "Benfica"}])
    aligned = ensure_columns(df, FIRST_LEG_COLUMNS)
    assert list(aligned.columns) == FIRST_LEG_COLUMNS


def test_integration_smoke_script_import_and_build_url():
    # Smoke-level integration substitute for environments without Playwright.
    # If Playwright is available in runtime, this confirms expected import path.
    try:
        import playwright.sync_api  # noqa: F401
    except Exception:
        return

    url = build_matchlog_url("https://fbref.com/en/squads/a77c513e/Benfica-Stats")
    assert "matchlogs/all_comps/schedule" in url
