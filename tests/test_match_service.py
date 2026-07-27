import importlib

import pandas as pd


def _load_service():
    import backend.match_service as match_service
    return importlib.reload(match_service)


class FakeQuery:
    def __init__(self):
        self.filters = []

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    @property
    def not_(self):
        return self

    def is_(self, *_args, **_kwargs):
        return self

    def in_(self, column, values):
        self.filters.append(("in", column, tuple(values)))
        return self

    def order(self, *_args, **_kwargs):
        return self

    def range(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self


def test_get_recent_matches_returns_separate_lists_per_team(monkeypatch):
    service = _load_service()

    def fake_fetch(team, last_matches, mode="national", request_id="-"):
        assert last_matches == 2
        assert mode == "national"
        if team == "Brazil":
            return pd.DataFrame(
                [
                    {"date": "2026-03-10", "home_team": "Brazil", "away_team": "Chile", "home_score": 2, "away_score": 0},
                    {"date": "2026-03-01", "home_team": "Argentina", "away_team": "Brazil", "home_score": 1, "away_score": 1},
                ]
            )
        return pd.DataFrame(
            [
                {"date": "2026-03-05", "home_team": "Argentina", "away_team": "Uruguay", "home_score": 3, "away_score": 1},
                {"date": "2026-03-01", "home_team": "Argentina", "away_team": "Brazil", "home_score": 1, "away_score": 1},
            ]
        )

    monkeypatch.setattr(service, "_fetch_recent_team_matches", fake_fetch)

    result = service.get_recent_matches("Brazil", "Argentina", last_matches=2)

    assert result == {
        "home_team": "Brazil",
        "away_team": "Argentina",
        "home_matches": [
            {"date": "2026-03-10", "home_team": "Brazil", "away_team": "Chile", "home_score": 2, "away_score": 0},
            {"date": "2026-03-01", "home_team": "Argentina", "away_team": "Brazil", "home_score": 1, "away_score": 1},
        ],
        "away_matches": [
            {"date": "2026-03-05", "home_team": "Argentina", "away_team": "Uruguay", "home_score": 3, "away_score": 1},
            {"date": "2026-03-01", "home_team": "Argentina", "away_team": "Brazil", "home_score": 1, "away_score": 1},
        ],
    }


def test_get_head_to_head_builds_match_list_and_form(monkeypatch):
    service = _load_service()
    h2h_df = pd.DataFrame(
        [
            {
                "date": "2026-03-10",
                "home_team": "Brazil",
                "away_team": "Argentina",
                "home_score": 2,
                "away_score": 1,
                "tournament": "Friendly",
            },
            {
                "date": "2025-11-10",
                "home_team": "Argentina",
                "away_team": "Brazil",
                "home_score": 0,
                "away_score": 0,
                "tournament": "FIFA World Cup qualification",
            },
            {
                "date": "2024-07-10",
                "home_team": "Brazil",
                "away_team": "Argentina",
                "home_score": 1,
                "away_score": 0,
                "tournament": "Copa América",
            },
        ]
    )

    monkeypatch.setattr(service, "_fetch_head_to_head_matches", lambda *_args, **_kwargs: h2h_df)

    result = service.get_head_to_head("Brazil", "Argentina", tournaments=["Friendly"])

    assert result["matches"] == [
        {"date": "2026-03-10", "home_team": "Brazil", "away_team": "Argentina", "home_score": 2, "away_score": 1},
        {"date": "2025-11-10", "home_team": "Argentina", "away_team": "Brazil", "home_score": 0, "away_score": 0},
        {"date": "2024-07-10", "home_team": "Brazil", "away_team": "Argentina", "home_score": 1, "away_score": 0},
    ]
    assert result["home_form"] == {"team": "Brazil", "wins": 2, "draws": 1, "losses": 0, "goals": 3}
    assert result["away_form"] == {"team": "Argentina", "wins": 0, "draws": 1, "losses": 2, "goals": 1}


def test_get_team_vs_confed_returns_aggregate_record(monkeypatch):
    service = _load_service()
    confed_df = pd.DataFrame(
        [
            {
                "date": "2026-03-10",
                "home_team": "Brazil",
                "away_team": "Spain",
                "home_score": 2,
                "away_score": 0,
                "tournament": "Friendly",
            },
            {
                "date": "2025-06-01",
                "home_team": "France",
                "away_team": "Brazil",
                "home_score": 1,
                "away_score": 1,
                "tournament": "Friendly",
            },
        ]
    )

    monkeypatch.setattr(service, "_fetch_team_vs_confed_matches", lambda *_args, **_kwargs: confed_df)

    result = service.get_team_vs_confed("Brazil", "uefa")

    assert result == {
        "team": "Brazil",
        "opponent_confederation": "UEFA",
        "matches_count": 2,
        "wins": 1,
        "draws": 1,
        "losses": 0,
        "goals_for": 3,
        "goals_against": 1,
    }


def test_get_recent_matches_ignores_incomplete_future_rows(monkeypatch):
    service = _load_service()

    def fake_fetch(team, last_matches, mode="national", request_id="-"):
        assert mode == "national"
        if team == "Brazil":
            return pd.DataFrame(
                [
                    {"date": "2026-06-13", "home_team": "Brazil", "away_team": "Morocco", "home_score": None, "away_score": None},
                    {"date": "2026-03-31", "home_team": "Brazil", "away_team": "Croatia", "home_score": 3, "away_score": 1},
                ]
            )
        return pd.DataFrame(
            [
                {"date": "2026-06-16", "home_team": "Argentina", "away_team": "Algeria", "home_score": None, "away_score": None},
                {"date": "2026-03-26", "home_team": "Brazil", "away_team": "Argentina", "home_score": 1, "away_score": 2},
            ]
        )

    monkeypatch.setattr(service, "_fetch_recent_team_matches", fake_fetch)

    result = service.get_recent_matches("Brazil", "Argentina", last_matches=2)

    assert result["home_matches"] == [
        {"date": "2026-03-31", "home_team": "Brazil", "away_team": "Croatia", "home_score": 3, "away_score": 1},
    ]
    assert result["away_matches"] == [
        {"date": "2026-03-26", "home_team": "Brazil", "away_team": "Argentina", "home_score": 1, "away_score": 2},
    ]


def test_get_recent_matches_libertadores_uses_libertadores_mode(monkeypatch):
    service = _load_service()

    def fake_fetch(team, last_matches, mode="national", request_id="-"):
        assert last_matches == 2
        assert mode == "libertadores"
        if team == "Palmeiras":
            return pd.DataFrame(
                [
                    {"date": "2026-04-08", "home_team": "Palmeiras", "away_team": "Sporting Cristal", "home_score": 2, "away_score": 0},
                    {"date": "2026-04-03", "home_team": "Palmeiras", "away_team": "Santos", "home_score": 1, "away_score": 1},
                ]
            )
        return pd.DataFrame(
            [
                {"date": "2026-04-09", "home_team": "River Plate", "away_team": "Universitario", "home_score": 3, "away_score": 1},
                {"date": "2026-04-05", "home_team": "Alianza Lima", "away_team": "Universitario", "home_score": 0, "away_score": 2},
            ]
        )

    monkeypatch.setattr(service, "_fetch_recent_team_matches", fake_fetch)

    result = service.get_recent_matches("Palmeiras", "Universitario", last_matches=2, mode="libertadores")

    assert result == {
        "home_team": "Palmeiras",
        "away_team": "Universitario",
        "home_matches": [
            {"date": "2026-04-08", "home_team": "Palmeiras", "away_team": "Sporting Cristal", "home_score": 2, "away_score": 0},
            {"date": "2026-04-03", "home_team": "Palmeiras", "away_team": "Santos", "home_score": 1, "away_score": 1},
        ],
        "away_matches": [
            {"date": "2026-04-09", "home_team": "River Plate", "away_team": "Universitario", "home_score": 3, "away_score": 1},
            {"date": "2026-04-05", "home_team": "Alianza Lima", "away_team": "Universitario", "home_score": 0, "away_score": 2},
        ],
    }


def test_fetch_recent_team_matches_uses_national_aliases_and_dedupes(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(service, "_matches_table", lambda: FakeQuery())
    monkeypatch.setattr(
        service,
        "_national_team_name_variants",
        lambda team: ["DR Congo", "Congo DR", "RD del Congo"],
    )

    def fake_execute(_query, *, request_id, operation, context=None, **_kwargs):
        query_team = context["query_team"]
        if operation == "recent_team_matches_home" and query_team == "DR Congo":
            return [
                {
                    "match_date": "2026-03-31",
                    "home_team": "DR Congo",
                    "away_team": "Jamaica",
                    "home_score": 1,
                    "away_score": 0,
                }
            ]
        if operation == "recent_team_matches_home" and query_team == "Congo DR":
            return [
                {
                    "match_date": "2026-03-31",
                    "home_team": "Congo DR",
                    "away_team": "Jamaica",
                    "home_score": 1,
                    "away_score": 0,
                },
                {
                    "match_date": "2025-12-23",
                    "home_team": "Congo DR",
                    "away_team": "Benin",
                    "home_score": 1,
                    "away_score": 0,
                },
            ]
        if operation == "recent_team_matches_away" and query_team == "Congo DR":
            return [
                {
                    "match_date": "2026-03-25",
                    "home_team": "Bermuda",
                    "away_team": "Congo DR",
                    "home_score": 0,
                    "away_score": 2,
                },
                {
                    "match_date": "2026-03-31",
                    "home_team": "Jamaica",
                    "away_team": "Congo DR",
                    "home_score": 0,
                    "away_score": 1,
                }
            ]
        return []

    monkeypatch.setattr(service, "_execute_query", fake_execute)

    result = service._fetch_recent_team_matches("DR Congo", last_matches=3)

    assert result[["date", "home_team", "away_team", "home_score", "away_score"]].to_dict(orient="records") == [
        {
            "date": pd.Timestamp("2026-03-31"),
            "home_team": "DR Congo",
            "away_team": "Jamaica",
            "home_score": 1,
            "away_score": 0,
        },
        {
            "date": pd.Timestamp("2026-03-25"),
            "home_team": "Bermuda",
            "away_team": "Congo DR",
            "home_score": 0,
            "away_score": 2,
        },
        {
            "date": pd.Timestamp("2025-12-23"),
            "home_team": "Congo DR",
            "away_team": "Benin",
            "home_score": 1,
            "away_score": 0,
        },
    ]


def test_fetch_head_to_head_matches_uses_national_aliases_and_dedupes(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(service, "_matches_table", lambda: FakeQuery())
    monkeypatch.setattr(
        service,
        "_national_team_name_variants",
        lambda team: (
            ["RD del Congo", "RD Congo", "Congo DR", "DR Congo"]
            if team == "RD del Congo"
            else [team]
        ),
    )

    observed_pairs = []

    def fake_execute(query, *, request_id, operation, context=None, **_kwargs):
        filters = {(column, value) for _, column, value in query.filters}
        home_team = next(value for op, column, value in query.filters if op == "eq" and column == "home_team")
        away_team = next(value for op, column, value in query.filters if op == "eq" and column == "away_team")
        observed_pairs.append((home_team, away_team))
        assert ("tournament", ("Friendly",)) in filters
        if (home_team, away_team) == ("RD del Congo", "Jamaica"):
            return [
                {
                    "match_date": "2026-03-31",
                    "home_team": "RD del Congo",
                    "away_team": "Jamaica",
                    "home_score": 1,
                    "away_score": 0,
                    "tournament": "Friendly",
                }
            ]
        if (home_team, away_team) == ("Congo DR", "Jamaica"):
            return [
                {
                    "match_date": "2026-03-31",
                    "home_team": "Congo DR",
                    "away_team": "Jamaica",
                    "home_score": 1,
                    "away_score": 0,
                    "tournament": "Friendly",
                }
            ]
        if (home_team, away_team) == ("Jamaica", "DR Congo"):
            return [
                {
                    "match_date": "2025-03-25",
                    "home_team": "Jamaica",
                    "away_team": "DR Congo",
                    "home_score": 2,
                    "away_score": 2,
                    "tournament": "Friendly",
                }
            ]
        return []

    monkeypatch.setattr(service, "_execute_query", fake_execute)

    result = service._fetch_head_to_head_matches("RD del Congo", "Jamaica", ["Friendly"])

    assert ("Congo DR", "Jamaica") in observed_pairs
    assert ("Jamaica", "RD Congo") in observed_pairs
    assert result[["date", "home_team", "away_team", "home_score", "away_score"]].to_dict(orient="records") == [
        {
            "date": pd.Timestamp("2026-03-31"),
            "home_team": "RD del Congo",
            "away_team": "Jamaica",
            "home_score": 1,
            "away_score": 0,
        },
        {
            "date": pd.Timestamp("2025-03-25"),
            "home_team": "Jamaica",
            "away_team": "DR Congo",
            "home_score": 2,
            "away_score": 2,
        },
    ]


def test_get_head_to_head_counts_alias_rows_from_requested_team_perspective(monkeypatch):
    service = _load_service()
    h2h_df = pd.DataFrame(
        [
            {
                "date": "2026-03-31",
                "home_team": "Congo DR",
                "away_team": "Jamaica",
                "home_score": 1,
                "away_score": 0,
                "tournament": "Friendly",
            },
            {
                "date": "2025-03-25",
                "home_team": "Jamaica",
                "away_team": "DR Congo",
                "home_score": 2,
                "away_score": 2,
                "tournament": "Friendly",
            },
        ]
    )
    monkeypatch.setattr(
        service,
        "_national_team_name_variants",
        lambda team: (
            ["RD del Congo", "RD Congo", "Congo DR", "DR Congo"]
            if team == "RD del Congo"
            else [team]
        ),
    )
    monkeypatch.setattr(service, "_fetch_head_to_head_matches", lambda *_args, **_kwargs: h2h_df)

    result = service.get_head_to_head("RD del Congo", "Jamaica", tournaments=["Friendly"])

    assert result["home_form"] == {"team": "RD del Congo", "wins": 1, "draws": 1, "losses": 0, "goals": 3}
    assert result["away_form"] == {"team": "Jamaica", "wins": 0, "draws": 1, "losses": 1, "goals": 2}


def test_fetch_team_vs_confed_matches_uses_national_aliases_and_dedupes(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(service, "_matches_table", lambda: FakeQuery())
    monkeypatch.setattr(
        service,
        "_national_team_name_variants",
        lambda team: ["RD del Congo", "RD Congo", "Congo DR", "DR Congo"],
    )

    observed_home_queries = []
    observed_away_queries = []

    def fake_execute(query, *, request_id, operation, context=None, **_kwargs):
        filters = {(column, value) for _, column, value in query.filters}
        if operation == "team_vs_confed_home":
            query_team = next(value for op, column, value in query.filters if op == "eq" and column == "home_team")
            observed_home_queries.append(query_team)
            assert ("away_team_confederation", "CONCACAF") in filters
            if query_team in {"RD del Congo", "Congo DR"}:
                return [
                    {
                        "match_date": "2026-03-31",
                        "home_team": query_team,
                        "away_team": "Jamaica",
                        "home_score": 1,
                        "away_score": 0,
                        "tournament": "Friendly",
                    }
                ]
        if operation == "team_vs_confed_away":
            query_team = next(value for op, column, value in query.filters if op == "eq" and column == "away_team")
            observed_away_queries.append(query_team)
            assert ("home_team_confederation", "CONCACAF") in filters
            if query_team == "DR Congo":
                return [
                    {
                        "match_date": "2026-03-25",
                        "home_team": "Bermuda",
                        "away_team": "DR Congo",
                        "home_score": 0,
                        "away_score": 2,
                        "tournament": "Friendly",
                    }
                ]
        return []

    monkeypatch.setattr(service, "_execute_query", fake_execute)

    result = service._fetch_team_vs_confed_matches("RD del Congo", "concacaf")

    assert "Congo DR" in observed_home_queries
    assert "RD Congo" in observed_away_queries
    assert result[["date", "home_team", "away_team", "home_score", "away_score"]].to_dict(orient="records") == [
        {
            "date": pd.Timestamp("2026-03-31"),
            "home_team": "RD del Congo",
            "away_team": "Jamaica",
            "home_score": 1,
            "away_score": 0,
        },
        {
            "date": pd.Timestamp("2026-03-25"),
            "home_team": "Bermuda",
            "away_team": "DR Congo",
            "home_score": 0,
            "away_score": 2,
        },
    ]


def test_team_vs_confed_returns_empty_when_supabase_has_no_rows(monkeypatch):
    service = _load_service()
    monkeypatch.setattr(service, "_matches_table", lambda: FakeQuery())
    monkeypatch.setattr(
        service,
        "_national_team_name_variants",
        lambda _team: ["RD del Congo", "RD Congo", "Congo DR", "DR Congo"],
    )
    monkeypatch.setattr(service, "_fetch_paginated_rows", lambda *_args, **_kwargs: [])

    result = service._fetch_team_vs_confed_matches(
        "RD del Congo",
        "CONCACAF",
    )

    assert result.empty


def test_get_team_vs_confed_ignores_incomplete_rows(monkeypatch):
    service = _load_service()
    confed_df = pd.DataFrame(
        [
            {
                "date": "2026-06-24",
                "home_team": "Scotland",
                "away_team": "Brazil",
                "home_score": None,
                "away_score": None,
                "tournament": "FIFA World Cup",
            },
            {
                "date": "2026-03-31",
                "home_team": "Brazil",
                "away_team": "Croatia",
                "home_score": 3,
                "away_score": 1,
                "tournament": "Friendly",
            },
        ]
    )

    monkeypatch.setattr(service, "_fetch_team_vs_confed_matches", lambda *_args, **_kwargs: confed_df)

    result = service.get_team_vs_confed("Brazil", "uefa")

    assert result == {
        "team": "Brazil",
        "opponent_confederation": "UEFA",
        "matches_count": 1,
        "wins": 1,
        "draws": 0,
        "losses": 0,
        "goals_for": 3,
        "goals_against": 1,
    }
