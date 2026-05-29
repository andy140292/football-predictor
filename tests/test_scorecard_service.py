import importlib

import pandas as pd


def _load_service():
    import backend.scorecard_service as scorecard_service
    return importlib.reload(scorecard_service)


def test_derive_actual_outcome_maps_scores():
    service = _load_service()

    assert service.derive_actual_outcome(2, 1) == "home_win"
    assert service.derive_actual_outcome(1, 2) == "away_win"
    assert service.derive_actual_outcome(0, 0) == "draw"


def test_build_consensus_averages_model_probabilities():
    service = _load_service()

    consensus = service.build_consensus_from_model_probs(
        {
            "random_forest": {"home_win": 0.60, "draw": 0.20, "away_win": 0.20},
            "logistic_regression": {"home_win": 0.30, "draw": 0.40, "away_win": 0.30},
            "mlp": {"home_win": 0.45, "draw": 0.15, "away_win": 0.40},
        }
    )

    assert round(consensus["consensus_prob_home_win"], 6) == round((0.60 + 0.30 + 0.45) / 3.0, 6)
    assert round(consensus["consensus_prob_draw"], 6) == round((0.20 + 0.40 + 0.15) / 3.0, 6)
    assert round(consensus["consensus_prob_away_win"], 6) == round((0.20 + 0.30 + 0.40) / 3.0, 6)
    assert consensus["consensus_predicted_outcome"] == "home_win"


def test_get_model_scorecard_counts_correct_and_incorrect(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(service, "_fetch_model_scorecard_snapshot", lambda **_: None)
    monkeypatch.setattr(
        service,
        "_fetch_evaluation_rows",
        lambda **_: (
            [
                {"is_correct": True},
                {"is_correct": False},
                {"is_correct": True},
            ],
            3,
        ),
    )

    summary = service.get_model_scorecard(mode="national", model_version="2026_01_national_v1")

    assert summary["correct_count"] == 2
    assert summary["incorrect_count"] == 1
    assert summary["total_scored"] == 3
    assert summary["accuracy_pct"] == 66.6667


def test_get_model_scorecard_uses_snapshot_when_available(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(
        service,
        "_fetch_model_scorecard_snapshot",
        lambda **_: {
            "mode": "national",
            "model_version": "2026_01_national_v1",
            "period_start": "2025-09-01",
            "period_end": "2026-01-31",
            "correct_count": 10,
            "incorrect_count": 5,
            "total_scored": 15,
            "accuracy_pct": 66.6667,
        },
    )
    monkeypatch.setattr(
        service,
        "_fetch_evaluation_rows",
        lambda **_: (_ for _ in ()).throw(AssertionError("evaluation rows should not be fetched when snapshot exists")),
    )

    summary = service.get_model_scorecard(
        mode="national",
        model_version="2026_01_national_v1",
        from_date="2025-09-01",
        to_date="2026-01-31",
    )

    assert summary["correct_count"] == 10
    assert summary["incorrect_count"] == 5
    assert summary["total_scored"] == 15
    assert summary["accuracy_pct"] == 66.6667


def test_get_model_scorecard_uses_env_model_version_when_missing(monkeypatch):
    service = _load_service()

    monkeypatch.setenv("MODEL_VERSION", "2026_01_national_v1")
    monkeypatch.setattr(service, "_fetch_model_scorecard_snapshot", lambda **_: None)
    monkeypatch.setattr(
        service,
        "_fetch_evaluation_rows",
        lambda **kwargs: (
            [
                {"is_correct": True},
                {"is_correct": False},
            ],
            2,
        ),
    )

    summary = service.get_model_scorecard(mode="national", model_version="", from_date="2025-09-01", to_date="2025-09-30")

    assert summary["model_version"] == "2026_01_national_v1"
    assert summary["total_scored"] == 2


def test_list_model_scorecard_matches_validates_verdict(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(service, "_fetch_evaluation_rows", lambda **_: ([], 0))

    raised = False
    try:
        service.list_model_scorecard_matches(
            mode="national",
            model_version="2026_01_national_v1",
            verdict="unknown",
        )
    except ValueError:
        raised = True

    assert raised is True


def test_load_matches_for_backfill_reads_from_matches_table(monkeypatch):
    service = _load_service()

    rows = [
        {
            "match_date": "2026-03-30",
            "home_team": "Brazil",
            "away_team": "Argentina",
            "home_score": 2,
            "away_score": 1,
            "tournament": "Friendly",
        },
        {
            "match_date": "2026-03-31",
            "home_team": "Spain",
            "away_team": "France",
            "home_score": 1,
            "away_score": 1,
            "tournament": "UEFA Nations League",
        },
        {
            "match_date": "2026-03-31",
            "home_team": "Club A",
            "away_team": "Club B",
            "home_score": 3,
            "away_score": 2,
            "tournament": "Club Friendly",
        },
    ]

    class FakeExecuteResult:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, data):
            self._data = list(data)
            self._gte = None
            self._lte = None
            self._tournaments = None
            self._range = (0, len(self._data) - 1)

        def select(self, _columns):
            return self

        def gte(self, field, value):
            assert field == "match_date"
            self._gte = value
            return self

        def lte(self, field, value):
            assert field == "match_date"
            self._lte = value
            return self

        def in_(self, field, values):
            assert field == "tournament"
            self._tournaments = set(values)
            return self

        def order(self, _field):
            return self

        def range(self, start, end):
            self._range = (start, end)
            return self

        def execute(self):
            filtered = list(self._data)
            if self._gte is not None:
                filtered = [row for row in filtered if row["match_date"] >= self._gte]
            if self._lte is not None:
                filtered = [row for row in filtered if row["match_date"] <= self._lte]
            if self._tournaments is not None:
                filtered = [row for row in filtered if row["tournament"] in self._tournaments]
            start, end = self._range
            return FakeExecuteResult(filtered[start : end + 1])

    class FakeClient:
        def table(self, name):
            assert name == "matches"
            return FakeQuery(rows)

    monkeypatch.setattr(service, "_service_role_client", lambda: FakeClient())

    matches_df = service.load_matches_for_backfill(
        from_date="2026-03-30",
        to_date="2026-03-31",
        tournaments={"Friendly", "UEFA Nations League"},
        chunk_size=1,
    )

    assert matches_df.to_dict(orient="records") == [
        {
            "match_date": "2026-03-30",
            "home_team": "Brazil",
            "away_team": "Argentina",
            "home_score": 2,
            "away_score": 1,
            "tournament": "Friendly",
        },
        {
            "match_date": "2026-03-31",
            "home_team": "Spain",
            "away_team": "France",
            "home_score": 1,
            "away_score": 1,
            "tournament": "UEFA Nations League",
        },
    ]


def test_upsert_match_results_reconciles_calendar_names_by_team_code(monkeypatch):
    service = _load_service()
    captured = {}

    matches_df = pd.DataFrame(
        [
            {
                "match_date": "2026-03-31",
                "home_team": "Netherlands",
                "away_team": "Ecuador",
                "home_score": 1,
                "away_score": 1,
                "tournament": "Friendly",
            }
        ]
    )

    monkeypatch.setattr(
        service,
        "_fetch_calendar_rows_for_backfill",
        lambda from_date, to_date, chunk_size=1000: [
            {
                "match_id": "abc",
                "match_date": "2026-03-31",
                "home_team": "Paises Bajos",
                "away_team": "Ecuador",
                "home_team_code": "NED",
                "away_team_code": "ECU",
            }
        ],
    )
    monkeypatch.setattr(service, "_resolve_team_code", lambda team: {"Netherlands": "NED", "Ecuador": "ECU"}.get(team, ""))

    class FakeTable:
        def upsert(self, rows, on_conflict):
            captured["rows"] = rows
            captured["on_conflict"] = on_conflict
            return self

        def execute(self):
            return type("Result", (), {"data": []})()

    class FakeClient:
        def table(self, name):
            assert name == "matches_calendar"
            return FakeTable()

    monkeypatch.setattr(service, "_service_role_client", lambda: FakeClient())

    summary = service.upsert_match_results(matches_df, result_source="supabase.matches")

    assert summary == {"upserted": 1}
    assert captured["on_conflict"] == "home_team,away_team,match_date"
    assert captured["rows"] == [
        {
            "home_team": "Paises Bajos",
            "away_team": "Ecuador",
            "match_date": "2026-03-31",
            "home_score": 1,
            "away_score": 1,
            "actual_outcome": "draw",
            "tournament": "Friendly",
            "result_source": "supabase.matches",
            "result_updated_at": captured["rows"][0]["result_updated_at"],
        }
    ]


def test_transform_result_for_canonical_flips_reversed_scores(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(
        service,
        "_resolve_team_code",
        lambda team: {
            "Jamaica": "JAM",
            "DR Congo": "COD",
        }.get(team, ""),
    )

    canonical = {
        "match_id": "canonical",
        "match_date": "2026-03-31",
        "home_team": "Jamaica",
        "away_team": "DR Congo",
        "home_team_code": "JAM",
        "away_team_code": "COD",
    }
    duplicate = {
        "match_id": "duplicate",
        "match_date": "2026-03-31",
        "home_team": "DR Congo",
        "away_team": "Jamaica",
        "home_score": 1,
        "away_score": 0,
        "actual_outcome": "home_win",
        "result_source": "supabase.matches",
        "result_updated_at": "2026-04-01T00:00:00+00:00",
    }

    transformed = service._transform_result_for_canonical(duplicate, canonical)

    assert transformed["home_score"] == 0
    assert transformed["away_score"] == 1
    assert transformed["actual_outcome"] == "away_win"
    assert transformed["home_team_code"] == "JAM"
    assert transformed["away_team_code"] == "COD"


def test_prepare_prediction_merge_plan_keeps_latest_and_flips_outcome(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(
        service,
        "_resolve_team_code",
        lambda team: {
            "Jamaica": "JAM",
            "DR Congo": "COD",
        }.get(team, ""),
    )

    canonical = {
        "match_id": "canonical",
        "match_date": "2026-03-31",
        "home_team": "Jamaica",
        "away_team": "DR Congo",
        "home_team_code": "JAM",
        "away_team_code": "COD",
    }
    predictions = [
        {
            "prediction_id": "old-canonical",
            "match_id": "canonical",
            "user_id": "user-1",
            "email": "user@example.com",
            "predicted_outcome": "draw",
            "created_at": "2026-03-30T10:00:00+00:00",
        },
        {
            "prediction_id": "new-duplicate",
            "match_id": "duplicate",
            "user_id": "user-1",
            "email": "user@example.com",
            "predicted_outcome": "home_win",
            "created_at": "2026-03-30T12:00:00+00:00",
        },
    ]

    updates, deletes = service._prepare_prediction_merge_plan(
        predictions,
        canonical,
        {
            "canonical": canonical,
            "duplicate": {
                "match_id": "duplicate",
                "match_date": "2026-03-31",
                "home_team": "DR Congo",
                "away_team": "Jamaica",
            },
        },
    )

    assert updates == [
        {
            "prediction_id": "new-duplicate",
            "payload": {
                "match_id": "canonical",
                "predicted_outcome": "away_win",
                "email": "user@example.com",
                "user_id": "user-1",
            },
        }
    ]
    assert deletes == ["old-canonical"]


def test_prepare_evaluation_merge_plan_flips_reversed_probabilities(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(
        service,
        "_resolve_team_code",
        lambda team: {
            "Bolivia": "BOL",
            "Iraq": "IRQ",
        }.get(team, ""),
    )

    canonical = {
        "match_id": "canonical",
        "match_date": "2026-03-31",
        "home_team": "Bolivia",
        "away_team": "Iraq",
        "home_team_code": "BOL",
        "away_team_code": "IRQ",
        "tournament": "FIFA World Cup qualification",
    }
    evaluations = [
        {
            "evaluation_id": "dup-eval",
            "match_id": "duplicate",
            "mode": "national",
            "model_version": "2026_01_national_v1",
            "match_date": "2026-03-31",
            "home_team": "Iraq",
            "away_team": "Bolivia",
            "tournament": "FIFA World Cup qualification",
            "actual_outcome": "home_win",
            "consensus_predicted_outcome": "away_win",
            "consensus_prob_home_win": 0.6,
            "consensus_prob_draw": 0.2,
            "consensus_prob_away_win": 0.2,
            "is_correct": False,
            "created_at": "2026-04-01T00:00:00+00:00",
        }
    ]

    updates, deletes = service._prepare_evaluation_merge_plan(evaluations, canonical)

    assert updates == [
        {
            "evaluation_id": "dup-eval",
            "payload": {
                "match_id": "canonical",
                "match_date": "2026-03-31",
                "home_team": "Bolivia",
                "away_team": "Iraq",
                "tournament": "FIFA World Cup qualification",
                "actual_outcome": "away_win",
                "consensus_predicted_outcome": "home_win",
                "consensus_prob_home_win": 0.2,
                "consensus_prob_draw": 0.2,
                "consensus_prob_away_win": 0.6,
                "is_correct": False,
            },
        }
    ]
    assert deletes == []


def test_list_prediction_rankings_counts_only_resolved_rows(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(
        service,
        "_fetch_prediction_ranking_rows",
        lambda: [
            {
                "user_id": "user-1",
                "email": "ana@example.com",
                "predicted_outcome": "home_win",
                "actual_outcome": "home_win",
                "mode": "national",
            },
            {
                "user_id": "user-1",
                "email": "ana@example.com",
                "predicted_outcome": "draw",
                "actual_outcome": "away_win",
                "mode": "national",
            },
            {
                "user_id": "user-2",
                "email": "ben@example.com",
                "predicted_outcome": "draw",
                "actual_outcome": "draw",
                "mode": "champions",
            },
        ],
    )
    monkeypatch.setattr(
        service,
        "_fetch_auth_user_display_name_map",
        lambda: {"user-1": "Ana", "user-2": "Ben"},
    )

    result = service.list_prediction_rankings(mode="national")

    assert result["mode"] == "national"
    assert result["total_users"] == 1
    assert result["rankings"] == [
        {
            "rank": 1,
            "user_id": "user-1",
            "display_name": "Ana",
            "correct_count": 1,
            "incorrect_count": 1,
            "total_resolved_predictions": 2,
            "accuracy_pct": 50.0,
        }
    ]


def test_list_prediction_rankings_sorts_and_paginates(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(
        service,
        "_fetch_prediction_ranking_rows",
        lambda: [
            {
                "user_id": "user-1",
                "email": "ana@example.com",
                "predicted_outcome": "home_win",
                "actual_outcome": "home_win",
                "mode": "national",
            },
            {
                "user_id": "user-1",
                "email": "ana@example.com",
                "predicted_outcome": "away_win",
                "actual_outcome": "draw",
                "mode": "national",
            },
            {
                "user_id": "user-2",
                "email": "bea@example.com",
                "predicted_outcome": "draw",
                "actual_outcome": "draw",
                "mode": "national",
            },
            {
                "user_id": "user-2",
                "email": "bea@example.com",
                "predicted_outcome": "home_win",
                "actual_outcome": "home_win",
                "mode": "national",
            },
            {
                "user_id": "user-3",
                "email": "cora@example.com",
                "predicted_outcome": "home_win",
                "actual_outcome": "away_win",
                "mode": "national",
            },
        ],
    )
    monkeypatch.setattr(
        service,
        "_fetch_auth_user_display_name_map",
        lambda: {"user-1": "Ana", "user-2": "Bea", "user-3": "Cora"},
    )

    result = service.list_prediction_rankings(
        mode="national",
        sort_by="accuracy_pct",
        sort_order="desc",
        page=2,
        page_size=1,
    )

    assert result["total_users"] == 3
    assert result["rankings"] == [
        {
            "rank": 2,
            "user_id": "user-1",
            "display_name": "Ana",
            "correct_count": 1,
            "incorrect_count": 1,
            "total_resolved_predictions": 2,
            "accuracy_pct": 50.0,
        }
    ]


def test_list_prediction_rankings_falls_back_to_email_display_name(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(
        service,
        "_fetch_prediction_ranking_rows",
        lambda: [
            {
                "user_id": "user-9",
                "email": "fallback@example.com",
                "predicted_outcome": "draw",
                "actual_outcome": "draw",
                "mode": "champions",
            }
        ],
    )
    monkeypatch.setattr(service, "_fetch_auth_user_display_name_map", lambda: {})

    result = service.list_prediction_rankings(mode="champions")

    assert result["rankings"][0]["display_name"] == "fallback"
    assert result["rankings"][0]["user_id"] == "user-9"


def test_list_prediction_rankings_validates_inputs(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(service, "_fetch_prediction_ranking_rows", lambda: [])

    for kwargs in (
        {"mode": "club"},
        {"mode": "national", "sort_by": "wins"},
        {"mode": "national", "sort_order": "down"},
        {"mode": "national", "page": 0},
        {"mode": "national", "page_size": 0},
    ):
        raised = False
        try:
            service.list_prediction_rankings(**kwargs)
        except ValueError:
            raised = True
        assert raised is True


def test_list_prediction_rankings_supports_all_mode(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(
        service,
        "_fetch_prediction_ranking_rows",
        lambda: [
            {
                "user_id": "user-1",
                "email": "ana@example.com",
                "predicted_outcome": "home_win",
                "actual_outcome": "home_win",
                "mode": "national",
            },
            {
                "user_id": "user-1",
                "email": "ana@example.com",
                "predicted_outcome": "draw",
                "actual_outcome": "draw",
                "mode": "champions",
            },
            {
                "user_id": "user-2",
                "email": "bea@example.com",
                "predicted_outcome": "home_win",
                "actual_outcome": "away_win",
                "mode": "champions",
            },
        ],
    )
    monkeypatch.setattr(service, "_fetch_auth_user_display_name_map", lambda: {"user-1": "Ana", "user-2": "Bea"})

    result = service.list_prediction_rankings(mode="")

    assert result["mode"] == "all"
    assert result["total_users"] == 2
    assert result["rankings"][0]["user_id"] == "user-1"
    assert result["rankings"][0]["total_resolved_predictions"] == 2


def test_list_prediction_rankings_excludes_configured_user_id(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(
        service,
        "_fetch_prediction_ranking_rows",
        lambda: [
            {
                "user_id": "6964523a-e976-4def-a369-ac5800bb313e",
                "email": "owner@example.com",
                "predicted_outcome": "home_win",
                "actual_outcome": "home_win",
                "mode": "national",
            },
            {
                "user_id": "user-2",
                "email": "bea@example.com",
                "predicted_outcome": "draw",
                "actual_outcome": "draw",
                "mode": "national",
            },
        ],
    )
    monkeypatch.setattr(service, "_fetch_auth_user_display_name_map", lambda: {"user-2": "Bea"})

    result = service.list_prediction_rankings(mode="national")

    assert result["total_users"] == 1
    assert result["rankings"][0]["user_id"] == "user-2"


def test_list_prediction_rankings_preserves_auth_display_name(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(
        service,
        "_fetch_prediction_ranking_rows",
        lambda: [
            {
                "user_id": "user-7",
                "email": "fallback@example.com",
                "predicted_outcome": "draw",
                "actual_outcome": "draw",
                "mode": "national",
            }
        ],
    )
    monkeypatch.setattr(
        service,
        "_fetch_auth_user_display_name_map",
        lambda: {"user-7": "Visible Name"},
    )

    result = service.list_prediction_rankings(mode="national")

    assert result["rankings"][0]["display_name"] == "Visible Name"


def test_list_prediction_rankings_falls_back_to_direct_auth_lookup_when_map_misses(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(
        service,
        "_fetch_prediction_ranking_rows",
        lambda: [
            {
                "user_id": "user-missed",
                "email": "fallback@example.com",
                "predicted_outcome": "draw",
                "actual_outcome": "draw",
                "mode": "national",
            }
        ],
    )
    monkeypatch.setattr(service, "_fetch_auth_user_display_name_map", lambda: {})
    monkeypatch.setattr(service, "_fetch_auth_user_display_name", lambda user_id: "José Ramón" if user_id == "user-missed" else None)

    result = service.list_prediction_rankings(mode="national")

    assert result["rankings"][0]["display_name"] == "José Ramón"


def test_list_prediction_rankings_strips_email_shaped_auth_display_name(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(
        service,
        "_fetch_prediction_ranking_rows",
        lambda: [
            {
                "user_id": "user-email",
                "email": "christheboymma@gmail.com",
                "predicted_outcome": "draw",
                "actual_outcome": "draw",
                "mode": "national",
            }
        ],
    )
    monkeypatch.setattr(
        service,
        "_fetch_auth_user_display_name_map",
        lambda: {"user-email": "christheboymma@gmail.com"},
    )

    result = service.list_prediction_rankings(mode="national")

    assert result["rankings"][0]["display_name"] == "christheboymma"


def test_run_consensus_scorecard_backfill_supports_libertadores(monkeypatch):
    service = _load_service()

    monkeypatch.setattr(service, "_resolve_model_version", lambda model_version: model_version or "2026_01_libertadores_v1")
    monkeypatch.setattr(
        service,
        "_fetch_completed_matches",
        lambda from_date, to_date, tournaments, chunk_size=1000: [
            {
                "match_id": "m1",
                "match_date": "2026-04-28",
                "home_team": "Cruzeiro",
                "away_team": "Boca Juniors",
                "tournament": "Libertadores",
                "home_score": 1,
                "away_score": 0,
                "actual_outcome": "home_win",
            }
        ]
        if tournaments == {"Libertadores"}
        else [],
    )

    captured = {}

    def fake_evaluate(matches, mode, model_version, run_id, chunk_size=300):
        captured["evaluate"] = {
            "matches": matches,
            "mode": mode,
            "model_version": model_version,
            "run_id": run_id,
        }
        return {"evaluated": len(matches), "skipped": 0}

    def fake_snapshot(run_id, mode, model_version, period_start, period_end):
        captured["snapshot"] = {
            "run_id": run_id,
            "mode": mode,
            "model_version": model_version,
            "period_start": period_start,
            "period_end": period_end,
        }
        return {"mode": mode, "model_version": model_version}

    monkeypatch.setattr(service, "evaluate_consensus_matches", fake_evaluate)
    monkeypatch.setattr(service, "upsert_model_scorecard_snapshot", fake_snapshot)
    monkeypatch.setattr(
        service,
        "load_matches_for_backfill",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("national-only loader should not run")),
    )
    monkeypatch.setattr(
        service,
        "upsert_match_results",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("national-only result upsert should not run")),
    )

    summary = service.run_consensus_scorecard_backfill(
        from_date="2026-04-28",
        to_date="2026-04-30",
        mode="libertadores",
        model_version="2026_01_libertadores_v1",
    )

    assert summary["mode"] == "libertadores"
    assert summary["model_version"] == "2026_01_libertadores_v1"
    assert summary["results_upsert"] == {"upserted": 0}
    assert summary["completed_matches"] == 1
    assert captured["evaluate"]["mode"] == "libertadores"
    assert captured["evaluate"]["model_version"] == "2026_01_libertadores_v1"
    assert captured["snapshot"]["mode"] == "libertadores"
