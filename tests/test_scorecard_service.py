import importlib


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
