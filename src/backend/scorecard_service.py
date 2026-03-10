from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from uuid import uuid4

import pandas as pd

try:
    from .match_service import _FORM_TOURNAMENTS
    from .paths import MATCHES_PATH
    from .predict_match import MODEL_NAMES, predict_match_probabilities_offline
    from .supabase_client import get_supabase_client
except ImportError:  # pragma: no cover - fallback for direct module execution
    from src.backend.match_service import _FORM_TOURNAMENTS
    from src.backend.paths import MATCHES_PATH
    from src.backend.predict_match import MODEL_NAMES, predict_match_probabilities_offline
    from src.backend.supabase_client import get_supabase_client


logger = logging.getLogger("futbolconu.scorecard")


def _parse_iso_date(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def derive_actual_outcome(home_score: int, away_score: int) -> Optional[str]:
    if home_score is None or away_score is None:
        return None
    try:
        home = int(home_score)
        away = int(away_score)
    except (TypeError, ValueError):
        return None
    if home > away:
        return "home_win"
    if away > home:
        return "away_win"
    return "draw"


def build_consensus_from_model_probs(model_probabilities: dict[str, dict[str, float]]) -> dict[str, float | str]:
    available_models = [name for name in MODEL_NAMES if name in model_probabilities]
    if not available_models:
        raise ValueError("No model probabilities available to build consensus")

    home_sum = 0.0
    draw_sum = 0.0
    away_sum = 0.0
    for model_name in available_models:
        model_values = model_probabilities.get(model_name, {})
        home_sum += float(model_values.get("home_win", 0.0))
        draw_sum += float(model_values.get("draw", 0.0))
        away_sum += float(model_values.get("away_win", 0.0))

    size = float(len(available_models))
    home_prob = home_sum / size
    draw_prob = draw_sum / size
    away_prob = away_sum / size
    options = {
        "home_win": home_prob,
        "draw": draw_prob,
        "away_win": away_prob,
    }
    predicted_outcome = max(options.items(), key=lambda item: item[1])[0]
    return {
        "consensus_prob_home_win": home_prob,
        "consensus_prob_draw": draw_prob,
        "consensus_prob_away_win": away_prob,
        "consensus_predicted_outcome": predicted_outcome,
    }


def _chunks(items: list[dict], size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _service_role_client():
    return get_supabase_client(use_service_role=True)


def load_matches_for_backfill(
    file_path: Optional[str],
    from_date: str,
    to_date: str,
    tournaments: Optional[set[str]] = None,
) -> pd.DataFrame:
    csv_path = Path(file_path) if file_path else Path(MATCHES_PATH)
    frame = pd.read_csv(csv_path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    start = pd.Timestamp(from_date)
    end = pd.Timestamp(to_date)
    filtered = frame[(frame["date"] >= start) & (frame["date"] <= end)].copy()
    if tournaments:
        filtered = filtered[filtered["tournament"].isin(tournaments)].copy()
    filtered = filtered.dropna(
        subset=["date", "home_team", "away_team", "home_score", "away_score"]
    ).copy()
    filtered["match_date"] = filtered["date"].dt.strftime("%Y-%m-%d")
    filtered["home_score"] = pd.to_numeric(filtered["home_score"], errors="coerce")
    filtered["away_score"] = pd.to_numeric(filtered["away_score"], errors="coerce")
    filtered = filtered.dropna(subset=["home_score", "away_score"]).copy()
    filtered["home_score"] = filtered["home_score"].astype(int)
    filtered["away_score"] = filtered["away_score"].astype(int)
    return filtered


def upsert_match_results(
    matches_df: pd.DataFrame,
    result_source: str,
    chunk_size: int = 500,
) -> dict:
    if matches_df.empty:
        return {"upserted": 0}

    now_iso = datetime.utcnow().isoformat()
    payload = []
    for row in matches_df.itertuples(index=False):
        actual_outcome = derive_actual_outcome(getattr(row, "home_score"), getattr(row, "away_score"))
        if actual_outcome is None:
            continue
        payload.append(
            {
                "home_team": str(getattr(row, "home_team")),
                "away_team": str(getattr(row, "away_team")),
                "match_date": str(getattr(row, "match_date")),
                "home_score": int(getattr(row, "home_score")),
                "away_score": int(getattr(row, "away_score")),
                "actual_outcome": actual_outcome,
                "tournament": str(getattr(row, "tournament", "") or ""),
                "result_source": result_source,
                "result_updated_at": now_iso,
            }
        )

    if not payload:
        return {"upserted": 0}

    client = _service_role_client()
    for batch in _chunks(payload, chunk_size):
        client.table("matches_calendar").upsert(
            batch,
            on_conflict="home_team,away_team,match_date",
        ).execute()
    return {"upserted": len(payload)}


def _fetch_completed_matches(
    from_date: str,
    to_date: str,
    tournaments: Optional[set[str]],
    chunk_size: int = 1000,
) -> list[dict]:
    client = _service_role_client()
    rows: list[dict] = []
    offset = 0
    select_cols = (
        "match_id,home_team,away_team,match_date,tournament,home_score,away_score,actual_outcome"
    )
    while True:
        query = (
            client.table("matches_calendar")
            .select(select_cols)
            .gte("match_date", from_date)
            .lte("match_date", to_date)
            .order("match_date")
            .range(offset, offset + chunk_size - 1)
        )
        batch = query.execute().data or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < chunk_size:
            break
        offset += chunk_size

    completed = []
    for row in rows:
        if tournaments and row.get("tournament") not in tournaments:
            continue
        if row.get("home_score") is None or row.get("away_score") is None:
            continue
        actual = row.get("actual_outcome") or derive_actual_outcome(row.get("home_score"), row.get("away_score"))
        if actual not in {"home_win", "away_win", "draw"}:
            continue
        if not row.get("match_id"):
            continue
        normalized = dict(row)
        normalized["actual_outcome"] = actual
        completed.append(normalized)
    return completed


def evaluate_consensus_matches(
    matches: list[dict],
    mode: str,
    model_version: str,
    run_id: str,
    chunk_size: int = 300,
) -> dict:
    if not matches:
        return {"evaluated": 0, "skipped": 0}

    payload: list[dict] = []
    skipped = 0
    for row in matches:
        try:
            probabilities = predict_match_probabilities_offline(
                home_team=str(row.get("home_team")),
                away_team=str(row.get("away_team")),
                mode=mode,
                request_id=f"scorecard-{run_id}",
            )
            consensus = build_consensus_from_model_probs(probabilities)
        except Exception as exc:
            skipped += 1
            logger.warning(
                "consensus_evaluation_skipped match_id=%s home_team=%s away_team=%s error=%s",
                row.get("match_id"),
                row.get("home_team"),
                row.get("away_team"),
                exc,
            )
            continue

        actual_outcome = str(row.get("actual_outcome"))
        predicted = str(consensus["consensus_predicted_outcome"])
        payload.append(
            {
                "run_id": run_id,
                "mode": mode,
                "model_version": model_version,
                "match_id": row.get("match_id"),
                "match_date": row.get("match_date"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "tournament": row.get("tournament"),
                "actual_outcome": actual_outcome,
                "consensus_predicted_outcome": predicted,
                "consensus_prob_home_win": float(consensus["consensus_prob_home_win"]),
                "consensus_prob_draw": float(consensus["consensus_prob_draw"]),
                "consensus_prob_away_win": float(consensus["consensus_prob_away_win"]),
                "is_correct": predicted == actual_outcome,
            }
        )

    if not payload:
        return {"evaluated": 0, "skipped": skipped}

    client = _service_role_client()
    for batch in _chunks(payload, chunk_size):
        client.table("consensus_match_evaluations").upsert(
            batch,
            on_conflict="mode,model_version,match_id",
        ).execute()

    return {"evaluated": len(payload), "skipped": skipped}


def _compute_accuracy(correct_count: int, total_scored: int) -> float:
    if total_scored <= 0:
        return 0.0
    return round((float(correct_count) / float(total_scored)) * 100.0, 4)


def _fetch_evaluation_rows(
    mode: str,
    model_version: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    verdict: str = "all",
    page: Optional[int] = None,
    page_size: Optional[int] = None,
) -> tuple[list[dict], Optional[int]]:
    client = _service_role_client()
    select_cols = (
        "match_id,match_date,home_team,away_team,tournament,actual_outcome,"
        "consensus_predicted_outcome,consensus_prob_home_win,consensus_prob_draw,"
        "consensus_prob_away_win,is_correct"
    )
    query = (
        client.table("consensus_match_evaluations")
        .select(select_cols, count="exact")
        .eq("mode", mode)
        .eq("model_version", model_version)
        .order("match_date", desc=True)
    )
    if from_date:
        query = query.gte("match_date", from_date)
    if to_date:
        query = query.lte("match_date", to_date)
    if verdict == "correct":
        query = query.eq("is_correct", True)
    elif verdict == "incorrect":
        query = query.eq("is_correct", False)
    if page is not None and page_size is not None:
        start = max(page - 1, 0) * page_size
        end = start + page_size - 1
        query = query.range(start, end)

    result = query.execute()
    rows = result.data or []
    return rows, int(result.count or 0)


def _fetch_model_scorecard_snapshot(
    mode: str,
    model_version: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> Optional[dict]:
    if not from_date or not to_date:
        return None
    client = _service_role_client()
    result = (
        client.table("model_scorecard_snapshots")
        .select(
            "mode,model_version,period_start,period_end,correct_count,incorrect_count,total_scored,accuracy_pct"
        )
        .eq("mode", mode)
        .eq("model_version", model_version)
        .eq("period_start", from_date)
        .eq("period_end", to_date)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return rows[0] if rows else None


def get_model_scorecard(
    mode: str,
    model_version: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict:
    if not str(model_version or "").strip():
        raise ValueError("model_version is required")
    normalized_from = _parse_iso_date(from_date)
    normalized_to = _parse_iso_date(to_date)

    snapshot = _fetch_model_scorecard_snapshot(
        mode=mode,
        model_version=model_version,
        from_date=normalized_from,
        to_date=normalized_to,
    )
    if snapshot:
        return {
            "mode": snapshot.get("mode") or mode,
            "model_version": snapshot.get("model_version") or model_version,
            "from_date": snapshot.get("period_start") or normalized_from,
            "to_date": snapshot.get("period_end") or normalized_to,
            "correct_count": int(snapshot.get("correct_count") or 0),
            "incorrect_count": int(snapshot.get("incorrect_count") or 0),
            "total_scored": int(snapshot.get("total_scored") or 0),
            "accuracy_pct": float(snapshot.get("accuracy_pct") or 0.0),
        }

    rows, _ = _fetch_evaluation_rows(
        mode=mode,
        model_version=model_version,
        from_date=normalized_from,
        to_date=normalized_to,
        verdict="all",
    )
    total = len(rows)
    correct = sum(1 for row in rows if bool(row.get("is_correct")))
    incorrect = total - correct
    return {
        "mode": mode,
        "model_version": model_version,
        "from_date": normalized_from,
        "to_date": normalized_to,
        "correct_count": correct,
        "incorrect_count": incorrect,
        "total_scored": total,
        "accuracy_pct": _compute_accuracy(correct, total),
    }


def list_model_scorecard_matches(
    mode: str,
    model_version: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    verdict: str = "all",
    page: int = 1,
    page_size: int = 50,
) -> dict:
    if not str(model_version or "").strip():
        raise ValueError("model_version is required")
    if verdict not in {"all", "correct", "incorrect"}:
        raise ValueError("verdict must be one of: all, correct, incorrect")
    normalized_from = _parse_iso_date(from_date)
    normalized_to = _parse_iso_date(to_date)
    page = max(int(page), 1)
    page_size = max(min(int(page_size), 200), 1)
    rows, total = _fetch_evaluation_rows(
        mode=mode,
        model_version=model_version,
        from_date=normalized_from,
        to_date=normalized_to,
        verdict=verdict,
        page=page,
        page_size=page_size,
    )
    return {
        "mode": mode,
        "model_version": model_version,
        "from_date": normalized_from,
        "to_date": normalized_to,
        "verdict": verdict,
        "page": page,
        "page_size": page_size,
        "total": total,
        "matches": rows,
    }


def upsert_model_scorecard_snapshot(
    run_id: str,
    mode: str,
    model_version: str,
    period_start: str,
    period_end: str,
) -> dict:
    summary = get_model_scorecard(
        mode=mode,
        model_version=model_version,
        from_date=period_start,
        to_date=period_end,
    )
    payload = {
        "run_id": run_id,
        "mode": mode,
        "model_version": model_version,
        "period_start": period_start,
        "period_end": period_end,
        "correct_count": summary["correct_count"],
        "incorrect_count": summary["incorrect_count"],
        "total_scored": summary["total_scored"],
        "accuracy_pct": summary["accuracy_pct"],
    }
    _service_role_client().table("model_scorecard_snapshots").upsert(
        payload,
        on_conflict="mode,model_version,period_start,period_end",
    ).execute()
    return payload


def run_consensus_scorecard_backfill(
    from_date: str,
    to_date: str,
    mode: str = "national",
    model_version: Optional[str] = None,
    matches_file_path: Optional[str] = None,
) -> dict:
    normalized_from = _parse_iso_date(from_date)
    normalized_to = _parse_iso_date(to_date)
    if not normalized_from or not normalized_to:
        raise ValueError("from_date and to_date must use YYYY-MM-DD format")
    if normalized_from > normalized_to:
        raise ValueError("from_date cannot be greater than to_date")
    if mode != "national":
        raise ValueError("Only mode='national' is currently supported by this backfill")

    resolved_model_version = str(model_version or "").strip()
    if not resolved_model_version:
        raise ValueError("model_version is required")

    run_id = str(uuid4())
    tournaments = set(_FORM_TOURNAMENTS)
    matches_df = load_matches_for_backfill(
        file_path=matches_file_path,
        from_date=normalized_from,
        to_date=normalized_to,
        tournaments=tournaments,
    )
    results_upsert = upsert_match_results(
        matches_df=matches_df,
        result_source=Path(matches_file_path).name if matches_file_path else Path(MATCHES_PATH).name,
    )
    completed_matches = _fetch_completed_matches(
        from_date=normalized_from,
        to_date=normalized_to,
        tournaments=tournaments,
    )
    evaluation_summary = evaluate_consensus_matches(
        matches=completed_matches,
        mode=mode,
        model_version=resolved_model_version,
        run_id=run_id,
    )
    snapshot = upsert_model_scorecard_snapshot(
        run_id=run_id,
        mode=mode,
        model_version=resolved_model_version,
        period_start=normalized_from,
        period_end=normalized_to,
    )
    return {
        "run_id": run_id,
        "mode": mode,
        "model_version": resolved_model_version,
        "from_date": normalized_from,
        "to_date": normalized_to,
        "results_upsert": results_upsert,
        "completed_matches": len(completed_matches),
        "evaluations": evaluation_summary,
        "snapshot": snapshot,
    }
