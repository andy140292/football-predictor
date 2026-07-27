from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional
import unicodedata
from uuid import uuid4

import pandas as pd

try:
    from .match_service import _FORM_TOURNAMENTS
    from .paths import DATA_DIR
    from .predict_match import MODEL_NAMES, predict_match_probabilities_offline
    from .supabase_client import get_supabase_client
except ImportError:  # pragma: no cover - fallback for direct module execution
    from src.backend.match_service import _FORM_TOURNAMENTS
    from src.backend.paths import DATA_DIR
    from src.backend.predict_match import MODEL_NAMES, predict_match_probabilities_offline
    from src.backend.supabase_client import get_supabase_client


logger = logging.getLogger("futbolconu.scorecard")
_MATCH_RESULT_SOURCE = "supabase.matches"
_TEAM_CODES_PATH = DATA_DIR / "fifa_country_codes.csv"
_TEAM_ALIASES_PATH = DATA_DIR / "team_code_aliases.csv"
_PREDICTION_RANKING_MODES = {"all", "national", "champions", "libertadores", "world_cup"}
_PREDICTION_RANKING_SORT_FIELDS = {
    "correct_count",
    "incorrect_count",
    "total_resolved_predictions",
    "accuracy_pct",
}
_EXCLUDED_PREDICTION_RANKING_USER_IDS = {
    "6964523a-e976-4def-a369-ac5800bb313e",
}


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


def _resolve_model_version(model_version: Optional[str]) -> str:
    resolved = str(model_version or os.getenv("MODEL_VERSION") or "").strip()
    if not resolved:
        raise ValueError("model_version is required")
    return resolved


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


def _normalized_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch for ch in text if ch.isalnum())


def _parse_timestamp(value: Optional[str]) -> tuple[int, str]:
    text = str(value or "").strip()
    if not text:
        return (0, "")
    normalized = text.replace("Z", "+00:00")
    try:
        return (1, datetime.fromisoformat(normalized).isoformat())
    except ValueError:
        return (1, text)


def _flip_outcome(value: Optional[str]) -> Optional[str]:
    outcome = str(value or "").strip()
    if outcome == "home_win":
        return "away_win"
    if outcome == "away_win":
        return "home_win"
    if outcome == "draw":
        return "draw"
    return value


@lru_cache(maxsize=1)
def _load_team_code_maps() -> tuple[dict[str, str], dict[str, str]]:
    name_to_code: dict[str, str] = {}
    alias_to_code: dict[str, str] = {}

    if _TEAM_CODES_PATH.exists():
        with _TEAM_CODES_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                code = str(
                    row.get("team_code") or row.get("code") or row.get("fifa_code") or row.get("Code") or ""
                ).strip().upper()
                name = str(
                    row.get("country_name") or row.get("country") or row.get("Country") or row.get("team") or ""
                ).strip()
                if code and name:
                    name_to_code[_normalized_text(name)] = code

    if _TEAM_ALIASES_PATH.exists():
        with _TEAM_ALIASES_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                code = str(
                    row.get("team_code") or row.get("code") or row.get("fifa_code") or ""
                ).strip().upper()
                alias = str(
                    row.get("alias_name") or row.get("alias") or row.get("team_name") or row.get("name") or ""
                ).strip()
                if code and alias:
                    alias_to_code[_normalized_text(alias)] = code

    return name_to_code, alias_to_code


def _resolve_team_code(team_name: str) -> str:
    key = _normalized_text(team_name)
    if not key:
        return ""
    name_to_code, alias_to_code = _load_team_code_maps()
    if key in alias_to_code:
        return alias_to_code[key]
    return name_to_code.get(key, "")


def _resolved_row_codes(row: dict) -> tuple[str, str]:
    home_code = str(row.get("home_team_code") or "").strip().upper()
    away_code = str(row.get("away_team_code") or "").strip().upper()
    if not home_code:
        home_code = _resolve_team_code(str(row.get("home_team") or ""))
    if not away_code:
        away_code = _resolve_team_code(str(row.get("away_team") or ""))
    return home_code, away_code


def _calendar_duplicate_group_key(row: dict) -> tuple[str, str, str]:
    match_date = str(row.get("match_date") or "").strip()
    home_code, away_code = _resolved_row_codes(row)
    if home_code and away_code:
        team_a, team_b = sorted([home_code, away_code])
        return match_date, team_a, team_b

    home_team = _normalized_text(str(row.get("home_team") or ""))
    away_team = _normalized_text(str(row.get("away_team") or ""))
    team_a, team_b = sorted([home_team, away_team])
    return match_date, team_a, team_b


def _rows_are_reversed(left: dict, right: dict) -> bool:
    left_home_code, left_away_code = _resolved_row_codes(left)
    right_home_code, right_away_code = _resolved_row_codes(right)
    if left_home_code and left_away_code and right_home_code and right_away_code:
        return left_home_code == right_away_code and left_away_code == right_home_code

    left_home = _normalized_text(str(left.get("home_team") or ""))
    left_away = _normalized_text(str(left.get("away_team") or ""))
    right_home = _normalized_text(str(right.get("home_team") or ""))
    right_away = _normalized_text(str(right.get("away_team") or ""))
    return left_home == right_away and left_away == right_home


def _transform_result_for_canonical(source_row: dict, canonical_row: dict) -> dict:
    reverse = _rows_are_reversed(canonical_row, source_row)
    home_score = source_row.get("home_score")
    away_score = source_row.get("away_score")
    actual_outcome = source_row.get("actual_outcome")

    if reverse:
        home_score, away_score = away_score, home_score
        actual_outcome = _flip_outcome(actual_outcome)

    return {
        "home_score": home_score,
        "away_score": away_score,
        "actual_outcome": actual_outcome,
        "result_source": source_row.get("result_source"),
        "result_updated_at": source_row.get("result_updated_at"),
        "tournament": source_row.get("tournament"),
        "home_team_code": _resolved_row_codes(canonical_row)[0]
        or (_resolved_row_codes(source_row)[1] if reverse else _resolved_row_codes(source_row)[0]),
        "away_team_code": _resolved_row_codes(canonical_row)[1]
        or (_resolved_row_codes(source_row)[0] if reverse else _resolved_row_codes(source_row)[1]),
    }


def _transform_prediction_outcome_for_canonical(predicted_outcome: Optional[str], reverse: bool) -> Optional[str]:
    if not reverse:
        return predicted_outcome
    return _flip_outcome(predicted_outcome)


def _transform_evaluation_for_canonical(evaluation: dict, canonical_row: dict) -> dict:
    reverse = _rows_are_reversed(canonical_row, evaluation)
    payload = {
        "match_id": canonical_row.get("match_id"),
        "match_date": canonical_row.get("match_date") or evaluation.get("match_date"),
        "home_team": canonical_row.get("home_team") or evaluation.get("home_team"),
        "away_team": canonical_row.get("away_team") or evaluation.get("away_team"),
        "tournament": canonical_row.get("tournament") or evaluation.get("tournament"),
        "actual_outcome": evaluation.get("actual_outcome"),
        "consensus_predicted_outcome": evaluation.get("consensus_predicted_outcome"),
        "consensus_prob_home_win": evaluation.get("consensus_prob_home_win"),
        "consensus_prob_draw": evaluation.get("consensus_prob_draw"),
        "consensus_prob_away_win": evaluation.get("consensus_prob_away_win"),
        "is_correct": evaluation.get("is_correct"),
    }
    if reverse:
        payload["actual_outcome"] = _flip_outcome(payload["actual_outcome"])
        payload["consensus_predicted_outcome"] = _flip_outcome(payload["consensus_predicted_outcome"])
        payload["consensus_prob_home_win"] = evaluation.get("consensus_prob_away_win")
        payload["consensus_prob_away_win"] = evaluation.get("consensus_prob_home_win")
        payload["is_correct"] = payload["consensus_predicted_outcome"] == payload["actual_outcome"]
    return payload


def _fetch_matches_calendar_rows(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    chunk_size: int = 1000,
) -> list[dict]:
    client = _service_role_client()
    rows: list[dict] = []
    offset = 0
    select_cols = (
        "match_id,match_date,home_team,away_team,home_team_code,away_team_code,"
        "home_score,away_score,actual_outcome,tournament,result_source,result_updated_at"
    )

    while True:
        query = client.table("matches_calendar").select(select_cols).order("match_date")
        if from_date:
            query = query.gte("match_date", from_date)
        if to_date:
            query = query.lte("match_date", to_date)
        batch = query.range(offset, offset + chunk_size - 1).execute().data or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < chunk_size:
            break
        offset += chunk_size
    return rows


def _fetch_rows_by_match_ids(
    table_name: str,
    select_cols: str,
    match_ids: list[str],
    chunk_size: int = 200,
) -> list[dict]:
    if not match_ids:
        return []
    client = _service_role_client()
    rows: list[dict] = []
    for batch in _chunks([{"match_id": match_id} for match_id in match_ids], chunk_size):
        ids = [row["match_id"] for row in batch]
        result = client.table(table_name).select(select_cols).in_("match_id", ids).execute()
        rows.extend(result.data or [])
    return rows


def _delete_rows_by_ids(table_name: str, id_column: str, ids: list[str], chunk_size: int = 200) -> int:
    if not ids:
        return 0
    client = _service_role_client()
    deleted = 0
    for batch in _chunks([{id_column: row_id} for row_id in ids], chunk_size):
        batch_ids = [row[id_column] for row in batch]
        client.table(table_name).delete().in_(id_column, batch_ids).execute()
        deleted += len(batch_ids)
    return deleted


def _update_row_by_id(table_name: str, id_column: str, row_id: str, payload: dict) -> None:
    client = _service_role_client()
    client.table(table_name).update(payload).eq(id_column, row_id).execute()


def _pick_canonical_calendar_row(rows: list[dict], ref_counts: dict[str, int]) -> dict:
    def _sort_key(row: dict) -> tuple[int, int, int, str]:
        match_id = str(row.get("match_id") or "")
        home_code = str(row.get("home_team_code") or "").strip().upper()
        away_code = str(row.get("away_team_code") or "").strip().upper()
        has_codes = int(bool(home_code and away_code))
        has_result = int(row.get("home_score") is not None and row.get("away_score") is not None)
        ref_count = int(ref_counts.get(match_id, 0))
        return (has_codes, ref_count, has_result, match_id)

    return max(rows, key=_sort_key)


def _select_best_result_source(rows: list[dict], canonical_row: dict) -> Optional[dict]:
    candidates = [row for row in rows if row.get("home_score") is not None and row.get("away_score") is not None]
    if not candidates:
        return None
    transformed = [_transform_result_for_canonical(row, canonical_row) for row in candidates]
    score_pairs = {(item.get("home_score"), item.get("away_score"), item.get("actual_outcome")) for item in transformed}
    if len(score_pairs) > 1:
        logger.warning(
            "calendar_duplicate_conflicting_results canonical_match_id=%s variants=%s",
            canonical_row.get("match_id"),
            sorted(score_pairs),
        )

    def _sort_key(row: dict) -> tuple[int, tuple[int, str], str]:
        transformed_row = _transform_result_for_canonical(row, canonical_row)
        result_source = str(row.get("result_source") or "").strip()
        has_source = int(bool(result_source))
        timestamp_key = _parse_timestamp(str(row.get("result_updated_at") or ""))
        match_id = str(row.get("match_id") or "")
        return (has_source, timestamp_key, match_id)

    return max(candidates, key=_sort_key)


def _prepare_prediction_merge_plan(
    predictions: list[dict],
    canonical_row: dict,
    match_rows_by_id: dict[str, dict],
) -> tuple[list[dict], list[str]]:
    updates: list[dict] = []
    deletes: list[str] = []
    canonical_match_id = str(canonical_row.get("match_id") or "")
    by_user: dict[str, list[dict]] = {}
    for row in predictions:
        user_key = str(row.get("user_id") or row.get("email") or row.get("prediction_id") or "")
        by_user.setdefault(user_key, []).append(row)

    for rows in by_user.values():
        winner = max(
            rows,
            key=lambda row: (
                _parse_timestamp(str(row.get("created_at") or "")),
                int(str(row.get("match_id") or "") == canonical_match_id),
                str(row.get("prediction_id") or ""),
            ),
        )
        source_row = match_rows_by_id.get(str(winner.get("match_id") or ""), canonical_row)
        reverse = _rows_are_reversed(canonical_row, source_row)
        predicted_outcome = _transform_prediction_outcome_for_canonical(winner.get("predicted_outcome"), reverse)
        payload = {"match_id": canonical_match_id, "predicted_outcome": predicted_outcome}
        if winner.get("email"):
            payload["email"] = winner.get("email")
        if winner.get("user_id"):
            payload["user_id"] = winner.get("user_id")
        updates.append({"prediction_id": winner.get("prediction_id"), "payload": payload})
        for row in rows:
            if row.get("prediction_id") != winner.get("prediction_id") and row.get("prediction_id"):
                deletes.append(str(row["prediction_id"]))
    return updates, deletes


def _prepare_evaluation_merge_plan(evaluations: list[dict], canonical_row: dict) -> tuple[list[dict], list[str]]:
    updates: list[dict] = []
    deletes: list[str] = []
    canonical_match_id = str(canonical_row.get("match_id") or "")
    by_key: dict[tuple[str, str], list[dict]] = {}
    for row in evaluations:
        key = (str(row.get("mode") or ""), str(row.get("model_version") or ""))
        by_key.setdefault(key, []).append(row)

    for rows in by_key.values():
        canonical_rows = [row for row in rows if str(row.get("match_id") or "") == canonical_match_id]
        if canonical_rows:
            winner = max(
                canonical_rows,
                key=lambda row: (_parse_timestamp(str(row.get("created_at") or "")), str(row.get("evaluation_id") or "")),
            )
        else:
            winner = max(
                rows,
                key=lambda row: (_parse_timestamp(str(row.get("created_at") or "")), str(row.get("evaluation_id") or "")),
            )
        payload = _transform_evaluation_for_canonical(winner, canonical_row)
        updates.append({"evaluation_id": winner.get("evaluation_id"), "payload": payload})
        for row in rows:
            if row.get("evaluation_id") != winner.get("evaluation_id") and row.get("evaluation_id"):
                deletes.append(str(row["evaluation_id"]))
    return updates, deletes


def cleanup_matches_calendar_duplicates(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict:
    calendar_rows = _fetch_matches_calendar_rows(from_date=from_date, to_date=to_date)
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in calendar_rows:
        grouped.setdefault(_calendar_duplicate_group_key(row), []).append(row)

    duplicate_groups = [rows for rows in grouped.values() if len(rows) > 1]
    if not duplicate_groups:
        return {
            "duplicate_groups": 0,
            "duplicate_rows": 0,
            "canonical_updates": 0,
            "predictions_updated": 0,
            "predictions_deleted": 0,
            "evaluations_updated": 0,
            "evaluations_deleted": 0,
            "calendar_duplicates_deleted": 0,
        }

    all_match_ids = [str(row.get("match_id")) for rows in duplicate_groups for row in rows if row.get("match_id")]
    predictions = _fetch_rows_by_match_ids(
        "match_predictions",
        "prediction_id,match_id,user_id,email,predicted_outcome,created_at",
        all_match_ids,
    )
    evaluations = _fetch_rows_by_match_ids(
        "consensus_match_evaluations",
        (
            "evaluation_id,match_id,mode,model_version,match_date,home_team,away_team,tournament,"
            "actual_outcome,consensus_predicted_outcome,consensus_prob_home_win,"
            "consensus_prob_draw,consensus_prob_away_win,is_correct,created_at"
        ),
        all_match_ids,
    )

    prediction_counts: dict[str, int] = {}
    evaluation_counts: dict[str, int] = {}
    for row in predictions:
        match_id = str(row.get("match_id") or "")
        prediction_counts[match_id] = prediction_counts.get(match_id, 0) + 1
    for row in evaluations:
        match_id = str(row.get("match_id") or "")
        evaluation_counts[match_id] = evaluation_counts.get(match_id, 0) + 1

    summary = {
        "duplicate_groups": len(duplicate_groups),
        "duplicate_rows": sum(len(rows) for rows in duplicate_groups),
        "canonical_updates": 0,
        "predictions_updated": 0,
        "predictions_deleted": 0,
        "evaluations_updated": 0,
        "evaluations_deleted": 0,
        "calendar_duplicates_deleted": 0,
    }

    for rows in duplicate_groups:
        match_ids = {str(row.get("match_id") or "") for row in rows if row.get("match_id")}
        ref_counts = {
            match_id: prediction_counts.get(match_id, 0) + evaluation_counts.get(match_id, 0)
            for match_id in match_ids
        }
        canonical_row = _pick_canonical_calendar_row(rows, ref_counts)
        duplicate_rows = [row for row in rows if row.get("match_id") != canonical_row.get("match_id")]
        if not duplicate_rows:
            continue

        best_result_source = _select_best_result_source(rows, canonical_row)
        canonical_update: dict[str, object] = {}
        if best_result_source is not None:
            transformed = _transform_result_for_canonical(best_result_source, canonical_row)
            for field in (
                "home_score",
                "away_score",
                "actual_outcome",
                "result_source",
                "result_updated_at",
                "tournament",
                "home_team_code",
                "away_team_code",
            ):
                value = transformed.get(field)
                if value is not None and value != canonical_row.get(field):
                    canonical_update[field] = value
        else:
            for code_field in ("home_team_code", "away_team_code"):
                if not canonical_row.get(code_field):
                    for row in duplicate_rows:
                        transformed = _transform_result_for_canonical(row, canonical_row)
                        code_value = transformed.get(code_field)
                        if code_value:
                            canonical_update[code_field] = code_value
                            break
        if canonical_update:
            _update_row_by_id("matches_calendar", "match_id", str(canonical_row["match_id"]), canonical_update)
            canonical_row = {**canonical_row, **canonical_update}
            summary["canonical_updates"] += 1

        group_predictions = [row for row in predictions if str(row.get("match_id") or "") in match_ids]
        match_rows_by_id = {str(row.get("match_id") or ""): row for row in rows}
        prediction_updates, prediction_deletes = _prepare_prediction_merge_plan(
            group_predictions,
            canonical_row,
            match_rows_by_id,
        )
        for update in prediction_updates:
            prediction_id = str(update.get("prediction_id") or "")
            if prediction_id:
                _update_row_by_id("match_predictions", "prediction_id", prediction_id, update["payload"])
                summary["predictions_updated"] += 1
        summary["predictions_deleted"] += _delete_rows_by_ids(
            "match_predictions", "prediction_id", prediction_deletes
        )

        group_evaluations = [row for row in evaluations if str(row.get("match_id") or "") in match_ids]
        evaluation_updates, evaluation_deletes = _prepare_evaluation_merge_plan(group_evaluations, canonical_row)
        for update in evaluation_updates:
            evaluation_id = str(update.get("evaluation_id") or "")
            if evaluation_id:
                _update_row_by_id(
                    "consensus_match_evaluations",
                    "evaluation_id",
                    evaluation_id,
                    update["payload"],
                )
                summary["evaluations_updated"] += 1
        summary["evaluations_deleted"] += _delete_rows_by_ids(
            "consensus_match_evaluations", "evaluation_id", evaluation_deletes
        )

        duplicate_ids = [str(row.get("match_id")) for row in duplicate_rows if row.get("match_id")]
        summary["calendar_duplicates_deleted"] += _delete_rows_by_ids(
            "matches_calendar", "match_id", duplicate_ids
        )

        logger.info(
            "calendar_duplicate_group_merged canonical_match_id=%s duplicate_match_ids=%s predictions_updated=%s predictions_deleted=%s evaluations_updated=%s evaluations_deleted=%s",
            canonical_row.get("match_id"),
            duplicate_ids,
            len(prediction_updates),
            len(prediction_deletes),
            len(evaluation_updates),
            len(evaluation_deletes),
        )

    return summary


def _find_calendar_match_for_row(row: dict, calendar_rows: list[dict]) -> Optional[dict]:
    if not calendar_rows:
        return None

    by_code_exact: dict[tuple[str, str, str], dict] = {}
    by_code_reverse: dict[tuple[str, str, str], dict] = {}
    by_name_exact: dict[tuple[str, str, str], dict] = {}
    by_name_reverse: dict[tuple[str, str, str], dict] = {}

    for calendar_row in calendar_rows:
        match_date = str(calendar_row.get("match_date") or "").strip()
        home_code, away_code = _resolved_row_codes(calendar_row)
        home_name = _normalized_text(str(calendar_row.get("home_team") or ""))
        away_name = _normalized_text(str(calendar_row.get("away_team") or ""))

        if match_date and home_code and away_code:
            by_code_exact[(match_date, home_code, away_code)] = calendar_row
            by_code_reverse[(match_date, away_code, home_code)] = calendar_row
        if match_date and home_name and away_name:
            by_name_exact[(match_date, home_name, away_name)] = calendar_row
            by_name_reverse[(match_date, away_name, home_name)] = calendar_row

    match_date = str(row.get("match_date") or "").strip()
    home_code, away_code = _resolved_row_codes(row)
    home_name = _normalized_text(str(row.get("home_team") or ""))
    away_name = _normalized_text(str(row.get("away_team") or ""))

    if match_date and home_code and away_code:
        target = by_code_exact.get((match_date, home_code, away_code))
        if target is not None:
            return target
        target = by_code_reverse.get((match_date, home_code, away_code))
        if target is not None:
            return target
    if match_date and home_name and away_name:
        target = by_name_exact.get((match_date, home_name, away_name))
        if target is not None:
            return target
        target = by_name_reverse.get((match_date, home_name, away_name))
        if target is not None:
            return target
    return None


def repair_orphaned_consensus_match_evaluations() -> dict:
    client = _service_role_client()
    calendar_rows = _fetch_matches_calendar_rows()
    calendar_ids = {str(row.get("match_id") or "") for row in calendar_rows if row.get("match_id")}

    orphaned_rows: list[dict] = []
    offset = 0
    select_cols = (
        "evaluation_id,match_id,mode,model_version,match_date,home_team,away_team,tournament,"
        "actual_outcome,consensus_predicted_outcome,consensus_prob_home_win,"
        "consensus_prob_draw,consensus_prob_away_win,is_correct,created_at"
    )
    while True:
        batch = client.table("consensus_match_evaluations").select(select_cols).range(offset, offset + 999).execute().data or []
        if not batch:
            break
        for row in batch:
            if str(row.get("match_id") or "") not in calendar_ids:
                orphaned_rows.append(row)
        if len(batch) < 1000:
            break
        offset += 1000

    if not orphaned_rows:
        return {"orphaned_found": 0, "updated": 0, "deleted": 0, "unmatched": 0}

    current_rows = client.table("consensus_match_evaluations").select(select_cols).execute().data or []
    current_by_key: dict[tuple[str, str, str], list[dict]] = {}
    for row in current_rows:
        key = (
            str(row.get("mode") or ""),
            str(row.get("model_version") or ""),
            str(row.get("match_id") or ""),
        )
        current_by_key.setdefault(key, []).append(row)

    updated = 0
    deleted = 0
    unmatched = 0

    for row in orphaned_rows:
        target = _find_calendar_match_for_row(row, calendar_rows)
        if target is None or not target.get("match_id"):
            unmatched += 1
            continue

        target_key = (
            str(row.get("mode") or ""),
            str(row.get("model_version") or ""),
            str(target.get("match_id") or ""),
        )
        existing = [
            item
            for item in current_by_key.get(target_key, [])
            if str(item.get("evaluation_id") or "") != str(row.get("evaluation_id") or "")
        ]
        if existing:
            chosen = max(
                existing,
                key=lambda item: (_parse_timestamp(str(item.get("created_at") or "")), str(item.get("evaluation_id") or "")),
            )
            incoming_key = (_parse_timestamp(str(row.get("created_at") or "")), str(row.get("evaluation_id") or ""))
            chosen_key = (_parse_timestamp(str(chosen.get("created_at") or "")), str(chosen.get("evaluation_id") or ""))
            if incoming_key > chosen_key:
                payload = _transform_evaluation_for_canonical(row, target)
                _update_row_by_id(
                    "consensus_match_evaluations",
                    "evaluation_id",
                    str(chosen["evaluation_id"]),
                    payload,
                )
                updated += 1
            _delete_rows_by_ids(
                "consensus_match_evaluations",
                "evaluation_id",
                [str(row["evaluation_id"])],
            )
            deleted += 1
            continue

        payload = _transform_evaluation_for_canonical(row, target)
        _update_row_by_id(
            "consensus_match_evaluations",
            "evaluation_id",
            str(row["evaluation_id"]),
            payload,
        )
        updated += 1

    return {
        "orphaned_found": len(orphaned_rows),
        "updated": updated,
        "deleted": deleted,
        "unmatched": unmatched,
    }


def load_matches_for_backfill(
    from_date: str,
    to_date: str,
    tournaments: Optional[set[str]] = None,
    chunk_size: int = 1000,
) -> pd.DataFrame:
    client = _service_role_client()
    rows: list[dict] = []
    offset = 0
    select_cols = "match_date,home_team,away_team,home_score,away_score,tournament"

    while True:
        query = (
            client.table("matches")
            .select(select_cols)
            .gte("match_date", from_date)
            .lte("match_date", to_date)
            .order("match_date")
            .range(offset, offset + chunk_size - 1)
        )
        if tournaments:
            query = query.in_("tournament", sorted(tournaments))

        batch = query.execute().data or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < chunk_size:
            break
        offset += chunk_size

    if not rows:
        return pd.DataFrame(
            columns=["match_date", "home_team", "away_team", "home_score", "away_score", "tournament"]
        )

    frame = pd.DataFrame(rows)
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce")
    frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce")
    frame = frame.dropna(
        subset=["match_date", "home_team", "away_team", "home_score", "away_score"]
    ).copy()
    frame["home_score"] = frame["home_score"].astype(int)
    frame["away_score"] = frame["away_score"].astype(int)
    return frame


def _fetch_calendar_rows_for_backfill(
    from_date: str,
    to_date: str,
    chunk_size: int = 1000,
) -> list[dict]:
    client = _service_role_client()
    rows: list[dict] = []
    offset = 0
    select_cols = "match_id,match_date,home_team,away_team,home_team_code,away_team_code"

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
    return rows


def _reconcile_result_rows_to_calendar(matches_df: pd.DataFrame, calendar_rows: list[dict]) -> pd.DataFrame:
    if matches_df.empty or not calendar_rows:
        return matches_df

    by_code: dict[tuple[str, str, str], dict] = {}
    by_name: dict[tuple[str, str, str], dict] = {}

    for row in calendar_rows:
        match_date = str(row.get("match_date") or "").strip()
        home_team = str(row.get("home_team") or "").strip()
        away_team = str(row.get("away_team") or "").strip()
        home_code = str(row.get("home_team_code") or "").strip().upper()
        away_code = str(row.get("away_team_code") or "").strip().upper()

        if match_date and home_code and away_code:
            by_code[(match_date, home_code, away_code)] = row
        if match_date and home_team and away_team:
            by_name[(match_date, _normalized_text(home_team), _normalized_text(away_team))] = row

    reconciled_rows: list[dict] = []
    matched_by_code = 0
    matched_by_name = 0
    unmatched = 0

    for row in matches_df.to_dict(orient="records"):
        match_date = str(row.get("match_date") or "").strip()
        home_team = str(row.get("home_team") or "").strip()
        away_team = str(row.get("away_team") or "").strip()
        home_code = _resolve_team_code(home_team)
        away_code = _resolve_team_code(away_team)

        target = None
        if match_date and home_code and away_code:
            target = by_code.get((match_date, home_code, away_code))
            if target is not None:
                matched_by_code += 1
        if target is None and match_date and home_team and away_team:
            target = by_name.get((match_date, _normalized_text(home_team), _normalized_text(away_team)))
            if target is not None:
                matched_by_name += 1
        if target is None:
            unmatched += 1
            reconciled_rows.append(row)
            continue

        updated = dict(row)
        updated["home_team"] = str(target.get("home_team") or row["home_team"])
        updated["away_team"] = str(target.get("away_team") or row["away_team"])
        reconciled_rows.append(updated)

    logger.info(
        "scorecard_calendar_reconciliation matched_by_code=%s matched_by_name=%s unmatched=%s total=%s",
        matched_by_code,
        matched_by_name,
        unmatched,
        len(reconciled_rows),
    )
    return pd.DataFrame(reconciled_rows)


def upsert_match_results(
    matches_df: pd.DataFrame,
    result_source: str,
    chunk_size: int = 500,
) -> dict:
    if matches_df.empty:
        return {"upserted": 0}

    from_date = str(matches_df["match_date"].min())
    to_date = str(matches_df["match_date"].max())
    calendar_rows = _fetch_calendar_rows_for_backfill(from_date=from_date, to_date=to_date)
    matches_df = _reconcile_result_rows_to_calendar(matches_df, calendar_rows)

    now_iso = datetime.utcnow().isoformat()
    deduped_payload: dict[tuple[str, str, str], dict] = {}
    for row in matches_df.itertuples(index=False):
        actual_outcome = derive_actual_outcome(getattr(row, "home_score"), getattr(row, "away_score"))
        if actual_outcome is None:
            continue
        payload_row = {
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
        dedupe_key = (
            payload_row["home_team"],
            payload_row["away_team"],
            payload_row["match_date"],
        )
        existing = deduped_payload.get(dedupe_key)
        if existing and (
            existing["home_score"] != payload_row["home_score"]
            or existing["away_score"] != payload_row["away_score"]
            or existing["actual_outcome"] != payload_row["actual_outcome"]
        ):
            logger.warning(
                "scorecard_result_conflict home_team=%s away_team=%s match_date=%s existing=%s incoming=%s",
                payload_row["home_team"],
                payload_row["away_team"],
                payload_row["match_date"],
                existing,
                payload_row,
            )
        deduped_payload[dedupe_key] = payload_row

    payload = list(deduped_payload.values())
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
    prediction_mode = "national" if mode == "world_cup" else mode
    for row in matches:
        try:
            probabilities = predict_match_probabilities_offline(
                home_team=str(row.get("home_team")),
                away_team=str(row.get("away_team")),
                mode=prediction_mode,
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


def _normalize_prediction_ranking_mode(mode: Optional[str]) -> str:
    normalized = str(mode or "").strip().lower()
    if not normalized:
        return "all"
    if normalized == "total":
        return "all"
    if normalized not in _PREDICTION_RANKING_MODES:
        raise ValueError("mode must be one of: all, national, champions, libertadores, world_cup")
    return normalized


def _normalize_prediction_ranking_sort_by(sort_by: str) -> str:
    normalized = str(sort_by or "correct_count").strip()
    if normalized not in _PREDICTION_RANKING_SORT_FIELDS:
        raise ValueError(
            "sort_by must be one of: correct_count, incorrect_count, total_resolved_predictions, accuracy_pct"
        )
    return normalized


def _normalize_prediction_ranking_sort_order(sort_order: str) -> str:
    normalized = str(sort_order or "desc").strip().lower()
    if normalized not in {"asc", "desc"}:
        raise ValueError("sort_order must be one of: asc, desc")
    return normalized


def _ranking_mode_for_tournament(tournament: Optional[str]) -> str:
    tournament_text = str(tournament or "").strip().lower()
    if tournament_text == "uefa champions league":
        return "champions"
    if "libertadores" in tournament_text:
        return "libertadores"
    if tournament_text == "fifa world cup":
        return "world_cup"
    return "national"


def _fetch_match_prediction_rows(chunk_size: int = 1000) -> list[dict]:
    client = _service_role_client()
    rows: list[dict] = []
    offset = 0
    select_cols = "prediction_id,user_id,email,match_id,predicted_outcome"

    while True:
        batch = (
            client.table("match_predictions")
            .select(select_cols)
            .range(offset, offset + chunk_size - 1)
            .execute()
            .data
            or []
        )
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < chunk_size:
            break
        offset += chunk_size
    return rows


def _fetch_calendar_rows_for_prediction_rankings(match_ids: list[str]) -> dict[str, dict]:
    rows = _fetch_rows_by_match_ids(
        "matches_calendar",
        "match_id,tournament,actual_outcome",
        match_ids,
    )
    return {str(row.get("match_id") or ""): row for row in rows if row.get("match_id")}


def _fetch_prediction_ranking_rows() -> list[dict]:
    prediction_rows = _fetch_match_prediction_rows()
    if not prediction_rows:
        return []

    match_ids = [
        str(row.get("match_id") or "")
        for row in prediction_rows
        if str(row.get("match_id") or "").strip()
    ]
    calendar_rows_by_id = _fetch_calendar_rows_for_prediction_rankings(match_ids)
    resolved_rows: list[dict] = []
    for row in prediction_rows:
        match_id = str(row.get("match_id") or "").strip()
        calendar_row = calendar_rows_by_id.get(match_id)
        if not calendar_row:
            continue
        actual_outcome = str(calendar_row.get("actual_outcome") or "").strip()
        predicted_outcome = str(row.get("predicted_outcome") or "").strip()
        if actual_outcome not in {"home_win", "away_win", "draw"}:
            continue
        if predicted_outcome not in {"home_win", "away_win", "draw"}:
            continue
        resolved_rows.append(
            {
                "user_id": str(row.get("user_id") or "").strip(),
                "email": str(row.get("email") or "").strip(),
                "predicted_outcome": predicted_outcome,
                "actual_outcome": actual_outcome,
                "mode": _ranking_mode_for_tournament(calendar_row.get("tournament")),
            }
        )
    return resolved_rows


def _aggregate_prediction_rankings(rows: list[dict], mode: str) -> list[dict]:
    aggregates: dict[str, dict] = {}
    for row in rows:
        if mode != "all" and row.get("mode") != mode:
            continue
        user_id = str(row.get("user_id") or "").strip()
        email = str(row.get("email") or "").strip()
        if user_id in _EXCLUDED_PREDICTION_RANKING_USER_IDS:
            continue
        user_key = user_id or email
        if not user_key:
            continue
        aggregate = aggregates.setdefault(
            user_key,
            {
                "user_id": user_id,
                "email": email,
                "correct_count": 0,
                "incorrect_count": 0,
                "total_resolved_predictions": 0,
            },
        )
        if not aggregate.get("user_id") and user_id:
            aggregate["user_id"] = user_id
        if not aggregate.get("email") and email:
            aggregate["email"] = email
        aggregate["total_resolved_predictions"] += 1
        if row.get("predicted_outcome") == row.get("actual_outcome"):
            aggregate["correct_count"] += 1
        else:
            aggregate["incorrect_count"] += 1

    ranking_rows = list(aggregates.values())
    for row in ranking_rows:
        row["accuracy_pct"] = _compute_accuracy(
            int(row.get("correct_count") or 0),
            int(row.get("total_resolved_predictions") or 0),
        )
    return ranking_rows


def _display_name_from_auth_user(user) -> Optional[str]:
    user_metadata = getattr(user, "user_metadata", None) or {}
    for field in ("full_name", "name"):
        value = str(user_metadata.get(field) or "").strip()
        if value:
            return value
    email = str(getattr(user, "email", "") or "").strip()
    if email:
        return email
    return None


@lru_cache(maxsize=512)
def _fetch_auth_user_display_name(user_id: str) -> Optional[str]:
    normalized_user_id = str(user_id or "").strip()
    if not normalized_user_id:
        return None
    try:
        result = _service_role_client().auth.admin.get_user_by_id(normalized_user_id)
    except Exception as exc:  # pragma: no cover - defensive network fallback
        logger.warning("prediction_ranking_auth_lookup_failed user_id=%s error=%s", normalized_user_id, exc)
        return None
    user = getattr(result, "user", None) or result
    return _display_name_from_auth_user(user)


@lru_cache(maxsize=1)
def _fetch_auth_user_display_name_map() -> dict[str, str]:
    try:
        auth_admin = _service_role_client().auth.admin
    except Exception as exc:  # pragma: no cover - defensive network fallback
        logger.warning("prediction_ranking_auth_list_failed error=%s", exc)
        return {}

    display_name_by_user_id: dict[str, str] = {}
    page = 1
    per_page = 200
    while True:
        try:
            users = auth_admin.list_users(page=page, per_page=per_page) or []
        except Exception as exc:  # pragma: no cover - defensive network fallback
            logger.warning("prediction_ranking_auth_list_page_failed page=%s error=%s", page, exc)
            break
        if not users:
            break
        for user in users:
            user_id = str(getattr(user, "id", "") or "").strip()
            if not user_id:
                continue
            display_name = _display_name_from_auth_user(user)
            if display_name:
                display_name_by_user_id[user_id] = display_name
        if len(users) < per_page:
            break
        page += 1
    return display_name_by_user_id


def _apply_prediction_ranking_display_names(rows: list[dict]) -> list[dict]:
    display_name_by_user_id = _fetch_auth_user_display_name_map()
    enriched: list[dict] = []
    for row in rows:
        user_id = str(row.get("user_id") or "").strip()
        email = str(row.get("email") or "").strip()
        auth_display_name = display_name_by_user_id.get(user_id) if user_id else None
        if not auth_display_name and user_id:
            auth_display_name = _fetch_auth_user_display_name(user_id)
        if auth_display_name:
            display_name = str(auth_display_name).strip()
        else:
            display_name = email or user_id
        display_name = str(display_name or "").strip()
        if "@" in display_name:
            display_name = display_name.split("@", 1)[0].strip() or display_name
        enriched.append({**row, "display_name": str(display_name or "")})
    return enriched


def _sort_prediction_rankings(rows: list[dict], sort_by: str, sort_order: str) -> list[dict]:
    sort_steps = [
        ("correct_count", "desc"),
        ("accuracy_pct", "desc"),
        ("total_resolved_predictions", "desc"),
        ("incorrect_count", "asc"),
        ("display_name", "asc"),
    ]
    ordered_steps = [(sort_by, sort_order)] + [
        step for step in sort_steps if step[0] != sort_by
    ]
    sorted_rows = list(rows)
    for field, direction in reversed(ordered_steps):
        reverse = direction == "desc"
        if field == "display_name":
            sorted_rows.sort(key=lambda row: str(row.get("display_name") or "").casefold(), reverse=reverse)
        else:
            sorted_rows.sort(key=lambda row: float(row.get(field) or 0), reverse=reverse)
    return sorted_rows


def list_prediction_rankings(
    mode: str,
    sort_by: str = "correct_count",
    sort_order: str = "desc",
    page: int = 1,
    page_size: int = 50,
) -> dict:
    normalized_mode = _normalize_prediction_ranking_mode(mode)
    normalized_sort_by = _normalize_prediction_ranking_sort_by(sort_by)
    normalized_sort_order = _normalize_prediction_ranking_sort_order(sort_order)
    page = int(page)
    page_size = int(page_size)
    if page < 1:
        raise ValueError("page must be greater than or equal to 1")
    if page_size < 1 or page_size > 200:
        raise ValueError("page_size must be between 1 and 200")

    ranking_rows = _aggregate_prediction_rankings(
        _fetch_prediction_ranking_rows(),
        normalized_mode,
    )
    ranking_rows = _apply_prediction_ranking_display_names(ranking_rows)
    ranking_rows = _sort_prediction_rankings(
        ranking_rows,
        sort_by=normalized_sort_by,
        sort_order=normalized_sort_order,
    )

    total_users = len(ranking_rows)
    start = (page - 1) * page_size
    end = start + page_size
    page_rows = ranking_rows[start:end]
    rankings = []
    for index, row in enumerate(page_rows, start=start + 1):
        rankings.append(
            {
                "rank": index,
                "user_id": str(row.get("user_id") or row.get("email") or ""),
                "display_name": str(row.get("display_name") or ""),
                "correct_count": int(row.get("correct_count") or 0),
                "incorrect_count": int(row.get("incorrect_count") or 0),
                "total_resolved_predictions": int(row.get("total_resolved_predictions") or 0),
                "accuracy_pct": float(row.get("accuracy_pct") or 0.0),
            }
        )

    return {
        "mode": normalized_mode,
        "sort_by": normalized_sort_by,
        "sort_order": normalized_sort_order,
        "page": page,
        "page_size": page_size,
        "total_users": total_users,
        "rankings": rankings,
    }


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


def _compute_model_scorecard_from_evaluations(
    mode: str,
    model_version: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict:
    rows, _ = _fetch_evaluation_rows(
        mode=mode,
        model_version=model_version,
        from_date=from_date,
        to_date=to_date,
        verdict="all",
    )
    total = len(rows)
    correct = sum(1 for row in rows if bool(row.get("is_correct")))
    incorrect = total - correct
    return {
        "mode": mode,
        "model_version": model_version,
        "from_date": from_date,
        "to_date": to_date,
        "correct_count": correct,
        "incorrect_count": incorrect,
        "total_scored": total,
        "accuracy_pct": _compute_accuracy(correct, total),
    }


def get_model_scorecard(
    mode: str,
    model_version: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict:
    resolved_model_version = _resolve_model_version(model_version)
    normalized_from = _parse_iso_date(from_date)
    normalized_to = _parse_iso_date(to_date)

    snapshot = _fetch_model_scorecard_snapshot(
        mode=mode,
        model_version=resolved_model_version,
        from_date=normalized_from,
        to_date=normalized_to,
    )
    if snapshot:
        return {
            "mode": snapshot.get("mode") or mode,
            "model_version": snapshot.get("model_version") or resolved_model_version,
            "from_date": snapshot.get("period_start") or normalized_from,
            "to_date": snapshot.get("period_end") or normalized_to,
            "correct_count": int(snapshot.get("correct_count") or 0),
            "incorrect_count": int(snapshot.get("incorrect_count") or 0),
            "total_scored": int(snapshot.get("total_scored") or 0),
            "accuracy_pct": float(snapshot.get("accuracy_pct") or 0.0),
        }

    return _compute_model_scorecard_from_evaluations(
        mode=mode,
        model_version=resolved_model_version,
        from_date=normalized_from,
        to_date=normalized_to,
    )


def list_model_scorecard_matches(
    mode: str,
    model_version: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    verdict: str = "all",
    page: int = 1,
    page_size: int = 50,
) -> dict:
    resolved_model_version = _resolve_model_version(model_version)
    if verdict not in {"all", "correct", "incorrect"}:
        raise ValueError("verdict must be one of: all, correct, incorrect")
    normalized_from = _parse_iso_date(from_date)
    normalized_to = _parse_iso_date(to_date)
    page = max(int(page), 1)
    page_size = max(min(int(page_size), 200), 1)
    rows, total = _fetch_evaluation_rows(
        mode=mode,
        model_version=resolved_model_version,
        from_date=normalized_from,
        to_date=normalized_to,
        verdict=verdict,
        page=page,
        page_size=page_size,
    )
    return {
        "mode": mode,
        "model_version": resolved_model_version,
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
    resolved_model_version = _resolve_model_version(model_version)
    summary = _compute_model_scorecard_from_evaluations(
        mode=mode,
        model_version=resolved_model_version,
        from_date=period_start,
        to_date=period_end,
    )
    payload = {
        "run_id": run_id,
        "mode": mode,
        "model_version": resolved_model_version,
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
) -> dict:
    normalized_from = _parse_iso_date(from_date)
    normalized_to = _parse_iso_date(to_date)
    if not normalized_from or not normalized_to:
        raise ValueError("from_date and to_date must use YYYY-MM-DD format")
    if normalized_from > normalized_to:
        raise ValueError("from_date cannot be greater than to_date")

    normalized_mode = str(mode or "").strip().lower() or "national"
    if normalized_mode not in {"national", "champions", "libertadores", "world_cup"}:
        raise ValueError("mode must be one of: national, champions, libertadores, world_cup")

    resolved_model_version = _resolve_model_version(model_version)

    run_id = str(uuid4())
    if normalized_mode == "national":
        tournaments = set(_FORM_TOURNAMENTS)
    elif normalized_mode == "champions":
        tournaments = {"UEFA Champions League"}
    elif normalized_mode == "libertadores":
        tournaments = {"Libertadores"}
    else:
        tournaments = {"FIFA World Cup"}

    results_upsert = {"upserted": 0}
    if normalized_mode in {"national", "world_cup"}:
        matches_df = load_matches_for_backfill(
            from_date=normalized_from,
            to_date=normalized_to,
            tournaments=tournaments,
        )
        results_upsert = upsert_match_results(
            matches_df=matches_df,
            result_source=_MATCH_RESULT_SOURCE,
        )

    completed_matches = _fetch_completed_matches(
        from_date=normalized_from,
        to_date=normalized_to,
        tournaments=tournaments,
    )
    evaluation_summary = evaluate_consensus_matches(
        matches=completed_matches,
        mode=normalized_mode,
        model_version=resolved_model_version,
        run_id=run_id,
    )
    snapshot = upsert_model_scorecard_snapshot(
        run_id=run_id,
        mode=normalized_mode,
        model_version=resolved_model_version,
        period_start=normalized_from,
        period_end=normalized_to,
    )
    return {
        "run_id": run_id,
        "mode": normalized_mode,
        "model_version": resolved_model_version,
        "from_date": normalized_from,
        "to_date": normalized_to,
        "results_upsert": results_upsert,
        "completed_matches": len(completed_matches),
        "evaluations": evaluation_summary,
        "snapshot": snapshot,
    }
