from __future__ import annotations

from datetime import date
from functools import lru_cache
import logging
from pathlib import Path
from time import perf_counter

import pandas as pd

try:
    from .paths import HISTORICAL_DATA_DIR, RANKING_PATH
    from .supabase_client import get_supabase_client
    from utils.add_ranking_fifa_to_matches import add_ranking_fifa_to_matches
    from utils.confederation_mapping import add_confederation_to_matches
    from utils.matches_common import (
        build_fixture_identity_key,
        build_match_key,
        match_row_quality_score,
        normalize_match_text,
    )
except ImportError:  # pragma: no cover - fallback for direct module execution
    from src.backend.paths import HISTORICAL_DATA_DIR, RANKING_PATH
    from src.backend.supabase_client import get_supabase_client
    from src.utils.add_ranking_fifa_to_matches import add_ranking_fifa_to_matches
    from src.utils.confederation_mapping import add_confederation_to_matches
    from src.utils.matches_common import (
        build_fixture_identity_key,
        build_match_key,
        match_row_quality_score,
        normalize_match_text,
    )


logger = logging.getLogger("futbolconu.matches_admin")

CONFED_PATH = HISTORICAL_DATA_DIR / "unique_countries.csv"


def _service_role_client():
    return get_supabase_client(use_service_role=True)


@lru_cache(maxsize=1)
def _load_ranking_df() -> pd.DataFrame:
    return pd.read_csv(RANKING_PATH)


def _normalize_date(value: str) -> tuple[str | None, str]:
    text = str(value or "").strip()
    if not text:
        return None, "date is required"
    try:
        return date.fromisoformat(text).isoformat(), ""
    except ValueError:
        return None, "date must be in YYYY-MM-DD format"


def _normalize_score(value, field_name: str) -> tuple[int | None, str]:
    try:
        return int(value), ""
    except (TypeError, ValueError):
        return None, f"{field_name} must be an integer"


def _validate_match_row(row: dict) -> tuple[dict | None, str]:
    normalized = {
        "home_team": normalize_match_text((row or {}).get("home_team")),
        "away_team": normalize_match_text((row or {}).get("away_team")),
        "tournament": normalize_match_text((row or {}).get("tournament")),
        "city": normalize_match_text((row or {}).get("city")),
        "country": normalize_match_text((row or {}).get("country")),
        "neutral": bool((row or {}).get("neutral")),
    }

    for field in ("home_team", "away_team", "tournament", "city", "country"):
        if not normalized[field]:
            return None, f"{field} is required"

    normalized_date, date_error = _normalize_date((row or {}).get("date"))
    if date_error:
        return None, date_error
    normalized["date"] = normalized_date

    home_score, home_score_error = _normalize_score((row or {}).get("home_score"), "home_score")
    if home_score_error:
        return None, home_score_error
    away_score, away_score_error = _normalize_score((row or {}).get("away_score"), "away_score")
    if away_score_error:
        return None, away_score_error
    normalized["home_score"] = home_score
    normalized["away_score"] = away_score

    return normalized, ""


def _enrich_match_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    enriched = add_confederation_to_matches(df, confed_path=str(CONFED_PATH))
    return add_ranking_fifa_to_matches(enriched, _load_ranking_df())


def _build_match_record(row: dict) -> tuple[dict | None, str]:
    if pd.isna(row.get("home_team_confederation")) or pd.isna(row.get("away_team_confederation")):
        return None, "missing confederation mapping"

    payload = {
        "match_date": row["date"],
        "home_team": normalize_match_text(row["home_team"]),
        "away_team": normalize_match_text(row["away_team"]),
        "home_score": int(row["home_score"]),
        "away_score": int(row["away_score"]),
        "tournament": normalize_match_text(row["tournament"]),
        "city": normalize_match_text(row["city"]),
        "country": normalize_match_text(row["country"]),
        "neutral": bool(row["neutral"]),
        "home_team_confederation": normalize_match_text(row["home_team_confederation"]),
        "away_team_confederation": normalize_match_text(row["away_team_confederation"]),
        "home_team_fifa_rank": float(row["home_team_fifa_rank"]),
        "home_team_fifa_points": float(row["home_team_fifa_points"]),
        "away_team_fifa_rank": float(row["away_team_fifa_rank"]),
        "away_team_fifa_points": float(row["away_team_fifa_points"]),
        "source": "manual",
        "source_file": None,
    }
    payload["match_key"] = build_match_key(payload)
    return payload, ""


def _fetch_existing_match_keys(match_keys: list[str]) -> set[str]:
    if not match_keys:
        return set()

    existing = set()
    client = _service_role_client()
    chunk_size = 500
    for start in range(0, len(match_keys), chunk_size):
        chunk = match_keys[start : start + chunk_size]
        rows = (
            client.table("matches")
            .select("match_key")
            .in_("match_key", chunk)
            .execute()
            .data
            or []
        )
        existing.update(str(row.get("match_key")) for row in rows if row.get("match_key"))
    return existing


def _bulk_upsert_match_rows(rows: list[dict]) -> None:
    if not rows:
        return
    _service_role_client().table("matches").upsert(
        rows,
        on_conflict="match_key",
    ).execute()


def _insert_match_rows(rows: list[dict]) -> None:
    if not rows:
        return
    _service_role_client().table("matches").insert(rows).execute()


def _update_match_row(match_id: int, payload: dict) -> None:
    _service_role_client().table("matches").update(payload).eq("match_id", match_id).execute()


def _fetch_matches_rows_for_cleanup(
    from_date: str,
    to_date: str,
    chunk_size: int = 1000,
) -> list[dict]:
    client = _service_role_client()
    rows: list[dict] = []
    offset = 0
    select_cols = (
        "match_id,match_key,match_date,home_team,away_team,home_score,away_score,tournament,"
        "city,country,neutral,source,source_file,created_at,updated_at"
    )
    while True:
        batch = (
            client.table("matches")
            .select(select_cols)
            .gte("match_date", from_date)
            .lte("match_date", to_date)
            .order("match_date")
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


def _fetch_existing_matches_for_range(
    from_date: str,
    to_date: str,
    chunk_size: int = 1000,
) -> list[dict]:
    return _fetch_matches_rows_for_cleanup(from_date=from_date, to_date=to_date, chunk_size=chunk_size)


def _delete_matches_rows(match_ids: list[int], chunk_size: int = 200) -> int:
    if not match_ids:
        return 0
    client = _service_role_client()
    deleted = 0
    for start in range(0, len(match_ids), chunk_size):
        chunk = match_ids[start : start + chunk_size]
        client.table("matches").delete().in_("match_id", chunk).execute()
        deleted += len(chunk)
    return deleted


def _match_duplicate_group_key(row: dict) -> tuple[str, str, str]:
    parts = build_fixture_identity_key(row).split("|")
    return tuple(parts)  # type: ignore[return-value]


def _match_row_quality_score(row: dict) -> tuple[int, int, int, int, str]:
    return match_row_quality_score(row)


def _pick_canonical_match_row(rows: list[dict]) -> dict:
    return max(rows, key=lambda row: (_match_row_quality_score(row), -int(row.get("match_id") or 0)))


def _pick_best_match_record(rows: list[dict]) -> dict:
    return max(rows, key=lambda row: (_match_row_quality_score(row), row.get("match_key") or ""))


def _reconcile_match_records(records: list[dict]) -> tuple[list[dict], list[dict], list[int], set[str]]:
    if not records:
        return [], [], [], set()

    date_values = [str(record["match_date"]) for record in records]
    existing_rows = _fetch_existing_matches_for_range(min(date_values), max(date_values))
    existing_by_fixture: dict[str, list[dict]] = {}
    for row in existing_rows:
        existing_by_fixture.setdefault(build_fixture_identity_key(row), []).append(row)

    inserts: list[dict] = []
    updates: list[dict] = []
    deletes: list[int] = []
    existing_match_keys: set[str] = set()

    incoming_by_fixture: dict[str, list[dict]] = {}
    for record in records:
        incoming_by_fixture.setdefault(build_fixture_identity_key(record), []).append(record)

    for fixture_key, incoming_group in incoming_by_fixture.items():
        incoming_record = _pick_best_match_record(incoming_group)
        existing_group = existing_by_fixture.get(fixture_key, [])
        if not existing_group:
            inserts.append(incoming_record)
            continue

        canonical_existing = _pick_canonical_match_row(existing_group)
        existing_match_keys.add(str(canonical_existing.get("match_key") or ""))
        for row in existing_group:
            match_id = row.get("match_id")
            if match_id != canonical_existing.get("match_id") and match_id is not None:
                deletes.append(int(match_id))

        best_row = _pick_best_match_record([incoming_record, canonical_existing])
        if best_row is incoming_record:
            updates.append({"match_id": int(canonical_existing["match_id"]), "payload": incoming_record})

    return inserts, updates, deletes, existing_match_keys


def cleanup_matches_duplicates(
    from_date: str,
    to_date: str,
) -> dict:
    normalized_from, from_error = _normalize_date(from_date)
    if from_error:
        raise ValueError(from_error)
    normalized_to, to_error = _normalize_date(to_date)
    if to_error:
        raise ValueError(to_error)
    if normalized_from > normalized_to:
        raise ValueError("from_date cannot be greater than to_date")

    rows = _fetch_matches_rows_for_cleanup(normalized_from, normalized_to)
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault(_match_duplicate_group_key(row), []).append(row)

    duplicate_groups = [group for group in grouped.values() if len(group) > 1]
    deleted_ids: list[int] = []
    exact_groups = 0
    reversed_groups = 0

    for group in duplicate_groups:
        canonical = _pick_canonical_match_row(group)
        pairings = {(normalize_match_text(row.get("home_team")), normalize_match_text(row.get("away_team"))) for row in group}
        if len(pairings) > 1:
            reversed_groups += 1
        else:
            exact_groups += 1
        for row in group:
            match_id = row.get("match_id")
            if match_id != canonical.get("match_id") and match_id is not None:
                deleted_ids.append(int(match_id))

        logger.info(
            "matches_duplicate_group_cleaned canonical_match_id=%s deleted_match_ids=%s match_date=%s home_team=%s away_team=%s",
            canonical.get("match_id"),
            [row.get("match_id") for row in group if row.get("match_id") != canonical.get("match_id")],
            canonical.get("match_date"),
            canonical.get("home_team"),
            canonical.get("away_team"),
        )

    deleted = _delete_matches_rows(deleted_ids)
    return {
        "from_date": normalized_from,
        "to_date": normalized_to,
        "rows_scanned": len(rows),
        "duplicate_groups": len(duplicate_groups),
        "exact_duplicate_groups": exact_groups,
        "reversed_duplicate_groups": reversed_groups,
        "deleted_rows": deleted,
    }


def upsert_manual_matches_batch(matches: list[dict], request_id: str = "-") -> dict:
    start_time = perf_counter()
    summary = {
        "received": len(matches or []),
        "inserted": 0,
        "updated": 0,
        "skipped": 0,
        "errors": [],
    }

    logger.info(
        "manual_matches_upsert_started request_id=%s received=%s",
        request_id,
        summary["received"],
    )

    if not matches:
        return summary

    valid_rows = []
    seen_payload_keys = set()
    for index, row in enumerate(matches):
        normalized_row, error = _validate_match_row(row)
        if error:
            summary["skipped"] += 1
            summary["errors"].append({"row_index": index, "reason": error})
            continue

        payload_key = build_fixture_identity_key(
            {
                "match_date": normalized_row["date"],
                "home_team": normalized_row["home_team"],
                "away_team": normalized_row["away_team"],
                "tournament": normalized_row["tournament"],
            }
        )
        if payload_key in seen_payload_keys:
            summary["skipped"] += 1
            summary["errors"].append({"row_index": index, "reason": "duplicate fixture in payload"})
            continue

        seen_payload_keys.add(payload_key)
        normalized_row["_row_index"] = index
        valid_rows.append(normalized_row)

    if not valid_rows:
        return summary

    enriched = _enrich_match_rows(valid_rows)

    records = []
    for enriched_row in enriched.to_dict(orient="records"):
        row_index = enriched_row.pop("_row_index")
        record, error = _build_match_record(enriched_row)
        if error:
            summary["skipped"] += 1
            summary["errors"].append({"row_index": row_index, "reason": error})
            continue
        records.append(record)

    if not records:
        return summary

    inserts, updates, deletes, existing_keys = _reconcile_match_records(records)
    _insert_match_rows(inserts)
    for update in updates:
        _update_match_row(update["match_id"], update["payload"])
    _delete_matches_rows(deletes)

    summary["inserted"] = len(inserts)
    summary["updated"] = len(existing_keys)

    logger.info(
        "manual_matches_upsert_completed request_id=%s inserted=%s updated=%s skipped=%s errors=%s elapsed_ms=%.2f",
        request_id,
        summary["inserted"],
        summary["updated"],
        summary["skipped"],
        len(summary["errors"]),
        (perf_counter() - start_time) * 1000.0,
    )
    return summary
