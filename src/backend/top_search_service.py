from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
import logging
from typing import Optional

import pandas as pd

try:
    from .paths import CLUB_COEFFICIENTS_HISTORY_PATH, MATCHES_PATH, RANKING_PATH
    from .predict_match import (
        _build_club_coeff_lookup,
        _lookup_club_coeff_row,
        _normalized_text,
        normalize_search_bucket_mode,
    )
    from .supabase_client import get_supabase_client
except ImportError:  # pragma: no cover - fallback for direct module execution
    from src.backend.paths import CLUB_COEFFICIENTS_HISTORY_PATH, MATCHES_PATH, RANKING_PATH
    from src.backend.predict_match import (
        _build_club_coeff_lookup,
        _lookup_club_coeff_row,
        _normalized_text,
        normalize_search_bucket_mode,
    )
    from src.backend.supabase_client import get_supabase_client


logger = logging.getLogger("futbolconu.top_search")

TOP_SEARCHED_LIMIT = 5
TOP_SEARCHED_DEFAULT_LOOKBACK_DAYS = 30
TOP_SEARCHED_NATIONAL_LOOKBACK_DAYS = 7

_national_team_keys: Optional[set[str]] = None
_club_coeff_lookup: Optional[dict] = None


def _service_role_client():
    return get_supabase_client(use_service_role=True)


def _utc_now() -> datetime:
    return datetime.utcnow()


def _snapshot_date_for(now: Optional[datetime] = None) -> str:
    return (now or _utc_now()).date().isoformat()


def _load_national_team_keys() -> set[str]:
    global _national_team_keys
    if _national_team_keys is not None:
        return _national_team_keys

    names: set[str] = set()

    try:
        ranking = pd.read_csv(RANKING_PATH)
        for value in ranking.get("team", []):
            key = _normalized_text(value)
            if key:
                names.add(key)
    except Exception as exc:  # pragma: no cover - defensive only
        logger.warning("top_search_national_ranking_load_failed error=%s", exc)

    try:
        matches = pd.read_csv(MATCHES_PATH, usecols=["home_team", "away_team"])
        for column in ("home_team", "away_team"):
            for value in matches.get(column, []):
                key = _normalized_text(value)
                if key:
                    names.add(key)
    except Exception as exc:  # pragma: no cover - defensive only
        logger.warning("top_search_matches_load_failed error=%s", exc)

    _national_team_keys = names
    return _national_team_keys


def _load_club_coeff_lookup() -> dict:
    global _club_coeff_lookup
    if _club_coeff_lookup is not None:
        return _club_coeff_lookup

    try:
        coefficients = pd.read_csv(CLUB_COEFFICIENTS_HISTORY_PATH)
    except Exception as exc:  # pragma: no cover - defensive only
        logger.warning("top_search_club_coefficients_load_failed error=%s", exc)
        _club_coeff_lookup = {}
        return _club_coeff_lookup

    _club_coeff_lookup = _build_club_coeff_lookup(coefficients)
    return _club_coeff_lookup


def _is_known_national_team(team_name: str) -> bool:
    key = _normalized_text(team_name)
    return bool(key and key in _load_national_team_keys())


def _is_known_club_team(team_name: str) -> bool:
    return _lookup_club_coeff_row(team_name, _load_club_coeff_lookup()) is not None


def infer_prediction_bucket_mode(home_team: str, away_team: str) -> Optional[str]:
    national_match = _is_known_national_team(home_team) and _is_known_national_team(away_team)
    club_match = _is_known_club_team(home_team) and _is_known_club_team(away_team)

    if national_match and not club_match:
        return "national"
    if club_match and not national_match:
        return "champions"
    return None


def _row_bucket_mode(row: dict) -> Optional[str]:
    mode = normalize_search_bucket_mode(row.get("mode"), default=None, strict=False)
    if mode in {"national", "champions", "libertadores"}:
        return mode
    return infer_prediction_bucket_mode(row.get("home_team", ""), row.get("away_team", ""))


def _fetch_daily_snapshot(mode: str, snapshot_date: str) -> Optional[dict]:
    result = (
        _service_role_client()
        .table("top_searched_team_snapshots")
        .select("mode,snapshot_date,lookback_days_used,teams,calculated_at")
        .eq("mode", mode)
        .eq("snapshot_date", snapshot_date)
        .order("calculated_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return rows[0] if rows else None


def _fetch_recent_user_predictions(start_iso: str, chunk_size: int = 1000) -> list[dict]:
    client = _service_role_client()
    rows: list[dict] = []
    offset = 0
    select_with_mode = "home_team,away_team,timestamp,mode"
    select_without_mode = "home_team,away_team,timestamp"
    supports_mode = True

    while True:
        try:
            select_cols = select_with_mode if supports_mode else select_without_mode
            query = (
                client.table("user_predictions")
                .select(select_cols)
                .gte("timestamp", start_iso)
                .order("timestamp")
                .range(offset, offset + chunk_size - 1)
            )
            batch = query.execute().data or []
        except Exception as exc:
            error_text = str(exc).lower()
            if supports_mode and "column" in error_text and "mode" in error_text:
                supports_mode = False
                continue
            raise

        if not batch:
            break

        rows.extend(batch)
        if len(batch) < chunk_size:
            break
        offset += chunk_size

    return rows


def _build_snapshot_for_rows(
    mode: str,
    rows: list[dict],
    current_time: datetime,
    lookback_days_used: int,
) -> dict:
    snapshot_date = _snapshot_date_for(current_time)

    counts: Counter[str] = Counter()
    for row in rows:
        if _row_bucket_mode(row) != mode:
            continue
        for field in ("home_team", "away_team"):
            team = str(row.get(field, "") or "").strip()
            if team:
                counts[team] += 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:TOP_SEARCHED_LIMIT]
    teams = [
        {"rank": idx + 1, "team": team, "searches": searches}
        for idx, (team, searches) in enumerate(ranked)
    ]
    return {
        "mode": mode,
        "snapshot_date": snapshot_date,
        "lookback_days_used": lookback_days_used,
        "teams": teams,
        "calculated_at": current_time.isoformat() + "Z",
    }


def _build_snapshot_for_lookback(mode: str, current_time: datetime, lookback_days: int) -> dict:
    start_iso = (current_time - timedelta(days=lookback_days)).isoformat()
    rows = _fetch_recent_user_predictions(start_iso)
    return _build_snapshot_for_rows(mode, rows, current_time, lookback_days)


def build_daily_top_searched_snapshot(mode: str, now: Optional[datetime] = None) -> dict:
    normalized_mode = normalize_search_bucket_mode(mode, strict=True)
    if normalized_mode not in {"national", "champions", "libertadores"}:
        raise ValueError("mode must be one of: national, champions, libertadores")

    current_time = now or _utc_now()
    if normalized_mode == "national":
        snapshot = _build_snapshot_for_lookback(
            normalized_mode,
            current_time,
            TOP_SEARCHED_NATIONAL_LOOKBACK_DAYS,
        )
        if snapshot["teams"]:
            return snapshot

    return _build_snapshot_for_lookback(
        normalized_mode,
        current_time,
        TOP_SEARCHED_DEFAULT_LOOKBACK_DAYS,
    )


def upsert_daily_snapshot(snapshot: dict) -> dict:
    payload = {
        "mode": snapshot["mode"],
        "snapshot_date": snapshot["snapshot_date"],
        "lookback_days_used": snapshot["lookback_days_used"],
        "teams": snapshot["teams"],
        "calculated_at": snapshot["calculated_at"],
    }
    _service_role_client().table("top_searched_team_snapshots").upsert(
        payload,
        on_conflict="mode,snapshot_date",
    ).execute()
    return payload


def get_top_searched_teams_snapshot(mode: str, now: Optional[datetime] = None) -> dict:
    normalized_mode = normalize_search_bucket_mode(mode, strict=True)
    if normalized_mode not in {"national", "champions", "libertadores"}:
        raise ValueError("mode must be one of: national, champions, libertadores")

    snapshot_date = _snapshot_date_for(now)
    existing = _fetch_daily_snapshot(normalized_mode, snapshot_date)
    if existing and existing.get("lookback_days_used") is not None:
        return {
            "mode": existing.get("mode") or normalized_mode,
            "snapshot_date": existing.get("snapshot_date") or snapshot_date,
            "lookback_days_used": existing.get("lookback_days_used"),
            "teams": existing.get("teams") or [],
            "calculated_at": existing.get("calculated_at"),
        }

    snapshot = build_daily_top_searched_snapshot(normalized_mode, now=now)
    return upsert_daily_snapshot(snapshot)
