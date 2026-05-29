from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backend.supabase_client import get_supabase_client
except ImportError:  # pragma: no cover
    from src.backend.supabase_client import get_supabase_client

DEFAULT_SNAPSHOT_PATH = PROJECT_ROOT / "data" / "ucl_matches_calendar_snapshot_2026_04_23.csv"
DEFAULT_ALIAS_PATH = PROJECT_ROOT / "preprocessing_data" / "club_team_aliases.csv"
UCL_TOURNAMENT = "UEFA Champions League"
FBREF_SOURCE_URL = "https://fbref.com/en/comps/8/schedule/Champions-League-Scores-and-Fixtures"


@dataclass(frozen=True)
class SnapshotFixture:
    match_date: str
    home_team: str
    away_team: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    round_name: str = ""

    @property
    def is_completed(self) -> bool:
        return self.home_score is not None and self.away_score is not None


@dataclass(frozen=True)
class PlannedCalendarRow:
    action: str
    payload: dict
    existing_row: Optional[dict] = None


def _normalized_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch for ch in text if ch.isalnum())


def _canonical_key(team: str, alias_to_canonical: dict[str, str]) -> str:
    normalized = _normalized_text(team)
    canonical = alias_to_canonical.get(normalized, str(team or "").strip())
    return _normalized_text(canonical)


def _parse_optional_score(value: str) -> Optional[int]:
    raw = str(value or "").strip()
    if raw == "":
        return None
    return int(raw)


def load_snapshot_fixtures(snapshot_path: Path) -> list[SnapshotFixture]:
    fixtures: list[SnapshotFixture] = []
    with snapshot_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fixtures.append(
                SnapshotFixture(
                    match_date=str(row.get("match_date") or "").strip(),
                    home_team=str(row.get("home_team") or "").strip(),
                    away_team=str(row.get("away_team") or "").strip(),
                    home_score=_parse_optional_score(row.get("home_score")),
                    away_score=_parse_optional_score(row.get("away_score")),
                    round_name=str(row.get("round_name") or "").strip(),
                )
            )
    return fixtures


def load_alias_maps(alias_path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    alias_to_canonical: dict[str, str] = {}
    canonical_to_names: dict[str, set[str]] = defaultdict(set)
    if not alias_path.exists():
        return alias_to_canonical, canonical_to_names

    with alias_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            alias = str(row.get("alias") or row.get("source_name") or row.get("source_team") or "").strip()
            canonical = str(
                row.get("canonical") or row.get("target_name") or row.get("target_team") or ""
            ).strip()
            if not alias or not canonical:
                continue
            alias_to_canonical[_normalized_text(alias)] = canonical
            canonical_key = _normalized_text(canonical)
            canonical_to_names[canonical_key].update({alias, canonical})

    return alias_to_canonical, canonical_to_names


def build_existing_label_preferences(
    existing_rows: list[dict],
    alias_to_canonical: dict[str, str],
) -> dict[str, str]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in existing_rows:
        for field in ("home_team", "away_team"):
            team = str(row.get(field) or "").strip()
            if not team:
                continue
            counts[_canonical_key(team, alias_to_canonical)][team] += 1

    return {
        canonical_key: counter.most_common(1)[0][0]
        for canonical_key, counter in counts.items()
        if counter
    }


def preferred_calendar_team_name(
    team: str,
    label_preferences: dict[str, str],
    alias_to_canonical: dict[str, str],
) -> str:
    preferred = label_preferences.get(_canonical_key(team, alias_to_canonical))
    return preferred or str(team or "").strip()


def build_existing_match_index(
    existing_rows: list[dict],
    alias_to_canonical: dict[str, str],
) -> dict[tuple[str, str, str], dict]:
    index: dict[tuple[str, str, str], dict] = {}
    for row in existing_rows:
        match_date = str(row.get("match_date") or "").strip()
        home_team = str(row.get("home_team") or "").strip()
        away_team = str(row.get("away_team") or "").strip()
        if not match_date or not home_team or not away_team:
            continue
        key = (
            match_date,
            _canonical_key(home_team, alias_to_canonical),
            _canonical_key(away_team, alias_to_canonical),
        )
        index[key] = row
    return index


def build_sync_plan(
    fixtures: list[SnapshotFixture],
    existing_rows: list[dict],
    alias_to_canonical: dict[str, str],
) -> list[PlannedCalendarRow]:
    plans: list[PlannedCalendarRow] = []
    label_preferences = build_existing_label_preferences(existing_rows, alias_to_canonical)
    existing_index = build_existing_match_index(existing_rows, alias_to_canonical)
    now_iso = datetime.now(timezone.utc).isoformat()

    for fixture in fixtures:
        key = (
            fixture.match_date,
            _canonical_key(fixture.home_team, alias_to_canonical),
            _canonical_key(fixture.away_team, alias_to_canonical),
        )
        existing = existing_index.get(key)

        home_team = (
            str(existing.get("home_team") or "").strip()
            if existing
            else preferred_calendar_team_name(fixture.home_team, label_preferences, alias_to_canonical)
        )
        away_team = (
            str(existing.get("away_team") or "").strip()
            if existing
            else preferred_calendar_team_name(fixture.away_team, label_preferences, alias_to_canonical)
        )

        payload = {
            "match_date": fixture.match_date,
            "home_team": home_team,
            "away_team": away_team,
            "tournament": UCL_TOURNAMENT,
        }
        if fixture.is_completed:
            payload["home_score"] = fixture.home_score
            payload["away_score"] = fixture.away_score
            payload["result_source"] = FBREF_SOURCE_URL
            payload["result_updated_at"] = now_iso

        if existing is None:
            plans.append(PlannedCalendarRow(action="insert", payload=payload))
            continue

        existing_home_score = existing.get("home_score")
        existing_away_score = existing.get("away_score")
        existing_tournament = str(existing.get("tournament") or "").strip()
        existing_result_source = str(existing.get("result_source") or "").strip()

        needs_score_update = fixture.is_completed and (
            existing_home_score != fixture.home_score or existing_away_score != fixture.away_score
        )
        needs_tournament_update = existing_tournament != UCL_TOURNAMENT
        needs_source_update = fixture.is_completed and existing_result_source != FBREF_SOURCE_URL

        if needs_score_update or needs_tournament_update or needs_source_update:
            plans.append(PlannedCalendarRow(action="update", payload=payload, existing_row=existing))
        else:
            plans.append(PlannedCalendarRow(action="skip", payload=payload, existing_row=existing))

    return plans


def fetch_existing_ucl_rows() -> list[dict]:
    client = get_supabase_client(use_service_role=True)
    response = (
        client.table("matches_calendar")
        .select("match_id,match_date,home_team,away_team,home_score,away_score,tournament,result_source")
        .eq("tournament", UCL_TOURNAMENT)
        .order("match_date")
        .execute()
    )
    return response.data or []


def apply_sync_plan(plan: list[PlannedCalendarRow]) -> dict[str, int]:
    rows = [item.payload for item in plan if item.action in {"insert", "update"}]
    if not rows:
        return {"inserted": 0, "updated": 0}

    client = get_supabase_client(use_service_role=True)
    client.table("matches_calendar").upsert(
        rows,
        on_conflict="home_team,away_team,match_date",
    ).execute()
    return {
        "inserted": sum(1 for item in plan if item.action == "insert"),
        "updated": sum(1 for item in plan if item.action == "update"),
    }


def summarize_plan(plan: list[PlannedCalendarRow]) -> dict[str, object]:
    return {
        "received": len(plan),
        "inserted": sum(1 for item in plan if item.action == "insert"),
        "updated": sum(1 for item in plan if item.action == "update"),
        "skipped": sum(1 for item in plan if item.action == "skip"),
        "rows": [item.payload for item in plan if item.action in {"insert", "update"}],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync a snapshot of UEFA Champions League fixtures into matches_calendar."
    )
    parser.add_argument(
        "--snapshot-csv",
        type=Path,
        default=DEFAULT_SNAPSHOT_PATH,
        help=f"CSV snapshot to sync (default: {DEFAULT_SNAPSHOT_PATH}).",
    )
    parser.add_argument(
        "--alias-csv",
        type=Path,
        default=DEFAULT_ALIAS_PATH,
        help=f"Club alias CSV (default: {DEFAULT_ALIAS_PATH}).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply inserts and updates to Supabase. Without this flag the script is a dry run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixtures = load_snapshot_fixtures(args.snapshot_csv)
    alias_to_canonical, _ = load_alias_maps(args.alias_csv)
    existing_rows = fetch_existing_ucl_rows()
    plan = build_sync_plan(fixtures, existing_rows, alias_to_canonical)
    summary = summarize_plan(plan)

    if args.apply:
        applied = apply_sync_plan(plan)
        summary.update(applied)
        summary["applied"] = True
    else:
        summary["applied"] = False

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
