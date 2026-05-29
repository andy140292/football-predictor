import argparse
import csv
import os
import time
from pathlib import Path
import sys
from typing import Optional

from dotenv import load_dotenv
from supabase import create_client

try:
    from matches_common import (
        build_fixture_identity_key,
        build_match_key,
        match_row_quality_score,
        normalize_match_text,
        resolve_latest_matches_csv,
    )
except ImportError:  # pragma: no cover - fallback for package-style execution
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from src.utils.matches_common import (
        build_fixture_identity_key,
        build_match_key,
        match_row_quality_score,
        normalize_match_text,
        resolve_latest_matches_csv,
    )


load_dotenv()

IDENTITY_FIELDS = (
    "match_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
)
ENRICHABLE_FIELDS = (
    "city",
    "country",
)
NON_SUBSTANTIVE_FIELDS = (
    "source",
    "source_file",
)


def _nullable_int(value: str):
    text = str(value or "").strip()
    if not text:
        return None
    return int(float(text))


def _nullable_float(value: str):
    text = str(value or "").strip()
    if not text:
        return None
    return float(text)


def _nullable_text(value: str):
    text = str(value or "").strip()
    return text or None


def _parse_bool(value: str) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def parse_csv_row(row: dict, source_file: str) -> dict:
    record = {
        "match_date": row["date"],
        "home_team": normalize_match_text(row["home_team"]),
        "away_team": normalize_match_text(row["away_team"]),
        "home_score": _nullable_int(row["home_score"]),
        "away_score": _nullable_int(row["away_score"]),
        "tournament": _nullable_text(row["tournament"]),
        "city": _nullable_text(row["city"]),
        "country": _nullable_text(row["country"]),
        "neutral": _parse_bool(row["neutral"]),
        "home_team_confederation": _nullable_text(row["home_team_confederation"]),
        "away_team_confederation": _nullable_text(row["away_team_confederation"]),
        "home_team_fifa_rank": _nullable_float(row["home_team_fifa_rank"]),
        "home_team_fifa_points": _nullable_float(row["home_team_fifa_points"]),
        "away_team_fifa_rank": _nullable_float(row["away_team_fifa_rank"]),
        "away_team_fifa_points": _nullable_float(row["away_team_fifa_points"]),
        "source": "csv_import",
        "source_file": source_file,
    }
    record["match_key"] = build_match_key(record)
    return record


def load_records(csv_path: Path) -> tuple[list[dict], int]:
    deduped: dict[str, dict] = {}
    duplicates_replaced = 0
    source_file = csv_path.name

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record = parse_csv_row(row, source_file)
            fixture_key = build_fixture_identity_key(record)
            existing = deduped.get(fixture_key)
            if existing is not None:
                duplicates_replaced += 1
                if match_row_quality_score(record) <= match_row_quality_score(existing):
                    continue
            deduped[fixture_key] = record

    return list(deduped.values()), duplicates_replaced


def chunked(items: list[dict], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def get_supabase_admin_client():
    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_role_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
    return create_client(supabase_url, service_role_key)


def _upsert_batch(client, table_name: str, batch: list[dict], max_retries: int) -> None:
    attempt = 0
    while True:
        try:
            client.table(table_name).upsert(batch, on_conflict="match_key").execute()
            return
        except Exception:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_seconds = min(2 ** attempt, 10)
            print(f"batch_retry attempt={attempt} sleep_seconds={sleep_seconds}")
            time.sleep(sleep_seconds)


def _fetch_existing_rows_for_date_range(client, table_name: str, from_date: str, to_date: str) -> list[dict]:
    rows = []
    offset = 0
    select_cols = (
        "match_id,match_key,match_date,home_team,away_team,home_score,away_score,tournament,"
        "city,country,neutral,source,source_file,created_at"
    )
    while True:
        batch = (
            client.table(table_name)
            .select(select_cols)
            .gte("match_date", from_date)
            .lte("match_date", to_date)
            .order("match_date")
            .range(offset, offset + 999)
            .execute()
            .data
            or []
        )
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000
    return rows


def _delete_rows_by_match_id(client, table_name: str, match_ids: list[int]) -> int:
    if not match_ids:
        return 0
    deleted = 0
    for start in range(0, len(match_ids), 200):
        chunk = match_ids[start : start + 200]
        client.table(table_name).delete().in_("match_id", chunk).execute()
        deleted += len(chunk)
    return deleted


def _split_completed_vs_calendar(records: list[dict]) -> tuple[list[dict], list[dict]]:
    completed = []
    calendar_rows = []
    for record in records:
        if record.get("home_score") is None or record.get("away_score") is None:
            calendar_rows.append(
                {
                    "match_date": record.get("match_date"),
                    "home_team": record.get("home_team"),
                    "away_team": record.get("away_team"),
                    "tournament": record.get("tournament"),
                }
            )
            continue
        completed.append(record)
    return completed, calendar_rows


def _dedupe_calendar_rows(rows: list[dict]) -> list[dict]:
    deduped: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (
            str(row.get("match_date") or "").strip(),
            normalize_match_text(row.get("home_team")).casefold(),
            normalize_match_text(row.get("away_team")).casefold(),
        )
        deduped[key] = row
    return list(deduped.values())


def _upsert_calendar_rows(client, rows: list[dict], max_retries: int) -> int:
    if not rows:
        return 0
    payload = _dedupe_calendar_rows(rows)
    attempt = 0
    while True:
        try:
            client.table("matches_calendar").upsert(
                payload,
                on_conflict="home_team,away_team,match_date",
            ).execute()
            return len(payload)
        except Exception:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_seconds = min(2 ** attempt, 10)
            print(f"calendar_batch_retry attempt={attempt} sleep_seconds={sleep_seconds}")
            time.sleep(sleep_seconds)


def _normalize_compare_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        return normalize_match_text(value)
    return value


def _build_row_diff(existing: dict, incoming: dict) -> dict[str, dict[str, object]]:
    diffs: dict[str, dict[str, object]] = {}
    for field in [*IDENTITY_FIELDS, *ENRICHABLE_FIELDS, *NON_SUBSTANTIVE_FIELDS]:
        existing_value = _normalize_compare_value(existing.get(field))
        incoming_value = _normalize_compare_value(incoming.get(field))
        if existing_value != incoming_value:
            diffs[field] = {
                "existing": existing.get(field),
                "incoming": incoming.get(field),
            }
    return diffs


def _is_blank(value) -> bool:
    return _normalize_compare_value(value) in (None, "")


def _is_safe_enrichment_update(existing: dict, incoming: dict, diffs: dict[str, dict[str, object]]) -> bool:
    if not diffs:
        return False
    for field in diffs:
        if field in NON_SUBSTANTIVE_FIELDS:
            continue
        if field not in ENRICHABLE_FIELDS:
            return False
        if not _is_blank(existing.get(field)):
            return False
        if _is_blank(incoming.get(field)):
            return False
    return True


def _reconcile_records_with_existing(
    client,
    table_name: str,
    records: list[dict],
    resolve_conflicts: str = "report",
) -> tuple[list[dict], list[dict], list[int], list[dict], list[dict]]:
    if not records:
        return [], [], [], [], []

    date_values = [str(record["match_date"]) for record in records]
    existing_rows = _fetch_existing_rows_for_date_range(client, table_name, min(date_values), max(date_values))
    existing_by_fixture = {}
    for row in existing_rows:
        existing_by_fixture.setdefault(build_fixture_identity_key(row), []).append(row)

    incoming_by_fixture = {}
    for record in records:
        fixture_key = build_fixture_identity_key(record)
        current = incoming_by_fixture.get(fixture_key)
        if current is None or match_row_quality_score(record) > match_row_quality_score(current):
            incoming_by_fixture[fixture_key] = record

    inserts = []
    updates = []
    deletes = []
    conflicts = []
    resolved_conflicts = []
    for fixture_key, record in incoming_by_fixture.items():
        existing_group = existing_by_fixture.get(fixture_key, [])
        if not existing_group:
            inserts.append(record)
            continue

        canonical_existing = max(existing_group, key=lambda row: (match_row_quality_score(row), -(int(row.get("match_id") or 0))))
        for row in existing_group:
            match_id = row.get("match_id")
            if match_id != canonical_existing.get("match_id") and match_id is not None:
                deletes.append(int(match_id))

        diffs = _build_row_diff(canonical_existing, record)
        if not diffs:
            continue
        if set(diffs).issubset(set(NON_SUBSTANTIVE_FIELDS)):
            continue
        if _is_safe_enrichment_update(canonical_existing, record, diffs):
            updates.append({"match_id": int(canonical_existing["match_id"]), "payload": record})
            continue
        if resolve_conflicts == "incoming":
            updates.append({"match_id": int(canonical_existing["match_id"]), "payload": record})
            resolved_conflicts.append(
                {
                    "match_id": int(canonical_existing["match_id"]),
                    "match_date": record["match_date"],
                    "home_team": record["home_team"],
                    "away_team": record["away_team"],
                    "diff_fields": ",".join(sorted(diffs)),
                    "diffs": diffs,
                    "existing_source": canonical_existing.get("source"),
                    "existing_source_file": canonical_existing.get("source_file"),
                    "incoming_source_file": record.get("source_file"),
                }
            )
            continue
        conflicts.append(
            {
                "match_id": int(canonical_existing["match_id"]),
                "match_date": record["match_date"],
                "home_team": record["home_team"],
                "away_team": record["away_team"],
                "diff_fields": ",".join(sorted(diffs)),
                "diffs": diffs,
                "existing_source": canonical_existing.get("source"),
                "existing_source_file": canonical_existing.get("source_file"),
                "incoming_source_file": record.get("source_file"),
            }
        )

    return inserts, updates, deletes, conflicts, resolved_conflicts


def _write_conflicts_csv(conflicts: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "match_id",
                "match_date",
                "home_team",
                "away_team",
                "diff_fields",
                "existing_source",
                "existing_source_file",
                "incoming_source_file",
                "diffs",
            ],
        )
        writer.writeheader()
        for row in conflicts:
            serialized = dict(row)
            serialized["diffs"] = str(row.get("diffs") or "")
            writer.writerow(serialized)


def upload_records(
    csv_path: Path,
    table_name: str,
    batch_size: int,
    dry_run: bool,
    max_retries: int,
    conflicts_csv: Optional[Path] = None,
    resolve_conflicts: str = "report",
) -> None:
    records, duplicates_replaced = load_records(csv_path)
    print(f"csv_rows_loaded={len(records)} duplicates_replaced={duplicates_replaced} source={csv_path}")
    completed_records, calendar_rows = _split_completed_vs_calendar(records)
    print(
        f"record_split completed_rows={len(completed_records)} "
        f"calendar_rows={len(calendar_rows)}"
    )

    client = get_supabase_admin_client()
    inserts, updates, deletes, conflicts, resolved_conflicts = _reconcile_records_with_existing(
        client,
        table_name,
        completed_records,
        resolve_conflicts=resolve_conflicts,
    )

    if conflicts_csv is not None and conflicts:
        _write_conflicts_csv(conflicts, conflicts_csv)

    print(
        f"reconcile_summary inserts={len(inserts)} safe_updates={len(updates)} "
        f"deletes={len(deletes)} conflicts={len(conflicts)} resolved_conflicts={len(resolved_conflicts)}"
    )

    if dry_run:
        if conflicts_csv is not None and conflicts:
            print(f"conflicts_report={conflicts_csv}")
        print("dry_run=true no rows uploaded")
        return

    calendar_upserted = _upsert_calendar_rows(client, calendar_rows, max_retries=max_retries)

    uploaded = 0
    for batch in chunked(inserts, batch_size):
        _upsert_batch(client, table_name, batch, max_retries=max_retries)
        uploaded += len(batch)
        print(f"uploaded_rows={uploaded}")

    updated = 0
    for item in updates:
        client.table(table_name).update(item["payload"]).eq("match_id", item["match_id"]).execute()
        updated += 1

    deleted = _delete_rows_by_match_id(client, table_name, deletes)

    if conflicts_csv is not None and conflicts:
        print(f"conflicts_report={conflicts_csv}")
    print(
        f"upload_complete inserted_rows={uploaded} updated_rows={updated} "
        f"deleted_rows={deleted} conflict_rows={len(conflicts)} "
        f"resolved_conflict_rows={len(resolved_conflicts)} calendar_upserted_rows={calendar_upserted} "
        f"table={table_name}"
    )


def main():
    parser = argparse.ArgumentParser(description="Upload matches CSV into Supabase table with upsert.")
    parser.add_argument(
        "--csv-path",
        default=str(resolve_latest_matches_csv(Path("data"))),
        help="Path to the CSV file",
    )
    parser.add_argument("--table-name", default="matches", help="Supabase table name")
    parser.add_argument("--batch-size", type=int, default=500, help="Rows per upsert batch")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries per failed batch")
    parser.add_argument("--dry-run", action="store_true", help="Parse and dedupe without uploading")
    parser.add_argument(
        "--conflicts-csv",
        default="tmp/matches_upload_conflicts.csv",
        help="Path to write overlap conflicts report",
    )
    parser.add_argument(
        "--resolve-conflicts",
        choices=("report", "incoming"),
        default="report",
        help="How to handle overlapping rows with substantive differences",
    )
    args = parser.parse_args()

    upload_records(
        csv_path=Path(args.csv_path),
        table_name=args.table_name,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        conflicts_csv=Path(args.conflicts_csv) if args.conflicts_csv else None,
        resolve_conflicts=args.resolve_conflicts,
    )


if __name__ == "__main__":
    main()
