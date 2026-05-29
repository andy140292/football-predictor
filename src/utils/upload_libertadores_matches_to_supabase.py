import argparse
import csv
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

try:
    from matches_common import normalize_match_text
except ImportError:  # pragma: no cover - fallback for package-style execution
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from src.utils.matches_common import normalize_match_text


load_dotenv()


def _nullable_int(value: str):
    text = str(value or "").strip()
    if not text:
        return None
    return int(float(text))


def _parse_bool(value: str) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def parse_csv_row(row: dict, source_file: str) -> dict:
    return {
        "match_date": str(row.get("date") or "").strip(),
        "home_team": normalize_match_text(row.get("home_team")),
        "away_team": normalize_match_text(row.get("away_team")),
        "home_score": _nullable_int(row.get("home_score")),
        "away_score": _nullable_int(row.get("away_score")),
        "tournament": normalize_match_text(row.get("competition")),
        "season": _nullable_int(row.get("season")),
        "round": normalize_match_text(row.get("round")) or None,
        "neutral": _parse_bool(row.get("neutral")),
        "source": normalize_match_text(row.get("source")) or "csv_import",
        "source_file": normalize_match_text(row.get("source_file")) or source_file,
    }


def load_records(csv_path: Path) -> list[dict]:
    deduped: dict[tuple[str, str, str, str], dict] = {}
    source_file = csv_path.name

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record = parse_csv_row(row, source_file)
            key = (
                record["match_date"],
                record["home_team"].casefold(),
                record["away_team"].casefold(),
                record["tournament"].casefold(),
            )
            deduped[key] = record

    return list(deduped.values())


def chunked(items: list[dict], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def get_supabase_admin_client():
    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_role_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
    return create_client(supabase_url, service_role_key)


def upsert_rows(rows: list[dict], batch_size: int, max_retries: int) -> int:
    if not rows:
        return 0

    client = get_supabase_admin_client()
    uploaded = 0
    for batch in chunked(rows, batch_size):
        attempt = 0
        while True:
            try:
                client.table("libertadores_matches").upsert(
                    batch,
                    on_conflict="match_date,home_team,away_team,tournament",
                ).execute()
                break
            except Exception:
                attempt += 1
                if attempt > max_retries:
                    raise
                sleep_seconds = min(2 ** attempt, 10)
                print(f"batch_retry attempt={attempt} sleep_seconds={sleep_seconds}")
                time.sleep(sleep_seconds)
        uploaded += len(batch)
        print(f"uploaded_rows={uploaded}")
    return uploaded


def main():
    parser = argparse.ArgumentParser(description="Upload Libertadores club history CSV into Supabase.")
    parser.add_argument(
        "--csv-path",
        default="preprocessing_data/club_matches_historical_libertadores_2025_2026.csv",
        help="Path to the Libertadores history CSV",
    )
    parser.add_argument("--batch-size", type=int, default=500, help="Rows per upsert batch")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries per failed batch")
    parser.add_argument("--dry-run", action="store_true", help="Parse without uploading")
    args = parser.parse_args()

    rows = load_records(Path(args.csv_path))
    print(f"csv_rows_loaded={len(rows)} source={args.csv_path}")
    if args.dry_run:
        print("dry_run=true no rows uploaded")
        return

    uploaded = upsert_rows(rows, batch_size=args.batch_size, max_retries=args.max_retries)
    print(f"upload_complete uploaded_rows={uploaded} table=libertadores_matches")


if __name__ == "__main__":
    main()
