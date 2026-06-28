"""Bulk upload matches into /admin/matches-calendar/upsert-batch from a CSV file.

Expected CSV columns:
- home_team
- away_team
- match_date (YYYY-MM-DD)

Optional CSV columns:
- tournament
- home_team_code
- away_team_code
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import re
import unicodedata

import requests
from dotenv import load_dotenv


@dataclass
class LocalValidationError:
    row_index: int
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bulk upsert matches_calendar rows through backend admin API."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV with columns: home_team, away_team, match_date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000"),
        help="Backend base URL (default: BACKEND_BASE_URL env or http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--admin-key",
        default=os.getenv("ADMIN_API_KEY"),
        help="Admin API key (default: ADMIN_API_KEY env).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Rows per API request (default: 200).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds per request (default: 30).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize without calling the API.",
    )
    parser.add_argument(
        "--no-resolve-codes",
        action="store_true",
        help="Do not resolve team codes from local maps.",
    )
    parser.add_argument(
        "--allow-missing-codes",
        action="store_true",
        help="Allow rows without resolved team codes.",
    )
    parser.add_argument(
        "--fifa-codes-csv",
        default="data/fifa_country_codes.csv",
        help="Path to static FIFA code map CSV (default: data/fifa_country_codes.csv).",
    )
    parser.add_argument(
        "--alias-csv",
        default="data/team_code_aliases.csv",
        help="Path to team alias map CSV (default: data/team_code_aliases.csv).",
    )
    return parser.parse_args()


def _normalize_header(value: str) -> str:
    return str(value or "").strip().lower().replace("\ufeff", "")


def _parse_iso_date(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return datetime.strptime(raw[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        return ""


def _normalized_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch for ch in text if ch.isalnum())


def _normalize_team_code(value: str) -> str:
    code = str(value or "").strip().upper()
    return code if re.fullmatch(r"[A-Z]{3}", code) else ""


def load_code_maps(fifa_codes_path: Path, alias_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    name_to_code: Dict[str, str] = {}
    alias_to_code: Dict[str, str] = {}

    if fifa_codes_path.exists():
        with fifa_codes_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                code = _normalize_team_code(
                    row.get("team_code") or row.get("code") or row.get("fifa_code") or row.get("Code")
                )
                name = str(
                    row.get("country_name") or row.get("country") or row.get("Country") or row.get("team") or ""
                ).strip()
                if code and name:
                    name_to_code[_normalized_text(name)] = code

    if alias_path.exists():
        with alias_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                code = _normalize_team_code(
                    row.get("team_code") or row.get("code") or row.get("fifa_code")
                )
                alias = str(
                    row.get("alias_name") or row.get("alias") or row.get("team_name") or row.get("name") or ""
                ).strip()
                if code and alias:
                    alias_to_code[_normalized_text(alias)] = code

    return name_to_code, alias_to_code


def resolve_team_code(name: str, name_to_code: Dict[str, str], alias_to_code: Dict[str, str]) -> str:
    key = _normalized_text(name)
    if key in alias_to_code:
        return alias_to_code[key]
    return name_to_code.get(key, "")


def read_and_validate_csv(
    path: Path,
    resolve_codes: bool,
    allow_missing_codes: bool,
    name_to_code: Dict[str, str],
    alias_to_code: Dict[str, str],
) -> Tuple[List[Dict[str, str]], List[LocalValidationError]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    rows: List[Dict[str, str]] = []
    errors: List[LocalValidationError] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")

        normalized_to_original = {_normalize_header(name): name for name in reader.fieldnames}
        required_cols = ["home_team", "away_team", "match_date"]
        missing = [col for col in required_cols if col not in normalized_to_original]
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

        for idx, source_row in enumerate(reader, start=2):
            home_team = str(source_row.get(normalized_to_original["home_team"], "")).strip()
            away_team = str(source_row.get(normalized_to_original["away_team"], "")).strip()
            match_date = _parse_iso_date(
                str(source_row.get(normalized_to_original["match_date"], "")).strip()
            )
            tournament = ""
            if "tournament" in normalized_to_original:
                tournament = str(source_row.get(normalized_to_original["tournament"], "")).strip()

            if not home_team:
                errors.append(LocalValidationError(row_index=idx, reason="home_team is required"))
                continue
            if not away_team:
                errors.append(LocalValidationError(row_index=idx, reason="away_team is required"))
                continue
            if home_team.casefold() == away_team.casefold():
                errors.append(
                    LocalValidationError(
                        row_index=idx, reason="home_team and away_team must be different"
                    )
                )
                continue
            if not match_date:
                errors.append(
                    LocalValidationError(
                        row_index=idx, reason="match_date must use YYYY-MM-DD format"
                    )
                )
                continue

            home_team_code = ""
            away_team_code = ""
            if "home_team_code" in normalized_to_original:
                home_team_code = _normalize_team_code(
                    str(source_row.get(normalized_to_original["home_team_code"], "")).strip()
                )
            if "away_team_code" in normalized_to_original:
                away_team_code = _normalize_team_code(
                    str(source_row.get(normalized_to_original["away_team_code"], "")).strip()
                )

            if resolve_codes:
                if not home_team_code:
                    home_team_code = resolve_team_code(home_team, name_to_code, alias_to_code)
                if not away_team_code:
                    away_team_code = resolve_team_code(away_team, name_to_code, alias_to_code)

            if (home_team_code and not away_team_code) or (away_team_code and not home_team_code):
                errors.append(
                    LocalValidationError(
                        row_index=idx,
                        reason="both home_team_code and away_team_code are required when one is present",
                    )
                )
                continue

            if home_team_code and away_team_code and home_team_code == away_team_code:
                errors.append(
                    LocalValidationError(
                        row_index=idx,
                        reason="home_team_code and away_team_code must be different",
                    )
                )
                continue

            if resolve_codes and (not home_team_code or not away_team_code) and not allow_missing_codes:
                errors.append(
                    LocalValidationError(
                        row_index=idx,
                        reason=f"could not resolve FIFA codes for '{home_team}' vs '{away_team}'",
                    )
                )
                continue

            payload = {"home_team": home_team, "away_team": away_team, "match_date": match_date}
            if tournament:
                payload["tournament"] = tournament
            if home_team_code and away_team_code:
                payload["home_team_code"] = home_team_code
                payload["away_team_code"] = away_team_code
            rows.append(payload)

    return rows, errors


def chunked(values: List[Dict[str, str]], size: int) -> List[List[Dict[str, str]]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def main() -> int:
    load_dotenv()
    args = parse_args()

    if args.chunk_size <= 0:
        print("chunk-size must be greater than 0", file=sys.stderr)
        return 2

    if not args.admin_key and not args.dry_run:
        print("Missing admin key. Set --admin-key or ADMIN_API_KEY env.", file=sys.stderr)
        return 2

    csv_path = Path(args.csv).expanduser().resolve()
    fifa_codes_path = Path(args.fifa_codes_csv).expanduser().resolve()
    alias_path = Path(args.alias_csv).expanduser().resolve()
    resolve_codes = not args.no_resolve_codes

    name_to_code: Dict[str, str] = {}
    alias_to_code: Dict[str, str] = {}
    if resolve_codes:
        name_to_code, alias_to_code = load_code_maps(fifa_codes_path, alias_path)
        if not name_to_code and not alias_to_code:
            print(
                "Warning: code resolution enabled but no code maps were loaded.",
                file=sys.stderr,
            )

    try:
        valid_rows, local_errors = read_and_validate_csv(
            csv_path,
            resolve_codes=resolve_codes,
            allow_missing_codes=args.allow_missing_codes,
            name_to_code=name_to_code,
            alias_to_code=alias_to_code,
        )
    except Exception as exc:
        print(f"Failed to read CSV: {exc}", file=sys.stderr)
        return 2

    summary = {
        "received": len(valid_rows) + len(local_errors),
        "inserted": 0,
        "updated": 0,
        "skipped": len(local_errors),
        "errors": [{"row_index": e.row_index, "reason": e.reason} for e in local_errors],
    }

    if args.dry_run:
        print(json.dumps({"mode": "dry-run", "summary": summary}, indent=2, ensure_ascii=False))
        return 0

    url = args.base_url.rstrip("/") + "/admin/matches-calendar/upsert-batch"
    headers = {
        "Content-Type": "application/json",
        "X-Admin-Key": args.admin_key,
    }

    request_failed = False
    for chunk_idx, batch in enumerate(chunked(valid_rows, args.chunk_size), start=1):
        response = requests.post(
            url,
            headers=headers,
            json={"matches": batch},
            timeout=args.timeout,
        )
        if response.status_code != 200:
            request_failed = True
            summary["errors"].append(
                {
                    "row_index": -1,
                    "reason": f"chunk {chunk_idx} failed with status {response.status_code}: {response.text}",
                }
            )
            continue

        payload = response.json()
        summary["inserted"] += int(payload.get("inserted", 0))
        summary["updated"] += int(payload.get("updated", 0))
        summary["skipped"] += int(payload.get("skipped", 0))
        chunk_errors = payload.get("errors", []) or []
        for err in chunk_errors:
            summary["errors"].append(
                {
                    "row_index": err.get("row_index", -1),
                    "reason": f"chunk {chunk_idx}: {err.get('reason', 'unknown')}",
                }
            )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if request_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
