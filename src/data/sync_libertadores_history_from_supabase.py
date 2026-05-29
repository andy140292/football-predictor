from __future__ import annotations

import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable

from dotenv import load_dotenv
from supabase import create_client


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


HISTORY_CSV_PATH = ROOT_DIR / "preprocessing_data" / "club_matches_historical_libertadores_2025_2026.csv"
COEFFICIENTS_PATH = ROOT_DIR / "preprocessing_data" / "libertadores_conmebol_coefficients_2026.csv"

FIELDNAMES = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "result",
    "competition",
    "country",
    "season",
    "round",
    "neutral",
    "source",
    "source_file",
    "div_code",
]


def _normalized(value: str) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _parse_int(value) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def _compute_result(home_score: int | None, away_score: int | None) -> str:
    if home_score is None or away_score is None:
        return ""
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def _season_from_date(date_text: str) -> str:
    return str(date_text).split("-", 1)[0]


def load_competition_country_map(rows: Iterable[dict]) -> Dict[str, str]:
    country_counts: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        competition = str(row.get("competition") or "").strip()
        country = str(row.get("country") or "").strip()
        if competition and country:
            country_counts[_normalized(competition)][country] += 1
    return {
        key: counter.most_common(1)[0][0]
        for key, counter in country_counts.items()
        if counter
    }


def load_team_country_map() -> Dict[str, str]:
    if not COEFFICIENTS_PATH.exists():
        return {}
    with COEFFICIENTS_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        mapping = {}
        for row in reader:
            team = str(row.get("team") or row.get("display_name") or "").strip()
            country = str(row.get("country") or "").strip()
            if team and country:
                mapping[_normalized(team)] = country
    return mapping


def infer_country(competition: str, home_team: str, away_team: str, competition_country_map: Dict[str, str], team_country_map: Dict[str, str]) -> str:
    competition_text = str(competition or "").strip()
    competition_key = _normalized(competition_text)
    if competition_key in competition_country_map:
        return competition_country_map[competition_key]
    if "libertadores" in competition_key:
        return "South America"
    return (
        team_country_map.get(_normalized(home_team))
        or team_country_map.get(_normalized(away_team))
        or ""
    )


def load_existing_csv_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fetch_all_supabase_rows() -> list[dict]:
    load_dotenv(ROOT_DIR / ".env")
    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_role_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

    client = create_client(supabase_url, service_role_key)
    rows = []
    page_size = 1000
    offset = 0
    while True:
        result = (
            client.table("libertadores_matches")
            .select("*")
            .order("match_date", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = result.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def is_completed_match(row: dict) -> bool:
    return _parse_int(row.get("home_score")) is not None and _parse_int(row.get("away_score")) is not None


def supabase_to_csv_row(row: dict, competition_country_map: Dict[str, str], team_country_map: Dict[str, str]) -> dict:
    date_text = str(row.get("match_date") or "").strip()
    home_team = str(row.get("home_team") or "").strip()
    away_team = str(row.get("away_team") or "").strip()
    competition = str(row.get("tournament") or "").strip()
    home_score = _parse_int(row.get("home_score"))
    away_score = _parse_int(row.get("away_score"))
    return {
        "date": date_text,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": "" if home_score is None else str(home_score),
        "away_score": "" if away_score is None else str(away_score),
        "result": _compute_result(home_score, away_score),
        "competition": competition,
        "country": infer_country(competition, home_team, away_team, competition_country_map, team_country_map),
        "season": str(row.get("season") or _season_from_date(date_text)),
        "round": str(row.get("round") or "").strip(),
        "neutral": "1" if bool(row.get("neutral")) else "0",
        "source": str(row.get("source") or "fbref.com").strip(),
        "source_file": str(row.get("source_file") or "").strip(),
        "div_code": "",
    }


def merge_rows(existing_rows: list[dict], supabase_rows: list[dict]) -> list[dict]:
    competition_country_map = load_competition_country_map(existing_rows)
    team_country_map = load_team_country_map()

    deduped: Dict[tuple[str, str, str, str], dict] = {}
    for row in existing_rows:
        if _parse_int(row.get("home_score")) is None or _parse_int(row.get("away_score")) is None:
            continue
        key = (
            str(row.get("date") or "").strip(),
            _normalized(row.get("home_team")),
            _normalized(row.get("away_team")),
            _normalized(row.get("competition")),
        )
        deduped[key] = {field: str(row.get(field) or "").strip() for field in FIELDNAMES}

    for row in supabase_rows:
        if not is_completed_match(row):
            continue
        csv_row = supabase_to_csv_row(row, competition_country_map, team_country_map)
        key = (
            csv_row["date"],
            _normalized(csv_row["home_team"]),
            _normalized(csv_row["away_team"]),
            _normalized(csv_row["competition"]),
        )
        deduped[key] = csv_row

    return sorted(
        deduped.values(),
        key=lambda row: (
            row["date"],
            row["home_team"],
            row["away_team"],
            row["competition"],
        ),
    )


def write_rows(csv_path: Path, rows: list[dict]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    existing_rows = load_existing_csv_rows(HISTORY_CSV_PATH)
    supabase_rows = fetch_all_supabase_rows()
    merged_rows = merge_rows(existing_rows, supabase_rows)
    write_rows(HISTORY_CSV_PATH, merged_rows)
    print(
        f"existing_rows={len(existing_rows)} "
        f"supabase_rows={len(supabase_rows)} "
        f"merged_rows={len(merged_rows)} "
        f"output={HISTORY_CSV_PATH}"
    )


if __name__ == "__main__":
    main()
