from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

from dotenv import load_dotenv
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.extract_libertadores_historical_fbref import (
    FbrefPlaywrightExtractor,
    TeamSource,
    ensure_columns,
    team_rows_to_canonical,
)
from src.utils.upload_libertadores_matches_to_supabase import upsert_rows


DEFAULT_ALIAS_PATH = ROOT_DIR / "preprocessing_data" / "club_team_aliases.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "preprocessing_data"
DEFAULT_TEAM_SOURCES_PATH = ROOT_DIR / "preprocessing_data" / "libertadores_team_sources_2026.csv"
COUNTRY_CODE_PREFIXES = {"ar", "bo", "br", "cl", "co", "ec", "pe", "py", "uy", "ve"}


def _normalized(value: str) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _strip_country_code_prefix(value: str) -> str:
    parts = str(value or "").strip().split()
    if len(parts) >= 2 and parts[0].casefold() in COUNTRY_CODE_PREFIXES:
        return " ".join(parts[1:]).strip()
    return str(value or "").strip()


def load_alias_groups(alias_path: Path) -> Dict[str, Set[str]]:
    groups: Dict[str, Set[str]] = defaultdict(set)
    if not alias_path.exists():
        return groups

    with alias_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            alias = str(row.get("alias") or row.get("source_name") or "").strip()
            canonical = str(row.get("canonical") or row.get("target_name") or "").strip()
            if not alias or not canonical:
                continue
            groups[_normalized(alias)].update({alias, canonical})
            groups[_normalized(canonical)].update({alias, canonical})
    return groups


def load_existing_table_team_names() -> Set[str]:
    from supabase import create_client

    load_dotenv(ROOT_DIR / ".env")
    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_role_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

    client = create_client(supabase_url, service_role_key)
    teams: Set[str] = set()
    page_size = 1000
    offset = 0
    while True:
        response = (
            client.table("libertadores_matches")
            .select("home_team,away_team")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = response.data or []
        for row in rows:
            home_team = str(row.get("home_team") or "").strip()
            away_team = str(row.get("away_team") or "").strip()
            if home_team:
                teams.add(home_team)
            if away_team:
                teams.add(away_team)
        if len(rows) < page_size:
            break
        offset += page_size
    return teams


def build_name_resolver(existing_team_names: Iterable[str], alias_groups: Dict[str, Set[str]]):
    existing_by_key: Dict[str, List[str]] = defaultdict(list)
    for name in existing_team_names:
        key = _normalized(name)
        if key:
            existing_by_key[key].append(name)

    def resolve(name: str) -> str:
        text = _strip_country_code_prefix(name)
        if not text:
            return text

        key = _normalized(text)
        direct_matches = existing_by_key.get(key, [])
        if direct_matches:
            return direct_matches[0]

        for candidate in sorted(alias_groups.get(key, set()), key=lambda value: (_normalized(value), value)):
            candidate_matches = existing_by_key.get(_normalized(candidate), [])
            if candidate_matches:
                return candidate_matches[0]

        return text

    return resolve


def remap_team_names(rows: List[dict], resolver) -> List[dict]:
    remapped: List[dict] = []
    for row in rows:
        item = dict(row)
        item["home_team"] = resolver(item.get("home_team"))
        item["away_team"] = resolver(item.get("away_team"))
        remapped.append(item)
    return remapped


def dedupe_rows(rows: List[dict]) -> List[dict]:
    deduped: Dict[tuple[str, str, str, str], dict] = {}
    for row in rows:
        key = (
            str(row.get("date") or ""),
            str(row.get("home_team") or ""),
            str(row.get("away_team") or ""),
            str(row.get("competition") or ""),
        )
        deduped[key] = row
    return list(deduped.values())


def load_team_sources(team_sources_path: Path) -> List[TeamSource]:
    with team_sources_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        sources = []
        for row in reader:
            team = str(row.get("team") or "").strip()
            team_country = str(row.get("team_country") or "").strip()
            team_country_code = str(row.get("team_country_code") or "").strip().lower()
            team_url = str(row.get("team_url") or "").strip()
            if team and team_url:
                sources.append(
                    TeamSource(
                        team=team,
                        team_country=team_country,
                        team_country_code=team_country_code,
                        team_url=team_url,
                    )
                )
    return sources


def fetch_recent_team_rows(
    team_sources: List[TeamSource],
    season_year: str,
    start_date: str,
    headless: bool,
    user_data_dir: str | None,
    manual_challenge_seconds: int,
    cdp_url: str | None,
) -> tuple[List[dict], List[TeamSource]]:
    extractor = FbrefPlaywrightExtractor(
        headless=headless,
        user_data_dir=user_data_dir,
        manual_challenge_seconds=manual_challenge_seconds,
        cdp_url=cdp_url,
    )
    all_team_rows: List[dict] = []
    with extractor:
        for source in team_sources:
            rows = extractor.fetch_team_matchlogs(source=source, season_year=season_year)
            all_team_rows.extend(rows)

    team_rows_df = ensure_columns(pd.DataFrame(all_team_rows), [
        "date",
        "time",
        "competition",
        "round",
        "day",
        "venue",
        "result",
        "gf",
        "ga",
        "opponent",
        "possession",
        "attendance",
        "captain",
        "formation",
        "opp_formation",
        "referee",
        "match_report",
        "notes",
        "team",
        "team_country",
        "team_url",
        "source_competition_page",
        "source_matchlog_page",
        "fbref_season",
        "scraped_at_utc",
    ])
    canonical = team_rows_to_canonical(team_rows_df).to_dict("records")
    filtered = [row for row in canonical if str(row.get("date") or "") >= start_date]
    return filtered, team_sources


def write_csv(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill recent Libertadores club matches from FBref.")
    parser.add_argument("--season-year", default="2026")
    parser.add_argument("--start-date", default="2026-04-01")
    parser.add_argument("--alias-path", default=str(DEFAULT_ALIAS_PATH))
    parser.add_argument("--team-sources-path", default=str(DEFAULT_TEAM_SOURCES_PATH))
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_DIR / "libertadores_recent_matches_2026_04_01.csv"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--user-data-dir", default="")
    parser.add_argument("--manual-challenge-seconds", type=int, default=0)
    parser.add_argument("--cdp-url", default="")
    args = parser.parse_args()

    existing_team_names = load_existing_table_team_names()
    alias_groups = load_alias_groups(Path(args.alias_path))
    resolver = build_name_resolver(existing_team_names, alias_groups)
    team_sources = load_team_sources(Path(args.team_sources_path))

    rows, team_sources = fetch_recent_team_rows(
        team_sources=team_sources,
        season_year=str(args.season_year),
        start_date=str(args.start_date),
        headless=not args.headed,
        user_data_dir=args.user_data_dir or None,
        manual_challenge_seconds=args.manual_challenge_seconds,
        cdp_url=args.cdp_url or None,
    )
    rows = dedupe_rows(remap_team_names(rows, resolver))
    rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("date") or ""),
            str(row.get("home_team") or ""),
            str(row.get("away_team") or ""),
            str(row.get("competition") or ""),
        ),
    )

    output_csv = Path(args.output_csv)
    write_csv(rows, output_csv)
    print(f"team_sources={len(team_sources)} recent_rows={len(rows)} output_csv={output_csv}")

    if args.dry_run:
        return

    upload_rows = []
    for row in rows:
        upload_rows.append(
            {
                "match_date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_score": int(float(row["home_score"])),
                "away_score": int(float(row["away_score"])),
                "tournament": row["competition"],
                "season": int(row["season"]) if str(row.get("season") or "").strip() else None,
                "round": row["round"] or None,
                "neutral": bool(int(row.get("neutral", 0))),
                "source": row["source"],
                "source_file": row["source_file"],
            }
        )

    uploaded = upsert_rows(upload_rows, batch_size=500, max_retries=5)
    print(f"uploaded_rows={uploaded}")


if __name__ == "__main__":
    main()
