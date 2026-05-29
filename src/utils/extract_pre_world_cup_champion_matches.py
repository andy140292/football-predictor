import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from matches_common import resolve_latest_matches_csv
except ImportError:  # pragma: no cover - fallback for package-style execution
    from src.utils.matches_common import resolve_latest_matches_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATCHES_PATH = resolve_latest_matches_csv(PROJECT_ROOT / "data")
DEFAULT_ALIAS_PATH = PROJECT_ROOT / "data" / "team_code_aliases.csv"
DEFAULT_COUNTRY_CODES_PATH = PROJECT_ROOT / "data" / "fifa_country_codes.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "world_cup_champions_last5_pre_world_cup.csv"
DEFAULT_RANKINGS_PATH = Path("/Users/andresuribe/Downloads/archive (9)/fifa_ranking-2024-06-20.csv")

CHAMPION_YEARS: Sequence[Tuple[str, int]] = (
    ("Brazil", 2002),
    ("Italy", 2006),
    ("Spain", 2010),
    ("Germany", 2014),
    ("France", 2018),
    ("Argentina", 2022),
)

ADDED_FIELDNAMES: Sequence[str] = (
    "champion_team",
    "world_cup_year",
    "first_world_cup_match_date",
    "selected_rank_date",
    "champion_rank_pre_wc",
    "opponent_rank_pre_wc",
)

TEAM_CODE_OVERRIDES: Dict[str, str] = {
    "Russian Federation": "RUS",
}

IRRELEVANT_TOURNAMENTS = {
    "Viva World Cup",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the last five matches each selected World Cup champion played before its title-winning tournament."
    )
    parser.add_argument("--matches-path", type=Path, default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--rankings-path", type=Path, default=DEFAULT_RANKINGS_PATH)
    parser.add_argument("--alias-path", type=Path, default=DEFAULT_ALIAS_PATH)
    parser.add_argument("--country-codes-path", type=Path, default=DEFAULT_COUNTRY_CODES_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def load_team_code_lookup(alias_path: Path, country_codes_path: Path) -> Dict[str, str]:
    lookup: Dict[str, str] = {}

    with alias_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lookup[row["alias_name"]] = row["team_code"]

    with country_codes_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lookup[row["country_name"]] = row["team_code"]

    lookup.update(TEAM_CODE_OVERRIDES)
    return lookup


def load_rankings_by_date(rankings_path: Path) -> Dict[str, Dict[str, str]]:
    rankings: Dict[str, Dict[str, str]] = {}

    with rankings_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rankings.setdefault(row["rank_date"], {})[row["country_abrv"]] = row["rank"]

    return rankings


def is_relevant_tournament(tournament: str) -> bool:
    return bool(tournament) and tournament not in IRRELEVANT_TOURNAMENTS


def get_selected_rank_date(rank_dates: Iterable[str], first_world_cup_match_date: str) -> str:
    eligible_dates = [rank_date for rank_date in rank_dates if rank_date <= first_world_cup_match_date]
    if not eligible_dates:
        raise ValueError(f"No FIFA ranking date available on or before {first_world_cup_match_date}")
    return max(eligible_dates)


def get_first_world_cup_match(matches_rows: Sequence[Dict[str, str]], champion_team: str, world_cup_year: int) -> Dict[str, str]:
    matches = [
        row
        for row in matches_rows
        if row["tournament"] == "FIFA World Cup"
        and row["date"].startswith(str(world_cup_year))
        and champion_team in (row["home_team"], row["away_team"])
    ]
    if not matches:
        raise ValueError(f"Missing FIFA World Cup matches for {champion_team} in {world_cup_year}")
    return min(matches, key=lambda row: row["date"])


def get_last_five_matches(
    matches_rows: Sequence[Dict[str, str]],
    champion_team: str,
    first_world_cup_match_date: str,
) -> List[Dict[str, str]]:
    prior_matches = [
        row
        for row in matches_rows
        if row["date"] < first_world_cup_match_date
        and champion_team in (row["home_team"], row["away_team"])
        and is_relevant_tournament(row["tournament"])
    ]
    prior_matches.sort(key=lambda row: row["date"])
    selected = prior_matches[-5:]
    if len(selected) != 5:
        raise ValueError(f"Expected 5 matches for {champion_team}, found {len(selected)}")
    return selected


def get_team_rank(team_name: str, rank_snapshot: Dict[str, str], team_code_lookup: Dict[str, str]) -> str:
    team_code = team_code_lookup.get(team_name)
    if not team_code:
        return ""
    return rank_snapshot.get(team_code, "")


def enrich_match_row(
    row: Dict[str, str],
    champion_team: str,
    world_cup_year: int,
    first_world_cup_match_date: str,
    selected_rank_date: str,
    rank_snapshot: Dict[str, str],
    team_code_lookup: Dict[str, str],
) -> Dict[str, str]:
    opponent = row["away_team"] if row["home_team"] == champion_team else row["home_team"]
    enriched_row = dict(row)
    enriched_row.update(
        {
            "champion_team": champion_team,
            "world_cup_year": str(world_cup_year),
            "first_world_cup_match_date": first_world_cup_match_date,
            "selected_rank_date": selected_rank_date,
            "champion_rank_pre_wc": get_team_rank(champion_team, rank_snapshot, team_code_lookup),
            "opponent_rank_pre_wc": get_team_rank(opponent, rank_snapshot, team_code_lookup),
        }
    )
    return enriched_row


def extract_matches(
    matches_rows: Sequence[Dict[str, str]],
    rankings_by_date: Dict[str, Dict[str, str]],
    team_code_lookup: Dict[str, str],
) -> List[Dict[str, str]]:
    extracted_rows: List[Dict[str, str]] = []
    rank_dates = rankings_by_date.keys()

    for champion_team, world_cup_year in CHAMPION_YEARS:
        first_world_cup_match = get_first_world_cup_match(matches_rows, champion_team, world_cup_year)
        first_world_cup_match_date = first_world_cup_match["date"]
        selected_rank_date = get_selected_rank_date(rank_dates, first_world_cup_match_date)
        rank_snapshot = rankings_by_date[selected_rank_date]

        for row in get_last_five_matches(matches_rows, champion_team, first_world_cup_match_date):
            extracted_rows.append(
                enrich_match_row(
                    row=row,
                    champion_team=champion_team,
                    world_cup_year=world_cup_year,
                    first_world_cup_match_date=first_world_cup_match_date,
                    selected_rank_date=selected_rank_date,
                    rank_snapshot=rank_snapshot,
                    team_code_lookup=team_code_lookup,
                )
            )

    return extracted_rows


def write_output(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: Sequence[Dict[str, str]]) -> List[str]:
    summary_lines: List[str] = []
    current_group = None

    for row in rows:
        group = (row["champion_team"], row["world_cup_year"], row["selected_rank_date"])
        if group != current_group:
            champion_team, world_cup_year, selected_rank_date = group
            summary_lines.append(f"{champion_team} {world_cup_year} (ranking date {selected_rank_date})")
            current_group = group

        opponent = row["away_team"] if row["home_team"] == row["champion_team"] else row["home_team"]
        summary_lines.append(
            f"  {row['date']} | {row['tournament']} | {row['home_team']} vs {row['away_team']} | opponent: {opponent} | champion rank {row['champion_rank_pre_wc'] or 'NA'} | opponent rank {row['opponent_rank_pre_wc'] or 'NA'}"
        )

    return summary_lines


def main() -> None:
    args = parse_args()
    original_fieldnames, matches_rows = read_csv_rows(args.matches_path)
    rankings_by_date = load_rankings_by_date(args.rankings_path)
    team_code_lookup = load_team_code_lookup(args.alias_path, args.country_codes_path)
    extracted_rows = extract_matches(matches_rows, rankings_by_date, team_code_lookup)
    output_fieldnames = original_fieldnames + list(ADDED_FIELDNAMES)
    write_output(args.output_path, output_fieldnames, extracted_rows)

    print(f"Wrote {len(extracted_rows)} rows to {args.output_path}")
    for line in build_summary(extracted_rows):
        print(line)


if __name__ == "__main__":
    main()
