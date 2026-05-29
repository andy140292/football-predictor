from pathlib import Path


def normalize_match_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def build_match_key(row: dict) -> str:
    parts = [
        row["match_date"],
        normalize_match_text(row["home_team"]).lower(),
        normalize_match_text(row["away_team"]).lower(),
        normalize_match_text(row["tournament"]).lower(),
        normalize_match_text(row["city"]).lower(),
        normalize_match_text(row["country"]).lower(),
        "1" if row["neutral"] else "0",
    ]
    return "|".join(parts)


def build_fixture_identity_key(row: dict) -> str:
    match_date = str(row.get("match_date") or row.get("date") or "").strip()
    tournament = normalize_match_text(row.get("tournament")).lower()
    home_team = normalize_match_text(row.get("home_team")).lower()
    away_team = normalize_match_text(row.get("away_team")).lower()
    team_a, team_b = sorted([home_team, away_team])
    return "|".join([match_date, team_a, team_b, tournament])


def match_row_quality_score(row: dict) -> tuple[int, int, int, int, str]:
    city = normalize_match_text(row.get("city"))
    country = normalize_match_text(row.get("country"))
    source_file = str(row.get("source_file") or "").strip()
    source = str(row.get("source") or "").strip().lower()
    source_priority = {
        "manual": 4,
        "csv_import": 3,
        "sofascore": 2,
        "soccerway_feed": 1,
    }.get(source, 0)
    metadata_score = int(bool(city)) + int(bool(country)) + int(bool(source_file))
    explicit_neutral = int(row.get("neutral") is not None)
    created_at = str(row.get("created_at") or "")
    return (metadata_score, source_priority, explicit_neutral, -len(created_at), created_at)


def resolve_latest_matches_csv(data_dir: Path) -> Path:
    newest_file = None
    newest_name = ""
    for path in data_dir.glob("matches_*.csv"):
        if path.name > newest_name:
            newest_name = path.name
            newest_file = path

    if newest_file is not None:
        return newest_file
    return data_dir / "matches.csv"
