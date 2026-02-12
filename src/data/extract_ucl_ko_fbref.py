"""Extract UEFA Champions League KO playoff team match logs from FBref.

Outputs:
- data/ucl_ko_teams_scores_fixtures_2025_2026.csv
- data/ucl_ko_playoff_first_legs_2026_02_17_18.csv
- data/ucl_ko_team_sources_2025_2026.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urljoin

import pandas as pd

FBREF_BASE = "https://fbref.com"
UCL_SCHEDULE_URL = f"{FBREF_BASE}/en/comps/8/schedule/Champions-League-Scores-and-Fixtures"
KO_ROUND_LABEL = "knockout phase play-offs"
FIRST_LEG_DATES = {"2026-02-17", "2026-02-18"}

TEAM_FIXTURE_COLUMNS = [
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
    "team_url",
    "source_competition_page",
    "scraped_at_utc",
]

FIRST_LEG_COLUMNS = [
    "date",
    "time",
    "home_team",
    "away_team",
    "home_team_url",
    "away_team_url",
    "round",
]

TEAM_SOURCE_COLUMNS = ["team", "team_url"]

COUNTRY_CODES = {
    "ad", "ae", "al", "am", "ar", "at", "az", "ba", "be", "bg", "br", "by", "ch", "cl",
    "co", "cy", "cz", "de", "dk", "dz", "ec", "eg", "eng", "es", "fi", "fr", "ge", "gh",
    "gr", "hr", "hu", "id", "ie", "il", "in", "ir", "is", "it", "jp", "kr", "mx", "ng",
    "nl", "no", "pl", "pt", "ro", "rs", "ru", "sa", "sc", "se", "si", "sk", "tr", "ua",
    "us", "uy", "ve", "wa",
}


@dataclass(frozen=True)
class TeamSource:
    team: str
    team_url: str


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def normalize_team_display(raw: str) -> str:
    """Normalize schedule cell team text by removing country-code tokens."""
    text = normalize_whitespace(raw)
    if not text:
        return text

    tokens = text.split(" ")
    if tokens and tokens[-1].lower() in COUNTRY_CODES:
        tokens = tokens[:-1]
    if tokens and tokens[0].lower() in COUNTRY_CODES:
        tokens = tokens[1:]

    return normalize_whitespace(" ".join(tokens))


def normalize_url(path_or_url: str) -> str:
    if not path_or_url:
        return ""
    absolute = urljoin(FBREF_BASE, path_or_url)
    return absolute.split("?", 1)[0]


def is_ko_playoff_round(round_label: str) -> bool:
    return KO_ROUND_LABEL in normalize_whitespace(round_label).lower()


def parse_team_slug(team_url: str) -> str:
    match = re.search(r"/en/squads/[^/]+/([^/?#]+)-Stats$", team_url)
    if not match:
        raise ValueError(f"Unexpected FBref squad URL format: {team_url}")
    return match.group(1)


def build_matchlog_url(team_url: str, season: str = "2025-2026") -> str:
    team_url = normalize_url(team_url)
    team_slug = parse_team_slug(team_url)
    base = team_url.rsplit("/", 1)[0]
    return (
        f"{base}/{season}/matchlogs/all_comps/schedule/"
        f"{team_slug}-Scores-and-Fixtures-All-Competitions"
    )


def dedupe_team_sources(candidates: Iterable[TeamSource]) -> List[TeamSource]:
    seen: Dict[str, TeamSource] = {}
    for source in candidates:
        team = normalize_team_display(source.team)
        url = normalize_url(source.team_url)
        if not team or not url:
            continue
        if team not in seen:
            seen[team] = TeamSource(team=team, team_url=url)
    return sorted(seen.values(), key=lambda item: item.team.lower())


def filter_first_legs(fixtures: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        {
            "date": row.get("date", ""),
            "time": row.get("time", ""),
            "home_team": normalize_team_display(row.get("home_team", "")),
            "away_team": normalize_team_display(row.get("away_team", "")),
            "home_team_url": normalize_url(row.get("home_team_url", "")),
            "away_team_url": normalize_url(row.get("away_team_url", "")),
            "round": normalize_whitespace(row.get("round", "")),
        }
        for row in fixtures
        if row.get("date") in FIRST_LEG_DATES and is_ko_playoff_round(row.get("round", ""))
    ]


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df.loc[:, list(columns)]


class FbrefPlaywrightExtractor:
    def __init__(self, headless: bool = True, max_attempts: int = 3):
        self.headless = headless
        self.max_attempts = max_attempts

    def _goto_with_retry(self, page, url: str) -> None:
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                self._wait_for_challenge(page)
                return
            except Exception as exc:  # pragma: no cover - runtime path
                last_error = exc
                if attempt == self.max_attempts:
                    break
                time.sleep(2 * attempt)
        raise RuntimeError(f"Failed navigation after {self.max_attempts} attempts: {url}") from last_error

    def _wait_for_challenge(self, page) -> None:
        deadline = time.time() + 45
        while time.time() < deadline:
            title = page.title().strip().lower()
            if "just a moment" not in title:
                return
            time.sleep(1)
        raise TimeoutError("Cloudflare challenge did not clear in time")

    def _wait_for_stats_table(self, page) -> None:
        page.wait_for_selector("table.stats_table", timeout=30000)

    def fetch_ko_schedule(self) -> Tuple[List[Dict[str, str]], List[TeamSource]]:
        from playwright.sync_api import sync_playwright  # lazy import

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page = browser.new_page()
            self._goto_with_retry(page, UCL_SCHEDULE_URL)
            self._wait_for_stats_table(page)

            fixtures = page.evaluate(
                """
                () => {
                  const rows = Array.from(document.querySelectorAll('table.stats_table tbody tr'));
                  return rows.map((tr) => {
                    const cells = tr.querySelectorAll('th,td');
                    return {
                      round: (cells[0]?.textContent || '').trim(),
                      date: (cells[3]?.textContent || '').trim(),
                      time: (cells[4]?.textContent || '').trim(),
                      home_team: (cells[5]?.textContent || '').trim(),
                      home_team_url: cells[5]?.querySelector('a')?.getAttribute('href') || '',
                      away_team: (cells[7]?.textContent || '').trim(),
                      away_team_url: cells[7]?.querySelector('a')?.getAttribute('href') || '',
                    };
                  }).filter((row) => row.round && row.date);
                }
                """
            )

            browser.close()

        ko_rows = [row for row in fixtures if is_ko_playoff_round(row.get("round", ""))]
        sources = dedupe_team_sources(
            [
                TeamSource(team=row.get("home_team", ""), team_url=row.get("home_team_url", ""))
                for row in ko_rows
            ]
            + [
                TeamSource(team=row.get("away_team", ""), team_url=row.get("away_team_url", ""))
                for row in ko_rows
            ]
        )
        return ko_rows, sources

    def fetch_team_matchlogs(
        self,
        source: TeamSource,
        source_competition_page: str,
        season: str = "2025-2026",
    ) -> List[Dict[str, str]]:
        from playwright.sync_api import sync_playwright  # lazy import

        matchlog_url = build_matchlog_url(source.team_url, season=season)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page = browser.new_page()
            self._goto_with_retry(page, matchlog_url)
            self._wait_for_stats_table(page)

            rows = page.evaluate(
                r"""
                () => {
                  const table = document.querySelector('table.stats_table');
                  if (!table) return [];
                  const headers = Array.from(table.querySelectorAll('thead tr:last-child th')).map((th) => (th.textContent || '').trim());
                  const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim();
                  const out = [];
                  for (const tr of table.querySelectorAll('tbody tr')) {
                    if (tr.classList.contains('thead')) continue;
                    const cells = Array.from(tr.querySelectorAll('th,td'));
                    if (!cells.length) continue;
                    const row = {};
                    for (let i = 0; i < headers.length && i < cells.length; i += 1) {
                      row[headers[i]] = normalize(cells[i].textContent);
                    }
                    row['_match_report_href'] = tr.querySelector('td[data-stat="match_report"] a')?.getAttribute('href') || '';
                    out.push(row);
                  }
                  return out;
                }
                """
            )

            browser.close()

        scraped_at_utc = datetime.now(timezone.utc).isoformat()
        normalized_rows = []
        for row in rows:
            normalized_rows.append(
                {
                    "date": row.get("Date", ""),
                    "time": row.get("Time", ""),
                    "competition": row.get("Comp", ""),
                    "round": row.get("Round", ""),
                    "day": row.get("Day", ""),
                    "venue": row.get("Venue", ""),
                    "result": row.get("Result", ""),
                    "gf": row.get("GF", ""),
                    "ga": row.get("GA", ""),
                    "opponent": normalize_team_display(row.get("Opponent", "")),
                    "possession": row.get("Poss", ""),
                    "attendance": row.get("Attendance", ""),
                    "captain": row.get("Captain", ""),
                    "formation": row.get("Formation", ""),
                    "opp_formation": row.get("Opp Formation", ""),
                    "referee": row.get("Referee", ""),
                    "match_report": normalize_url(row.get("_match_report_href", "")),
                    "notes": row.get("Notes", ""),
                    "team": source.team,
                    "team_url": source.team_url,
                    "source_competition_page": source_competition_page,
                    "scraped_at_utc": scraped_at_utc,
                }
            )
        return normalized_rows


def run_extraction(output_dir: Path, season: str, headless: bool = True) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = FbrefPlaywrightExtractor(headless=headless)
    ko_rows, team_sources = extractor.fetch_ko_schedule()

    first_legs = filter_first_legs(ko_rows)
    first_legs_df = ensure_columns(pd.DataFrame(first_legs), FIRST_LEG_COLUMNS)

    team_sources_df = ensure_columns(
        pd.DataFrame([{"team": item.team, "team_url": item.team_url} for item in team_sources]),
        TEAM_SOURCE_COLUMNS,
    )

    all_rows: List[Dict[str, str]] = []
    failed_teams: List[str] = []

    for source in team_sources:
        try:
            team_rows = extractor.fetch_team_matchlogs(
                source=source,
                source_competition_page=UCL_SCHEDULE_URL,
                season=season,
            )
            all_rows.extend(team_rows)
        except Exception as exc:  # pragma: no cover - runtime path
            print(f"[WARN] Failed extracting {source.team}: {exc}")
            failed_teams.append(source.team)

    team_df = ensure_columns(pd.DataFrame(all_rows), TEAM_FIXTURE_COLUMNS)

    teams_file = output_dir / "ucl_ko_teams_scores_fixtures_2025_2026.csv"
    first_legs_file = output_dir / "ucl_ko_playoff_first_legs_2026_02_17_18.csv"
    team_sources_file = output_dir / "ucl_ko_team_sources_2025_2026.csv"

    team_df.to_csv(teams_file, index=False)
    first_legs_df.to_csv(first_legs_file, index=False)
    team_sources_df.to_csv(team_sources_file, index=False)

    return {
        "teams_requested": len(team_sources),
        "teams_succeeded": len(team_sources) - len(failed_teams),
        "teams_failed": len(failed_teams),
        "team_rows": len(team_df),
        "first_leg_rows": len(first_legs_df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract UCL KO playoff team scores & fixtures from FBref.")
    parser.add_argument("--output-dir", default="data", help="Output directory for CSV artifacts")
    parser.add_argument("--season", default="2025-2026", help="Season segment used in FBref matchlog URLs")
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser in headed mode (default is headless)",
    )
    args = parser.parse_args()

    stats = run_extraction(
        output_dir=Path(args.output_dir),
        season=args.season,
        headless=not args.headed,
    )
    print(
        "Extraction report | "
        f"teams_requested={stats['teams_requested']} "
        f"teams_succeeded={stats['teams_succeeded']} "
        f"teams_failed={stats['teams_failed']} "
        f"team_rows={stats['team_rows']} "
        f"first_leg_rows={stats['first_leg_rows']}"
    )


if __name__ == "__main__":
    main()
