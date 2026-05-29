"""Extract Libertadores historical match data from FBref.

Builds a canonical match-level CSV with the same core columns used by
`preprocessing_data/club_matches_historical_1993_2026.csv`.

Workflow:
- Read all clubs competing in the current Copa Libertadores from FBref.
- For each club, fetch `Scores & Fixtures` for the previous season and current season.
- Optionally fetch competition-wide `Scores & Fixtures` pages for a range of
  historical Libertadores seasons.
- Convert both sources into canonical match-level rows.
- Save a deduplicated historical CSV for downstream model training.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
from urllib.parse import urljoin

import pandas as pd

FBREF_BASE = "https://fbref.com"
LIBERTADORES_STATS_URL = f"{FBREF_BASE}/en/comps/14/Copa-Libertadores-Stats"

TEAM_MATCHLOG_COLUMNS = [
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
]

CANONICAL_COLUMNS = [
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

TEAM_SOURCE_COLUMNS = [
    "team",
    "team_country",
    "team_country_code",
    "team_url",
]

COUNTRY_CODES = {
    "ar": "Argentina",
    "bo": "Bolivia",
    "br": "Brazil",
    "cl": "Chile",
    "co": "Colombia",
    "ec": "Ecuador",
    "pe": "Peru",
    "py": "Paraguay",
    "uy": "Uruguay",
    "ve": "Venezuela",
}


@dataclass(frozen=True)
class TeamSource:
    team: str
    team_country: str
    team_country_code: str
    team_url: str


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def normalize_team_display(raw: str) -> str:
    return normalize_whitespace(raw).replace("\u2013", "-")


def normalize_competition_schedule_team(raw: str) -> str:
    text = normalize_team_display(raw)
    country_tokens = {"ar", "bo", "br", "cl", "co", "ec", "pe", "py", "uy", "ve"}
    parts = text.split()
    if parts and parts[0].lower() in country_tokens:
        parts = parts[1:]
    if parts and parts[-1].lower() in country_tokens:
        parts = parts[:-1]
    text = " ".join(parts)
    return normalize_team_display(text)


def normalize_url(path_or_url: str) -> str:
    if not path_or_url:
        return ""
    return urljoin(FBREF_BASE, path_or_url).split("?", 1)[0]


def parse_team_slug(team_url: str) -> str:
    match = re.search(r"/en/squads/[^/]+/(?:\d{4}/)?([^/?#]+)-Stats", team_url)
    if not match:
        raise ValueError(f"Unexpected FBref squad URL format: {team_url}")
    return match.group(1)


def build_season_matchlog_url(team_url: str, season_year: str) -> str:
    team_url = normalize_url(team_url)
    slug = parse_team_slug(team_url)
    base = team_url.split("/en/squads/", 1)[1].split("/", 1)[0]
    return (
        f"{FBREF_BASE}/en/squads/{base}/{season_year}/matchlogs/all_comps/schedule/"
        f"{slug}-Scores-and-Fixtures-All-Competitions"
    )


def build_competition_schedule_url(season_year: str) -> str:
    return (
        f"{FBREF_BASE}/en/comps/14/{season_year}/schedule/"
        f"{season_year}-Copa-Libertadores-Scores-and-Fixtures"
    )


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df.loc[:, list(columns)]


def dedupe_team_sources(candidates: Iterable[TeamSource]) -> List[TeamSource]:
    seen: Dict[str, TeamSource] = {}
    for item in candidates:
        team = normalize_team_display(item.team)
        url = normalize_url(item.team_url)
        if not team or not url:
            continue
        if team not in seen:
            seen[team] = TeamSource(
                team=team,
                team_country=item.team_country,
                team_country_code=item.team_country_code,
                team_url=url,
            )
    return sorted(seen.values(), key=lambda row: row.team.lower())


def infer_row_country(competition: str, team_country: str) -> str:
    comp = normalize_whitespace(competition).lower()
    if "libertadores" in comp or "sudamericana" in comp or "recopa" in comp:
        return "South America"
    return team_country


def compute_home_result(home_score: float, away_score: float) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def team_rows_to_canonical(rows: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    for row in rows.itertuples(index=False):
        team = normalize_team_display(getattr(row, "team", ""))
        opponent = normalize_team_display(getattr(row, "opponent", ""))
        venue = normalize_whitespace(getattr(row, "venue", "")).lower()
        date = getattr(row, "date", "")
        competition = normalize_whitespace(getattr(row, "competition", ""))
        round_name = normalize_whitespace(getattr(row, "round", ""))
        team_country = normalize_whitespace(getattr(row, "team_country", ""))
        season = str(getattr(row, "fbref_season", "") or "")
        source_file = normalize_url(getattr(row, "source_matchlog_page", ""))

        if not team or not opponent or not venue:
            continue

        gf = pd.to_numeric(getattr(row, "gf", None), errors="coerce")
        ga = pd.to_numeric(getattr(row, "ga", None), errors="coerce")
        parsed_date = pd.to_datetime(date, errors="coerce")
        if pd.isna(parsed_date) or pd.isna(gf) or pd.isna(ga):
            continue

        if venue == "home":
            home_team, away_team = team, opponent
            home_score, away_score = float(gf), float(ga)
            neutral = 0
        elif venue == "away":
            home_team, away_team = opponent, team
            home_score, away_score = float(ga), float(gf)
            neutral = 0
        else:
            ordered = sorted([team, opponent])
            home_team, away_team = ordered[0], ordered[1]
            if team == home_team:
                home_score, away_score = float(gf), float(ga)
            else:
                home_score, away_score = float(ga), float(gf)
            neutral = 1

        records.append(
            {
                "date": parsed_date.date().isoformat(),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "result": compute_home_result(home_score, away_score),
                "competition": competition,
                "country": infer_row_country(competition, team_country),
                "season": season,
                "round": round_name,
                "neutral": neutral,
                "source": "fbref.com",
                "source_file": source_file,
                "div_code": "",
            }
        )

    canonical = pd.DataFrame(records)
    if canonical.empty:
        return ensure_columns(canonical, CANONICAL_COLUMNS)

    dedupe_cols = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "competition",
        "round",
        "neutral",
    ]
    canonical = canonical.sort_values(["date", "home_team", "away_team"]).drop_duplicates(
        subset=dedupe_cols,
        keep="first",
    )
    return ensure_columns(canonical.reset_index(drop=True), CANONICAL_COLUMNS)


def schedule_rows_to_canonical(rows: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    for row in rows.itertuples(index=False):
        home_team = normalize_competition_schedule_team(getattr(row, "home_team", ""))
        away_team = normalize_competition_schedule_team(getattr(row, "away_team", ""))
        date = getattr(row, "date", "")
        round_name = normalize_whitespace(getattr(row, "round", ""))
        season = str(getattr(row, "season", "") or "")
        source_file = normalize_url(getattr(row, "source_file", ""))
        notes = normalize_whitespace(getattr(row, "notes", ""))

        if not home_team or not away_team:
            continue

        parsed_date = pd.to_datetime(date, errors="coerce")
        if pd.isna(parsed_date):
            continue

        score_text = normalize_whitespace(getattr(row, "score", ""))
        score_match = re.search(r"(\d+)\s*[–-]\s*(\d+)", score_text)
        if not score_match:
            continue
        home_score = float(score_match.group(1))
        away_score = float(score_match.group(2))

        neutral = 1 if "neutral" in notes.lower() else 0

        records.append(
            {
                "date": parsed_date.date().isoformat(),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "result": compute_home_result(home_score, away_score),
                "competition": "Libertadores",
                "country": "South America",
                "season": season,
                "round": round_name,
                "neutral": neutral,
                "source": "fbref.com",
                "source_file": source_file,
                "div_code": "",
            }
        )

    canonical = pd.DataFrame(records)
    if canonical.empty:
        return ensure_columns(canonical, CANONICAL_COLUMNS)

    dedupe_cols = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "competition",
        "round",
        "neutral",
    ]
    canonical = canonical.sort_values(["date", "home_team", "away_team"]).drop_duplicates(
        subset=dedupe_cols,
        keep="first",
    )
    return ensure_columns(canonical.reset_index(drop=True), CANONICAL_COLUMNS)


class FbrefPlaywrightExtractor:
    def __init__(
        self,
        headless: bool = True,
        max_attempts: int = 3,
        user_data_dir: str | None = None,
        manual_challenge_seconds: int = 0,
        cdp_url: str | None = None,
    ):
        self.headless = headless
        self.max_attempts = max_attempts
        self.user_data_dir = user_data_dir
        self.manual_challenge_seconds = max(0, int(manual_challenge_seconds))
        self.cdp_url = cdp_url
        self._playwright = None
        self._browser = None
        self._context = None
        self._challenge_primed = False
        self._owns_browser = True

    def __enter__(self):
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        if self.cdp_url:
            self._browser = self._playwright.chromium.connect_over_cdp(self.cdp_url)
            self._owns_browser = False
            existing_contexts = list(self._browser.contexts)
            self._context = existing_contexts[0] if existing_contexts else self._browser.new_context()
        else:
            launch_kwargs = {"headless": self.headless}
            if not self.headless:
                launch_kwargs["channel"] = "chrome"

            if self.user_data_dir or not self.headless:
                persistent_dir = str(
                    Path(self.user_data_dir).expanduser()
                    if self.user_data_dir
                    else (Path.home() / ".cache" / "fbref_playwright_profile")
                )
                self._context = self._playwright.chromium.launch_persistent_context(
                    user_data_dir=persistent_dir,
                    **launch_kwargs,
                )
            else:
                self._browser = self._playwright.chromium.launch(**launch_kwargs)
                self._context = self._browser.new_context()
            self._owns_browser = True

        self._prime_challenge_if_needed()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._context is not None:
            if self._owns_browser or self.cdp_url is None:
                self._context.close()
            self._context = None
        if self._browser is not None:
            if self._owns_browser:
                self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def _prime_challenge_if_needed(self) -> None:
        if self.headless or self.manual_challenge_seconds <= 0 or self._challenge_primed:
            return
        page = self._context.new_page()
        try:
            try:
                page.goto(LIBERTADORES_STATS_URL, wait_until="domcontentloaded", timeout=60000)
            except Exception:
                pass
            print(
                "[INFO] Headed browser opened for FBref. "
                f"Use the next {self.manual_challenge_seconds}s to clear any Cloudflare challenge.",
                flush=True,
            )
            time.sleep(self.manual_challenge_seconds)
        finally:
            page.close()
        self._challenge_primed = True

    def _new_page(self):
        if self._context is None:
            raise RuntimeError("Browser context is not initialized. Use extractor as a context manager.")
        return self._context.new_page()

    def _goto_with_retry(self, page, url: str) -> None:
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                self._wait_for_challenge(page)
                return
            except Exception as exc:
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

    def fetch_team_sources(self) -> List[TeamSource]:
        page = self._new_page()
        try:
            self._goto_with_retry(page, LIBERTADORES_STATS_URL)
            self._wait_for_stats_table(page)

            payload = page.evaluate(
                """
                () => {
                  const tables = Array.from(document.querySelectorAll('table[id^="results"][id$="_overall"]'));
                  const table = tables.find((candidate) => {
                    const heading = document.querySelector('h1')?.textContent || '';
                    return heading.includes('Copa Libertadores');
                  }) || tables[0];
                  if (!table) return [];

                  const normalize = (value) => (value || '').replace(/\\s+/g, ' ').trim();
                  const rows = [];
                  for (const tr of table.querySelectorAll('tbody tr')) {
                    if (tr.classList.contains('thead')) continue;
                    const squadCell = tr.querySelector('th[data-stat="team"], td[data-stat="team"], th[data-stat="squad"], td[data-stat="squad"]');
                    const anchor = squadCell?.querySelector('a');
                    if (!anchor) continue;
                    const rawText = normalize(squadCell.textContent);
                    const match = rawText.match(/^([a-z]{2})\\s+(.*)$/i);
                    rows.push({
                      team_country_code: match ? match[1].toLowerCase() : '',
                      team: match ? match[2] : normalize(anchor.textContent),
                      team_url: anchor.getAttribute('href') || '',
                    });
                  }
                  return rows;
                }
                """
            )
        finally:
            page.close()

        sources = []
        for row in payload:
            code = normalize_whitespace(row.get("team_country_code", "")).lower()
            sources.append(
                TeamSource(
                    team=normalize_team_display(row.get("team", "")),
                    team_country=COUNTRY_CODES.get(code, code.upper()),
                    team_country_code=code,
                    team_url=normalize_url(row.get("team_url", "")),
                )
            )
        return dedupe_team_sources(sources)

    def fetch_team_matchlogs(self, source: TeamSource, season_year: str) -> List[Dict[str, str]]:
        matchlog_url = build_season_matchlog_url(source.team_url, season_year=season_year)
        page = self._new_page()
        try:
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
                    row["_match_report_href"] = tr.querySelector('td[data-stat="match_report"] a')?.getAttribute('href') || '';
                    out.push(row);
                  }
                  return out;
                }
                """
            )
        finally:
            page.close()

        scraped_at_utc = datetime.now(timezone.utc).isoformat()
        normalized_rows: List[Dict[str, str]] = []
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
                    "team_country": source.team_country,
                    "team_url": source.team_url,
                    "source_competition_page": LIBERTADORES_STATS_URL,
                    "source_matchlog_page": matchlog_url,
                    "fbref_season": str(season_year),
                    "scraped_at_utc": scraped_at_utc,
                }
            )
        return normalized_rows

    def fetch_competition_schedule_rows(self, season_year: str) -> List[Dict[str, str]]:
        schedule_url = build_competition_schedule_url(str(season_year))
        page = self._new_page()
        try:
            self._goto_with_retry(page, schedule_url)
            self._wait_for_stats_table(page)

            rows = page.evaluate(
                r"""
                () => {
                  const table = document.querySelector('table.stats_table');
                  if (!table) return [];
                  const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim();
                  const out = [];
                  for (const tr of table.querySelectorAll('tbody tr')) {
                    if (tr.classList.contains('thead')) continue;
                    const scoreCell = tr.querySelector('td[data-stat="score"], th[data-stat="score"]');
                    const homeCell = tr.querySelector('td[data-stat="home_team"], th[data-stat="home_team"]');
                    const awayCell = tr.querySelector('td[data-stat="away_team"], th[data-stat="away_team"]');
                    const dateCell = tr.querySelector('td[data-stat="date"], th[data-stat="date"]');
                    const roundCell = tr.querySelector('td[data-stat="round"], th[data-stat="round"]');
                    const notesCell = tr.querySelector('td[data-stat="notes"], th[data-stat="notes"]');
                    if (!scoreCell || !homeCell || !awayCell || !dateCell) continue;
                    out.push({
                      date: normalize(dateCell.textContent),
                      round: normalize(roundCell?.textContent || ''),
                      home_team: normalize(homeCell.textContent),
                      away_team: normalize(awayCell.textContent),
                      score: normalize(scoreCell.textContent),
                      notes: normalize(notesCell?.textContent || ''),
                      source_file: window.location.href,
                    });
                  }
                  return out;
                }
                """
            )
        finally:
            page.close()

        for row in rows:
            row["season"] = str(season_year)
        return rows


def run_extraction(
    output_dir: Path,
    previous_season: str,
    current_season: str,
    historical_start_season: str | None = None,
    historical_end_season: str | None = None,
    headless: bool = True,
    user_data_dir: str | None = None,
    manual_challenge_seconds: int = 0,
    cdp_url: str | None = None,
) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = FbrefPlaywrightExtractor(
        headless=headless,
        user_data_dir=user_data_dir,
        manual_challenge_seconds=manual_challenge_seconds,
        cdp_url=cdp_url,
    )
    all_team_rows: List[Dict[str, str]] = []
    historical_schedule_rows: List[Dict[str, str]] = []
    failed: List[str] = []
    seasons = [str(previous_season), str(current_season)]
    historical_failures = 0

    with extractor:
        team_sources = extractor.fetch_team_sources()

        for source in team_sources:
            for season_year in seasons:
                try:
                    team_rows = extractor.fetch_team_matchlogs(source=source, season_year=season_year)
                    all_team_rows.extend(team_rows)
                except Exception as exc:
                    print(f"[WARN] Failed extracting {source.team} season={season_year}: {exc}", flush=True)
                    failed.append(f"{source.team}:{season_year}")

        if historical_start_season and historical_end_season:
            for season_year in range(int(historical_start_season), int(historical_end_season) + 1):
                try:
                    print(f"[INFO] Fetching Libertadores schedule season={season_year}", flush=True)
                    historical_schedule_rows.extend(
                        extractor.fetch_competition_schedule_rows(season_year=str(season_year))
                    )
                except Exception as exc:
                    print(f"[WARN] Failed extracting competition schedule season={season_year}: {exc}", flush=True)
                    historical_failures += 1

    team_rows_df = ensure_columns(pd.DataFrame(all_team_rows), TEAM_MATCHLOG_COLUMNS)
    canonical_df = team_rows_to_canonical(team_rows_df)
    historical_schedule_df = pd.DataFrame(historical_schedule_rows)
    historical_canonical_df = schedule_rows_to_canonical(historical_schedule_df)
    canonical_df = (
        pd.concat([historical_canonical_df, canonical_df], ignore_index=True)
        .sort_values(["date", "home_team", "away_team"])
        .drop_duplicates(
            subset=[
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "competition",
                "round",
                "neutral",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )
    team_sources_df = ensure_columns(
        pd.DataFrame(
            [
                {
                    "team": item.team,
                    "team_country": item.team_country,
                    "team_country_code": item.team_country_code,
                    "team_url": item.team_url,
                }
                for item in team_sources
            ]
        ),
        TEAM_SOURCE_COLUMNS,
    )

    raw_file = output_dir / f"libertadores_teams_scores_fixtures_{previous_season}_{current_season}.csv"
    canonical_file = output_dir / f"club_matches_historical_libertadores_{previous_season}_{current_season}.csv"
    teams_file = output_dir / f"libertadores_team_sources_{current_season}.csv"

    team_rows_df.to_csv(raw_file, index=False)
    canonical_df.to_csv(canonical_file, index=False)
    team_sources_df.to_csv(teams_file, index=False)

    return {
        "teams_requested": len(team_sources),
        "team_season_failures": len(failed),
        "historical_schedule_failures": historical_failures,
        "team_rows": int(len(team_rows_df)),
        "historical_schedule_rows": int(len(historical_schedule_df)),
        "canonical_rows": int(len(canonical_df)),
    }


def main() -> None:
    now_year = datetime.now(timezone.utc).year

    parser = argparse.ArgumentParser(
        description="Extract Libertadores participant Scores & Fixtures history from FBref."
    )
    parser.add_argument(
        "--output-dir",
        default="preprocessing_data",
        help="Output directory for CSV artifacts",
    )
    parser.add_argument(
        "--previous-season",
        default=str(now_year - 1),
        help="Previous FBref season year, e.g. 2025",
    )
    parser.add_argument(
        "--current-season",
        default=str(now_year),
        help="Current FBref season year, e.g. 2026",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser in headed mode (default is headless)",
    )
    parser.add_argument(
        "--user-data-dir",
        default="",
        help="Persistent browser profile dir to reuse Cloudflare cookies when headed",
    )
    parser.add_argument(
        "--manual-challenge-seconds",
        type=int,
        default=0,
        help="When headed, keep the browser open this many seconds so you can clear Cloudflare manually",
    )
    parser.add_argument(
        "--cdp-url",
        default="",
        help="Connect to an already-running Chrome with remote debugging, e.g. http://127.0.0.1:9222",
    )
    parser.add_argument(
        "--historical-start-season",
        default="2014",
        help="First Libertadores season year to fetch from competition schedule pages",
    )
    parser.add_argument(
        "--historical-end-season",
        default=str(now_year - 2),
        help="Last Libertadores season year to fetch from competition schedule pages",
    )
    args = parser.parse_args()

    stats = run_extraction(
        output_dir=Path(args.output_dir),
        previous_season=args.previous_season,
        current_season=args.current_season,
        historical_start_season=args.historical_start_season,
        historical_end_season=args.historical_end_season,
        headless=not args.headed,
        user_data_dir=args.user_data_dir or None,
        manual_challenge_seconds=args.manual_challenge_seconds,
        cdp_url=args.cdp_url or None,
    )
    print(
        "Extraction report | "
        f"teams_requested={stats['teams_requested']} "
        f"team_season_failures={stats['team_season_failures']} "
        f"historical_schedule_failures={stats['historical_schedule_failures']} "
        f"team_rows={stats['team_rows']} "
        f"historical_schedule_rows={stats['historical_schedule_rows']} "
        f"canonical_rows={stats['canonical_rows']}"
    )


if __name__ == "__main__":
    main()
