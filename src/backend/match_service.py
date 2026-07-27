from functools import lru_cache
from typing import List, Dict, Callable, Optional
import logging
import os
import re
from time import perf_counter
import unicodedata

import pandas as pd

try:
    from .supabase_client import get_supabase_client
    from .paths import DATA_DIR, LIBERTADORES_MATCHES_HISTORY_PATH
except ImportError:  # pragma: no cover - fallback for direct module execution
    from src.backend.supabase_client import get_supabase_client
    from src.backend.paths import DATA_DIR, LIBERTADORES_MATCHES_HISTORY_PATH

logger = logging.getLogger("futbolconu.match_service")

_MATCH_SELECT_COLUMNS = (
    "match_date,home_team,away_team,home_score,away_score,"
    "tournament,home_team_confederation,away_team_confederation"
)
_LIBERTADORES_MATCH_SELECT_COLUMNS = "match_date,home_team,away_team,home_score,away_score,tournament"
FIFA_CODES_PATH = os.getenv("FIFA_CODES_PATH", str(DATA_DIR / "fifa_country_codes.csv"))
TEAM_CODE_ALIASES_PATH = os.getenv("TEAM_CODE_ALIASES_PATH", str(DATA_DIR / "team_code_aliases.csv"))


def _service_role_client():
    return get_supabase_client(use_service_role=True)


def _matches_table():
    return _service_role_client().table("matches")


def _libertadores_matches_table():
    return _service_role_client().table("libertadores_matches")


@lru_cache(maxsize=1)
def _libertadores_matches_table_available() -> bool:
    try:
        _libertadores_matches_table().select("match_id").limit(1).execute()
        return True
    except Exception as exc:
        logger.warning("libertadores_recent_form_table_missing error=%s fallback=csv", exc)
        return False


def _execute_query(
    query,
    *,
    request_id: str,
    operation: str,
    context: Optional[dict] = None,
    log_exception: bool = True,
) -> list[dict]:
    start_time = perf_counter()
    context = context or {}
    logger.info(
        "match_query_started request_id=%s operation=%s context=%s",
        request_id,
        operation,
        context,
    )
    try:
        rows = query.execute().data or []
        logger.info(
            "match_query_succeeded request_id=%s operation=%s rows=%s elapsed_ms=%.2f",
            request_id,
            operation,
            len(rows),
            (perf_counter() - start_time) * 1000.0,
        )
        return rows
    except Exception as exc:
        log_method = logger.exception if log_exception else logger.warning
        log_method(
            "match_query_failed request_id=%s operation=%s elapsed_ms=%.2f context=%s error=%s",
            request_id,
            operation,
            (perf_counter() - start_time) * 1000.0,
            context,
            exc,
        )
        raise


def _fetch_paginated_rows(
    build_query: Callable[[int, int], object],
    *,
    request_id: str,
    operation: str,
    chunk_size: int = 1000,
    log_exception: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    offset = 0

    while True:
        end = offset + chunk_size - 1
        batch = _execute_query(
            build_query(offset, end),
            request_id=request_id,
            operation=operation,
            context={"offset": offset, "end": end},
            log_exception=log_exception,
        )
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < chunk_size:
            break
        offset += chunk_size

    return rows


def _rows_to_matches_df(rows: list[dict]) -> pd.DataFrame:
    columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "home_team_confederation",
        "away_team_confederation",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows).rename(columns={"match_date": "date"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for score_col in ("home_score", "away_score"):
        if score_col in df.columns:
            df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    return df


def _normalized_text(value: str) -> str:
    text = str(value or "").replace("\xa0", " ").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def _normalize_team_code(value: str) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    return raw if re.fullmatch(r"[A-Z]{3}", raw) else ""


@lru_cache(maxsize=1)
def _load_national_team_alias_maps() -> tuple[dict[str, str], dict[str, list[str]]]:
    name_to_code: dict[str, str] = {}
    code_to_names: dict[str, set[str]] = {}

    def add_mapping(name: object, code: object):
        normalized_code = _normalize_team_code(code)
        display_name = str(name or "").strip()
        if not normalized_code or not display_name:
            return
        name_to_code[_normalized_text(display_name)] = normalized_code
        code_to_names.setdefault(normalized_code, set()).add(display_name)

    if os.path.exists(FIFA_CODES_PATH):
        try:
            fifa_df = pd.read_csv(FIFA_CODES_PATH)
        except Exception as exc:
            logger.warning("fifa_codes_load_failed path=%s error=%s", FIFA_CODES_PATH, exc)
            fifa_df = pd.DataFrame()

        if not fifa_df.empty:
            code_col = next(
                (col for col in ["team_code", "code", "fifa_code", "Code"] if col in fifa_df.columns),
                None,
            )
            name_col = next(
                (col for col in ["country_name", "country", "team", "Country"] if col in fifa_df.columns),
                None,
            )
            if code_col and name_col:
                for _, row in fifa_df[[code_col, name_col]].iterrows():
                    add_mapping(row.get(name_col), row.get(code_col))

    if os.path.exists(TEAM_CODE_ALIASES_PATH):
        try:
            alias_df = pd.read_csv(TEAM_CODE_ALIASES_PATH)
        except Exception as exc:
            logger.warning("team_code_aliases_load_failed path=%s error=%s", TEAM_CODE_ALIASES_PATH, exc)
            alias_df = pd.DataFrame()

        if not alias_df.empty:
            code_col = next(
                (col for col in ["team_code", "code", "fifa_code"] if col in alias_df.columns),
                None,
            )
            alias_col = next(
                (col for col in ["alias_name", "alias", "team_name", "name"] if col in alias_df.columns),
                None,
            )
            if code_col and alias_col:
                for _, row in alias_df[[code_col, alias_col]].iterrows():
                    add_mapping(row.get(alias_col), row.get(code_col))

    return name_to_code, {code: sorted(names) for code, names in code_to_names.items()}


def _national_team_name_variants(team: str) -> list[str]:
    team_name = str(team or "").strip()
    if not team_name:
        return []

    name_to_code, code_to_names = _load_national_team_alias_maps()
    code = name_to_code.get(_normalized_text(team_name), "")
    variants = [team_name]
    if code:
        variants.extend(code_to_names.get(code, []))

    unique_variants = []
    seen = set()
    for variant in variants:
        key = _normalized_text(variant)
        if key and key not in seen:
            unique_variants.append(variant)
            seen.add(key)
    return unique_variants


def _dedupe_recent_team_matches(df: pd.DataFrame, team_variants: list[str]) -> pd.DataFrame:
    if df.empty:
        return df

    variant_keys = {_normalized_text(variant) for variant in team_variants}

    def team_key(value: object) -> str:
        key = _normalized_text(value)
        return "__team__" if key in variant_keys else key

    dedupe = df.copy()
    if "home_team" in dedupe.columns:
        dedupe["_home_team_key"] = dedupe["home_team"].map(team_key)
    if "away_team" in dedupe.columns:
        dedupe["_away_team_key"] = dedupe["away_team"].map(team_key)
    if "date" in dedupe.columns:
        dedupe["_date_key"] = pd.to_datetime(dedupe["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if {"home_team", "away_team", "home_score", "away_score"}.issubset(dedupe.columns):
        def perspective_key(row: pd.Series) -> str:
            home_key = team_key(row.get("home_team"))
            away_key = team_key(row.get("away_team"))
            if home_key == "__team__":
                return f"{row.get('_date_key')}|{away_key}|{row.get('home_score')}|{row.get('away_score')}"
            if away_key == "__team__":
                return f"{row.get('_date_key')}|{home_key}|{row.get('away_score')}|{row.get('home_score')}"
            return ""

        dedupe["_team_perspective_key"] = dedupe.apply(perspective_key, axis=1)

    if "_team_perspective_key" in dedupe.columns and dedupe["_team_perspective_key"].astype(bool).all():
        subset = ["_team_perspective_key"]
    else:
        subset = [
            column
            for column in ["_date_key", "_home_team_key", "_away_team_key", "home_score", "away_score"]
            if column in dedupe.columns
        ]
    if subset:
        dedupe = dedupe.drop_duplicates(subset=subset)
    return dedupe.drop(columns=[column for column in dedupe.columns if column.startswith("_")])


@lru_cache(maxsize=1)
def _load_libertadores_matches_df() -> pd.DataFrame:
    path = LIBERTADORES_MATCHES_HISTORY_PATH
    if not path.exists():
        logger.warning("libertadores_recent_form_csv_missing path=%s", path)
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_score", "away_score", "tournament"])

    df = pd.read_csv(path)
    rename_map = {}
    if "competition" in df.columns and "tournament" not in df.columns:
        rename_map["competition"] = "tournament"
    if rename_map:
        df = df.rename(columns=rename_map)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for score_col in ("home_score", "away_score"):
        if score_col in df.columns:
            df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    return df


def _sort_matches(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        return df
    return df.sort_values("date", ascending=False)


def _dedupe_matches(df: pd.DataFrame) -> pd.DataFrame:
    subset = ["date", "home_team", "away_team", "home_score", "away_score"]
    available_subset = [column for column in subset if column in df.columns]
    if not available_subset:
        return df
    return df.drop_duplicates(subset=available_subset)


def _dedupe_matches_with_aliases(
    df: pd.DataFrame,
    home_variants: list[str],
    away_variants: list[str],
) -> pd.DataFrame:
    if df.empty:
        return df

    home_keys = {_normalized_text(variant) for variant in home_variants}
    away_keys = {_normalized_text(variant) for variant in away_variants}
    dedupe = df.copy()

    def side_key(value: object) -> str:
        key = _normalized_text(value)
        if key in home_keys:
            return "__home_team__"
        if key in away_keys:
            return "__away_team__"
        return key

    if "date" in dedupe.columns:
        dedupe["_date_key"] = pd.to_datetime(dedupe["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "home_team" in dedupe.columns:
        dedupe["_home_team_key"] = dedupe["home_team"].map(side_key)
    if "away_team" in dedupe.columns:
        dedupe["_away_team_key"] = dedupe["away_team"].map(side_key)

    subset = [
        column
        for column in ["_date_key", "_home_team_key", "_away_team_key", "home_score", "away_score", "tournament"]
        if column in dedupe.columns
    ]
    if subset:
        dedupe = dedupe.drop_duplicates(subset=subset)
    return dedupe.drop(columns=[column for column in dedupe.columns if column.startswith("_")])


def _only_completed_matches(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    score_cols = [column for column in ("home_score", "away_score") if column in df.columns]
    if not score_cols:
        return df
    return df.dropna(subset=score_cols)


def _format_matches(df: pd.DataFrame) -> List[Dict[str, int]]:
    formatted = _only_completed_matches(df).copy()
    if "date" in formatted.columns:
        formatted["date"] = pd.to_datetime(formatted["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        formatted["date"] = formatted["date"].where(formatted["date"].notna(), None)
    for score_col in ("home_score", "away_score"):
        if score_col in formatted.columns:
            formatted[score_col] = formatted[score_col].astype(int)
    return formatted[["date", "home_team", "away_team", "home_score", "away_score"]].to_dict(orient="records")


def _fetch_recent_team_matches_from_libertadores_source(team: str, last_matches: int, request_id: str = "-") -> pd.DataFrame:
    if _libertadores_matches_table_available():
        home_rows = _execute_query(
            _libertadores_matches_table()
            .select(_LIBERTADORES_MATCH_SELECT_COLUMNS)
            .eq("home_team", team)
            .not_.is_("home_score", "null")
            .not_.is_("away_score", "null")
            .order("match_date", desc=True)
            .limit(last_matches),
            request_id=request_id,
            operation="recent_team_matches_libertadores_home",
            context={"team": team, "limit": last_matches},
        )
        away_rows = _execute_query(
            _libertadores_matches_table()
            .select(_LIBERTADORES_MATCH_SELECT_COLUMNS)
            .eq("away_team", team)
            .not_.is_("home_score", "null")
            .not_.is_("away_score", "null")
            .order("match_date", desc=True)
            .limit(last_matches),
            request_id=request_id,
            operation="recent_team_matches_libertadores_away",
            context={"team": team, "limit": last_matches},
        )
        combined = _rows_to_matches_df(home_rows + away_rows)
    else:
        source_df = _load_libertadores_matches_df()
        combined = source_df[
            (source_df["home_team"] == team) | (source_df["away_team"] == team)
        ].copy()

    combined = _only_completed_matches(combined)
    combined = _sort_matches(combined)
    combined = _dedupe_matches(combined)
    return combined.head(last_matches)


def _fetch_recent_team_matches(team: str, last_matches: int, mode: str = "national", request_id: str = "-") -> pd.DataFrame:
    if str(mode or "").strip().lower() == "libertadores":
        return _fetch_recent_team_matches_from_libertadores_source(team, last_matches, request_id=request_id)

    team_variants = _national_team_name_variants(team)
    home_rows = []
    away_rows = []
    for variant in team_variants:
        home_rows.extend(
            _execute_query(
                _matches_table()
                .select(_MATCH_SELECT_COLUMNS)
                .eq("home_team", variant)
                .not_.is_("home_score", "null")
                .not_.is_("away_score", "null")
                .order("match_date", desc=True)
                .limit(last_matches),
                request_id=request_id,
                operation="recent_team_matches_home",
                context={"team": team, "query_team": variant, "limit": last_matches},
            )
        )
        away_rows.extend(
            _execute_query(
                _matches_table()
                .select(_MATCH_SELECT_COLUMNS)
                .eq("away_team", variant)
                .not_.is_("home_score", "null")
                .not_.is_("away_score", "null")
                .order("match_date", desc=True)
                .limit(last_matches),
                request_id=request_id,
                operation="recent_team_matches_away",
                context={"team": team, "query_team": variant, "limit": last_matches},
            )
        )
    combined = _rows_to_matches_df(home_rows + away_rows)
    logger.info(
        "recent_team_matches_combined request_id=%s team=%s variants=%s home_rows=%s away_rows=%s combined_rows=%s",
        request_id,
        team,
        team_variants,
        len(home_rows),
        len(away_rows),
        len(combined),
    )
    unique_matches = _dedupe_recent_team_matches(
        _dedupe_matches(_sort_matches(_only_completed_matches(combined))),
        team_variants,
    )
    return unique_matches.head(last_matches)


def get_recent_matches(
    home_team: str,
    away_team: str,
    last_matches: int,
    mode: str = "national",
    request_id: str = "-",
) -> Dict[str, object]:
    start_time = perf_counter()
    logger.info(
        "recent_form_service_started request_id=%s home_team=%s away_team=%s last_matches=%s mode=%s source=%s",
        request_id,
        home_team,
        away_team,
        last_matches,
        mode,
        "libertadores_matches/csv" if str(mode or "").strip().lower() == "libertadores" else "supabase",
    )

    recent_home = _fetch_recent_team_matches(home_team, last_matches, mode=mode, request_id=request_id)
    recent_away = _fetch_recent_team_matches(away_team, last_matches, mode=mode, request_id=request_id)
    result = {
        "home_team": home_team,
        "away_team": away_team,
        "home_matches": _format_matches(_only_completed_matches(recent_home)),
        "away_matches": _format_matches(_only_completed_matches(recent_away)),
    }
    logger.info(
        "recent_form_service_completed request_id=%s home_returned=%s away_returned=%s elapsed_ms=%.2f",
        request_id,
        len(result["home_matches"]),
        len(result["away_matches"]),
        (perf_counter() - start_time) * 1000.0,
    )
    return result


_FORM_TOURNAMENTS = {
    "AFC Asian Cup",
    "AFC Asian Cup qualification",
    "African Cup of Nations",
    "African Cup of Nations qualification",
    "Arab Cup",
    "CONCACAF Championship",
    "CONCACAF Championship qualification",
    "CONCACAF Nations League",
    "CONCACAF Nations League qualification",
    "Copa América",
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "Friendly",
    "Gold Cup",
    "Gold Cup qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "UEFA Nations League",
    "Confederations Cup",
}


def _team_form(team: str, matches: pd.DataFrame, team_variants: Optional[list[str]] = None) -> Dict[str, int]:
    wins = draws = losses = goals = 0
    filtered = matches[matches["tournament"].isin(_FORM_TOURNAMENTS)]
    variant_keys = {_normalized_text(variant) for variant in (team_variants or [team])}
    for _, row in filtered.iterrows():
        if _normalized_text(row["home_team"]) in variant_keys:
            team_goals = row["home_score"]
            opp_goals = row["away_score"]
        else:
            team_goals = row["away_score"]
            opp_goals = row["home_score"]

        goals += int(team_goals)
        if team_goals > opp_goals:
            wins += 1
        elif team_goals == opp_goals:
            draws += 1
        else:
            losses += 1

    return {"team": team, "wins": wins, "draws": draws, "losses": losses, "goals": goals}


def _fetch_head_to_head_matches(
    home_team: str,
    away_team: str,
    tournaments: List[str],
    request_id: str = "-",
) -> pd.DataFrame:
    home_variants = _national_team_name_variants(home_team)
    away_variants = _national_team_name_variants(away_team)

    def build_directional_query(first_team: str, second_team: str):
        def build_query(offset: int, end: int):
            query = (
                _matches_table()
                .select(_MATCH_SELECT_COLUMNS)
                .eq("home_team", first_team)
                .eq("away_team", second_team)
                .not_.is_("home_score", "null")
                .not_.is_("away_score", "null")
                .order("match_date", desc=True)
                .range(offset, end)
            )
            if tournaments:
                query = query.in_("tournament", tournaments)
            return query

        return build_query

    rows = []
    for first_team in home_variants:
        for second_team in away_variants:
            try:
                rows.extend(
                    _fetch_paginated_rows(
                        build_directional_query(first_team, second_team),
                        request_id=request_id,
                        operation="head_to_head_forward",
                        log_exception=False,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "head_to_head_supabase_query_failed request_id=%s direction=forward first_team=%s second_team=%s error=%s",
                    request_id,
                    first_team,
                    second_team,
                    exc,
                )
    for first_team in away_variants:
        for second_team in home_variants:
            try:
                rows.extend(
                    _fetch_paginated_rows(
                        build_directional_query(first_team, second_team),
                        request_id=request_id,
                        operation="head_to_head_reverse",
                        log_exception=False,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "head_to_head_supabase_query_failed request_id=%s direction=reverse first_team=%s second_team=%s error=%s",
                    request_id,
                    first_team,
                    second_team,
                    exc,
                )
    logger.info(
        "head_to_head_rows_combined request_id=%s home_team=%s away_team=%s home_variants=%s away_variants=%s rows=%s tournaments=%s",
        request_id,
        home_team,
        away_team,
        home_variants,
        away_variants,
        len(rows),
        len(tournaments or []),
    )
    matches = _sort_matches(
        _dedupe_matches_with_aliases(
            _only_completed_matches(_rows_to_matches_df(rows)),
            home_variants,
            away_variants,
        )
    )
    return matches


def _fetch_team_vs_confed_matches(team: str, confed: str, request_id: str = "-") -> pd.DataFrame:
    confed = (confed or "").strip().upper()
    team_variants = _national_team_name_variants(team)
    logger.info(
        "TEAM_VS_CONFED_DEBUG aliases_resolved request_id=%s team=%s confed=%s variants=%s",
        request_id,
        team,
        confed,
        team_variants,
    )

    def build_side_query(team_column: str, confed_column: str, query_team: str):
        def build_query(offset: int, end: int):
            return (
                _matches_table()
                .select(_MATCH_SELECT_COLUMNS)
                .eq(team_column, query_team)
                .eq(confed_column, confed)
                .not_.is_("home_score", "null")
                .not_.is_("away_score", "null")
                .in_("tournament", list(_FORM_TOURNAMENTS))
                .order("match_date", desc=True)
                .range(offset, end)
            )

        return build_query

    rows = []
    for query_team in team_variants:
        try:
            home_rows = _fetch_paginated_rows(
                build_side_query("home_team", "away_team_confederation", query_team),
                request_id=request_id,
                operation="team_vs_confed_home",
                log_exception=False,
            )
            rows.extend(home_rows)
            logger.info(
                "TEAM_VS_CONFED_DEBUG supabase_side_result request_id=%s side=home query_team=%s opponent_confed=%s rows=%s",
                request_id,
                query_team,
                confed,
                len(home_rows),
            )
        except Exception as exc:
            logger.warning(
                "team_vs_confed_supabase_query_failed request_id=%s side=home query_team=%s confed=%s error=%s",
                request_id,
                query_team,
                confed,
                exc,
            )
            logger.warning(
                "TEAM_VS_CONFED_DEBUG supabase_side_failed request_id=%s side=home query_team=%s opponent_confed=%s error=%s",
                request_id,
                query_team,
                confed,
                exc,
            )
        try:
            away_rows = _fetch_paginated_rows(
                build_side_query("away_team", "home_team_confederation", query_team),
                request_id=request_id,
                operation="team_vs_confed_away",
                log_exception=False,
            )
            rows.extend(away_rows)
            logger.info(
                "TEAM_VS_CONFED_DEBUG supabase_side_result request_id=%s side=away query_team=%s opponent_confed=%s rows=%s",
                request_id,
                query_team,
                confed,
                len(away_rows),
            )
        except Exception as exc:
            logger.warning(
                "team_vs_confed_supabase_query_failed request_id=%s side=away query_team=%s confed=%s error=%s",
                request_id,
                query_team,
                confed,
                exc,
            )
            logger.warning(
                "TEAM_VS_CONFED_DEBUG supabase_side_failed request_id=%s side=away query_team=%s opponent_confed=%s error=%s",
                request_id,
                query_team,
                confed,
                exc,
            )
    logger.info(
        "team_vs_confed_rows_combined request_id=%s team=%s variants=%s confed=%s rows=%s",
        request_id,
        team,
        team_variants,
        confed,
        len(rows),
    )
    matches = _sort_matches(
        _dedupe_recent_team_matches(
            _dedupe_matches(_only_completed_matches(_rows_to_matches_df(rows))),
            team_variants,
        )
    )
    logger.info(
        "TEAM_VS_CONFED_DEBUG source_selected request_id=%s source=supabase rows=%s sample=%s",
        request_id,
        len(matches),
        _format_matches(matches.head(3)),
    )
    return matches


def get_head_to_head(
    home_team: str,
    away_team: str,
    tournaments: List[str],
    request_id: str = "-",
) -> Dict[str, object]:
    start_time = perf_counter()
    logger.info(
        "head_to_head_service_started request_id=%s home_team=%s away_team=%s tournaments=%s source=supabase",
        request_id,
        home_team,
        away_team,
        len(tournaments or []),
    )

    home_variants = _national_team_name_variants(home_team)
    away_variants = _national_team_name_variants(away_team)
    h2h = _only_completed_matches(
        _fetch_head_to_head_matches(
            home_team,
            away_team,
            tournaments or [],
            request_id=request_id,
        )
    )

    matches = _format_matches(h2h)
    result = {
        "matches": matches,
        "home_form": _team_form(home_team, h2h, home_variants),
        "away_form": _team_form(away_team, h2h, away_variants),
    }
    logger.info(
        "head_to_head_service_completed request_id=%s returned_matches=%s elapsed_ms=%.2f",
        request_id,
        len(matches),
        (perf_counter() - start_time) * 1000.0,
    )
    return result


def _team_record(team: str, matches: pd.DataFrame, team_variants: Optional[list[str]] = None) -> Dict[str, int]:
    wins = draws = losses = goals_for = goals_against = 0
    variant_keys = {_normalized_text(variant) for variant in (team_variants or [team])}
    for _, row in matches.iterrows():
        if _normalized_text(row["home_team"]) in variant_keys:
            team_goals = row["home_score"]
            opp_goals = row["away_score"]
        else:
            team_goals = row["away_score"]
            opp_goals = row["home_score"]

        goals_for += int(team_goals)
        goals_against += int(opp_goals)
        if team_goals > opp_goals:
            wins += 1
        elif team_goals == opp_goals:
            draws += 1
        else:
            losses += 1

    return {
        "matches_count": int(len(matches)),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
    }


def get_team_vs_confed(
    team: str,
    opponent_confederation: str,
    request_id: str = "-",
) -> Dict[str, object]:
    start_time = perf_counter()
    confed = (opponent_confederation or "").strip().upper()
    logger.info(
        "team_vs_confed_service_started request_id=%s team=%s opponent_confed=%s source=supabase",
        request_id,
        team,
        confed,
    )

    team_variants = _national_team_name_variants(team)
    filtered = _only_completed_matches(_fetch_team_vs_confed_matches(team, confed, request_id=request_id))

    record = _team_record(team, filtered, team_variants)
    result = {
        "team": team,
        "opponent_confederation": confed,
        **record,
    }
    logger.info(
        "team_vs_confed_service_completed request_id=%s matches_count=%s elapsed_ms=%.2f",
        request_id,
        result.get("matches_count"),
        (perf_counter() - start_time) * 1000.0,
    )
    return result
