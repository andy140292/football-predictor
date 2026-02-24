import os
import re
import sys
import types
import unicodedata
import logging
from difflib import get_close_matches
from datetime import datetime
from time import perf_counter

import joblib
import jwt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from .club_feature_engineering import ClubFeatureEngineer
    from .supabase_client import get_supabase_client
    from .paths import (
        CHAMPIONS_MODEL_PATHS,
        CHAMPIONS_PROCESSED_X_FULL_PATH,
        CHAMPIONS_PROCESSED_X_PATH,
        CHAMPIONS_STATE_SNAPSHOT_PATH,
        CLUB_COEFFICIENTS_HISTORY_PATH,
        CLUB_COEFFICIENTS_PATH,
        CLUB_STATE_SNAPSHOT_PATH,
        CLUB_TEAM_ALIASES_PATH,
        COUNTRY_COEFFICIENTS_HISTORY_PATH,
        CLUB_MODEL_PATHS,
        CLUB_PROCESSED_X_FULL_PATH,
        CLUB_PROCESSED_X_PATH,
        MODEL_PATHS,
        PROCESSED_X_PATH,
        RANKING_PATH,
    )
except ImportError:  # pragma: no cover - fallback for direct module execution
    from src.backend.club_feature_engineering import ClubFeatureEngineer
    from src.backend.supabase_client import get_supabase_client
    from src.backend.paths import (
        CHAMPIONS_MODEL_PATHS,
        CHAMPIONS_PROCESSED_X_FULL_PATH,
        CHAMPIONS_PROCESSED_X_PATH,
        CHAMPIONS_STATE_SNAPSHOT_PATH,
        CLUB_COEFFICIENTS_HISTORY_PATH,
        CLUB_COEFFICIENTS_PATH,
        CLUB_STATE_SNAPSHOT_PATH,
        CLUB_TEAM_ALIASES_PATH,
        COUNTRY_COEFFICIENTS_HISTORY_PATH,
        CLUB_MODEL_PATHS,
        CLUB_PROCESSED_X_FULL_PATH,
        CLUB_PROCESSED_X_PATH,
        MODEL_PATHS,
        PROCESSED_X_PATH,
        RANKING_PATH,
    )

supabase = get_supabase_client()
API_ENV = os.getenv("API_ENV", "prod")
PREDICTION_PROBA_FLOOR = float(os.getenv("PREDICTION_PROBA_FLOOR", "0.005"))
CLUB_NAME_TOKENS = {"fc", "cf", "sc", "afc", "fk", "ac", "ss", "sv", "as"}
logger = logging.getLogger("futbolconu.predict")

_assets = None
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIFA_CODES_PATH = os.getenv("FIFA_CODES_PATH", os.path.join(_PROJECT_ROOT, "data", "fifa_country_codes.csv"))
TEAM_CODE_ALIASES_PATH = os.getenv(
    "TEAM_CODE_ALIASES_PATH", os.path.join(_PROJECT_ROOT, "data", "team_code_aliases.csv")
)
_team_name_to_code = None
_team_alias_to_code = None


class FootballMatchPredictor:
    """Compatibility wrapper for legacy pickled predictor objects."""

    def __init__(self, model_type="random_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.X_train_columns = None

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Loaded predictor has no underlying model.")
        if self.model_type in ["logistic_regression", "mlp"]:
            if self.scaler is None:
                raise ValueError("Loaded predictor missing scaler for scaled model type.")
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
        return self.model.predict_proba(X)


def _register_pickle_compat_modules():
    """Expose legacy module paths expected by existing pickled models."""
    def _ensure_package(name: str):
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        elif not hasattr(module, "__path__"):
            module.__path__ = []
        return module

    _ensure_package("prediction")
    _ensure_package("src")
    _ensure_package("src.prediction")
    _ensure_package("backend")
    _ensure_package("backend.prediction")
    _ensure_package("src.backend")
    _ensure_package("src.backend.prediction")

    compat_module = types.ModuleType("prediction.football_match_predictor")
    compat_module.FootballMatchPredictor = FootballMatchPredictor
    for alias in (
        "prediction.football_match_predictor",
        "src.prediction.football_match_predictor",
        "backend.prediction.football_match_predictor",
        "src.backend.prediction.football_match_predictor",
    ):
        sys.modules[alias] = compat_module


_register_pickle_compat_modules()


def _load_assets():
    global _assets
    if _assets is None:
        X = pd.read_csv(PROCESSED_X_PATH)
        fifa_rank = pd.read_csv(RANKING_PATH)
        rf_predictor = joblib.load(MODEL_PATHS["random_forest"])
        lr_predictor = joblib.load(MODEL_PATHS["logistic_regression"])
        mlp_predictor = joblib.load(MODEL_PATHS["mlp"])
        _assets = {
            "X": X,
            "fifa_rank": fifa_rank,
            "models": {
                "random_forest": rf_predictor,
                "logistic_regression": lr_predictor,
                "mlp": mlp_predictor,
            },
        }
    return _assets


def _stabilize_probabilities(raw_probs):
    probs = np.asarray(raw_probs, dtype=float)
    floor = float(PREDICTION_PROBA_FLOOR)
    if floor <= 0.0:
        return probs

    capped = np.clip(probs, floor, 1.0 - floor)
    # Preserve original values when no clipping is needed.
    if np.array_equal(capped, probs):
        return probs

    total = float(capped.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    return capped / total


_club_assets = None
_champions_assets = None
_club_team_aliases = None
_STATE_HISTORY_COLS = ["date", "home_team", "away_team", "home_score", "away_score"]


def _is_dev_mode() -> bool:
    # Resolve at runtime to avoid import-order issues with load_dotenv.
    return os.getenv("API_ENV", API_ENV) == "dev"


def _get_request_supabase_client(token: str = None):
    if token:
        return get_supabase_client(access_token=token)
    return supabase


def _load_latest_club_coefficients():
    coeff_history_path = str(CLUB_COEFFICIENTS_HISTORY_PATH)
    if os.path.exists(coeff_history_path):
        club_coefficients = pd.read_csv(coeff_history_path)
        if "uefa_season_year" in club_coefficients.columns:
            club_coefficients["uefa_season_year"] = pd.to_numeric(
                club_coefficients["uefa_season_year"], errors="coerce"
            )
            club_coefficients = (
                club_coefficients.sort_values("uefa_season_year", ascending=False)
                .drop_duplicates(subset=["team"], keep="first")
                .reset_index(drop=True)
            )
        country_coeff_path = str(COUNTRY_COEFFICIENTS_HISTORY_PATH)
        if os.path.exists(country_coeff_path):
            country_coefficients = pd.read_csv(country_coeff_path)
            if not country_coefficients.empty and "country" in country_coefficients.columns:
                if "uefa_season_year" in country_coefficients.columns:
                    country_coefficients["uefa_season_year"] = pd.to_numeric(
                        country_coefficients["uefa_season_year"], errors="coerce"
                    )
                    country_coefficients = (
                        country_coefficients.sort_values("uefa_season_year", ascending=False)
                        .drop_duplicates(subset=["country"], keep="first")
                        .reset_index(drop=True)
                    )
                merge_cols = [
                    col
                    for col in [
                        "country",
                        "overall_country_coefficient",
                        "season_country_coefficient",
                        "overall_rank",
                        "season_rank",
                    ]
                    if col in country_coefficients.columns
                ]
                if set(["country", "overall_country_coefficient", "season_country_coefficient"]).issubset(
                    set(merge_cols)
                ):
                    country_for_merge = country_coefficients[merge_cols].rename(
                        columns={
                            "overall_rank": "country_uefa_overall_rank",
                            "season_rank": "country_uefa_season_rank",
                        }
                    )
                    club_coefficients = club_coefficients.merge(
                        country_for_merge, on="country", how="left"
                    )
        return club_coefficients
    if os.path.exists(CLUB_COEFFICIENTS_PATH):
        return pd.read_csv(CLUB_COEFFICIENTS_PATH)
    return pd.DataFrame()


def _load_state_history_frame(csv_path):
    path = str(csv_path)
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return pd.DataFrame(columns=_STATE_HISTORY_COLS)

    if not set(_STATE_HISTORY_COLS).issubset(set(header.columns)):
        return pd.DataFrame(columns=_STATE_HISTORY_COLS)

    return pd.read_csv(path, usecols=_STATE_HISTORY_COLS)


def _load_state_snapshot(snapshot_path):
    path = str(snapshot_path)
    if not path or not os.path.exists(path):
        return None, None, None

    try:
        payload = joblib.load(path)
    except Exception as exc:
        logger.warning("state_snapshot_load_failed path=%s error=%s", path, exc)
        return None, None, None

    if not isinstance(payload, dict):
        logger.warning("state_snapshot_invalid_payload path=%s type=%s", path, type(payload).__name__)
        return None, None, None

    team_states = payload.get("team_states")
    pair_states = payload.get("pair_states")
    if not isinstance(team_states, dict) or not isinstance(pair_states, dict):
        logger.warning("state_snapshot_missing_dicts path=%s", path)
        return None, None, None

    return team_states, pair_states, payload.get("meta", {})


def _build_club_mode_assets(processed_x_path, processed_x_full_path, model_paths, state_snapshot_path=None):
    X = pd.read_csv(processed_x_path)
    # Keep only the minimal columns required to derive current team/pair states.
    # Loading the full engineered matrix can exceed Fly machine memory limits.
    X_full = _load_state_history_frame(processed_x_full_path)
    club_coefficients = _load_latest_club_coefficients()

    team_states, pair_states = ({}, {})
    snapshot_loaded = False
    if state_snapshot_path is not None:
        snapshot_team_states, snapshot_pair_states, snapshot_meta = _load_state_snapshot(state_snapshot_path)
        if snapshot_team_states is not None and snapshot_pair_states is not None:
            team_states, pair_states = snapshot_team_states, snapshot_pair_states
            snapshot_loaded = True
            logger.info(
                "state_snapshot_loaded path=%s teams=%s pairs=%s generated_at_utc=%s rows=%s",
                state_snapshot_path,
                len(team_states),
                len(pair_states),
                snapshot_meta.get("generated_at_utc"),
                snapshot_meta.get("rows"),
            )

    if (not snapshot_loaded) and {"date", "home_team", "away_team", "home_score", "away_score"}.issubset(set(X_full.columns)):
        state_start = perf_counter()
        team_states, pair_states = ClubFeatureEngineer.build_current_states(X_full)
        logger.warning(
            "state_snapshot_missing_or_invalid path=%s recomputed_states teams=%s pairs=%s elapsed_ms=%.2f",
            state_snapshot_path,
            len(team_states),
            len(pair_states),
            (perf_counter() - state_start) * 1000.0,
        )

    rf_predictor = joblib.load(model_paths["random_forest"])
    lr_predictor = joblib.load(model_paths["logistic_regression"])
    mlp_predictor = joblib.load(model_paths["mlp"])
    return {
        "X": X,
        "X_full": X_full,
        "club_coefficients": club_coefficients,
        "team_states": team_states,
        "pair_states": pair_states,
        "models": {
            "random_forest": rf_predictor,
            "logistic_regression": lr_predictor,
            "mlp": mlp_predictor,
        },
    }


def _load_club_assets():
    global _club_assets
    if _club_assets is None:
        _club_assets = _build_club_mode_assets(
            CLUB_PROCESSED_X_PATH,
            CLUB_PROCESSED_X_FULL_PATH,
            CLUB_MODEL_PATHS,
            CLUB_STATE_SNAPSHOT_PATH,
        )
    return _club_assets


def _load_champions_assets():
    global _champions_assets
    if _champions_assets is None:
        _champions_assets = _build_club_mode_assets(
            CHAMPIONS_PROCESSED_X_PATH,
            CHAMPIONS_PROCESSED_X_FULL_PATH,
            CHAMPIONS_MODEL_PATHS,
            CHAMPIONS_STATE_SNAPSHOT_PATH,
        )
    return _champions_assets


def extract_email_from_token(token: str) -> str:
    if _is_dev_mode():
        return "dev@example.com"

    if not token:
        return None

    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload.get("email")
    except Exception as e:
        print(f"❌ Error al decodificar token: {e}")
        return None


def extract_user_id_from_token(token: str) -> str:
    if _is_dev_mode():
        return "dev-user"

    if not token:
        return None

    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload.get("sub")
    except Exception as e:
        print(f"❌ Error al decodificar token para user_id: {e}")
        return None


def _normalize_user_id(user_id: str) -> str:
    if not user_id:
        return None

    if _is_dev_mode() and user_id == "dev-user":
        return "00000000-0000-0000-0000-000000000000"

    return str(user_id)


def _to_iso_date(value) -> str:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date().isoformat()

    raw = str(value).strip()
    if not raw:
        return None

    try:
        return datetime.strptime(raw[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def _normalize_team_code(value: str) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    return raw if re.fullmatch(r"[A-Z]{3}", raw) else ""


def _load_team_code_mappings() -> tuple[dict, dict]:
    global _team_name_to_code, _team_alias_to_code
    if _team_name_to_code is not None and _team_alias_to_code is not None:
        return _team_name_to_code, _team_alias_to_code

    _team_name_to_code = {}
    _team_alias_to_code = {}

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
                    code = _normalize_team_code(row.get(code_col))
                    name = str(row.get(name_col, "") or "").strip()
                    if code and name:
                        _team_name_to_code[_normalized_text(name)] = code

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
                    code = _normalize_team_code(row.get(code_col))
                    alias = str(row.get(alias_col, "") or "").strip()
                    if code and alias:
                        _team_alias_to_code[_normalized_text(alias)] = code

    return _team_name_to_code, _team_alias_to_code


def _resolve_team_code(team_name: str) -> str:
    if not team_name:
        return ""
    name_to_code, alias_to_code = _load_team_code_mappings()
    key = _normalized_text(team_name)
    if key in alias_to_code:
        return alias_to_code[key]
    return name_to_code.get(key, "")


def _query_future_matches_pair(
    home_col: str,
    away_col: str,
    home_value: str,
    away_value: str,
    today_iso: str,
    token: str = None,
) -> list[dict]:
    client = _get_request_supabase_client(token)
    rows = []
    for home, away in ((home_value, away_value), (away_value, home_value)):
        result = (
            client.table("matches_calendar")
            .select("match_id,home_team,away_team,match_date")
            .eq(home_col, home)
            .eq(away_col, away)
            .gte("match_date", today_iso)
            .order("match_date")
            .execute()
        )
        rows.extend(result.data or [])
    return rows


def _fetch_future_matches_for_pair(home_team: str, away_team: str, token: str = None) -> list[dict]:
    today_iso = datetime.utcnow().date().isoformat()
    rows = []
    home_code = _resolve_team_code(home_team)
    away_code = _resolve_team_code(away_team)

    if home_code and away_code:
        try:
            rows.extend(
                _query_future_matches_pair(
                    home_col="home_team_code",
                    away_col="away_team_code",
                    home_value=home_code,
                    away_value=away_code,
                    today_iso=today_iso,
                    token=token,
                )
            )
        except Exception as exc:
            logger.warning(
                "future_matches_code_lookup_failed home_team=%s away_team=%s home_code=%s away_code=%s error=%s",
                home_team,
                away_team,
                home_code,
                away_code,
                exc,
            )

    rows.extend(
        _query_future_matches_pair(
            home_col="home_team",
            away_col="away_team",
            home_value=home_team,
            away_value=away_team,
            today_iso=today_iso,
            token=token,
        )
    )

    deduped = {}
    for row in rows:
        match_id = str(row.get("match_id") or "")
        match_date = _to_iso_date(row.get("match_date"))
        key = match_id or f"{row.get('home_team')}|{row.get('away_team')}|{match_date}"
        if key in deduped:
            continue
        deduped[key] = {
            "match_id": match_id,
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "match_date": match_date,
        }

    return sorted(
        deduped.values(),
        key=lambda item: (item.get("match_date") or "", item.get("home_team") or "", item.get("away_team") or ""),
    )


def _get_user_predicted_match_ids(user_id: str, token: str = None) -> set[str]:
    normalized_user_id = _normalize_user_id(user_id)
    if not normalized_user_id:
        return set()

    client = _get_request_supabase_client(token)
    result = (
        client.table("match_predictions")
        .select("match_id")
        .eq("user_id", normalized_user_id)
        .execute()
    )
    return {str(row.get("match_id")) for row in (result.data or []) if row.get("match_id")}


def get_unpredicted_future_matches(
    home_team: str,
    away_team: str,
    user_id: str = None,
    token: str = None,
    request_id: str = "-",
) -> list[dict]:
    try:
        if token:
            matches = _fetch_future_matches_for_pair(home_team, away_team, token=token)
        else:
            matches = _fetch_future_matches_for_pair(home_team, away_team)
    except Exception as exc:
        logger.warning(
            "future_matches_lookup_failed request_id=%s home_team=%s away_team=%s error=%s",
            request_id,
            home_team,
            away_team,
            exc,
        )
        return []

    if not matches or not user_id:
        return matches

    try:
        if token:
            predicted_match_ids = _get_user_predicted_match_ids(user_id, token=token)
        else:
            predicted_match_ids = _get_user_predicted_match_ids(user_id)
    except Exception as exc:
        logger.warning(
            "future_matches_user_filter_failed request_id=%s user_id=%s error=%s",
            request_id,
            user_id,
            exc,
        )
        return matches

    return [match for match in matches if str(match.get("match_id")) not in predicted_match_ids]


def _serialize_match_prediction(row: dict) -> dict:
    return {
        "prediction_id": row.get("prediction_id"),
        "match_id": row.get("match_id"),
        "predicted_outcome": row.get("predicted_outcome"),
        "created_at": str(row.get("created_at") or ""),
    }


def _get_calendar_match_by_id(match_id: str, token: str = None) -> dict:
    client = _get_request_supabase_client(token)
    result = (
        client.table("matches_calendar")
        .select("match_id,home_team,away_team,match_date")
        .eq("match_id", match_id)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return rows[0] if rows else None


def _get_existing_match_prediction(user_id: str, match_id: str, token: str = None) -> dict:
    normalized_user_id = _normalize_user_id(user_id)
    client = _get_request_supabase_client(token)
    result = (
        client.table("match_predictions")
        .select("prediction_id,match_id,predicted_outcome,created_at")
        .eq("user_id", normalized_user_id)
        .eq("match_id", match_id)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return _serialize_match_prediction(rows[0]) if rows else None


def _insert_match_prediction(
    user_id: str,
    email: str,
    match_id: str,
    predicted_outcome: str,
    token: str = None,
) -> dict:
    normalized_user_id = _normalize_user_id(user_id)
    payload = {
        "user_id": normalized_user_id,
        "email": email,
        "match_id": match_id,
        "predicted_outcome": predicted_outcome,
    }
    client = _get_request_supabase_client(token)
    result = client.table("match_predictions").insert(payload).execute()
    rows = result.data or []
    if rows:
        return _serialize_match_prediction(rows[0])
    if token:
        return _get_existing_match_prediction(normalized_user_id, match_id, token=token)
    return _get_existing_match_prediction(normalized_user_id, match_id)


def create_or_get_match_prediction(
    user_id: str,
    email: str,
    match_id: str,
    predicted_outcome: str,
    token: str = None,
    request_id: str = "-",
) -> dict:
    normalized_user_id = _normalize_user_id(user_id)
    if not normalized_user_id:
        raise ValueError("No se pudo resolver el user_id del token")

    if token:
        calendar_match = _get_calendar_match_by_id(match_id, token=token)
    else:
        calendar_match = _get_calendar_match_by_id(match_id)
    if not calendar_match:
        raise ValueError("El partido no existe en matches_calendar")

    today_iso = datetime.utcnow().date().isoformat()
    match_date = _to_iso_date(calendar_match.get("match_date"))
    if not match_date or match_date < today_iso:
        raise ValueError("No se puede registrar una predicción para un partido pasado")

    if token:
        existing = _get_existing_match_prediction(normalized_user_id, match_id, token=token)
    else:
        existing = _get_existing_match_prediction(normalized_user_id, match_id)
    if existing:
        return {"status": "exists", "prediction": existing}

    try:
        created = _insert_match_prediction(
            user_id=normalized_user_id,
            email=email,
            match_id=match_id,
            predicted_outcome=predicted_outcome,
            token=token,
        )
    except Exception as exc:
        error_text = str(exc).lower()
        if "duplicate key" in error_text or "23505" in error_text:
            if token:
                existing = _get_existing_match_prediction(normalized_user_id, match_id, token=token)
            else:
                existing = _get_existing_match_prediction(normalized_user_id, match_id)
            if existing:
                return {"status": "exists", "prediction": existing}
        logger.exception(
            "match_prediction_insert_failed request_id=%s user_id=%s match_id=%s",
            request_id,
            normalized_user_id,
            match_id,
        )
        raise

    if not created:
        raise RuntimeError("No se pudo registrar la predicción del usuario")

    return {"status": "created", "prediction": created}


def _validate_calendar_row(
    home_team: str,
    away_team: str,
    match_date: str,
    home_team_code: str = None,
    away_team_code: str = None,
) -> tuple[dict, str]:
    home = str(home_team or "").strip()
    away = str(away_team or "").strip()
    if not home:
        return None, "home_team is required"
    if not away:
        return None, "away_team is required"
    if home.casefold() == away.casefold():
        return None, "home_team and away_team must be different"

    parsed_date = _to_iso_date(match_date)
    if not parsed_date:
        return None, "match_date must use YYYY-MM-DD format"

    home_code = _normalize_team_code(home_team_code)
    away_code = _normalize_team_code(away_team_code)
    if (home_team_code or away_team_code) and (not home_code or not away_code):
        return None, "home_team_code and away_team_code must be 3-letter FIFA codes"
    if home_code and away_code and home_code == away_code:
        return None, "home_team_code and away_team_code must be different"

    row = {
        "home_team": home,
        "away_team": away,
        "match_date": parsed_date,
    }
    if home_code and away_code:
        row["home_team_code"] = home_code
        row["away_team_code"] = away_code
    return row, ""


def _calendar_row_exists(row: dict) -> bool:
    home_team = row.get("home_team")
    away_team = row.get("away_team")
    match_date = row.get("match_date")
    home_team_code = row.get("home_team_code")
    away_team_code = row.get("away_team_code")

    if home_team_code and away_team_code:
        try:
            by_code = (
                supabase.table("matches_calendar")
                .select("match_id")
                .eq("home_team_code", home_team_code)
                .eq("away_team_code", away_team_code)
                .eq("match_date", match_date)
                .limit(1)
                .execute()
            )
            if by_code.data:
                return True
        except Exception as exc:
            logger.warning("calendar_exists_by_code_failed error=%s", exc)

    by_name = (
        supabase.table("matches_calendar")
        .select("match_id")
        .eq("home_team", home_team)
        .eq("away_team", away_team)
        .eq("match_date", match_date)
        .limit(1)
        .execute()
    )
    return bool(by_name.data)


def _bulk_upsert_calendar_rows(rows: list[dict]) -> None:
    if not rows:
        return
    try:
        supabase.table("matches_calendar").upsert(
            rows,
            on_conflict="home_team,away_team,match_date",
        ).execute()
    except Exception as exc:
        if "home_team_code" not in str(exc) and "away_team_code" not in str(exc):
            raise
        logger.warning("calendar_upsert_code_columns_missing_fallback error=%s", exc)
        stripped_rows = [
            {
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "match_date": row.get("match_date"),
            }
            for row in rows
        ]
        supabase.table("matches_calendar").upsert(
            stripped_rows,
            on_conflict="home_team,away_team,match_date",
        ).execute()


def upsert_matches_calendar_batch(matches: list[dict], request_id: str = "-") -> dict:
    summary = {
        "received": len(matches or []),
        "inserted": 0,
        "updated": 0,
        "skipped": 0,
        "errors": [],
    }

    if not matches:
        return summary

    seen_keys = set()
    valid_rows = []
    for index, row in enumerate(matches):
        normalized_row, error = _validate_calendar_row(
            (row or {}).get("home_team"),
            (row or {}).get("away_team"),
            (row or {}).get("match_date"),
            (row or {}).get("home_team_code"),
            (row or {}).get("away_team_code"),
        )
        if error:
            summary["skipped"] += 1
            summary["errors"].append({"row_index": index, "reason": error})
            continue

        key = (
            normalized_row.get("home_team_code"),
            normalized_row.get("away_team_code"),
            normalized_row.get("home_team"),
            normalized_row.get("away_team"),
            normalized_row.get("match_date"),
        )
        if key in seen_keys:
            summary["skipped"] += 1
            summary["errors"].append({"row_index": index, "reason": "duplicate row in payload"})
            continue

        seen_keys.add(key)
        valid_rows.append(normalized_row)

    if not valid_rows:
        return summary

    existing_keys = set()
    for row in valid_rows:
        key = (
            row.get("home_team_code"),
            row.get("away_team_code"),
            row["home_team"],
            row["away_team"],
            row["match_date"],
        )
        if _calendar_row_exists(row):
            existing_keys.add(key)

    _bulk_upsert_calendar_rows(valid_rows)

    for row in valid_rows:
        key = (
            row.get("home_team_code"),
            row.get("away_team_code"),
            row["home_team"],
            row["away_team"],
            row["match_date"],
        )
        if key in existing_keys:
            summary["updated"] += 1
        else:
            summary["inserted"] += 1

    logger.info(
        "calendar_batch_upsert_completed request_id=%s received=%s inserted=%s updated=%s skipped=%s errors=%s",
        request_id,
        summary["received"],
        summary["inserted"],
        summary["updated"],
        summary["skipped"],
        len(summary["errors"]),
    )
    return summary


def get_remaining_predictions(email: str, limit=15, token: str = None, user_id: str = None):
    if _is_dev_mode():
        return True

    if email == "andresuribe1402@gmail.com":
        return True

    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    client = _get_request_supabase_client(token)
    normalized_user_id = _normalize_user_id(user_id or "")
    if normalized_user_id:
        try:
            result = (
                client.table("user_predictions")
                .select("*")
                .eq("user_id", normalized_user_id)
                .gte("timestamp", today_start)
                .execute()
            )
        except Exception as exc:
            error_text = str(exc).lower()
            if "column" in error_text and "user_id" in error_text:
                result = (
                    client.table("user_predictions")
                    .select("*")
                    .eq("email", email)
                    .gte("timestamp", today_start)
                    .execute()
                )
            else:
                raise
    else:
        result = (
            client.table("user_predictions")
            .select("*")
            .eq("email", email)
            .gte("timestamp", today_start)
            .execute()
        )

    used = len(result.data)
    return max(limit - used, 0)


def register_prediction(
    email: str,
    home_team: str,
    away_team: str,
    token: str = None,
    user_id: str = None,
):
    if _is_dev_mode():
        return

    client = _get_request_supabase_client(token)
    payload = {
        "email": email,
        "home_team": home_team,
        "away_team": away_team,
        "timestamp": datetime.utcnow().isoformat(),
    }
    normalized_user_id = _normalize_user_id(user_id or "")
    if normalized_user_id:
        payload["user_id"] = normalized_user_id

    try:
        client.table("user_predictions").insert(payload).execute()
    except Exception as exc:
        error_text = str(exc).lower()
        if "column" in error_text and "user_id" in error_text and "user_id" in payload:
            payload.pop("user_id", None)
            client.table("user_predictions").insert(payload).execute()
            return
        raise


def predict_outcome(
    home_team,
    away_team,
    token=None,
    user_email=None,
    user_id=None,
    mode="national",
    competition=None,
    round_name=None,
    neutral=None,
    request_id=None,
):
    request_id = request_id or "-"
    request_start = perf_counter()
    logger.info(
        "predict_outcome_started request_id=%s mode=%s home_team=%s away_team=%s competition=%s round=%s neutral=%s",
        request_id,
        mode,
        home_team,
        away_team,
        competition,
        round_name,
        neutral,
    )

    email = user_email or extract_email_from_token(token)
    if not email:
        logger.warning("predict_outcome_failed_no_email request_id=%s", request_id)
        raise ValueError("No se pudo extraer el email del token")
    user_id = user_id or extract_user_id_from_token(token)
    logger.info("predict_outcome_user_resolved request_id=%s", request_id)

    quota_start = perf_counter()
    remaining_predictions = get_remaining_predictions(email, 15, token, user_id)
    logger.info(
        "predict_outcome_quota_checked request_id=%s remaining=%s elapsed_ms=%.2f",
        request_id,
        remaining_predictions,
        (perf_counter() - quota_start) * 1000.0,
    )
    if not remaining_predictions:
        logger.warning("predict_outcome_failed_quota_exceeded request_id=%s", request_id)
        raise ValueError("Límite diario de predicciones alcanzado (15)")

    registration_start = perf_counter()
    register_prediction(email, home_team, away_team, token, user_id)
    logger.info(
        "predict_outcome_registered_prediction request_id=%s elapsed_ms=%.2f",
        request_id,
        (perf_counter() - registration_start) * 1000.0,
    )

    if mode not in {"national", "club", "champions"}:
        logger.warning("predict_outcome_failed_invalid_mode request_id=%s mode=%s", request_id, mode)
        raise ValueError(f"Modo inválido: {mode}")

    assets_start = perf_counter()
    if mode == "national":
        assets = _load_assets()
    elif mode == "club":
        assets = _load_club_assets()
    else:
        assets = _load_champions_assets()
    logger.info(
        "predict_outcome_assets_loaded request_id=%s mode=%s elapsed_ms=%.2f",
        request_id,
        mode,
        (perf_counter() - assets_start) * 1000.0,
    )

    X = assets["X"]
    models = assets["models"]
    ranking_data = assets.get("fifa_rank") if mode == "national" else assets.get("club_coefficients")

    feature_start = perf_counter()
    logger.info(
        "predict_outcome_feature_vector_build_started request_id=%s mode=%s template_cols=%s",
        request_id,
        mode,
        len(X.columns),
    )
    match_vector = build_feature_vector(
        home_team,
        away_team,
        X,
        ranking_data,
        mode=mode,
        history_df=assets.get("X_full"),
        competition=competition,
        round_name=round_name,
        neutral=neutral,
        team_states=assets.get("team_states"),
        pair_states=assets.get("pair_states"),
        request_id=request_id,
    )
    non_zero_features = int((match_vector.iloc[0] != 0).sum()) if len(match_vector) > 0 else 0
    logger.info(
        "predict_outcome_feature_vector_build_completed request_id=%s elapsed_ms=%.2f vector_rows=%s vector_cols=%s non_zero_features=%s",
        request_id,
        (perf_counter() - feature_start) * 1000.0,
        len(match_vector),
        len(match_vector.columns),
        non_zero_features,
    )

    results = {}

    rf_predictor = models.get("random_forest")
    if rf_predictor:
        model_start = perf_counter()
        rf_probs = _stabilize_probabilities(rf_predictor.predict_proba(match_vector)[0])
        results["random_forest"] = {
            "home_win": rf_probs[2],
            "draw": rf_probs[1],
            "away_win": rf_probs[0],
        }
        logger.info(
            "predict_outcome_model_scored request_id=%s model=random_forest elapsed_ms=%.2f",
            request_id,
            (perf_counter() - model_start) * 1000.0,
        )
    else:
        logger.warning("predict_outcome_model_missing request_id=%s model=random_forest", request_id)

    lr_predictor = models.get("logistic_regression")
    if lr_predictor:
        model_start = perf_counter()
        lr_probs = _stabilize_probabilities(lr_predictor.predict_proba(match_vector)[0])
        results["logistic_regression"] = {
            "home_win": lr_probs[2],
            "draw": lr_probs[1],
            "away_win": lr_probs[0],
        }
        logger.info(
            "predict_outcome_model_scored request_id=%s model=logistic_regression elapsed_ms=%.2f",
            request_id,
            (perf_counter() - model_start) * 1000.0,
        )
    else:
        logger.warning("predict_outcome_model_missing request_id=%s model=logistic_regression", request_id)

    mlp_predictor = models.get("mlp")
    if mlp_predictor:
        model_start = perf_counter()
        mlp_probs = _stabilize_probabilities(mlp_predictor.predict_proba(match_vector)[0])
        results["mlp"] = {
            "home_win": mlp_probs[2],
            "draw": mlp_probs[1],
            "away_win": mlp_probs[0],
        }
        logger.info(
            "predict_outcome_model_scored request_id=%s model=mlp elapsed_ms=%.2f",
            request_id,
            (perf_counter() - model_start) * 1000.0,
        )
    else:
        logger.warning("predict_outcome_model_missing request_id=%s model=mlp", request_id)

    future_matches_start = perf_counter()
    results["future_matches"] = get_unpredicted_future_matches(
        home_team=home_team,
        away_team=away_team,
        user_id=user_id,
        token=token,
        request_id=request_id,
    )
    logger.info(
        "predict_outcome_future_matches_loaded request_id=%s count=%s elapsed_ms=%.2f",
        request_id,
        len(results["future_matches"]),
        (perf_counter() - future_matches_start) * 1000.0,
    )

    model_keys = [key for key in results.keys() if key != "future_matches"]
    logger.info(
        "predict_outcome_completed request_id=%s models=%s elapsed_ms=%.2f",
        request_id,
        ",".join(sorted(model_keys)),
        (perf_counter() - request_start) * 1000.0,
    )
    return results


def _build_national_feature_vector(home_team, away_team, feature_template_df, ranking_df):
    import numpy as np

    vector = pd.DataFrame([np.zeros(len(feature_template_df.columns))], columns=feature_template_df.columns)

    home_row = ranking_df[ranking_df["team"] == home_team]
    away_row = ranking_df[ranking_df["team"] == away_team]
    if home_row.empty or away_row.empty:
        raise ValueError(f"Team not found in ranking: {home_team} or {away_team}")
    home_info = home_row.iloc[0]
    away_info = away_row.iloc[0]

    if "home_team_fifa_rank" in vector.columns:
        vector["home_team_fifa_rank"] = home_info["ranking"]
    if "away_team_fifa_rank" in vector.columns:
        vector["away_team_fifa_rank"] = away_info["ranking"]
    if "home_team_fifa_points" in vector.columns:
        vector["home_team_fifa_points"] = home_info["points"]
    if "away_team_fifa_points" in vector.columns:
        vector["away_team_fifa_points"] = away_info["points"]

    if "fifa_rank_diff" in vector.columns:
        vector["fifa_rank_diff"] = home_info["ranking"] - away_info["ranking"]
    if "fifa_points_diff" in vector.columns:
        vector["fifa_points_diff"] = home_info["points"] - away_info["points"]

    home_col = f"home_team_{home_team}"
    away_col = f"away_team_{away_team}"
    if home_col in vector.columns:
        vector[home_col] = 1
    if away_col in vector.columns:
        vector[away_col] = 1

    conf_pair_col = f"confed_pair_{home_info['confederation']}_vs_{away_info['confederation']}"
    if conf_pair_col in vector.columns:
        vector[conf_pair_col] = 1

    home_conf = f"home_team_confederation_{home_info['confederation']}"
    away_conf = f"away_team_confederation_{away_info['confederation']}"
    if home_conf in vector.columns:
        vector[home_conf] = 1
    if away_conf in vector.columns:
        vector[away_conf] = 1

    if "neutral" in vector.columns:
        vector["neutral"] = int(home_info["confederation"] != away_info["confederation"])

    return vector


def _normalized_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch for ch in text if ch.isalnum())


def _load_club_team_aliases() -> dict:
    global _club_team_aliases
    if _club_team_aliases is not None:
        return _club_team_aliases

    alias_path = str(CLUB_TEAM_ALIASES_PATH)
    if not os.path.exists(alias_path):
        _club_team_aliases = {}
        return _club_team_aliases

    try:
        aliases = pd.read_csv(alias_path)
    except Exception:
        _club_team_aliases = {}
        return _club_team_aliases

    source_candidates = ["alias", "source_name", "source_team", "match_team", "from"]
    target_candidates = ["canonical", "target_name", "target_team", "uefa_team", "to"]
    source_col = next((col for col in source_candidates if col in aliases.columns), None)
    target_col = next((col for col in target_candidates if col in aliases.columns), None)
    if source_col is None or target_col is None:
        _club_team_aliases = {}
        return _club_team_aliases

    mapped = {}
    for _, row in aliases[[source_col, target_col]].iterrows():
        source = str(row.get(source_col, "") or "").strip()
        target = str(row.get(target_col, "") or "").strip()
        if source and target:
            mapped[_normalized_text(source)] = target
    _club_team_aliases = mapped
    return _club_team_aliases


def _canonical_club_name(value: str) -> str:
    team = str(value or "").strip()
    aliases = _load_club_team_aliases()
    return aliases.get(_normalized_text(team), team)


def _club_name_keys(value: str) -> set[str]:
    text = _canonical_club_name(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    tokens = re.findall(r"[a-z0-9]+", text)
    if not tokens:
        return set()
    keys = {"".join(tokens)}
    filtered = [tok for tok in tokens if tok not in CLUB_NAME_TOKENS]
    if filtered:
        keys.add("".join(filtered))
    return {k for k in keys if k}


def _build_club_coeff_lookup(ranking_df: pd.DataFrame) -> dict:
    if ranking_df is None or ranking_df.empty:
        return {}

    ranking = ranking_df.copy()
    if "uefa_season_year" in ranking.columns:
        ranking["uefa_season_year"] = pd.to_numeric(ranking["uefa_season_year"], errors="coerce")
        ranking = ranking.sort_values(["uefa_season_year", "overall_rank"], ascending=[False, True])

    name_cols = [col for col in ["team", "display_name", "official_name"] if col in ranking.columns]
    lookup = {}
    for _, row in ranking.iterrows():
        for col in name_cols:
            for key in _club_name_keys(row.get(col, "")):
                if key not in lookup:
                    lookup[key] = row
    return lookup


def _lookup_club_coeff_row(team_name: str, coeff_lookup: dict):
    if not coeff_lookup:
        return None
    for key in _club_name_keys(team_name):
        if key in coeff_lookup:
            return coeff_lookup[key]
    return None


def _suggest_club_names(team_name: str, coeff_lookup: dict, max_suggestions: int = 3) -> list[str]:
    if not coeff_lookup:
        return []
    available = sorted({str(row.get("team", "")).strip() for row in coeff_lookup.values() if str(row.get("team", "")).strip()})
    if not available:
        return []
    return get_close_matches(team_name, available, n=max_suggestions, cutoff=0.72)


def _is_knockout_round(round_name: str) -> int:
    text = str(round_name or "").lower()
    tokens = ["knockout", "play-off", "playoff", "quarter", "semi", "final", "round of", "last "]
    return int(any(token in text for token in tokens))


def _extract_states_from_history_if_needed(home_team, away_team, history_df, team_states, pair_states):
    if team_states and pair_states:
        return team_states, pair_states

    if history_df is None or history_df.empty:
        return team_states or {}, pair_states or {}

    canonical_cols = {"date", "home_team", "away_team", "home_score", "away_score"}
    if canonical_cols.issubset(set(history_df.columns)):
        return ClubFeatureEngineer.build_current_states(history_df)

    return team_states or {}, pair_states or {}


def _resolve_team_name_from_state(team_name: str, team_states: dict) -> str:
    if not isinstance(team_states, dict) or not team_states:
        return team_name
    if team_name in team_states:
        return team_name

    key_to_team = {}
    for known_team in team_states.keys():
        for key in _club_name_keys(known_team):
            if key not in key_to_team:
                key_to_team[key] = known_team

    for key in _club_name_keys(team_name):
        if key in key_to_team:
            return key_to_team[key]
    return team_name


def _build_club_feature_vector(
    home_team,
    away_team,
    feature_template_df,
    history_df,
    ranking_df=None,
    strict_coefficients=False,
    competition=None,
    round_name=None,
    neutral=None,
    team_states=None,
    pair_states=None,
    request_id=None,
):
    import numpy as np
    request_id = request_id or "-"
    feature_build_start = perf_counter()
    logger.info(
        "club_feature_vector_build_started request_id=%s home_team=%s away_team=%s competition=%s round=%s neutral=%s template_cols=%s",
        request_id,
        home_team,
        away_team,
        competition,
        round_name,
        neutral,
        len(feature_template_df.columns),
    )

    vector = pd.DataFrame([np.zeros(len(feature_template_df.columns))], columns=feature_template_df.columns)

    competition = competition or "Champions Lg"
    round_name = round_name or ""
    neutral_value = int(neutral) if neutral is not None else 0

    competition_col = f"competition_{competition}"
    round_col = f"round_{round_name}"
    if competition_col in vector.columns:
        vector[competition_col] = 1
    if round_col in vector.columns:
        vector[round_col] = 1

    if "neutral" in vector.columns:
        vector["neutral"] = neutral_value
    if "is_ucl_match" in vector.columns:
        vector["is_ucl_match"] = int("champions" in str(competition).lower())
    if "is_knockout_round" in vector.columns:
        vector["is_knockout_round"] = _is_knockout_round(round_name)
    if "is_knockout_playoff" in vector.columns:
        vector["is_knockout_playoff"] = int("play-off" in str(round_name).lower() or "playoff" in str(round_name).lower())

    # Optional UEFA coefficient fallback for legacy feature templates.
    if ranking_df is not None and not ranking_df.empty and "team" in ranking_df.columns:
        coeff_lookup = _build_club_coeff_lookup(ranking_df)
        home_info = _lookup_club_coeff_row(home_team, coeff_lookup)
        away_info = _lookup_club_coeff_row(away_team, coeff_lookup)

        if strict_coefficients and (home_info is None or away_info is None):
            suggestions = []
            if home_info is None:
                suggestions.extend(_suggest_club_names(home_team, coeff_lookup))
            if away_info is None:
                suggestions.extend(_suggest_club_names(away_team, coeff_lookup))
            suggestions = sorted({item for item in suggestions if item})
            detail = f"Team not found in UEFA coefficients: {home_team} or {away_team}"
            if suggestions:
                detail += f". Did you mean: {', '.join(suggestions)}?"
            detail += " (you can add aliases in data/club_team_aliases.csv)"
            raise ValueError(detail)

        if home_info is not None and away_info is not None:
            def _safe_float(value, default=0.0):
                try:
                    out = float(value)
                    if np.isnan(out):
                        return float(default)
                    return out
                except Exception:
                    return float(default)

            home_country_missing = any(
                pd.isna(home_info.get(col))
                for col in [
                    "overall_country_coefficient",
                    "season_country_coefficient",
                    "country_uefa_overall_rank",
                    "country_uefa_season_rank",
                ]
            )
            away_country_missing = any(
                pd.isna(away_info.get(col))
                for col in [
                    "overall_country_coefficient",
                    "season_country_coefficient",
                    "country_uefa_overall_rank",
                    "country_uefa_season_rank",
                ]
            )

            coeff_feature_map = {
                "uefa_season_year": _safe_float(home_info.get("uefa_season_year", 0.0)),
                "home_team_uefa_overall_coefficient": _safe_float(home_info.get("overall_club_coefficient", 0.0)),
                "away_team_uefa_overall_coefficient": _safe_float(away_info.get("overall_club_coefficient", 0.0)),
                "home_team_uefa_season_coefficient": _safe_float(home_info.get("season_club_coefficient", 0.0)),
                "away_team_uefa_season_coefficient": _safe_float(away_info.get("season_club_coefficient", 0.0)),
                "home_team_uefa_overall_rank": _safe_float(home_info.get("overall_rank", 0.0)),
                "away_team_uefa_overall_rank": _safe_float(away_info.get("overall_rank", 0.0)),
                "home_team_uefa_season_rank": _safe_float(home_info.get("season_rank", 0.0)),
                "away_team_uefa_season_rank": _safe_float(away_info.get("season_rank", 0.0)),
                "home_overall_country_coefficient": _safe_float(home_info.get("overall_country_coefficient", 0.0)),
                "away_overall_country_coefficient": _safe_float(away_info.get("overall_country_coefficient", 0.0)),
                "home_season_country_coefficient": _safe_float(home_info.get("season_country_coefficient", 0.0)),
                "away_season_country_coefficient": _safe_float(away_info.get("season_country_coefficient", 0.0)),
                "home_country_uefa_overall_rank": _safe_float(home_info.get("country_uefa_overall_rank", 0.0)),
                "away_country_uefa_overall_rank": _safe_float(away_info.get("country_uefa_overall_rank", 0.0)),
                "home_country_uefa_season_rank": _safe_float(home_info.get("country_uefa_season_rank", 0.0)),
                "away_country_uefa_season_rank": _safe_float(away_info.get("country_uefa_season_rank", 0.0)),
                # Backward-compatible single-country fields.
                "overall_country_coefficient": _safe_float(home_info.get("overall_country_coefficient", 0.0)),
                "season_country_coefficient": _safe_float(home_info.get("season_country_coefficient", 0.0)),
                "country_uefa_overall_rank": _safe_float(home_info.get("country_uefa_overall_rank", 0.0)),
                "country_uefa_season_rank": _safe_float(home_info.get("country_uefa_season_rank", 0.0)),
                "home_uefa_missing": 0.0,
                "away_uefa_missing": 0.0,
                "country_uefa_missing": float(home_country_missing or away_country_missing),
            }
            coeff_feature_map["uefa_overall_coefficient_diff"] = (
                coeff_feature_map["home_team_uefa_overall_coefficient"]
                - coeff_feature_map["away_team_uefa_overall_coefficient"]
            )
            coeff_feature_map["uefa_season_coefficient_diff"] = (
                coeff_feature_map["home_team_uefa_season_coefficient"]
                - coeff_feature_map["away_team_uefa_season_coefficient"]
            )
            coeff_feature_map["uefa_overall_rank_diff"] = (
                coeff_feature_map["home_team_uefa_overall_rank"] - coeff_feature_map["away_team_uefa_overall_rank"]
            )
            coeff_feature_map["uefa_season_rank_diff"] = (
                coeff_feature_map["home_team_uefa_season_rank"] - coeff_feature_map["away_team_uefa_season_rank"]
            )
            coeff_feature_map["country_overall_coefficient_diff"] = (
                coeff_feature_map["home_overall_country_coefficient"]
                - coeff_feature_map["away_overall_country_coefficient"]
            )
            coeff_feature_map["country_season_coefficient_diff"] = (
                coeff_feature_map["home_season_country_coefficient"]
                - coeff_feature_map["away_season_country_coefficient"]
            )
            coeff_feature_map["country_overall_rank_diff"] = (
                coeff_feature_map["home_country_uefa_overall_rank"]
                - coeff_feature_map["away_country_uefa_overall_rank"]
            )
            coeff_feature_map["country_season_rank_diff"] = (
                coeff_feature_map["home_country_uefa_season_rank"]
                - coeff_feature_map["away_country_uefa_season_rank"]
            )

            for col, val in coeff_feature_map.items():
                if col in vector.columns:
                    vector[col] = val

    team_states, pair_states = _extract_states_from_history_if_needed(
        home_team,
        away_team,
        history_df,
        team_states,
        pair_states,
    )

    resolved_home_team = _resolve_team_name_from_state(home_team, team_states)
    resolved_away_team = _resolve_team_name_from_state(away_team, team_states)

    # Backward-compatible one-hot support if old feature templates are loaded.
    for team_name in {home_team, resolved_home_team, _canonical_club_name(home_team)}:
        home_col = f"home_team_{team_name}"
        if home_col in vector.columns:
            vector[home_col] = 1
    for team_name in {away_team, resolved_away_team, _canonical_club_name(away_team)}:
        away_col = f"away_team_{team_name}"
        if away_col in vector.columns:
            vector[away_col] = 1

    home_state = team_states.get(resolved_home_team, {})
    away_state = team_states.get(resolved_away_team, {})

    today = pd.Timestamp.utcnow().tz_localize(None)

    def _rest_days(state):
        last_date = state.get("last_match_date") if isinstance(state, dict) else None
        if last_date is None or pd.isna(last_date):
            return 7.0
        days = float((today - pd.to_datetime(last_date)).days)
        return float(max(0.0, min(days, 30.0)))

    feature_map = {
        "home_pre_elo": float(home_state.get("elo", 1500.0)),
        "away_pre_elo": float(away_state.get("elo", 1500.0)),
        "elo_diff": float(home_state.get("elo", 1500.0)) - float(away_state.get("elo", 1500.0)),
        "home_form_goals_for_5": float(home_state.get("form_goals_for_5", 0.0)),
        "home_form_goals_against_5": float(home_state.get("form_goals_against_5", 0.0)),
        "home_form_points_5": float(home_state.get("form_points_5", 0.0)),
        "home_form_goal_diff_5": float(home_state.get("form_goal_diff_5", 0.0)),
        "home_form_win_rate_5": float(home_state.get("form_win_rate_5", 0.0)),
        "home_form_draw_rate_5": float(home_state.get("form_draw_rate_5", 0.0)),
        "home_form_loss_rate_5": float(home_state.get("form_loss_rate_5", 0.0)),
        "home_form_clean_sheet_rate_5": float(home_state.get("form_clean_sheet_rate_5", 0.0)),
        "home_form_fail_to_score_rate_5": float(home_state.get("form_fail_to_score_rate_5", 0.0)),
        "home_form_btts_rate_5": float(home_state.get("form_btts_rate_5", 0.0)),
        "home_form_goals_for_10": float(home_state.get("form_goals_for_10", 0.0)),
        "home_form_goals_against_10": float(home_state.get("form_goals_against_10", 0.0)),
        "home_form_points_10": float(home_state.get("form_points_10", 0.0)),
        "home_form_goal_diff_10": float(home_state.get("form_goal_diff_10", 0.0)),
        "home_form_win_rate_10": float(home_state.get("form_win_rate_10", 0.0)),
        "home_form_draw_rate_10": float(home_state.get("form_draw_rate_10", 0.0)),
        "home_form_loss_rate_10": float(home_state.get("form_loss_rate_10", 0.0)),
        "home_form_clean_sheet_rate_10": float(home_state.get("form_clean_sheet_rate_10", 0.0)),
        "home_form_fail_to_score_rate_10": float(home_state.get("form_fail_to_score_rate_10", 0.0)),
        "home_form_btts_rate_10": float(home_state.get("form_btts_rate_10", 0.0)),
        "away_form_goals_for_5": float(away_state.get("form_goals_for_5", 0.0)),
        "away_form_goals_against_5": float(away_state.get("form_goals_against_5", 0.0)),
        "away_form_points_5": float(away_state.get("form_points_5", 0.0)),
        "away_form_goal_diff_5": float(away_state.get("form_goal_diff_5", 0.0)),
        "away_form_win_rate_5": float(away_state.get("form_win_rate_5", 0.0)),
        "away_form_draw_rate_5": float(away_state.get("form_draw_rate_5", 0.0)),
        "away_form_loss_rate_5": float(away_state.get("form_loss_rate_5", 0.0)),
        "away_form_clean_sheet_rate_5": float(away_state.get("form_clean_sheet_rate_5", 0.0)),
        "away_form_fail_to_score_rate_5": float(away_state.get("form_fail_to_score_rate_5", 0.0)),
        "away_form_btts_rate_5": float(away_state.get("form_btts_rate_5", 0.0)),
        "away_form_goals_for_10": float(away_state.get("form_goals_for_10", 0.0)),
        "away_form_goals_against_10": float(away_state.get("form_goals_against_10", 0.0)),
        "away_form_points_10": float(away_state.get("form_points_10", 0.0)),
        "away_form_goal_diff_10": float(away_state.get("form_goal_diff_10", 0.0)),
        "away_form_win_rate_10": float(away_state.get("form_win_rate_10", 0.0)),
        "away_form_draw_rate_10": float(away_state.get("form_draw_rate_10", 0.0)),
        "away_form_loss_rate_10": float(away_state.get("form_loss_rate_10", 0.0)),
        "away_form_clean_sheet_rate_10": float(away_state.get("form_clean_sheet_rate_10", 0.0)),
        "away_form_fail_to_score_rate_10": float(away_state.get("form_fail_to_score_rate_10", 0.0)),
        "away_form_btts_rate_10": float(away_state.get("form_btts_rate_10", 0.0)),
        "home_team_home_form_goals_for_5": float(home_state.get("home_only_form_goals_for_5", 0.0)),
        "home_team_home_form_goals_against_5": float(home_state.get("home_only_form_goals_against_5", 0.0)),
        "home_team_home_form_points_5": float(home_state.get("home_only_form_points_5", 0.0)),
        "home_team_home_form_goals_for_10": float(home_state.get("home_only_form_goals_for_10", 0.0)),
        "home_team_home_form_goals_against_10": float(home_state.get("home_only_form_goals_against_10", 0.0)),
        "home_team_home_form_points_10": float(home_state.get("home_only_form_points_10", 0.0)),
        "away_team_away_form_goals_for_5": float(away_state.get("away_only_form_goals_for_5", 0.0)),
        "away_team_away_form_goals_against_5": float(away_state.get("away_only_form_goals_against_5", 0.0)),
        "away_team_away_form_points_5": float(away_state.get("away_only_form_points_5", 0.0)),
        "away_team_away_form_goals_for_10": float(away_state.get("away_only_form_goals_for_10", 0.0)),
        "away_team_away_form_goals_against_10": float(away_state.get("away_only_form_goals_against_10", 0.0)),
        "away_team_away_form_points_10": float(away_state.get("away_only_form_points_10", 0.0)),
        "home_rest_days": _rest_days(home_state) if isinstance(home_state, dict) else 7.0,
        "away_rest_days": _rest_days(away_state) if isinstance(away_state, dict) else 7.0,
    }
    feature_map["rest_days_diff"] = feature_map["home_rest_days"] - feature_map["away_rest_days"]
    feature_map["venue_form_points_diff_5"] = (
        feature_map["home_team_home_form_points_5"] - feature_map["away_team_away_form_points_5"]
    )
    feature_map["venue_form_points_diff_10"] = (
        feature_map["home_team_home_form_points_10"] - feature_map["away_team_away_form_points_10"]
    )

    pair_key = tuple(sorted((resolved_home_team, resolved_away_team)))
    pair_info = pair_states.get(pair_key, {}) if isinstance(pair_states, dict) else {}
    h2h_count = int(pair_info.get("count", 0)) if isinstance(pair_info, dict) else 0
    h2h_sum = 0.0
    if isinstance(pair_info, dict):
        sums = pair_info.get("sum_by_team", {})
        if isinstance(sums, dict):
            h2h_sum = float(sums.get(resolved_home_team, 0.0))

    feature_map["h2h_matches_before"] = h2h_count
    feature_map["head_to_head_goal_diff"] = h2h_sum / h2h_count if h2h_count else 0.0
    feature_map["h2h_available"] = int(h2h_count > 0)

    # Legacy fallback for old engineered templates based on one-hot team columns.
    if history_df is not None and not history_df.empty:
        home_hist_col = f"home_team_{home_team}"
        away_hist_col = f"away_team_{away_team}"
        if home_hist_col in history_df.columns and away_hist_col in history_df.columns:
            h2h_mask = (history_df[home_hist_col] == 1) & (history_df[away_hist_col] == 1)
            if h2h_mask.any():
                if "head_to_head_goal_diff" in history_df.columns and "head_to_head_goal_diff" in vector.columns:
                    value = history_df.loc[h2h_mask, "head_to_head_goal_diff"].mean()
                    vector["head_to_head_goal_diff"] = 0.0 if pd.isna(value) else float(value)
                if "h2h_available" in vector.columns:
                    vector["h2h_available"] = 1

    for col, val in feature_map.items():
        if col in vector.columns:
            vector[col] = val

    non_zero_features = int((vector.iloc[0] != 0).sum()) if len(vector) > 0 else 0
    logger.info(
        "club_feature_vector_build_completed request_id=%s resolved_home_team=%s resolved_away_team=%s h2h_count=%s non_zero_features=%s elapsed_ms=%.2f",
        request_id,
        resolved_home_team,
        resolved_away_team,
        h2h_count,
        non_zero_features,
        (perf_counter() - feature_build_start) * 1000.0,
    )
    return vector


def build_feature_vector(
    home_team,
    away_team,
    feature_template_df,
    ranking_df=None,
    mode="national",
    history_df=None,
    competition=None,
    round_name=None,
    neutral=None,
    team_states=None,
    pair_states=None,
    request_id=None,
):
    if mode == "national":
        if ranking_df is None:
            raise ValueError("ranking_df is required for national mode")
        return _build_national_feature_vector(home_team, away_team, feature_template_df, ranking_df)
    if mode in {"club", "champions"}:
        return _build_club_feature_vector(
            home_team,
            away_team,
            feature_template_df,
            history_df,
            ranking_df=ranking_df,
            strict_coefficients=(mode == "champions"),
            competition=competition,
            round_name=round_name,
            neutral=neutral,
            team_states=team_states,
            pair_states=pair_states,
            request_id=request_id,
        )
    raise ValueError(f"Unsupported mode: {mode}")
