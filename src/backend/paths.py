import os
from pathlib import Path

# Si no hay APP_BASE_DIR en env, cae al root del repo (asumiendo este archivo está en src/...)
DEFAULT_BASE = Path(__file__).resolve().parents[2]

BASE_DIR = Path(os.getenv("APP_BASE_DIR", DEFAULT_BASE))


def _resolve_existing_dir(env_var: str, candidates: list[Path]) -> Path:
    env_value = os.getenv(env_var)
    if env_value:
        return Path(env_value)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_existing_file(filename: str, candidates: list[Path]) -> Path:
    for directory in candidates:
        path = directory / filename
        if path.exists():
            return path
    return candidates[0] / filename


DATA_DIR = _resolve_existing_dir(
    "DATA_DIR",
    [
        BASE_DIR / "data",
        BASE_DIR / "src" / "data",
    ],
)
MODELS_DIR = _resolve_existing_dir(
    "MODELS_DIR",
    [
        BASE_DIR / "models",
        BASE_DIR / "src" / "backend" / "models",
    ],
)
HISTORICAL_DATA_DIR = _resolve_existing_dir(
    "HISTORICAL_DATA_DIR",
    [
        BASE_DIR / "preprocessing_data",
        BASE_DIR / "preprocessing_Dta",
        DATA_DIR,
    ],
)

PROCESSED_X_PATH = DATA_DIR / "processed_X.csv"
PROCESSED_X__FULL_PATH = DATA_DIR / "processed_X_Full.csv"
PROCESSED_y_PATH = DATA_DIR / "processed_y.csv"
RANKING_PATH = DATA_DIR / "ranking_fifa_2025.csv"

CLUB_MATCHES_PATH = DATA_DIR / "ucl_ko_teams_scores_fixtures_2025_2026.csv"
CLUB_COEFFICIENTS_PATH = DATA_DIR / "ucl_ko_uefa_coefficients_2026.csv"
CLUB_COEFFICIENTS_HISTORY_PATH = DATA_DIR / "uefa_club_coefficients_2002_2026.csv"
COUNTRY_COEFFICIENTS_HISTORY_PATH = DATA_DIR / "uefa_country_coefficients_2002_2026.csv"
CLUB_LEAGUES_HISTORY_PATH = _resolve_existing_file(
    "club_leagues_historical_1993_2026.csv",
    [HISTORICAL_DATA_DIR, DATA_DIR],
)
CLUB_UCL_HISTORY_PATH = _resolve_existing_file(
    "ucl_scores_fixtures_historical.csv",
    [HISTORICAL_DATA_DIR, DATA_DIR],
)
CLUB_MATCHES_HISTORY_PATH = _resolve_existing_file(
    "club_matches_historical_1993_2026.csv",
    [HISTORICAL_DATA_DIR, DATA_DIR],
)
LIBERTADORES_MATCHES_HISTORY_PATH = _resolve_existing_file(
    "club_matches_historical_libertadores_2025_2026.csv",
    [HISTORICAL_DATA_DIR, DATA_DIR],
)
CLUB_TEAM_ALIASES_PATH = _resolve_existing_file(
    "club_team_aliases.csv",
    [DATA_DIR, HISTORICAL_DATA_DIR],
)
LIBERTADORES_COEFFICIENTS_PATH = _resolve_existing_file(
    "libertadores_conmebol_coefficients_2026.csv",
    [HISTORICAL_DATA_DIR, DATA_DIR],
)
CLUB_PROCESSED_X_PATH = DATA_DIR / "processed_X_club.csv"
CLUB_PROCESSED_X_FULL_PATH = DATA_DIR / "processed_X_Full_club.csv"
CLUB_PROCESSED_Y_PATH = DATA_DIR / "processed_y_club.csv"
CHAMPIONS_PROCESSED_X_PATH = DATA_DIR / "processed_X_champions.csv"
CHAMPIONS_PROCESSED_X_FULL_PATH = DATA_DIR / "processed_X_Full_champions.csv"
CHAMPIONS_PROCESSED_Y_PATH = DATA_DIR / "processed_y_champions.csv"
LIBERTADORES_PROCESSED_X_PATH = DATA_DIR / "processed_X_libertadores.csv"
LIBERTADORES_PROCESSED_X_FULL_PATH = DATA_DIR / "processed_X_Full_libertadores.csv"
LIBERTADORES_PROCESSED_Y_PATH = DATA_DIR / "processed_y_libertadores.csv"

MODEL_PATHS = {
    "logistic_regression": MODELS_DIR / "logistic_regression_predictor.pkl",
    "random_forest": MODELS_DIR / "random_forest_predictor.pkl",
    "mlp": MODELS_DIR / "mlp_predictor.pkl",
}

CLUB_MODEL_PATHS = {
    "logistic_regression": MODELS_DIR / "club_logistic_regression_predictor.pkl",
    "random_forest": MODELS_DIR / "club_random_forest_predictor.pkl",
    "mlp": MODELS_DIR / "club_mlp_predictor.pkl",
}

CHAMPIONS_MODEL_PATHS = {
    "logistic_regression": MODELS_DIR / "champions_logistic_regression_predictor.pkl",
    "random_forest": MODELS_DIR / "champions_random_forest_predictor.pkl",
    "mlp": MODELS_DIR / "champions_mlp_predictor.pkl",
}
LIBERTADORES_MODEL_PATHS = {
    "logistic_regression": MODELS_DIR / "libertadores_logistic_regression_predictor.pkl",
    "random_forest": MODELS_DIR / "libertadores_random_forest_predictor.pkl",
    "mlp": MODELS_DIR / "libertadores_mlp_predictor.pkl",
}

CLUB_STATE_SNAPSHOT_PATH = MODELS_DIR / "club_state_snapshot.pkl"
CHAMPIONS_STATE_SNAPSHOT_PATH = MODELS_DIR / "champions_state_snapshot.pkl"
LIBERTADORES_STATE_SNAPSHOT_PATH = MODELS_DIR / "libertadores_state_snapshot.pkl"
