from pathlib import Path
import os
from datetime import datetime
from datetime import timedelta

# Si no hay APP_BASE_DIR en env, cae al root del repo (asumiendo este archivo está en src/...)
DEFAULT_BASE = Path(__file__).resolve().parents[2]

BASE_DIR = Path(os.getenv("APP_BASE_DIR", DEFAULT_BASE))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))

month_str = (datetime.today() - timedelta(days=30)).strftime("%Y_%m")
MATCHES_PATH = DATA_DIR / f"matches_{month_str}.csv"
PROCESSED_X_PATH = DATA_DIR / "processed_X.csv"
PROCESSED_X__FULL_PATH = DATA_DIR / "processed_X_Full.csv"
PROCESSED_y_PATH = DATA_DIR / "processed_y.csv"
RANKING_PATH = DATA_DIR / "ranking_fifa_2025.csv"

CLUB_MATCHES_PATH = DATA_DIR / "ucl_ko_teams_scores_fixtures_2025_2026.csv"
CLUB_COEFFICIENTS_PATH = DATA_DIR / "ucl_ko_uefa_coefficients_2026.csv"
CLUB_PROCESSED_X_PATH = DATA_DIR / "processed_X_club.csv"
CLUB_PROCESSED_X_FULL_PATH = DATA_DIR / "processed_X_Full_club.csv"
CLUB_PROCESSED_Y_PATH = DATA_DIR / "processed_y_club.csv"

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
