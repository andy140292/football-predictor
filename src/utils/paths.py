from pathlib import Path
import os

# Si no hay APP_BASE_DIR en env, cae al root del repo (asumiendo este archivo está en src/...)
DEFAULT_BASE = Path(__file__).resolve().parents[2]

BASE_DIR = Path(os.getenv("APP_BASE_DIR", DEFAULT_BASE))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))

MATCHES_PATH = DATA_DIR / "matches.csv"
PROCESSED_X_PATH = DATA_DIR / "processed_X.csv"
PROCESSED_X__FULL_PATH = DATA_DIR / "processed_X_Full.csv"
PROCESSED_y_PATH = DATA_DIR / "processed_y.csv"
RANKING_PATH = DATA_DIR / "ranking_fifa_2025.csv"

MODEL_PATHS = {
    "logistic_regression": MODELS_DIR / "logistic_regression_predictor.pkl",
    "random_forest": MODELS_DIR / "random_forest_predictor.pkl",
    "mlp": MODELS_DIR / "mlp_predictor.pkl",
}