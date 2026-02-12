import pandas as pd
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datetime import datetime
import jwt
from src.supabase_client import get_supabase_client
from src.utils.paths import (
    CLUB_COEFFICIENTS_PATH,
    CLUB_MODEL_PATHS,
    CLUB_PROCESSED_X_FULL_PATH,
    CLUB_PROCESSED_X_PATH,
    MODEL_PATHS,
    PROCESSED_X_PATH,
    RANKING_PATH,
)

supabase = get_supabase_client()
API_ENV = os.getenv("API_ENV", "prod")

_assets = None

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


_club_assets = None


def _load_club_assets():
    global _club_assets
    if _club_assets is None:
        X = pd.read_csv(CLUB_PROCESSED_X_PATH)
        X_full = pd.read_csv(CLUB_PROCESSED_X_FULL_PATH)
        club_coefficients = pd.read_csv(CLUB_COEFFICIENTS_PATH)
        rf_predictor = joblib.load(CLUB_MODEL_PATHS["random_forest"])
        lr_predictor = joblib.load(CLUB_MODEL_PATHS["logistic_regression"])
        mlp_predictor = joblib.load(CLUB_MODEL_PATHS["mlp"])
        _club_assets = {
            "X": X,
            "X_full": X_full,
            "club_coefficients": club_coefficients,
            "models": {
                "random_forest": rf_predictor,
                "logistic_regression": lr_predictor,
                "mlp": mlp_predictor,
            },
        }
    return _club_assets

def extract_email_from_token(token: str) -> str:
    if API_ENV == "dev":
        # ✅ Email ficticio para pruebas
        return "dev@example.com"

    if not token:
        return None

    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload.get("email")
    except Exception as e:
        print(f"❌ Error al decodificar token: {e}")
        return None

def get_remaining_predictions(email: str, limit=15):
    if email == "andresuribe1402@gmail.com":
        return True  # No limit for this email
    
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    result = supabase.table("user_predictions").select("*")\
        .eq("email", email)\
        .gte("timestamp", today_start).execute()

    used = len(result.data)
    return max(limit - used, 0)

def register_prediction(email: str, home_team: str, away_team: str):
    supabase.table("user_predictions").insert({
        "email": email,
        "home_team": home_team,
        "away_team": away_team,
        "timestamp": datetime.utcnow().isoformat()
    }).execute()

def predict_outcome(home_team, away_team, token=None, mode="national"):
    email = extract_email_from_token(token)
    if not email:
        raise ValueError("No se pudo extraer el email del token")

    if not get_remaining_predictions(email):
        raise ValueError("Límite diario de predicciones alcanzado (15)")

    register_prediction(email, home_team, away_team)

    if mode not in {"national", "club"}:
        raise ValueError(f"Modo inválido: {mode}")

    assets = _load_assets() if mode == "national" else _load_club_assets()
    X = assets["X"]
    models = assets["models"]
    ranking_data = assets.get("fifa_rank") if mode == "national" else assets.get("club_coefficients")
    match_vector = build_feature_vector(
        home_team,
        away_team,
        X,
        ranking_data,
        mode=mode,
        history_df=assets.get("X_full"),
    )
    results = {}

    rf_predictor = models.get("random_forest")
    if rf_predictor:
        rf_probs = rf_predictor.predict_proba(match_vector)[0]
        results["random_forest"] = {
            "home_win": rf_probs[2],
            "draw": rf_probs[1],
            "away_win": rf_probs[0]
        }

    lr_predictor = models.get("logistic_regression")
    if lr_predictor:
        lr_probs = lr_predictor.predict_proba(match_vector)[0]
        results["logistic_regression"] = {
            "home_win": lr_probs[2],
            "draw": lr_probs[1],
            "away_win": lr_probs[0]
        }

    mlp_predictor = models.get("mlp")
    if mlp_predictor:
        mlp_probs = mlp_predictor.predict_proba(match_vector)[0]
        results["mlp"] = {
            "home_win": mlp_probs[2],
            "draw": mlp_probs[1],
            "away_win": mlp_probs[0]
        }

    return results

def _build_national_feature_vector(home_team, away_team, feature_template_df, ranking_df):
    import numpy as np
    import pandas as pd

    vector = pd.DataFrame([np.zeros(len(feature_template_df.columns))], columns=feature_template_df.columns)

    home_row = ranking_df[ranking_df["team"] == home_team]
    away_row = ranking_df[ranking_df["team"] == away_team]
    if home_row.empty or away_row.empty:
        raise ValueError(f"Team not found in ranking: {home_team} or {away_team}")
    home_info = home_row.iloc[0]
    away_info = away_row.iloc[0]

    # Optional individual ranking features
    if "home_team_fifa_rank" in vector.columns:
        vector["home_team_fifa_rank"] = home_info["ranking"]
    if "away_team_fifa_rank" in vector.columns:
        vector["away_team_fifa_rank"] = away_info["ranking"]
    if "home_team_fifa_points" in vector.columns:
        vector["home_team_fifa_points"] = home_info["points"]
    if "away_team_fifa_points" in vector.columns:
        vector["away_team_fifa_points"] = away_info["points"]

    # Aggregated features
    vector["fifa_rank_diff"] = home_info["ranking"] - away_info["ranking"]
    vector["fifa_points_diff"] = home_info["points"] - away_info["points"]

    # Team one-hot encoding
    home_col = f"home_team_{home_team}"
    away_col = f"away_team_{away_team}"
    if home_col in vector.columns:
        vector[home_col] = 1
    if away_col in vector.columns:
        vector[away_col] = 1

    # Confederation-based features
    conf_pair_col = f"confed_pair_{home_info['confederation']}_vs_{away_info['confederation']}"
    if conf_pair_col in vector.columns:
        vector[conf_pair_col] = 1

    home_conf = f"home_team_confederation_{home_info['confederation']}"
    away_conf = f"away_team_confederation_{away_info['confederation']}"
    if home_conf in vector.columns:
        vector[home_conf] = 1
    if away_conf in vector.columns:
        vector[away_conf] = 1

    # Neutral field logic
    if "neutral" in vector.columns:
        vector["neutral"] = int(home_info["confederation"] != away_info["confederation"])

    return vector


def _normalized_text(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("á", "a")
        .replace("à", "a")
        .replace("ä", "a")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ë", "e")
        .replace("í", "i")
        .replace("ì", "i")
        .replace("ï", "i")
        .replace("ó", "o")
        .replace("ò", "o")
        .replace("ö", "o")
        .replace("ú", "u")
        .replace("ù", "u")
        .replace("ü", "u")
        .replace("ğ", "g")
        .replace("ø", "o")
    )


def _build_club_feature_vector(home_team, away_team, feature_template_df, history_df, ranking_df=None):
    import numpy as np
    import pandas as pd

    vector = pd.DataFrame([np.zeros(len(feature_template_df.columns))], columns=feature_template_df.columns)

    # Team one-hot encoding
    home_col = f"home_team_{home_team}"
    away_col = f"away_team_{away_team}"
    if home_col in vector.columns:
        vector[home_col] = 1
    if away_col in vector.columns:
        vector[away_col] = 1

    # Default context for requested UCL knockout predictions.
    competition_col = "competition_Champions Lg"
    round_col = "round_Knockout phase play-offs"
    if competition_col in vector.columns:
        vector[competition_col] = 1
    if round_col in vector.columns:
        vector[round_col] = 1
    if "is_ucl_match" in vector.columns:
        vector["is_ucl_match"] = 1
    if "is_knockout_playoff" in vector.columns:
        vector["is_knockout_playoff"] = 1
    if "neutral" in vector.columns:
        vector["neutral"] = 0

    if ranking_df is not None and not ranking_df.empty:
        ranking = ranking_df.copy()
        ranking["team_norm"] = ranking["team"].map(_normalized_text)

        home_norm = _normalized_text(home_team)
        away_norm = _normalized_text(away_team)
        home_rank = ranking[ranking["team_norm"] == home_norm]
        away_rank = ranking[ranking["team_norm"] == away_norm]

        if home_rank.empty or away_rank.empty:
            raise ValueError(f"Team not found in UEFA coefficients: {home_team} or {away_team}")

        home_info = home_rank.iloc[0]
        away_info = away_rank.iloc[0]

        coeff_feature_map = {
            "home_team_uefa_overall_coefficient": float(home_info["overall_club_coefficient"]),
            "away_team_uefa_overall_coefficient": float(away_info["overall_club_coefficient"]),
            "home_team_uefa_season_coefficient": float(home_info["season_club_coefficient"]),
            "away_team_uefa_season_coefficient": float(away_info["season_club_coefficient"]),
            "home_team_uefa_overall_rank": float(home_info["overall_rank"]),
            "away_team_uefa_overall_rank": float(away_info["overall_rank"]),
            "home_team_uefa_season_rank": float(home_info["season_rank"]),
            "away_team_uefa_season_rank": float(away_info["season_rank"]),
            "uefa_overall_coefficient_diff": float(home_info["overall_club_coefficient"]) - float(away_info["overall_club_coefficient"]),
            "uefa_season_coefficient_diff": float(home_info["season_club_coefficient"]) - float(away_info["season_club_coefficient"]),
            "uefa_overall_rank_diff": float(home_info["overall_rank"]) - float(away_info["overall_rank"]),
            "uefa_season_rank_diff": float(home_info["season_rank"]) - float(away_info["season_rank"]),
        }
        for col, val in coeff_feature_map.items():
            if col in vector.columns:
                vector[col] = val

    if history_df is None or history_df.empty:
        if "h2h_available" in vector.columns:
            vector["h2h_available"] = 0
        return vector

    # Pull latest form proxies from historical engineered rows.
    home_hist_col = f"home_team_{home_team}"
    away_hist_col = f"away_team_{away_team}"
    reverse_home_hist_col = f"home_team_{away_team}"
    reverse_away_hist_col = f"away_team_{home_team}"

    same_pair = pd.Series([False] * len(history_df))
    rev_pair = pd.Series([False] * len(history_df))
    if home_hist_col in history_df.columns and away_hist_col in history_df.columns:
        same_pair = (history_df[home_hist_col] == 1) & (history_df[away_hist_col] == 1)
    if reverse_home_hist_col in history_df.columns and reverse_away_hist_col in history_df.columns:
        rev_pair = (history_df[reverse_home_hist_col] == 1) & (history_df[reverse_away_hist_col] == 1)
    h2h_mask = same_pair | rev_pair

    if "head_to_head_goal_diff" in history_df.columns and "head_to_head_goal_diff" in vector.columns:
        h2h_value = history_df.loc[h2h_mask, "head_to_head_goal_diff"].mean()
        vector["head_to_head_goal_diff"] = 0 if pd.isna(h2h_value) else float(h2h_value)
    if "h2h_available" in vector.columns:
        vector["h2h_available"] = int(h2h_mask.any())

    def _latest_value(mask, col):
        if col not in history_df.columns or not mask.any():
            return 0.0
        series = history_df.loc[mask, col]
        if series.empty:
            return 0.0
        value = series.iloc[-1]
        return 0.0 if pd.isna(value) else float(value)

    home_mask = history_df[home_hist_col] == 1 if home_hist_col in history_df.columns else pd.Series([False] * len(history_df))
    away_mask = history_df[away_hist_col] == 1 if away_hist_col in history_df.columns else pd.Series([False] * len(history_df))

    for col, mask in [
        ("home_team_avg_scored", home_mask),
        ("home_team_avg_conceded", home_mask),
        ("away_team_avg_scored", away_mask),
        ("away_team_avg_conceded", away_mask),
    ]:
        if col in vector.columns:
            vector[col] = _latest_value(mask, col)

    return vector


def build_feature_vector(home_team, away_team, feature_template_df, ranking_df=None, mode="national", history_df=None):
    if mode == "national":
        if ranking_df is None:
            raise ValueError("ranking_df is required for national mode")
        return _build_national_feature_vector(home_team, away_team, feature_template_df, ranking_df)
    if mode == "club":
        return _build_club_feature_vector(home_team, away_team, feature_template_df, history_df, ranking_df=ranking_df)
    raise ValueError(f"Unsupported mode: {mode}")
