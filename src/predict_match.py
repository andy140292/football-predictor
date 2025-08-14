import pandas as pd
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from supabase import create_client, Client
from dotenv import load_dotenv
import io
from datetime import datetime
from datetime import datetime, timedelta
import jwt
from src.supabase_client import get_supabase_client
from src.utils.paths import MODEL_PATHS, PROCESSED_X_PATH, PROCESSED_X__FULL_PATH, PROCESSED_y_PATH, MATCHES_PATH,RANKING_PATH

supabase = get_supabase_client()
API_ENV = os.getenv("API_ENV", "prod")

month_str = (datetime.today() - timedelta(days=30)).strftime("%Y_%m")
bucket = "model-artifacts"

def load_file_from_supabase(bucket: str, path: str, as_dataframe=False):
    response = supabase.storage.from_(bucket).download(path)
    if as_dataframe:
        return pd.read_csv(io.BytesIO(response))
    else:
        return joblib.load(io.BytesIO(response))
    
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
    # if email == "andresuribe1402@gmail.com":
    #     return True  # No limit for this email
    
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

# Load feature columns
X = pd.read_csv(PROCESSED_X_PATH)
fifa_rank = pd.read_csv(RANKING_PATH)
feature_columns = X.columns

rf_predictor = joblib.load(MODEL_PATHS["random_forest"])
lr_predictor = joblib.load(MODEL_PATHS["logistic_regression"])
mlp_predictor = joblib.load(MODEL_PATHS["mlp"])

def predict_outcome(home_team, away_team, token=None):
    email = extract_email_from_token(token)
    if not email:
        raise ValueError("No se pudo extraer el email del token")

    if not get_remaining_predictions(email):
        raise ValueError("Límite diario de predicciones alcanzado (15)")

    register_prediction(email, home_team, away_team)
    
    match_vector = build_feature_vector(home_team, away_team, X, fifa_rank)
    results = {}

    if rf_predictor:
        rf_probs = rf_predictor.predict_proba(match_vector)[0]
        results["random_forest"] = {
            "home_win": rf_probs[2],
            "draw": rf_probs[1],
            "away_win": rf_probs[0]
        }

    if lr_predictor:
        lr_probs = lr_predictor.predict_proba(match_vector)[0]
        results["logistic_regression"] = {
            "home_win": lr_probs[2],
            "draw": lr_probs[1],
            "away_win": lr_probs[0]
        }

    if mlp_predictor:
        mlp_probs = mlp_predictor.predict_proba(match_vector)[0]
        results["mlp"] = {
            "home_win": mlp_probs[2],
            "draw": mlp_probs[1],
            "away_win": mlp_probs[0]
        }

    return results

def build_feature_vector(home_team, away_team, feature_template_df, ranking_df):
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


