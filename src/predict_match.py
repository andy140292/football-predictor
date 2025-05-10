import pandas as pd
import joblib
from src.prediction.football_match_predictor import FootballMatchPredictor

# Load processed data and models
X = pd.read_csv("data/processed_X.csv")
feature_columns = X.columns

rf_model = joblib.load("models/random_forest_model.pkl")
lr_model = joblib.load("models/logistic_regression_model.pkl")

# Build prediction input
home_team = input("Equipo local: ")
away_team = input("Equipo visitante: ")

new_match_data = {col: 0 for col in feature_columns}
home_dummy = 'home_team_' + home_team
away_dummy = 'away_team_' + away_team

if home_dummy in new_match_data:
    new_match_data[home_dummy] = 1
if away_dummy in new_match_data:
    new_match_data[away_dummy] = 1

new_match_df = pd.DataFrame([new_match_data])

# Predict
def predict(model, name):
    probs = model.predict_proba(new_match_df)[0]
    print(f"\n📊 {name} Prediction:")
    print(f"  Home Win:  {probs[2]:.2%}")
    print(f"  Draw:      {probs[1]:.2%}")
    print(f"  Away Win:  {probs[0]:.2%}")

predict(rf_model, "Random Forest")
predict(lr_model, "Logistic Regression")
