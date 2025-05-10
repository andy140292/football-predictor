import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from src.prediction.match_data_preprocessor import MatchDataPreprocessor
from src.prediction.football_match_predictor import FootballMatchPredictor

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

# Load and preprocess data
south_america_teams = ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela']
preprocessor = MatchDataPreprocessor(file_path='data/matches.csv', south_america_teams=south_america_teams)
X, y = preprocessor.preprocess()

# Save processed data
X.to_csv("data/processed_X.csv", index=False)
y.to_csv("data/processed_y.csv", index=False)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save models
for model_type in ["random_forest", "logistic_regression"]:
    print(f"\n🔁 Training {model_type}...")
    predictor = FootballMatchPredictor(model_type)
    predictor.train(X_train, y_train, X)
    predictor.evaluate(X_test, y_test)

    if model_type in ["random_forest", "logistic_regression"]:
        predictor.feature_importance(X_train)  # Plot and analyze feature importance

    joblib.dump(predictor.model, f"models/{model_type}_model.pkl")

