import pandas as pd
from sklearn.model_selection import train_test_split
from src.prediction.match_data_preprocessor import MatchDataPreprocessor # Import the MatchDataPreprocessor class
from src.prediction.football_match_predictor import FootballMatchPredictor # Import the FootballMatchPredictor class

# ✅ List of South American teams
south_america_teams = ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela']

# ✅ Initialize and run preprocessor
preprocessor = MatchDataPreprocessor(file_path='data/matches.csv', south_america_teams=south_america_teams)
X, y = preprocessor.preprocess()

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Data preprocessing complete! Ready for model training.")

# ✅ Train & Evaluate a Football Predictor Model for random forest
rf_predictor = FootballMatchPredictor("random_forest")
rf_predictor.train(X_train, y_train,X)
rf_predictor.evaluate(X_test, y_test)
# rf_predictor.feature_importance(X_train)

# ✅ Train & Evaluate a Football Predictor Model for random forest
lr_predictor = FootballMatchPredictor("logistic_regression")
lr_predictor.train(X_train, y_train,X)
lr_predictor.evaluate(X_test, y_test)
# rf_predictor.feature_importance(X_train)

# Predicting the outcome of a match
# List of feature columns (should match X.columns)
feature_columns = X.columns

# Create a new data instance with default 0 values for all features
new_match_data = {col: 0 for col in feature_columns}

# Fill in the rolling averages for Ecuador (home) and Venezuela (away)
# Replace these with your actual computed values
home_team = input("Equipo local: ")
away_team = input("Equipo visitante: ")

# Set the appropriate dummy variables
home_team_dummy = 'home_team_' + home_team
away_team_dummy = 'away_team_' + away_team
new_match_data[home_team_dummy] = 1   # Ecuador is playing at home
new_match_data[away_team_dummy] = 1   # Venezuela is playing away

# ✅ Predict probability for a new match
new_match_features = pd.DataFrame([new_match_data])  # Ensure this matches X_train's structure
rf_probabilities = rf_predictor.predict_match(new_match_features)
lr_probabilities = lr_predictor.predict_match(new_match_features)

print("📊 Random Forest Match Prediction Probabilities:", rf_probabilities)
print("📊 Logistic Regression Match Prediction Probabilities:", lr_probabilities)
