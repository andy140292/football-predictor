import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

def get_home_average_scored(matches, team_name):
    # Filter matches for the home team
    home = matches[matches['home_team'] == team_name]
    home_team_avg_scored = 0.0
    home_team_avg_conceded = 0.0

    # Sort by date (if not already sorted) and get the most recent match
    home = home.sort_values(by='date')
    if not home.empty:
        last_home = home.iloc[-1]
        home_team_avg_scored = last_home['home_team_avg_scored']
        home_team_avg_conceded = last_home['home_team_avg_conceded']

    return home_team_avg_scored, home_team_avg_conceded

def get_away_average_scored(matches, team_name):
    # Filter matches for the away team
    away = matches[matches['away_team'] == team_name]
    away_team_avg_scored = 0.0
    away_team_avg_conceded = 0.0

    # Sort by date and get the most recent match
    away = away.sort_values(by='date')
    if not away.empty:
        last_away = away.iloc[-1]
        away_team_avg_scored = last_away['away_team_avg_scored']
        away_team_avg_conceded = last_away['away_team_avg_conceded']

    return away_team_avg_scored, away_team_avg_conceded

# def get_head_to_head_avg_goal_diff(matches, home_team, away_team):
#     # Filter matches where Venezuela is the away team
#     head_to_head_avg_goal_diff = matches[matches['away_team'] == away_team & matches['home_team'] == home_team]
#     result_head_to_head = 0.0


#     # Sort by date and get the most recent match
#     head_to_head_avg_goal_diff = head_to_head_avg_goal_diff.sort_values(by='date')
#     if not head_to_head_avg_goal_diff.empty:
#         last_head_to_head = head_to_head_avg_goal_diff.iloc[-1]
#         result_head_to_head = last_head_to_head['head_to_head_goal_diff']

#     return result_head_to_head

def get_head_to_head_stats(matches):
    # Create a matchup identifier by sorting the team names alphabetically
    matches['matchup_id'] = matches.apply(
        lambda row: '_'.join(sorted([row['home_team'], row['away_team']])), axis=1
    )

    # Calculate goal difference for each match (from the perspective of the listed home team)
    matches['goal_diff'] = matches['home_score'] - matches['away_score']

    # Group by matchup_id and compute the average goal difference
    head_to_head_stats = matches.groupby('matchup_id')['goal_diff'].mean().reset_index()
    head_to_head_stats.rename(
        columns={'goal_diff': 'head_to_head_goal_diff'}, inplace=True
    )
    return head_to_head_stats

# Define the new multi-class target variable
def get_match_outcome(row):
    if row['home_score'] > row['away_score']:
        return 2  # Home Win
    elif row['home_score'] < row['away_score']:
        return 0  # Away Win
    else:
        return 1  # Draw

def get_feature_importance_rf(model):
    # Get the best model from RandomizedSearchCV
    rf_best_model = model.best_estimator_  # Ensure we use the trained model 
    # Get feature importance scores
    feature_importances = rf_best_model.feature_importances_

    # Create DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.show()


south_america = ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela']

matches = pd.read_csv('./results.csv')

# Filter South American teams for the world cup qualification matches
matches_south_america = matches[matches['tournament'] == 'FIFA World Cup qualification' or matches['tournament'] == 'Copa America' or matches['tournament'] == 'FIFA World Cup' or matches['tournament'] == 'Friendly']
# matches_south_america = matches_south_america[matches_south_america['home_team'].isin(south_america)]
# matches_south_america = matches_south_america[matches_south_america['away_team'].isin(south_america)]

# Ensure 'date' column is in datetime format
matches_south_america['date'] = pd.to_datetime(matches_south_america['date'])

# Filter the DataFrame to only include matches after 2000-01-01
matches_south_america = matches_south_america[matches_south_america['date'] >= '2000-01-01']

# # Create a binary outcome for a home win: 1 if home wins, else 0.
# matches_south_america['home_win'] = (matches_south_america['home_score'] > matches_south_america['away_score']).astype(int)

# Create the matchup identifier in your training DataFrame
matches_south_america['matchup_id'] = matches_south_america.apply(
    lambda row: '_'.join(sorted([row['home_team'], row['away_team']])), axis=1
)

head_to_head_stats = get_head_to_head_stats(matches_south_america)
print(head_to_head_stats)

# Merge the head-to-head stats into your training DataFrame
matches_south_america = pd.merge(matches_south_america, head_to_head_stats, on='matchup_id', how='left')

# For matches with no head-to-head history, fill missing values (e.g., with 0 or another default)
matches_south_america['head_to_head_goal_diff'] = matches_south_america['head_to_head_goal_diff'].fillna(0)

# Rolling window size (e.g., the last 5 matches)
window_size = 10

# For the home team: Calculate rolling average of goals scored and conceded in home matches.
matches_south_america['home_team_avg_scored'] = matches_south_america.groupby('home_team')['home_score'] \
    .transform(lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())
matches_south_america['home_team_avg_conceded'] = matches_south_america.groupby('home_team')['away_score'] \
    .transform(lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())

# For the away team: Calculate rolling average of goals scored and conceded in away matches.
matches_south_america['away_team_avg_scored'] = matches_south_america.groupby('away_team')['away_score'] \
    .transform(lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())
matches_south_america['away_team_avg_conceded'] = matches_south_america.groupby('away_team')['home_score'] \
    .transform(lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())

# Create dummy variables for team names.
matches_south_america_no_dummies = matches_south_america.copy()

# Create dummy variables for team names.
matches_south_america = pd.get_dummies(matches_south_america, columns=['home_team', 'away_team'])

# Include rolling average of goals scored and conceded for each team.
matches_south_america = matches_south_america.sort_values('date')

# drop columns wih NaN values
matches_south_america = matches_south_america.dropna() 

# fixing indices to start at 0
matches_south_america = matches_south_america.reset_index(drop=True)

# Drop columns that are not used as predictors (like raw goals and match_date).
X = matches_south_america.drop(columns=['date', 'home_score', 'away_score', 'tournament','neutral','city','country','matchup_id','goal_diff'])
y = matches_south_america.apply(get_match_outcome, axis=1)

# print(list(X.columns))

# print("X shape:", X.shape)
# print("y shape:", y.shape)

# Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_test shape:", y_test.shape)

def get_logistic_regression_model(X_train, y_train, X_test, y_test):
    # # Scale the data
    scaler = StandardScaler()

    # # Fit the scaler on the training data and transform both training and testing data
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # Initialize the model; use scaled data if you performed scaling
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)  # Train logistic regression

    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
    calibrated_model.fit(X_train_scaled, y_train)

    # Predict probabilities
    y_pred_proba = calibrated_model.predict_proba(X_test_scaled)

    # Evaluate the model
    logloss_lr = log_loss(y_test, y_pred_proba, labels=[0, 1, 2])
    accuracy_lr = accuracy_score(y_test, model.predict(X_test))

    return (calibrated_model, logloss_lr, accuracy_lr)

def get_random_forest_model(X_train, y_train, X_test, y_test):
    # Initialize and train Random Forest classifier
    # Define parameter grid
    rf_param_grid = {
        'n_estimators': [100, 200, 500],  # Number of trees
        'max_depth': [10, 20, None],  # Tree depth
        'min_samples_split': [2, 5, 10],  # Min samples to split a node
        'min_samples_leaf': [1, 2, 4]  # Min samples at leaf node
    }

    # Initialize Random Forest model
    rf_model = RandomForestClassifier(random_state=42)

    # Perform Randomized Search
    rf_random_search = RandomizedSearchCV(
        estimator=rf_model, param_distributions=rf_param_grid,
        n_iter=10, scoring='neg_log_loss', cv=5, verbose=2, n_jobs=-1, random_state=42
    )

    rf_random_search.fit(X_train, y_train)

    # Best parameters
    # print("Best RF Parameters:", rf_random_search.best_params_)

    # Evaluate best model
    rf_best_model = rf_random_search.best_estimator_
    y_pred_proba_rf = rf_best_model.predict_proba(X_test)
    logloss_rf = log_loss(y_test, y_pred_proba_rf)
    accuracy_rf = accuracy_score(y_test, rf_best_model.predict(X_test))

    return (rf_random_search,logloss_rf,accuracy_rf)

def get_xgb_model(X_train, y_train, X_test, y_test):
    xgb_model = XGBClassifier(n_estimators=200, objective="multi:softprob", eval_metric="mlogloss")
    xgb_model.fit(X_train, y_train)

    # Predict probabilities
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)

    # Evaluate performance
    logloss_xgb = log_loss(y_test, y_pred_proba_xgb)
    accuracy_xgb = accuracy_score(y_test, xgb_model.predict(X_test))

    return (xgb_model,logloss_xgb,accuracy_xgb)

def get_cat_model(X_train, y_train, X_test, y_test):
    # Initialize and train CatBoost model
    cat_model = CatBoostClassifier(iterations=200, loss_function='MultiClass', verbose=0)
    cat_model.fit(X_train, y_train)

    # Predict probabilities
    y_pred_proba_cat = cat_model.predict_proba(X_test)

    # Evaluate performance
    logloss_cat = log_loss(y_test, y_pred_proba_cat)
    accuracy_cat = accuracy_score(y_test, cat_model.predict(X_test))

    return (cat_model,logloss_cat,accuracy_cat)


# Predicting the outcome of a match
# List of feature columns (should match X.columns)
feature_columns = X.columns

# Create a new data instance with default 0 values for all features
new_match_data = {col: 0 for col in feature_columns}

# Fill in the rolling averages for Ecuador (home) and Venezuela (away)
# Replace these with your actual computed values
home_team = input("Equipo local: ")
away_team = input("Equipo visitante: ")
# home_team_avg_scored, home_team_avg_conceded = get_home_average_scored(matches_south_america_no_dummies, home_team)
# away_team_avg_scored, away_team_avg_conceded = get_away_average_scored(matches_south_america_no_dummies, away_team)
# head_to_head_stats = get_head_to_head_avg_goal_diff(matches_south_america_no_dummies, home_team, away_team)
# new_match_data['home_team_avg_scored'] = home_team_avg_scored      # Example value for Ecuador at home
# new_match_data['home_team_avg_conceded'] = home_team_avg_conceded       # Example value for Ecuador at home
# new_match_data['away_team_avg_scored'] = away_team_avg_scored         # Example value for Venezuela away
# new_match_data['away_team_avg_conceded'] = away_team_avg_conceded        # Example value for Venezuela away
# new_match_data['head_to_head_goal_diff'] = head_to_head_stats        # Example value for head-to-head goal difference

# Set the appropriate dummy variables
home_team_dummy = 'home_team_' + home_team
away_team_dummy = 'away_team_' + away_team
new_match_data[home_team_dummy] = 1   # Ecuador is playing at home
new_match_data[away_team_dummy] = 1   # Venezuela is playing away

# Create the new match DataFrame
new_match_df = pd.DataFrame([new_match_data])
print(new_match_df)

# (Optional) Scale the new data if your model requires scaling
# new_match_scaled = scaler.transform(new_match_df)

# Predict the win probability using the trained model
model_list = []

lr_model, logloss_lr, accuracy_lr = get_logistic_regression_model(X_train, y_train, X_test, y_test)
model_list.append([lr_model, accuracy_lr, logloss_lr])
rf_model, logloss_rf, accuracy_rf = get_random_forest_model(X_train, y_train, X_test, y_test)
get_feature_importance_rf(rf_model)
model_list.append([rf_model, accuracy_rf, logloss_rf])
xgb_model, logloss_xgb, accuracy_xgb = get_xgb_model(X_train, y_train, X_test, y_test)
model_list.append([xgb_model, accuracy_xgb, logloss_xgb])
cat_model, logloss_cat, accuracy_cat = get_cat_model(X_train, y_train, X_test, y_test)
model_list.append([cat_model, accuracy_cat, logloss_cat])

probabilities_df = pd.DataFrame(index=['lr','rf','xgb','cat'], columns=['Away Win', 'Draw', 'Home Win', 'accuracy', 'logloss'])
i = 0

for model in model_list:
    probabilities = model[0].predict_proba(new_match_df)[0]
    probabilities_df.loc[i, 'Away Win'] = probabilities[0]
    probabilities_df.loc[i, 'Draw'] = probabilities[1]
    probabilities_df.loc[i,'Home Win'] = probabilities[2]
    probabilities_df.loc[i, 'accuracy'] = model[1]
    probabilities_df.loc[i, 'logloss'] = model[2]
    i += 1
print(probabilities_df)
print(f"Probabilities of the match: {home_team} vs {away_team}")