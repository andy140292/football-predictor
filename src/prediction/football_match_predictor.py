import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import log_loss, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import os

class FootballMatchPredictor:
    def __init__(self, model_type="random_forest"):
        """
        Initializes the model based on the selected model type.
        Supported model types: "random_forest", "xgboost", "catboost", "logistic_regression".
        """
        self.model_type = model_type
        self.model = self._initialize_model()
        self.scaler = StandardScaler()
        self.X_train_columns = None  # store feature columns for logistic regression

    def _initialize_model(self):
        """Private method to initialize the selected model."""
        if self.model_type == "random_forest":
            rf_param_grid = {
                'n_estimators': [100, 200, 500],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
            rf_random_search = RandomizedSearchCV(
                estimator=rf_model, param_distributions=rf_param_grid,
                n_iter=10, scoring='neg_log_loss', cv=5, verbose=2, n_jobs=-1, random_state=42
            )
            return rf_random_search
        elif self.model_type == "xgboost":
            return XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", use_label_encoder=False)
        elif self.model_type == "catboost":
            return CatBoostClassifier(loss_function='MultiClass', verbose=0)
        elif self.model_type == "logistic_regression":
            return LogisticRegression(solver='lbfgs', max_iter=1000)
        else:
            raise ValueError("Unsupported model type. Choose 'random_forest', 'xgboost', or 'catboost'.")

    def train(self, X_train, y_train, X=pd.DataFrame()):
        """Trains the model on the provided dataset."""
        self.X_train_columns = X_train.columns  # store for logistic regression feature importances
        if self.model_type == "logistic_regression":
            X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X.columns)
            self.model.fit(X_train_scaled, y_train)  # train on scaled data
        else:
            self.model.fit(X_train, y_train)
        print(f"{self.model_type} model trained successfully!")

    def evaluate(self, X_test, y_test):
        """Evaluates the model and prints performance metrics."""
        if self.model_type == "logistic_regression":
            X_test = pd.DataFrame(self.scaler.transform(X_test), columns=self.X_train_columns)
        y_pred_proba = self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)

        logloss = log_loss(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"{self.model_type} Log Loss: {logloss}")
        print(f"{self.model_type} Accuracy: {accuracy}")

        return logloss, accuracy

    def feature_importance(self, X_train):
        """Plots and saves feature importance or coefficients for supported models."""
        if self.model_type in ["random_forest", "xgboost"]:
            rf_best_model = self.model.best_estimator_
            feature_importances = rf_best_model.feature_importances_
            feature_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)

        elif self.model_type == "logistic_regression":
            coef = self.model.coef_[0]
            feature_df = pd.DataFrame({
                'Feature': self.X_train_columns,
                'Importance': coef
            }).sort_values(by='Importance', key=abs, ascending=False)

        else:
            print("Feature importance is only available for tree-based models or logistic regression.")
            return

        # Plot
        plt.figure(figsize=(12, 6))
        plt.barh(feature_df['Feature'], feature_df['Importance'])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"{self.model_type} Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def predict_match(self, match_features):
        """Predicts probabilities for a given match."""
        if self.model_type == "logistic_regression":
            match_features = pd.DataFrame(self.scaler.transform(match_features), columns=self.X_train_columns)
        probabilities = self.model.predict_proba(match_features)[0]
        return {
            "Home Win Probability": probabilities[2],
            "Draw Probability": probabilities[1],
            "Away Win Probability": probabilities[0]
        }
