import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import log_loss, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from typing import Optional

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
            base_model = LogisticRegression(solver='lbfgs', max_iter=1000)
            return CalibratedClassifierCV(base_model, method='isotonic', cv=5)
        elif self.model_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.01,
                learning_rate='adaptive',
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )

        else:
            raise ValueError("Unsupported model type. Choose 'random_forest', 'xgboost', or 'catboost'.")
    def train(self, X_train, y_train):
        if self.model_type in ["logistic_regression", "mlp"]:
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.X_train_columns = X_train.columns

            self.model.fit(X_train_scaled, y_train)
        else:
            self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        if self.model_type in ["logistic_regression", "mlp"]:
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
        else:
            return self.model.predict_proba(X)

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

    def feature_importance(self, X_train, top_n: Optional[int] = None):
        """Devuelve un gráfico de importancia de características.

        Args:
            X_train (pd.DataFrame): dataset de entrenamiento (solo usado para nombres).
            top_n (int | None): cuántas características mostrar (None = todas)
        """
        # Extraer el estimator correcto
        model = self.model.best_estimator_ if hasattr(self.model, "best_estimator_") else self.model

        if self.model_type in ["random_forest", "xgboost"] and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            names = X_train.columns
        elif self.model_type == "logistic_regression" and hasattr(model, "coef_"):
            importances = model.coef_[0]
            names = self.X_train_columns
        else:
            print("Feature importance not supported for this model.")
            return None

        # Crear DataFrame y ordenar por valor absoluto
        feat_df = pd.DataFrame({"Feature": names, "Importance": importances})
        feat_df["abs"] = feat_df["Importance"].abs()
        feat_df = feat_df.sort_values("abs", ascending=False)

        if top_n is not None:
            feat_df = feat_df.head(top_n)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.barh(feat_df["Feature"], feat_df["Importance"])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"{self.model_type} Feature Importance (top {top_n if top_n else 'all'})")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        return plt.gcf()
