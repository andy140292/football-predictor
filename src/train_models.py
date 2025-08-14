import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from prediction.match_data_preprocessor import MatchDataPreprocessor
from prediction.football_match_predictor import FootballMatchPredictor
from pathlib import Path
from utils.paths import MODEL_PATHS, PROCESSED_X_PATH, PROCESSED_X__FULL_PATH, PROCESSED_y_PATH, MATCHES_PATH
from datetime import datetime, timedelta
import io
import tempfile
from supabase_client import get_supabase_client

month_str = (datetime.today() - timedelta(days=30)).strftime("%Y_%m")

supabase = get_supabase_client()

def upload_model_to_supabase(predictor, filename, model_type):
    try:
        local_temp_path = f"/tmp/{filename}"  # puedes usar otra ruta si /tmp no existe

        joblib.dump(predictor, local_temp_path)

        # 2. Definir ruta en Supabase Storage
        supabase_path = f"{month_str}/{filename}"

        # 3. Subir el archivo desde disco
        supabase.storage.from_("model-artifacts").upload(
            supabase_path,
            local_temp_path,
            file_options={"content-type": "application/octet-stream", "cache-control": "3600"}
        )

        print(f"✅ Modelo {model_type} subido a Supabase en {supabase_path}")
    except Exception as e:
        print(f"❌ Error al subir el modelo {model_type} a Supabase.")

def upload_processed_data_to_supabase(X, y, X_full):
    # Diccionario con nombre del archivo y su DataFrame
    csv_dataframes = {
        f"{month_str}/X_processed.csv": X,
        f"{month_str}/y_processed.csv": y,
        f"{month_str}/X_full_processed.csv": X_full
    }
    
    for remote_path, df in csv_dataframes.items():
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                df.to_csv(tmp_file.name, index=False)
                tmp_file.flush()  # Asegura que se escriba en disco
                supabase.storage.from_("model-artifacts").upload(
                    remote_path,
                    tmp_file.name,
                    file_options={"content-type": "text/csv", "cache-control": "3600"}
                )
            print(f"✅ CSV subido exitosamente a Supabase: {remote_path}")
        except Exception as e:
            print(f"❌ Error al subir {remote_path} a Supabase.")
            print(f"🧨 {type(e)}: {e}")


def save_model_locally(predictor, model_type):
    # Save model locally
    local_model_path = MODEL_PATHS[model_type]
    joblib.dump(predictor, local_model_path)
    print(f"✅ Modelo {model_type} guardado localmente en {local_model_path}")

def save_processed_data_to_csv(X, y, X_full):
    # Save processed data
    X.to_csv(PROCESSED_X_PATH, index=False)
    y.to_csv(PROCESSED_y_PATH, index=False)
    X_full.to_csv(PROCESSED_X__FULL_PATH, index=False)

def train_model(model_type, X, y):
    predictor = FootballMatchPredictor(model_type=model_type)
    predictor.train(X, y)


    filename = f"{model_type}.pkl"
    
    # Upload model to Supabase. This is expensive, so we save the models locally
    # upload_model_to_supabase(predictor, filename, model_type)
    
    # Save model locally
    save_model_locally(predictor, model_type)
    print(f"✅ Modelo {model_type} entrenado y guardado.")

    # Store best estimator if needed
    best_model = (
        predictor.model.best_estimator_
        if hasattr(predictor.model, "best_estimator_")
        else predictor.model
    )
    return best_model


def train_models():
    file_path = f"matches_{month_str}"
    print(file_path)
    preprocessor = MatchDataPreprocessor(file_path, from_supabase=False)

    X, y = preprocessor.preprocess()

    # Save processed data for prediction module
    X_full = preprocessor.X_Full

    # Save processed data
    save_processed_data_to_csv(X, y, X_full)

    # Upload processed data to Supabase. Only doing it locally for now.
    # upload_processed_data_to_supabase(X, y, X_full)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {}

    for model_type in ["random_forest", "logistic_regression", "mlp"]:
        print(f"🔁 Entrenando modelo {model_type}...")
        best_model = train_model(model_type, X_train, y_train)
        models[model_type] = best_model

    return models, X, y, None

def plot_feature_importance(model, X, class_index=None, top_n=20):
    """
    Plot the top_n most important features for a given model.
    Supports Random Forest and Logistic Regression.
    """
    if hasattr(model, "feature_importances_"):  # Tree-based
        importances = model.feature_importances_
        title = "Feature Importance (Tree-based Model)"
    elif hasattr(model, "coef_"):  # Logistic Regression
        importances = model.coef_[class_index]
        title = f"Coeficiente (Regresión Logística) - Clase {class_index}"
    else:
        raise ValueError("Modelo no compatible con importancia de características.")

    # Get top N features
    indices = np.argsort(np.abs(importances))[::-1][:top_n]
    top_features = X.columns[indices]
    top_importance = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_features[::-1], top_importance[::-1])
    ax.set_title(title)
    ax.set_xlabel("Coeficiente / Importancia")
    ax.grid(True)
    return fig

def plot_mlp_feature_weights(mlp, X, class_index=0):
    # First layer weights → input layer to first hidden layer
    weights = mlp.coefs_[0][:, :]  # shape: (n_features, n_hidden)

    # Take mean or max absolute weight per input feature
    importance = abs(weights).mean(axis=1)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    indices = np.argsort(importance)[::-1]
    sorted_features = np.array(X.columns)[indices]
    sorted_importance = importance[indices]

    ax.barh(sorted_features[:15][::-1], sorted_importance[:15][::-1])
    ax.set_title("Importancia aproximada (MLP)")
    return fig

def train_all_models_if_needed():
    base_dir = Path(__file__).resolve().parent.parent  # raíz del proyecto
    models_dir = base_dir / "models"
    model_paths = {
        "random_forest": models_dir / "random_forest_predictor.pkl",
        "logistic_regression": models_dir / "logistic_regression_predictor.pkl",
        "mlp": models_dir / "mlp_predictor.pkl"
    }

    # Solo entrena si falta alguno
    if not all(p.exists() for p in model_paths.values()):
        print("🔁 Modelos no encontrados. Entrenando...")
        return train_models()
    else:
        print("✅ Modelos ya entrenados. Usando existentes.")
        return None, None, None
