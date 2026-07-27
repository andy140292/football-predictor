import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from backend.paths import PROCESSED_X_PATH, MODEL_PATHS, PROCESSED_y_PATH, RANKING_PATH
from backend.prediction.national_match_source import fetch_national_team_names
import requests
import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from supabase_client import get_supabase_client
from backend.predict_match import get_remaining_predictions, build_feature_vector


API_URL = "https://futbolconu-predictor.fly.dev/predict"  # Puedes cambiar esto más adelante por el endpoint de producción
LOCAL_MODEL_MODE = os.getenv("PREDICT_LOCAL_MODELS", "false").lower() == "true"

supabase = get_supabase_client()

bucket = "model-artifacts"

def load_file_from_supabase(bucket: str, path: str, as_dataframe=False):
    response = supabase.storage.from_(bucket).download(path)
    if as_dataframe:
        return pd.read_csv(io.BytesIO(response))
    else:
        return joblib.load(io.BytesIO(response))
    
def show_accuracy_log_loss(x_test, y_test, model, model_type):
    """Calcula y muestra la precisión y log loss del modelo."""
    if model_type == "logistic_regression":
        x_test_scaled = model.scaler.transform(x_test)
        x_test_eval = pd.DataFrame(x_test_scaled, columns=model.X_train_columns)
    else:
        x_test_eval = x_test

    y_pred_proba = model.predict_proba(x_test_eval)
    y_pred = model.model.predict(x_test_eval)

    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)

    st.write(f"📊 Modelo: {model_type}")
    st.write(f"✅ Accuracy: {acc:.3f}")
    st.write(f"📉 Log Loss: {loss:.3f}")

# Cargar modelos y data una sola vez (por ejemplo al inicio del módulo)
@st.cache_resource
def load_trained_models():
    """Carga los modelos entrenados y el dataset de features una sola vez."""
    X = pd.read_csv(PROCESSED_X_PATH)
    y = pd.read_csv(PROCESSED_y_PATH)
    rf_predictor = joblib.load(MODEL_PATHS["random_forest"])
    lr_predictor = joblib.load(MODEL_PATHS["logistic_regression"])
    mlp_predictor = joblib.load(MODEL_PATHS["mlp"])
    models = {
    "random_forest": rf_predictor,
    "logistic_regression": lr_predictor,
    "mlp": mlp_predictor,
    }
    return models, X, y


@st.cache_data(ttl=3600)
def load_national_team_names():
    return fetch_national_team_names()


def local_predict_outcome(home_team, away_team):
    models, X, _ = load_trained_models()
    fifa_rank = pd.read_csv(RANKING_PATH)
    match_vector = build_feature_vector(home_team, away_team, X, fifa_rank)
    results = {}

    for model_name, predictor in models.items():
        if predictor is None:
            continue
        probs = predictor.predict_proba(match_vector)[0]
        results[model_name] = {
            "home_win": probs[2],
            "draw": probs[1],
            "away_win": probs[0],
        }

    return results

def show_predict():
    # Verifica si el usuario está autenticado
    if "user" not in st.session_state or not st.session_state.user:
        st.warning("Debes iniciar sesión para hacer predicciones.")
        return

    # Equipos únicos
    teams = load_national_team_names()

    st.markdown(
        """
        Selecciona dos selecciones nacionales para predecir el resultado de un enfrentamiento entre ambos equipos 🔮.

        Datos que informan los modelos de predicción:

        - Enfrentamientos históricos entre ambos equipos.
        - Ranking FIFA.
        - Forma reciente de ambos equipos
        - Partidos en campo neutral. 

        El proximo paso es actualizar la aplicación para predicciones de clubes ⚽🏆.
        """
    )

    email = st.session_state.user

    remaining = get_remaining_predictions(email)

    # Mostrar el contador
    st.markdown(
        f"""
        <div style='background-color:#1E1E1E; padding:10px; border-radius:10px; margin-bottom:20px; text-align:center;'>
            <span style='font-size:18px; color:#ffffff;'>🧮 Predicciones restantes hoy: <strong style="color:#FFD700;">{remaining}</strong>/15</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Select teams
    st.markdown("<h3 style='text-align: left;'>Predicción de Partido</h3>", unsafe_allow_html=True)

    submitted = False
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            team1 = st.selectbox("Equipo Local", teams)
        
        with col2:
            team2 = st.selectbox("Equipo Visitante", teams)

        submitted = st.form_submit_button("Predecir resultado")

    if "last_prediction_results" in st.session_state and "last_prediction_teams" in st.session_state and st.session_state["last_prediction_teams"] == (team1, team2):
            results = st.session_state["last_prediction_results"]
            team1, team2 = st.session_state["last_prediction_teams"]

            if "random_forest" in results:
                st.markdown("### Modelo 1: 🌲 Bosque Aleatorio")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label=f"Victoria {team1} %", value=f"{results['random_forest']['home_win']:.0%}")
                with col2:
                    st.metric(label="Empate %", value=f"{results['random_forest']['draw']:.0%}")
                with col3:
                    st.metric(label=f"Victoria {team2} %", value=f"{results['random_forest']['away_win']:.0%}")

            if "logistic_regression" in results:
                st.markdown("### Modelo 2: 📈 Regresión Logística")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label=f"Victoria {team1} %", value=f"{results['logistic_regression']['home_win']:.0%}")
                with col2:
                    st.metric(label="Empate %", value=f"{results['logistic_regression']['draw']:.0%}")
                with col3:
                    st.metric(label=f"Victoria {team2} %", value=f"{results['logistic_regression']['away_win']:.0%}")

            if "mlp" in results and results["mlp"] is not None:
                st.markdown("### Modelo 3: 🤖 Red Neuronal (MLP)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label=f"Victoria {team1} %", value=f"{results['mlp']['home_win']:.0%}")
                with col2:
                    st.metric(label="Empate %", value=f"{results['mlp']['draw']:.0%}")
                with col3:
                    st.metric(label=f"Victoria {team2} %", value=f"{results['mlp']['away_win']:.0%}")
            else:
                st.markdown("### 🤖 Red Neuronal (MLP)")
                st.info("Modelo MLP no disponible.")

    if submitted:
        st.subheader("🔮 Probabilidades")
        
        # Verificación de que los equipos sean diferentes
        if team1 == team2:
            st.warning("Por favor selecciona dos equipos diferentes.")
            return

        if not LOCAL_MODEL_MODE and ("token" not in st.session_state or not st.session_state.token):
            st.warning("Necesitas iniciar sesión para hacer predicciones.")
            return
        
        if LOCAL_MODEL_MODE:
            results = local_predict_outcome(team1, team2)
            st.session_state["last_prediction_results"] = results
            st.session_state["last_prediction_teams"] = (team1, team2)
            st.rerun()

        headers = {
            "Authorization": f"Bearer {st.session_state.token}"
        }

        data = {
            "home_team": team1,
            "away_team": team2,
        }
        
        try:
            response = requests.post(API_URL, json=data, headers=headers)

            if response.status_code == 200:

                results = response.json().get("predicción", {})

                # Guarda resultados y equipos en session_state
                st.session_state["last_prediction_results"] = results
                st.session_state["last_prediction_teams"] = (team1, team2)
                
                # ✅ RERUN SOLO DESPUÉS DE MOSTRAR TODO
                st.rerun()

            elif response.status_code == 401:
                st.session_state.clear()
                st.error("Tu sesión expiró. Inicia sesión nuevamente.")
                st.stop()
            else:
                st.error(f"Error al obtener predicciones: {response.status_code} - {response.text}")

        except (ValueError, requests.exceptions.RequestException) as e:
            st.error(f"❌ {e}")


    st.markdown("""
                #### ¿Cómo funciona el algoritmo de Bosque Aleatorio?

                El modelo de Bosque Aleatorio es un conjunto de árboles de decisión que trabajan juntos para predecir el resultado de un partido. Cada árbol analiza diferentes combinaciones de datos como el historial entre selecciones, el ranking FIFA, el rendimiento reciente y si el partido se juega en campo neutral.

                En pocas palabras, es como tener varios analistas que evalúan el partido desde distintos ángulos y luego toman una decisión en conjunto.
                """)

    st.markdown("""
                #### ¿Cómo funciona el modelo de Regresión Logística? 📊⚽
                
                La Regresión Logística es un modelo estadístico que estima la probabilidad de cada posible resultado en un partido: victoria, empate o derrota. Para hacerlo, analiza diferentes factores como el ranking FIFA, la forma reciente de los equipos, enfrentamientos pasados y si el partido se juega en campo neutral.

                El modelo asigna un peso a cada una de estas variables y combina toda la información para calcular la probabilidad de cada resultado. Por ejemplo, puede predecir que hay un 60% de probabilidad de que gane el equipo local, 25% de empate y 15% de que gane el visitante.
                """)   

    st.markdown("""
                #### ¿Cómo funciona una Red Neuronal? 🧠⚽
                
                La Red Neuronal es un modelo inspirado en el cerebro humano que aprende a identificar patrones complejos entre datos. Para predecir el resultado de un partido, la red analiza múltiples factores como el ranking FIFA, la forma reciente, el rendimiento histórico entre equipos, y si el partido se juega en campo neutral.

                La información pasa por varias "capas" de neuronas artificiales, donde cada capa transforma y combina los datos de forma no lineal. A medida que la red entrena con más partidos reales, aprende a mejorar sus predicciones.
                """)    

    # -------------------------------------------------------------------------
    # Botón para mostrar/ocultar importancia de características
    # -------------------------------------------------------------------------
    if "trained_models" not in st.session_state:
        models, X, y = load_trained_models()
        st.session_state["trained_models"] = models
        st.session_state["trained_X"] = X
        st.session_state["trained_y"] = y

    models = st.session_state["trained_models"]
    X = st.session_state["trained_X"]
    y = st.session_state["trained_y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Dont show accuracy and log loss
    # for model_type, model in models.items():
    #     show_accuracy_log_loss(X_test, y_test, model, model_type)

    # Inicializar el estado del botón si no existe
    if "show_importance" not in st.session_state:
        st.session_state.show_importance = False

    # Toggle button con re-render inmediato
    if st.button(
        "🔍 Mostrar gráficas de importancia" if not st.session_state.show_importance else "🙈 Ocultar gráficas de importancia"
    ):
        st.session_state.show_importance = not st.session_state.show_importance
        st.rerun()

    # Mostrar gráficos si el toggle está activo
    if st.session_state.show_importance:
        st.header("🔍 Importancia de características por modelo")

        for model_name, predictor in models.items():
            if predictor is None:
                st.info(f"Modelo {model_name} no disponible.")
                continue

            st.subheader(model_name.replace("_", " ").title())

            plt.clf()
            fig = predictor.feature_importance(X, top_n=20)
            if fig is not None:
                st.pyplot(fig)
            else:
                st.warning("⚠️ Este modelo no permite visualizar la importancia de características.")
