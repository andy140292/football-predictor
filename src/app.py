# app.py
import streamlit as st
from streamlit_option_menu import option_menu
from components.login import show_login
from components.predict import show_predict

st.set_page_config(
    page_title="Predictor",
    layout="wide",
    page_icon="/src/static/images/logo.png",
    initial_sidebar_state="collapsed",
)

# Mostrar banner principal antes de cualquier navegación
st.markdown(
    """
    <style>
        .main-banner img {
            width: 100%;
            height: auto;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        .main-banner {
            padding: 0;
            margin: 0;
        }
    </style>
    <div class="main-banner">
        <img src="https://rfpkmlsmvypuonqstfpe.supabase.co/storage/v1/object/public/images//predictor_banner_site%20(2).png" alt="Futbol con U">
    </div>
    """,
    unsafe_allow_html=True
)

# ✅ Inicializar session_state
if "selected_option" not in st.session_state:
    st.session_state.selected_option = "Iniciar sesión"
    
selected = option_menu(
    menu_title=None,
    options=["Iniciar sesión", "Predicción"],
    icons=["box-arrow-in-right", "trophy"],
    menu_icon="cast",
    default_index=["Iniciar sesión", "Predicción"].index(st.session_state.selected_option),
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#0e1117"},
        "icon": {"color": "white", "font-size": "16px"},
        "nav-link": {"color": "white", "font-size": "16px", "text-align": "center"},
        "nav-link-selected": {"background-color": "#6c757d"},
    }
)
# Importante: actualiza el estado solo si cambia
if selected != st.session_state.selected_option:
    st.session_state.selected_option = selected

if st.session_state.selected_option == "Iniciar sesión":
    show_login()
elif st.session_state.selected_option == "Predicción":
    show_predict()