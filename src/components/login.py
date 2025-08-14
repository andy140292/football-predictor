import streamlit as st
import time
from components.auth import register_user, login_user, supabase  # asegúrate de importar supabase aquí

def show_login():
    st.header("🔐 Iniciar sesión o registrarse")

    if "user" in st.session_state and st.session_state.user:
        st.success(f"Sesión iniciada como: {st.session_state.user}")
        if st.button("Cerrar sesión"):
            st.session_state.user = None
            st.rerun()
        return

    option = st.radio("¿Qué deseas hacer?", ["Iniciar sesión", "Registrarse"])
    email = st.text_input("Correo electrónico")
    password = st.text_input("Contraseña", type="password")

    if option == "Registrarse":
        subscribe = st.checkbox("Deseo recibir actualizaciones y noticias por correo.")
        with st.expander("Ver Términos y Condiciones"):
            st.markdown("""
            Al crear una cuenta, aceptas nuestros términos y condiciones:

            - Tus correo se almacenará de forma segura y no se compartirá con terceros.
            - Podemos enviarte correos relacionados con mejoras del sistema o noticias deportivas.
            - Puedes darte de baja en cualquier momento.
            """)
        accepted_terms = st.checkbox("Acepto los términos y condiciones", key="terms")

        if st.button("Crear cuenta"):
            if not accepted_terms:
                st.error("Debes aceptar los términos y condiciones para continuar.")
            elif register_user(email, password, subscribe):
                st.success("Usuario registrado con éxito. Ahora puedes iniciar sesión.")
            else:
                st.error("Este correo ya está registrado.")

    else:
        if st.button("Iniciar sesión"):
            result = login_user(email, password)
            if result:
                st.session_state.user = email
                st.session_state.token = result.session.access_token  # Esto es clave
                st.success("Login exitoso")
                st.session_state.selected_option = "Predicción"
                st.rerun()
            else:
                st.error("Correo o contraseña incorrectos.")

        # 🔁 Recuperar contraseña
        if st.button("¿Olvidaste tu contraseña?"):
            if email:
                try:
                    supabase.auth.reset_password_email(email)
                    st.info("Se envió un enlace de recuperación a tu correo.")
                except Exception as e:
                    st.error(f"Error al enviar correo de recuperación: {e}")
            else:
                st.warning("Por favor, ingresa primero tu correo.")