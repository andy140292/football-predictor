import os
from gotrue.errors import AuthApiError
from supabase_client import get_supabase_client

supabase = get_supabase_client()

def register_user(email: str, password: str, subscribe: bool = False) -> bool:
    try:
        supabase.auth.sign_up(
            {
                "email": email,
                "password": password,
            }
        )
        # Autenticación (forzar inicio de sesión)
        login_response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        session = login_response.session
        user = login_response.user

        access_token = login_response.session.access_token
        refresh_token = login_response.session.refresh_token

        # 🔥 Esto es obligatorio para que auth.uid() funcione correctamente
        supabase.auth.set_session(access_token, refresh_token)

        user = login_response.user
        if user:
            # 👇 Guarda en tabla de suscriptores
            supabase.table("subscribers").insert({
                "email": email,
                "newsletter_opt_in": subscribe,
                "user_id": user.id,
            }).execute()

            return True
    except Exception as e:
        print(f"Error al registrar usuario: {e}")
        return False

def login_user(email: str, password: str) -> bool:
    """
    Devuelve True si las credenciales son correctas,
    False si son incorrectas o si ocurre cualquier otro error controlado.
    """
    try:
        response = supabase.auth.sign_in_with_password(
            {"email": email, "password": password}
        )
        return response if response.session else None
    except AuthApiError as e:
        # Credenciales inválidas
        print(f"Login error: {e}")
        return False
    except Exception as e:
        # Otro error (conexión, configuración, etc.)
        print(f"Error inesperado en login: {e}")
        return False