import os
from typing import Optional

from dotenv import load_dotenv
from supabase import ClientOptions, create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

_anon_supabase = None
_service_role_supabase = None


def get_supabase_client(access_token: Optional[str] = None, use_service_role: bool = False):
    global _anon_supabase
    global _service_role_supabase

    if access_token:
        return create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
            options=ClientOptions(
                headers={"Authorization": f"Bearer {access_token}"},
                auto_refresh_token=False,
                persist_session=False,
            ),
        )

    if use_service_role:
        if not SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY no está configurada")
        if _service_role_supabase is None:
            _service_role_supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        return _service_role_supabase

    if _anon_supabase is None:
        _anon_supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _anon_supabase
