from fastapi import FastAPI, HTTPException, Header, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
import requests
from src.backend.schema import MatchInput
from src.predict_match import predict_outcome
from dotenv import load_dotenv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Optional


load_dotenv()

# Usa tus claves de Supabase (desde supabase.com > Project > API)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
API_ENV = os.getenv("API_ENV", "prod")

app = FastAPI()

_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
ALLOWED_ORIGINS = [o.strip() for o in _raw.split(",") if o.strip()]

print(">>> ALLOWED_ORIGINS:", ALLOWED_ORIGINS)  # 👀 Debe mostrar ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_token(authorization: Optional[str] = Header(default=None)):

    if os.getenv("API_ENV") == "dev":
        return {"sub": "dev-user", "email": "dev@example.com", "token": None}

    if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format")


    token = authorization.split(" ")[1]

    # Verifica el token contra Supabase
    resp = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_KEY},
        timeout=10,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido o expirado")

    user_info = resp.json() or {}
    email = user_info.get("email") or user_info.get("user_metadata", {}).get("email")
    sub = user_info.get("id") or user_info.get("sub")

    # 🔙 DEVUELVE SIEMPRE ALGO
    return {"email": email, "sub": sub, "token": token, "claims": user_info}

    
@app.get("/healthz")
def healthz():
    return {"status": "ok", "env": API_ENV}

@app.post("/predict")
def predict(input: MatchInput, user=Depends(verify_token)):
    try:
        result = predict_outcome(
            home_team=input.home_team,
            away_team=input.away_team,
            token=user.get("token")
        )
        return {"predicción": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))