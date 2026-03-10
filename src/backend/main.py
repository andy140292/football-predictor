from fastapi import FastAPI, HTTPException, Header, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
import requests
from dotenv import load_dotenv
import sys
import os
import logging
from time import perf_counter
from uuid import uuid4
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Optional

try:
    from .schema import (
        MatchInput,
        RecentFormInput,
        HeadToHeadInput,
        HeadToHeadResponse,
        MatchOut,
        TeamVsConfedInput,
        TeamVsConfedResponse,
        MatchPredictionCreateInput,
        MatchPredictionCreateResponse,
        CalendarBatchUpsertInput,
        CalendarBatchUpsertResponse,
        ModelScorecardOut,
        ModelScorecardMatchesResponse,
    )
    from .match_service import get_recent_matches, get_head_to_head, get_team_vs_confed
    from .predict_match import (
        predict_outcome,
        create_or_get_match_prediction,
        upsert_matches_calendar_batch,
    )
    from .scorecard_service import get_model_scorecard, list_model_scorecard_matches
except ImportError:  # pragma: no cover - fallback for direct module execution
    from src.backend.schema import (
        MatchInput,
        RecentFormInput,
        HeadToHeadInput,
        HeadToHeadResponse,
        MatchOut,
        TeamVsConfedInput,
        TeamVsConfedResponse,
        MatchPredictionCreateInput,
        MatchPredictionCreateResponse,
        CalendarBatchUpsertInput,
        CalendarBatchUpsertResponse,
        ModelScorecardOut,
        ModelScorecardMatchesResponse,
    )
    from src.backend.match_service import get_recent_matches, get_head_to_head, get_team_vs_confed
    from src.backend.predict_match import (
        predict_outcome,
        create_or_get_match_prediction,
        upsert_matches_calendar_batch,
    )
    from src.backend.scorecard_service import get_model_scorecard, list_model_scorecard_matches


load_dotenv()

# Usa tus claves de Supabase (desde supabase.com > Project > API)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
API_ENV = os.getenv("API_ENV", "prod")
LOG_LEVEL = os.getenv("API_LOG_LEVEL", "INFO").upper()

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
logger = logging.getLogger("futbolconu.api")

app = FastAPI()

_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
ALLOWED_ORIGINS = [o.strip() for o in _raw.split(",") if o.strip()]

print(">>> ALLOWED_ORIGINS:", ALLOWED_ORIGINS)  # 👀 Debe mostrar ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _ensure_request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        return request_id

    request_id = request.headers.get("X-Request-ID") or uuid4().hex[:12]
    request.state.request_id = request_id
    return request_id


def verify_token(request: Request, authorization: Optional[str] = Header(default=None)):
    request_id = _ensure_request_id(request)
    start_time = perf_counter()
    logger.info("auth_verify_started request_id=%s", request_id)

    if os.getenv("API_ENV") == "dev":
        logger.info(
            "auth_verify_skipped_dev_mode request_id=%s elapsed_ms=%.2f",
            request_id,
            (perf_counter() - start_time) * 1000.0,
        )
        return {"sub": "dev-user", "email": "dev@example.com", "token": None}

    if not authorization or not authorization.startswith("Bearer "):
        logger.warning("auth_verify_failed_invalid_header request_id=%s", request_id)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format")


    token = authorization.split(" ")[1]

    # Verifica el token contra Supabase
    resp = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_KEY},
        timeout=10,
    )
    if resp.status_code != 200:
        logger.warning(
            "auth_verify_failed_supabase request_id=%s status_code=%s elapsed_ms=%.2f",
            request_id,
            resp.status_code,
            (perf_counter() - start_time) * 1000.0,
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido o expirado")

    user_info = resp.json() or {}
    email = user_info.get("email") or user_info.get("user_metadata", {}).get("email")
    sub = user_info.get("id") or user_info.get("sub")

    logger.info(
        "auth_verify_succeeded request_id=%s sub=%s elapsed_ms=%.2f",
        request_id,
        sub,
        (perf_counter() - start_time) * 1000.0,
    )

    # 🔙 DEVUELVE SIEMPRE ALGO
    return {"email": email, "sub": sub, "token": token, "claims": user_info}


def verify_admin_key(request: Request, x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key")):
    request_id = _ensure_request_id(request)
    expected_admin_key = os.getenv("ADMIN_API_KEY")
    if not expected_admin_key or x_admin_key != expected_admin_key:
        logger.warning("admin_key_verification_failed request_id=%s", request_id)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return True

    
@app.get("/healthz")
def healthz():
    return {"status": "ok", "env": API_ENV}

@app.post("/predict")
def predict(request: Request, response: Response, input: MatchInput, user=Depends(verify_token)):
    request_id = _ensure_request_id(request)
    response.headers["X-Request-ID"] = request_id
    start_time = perf_counter()
    logger.info(
        "predict_request_received request_id=%s sub=%s mode=%s home_team=%s away_team=%s competition=%s round=%s neutral=%s",
        request_id,
        user.get("sub"),
        input.mode,
        input.home_team,
        input.away_team,
        input.competition,
        input.round,
        input.neutral,
    )
    try:
        result = predict_outcome(
            home_team=input.home_team,
            away_team=input.away_team,
            token=user.get("token"),
            user_email=user.get("email"),
            user_id=user.get("sub"),
            mode=input.mode,
            competition=input.competition,
            round_name=input.round,
            neutral=input.neutral,
            request_id=request_id,
        )
        logger.info(
            "predict_request_succeeded request_id=%s models=%s elapsed_ms=%.2f",
            request_id,
            ",".join(sorted(result.keys())),
            (perf_counter() - start_time) * 1000.0,
        )
        return {"predicción": result}
    except Exception as e:
        logger.exception(
            "predict_request_failed request_id=%s elapsed_ms=%.2f",
            request_id,
            (perf_counter() - start_time) * 1000.0,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/match-predictions", response_model=MatchPredictionCreateResponse)
def create_match_prediction(
    request: Request,
    response: Response,
    input: MatchPredictionCreateInput,
    user=Depends(verify_token),
):
    request_id = _ensure_request_id(request)
    response.headers["X-Request-ID"] = request_id
    start_time = perf_counter()
    logger.info(
        "match_prediction_request_received request_id=%s sub=%s match_id=%s predicted_outcome=%s",
        request_id,
        user.get("sub"),
        input.match_id,
        input.predicted_outcome,
    )
    try:
        result = create_or_get_match_prediction(
            user_id=user.get("sub"),
            email=user.get("email"),
            match_id=str(input.match_id),
            predicted_outcome=input.predicted_outcome,
            token=user.get("token"),
            request_id=request_id,
        )
        logger.info(
            "match_prediction_request_succeeded request_id=%s status=%s elapsed_ms=%.2f",
            request_id,
            result.get("status"),
            (perf_counter() - start_time) * 1000.0,
        )
        return result
    except ValueError as e:
        logger.warning(
            "match_prediction_request_rejected request_id=%s reason=%s elapsed_ms=%.2f",
            request_id,
            str(e),
            (perf_counter() - start_time) * 1000.0,
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(
            "match_prediction_request_failed request_id=%s elapsed_ms=%.2f",
            request_id,
            (perf_counter() - start_time) * 1000.0,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/matches-calendar/upsert-batch", response_model=CalendarBatchUpsertResponse)
def upsert_matches_calendar(
    request: Request,
    response: Response,
    input: CalendarBatchUpsertInput,
    _: bool = Depends(verify_admin_key),
):
    request_id = _ensure_request_id(request)
    response.headers["X-Request-ID"] = request_id
    start_time = perf_counter()
    logger.info(
        "calendar_upsert_request_received request_id=%s rows=%s",
        request_id,
        len(input.matches or []),
    )
    try:
        payload = [
            row.model_dump() if hasattr(row, "model_dump") else row.dict()
            for row in (input.matches or [])
        ]
        result = upsert_matches_calendar_batch(payload, request_id=request_id)
        logger.info(
            "calendar_upsert_request_succeeded request_id=%s inserted=%s updated=%s skipped=%s errors=%s elapsed_ms=%.2f",
            request_id,
            result.get("inserted"),
            result.get("updated"),
            result.get("skipped"),
            len(result.get("errors", [])),
            (perf_counter() - start_time) * 1000.0,
        )
        return result
    except Exception as e:
        logger.exception(
            "calendar_upsert_request_failed request_id=%s elapsed_ms=%.2f",
            request_id,
            (perf_counter() - start_time) * 1000.0,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-scorecard", response_model=ModelScorecardOut)
def read_model_scorecard(
    request: Request,
    response: Response,
    mode: str = "national",
    model_version: str = "",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    user=Depends(verify_token),
):
    request_id = _ensure_request_id(request)
    response.headers["X-Request-ID"] = request_id
    start_time = perf_counter()
    logger.info(
        "model_scorecard_request_received request_id=%s sub=%s mode=%s model_version=%s from_date=%s to_date=%s",
        request_id,
        user.get("sub"),
        mode,
        model_version,
        from_date,
        to_date,
    )
    try:
        result = get_model_scorecard(
            mode=mode,
            model_version=model_version,
            from_date=from_date,
            to_date=to_date,
        )
        logger.info(
            "model_scorecard_request_succeeded request_id=%s total_scored=%s elapsed_ms=%.2f",
            request_id,
            result.get("total_scored"),
            (perf_counter() - start_time) * 1000.0,
        )
        return result
    except ValueError as e:
        logger.warning(
            "model_scorecard_request_rejected request_id=%s reason=%s elapsed_ms=%.2f",
            request_id,
            str(e),
            (perf_counter() - start_time) * 1000.0,
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(
            "model_scorecard_request_failed request_id=%s elapsed_ms=%.2f",
            request_id,
            (perf_counter() - start_time) * 1000.0,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-scorecard/matches", response_model=ModelScorecardMatchesResponse)
def read_model_scorecard_matches(
    request: Request,
    response: Response,
    mode: str = "national",
    model_version: str = "",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    verdict: str = "all",
    page: int = 1,
    page_size: int = 50,
    user=Depends(verify_token),
):
    request_id = _ensure_request_id(request)
    response.headers["X-Request-ID"] = request_id
    start_time = perf_counter()
    logger.info(
        "model_scorecard_matches_request_received request_id=%s sub=%s mode=%s model_version=%s verdict=%s page=%s page_size=%s from_date=%s to_date=%s",
        request_id,
        user.get("sub"),
        mode,
        model_version,
        verdict,
        page,
        page_size,
        from_date,
        to_date,
    )
    try:
        result = list_model_scorecard_matches(
            mode=mode,
            model_version=model_version,
            from_date=from_date,
            to_date=to_date,
            verdict=verdict,
            page=page,
            page_size=page_size,
        )
        logger.info(
            "model_scorecard_matches_request_succeeded request_id=%s total=%s returned=%s elapsed_ms=%.2f",
            request_id,
            result.get("total"),
            len(result.get("matches", [])),
            (perf_counter() - start_time) * 1000.0,
        )
        return result
    except ValueError as e:
        logger.warning(
            "model_scorecard_matches_request_rejected request_id=%s reason=%s elapsed_ms=%.2f",
            request_id,
            str(e),
            (perf_counter() - start_time) * 1000.0,
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(
            "model_scorecard_matches_request_failed request_id=%s elapsed_ms=%.2f",
            request_id,
            (perf_counter() - start_time) * 1000.0,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recent-form", response_model=list[MatchOut])
def recent_form(request: Request, response: Response, input: RecentFormInput, user=Depends(verify_token)):
    request_id = _ensure_request_id(request)
    response.headers["X-Request-ID"] = request_id
    start_time = perf_counter()
    logger.info(
        "recent_form_request_received request_id=%s sub=%s home_team=%s away_team=%s last_matches=%s",
        request_id,
        user.get("sub"),
        input.home_team,
        input.away_team,
        input.last_matches,
    )
    try:
        result = get_recent_matches(
            home_team=input.home_team,
            away_team=input.away_team,
            last_matches=input.last_matches,
            request_id=request_id,
        )
        logger.info(
            "recent_form_request_succeeded request_id=%s returned_matches=%s elapsed_ms=%.2f",
            request_id,
            len(result),
            (perf_counter() - start_time) * 1000.0,
        )
        return result
    except Exception as e:
        logger.exception(
            "recent_form_request_failed request_id=%s elapsed_ms=%.2f",
            request_id,
            (perf_counter() - start_time) * 1000.0,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/head-to-head", response_model=HeadToHeadResponse)
def head_to_head(request: Request, response: Response, input: HeadToHeadInput, user=Depends(verify_token)):
    request_id = _ensure_request_id(request)
    response.headers["X-Request-ID"] = request_id
    start_time = perf_counter()
    logger.info(
        "head_to_head_request_received request_id=%s sub=%s home_team=%s away_team=%s tournaments=%s",
        request_id,
        user.get("sub"),
        input.home_team,
        input.away_team,
        len(input.tournaments or []),
    )
    try:
        result = get_head_to_head(
            home_team=input.home_team,
            away_team=input.away_team,
            tournaments=input.tournaments,
            request_id=request_id,
        )
        logger.info(
            "head_to_head_request_succeeded request_id=%s returned_matches=%s elapsed_ms=%.2f",
            request_id,
            len(result.get("matches", [])),
            (perf_counter() - start_time) * 1000.0,
        )
        return result
    except Exception as e:
        logger.exception(
            "head_to_head_request_failed request_id=%s elapsed_ms=%.2f",
            request_id,
            (perf_counter() - start_time) * 1000.0,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/team-vs-confed", response_model=TeamVsConfedResponse)
def team_vs_confed(request: Request, response: Response, input: TeamVsConfedInput, user=Depends(verify_token)):
    request_id = _ensure_request_id(request)
    response.headers["X-Request-ID"] = request_id
    start_time = perf_counter()
    logger.info(
        "team_vs_confed_request_received request_id=%s sub=%s team=%s opponent_confed=%s",
        request_id,
        user.get("sub"),
        input.team,
        input.opponent_confederation,
    )
    try:
        result = get_team_vs_confed(
            team=input.team,
            opponent_confederation=input.opponent_confederation,
            request_id=request_id,
        )
        logger.info(
            "team_vs_confed_request_succeeded request_id=%s matches_count=%s elapsed_ms=%.2f",
            request_id,
            result.get("matches_count"),
            (perf_counter() - start_time) * 1000.0,
        )
        return result
    except Exception as e:
        logger.exception(
            "team_vs_confed_request_failed request_id=%s elapsed_ms=%.2f",
            request_id,
            (perf_counter() - start_time) * 1000.0,
        )
        raise HTTPException(status_code=500, detail=str(e))
