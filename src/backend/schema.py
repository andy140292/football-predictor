from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from uuid import UUID

class MatchInput(BaseModel):
    home_team: str
    away_team: str
    mode: str = "national"
    competition: Optional[str] = None
    round: Optional[str] = None
    neutral: Optional[int] = None


class RecentFormInput(BaseModel):
    home_team: str
    away_team: str
    last_matches: int = Field(gt=0, le=50)
    mode: Literal["national", "libertadores"] = "national"


class HeadToHeadInput(BaseModel):
    home_team: str
    away_team: str
    tournaments: List[str]


class MatchOut(BaseModel):
    date: Optional[str] = None
    home_team: str
    away_team: str
    home_score: int
    away_score: int


class RecentFormResponse(BaseModel):
    home_team: str
    away_team: str
    home_matches: List[MatchOut]
    away_matches: List[MatchOut]


class TeamForm(BaseModel):
    team: str
    wins: int
    draws: int
    losses: int
    goals: int


class HeadToHeadResponse(BaseModel):
    matches: List[MatchOut]
    home_form: TeamForm
    away_form: TeamForm


class TeamVsConfedInput(BaseModel):
    team: str
    opponent_confederation: str


class TeamVsConfedResponse(BaseModel):
    team: str
    opponent_confederation: str
    matches_count: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int


class FutureMatchOut(BaseModel):
    match_id: UUID
    home_team: str
    away_team: str
    match_date: str


class MatchPredictionCreateInput(BaseModel):
    match_id: UUID
    predicted_outcome: Literal["home_win", "away_win", "draw"]


class UserMatchPredictionOut(BaseModel):
    prediction_id: UUID
    match_id: UUID
    predicted_outcome: Literal["home_win", "away_win", "draw"]
    created_at: str


class MatchPredictionCreateResponse(BaseModel):
    status: Literal["created", "exists"]
    prediction: UserMatchPredictionOut


class CalendarMatchOut(BaseModel):
    match_id: UUID
    home_team: str
    away_team: str
    match_date: str
    tournament: Optional[str] = None
    predicted_outcome: Optional[Literal["home_win", "away_win", "draw"]] = None
    prediction_id: Optional[UUID] = None
    prediction_created_at: Optional[str] = None


class MatchesCalendarResponse(BaseModel):
    mode: Literal["world_cup"]
    matches: List[CalendarMatchOut]


class CalendarMatchUpsertIn(BaseModel):
    home_team: str
    away_team: str
    match_date: str
    tournament: Optional[str] = None
    home_team_code: Optional[str] = None
    away_team_code: Optional[str] = None


class CalendarUpsertError(BaseModel):
    row_index: int
    reason: str


class CalendarBatchUpsertInput(BaseModel):
    matches: List[CalendarMatchUpsertIn]


class CalendarBatchUpsertResponse(BaseModel):
    received: int
    inserted: int
    updated: int
    skipped: int
    errors: List[CalendarUpsertError]


class MatchHistoryUpsertIn(BaseModel):
    date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    tournament: str
    city: str
    country: str
    neutral: bool


class MatchHistoryBatchUpsertInput(BaseModel):
    matches: List[MatchHistoryUpsertIn]


class MatchHistoryBatchUpsertResponse(BaseModel):
    received: int
    inserted: int
    updated: int
    skipped: int
    errors: List[CalendarUpsertError]


class ModelScorecardOut(BaseModel):
    mode: str
    model_version: str
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    correct_count: int
    incorrect_count: int
    total_scored: int
    accuracy_pct: float


class ConsensusMatchEvaluationOut(BaseModel):
    match_id: UUID
    match_date: str
    home_team: str
    away_team: str
    tournament: Optional[str] = None
    actual_outcome: Literal["home_win", "away_win", "draw"]
    consensus_predicted_outcome: Literal["home_win", "away_win", "draw"]
    consensus_prob_home_win: float
    consensus_prob_draw: float
    consensus_prob_away_win: float
    is_correct: bool


class ModelScorecardMatchesResponse(BaseModel):
    mode: str
    model_version: str
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    verdict: Literal["all", "correct", "incorrect"]
    page: int
    page_size: int
    total: int
    matches: List[ConsensusMatchEvaluationOut]


class TopSearchedTeamOut(BaseModel):
    rank: int
    team: str
    searches: int


class TopSearchedTeamsSnapshotOut(BaseModel):
    mode: str
    snapshot_date: str
    lookback_days_used: int
    calculated_at: Optional[str] = None
    teams: List[TopSearchedTeamOut]


class PredictionRankingEntryOut(BaseModel):
    rank: int
    user_id: str
    display_name: str
    correct_count: int
    incorrect_count: int
    total_resolved_predictions: int
    accuracy_pct: float


class PredictionRankingsResponse(BaseModel):
    mode: Literal["all", "national", "champions", "libertadores"]
    sort_by: Literal["correct_count", "incorrect_count", "total_resolved_predictions", "accuracy_pct"]
    sort_order: Literal["asc", "desc"]
    page: int
    page_size: int
    total_users: int
    rankings: List[PredictionRankingEntryOut]
