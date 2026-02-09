from pydantic import BaseModel, Field
from typing import List, Optional

class MatchInput(BaseModel):
    home_team: str
    away_team: str


class RecentFormInput(BaseModel):
    home_team: str
    away_team: str
    last_matches: int = Field(gt=0, le=50)


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
