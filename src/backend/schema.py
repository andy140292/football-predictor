from pydantic import BaseModel

class MatchInput(BaseModel):
    home_team: str
    away_team: str