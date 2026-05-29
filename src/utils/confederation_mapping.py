import pandas as pd
from pathlib import Path


DEFAULT_CONFED_PATH = Path(__file__).resolve().parents[2] / "preprocessing_data" / "unique_countries.csv"


def add_confederation_to_matches(matches, confed_path=None):
    matches_copy = matches.copy()
    confed_copy = pd.read_csv(confed_path or DEFAULT_CONFED_PATH)
    # Merge to get home_team_confederation
    matches_copy = matches_copy.merge(
        confed_copy.rename(columns={'country': 'home_team', 'confederation': 'home_team_confederation'}),
        on='home_team',
        how='left'
    )

    # Merge to get away_team_confederation
    matches_copy = matches_copy.merge(
        confed_copy.rename(columns={'country': 'away_team', 'confederation': 'away_team_confederation'}),
        on='away_team',
        how='left'
    )
    return matches_copy
