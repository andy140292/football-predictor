import pandas as pd

# Load the match data
matches = pd.read_csv('data/results.csv')

# Load confederation mapping
CONFED_MAP = pd.read_csv('data/unique_countries.csv')

def add_confederation_to_matches(matches):
    matches_copy = matches.copy()
    confed_copy = CONFED_MAP.copy()
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
