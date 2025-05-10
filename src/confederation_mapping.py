import pandas as pd

# Load the match data
matches = pd.read_csv('data/results.csv')

# Load confederation mapping
confed_map = pd.read_csv('data/unique_countries.csv')

# Merge to get home_team_confederation
matches = matches.merge(
    confed_map.rename(columns={'country': 'home_team', 'confederation': 'home_team_confederation'}),
    on='home_team',
    how='left'
)

# Merge to get away_team_confederation
matches = matches.merge(
    confed_map.rename(columns={'country': 'away_team', 'confederation': 'away_team_confederation'}),
    on='away_team',
    how='left'
)

# Save final cleaned result
matches.to_csv('data/matches.csv', index=False)
