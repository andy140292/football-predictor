import pandas as pd

def get_head_to_head(matches, home_team, away_team):
    matches = matches[((matches['home_team'] == home_team) & (matches['away_team'] == away_team)) & (matches['tournament'] == 'FIFA World Cup qualification')]
    # matches = matches[((matches['home_team'] == home_team) & (matches['away_team'] == away_team))]
    matches = matches.sort_values(by='date')
    # fixing indices to start at 0
    filtered_matches = matches[['date', 'home_team', 'home_score', 'away_score', 'away_team']]
    filtered_matches = filtered_matches.reset_index(drop=True)
    return filtered_matches

def get_to_head_summary(matches):
    home_wins = matches[matches['home_score'] > matches['away_score']].shape[0]
    away_wins = matches[matches['home_score'] < matches['away_score']].shape[0]
    draws = matches[matches['home_score'] == matches['away_score']].shape[0]
    home_goals = matches['home_score'].sum()
    away_goals = matches['away_score'].sum()
    return home_wins, away_wins, draws, home_goals, away_goals


all_matches = pd.read_csv('data/results.csv')
home_team = input('Enter home team name: ')
away_team = input('Enter away team name: ')
head_to_head = get_head_to_head(all_matches, home_team, away_team)
home_wins, away_wins, draws, home_goals, away_goals = get_to_head_summary(head_to_head)

print(f"Head-to-Head Summary for {home_team} vs {away_team}:\nVictorias {home_team}: {home_wins}\nVictorias {away_team} {away_wins}\nEmpates: {draws}\nGoles {home_team}: {home_goals}\nGoles {away_team}: {away_goals}")

print(head_to_head) 