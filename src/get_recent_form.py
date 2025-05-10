import pandas as pd

def get_recent_form(matches, team):
    matches = matches[((matches['home_team'] == team) | (matches['away_team'] == team)) & (matches['tournament'] == 'FIFA World Cup qualification')]
    matches = matches.sort_values(by='date', ascending=False)
    matches = matches.head(5)
    # fixing indices to start at 0
    matches = matches.sort_values(by='date')
    filtered_matches = matches[['date', 'home_team', 'home_score', 'away_score', 'away_team']]
    filtered_matches = filtered_matches.reset_index(drop=True)   
    return filtered_matches


all_matches = pd.read_csv('data/results.csv')
team = input('Enter team name: ')
recent_form = get_recent_form(all_matches, team)

print(recent_form)