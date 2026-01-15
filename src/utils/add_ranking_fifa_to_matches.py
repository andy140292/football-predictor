import pandas as pd

# --- Manual name normalization ---
team_name_map = {
    "IR Iran": "Iran",
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Czechia": "Czech Republic",
    "Cape Verde Islands": "Cabo Verde",
    "DR Congo": "Congo DR",
    "Curaçao": "Curacao",
    "UAE": "United Arab Emirates",
    "St. Kitts and Nevis": "St Kitts and Nevis",
    "St. Vincent / Grenadines": "St Vincent and the Grenadines",
    "São Tomé e Príncipe": "Sao Tome and Principe",
    "Timor-Leste": "Timor Leste",
    "Russia": "Russian Federation",  # example if FIFA uses full names
    # Add more known mappings here
}

def add_ranking_fifa_to_matches(matches, fifa):
    matches_copy = matches.copy()
    fifa_copy = fifa.copy()

    # Apply to both dataframes
    matches_copy["home_team"] = matches_copy["home_team"].replace(team_name_map)
    matches_copy["away_team"] = matches_copy["away_team"].replace(team_name_map)
    fifa_copy["team"] = fifa_copy["team"].replace(team_name_map)

    home_rank = fifa_copy.rename(columns={
        "team": "home_team",
        "ranking": "home_team_fifa_rank",
        "points": "home_team_fifa_points"
    })
    away_rank = fifa_copy.rename(columns={
        "team": "away_team",
        "ranking": "away_team_fifa_rank",
        "points": "away_team_fifa_points"
    })

    matches_copy = matches_copy.merge(home_rank[["home_team", "home_team_fifa_rank", "home_team_fifa_points"]], on="home_team", how="left")
    matches_copy = matches_copy.merge(away_rank[["away_team", "away_team_fifa_rank", "away_team_fifa_points"]], on="away_team", how="left")

    # --- Fill missing values if any ---
    matches_copy["home_team_fifa_rank"] = matches_copy["home_team_fifa_rank"].fillna(110)
    matches_copy["away_team_fifa_rank"] = matches_copy["away_team_fifa_rank"].fillna(110)
    matches_copy["home_team_fifa_points"] = matches_copy["home_team_fifa_points"].fillna(1000)
    matches_copy["away_team_fifa_points"] = matches_copy["away_team_fifa_points"].fillna(1000)

    return matches_copy
