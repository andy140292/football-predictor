import pandas as pd
from utils.paths import RANKING_PATH

class FeatureEngineer:
    def __init__(self, matches_df, verbose=False):
        self.df = matches_df.copy()
        self.verbose = verbose

    def log(self, msg):
        if self.verbose:
            print(msg)

    def calculate_goal_diff(self):
        self.df["goal_diff"] = self.df["home_score"] - self.df["away_score"]

    def add_matchup_id(self):
        self.df["matchup_id"] = self.df.apply(
            lambda row: "_".join(sorted([row["home_team"], row["away_team"]])), axis=1
        )

    def compute_head_to_head_stats(self):
         
        # Calculate goal difference for each match (from the perspective of the listed home team)
        self.df['goal_diff'] = self.df['home_score'] - self.df['away_score']

        # Group by matchup_id and compute the average goal difference
        head_to_head_stats = self.df.groupby('matchup_id')['goal_diff'].mean().reset_index()
        head_to_head_stats.rename(
            columns={'goal_diff': 'head_to_head_goal_diff'}, inplace=True
        )

        """Merges head-to-head statistics into the dataset."""
        self.df = pd.merge(self.df, head_to_head_stats, on='matchup_id', how='left')

        # Fill missing values (for matchups with no history)
        self.df['head_to_head_goal_diff'] = self.df['head_to_head_goal_diff'].fillna(0)


    def add_context_flags(self):
        self.df["same_confederation"] = (
            self.df["home_team_confederation"] == self.df["away_team_confederation"]
        ).astype(int)
        self.df["h2h_available"] = self.df["head_to_head_goal_diff"].ne(0).astype(int)

    def calculate_rolling_averages(self, window_size=10):
        self.df['home_team_avg_scored'] = self.df.groupby('home_team')['home_score'] \
            .transform(lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())
        self.df['home_team_avg_conceded'] = self.df.groupby('home_team')['away_score'] \
            .transform(lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())

        self.df['away_team_avg_scored'] = self.df.groupby('away_team')['away_score'] \
            .transform(lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())
        self.df['away_team_avg_conceded'] = self.df.groupby('away_team')['home_score'] \
            .transform(lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())

    def calculate_fifa_ranking_diffs(self):
        # Asegúrate de que las columnas estén presentes
        required_cols = [
            "home_team_fifa_rank", "away_team_fifa_rank",
            "home_team_fifa_points", "away_team_fifa_points"
        ]
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en el dataset: {missing}")

        self.df["fifa_rank_diff"] = self.df["home_team_fifa_rank"] - self.df["away_team_fifa_rank"]
        self.df["fifa_points_diff"] = self.df["home_team_fifa_points"] - self.df["away_team_fifa_points"]

    def encode_teams(self):
        """Converts categorical team names into dummy variables (one-hot encoding)."""
        self.df = pd.get_dummies(self.df, columns=['home_team', 'away_team'])

    def encode_confederations(self):
        """Converts categorical team names into dummy variables (one-hot encoding)."""
        self.df = pd.get_dummies(self.df, columns=['home_team_confederation', 'away_team_confederation'])        


    def get_engineered_data(self):
        return self.df

    def generate_features(self):
        self.calculate_goal_diff()
        self.add_matchup_id()
        self.compute_head_to_head_stats()
        self.add_context_flags()
        self.calculate_rolling_averages()
        self.calculate_fifa_ranking_diffs()
        self.encode_teams()
        self.encode_confederations()
        return self.df