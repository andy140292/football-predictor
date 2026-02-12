import pandas as pd


class ClubFeatureEngineer:
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
        if "goal_diff" not in self.df.columns:
            self.calculate_goal_diff()

        # Historical-only H2H: for each row, use prior matches in the same matchup.
        self.df["h2h_matches_before"] = self.df.groupby("matchup_id").cumcount()
        self.df["head_to_head_goal_diff"] = self.df.groupby("matchup_id")["goal_diff"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        self.df["head_to_head_goal_diff"] = self.df["head_to_head_goal_diff"].fillna(0)

    def add_context_flags(self):
        if "neutral" not in self.df.columns:
            self.df["neutral"] = 0
        self.df["h2h_available"] = self.df["h2h_matches_before"].gt(0).astype(int)
        self.df["is_ucl_match"] = (self.df["competition"] == "Champions Lg").astype(int)
        self.df["is_knockout_playoff"] = (
            self.df["round"].str.lower() == "knockout phase play-offs"
        ).astype(int)

    def calculate_rolling_averages(self, window_size=10):
        # Mirror the national pipeline behavior: form of home teams in home matches
        # and away teams in away matches with shift to avoid leakage.
        self.df["home_team_avg_scored"] = self.df.groupby("home_team")["home_score"].transform(
            lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
        )
        self.df["home_team_avg_conceded"] = self.df.groupby("home_team")["away_score"].transform(
            lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
        )

        self.df["away_team_avg_scored"] = self.df.groupby("away_team")["away_score"].transform(
            lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
        )
        self.df["away_team_avg_conceded"] = self.df.groupby("away_team")["home_score"].transform(
            lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
        )

        rolling_cols = [
            "home_team_avg_scored",
            "home_team_avg_conceded",
            "away_team_avg_scored",
            "away_team_avg_conceded",
        ]
        self.df[rolling_cols] = self.df[rolling_cols].fillna(0)

    def encode_teams(self):
        self.df = pd.get_dummies(self.df, columns=["home_team", "away_team"])

    def encode_competition_context(self):
        self.df = pd.get_dummies(self.df, columns=["competition", "round"])

    def get_engineered_data(self):
        return self.df

    def generate_features(self):
        # Enforce temporal ordering to avoid leakage in rolling/history features.
        self.df = self.df.sort_values("date").reset_index(drop=True)
        self.calculate_goal_diff()
        self.add_matchup_id()
        self.compute_head_to_head_stats()
        self.add_context_flags()
        self.calculate_rolling_averages()
        self.encode_teams()
        self.encode_competition_context()
        return self.df
