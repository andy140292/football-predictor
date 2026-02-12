import pandas as pd

from prediction.club_feature_engineering import ClubFeatureEngineer
from utils.paths import CLUB_COEFFICIENTS_PATH


class ClubMatchDataPreprocessor:
    def __init__(self, file_path, verbose=False):
        self.file_path = (
            f"data/{file_path}.csv"
            if not str(file_path).startswith("data/") and not str(file_path).endswith(".csv")
            else file_path
        )
        self.verbose = verbose
        self.matches = None
        self.X_Full = None

    def log(self, msg):
        if self.verbose:
            print(msg)

    def _safe_numeric(self, value):
        return pd.to_numeric(value, errors="coerce")

    def _attach_uefa_coefficients(self, match_df: pd.DataFrame) -> pd.DataFrame:
        coeff = pd.read_csv(CLUB_COEFFICIENTS_PATH)
        required = [
            "team",
            "overall_club_coefficient",
            "season_club_coefficient",
            "overall_rank",
            "season_rank",
        ]
        missing = [col for col in required if col not in coeff.columns]
        if missing:
            raise ValueError(f"Missing columns in UEFA coefficients data: {missing}")

        coeff = coeff[required].copy()

        home_coeff = coeff.rename(
            columns={
                "team": "home_team",
                "overall_club_coefficient": "home_team_uefa_overall_coefficient",
                "season_club_coefficient": "home_team_uefa_season_coefficient",
                "overall_rank": "home_team_uefa_overall_rank",
                "season_rank": "home_team_uefa_season_rank",
            }
        )
        away_coeff = coeff.rename(
            columns={
                "team": "away_team",
                "overall_club_coefficient": "away_team_uefa_overall_coefficient",
                "season_club_coefficient": "away_team_uefa_season_coefficient",
                "overall_rank": "away_team_uefa_overall_rank",
                "season_rank": "away_team_uefa_season_rank",
            }
        )

        out = match_df.merge(home_coeff, on="home_team", how="left")
        out = out.merge(away_coeff, on="away_team", how="left")

        numeric_cols = [
            "home_team_uefa_overall_coefficient",
            "home_team_uefa_season_coefficient",
            "home_team_uefa_overall_rank",
            "home_team_uefa_season_rank",
            "away_team_uefa_overall_coefficient",
            "away_team_uefa_season_coefficient",
            "away_team_uefa_overall_rank",
            "away_team_uefa_season_rank",
        ]
        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].fillna(out[col].median())

        out["uefa_overall_coefficient_diff"] = (
            out["home_team_uefa_overall_coefficient"] - out["away_team_uefa_overall_coefficient"]
        )
        out["uefa_season_coefficient_diff"] = (
            out["home_team_uefa_season_coefficient"] - out["away_team_uefa_season_coefficient"]
        )
        out["uefa_overall_rank_diff"] = (
            out["home_team_uefa_overall_rank"] - out["away_team_uefa_overall_rank"]
        )
        out["uefa_season_rank_diff"] = (
            out["home_team_uefa_season_rank"] - out["away_team_uefa_season_rank"]
        )

        return out

    def _to_match_level(self, team_rows: pd.DataFrame) -> pd.DataFrame:
        records = []

        for row in team_rows.itertuples(index=False):
            team = getattr(row, "team")
            opponent = getattr(row, "opponent")
            venue = getattr(row, "venue")
            date = getattr(row, "date")
            competition = getattr(row, "competition")
            round_name = getattr(row, "round")
            gf = self._safe_numeric(getattr(row, "gf"))
            ga = self._safe_numeric(getattr(row, "ga"))

            if pd.isna(team) or pd.isna(opponent) or pd.isna(venue):
                continue

            venue_norm = str(venue).strip().lower()
            team = str(team).strip()
            opponent = str(opponent).strip()

            if venue_norm == "home":
                home_team, away_team = team, opponent
                home_score, away_score = gf, ga
                neutral = 0
            elif venue_norm == "away":
                home_team, away_team = opponent, team
                home_score, away_score = ga, gf
                neutral = 0
            else:
                # Deterministic assignment for neutral sites avoids duplicate polarity.
                ordered = sorted([team, opponent])
                home_team, away_team = ordered[0], ordered[1]
                if team == home_team:
                    home_score, away_score = gf, ga
                else:
                    home_score, away_score = ga, gf
                neutral = 1

            records.append(
                {
                    "date": date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": home_score,
                    "away_score": away_score,
                    "competition": competition,
                    "round": round_name,
                    "neutral": neutral,
                }
            )

        match_df = pd.DataFrame(records)
        if match_df.empty:
            return match_df

        # Drop mirrored duplicates for matches where both teams are in the source table.
        dedupe_cols = [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "competition",
            "round",
            "neutral",
        ]
        match_df = (
            match_df.sort_values("date")
            .drop_duplicates(subset=dedupe_cols, keep="first")
            .reset_index(drop=True)
        )

        return match_df

    def load_and_transform_data(self):
        raw = pd.read_csv(self.file_path)

        required_cols = ["date", "team", "opponent", "venue", "gf", "ga", "competition", "round"]
        missing = [col for col in required_cols if col not in raw.columns]
        if missing:
            raise ValueError(f"Missing columns in club data: {missing}")

        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna(subset=["date"]).copy()

        self.matches = self._to_match_level(raw)
        self.matches = self._attach_uefa_coefficients(self.matches)
        self.log(f"Loaded {len(self.matches)} club matches after transformation.")

    def get_match_outcome(self, row):
        if row["home_score"] > row["away_score"]:
            return 2  # Home Win
        if row["home_score"] < row["away_score"]:
            return 0  # Away Win
        return 1  # Draw

    def finalize_dataset(self):
        df = self.matches.copy()
        df = df.sort_values("date").reset_index(drop=True)

        # Only use completed matches for training/validation targets.
        completed_mask = df["home_score"].notna() & df["away_score"].notna()
        df = df.loc[completed_mask].copy()

        y = df.apply(self.get_match_outcome, axis=1)
        assert all(y.isin([0, 1, 2])), "Invalid club match outcome detected."

        columns_to_drop = [
            "date",
            "home_score",
            "away_score",
            "matchup_id",
            "goal_diff",
        ]

        self.X_Full = df
        X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        return X, y

    def preprocess(self):
        self.load_and_transform_data()
        feature_engineer = ClubFeatureEngineer(self.matches, verbose=self.verbose)
        self.matches = feature_engineer.generate_features()
        return self.finalize_dataset()
