from __future__ import annotations

from pathlib import Path
import unicodedata
import re

import pandas as pd

try:
    from backend.club_feature_engineering import ClubFeatureEngineer
    from backend.paths import (
        CLUB_COEFFICIENTS_HISTORY_PATH,
        CLUB_COEFFICIENTS_PATH,
        CLUB_TEAM_ALIASES_PATH,
        COUNTRY_COEFFICIENTS_HISTORY_PATH,
    )
except Exception:  # pragma: no cover - import path fallback
    from backend.club_feature_engineering import ClubFeatureEngineer
    from backend.paths import (
        CLUB_COEFFICIENTS_HISTORY_PATH,
        CLUB_COEFFICIENTS_PATH,
        CLUB_TEAM_ALIASES_PATH,
        COUNTRY_COEFFICIENTS_HISTORY_PATH,
    )


class ClubMatchDataPreprocessor:
    CLUB_NAME_TOKENS = {"fc", "cf", "sc", "afc", "fk", "ac", "ss", "sv", "as"}

    def __init__(self, file_path, verbose=False, include_uefa_coefficients=False):
        self.file_path = (
            f"data/{file_path}.csv"
            if not str(file_path).startswith("data/") and not str(file_path).endswith(".csv")
            else file_path
        )
        self.verbose = verbose
        self.include_uefa_coefficients = include_uefa_coefficients
        self.team_aliases = self._load_team_aliases()
        self.matches = None
        self.X_Full = None

    def log(self, msg):
        if self.verbose:
            print(msg)

    def _safe_numeric(self, value):
        return pd.to_numeric(value, errors="coerce")

    @staticmethod
    def _normalize_text(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        text = str(value).strip().lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return "".join(ch for ch in text if ch.isalnum())

    @staticmethod
    def _load_team_aliases() -> dict:
        alias_path = Path(CLUB_TEAM_ALIASES_PATH)
        if not alias_path.exists():
            return {}

        try:
            aliases = pd.read_csv(alias_path)
        except Exception:
            return {}
        if aliases.empty:
            return {}

        source_candidates = [
            "alias",
            "source_name",
            "source_team",
            "match_team",
            "from",
        ]
        target_candidates = [
            "canonical",
            "target_name",
            "target_team",
            "uefa_team",
            "to",
        ]
        source_col = next((col for col in source_candidates if col in aliases.columns), None)
        target_col = next((col for col in target_candidates if col in aliases.columns), None)
        if source_col is None or target_col is None:
            return {}

        alias_map = {}
        for _, row in aliases[[source_col, target_col]].iterrows():
            source_name = str(row.get(source_col, "") or "").strip()
            target_name = str(row.get(target_col, "") or "").strip()
            if not source_name or not target_name:
                continue
            source_key = ClubMatchDataPreprocessor._normalize_text(source_name)
            if source_key:
                alias_map[source_key] = target_name
        return alias_map

    def _canonical_team_name(self, value) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        team_name = str(value).strip()
        normalized = self._normalize_text(team_name)
        return self.team_aliases.get(normalized, team_name)

    def _team_name_key(self, value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        text = self._canonical_team_name(value).strip().lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        tokens = re.findall(r"[a-z0-9]+", text)
        if not tokens:
            return ""
        filtered = [tok for tok in tokens if tok not in self.CLUB_NAME_TOKENS]
        if filtered:
            return "".join(filtered)
        return "".join(tokens)

    @staticmethod
    def _uefa_season_year_from_date(date_series: pd.Series) -> pd.Series:
        dates = pd.to_datetime(date_series, errors="coerce")
        # UEFA season year follows season end year (e.g. Aug 2025 -> 2026).
        return dates.dt.year.where(dates.dt.month <= 6, dates.dt.year + 1).astype("Int64")

    def _to_match_level_from_team_rows(self, team_rows: pd.DataFrame) -> pd.DataFrame:
        records = []

        for row in team_rows.itertuples(index=False):
            team = getattr(row, "team", None)
            opponent = getattr(row, "opponent", None)
            venue = getattr(row, "venue", None)
            date = getattr(row, "date", None)
            competition = getattr(row, "competition", "")
            round_name = getattr(row, "round", "")
            country = getattr(row, "country", "")
            season = getattr(row, "season", "")
            source = getattr(row, "source", "fbref.com")
            source_file = getattr(row, "source_file", self.file_path)
            div_code = getattr(row, "div_code", "")
            result = getattr(row, "result", "")

            gf = self._safe_numeric(getattr(row, "gf", None))
            ga = self._safe_numeric(getattr(row, "ga", None))

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
                    "result": result,
                    "competition": competition,
                    "country": country,
                    "season": season,
                    "round": round_name,
                    "neutral": neutral,
                    "source": source,
                    "source_file": source_file,
                    "div_code": div_code,
                }
            )

        match_df = pd.DataFrame(records)
        if match_df.empty:
            return match_df

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

    def _attach_optional_uefa_coefficients(self, match_df: pd.DataFrame) -> pd.DataFrame:
        if not self.include_uefa_coefficients:
            return match_df

        history_coeff_path = Path(CLUB_COEFFICIENTS_HISTORY_PATH)
        coeff_path = history_coeff_path if history_coeff_path.exists() else Path(CLUB_COEFFICIENTS_PATH)
        if not coeff_path.exists():
            self.log("UEFA coefficients file not found; continuing without coefficients.")
            return match_df

        coeff = pd.read_csv(coeff_path)
        required = [
            "team",
            "overall_club_coefficient",
            "season_club_coefficient",
            "overall_rank",
            "season_rank",
        ]
        missing = [col for col in required if col not in coeff.columns]
        if missing:
            self.log(f"UEFA coefficients missing columns: {missing}. Skipping coefficient merge.")
            return match_df

        coeff = coeff.copy()
        if "country" not in coeff.columns:
            coeff["country"] = ""
        for col in required:
            if col not in coeff.columns:
                self.log(f"UEFA coefficients missing columns: {[col]}. Skipping coefficient merge.")
                return match_df

        name_cols = [col for col in ["team", "display_name", "official_name"] if col in coeff.columns]
        coeff["team_norm"] = coeff["team"].map(self._team_name_key)
        if "uefa_season_year" in coeff.columns:
            coeff["uefa_season_year"] = pd.to_numeric(coeff["uefa_season_year"], errors="coerce").astype("Int64")
        else:
            coeff["uefa_season_year"] = pd.NA

        coeff_variants = []
        for name_col in name_cols:
            tmp = coeff.copy()
            tmp["team_norm"] = tmp[name_col].map(self._team_name_key)
            tmp = tmp[tmp["team_norm"] != ""]
            coeff_variants.append(tmp)
        if coeff_variants:
            coeff = pd.concat(coeff_variants, ignore_index=True)
            dedupe_cols = ["team_norm"]
            if "uefa_season_year" in coeff.columns:
                dedupe_cols.append("uefa_season_year")
            coeff = coeff.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)

        home_coeff = coeff.rename(
            columns={
                "team_norm": "home_team_norm",
                "country": "home_team_country",
                "overall_club_coefficient": "home_team_uefa_overall_coefficient",
                "season_club_coefficient": "home_team_uefa_season_coefficient",
                "overall_rank": "home_team_uefa_overall_rank",
                "season_rank": "home_team_uefa_season_rank",
            }
        )
        away_coeff = coeff.rename(
            columns={
                "team_norm": "away_team_norm",
                "country": "away_team_country",
                "overall_club_coefficient": "away_team_uefa_overall_coefficient",
                "season_club_coefficient": "away_team_uefa_season_coefficient",
                "overall_rank": "away_team_uefa_overall_rank",
                "season_rank": "away_team_uefa_season_rank",
            }
        )

        out = match_df.copy()
        out["home_team_norm"] = out["home_team"].map(self._team_name_key)
        out["away_team_norm"] = out["away_team"].map(self._team_name_key)
        out["uefa_season_year"] = self._uefa_season_year_from_date(out["date"])

        home_keys = ["home_team_norm"]
        away_keys = ["away_team_norm"]
        if coeff["uefa_season_year"].notna().any():
            home_keys.append("uefa_season_year")
            away_keys.append("uefa_season_year")

        out = out.merge(
            home_coeff[
                home_keys
                + [
                    "home_team_country",
                    "home_team_uefa_overall_coefficient",
                    "home_team_uefa_season_coefficient",
                    "home_team_uefa_overall_rank",
                    "home_team_uefa_season_rank",
                ]
            ],
            on=home_keys,
            how="left",
        )
        out = out.merge(
            away_coeff[
                away_keys
                + [
                    "away_team_country",
                    "away_team_uefa_overall_coefficient",
                    "away_team_uefa_season_coefficient",
                    "away_team_uefa_overall_rank",
                    "away_team_uefa_season_rank",
                ]
            ],
            on=away_keys,
            how="left",
        )

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
        out["home_uefa_missing"] = (
            out["home_team_uefa_overall_coefficient"].isna()
            | out["home_team_uefa_season_coefficient"].isna()
            | out["home_team_uefa_overall_rank"].isna()
            | out["home_team_uefa_season_rank"].isna()
        ).astype(int)
        out["away_uefa_missing"] = (
            out["away_team_uefa_overall_coefficient"].isna()
            | out["away_team_uefa_season_coefficient"].isna()
            | out["away_team_uefa_overall_rank"].isna()
            | out["away_team_uefa_season_rank"].isna()
        ).astype(int)

        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].fillna(out[col].median())

        out["uefa_overall_coefficient_diff"] = (
            out["home_team_uefa_overall_coefficient"] - out["away_team_uefa_overall_coefficient"]
        )
        out["uefa_season_coefficient_diff"] = (
            out["home_team_uefa_season_coefficient"] - out["away_team_uefa_season_coefficient"]
        )
        out["uefa_overall_rank_diff"] = out["home_team_uefa_overall_rank"] - out["away_team_uefa_overall_rank"]
        out["uefa_season_rank_diff"] = out["home_team_uefa_season_rank"] - out["away_team_uefa_season_rank"]

        # Optional country-association context keyed by each team's association.
        country_coeff_path = Path(COUNTRY_COEFFICIENTS_HISTORY_PATH)
        if country_coeff_path.exists():
            country_coeff = pd.read_csv(country_coeff_path)
            required_country = [
                "country",
                "overall_country_coefficient",
                "season_country_coefficient",
                "overall_rank",
                "season_rank",
                "uefa_season_year",
            ]
            missing_country = [col for col in required_country if col not in country_coeff.columns]
            if not missing_country:
                country_coeff = country_coeff.copy()
                country_coeff["country_norm"] = country_coeff["country"].map(self._normalize_text)
                country_coeff["uefa_season_year"] = pd.to_numeric(
                    country_coeff["uefa_season_year"], errors="coerce"
                ).astype("Int64")
                out["home_country_norm"] = out["home_team_country"].map(self._normalize_text)
                out["away_country_norm"] = out["away_team_country"].map(self._normalize_text)

                home_country_coeff = country_coeff[
                    [
                        "country_norm",
                        "uefa_season_year",
                        "overall_country_coefficient",
                        "season_country_coefficient",
                        "overall_rank",
                        "season_rank",
                    ]
                ].rename(
                    columns={
                        "country_norm": "home_country_norm",
                        "overall_country_coefficient": "home_overall_country_coefficient",
                        "season_country_coefficient": "home_season_country_coefficient",
                        "overall_rank": "home_country_uefa_overall_rank",
                        "season_rank": "home_country_uefa_season_rank",
                    }
                )
                away_country_coeff = country_coeff[
                    [
                        "country_norm",
                        "uefa_season_year",
                        "overall_country_coefficient",
                        "season_country_coefficient",
                        "overall_rank",
                        "season_rank",
                    ]
                ].rename(
                    columns={
                        "country_norm": "away_country_norm",
                        "overall_country_coefficient": "away_overall_country_coefficient",
                        "season_country_coefficient": "away_season_country_coefficient",
                        "overall_rank": "away_country_uefa_overall_rank",
                        "season_rank": "away_country_uefa_season_rank",
                    }
                )

                out = out.merge(
                    home_country_coeff,
                    on=["home_country_norm", "uefa_season_year"],
                    how="left",
                )
                out = out.merge(
                    away_country_coeff,
                    on=["away_country_norm", "uefa_season_year"],
                    how="left",
                )

                country_cols = [
                    "home_overall_country_coefficient",
                    "home_season_country_coefficient",
                    "home_country_uefa_overall_rank",
                    "home_country_uefa_season_rank",
                    "away_overall_country_coefficient",
                    "away_season_country_coefficient",
                    "away_country_uefa_overall_rank",
                    "away_country_uefa_season_rank",
                ]
                home_country_missing = (
                    out["home_overall_country_coefficient"].isna()
                    | out["home_season_country_coefficient"].isna()
                    | out["home_country_uefa_overall_rank"].isna()
                    | out["home_country_uefa_season_rank"].isna()
                )
                away_country_missing = (
                    out["away_overall_country_coefficient"].isna()
                    | out["away_season_country_coefficient"].isna()
                    | out["away_country_uefa_overall_rank"].isna()
                    | out["away_country_uefa_season_rank"].isna()
                )
                out["country_uefa_missing"] = (
                    home_country_missing | away_country_missing
                ).astype(int)
                for col in country_cols:
                    out[col] = pd.to_numeric(out[col], errors="coerce")
                    out[col] = out[col].fillna(out[col].median())

                out["country_overall_coefficient_diff"] = (
                    out["home_overall_country_coefficient"] - out["away_overall_country_coefficient"]
                )
                out["country_season_coefficient_diff"] = (
                    out["home_season_country_coefficient"] - out["away_season_country_coefficient"]
                )
                out["country_overall_rank_diff"] = (
                    out["home_country_uefa_overall_rank"] - out["away_country_uefa_overall_rank"]
                )
                out["country_season_rank_diff"] = (
                    out["home_country_uefa_season_rank"] - out["away_country_uefa_season_rank"]
                )

                # Backward-compatible single country fields expected by existing templates.
                out["overall_country_coefficient"] = out["home_overall_country_coefficient"]
                out["season_country_coefficient"] = out["home_season_country_coefficient"]
                out["country_uefa_overall_rank"] = out["home_country_uefa_overall_rank"]
                out["country_uefa_season_rank"] = out["home_country_uefa_season_rank"]

        if self.verbose and {"home_uefa_missing", "away_uefa_missing"}.issubset(set(out.columns)):
            matched_rows = int(((out["home_uefa_missing"] == 0) & (out["away_uefa_missing"] == 0)).sum())
            total_rows = int(len(out))
            if total_rows > 0:
                self.log(
                    "UEFA team-coefficient match coverage: "
                    f"{matched_rows}/{total_rows} ({matched_rows / total_rows:.1%})"
                )
            missing_home = out.loc[out["home_uefa_missing"] == 1, "home_team"].value_counts().head(10)
            missing_away = out.loc[out["away_uefa_missing"] == 1, "away_team"].value_counts().head(10)
            if not missing_home.empty:
                self.log(
                    "Top unmatched home teams for UEFA coefficients: "
                    + ", ".join([f"{team} ({count})" for team, count in missing_home.items()])
                )
            if not missing_away.empty:
                self.log(
                    "Top unmatched away teams for UEFA coefficients: "
                    + ", ".join([f"{team} ({count})" for team, count in missing_away.items()])
                )

        drop_cols = [
            col
            for col in [
                "home_team_norm",
                "away_team_norm",
                "home_country_norm",
                "away_country_norm",
                "home_team_country",
                "away_team_country",
            ]
            if col in out.columns
        ]
        if drop_cols:
            out = out.drop(columns=drop_cols)

        return out

    def load_and_transform_data(self):
        raw = pd.read_csv(self.file_path)

        canonical_required = {"date", "home_team", "away_team", "home_score", "away_score"}
        legacy_required = {"date", "team", "opponent", "venue", "gf", "ga", "competition", "round"}

        if canonical_required.issubset(set(raw.columns)):
            df = raw.copy()
        elif legacy_required.issubset(set(raw.columns)):
            df = self._to_match_level_from_team_rows(raw)
        else:
            raise ValueError(
                "Club data must be canonical match-level or FBref team-centric format. "
                f"Got columns: {sorted(raw.columns.tolist())[:20]}"
            )

        if "competition" not in df.columns:
            df["competition"] = ""
        if "round" not in df.columns:
            df["round"] = ""
        if "neutral" not in df.columns:
            df["neutral"] = 0
        if "country" not in df.columns:
            df["country"] = ""
        if "season" not in df.columns:
            parsed_dates = pd.to_datetime(df["date"], errors="coerce")
            season_start = parsed_dates.dt.year.where(parsed_dates.dt.month >= 7, parsed_dates.dt.year - 1)
            df["season"] = season_start.astype("Int64").astype(str) + "-" + (season_start + 1).astype("Int64").astype(str)
        if "source" not in df.columns:
            df["source"] = ""
        if "source_file" not in df.columns:
            df["source_file"] = self.file_path
        if "div_code" not in df.columns:
            df["div_code"] = ""
        if "result" not in df.columns:
            df["result"] = ""

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
        df = df.dropna(subset=["date", "home_score", "away_score"]).copy()

        self.matches = self._attach_optional_uefa_coefficients(df)
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

        y = df.apply(self.get_match_outcome, axis=1).astype(int)
        assert all(y.isin([0, 1, 2])), "Invalid club match outcome detected."

        self.X_Full = df.copy()

        columns_to_drop = [
            "date",
            "home_score",
            "away_score",
            "home_team",
            "away_team",
            "competition",
            "country",
            "season",
            "round",
            "source",
            "source_file",
            "div_code",
            "result",
            "matchup_id",
            "goal_diff",
        ]
        X = df.drop(columns=[col for col in columns_to_drop if col in df.columns]).copy()

        # Keep only numeric model features for robust training/inference compatibility.
        non_numeric_cols = X.select_dtypes(include=["object", "datetime64[ns]"]).columns.tolist()
        if non_numeric_cols:
            X = X.drop(columns=non_numeric_cols)

        bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
        for col in bool_cols:
            X[col] = X[col].astype(int)

        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return X, y

    def preprocess(self):
        self.load_and_transform_data()
        feature_engineer = ClubFeatureEngineer(self.matches, verbose=self.verbose)
        self.matches = feature_engineer.generate_features()
        return self.finalize_dataset()
