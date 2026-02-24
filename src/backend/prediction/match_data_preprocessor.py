
import pandas as pd
import os
import gzip
from pathlib import Path
try:
    from backend.paths import DATA_DIR, HISTORICAL_DATA_DIR, RANKING_PATH
    from prediction.feature_engineering import FeatureEngineer
    from backend.supabase_client import get_supabase_client
except Exception:  # pragma: no cover - import path fallback
    from backend.paths import DATA_DIR, HISTORICAL_DATA_DIR, RANKING_PATH
    from backend.prediction.feature_engineering import FeatureEngineer
    try:
        from backend.supabase_client import get_supabase_client
    except Exception:  # pragma: no cover - import path fallback
        from src.backend.supabase_client import get_supabase_client

supabase = get_supabase_client()


def _resolve_local_matches_csv(file_path: str) -> str:
    raw = str(file_path).strip()
    if not raw:
        return raw

    candidate = Path(raw)
    if candidate.exists():
        return str(candidate)

    name = raw if raw.endswith(".csv") else f"{raw}.csv"
    named_candidate = Path(name)
    if named_candidate.exists():
        return str(named_candidate)

    search_dirs = [Path(DATA_DIR), Path("data"), Path(HISTORICAL_DATA_DIR), Path("preprocessing_data")]
    for base in search_dirs:
        path = base / name
        if path.exists():
            return str(path)

    return str(Path("data") / name)

class MatchDataPreprocessor:
    def __init__(self, file_path, from_supabase: bool = False, verbose=False):
        self.file_path = file_path
        self.verbose = verbose
        self.matches = None

        if from_supabase:
            self.file_path = self._download_and_extract_from_supabase(file_path)
        else:
            self.file_path = _resolve_local_matches_csv(file_path)

    def log(self, msg):
        if self.verbose:
            print(msg)

    def _download_and_extract_from_supabase(self, remote_filename: str, local_dir="data"):
        os.makedirs(local_dir, exist_ok=True)

        # Archivo .csv.gz en Supabase
        supabase_path = f"{remote_filename}.csv.gz"
        local_gz_path = os.path.join(local_dir, supabase_path)
        local_csv_path = os.path.join(local_dir, f"{remote_filename}.csv")

        # Descargar desde Supabase
        with open(local_gz_path, "wb") as f:
            content = supabase.storage.from_("match-datasets").download(supabase_path)
            f.write(content)

        # Descomprimir
        with gzip.open(local_gz_path, "rb") as f_in:
            with open(local_csv_path, "wb") as f_out:
                f_out.write(f_in.read())

        print(f"✅ Archivo descargado y descomprimido en: {local_csv_path}")
        return local_csv_path

    def load_and_filter_data(self):
        self.matches = pd.read_csv(self.file_path)

        required_cols = ["date", "home_score", "away_score", "tournament"]
        missing = [col for col in required_cols if col not in self.matches.columns]
        if missing:
            raise ValueError(f"Missing columns in data: {missing}")

        self.matches = self.matches[
            self.matches["tournament"].isin([
                "FIFA World Cup qualification",
                "Copa America",
                "FIFA World Cup",
                "Friendly"
            ])
        ]

        self.matches["date"] = pd.to_datetime(self.matches["date"], errors="coerce")
        self.matches = self.matches[self.matches["date"] >= "2000-01-01"]

        self.log(f"Loaded {len(self.matches)} matches after filtering.")

    def get_match_outcome(self, row):
        if row['home_score'] > row['away_score']:
            return 2  # Home Win
        elif row['home_score'] < row['away_score']:
            return 0  # Away Win
        else:
            return 1  # Draw

    def finalize_dataset(self):
        df = self.matches.copy()
        df = df.sort_values("date").dropna().reset_index(drop=True)
        y = df.apply(self.get_match_outcome, axis=1)
        assert all(y.isin([0, 1, 2])), "Invalid match outcome detected."

        columns_to_drop = [
            "date", "home_score", "away_score", "tournament",
            "city", "country", "matchup_id", "goal_diff",
            "home_team_fifa_rank", "home_team_fifa_points",
            "away_team_fifa_rank", "away_team_fifa_points",
        ]

        self.X_Full = df
        X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        print(X.columns)
        return X, y

    def preprocess(self):
        self.load_and_filter_data()
        feature_engineer = FeatureEngineer(self.matches)
        self.matches = feature_engineer.generate_features()

        return self.finalize_dataset()
