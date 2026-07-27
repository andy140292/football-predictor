import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, log_loss
from datetime import datetime
try:
    from backend.prediction.match_data_preprocessor import MatchDataPreprocessor
    from backend.prediction.club_match_data_preprocessor import ClubMatchDataPreprocessor
    from backend.prediction.football_match_predictor import FootballMatchPredictor
    from backend.prediction.libertadores_match_source import fetch_libertadores_matches
    from backend.prediction.national_match_source import fetch_national_matches
    from backend.paths import (
        CHAMPIONS_STATE_SNAPSHOT_PATH,
        CHAMPIONS_MODEL_PATHS,
        CHAMPIONS_PROCESSED_X_FULL_PATH,
        CHAMPIONS_PROCESSED_X_PATH,
        CHAMPIONS_PROCESSED_Y_PATH,
        CLUB_STATE_SNAPSHOT_PATH,
        CLUB_MATCHES_PATH,
        CLUB_MATCHES_HISTORY_PATH,
        CLUB_MODEL_PATHS,
        CLUB_PROCESSED_X_FULL_PATH,
        CLUB_PROCESSED_X_PATH,
        CLUB_PROCESSED_Y_PATH,
        LIBERTADORES_COEFFICIENTS_PATH,
        LIBERTADORES_MODEL_PATHS,
        LIBERTADORES_PROCESSED_X_FULL_PATH,
        LIBERTADORES_PROCESSED_X_PATH,
        LIBERTADORES_PROCESSED_Y_PATH,
        LIBERTADORES_STATE_SNAPSHOT_PATH,
        MODEL_PATHS,
        PROCESSED_X_PATH,
        PROCESSED_X__FULL_PATH,
        PROCESSED_y_PATH,
    )
except Exception:  # pragma: no cover - import path fallback
    from backend.prediction.match_data_preprocessor import MatchDataPreprocessor
    from backend.prediction.club_match_data_preprocessor import ClubMatchDataPreprocessor
    from backend.prediction.football_match_predictor import FootballMatchPredictor
    from backend.prediction.libertadores_match_source import fetch_libertadores_matches
    from backend.prediction.national_match_source import fetch_national_matches
    from backend.paths import (
        CHAMPIONS_STATE_SNAPSHOT_PATH,
        CHAMPIONS_MODEL_PATHS,
        CHAMPIONS_PROCESSED_X_FULL_PATH,
        CHAMPIONS_PROCESSED_X_PATH,
        CHAMPIONS_PROCESSED_Y_PATH,
        CLUB_STATE_SNAPSHOT_PATH,
        CLUB_MATCHES_PATH,
        CLUB_MATCHES_HISTORY_PATH,
        CLUB_MODEL_PATHS,
        CLUB_PROCESSED_X_FULL_PATH,
        CLUB_PROCESSED_X_PATH,
        CLUB_PROCESSED_Y_PATH,
        LIBERTADORES_COEFFICIENTS_PATH,
        LIBERTADORES_MODEL_PATHS,
        LIBERTADORES_PROCESSED_X_FULL_PATH,
        LIBERTADORES_PROCESSED_X_PATH,
        LIBERTADORES_PROCESSED_Y_PATH,
        LIBERTADORES_STATE_SNAPSHOT_PATH,
        MODEL_PATHS,
        PROCESSED_X_PATH,
        PROCESSED_X__FULL_PATH,
        PROCESSED_y_PATH,
    )
try:
    from backend.club_feature_engineering import ClubFeatureEngineer
except Exception:  # pragma: no cover - import path fallback
    from src.backend.club_feature_engineering import ClubFeatureEngineer
try:
    from data.build_club_historical_dataset import ensure_club_historical_dataset
except Exception:  # pragma: no cover - import path fallback
    try:
        from src.data.build_club_historical_dataset import ensure_club_historical_dataset
    except Exception:  # pragma: no cover - optional dependency in slim/runtime setups
        ensure_club_historical_dataset = None

CLUB_BASELINE_LOG_LOSS = float(os.getenv("CLUB_BASELINE_LOG_LOSS", "1.10"))
CHAMPIONS_BASELINE_LOG_LOSS = float(
    os.getenv("CHAMPIONS_BASELINE_LOG_LOSS", str(CLUB_BASELINE_LOG_LOSS))
)
CLUB_INCLUDE_UEFA_COEFFICIENTS = os.getenv("CLUB_INCLUDE_UEFA_COEFFICIENTS", "1").lower() not in {
    "0",
    "false",
    "no",
}
CLUB_MIN_DATE = os.getenv("CLUB_MIN_DATE", "").strip()
CLUB_FEATURE_SELECTION_ENABLED = os.getenv("CLUB_FEATURE_SELECTION_ENABLED", "1").lower() not in {
    "0",
    "false",
    "no",
}
CLUB_FEATURE_SELECTION_MODEL = os.getenv("CLUB_FEATURE_SELECTION_MODEL", "logistic_regression")
CLUB_FEATURE_SELECTION_STEP = int(os.getenv("CLUB_FEATURE_SELECTION_STEP", "5"))
CLUB_FEATURE_SELECTION_MIN_FEATURES = int(os.getenv("CLUB_FEATURE_SELECTION_MIN_FEATURES", "20"))
CLUB_FEATURE_SELECTION_PERM_REPEATS = int(os.getenv("CLUB_FEATURE_SELECTION_PERM_REPEATS", "2"))
CLUB_FEATURE_SELECTION_MIN_IMPROVEMENT = float(os.getenv("CLUB_FEATURE_SELECTION_MIN_IMPROVEMENT", "0.0005"))
CLUB_UCL_UPSAMPLE_FACTOR = int(os.getenv("CLUB_UCL_UPSAMPLE_FACTOR", "1"))
CLUB_ROLLOUT_GATE_ENABLED = os.getenv("CLUB_ROLLOUT_GATE_ENABLED", "1").lower() not in {
    "0",
    "false",
    "no",
}
CLUB_FORCE_SAVE_MODELS = os.getenv("CLUB_FORCE_SAVE_MODELS", "0").lower() in {
    "1",
    "true",
    "yes",
}
CLUB_PROTECTED_FEATURE_TOKENS = (
    "uefa",
    "country_coefficient",
)

def save_model_locally(predictor, model_type, mode="national"):
    # Save model locally
    if mode == "national":
        model_paths = MODEL_PATHS
    elif mode == "club":
        model_paths = CLUB_MODEL_PATHS
    elif mode == "champions":
        model_paths = CHAMPIONS_MODEL_PATHS
    elif mode == "libertadores":
        model_paths = LIBERTADORES_MODEL_PATHS
    else:
        raise ValueError(f"Modo inválido: {mode}")
    local_model_path = model_paths[model_type]
    joblib.dump(predictor, local_model_path)
    print(f"✅ Modelo {model_type} guardado localmente en {local_model_path}")

def save_processed_data_to_csv(X, y, X_full, mode="national"):
    if mode == "national":
        x_path = PROCESSED_X_PATH
        y_path = PROCESSED_y_PATH
        x_full_path = PROCESSED_X__FULL_PATH
    elif mode == "club":
        x_path = CLUB_PROCESSED_X_PATH
        y_path = CLUB_PROCESSED_Y_PATH
        x_full_path = CLUB_PROCESSED_X_FULL_PATH
    elif mode == "champions":
        x_path = CHAMPIONS_PROCESSED_X_PATH
        y_path = CHAMPIONS_PROCESSED_Y_PATH
        x_full_path = CHAMPIONS_PROCESSED_X_FULL_PATH
    elif mode == "libertadores":
        x_path = LIBERTADORES_PROCESSED_X_PATH
        y_path = LIBERTADORES_PROCESSED_Y_PATH
        x_full_path = LIBERTADORES_PROCESSED_X_FULL_PATH
    else:
        raise ValueError(f"Modo inválido: {mode}")

    X.to_csv(x_path, index=False)
    y.to_csv(y_path, index=False)
    X_full.to_csv(x_full_path, index=False)

def train_model(model_type, X, y, mode="national", save_model=True):
    predictor = FootballMatchPredictor(model_type=model_type)
    predictor.train(X, y)


    # Save model locally
    if save_model:
        save_model_locally(predictor, model_type, mode=mode)
        print(f"✅ Modelo {model_type} entrenado y guardado.")
    else:
        print(f"✅ Modelo {model_type} entrenado (pendiente de gate de despliegue).")

    # Store best estimator if needed
    best_model = (
        predictor.model.best_estimator_
        if hasattr(predictor.model, "best_estimator_")
        else predictor.model
    )
    return predictor, best_model

def _club_time_split_train_val_test(X, y, X_full):
    if "date" not in X_full.columns:
        n = len(X)
        train_end = max(1, int(n * 0.7))
        val_end = max(train_end + 1, int(n * 0.85))
        return (
            X.iloc[:train_end].copy(),
            X.iloc[train_end:val_end].copy(),
            X.iloc[val_end:].copy(),
            y.iloc[:train_end].copy(),
            y.iloc[train_end:val_end].copy(),
            y.iloc[val_end:].copy(),
        )

    dates = pd.to_datetime(X_full["date"], errors="coerce")
    train_mask = dates <= pd.Timestamp("2023-06-30")
    val_mask = (dates > pd.Timestamp("2023-06-30")) & (dates <= pd.Timestamp("2024-06-30"))
    test_mask = dates > pd.Timestamp("2024-06-30")

    # Ensure robust fallback for sparse datasets.
    if train_mask.sum() < 200 or val_mask.sum() < 100 or test_mask.sum() < 100:
        n = len(X)
        train_end = max(1, int(n * 0.7))
        val_end = max(train_end + 1, int(n * 0.85))
        return (
            X.iloc[:train_end].copy(),
            X.iloc[train_end:val_end].copy(),
            X.iloc[val_end:].copy(),
            y.iloc[:train_end].copy(),
            y.iloc[train_end:val_end].copy(),
            y.iloc[val_end:].copy(),
        )

    return (
        X.loc[train_mask].copy(),
        X.loc[val_mask].copy(),
        X.loc[test_mask].copy(),
        y.loc[train_mask].copy(),
        y.loc[val_mask].copy(),
        y.loc[test_mask].copy(),
    )


def _evaluate_predictor(predictor, X_eval, y_eval):
    if len(X_eval) == 0:
        return {"log_loss": float("nan"), "accuracy": float("nan"), "rows": 0}
    y_pred_proba = predictor.predict_proba(X_eval)
    y_pred = np.argmax(y_pred_proba, axis=1)
    return {
        "log_loss": float(log_loss(y_eval, y_pred_proba, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "rows": int(len(X_eval)),
    }


def _estimate_permutation_importance_logloss(predictor, X_val, y_val, n_repeats=2, seed=42):
    if len(X_val) == 0 or X_val.shape[1] == 0:
        return pd.Series(dtype=float), float("nan")

    baseline_proba = predictor.predict_proba(X_val)
    baseline_loss = float(log_loss(y_val, baseline_proba, labels=[0, 1, 2]))
    rng = np.random.default_rng(seed)
    importances = {}

    for col in X_val.columns:
        losses = []
        original = X_val[col].to_numpy(copy=True)
        for _ in range(max(1, n_repeats)):
            permuted = rng.permutation(original)
            Xp = X_val.copy()
            Xp[col] = permuted
            proba = predictor.predict_proba(Xp)
            losses.append(float(log_loss(y_val, proba, labels=[0, 1, 2])))
        importances[col] = float(np.mean(losses) - baseline_loss)

    return pd.Series(importances).sort_values(ascending=True), baseline_loss


def _select_club_features(X_train, y_train, X_val, y_val):
    feature_cols = list(X_train.columns)
    if len(feature_cols) <= CLUB_FEATURE_SELECTION_MIN_FEATURES:
        print(
            "ℹ️ Feature selection skipped: feature count is already at or below minimum "
            f"({len(feature_cols)} <= {CLUB_FEATURE_SELECTION_MIN_FEATURES})."
        )
        return feature_cols

    selector_model = CLUB_FEATURE_SELECTION_MODEL
    print(f"🧪 Feature selection model={selector_model} train_rows={len(X_train)} val_rows={len(X_val)}")

    baseline_predictor = FootballMatchPredictor(model_type=selector_model)
    baseline_predictor.train(X_train, y_train)
    baseline_metrics = _evaluate_predictor(baseline_predictor, X_val, y_val)
    print(
        f"🧭 Feature selection baseline | features={len(feature_cols)} "
        f"val_log_loss={baseline_metrics['log_loss']:.4f} val_acc={baseline_metrics['accuracy']:.4f}"
    )

    ranked_importance, perm_baseline = _estimate_permutation_importance_logloss(
        baseline_predictor,
        X_val,
        y_val,
        n_repeats=CLUB_FEATURE_SELECTION_PERM_REPEATS,
    )
    if ranked_importance.empty:
        return feature_cols
    print(f"🧮 Permutation baseline log_loss={perm_baseline:.4f}")

    protected_features = [
        col
        for col in feature_cols
        if any(token in str(col).lower() for token in CLUB_PROTECTED_FEATURE_TOKENS)
    ]
    removable = [
        col
        for col in ranked_importance.index.tolist()
        if col in feature_cols and col not in protected_features
    ]
    if protected_features:
        print(
            f"🛡️ Protected ranking/context features from removal: {len(protected_features)}"
        )
    selected = list(feature_cols)
    best_features = list(feature_cols)
    best_loss = baseline_metrics["log_loss"]
    cursor = 0
    step = max(1, CLUB_FEATURE_SELECTION_STEP)
    min_features = max(5, CLUB_FEATURE_SELECTION_MIN_FEATURES)

    while cursor < len(removable):
        drop_candidates = [col for col in removable[cursor:cursor + step] if col in selected]
        cursor += step
        if not drop_candidates:
            continue
        if len(selected) - len(drop_candidates) < min_features:
            drop_candidates = drop_candidates[: max(0, len(selected) - min_features)]
        if not drop_candidates:
            break

        trial_features = [col for col in selected if col not in drop_candidates]
        trial_predictor = FootballMatchPredictor(model_type=selector_model)
        trial_predictor.train(X_train[trial_features], y_train)
        trial_metrics = _evaluate_predictor(trial_predictor, X_val[trial_features], y_val)

        improved = trial_metrics["log_loss"] <= (best_loss - CLUB_FEATURE_SELECTION_MIN_IMPROVEMENT)
        print(
            f"🔍 Trial drop={len(drop_candidates)} -> features={len(trial_features)} "
            f"val_log_loss={trial_metrics['log_loss']:.4f} "
            f"{'ACCEPT' if improved else 'REJECT'}"
        )
        if improved:
            selected = trial_features
            best_features = trial_features
            best_loss = trial_metrics["log_loss"]

    removed = [col for col in feature_cols if col not in best_features]
    print(
        f"✅ Feature selection complete | kept={len(best_features)} removed={len(removed)} "
        f"best_val_log_loss={best_loss:.4f}"
    )
    if removed:
        print("🗑️ Removed features:", ", ".join(removed[:30]) + (" ..." if len(removed) > 30 else ""))
    return best_features


def _apply_optional_ucl_upsampling(X_train, y_train):
    factor = max(1, CLUB_UCL_UPSAMPLE_FACTOR)
    if factor <= 1:
        return X_train, y_train
    if "is_ucl_match" not in X_train.columns:
        print("ℹ️ UCL upsampling skipped: `is_ucl_match` feature not found.")
        return X_train, y_train

    ucl_mask = pd.to_numeric(X_train["is_ucl_match"], errors="coerce").fillna(0).astype(int) == 1
    ucl_rows = int(ucl_mask.sum())
    if ucl_rows == 0:
        print("ℹ️ UCL upsampling skipped: no UCL rows found in training split.")
        return X_train, y_train

    X_ucl = X_train.loc[ucl_mask]
    y_ucl = y_train.loc[ucl_mask]
    X_aug = pd.concat([X_train] + [X_ucl] * (factor - 1), axis=0, ignore_index=True)
    y_aug = pd.concat([y_train] + [y_ucl] * (factor - 1), axis=0, ignore_index=True)
    perm = np.random.default_rng(42).permutation(len(X_aug))
    X_aug = X_aug.iloc[perm].reset_index(drop=True)
    y_aug = y_aug.iloc[perm].reset_index(drop=True)
    print(
        f"📌 Applied UCL upsampling factor={factor} "
        f"base_rows={len(X_train)} ucl_rows={ucl_rows} augmented_rows={len(X_aug)}"
    )
    return X_aug, y_aug


def _apply_club_min_date_filter(X, y, X_full):
    if not CLUB_MIN_DATE:
        return X, y, X_full
    if "date" not in X_full.columns:
        print("ℹ️ CLUB_MIN_DATE ignored: no `date` column in X_full.")
        return X, y, X_full

    cutoff = pd.to_datetime(CLUB_MIN_DATE, errors="coerce")
    if pd.isna(cutoff):
        print(f"⚠️ Invalid CLUB_MIN_DATE='{CLUB_MIN_DATE}'. Skipping date filter.")
        return X, y, X_full

    dates = pd.to_datetime(X_full["date"], errors="coerce")
    mask = dates >= cutoff
    kept = int(mask.sum())
    if kept < 200:
        print(
            f"⚠️ CLUB_MIN_DATE filter keeps only {kept} rows (cutoff={cutoff.date()}). "
            "Skipping date filter."
        )
        return X, y, X_full

    print(f"🧹 Applied CLUB_MIN_DATE={cutoff.date()} | rows {len(X)} -> {kept}")
    return X.loc[mask].copy(), y.loc[mask].copy(), X_full.loc[mask].copy()


def _apply_competition_subset(X, y, X_full, mode):
    mode_text = str(mode or "").strip().lower()
    if mode_text == "champions":
        label = "Champions"
        competition_tokens = ["champions"]
    elif mode_text == "libertadores":
        label = "Libertadores"
        competition_tokens = ["libertadores"]
    else:
        return X, y, X_full

    if "competition" in X_full.columns:
        competition = X_full["competition"].astype(str).str.lower()
        mask = pd.Series(False, index=X_full.index)
        for token in competition_tokens:
            mask = mask | competition.str.contains(token)
    elif "is_ucl_match" in X.columns:
        mask = pd.to_numeric(X["is_ucl_match"], errors="coerce").fillna(0).astype(int) == 1
    else:
        print(f"⚠️ {label} subset could not be identified. Keeping full club dataset.")
        return X, y, X_full

    kept = int(mask.sum())
    if kept < 100:
        print(f"⚠️ {label} subset too small ({kept} rows). Keeping full club dataset.")
        return X, y, X_full

    print(f"🏆 Applied {mode_text} subset | rows {len(X)} -> {kept}")
    return X.loc[mask].copy(), y.loc[mask].copy(), X_full.loc[mask].copy()


def _save_state_snapshot(X_full, mode="club"):
    if mode not in {"club", "champions", "libertadores"}:
        return

    if mode == "club":
        snapshot_path = CLUB_STATE_SNAPSHOT_PATH
    elif mode == "champions":
        snapshot_path = CHAMPIONS_STATE_SNAPSHOT_PATH
    else:
        snapshot_path = LIBERTADORES_STATE_SNAPSHOT_PATH
    required_cols = ["date", "home_team", "away_team", "home_score", "away_score"]
    if not set(required_cols).issubset(set(X_full.columns)):
        print(
            f"⚠️ {mode} state snapshot skipped: missing required columns "
            f"{[col for col in required_cols if col not in X_full.columns]}"
        )
        return

    state_df = X_full[required_cols].copy()
    state_df["date"] = pd.to_datetime(state_df["date"], errors="coerce")
    state_df["home_score"] = pd.to_numeric(state_df["home_score"], errors="coerce")
    state_df["away_score"] = pd.to_numeric(state_df["away_score"], errors="coerce")
    state_df = state_df.dropna(subset=required_cols).sort_values("date").reset_index(drop=True)
    if state_df.empty:
        print(f"⚠️ {mode} state snapshot skipped: no valid rows after cleaning.")
        return

    team_states, pair_states = ClubFeatureEngineer.build_current_states(state_df)
    payload = {
        "team_states": team_states,
        "pair_states": pair_states,
        "meta": {
            "mode": mode,
            "rows": int(len(state_df)),
            "teams": int(len(team_states)),
            "pairs": int(len(pair_states)),
            "generated_at_utc": datetime.utcnow().isoformat(),
        },
    }
    joblib.dump(payload, snapshot_path)
    print(
        f"💾 Saved {mode} state snapshot to {snapshot_path} "
        f"(rows={len(state_df)} teams={len(team_states)} pairs={len(pair_states)})"
    )


def _prune_libertadores_duplicate_coeff_features(X: pd.DataFrame) -> pd.DataFrame:
    duplicate_cols = [
        # Backward-compatible single-country fields duplicate the home-country block
        # and create unstable linear behavior in Libertadores mode.
        "overall_country_coefficient",
        "season_country_coefficient",
        "country_uefa_overall_rank",
        "country_uefa_season_rank",
    ]
    cols_to_drop = [col for col in duplicate_cols if col in X.columns]
    if cols_to_drop:
        print(f"🧹 libertadores feature prune | dropping duplicate coeff cols={cols_to_drop}")
        X = X.drop(columns=cols_to_drop)
    return X


def train_models(mode="national"):
    if mode == "national":
        matches_df = fetch_national_matches()
        file_path = "supabase:matches"
        preprocessor = MatchDataPreprocessor(matches_df=matches_df)
    elif mode in {"club", "champions", "libertadores"}:
        if mode == "libertadores":
            matches_df = fetch_libertadores_matches()
            file_path = "supabase:libertadores_matches"
            preprocessor = ClubMatchDataPreprocessor(
                matches_df=matches_df,
                include_uefa_coefficients=CLUB_INCLUDE_UEFA_COEFFICIENTS,
                coeff_path=LIBERTADORES_COEFFICIENTS_PATH,
                coeff_history_path=LIBERTADORES_COEFFICIENTS_PATH,
                country_coeff_path=LIBERTADORES_COEFFICIENTS_PATH,
            )
        else:
            club_dataset_path = CLUB_MATCHES_HISTORY_PATH
            # Auto-build merged historical dataset if it does not exist yet.
            if not club_dataset_path.exists():
                if mode == "club" and ensure_club_historical_dataset is not None:
                    ensure_club_historical_dataset()
                if mode == "club" and CLUB_MATCHES_HISTORY_PATH.exists():
                    club_dataset_path = CLUB_MATCHES_HISTORY_PATH
                elif mode == "club" and Path(CLUB_MATCHES_PATH).exists():
                    print(
                        f"⚠️ {CLUB_MATCHES_HISTORY_PATH} not found. "
                        f"Falling back to {CLUB_MATCHES_PATH}."
                    )
                    club_dataset_path = Path(CLUB_MATCHES_PATH)
                else:
                    raise FileNotFoundError(
                        f"Missing required dataset for mode={mode}: {club_dataset_path}."
                    )
            file_path = str(club_dataset_path)
            preprocessor = ClubMatchDataPreprocessor(
                file_path,
                include_uefa_coefficients=CLUB_INCLUDE_UEFA_COEFFICIENTS,
            )
    else:
        raise ValueError(f"Modo inválido: {mode}")

    print(file_path)
    X, y = preprocessor.preprocess()

    # Save processed data for prediction module
    X_full_raw = preprocessor.X_Full.copy()
    X_full = X_full_raw.copy()
    if mode in {"club", "champions", "libertadores"}:
        X, y, X_full = _apply_club_min_date_filter(X, y, X_full)
    if mode in {"champions", "libertadores"}:
        X, y, X_full = _apply_competition_subset(X, y, X_full, mode)
    if mode == "libertadores":
        X = _prune_libertadores_duplicate_coeff_features(X)

    selected_features = list(X.columns)

    # Time-based split to avoid temporal leakage.
    if mode in {"club", "champions", "libertadores"}:
        X_train, X_val, X_test, y_train, y_val, y_test = _club_time_split_train_val_test(X, y, X_full)
        print(
            f"🗓️ {mode} OOT split | train={len(X_train)} val={len(X_val)} test={len(X_test)} "
            "cutoffs=(2023-06-30, 2024-06-30)"
        )
        if CLUB_FEATURE_SELECTION_ENABLED and len(X_train) > 0 and len(X_val) > 0:
            selected_features = _select_club_features(X_train, y_train, X_val, y_val)
        X_for_training = X[selected_features].copy()
        X_train_val = pd.concat([X_train[selected_features], X_val[selected_features]], axis=0).reset_index(drop=True)
        y_train_val = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        if mode == "club":
            X_train_val, y_train_val = _apply_optional_ucl_upsampling(X_train_val, y_train_val)
        X_test_eval = X_test[selected_features]
    else:
        split_idx = max(1, int(len(X) * 0.8))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        X_for_training = X.copy()
        X_train_val = X_train
        y_train_val = y_train
        X_test_eval = X_test

    X_full_for_inference = X_full_raw if mode in {"champions", "libertadores"} else X_full

    # Save processed data (feature template used by inference must match trained models).
    save_processed_data_to_csv(X_for_training, y, X_full_for_inference, mode=mode)
    if mode in {"club", "champions", "libertadores"}:
        _save_state_snapshot(X_full_for_inference, mode=mode)

    models = {}
    predictors = {}
    metrics = {}

    for model_type in ["random_forest", "logistic_regression", "mlp"]:
        print(f"🔁 Entrenando modelo {model_type}...")
        predictor, best_model = train_model(
            model_type,
            X_train_val,
            y_train_val,
            mode=mode,
            save_model=(mode not in {"club", "champions", "libertadores"}),
        )
        models[model_type] = best_model
        predictors[model_type] = predictor

        if len(X_test_eval) > 0:
            y_pred_proba = predictor.predict_proba(X_test_eval)
            y_pred = np.argmax(y_pred_proba, axis=1)
            model_log_loss = log_loss(y_test, y_pred_proba, labels=[0, 1, 2])
            model_acc = accuracy_score(y_test, y_pred)
            metrics[model_type] = {
                "log_loss": float(model_log_loss),
                "accuracy": float(model_acc),
                "test_rows": int(len(X_test_eval)),
                "feature_count": int(len(selected_features)),
            }
            if mode in {"club", "champions", "libertadores"} and "is_ucl_match" in X_test_eval.columns:
                ucl_mask_test = pd.to_numeric(X_test_eval["is_ucl_match"], errors="coerce").fillna(0).astype(int) == 1
                if int(ucl_mask_test.sum()) > 0:
                    ucl_probs = predictor.predict_proba(X_test_eval.loc[ucl_mask_test])
                    ucl_pred = np.argmax(ucl_probs, axis=1)
                    metrics[model_type]["ucl_test_rows"] = int(ucl_mask_test.sum())
                    metrics[model_type]["ucl_log_loss"] = float(
                        log_loss(y_test.loc[ucl_mask_test], ucl_probs, labels=[0, 1, 2])
                    )
                    metrics[model_type]["ucl_accuracy"] = float(
                        accuracy_score(y_test.loc[ucl_mask_test], ucl_pred)
                    )
            print(
                f"📈 {model_type} | test_rows={len(X_test_eval)} "
                f"log_loss={model_log_loss:.4f} accuracy={model_acc:.4f}"
            )
            if "ucl_log_loss" in metrics[model_type]:
                print(
                    f"   ↳ UCL subset | rows={metrics[model_type]['ucl_test_rows']} "
                    f"log_loss={metrics[model_type]['ucl_log_loss']:.4f} "
                    f"accuracy={metrics[model_type]['ucl_accuracy']:.4f}"
                )

    if mode in {"club", "champions", "libertadores"} and metrics:
        baseline_log_loss = CLUB_BASELINE_LOG_LOSS if mode == "club" else CHAMPIONS_BASELINE_LOG_LOSS
        best_logloss = min(m["log_loss"] for m in metrics.values())
        print(
            f"🏁 {mode} gate | best_log_loss={best_logloss:.4f} "
            f"baseline_log_loss={baseline_log_loss:.4f}"
        )
        gate_passed = (
            (not CLUB_ROLLOUT_GATE_ENABLED)
            or CLUB_FORCE_SAVE_MODELS
            or (best_logloss <= baseline_log_loss)
        )
        if gate_passed:
            for model_type, predictor in predictors.items():
                save_model_locally(predictor, model_type, mode=mode)
            if CLUB_FORCE_SAVE_MODELS and CLUB_ROLLOUT_GATE_ENABLED and best_logloss > baseline_log_loss:
                print(f"✅ {mode} models saved (forced with CLUB_FORCE_SAVE_MODELS=1).")
            elif not CLUB_ROLLOUT_GATE_ENABLED:
                print(f"✅ {mode} models saved (rollout gate disabled).")
            else:
                print(f"✅ {mode} models saved (gate passed).")
        else:
            print(f"⚠️ {mode} models not saved because rollout gate did not pass.")

    return models, X, y, metrics

if __name__ == "__main__":
    train_mode = os.getenv("TRAIN_MODE", "club")
    train_models(train_mode)
