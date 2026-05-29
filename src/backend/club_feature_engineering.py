from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import pandas as pd


@dataclass
class _TeamState:
    elo: float = 1500.0
    recent_goals_for: Deque[float] = None
    recent_goals_against: Deque[float] = None
    recent_points: Deque[float] = None
    recent_home_goals_for: Deque[float] = None
    recent_home_goals_against: Deque[float] = None
    recent_home_points: Deque[float] = None
    recent_away_goals_for: Deque[float] = None
    recent_away_goals_against: Deque[float] = None
    recent_away_points: Deque[float] = None
    last_match_date: pd.Timestamp = None

    def __post_init__(self):
        if self.recent_goals_for is None:
            self.recent_goals_for = deque(maxlen=10)
        if self.recent_goals_against is None:
            self.recent_goals_against = deque(maxlen=10)
        if self.recent_points is None:
            self.recent_points = deque(maxlen=10)
        if self.recent_home_goals_for is None:
            self.recent_home_goals_for = deque(maxlen=10)
        if self.recent_home_goals_against is None:
            self.recent_home_goals_against = deque(maxlen=10)
        if self.recent_home_points is None:
            self.recent_home_points = deque(maxlen=10)
        if self.recent_away_goals_for is None:
            self.recent_away_goals_for = deque(maxlen=10)
        if self.recent_away_goals_against is None:
            self.recent_away_goals_against = deque(maxlen=10)
        if self.recent_away_points is None:
            self.recent_away_points = deque(maxlen=10)


class ClubFeatureEngineer:
    HOME_ELO_ADVANTAGE = 60.0
    ELO_K_FACTOR = 20.0

    def __init__(self, matches_df, verbose=False):
        self.df = matches_df.copy()
        self.verbose = verbose
        self.team_states_: Dict[str, _TeamState] = {}
        self.pair_states_: Dict[Tuple[str, str], dict] = {}

    def log(self, msg):
        if self.verbose:
            print(msg)

    @staticmethod
    def _window_mean(values: Deque[float], size: int) -> float:
        if not values:
            return 0.0
        tail = list(values)[-size:]
        return float(sum(tail) / len(tail)) if tail else 0.0

    @staticmethod
    def _window_rate(predicate_values: list[bool], size: int) -> float:
        if not predicate_values:
            return 0.0
        tail = predicate_values[-size:]
        if not tail:
            return 0.0
        return float(sum(1 for value in tail if value) / len(tail))

    @classmethod
    def _window_win_rate(cls, points: Deque[float], size: int) -> float:
        values = list(points)
        return cls._window_rate([value >= 2.5 for value in values], size)

    @classmethod
    def _window_draw_rate(cls, points: Deque[float], size: int) -> float:
        values = list(points)
        return cls._window_rate([0.5 < value < 2.5 for value in values], size)

    @classmethod
    def _window_loss_rate(cls, points: Deque[float], size: int) -> float:
        values = list(points)
        return cls._window_rate([value <= 0.5 for value in values], size)

    @classmethod
    def _window_clean_sheet_rate(cls, goals_against: Deque[float], size: int) -> float:
        values = list(goals_against)
        return cls._window_rate([value <= 0.0 for value in values], size)

    @classmethod
    def _window_fail_to_score_rate(cls, goals_for: Deque[float], size: int) -> float:
        values = list(goals_for)
        return cls._window_rate([value <= 0.0 for value in values], size)

    @classmethod
    def _window_btts_rate(cls, goals_for: Deque[float], goals_against: Deque[float], size: int) -> float:
        gf_values = list(goals_for)
        ga_values = list(goals_against)
        combined = [(gf > 0.0) and (ga > 0.0) for gf, ga in zip(gf_values, ga_values)]
        return cls._window_rate(combined, size)

    @staticmethod
    def _points_for_result(goals_for: float, goals_against: float) -> int:
        if goals_for > goals_against:
            return 3
        if goals_for == goals_against:
            return 1
        return 0

    @staticmethod
    def _is_ucl_match(competition: str) -> int:
        value = str(competition or "").lower()
        return int(
            ("champions" in value and "lg" in value)
            or ("champions league" in value)
            or ("libertadores" in value)
        )

    @staticmethod
    def _is_knockout_round(round_name: str) -> int:
        text = str(round_name or "").lower()
        tokens = ["knockout", "play-off", "playoff", "quarter", "semi", "final", "round of", "last "]
        return int(any(token in text for token in tokens))

    def _get_team_state(self, team: str, states: Dict[str, _TeamState]) -> _TeamState:
        if team not in states:
            states[team] = _TeamState()
        return states[team]

    def _simulate(self, collect_feature_rows: bool = True):
        df = self.df.sort_values("date").reset_index(drop=True)

        feature_rows = []
        team_states: Dict[str, _TeamState] = {}
        pair_states: Dict[Tuple[str, str], dict] = {}

        for row in df.itertuples(index=False):
            home_team = str(getattr(row, "home_team")).strip()
            away_team = str(getattr(row, "away_team")).strip()
            date = pd.to_datetime(getattr(row, "date"), errors="coerce")
            home_score = float(getattr(row, "home_score"))
            away_score = float(getattr(row, "away_score"))

            if pd.isna(date):
                # Skip invalid rows in state simulation.
                if collect_feature_rows:
                    feature_rows.append({})
                continue

            home_state = self._get_team_state(home_team, team_states)
            away_state = self._get_team_state(away_team, team_states)

            home_pre_elo = float(home_state.elo)
            away_pre_elo = float(away_state.elo)

            pair_key = tuple(sorted((home_team, away_team)))
            pair_info = pair_states.setdefault(pair_key, {"count": 0, "sum_by_team": {}})
            h2h_count = int(pair_info["count"])
            h2h_sum_home = float(pair_info["sum_by_team"].get(home_team, 0.0))

            home_rest_days = (
                float((date - home_state.last_match_date).days)
                if home_state.last_match_date is not None
                else 7.0
            )
            away_rest_days = (
                float((date - away_state.last_match_date).days)
                if away_state.last_match_date is not None
                else 7.0
            )

            home_form_gf_5 = self._window_mean(home_state.recent_goals_for, 5)
            home_form_ga_5 = self._window_mean(home_state.recent_goals_against, 5)
            home_form_pts_5 = self._window_mean(home_state.recent_points, 5)
            home_form_goal_diff_5 = home_form_gf_5 - home_form_ga_5
            home_form_win_rate_5 = self._window_win_rate(home_state.recent_points, 5)
            home_form_draw_rate_5 = self._window_draw_rate(home_state.recent_points, 5)
            home_form_loss_rate_5 = self._window_loss_rate(home_state.recent_points, 5)
            home_form_clean_sheet_rate_5 = self._window_clean_sheet_rate(home_state.recent_goals_against, 5)
            home_form_fail_to_score_rate_5 = self._window_fail_to_score_rate(home_state.recent_goals_for, 5)
            home_form_btts_rate_5 = self._window_btts_rate(home_state.recent_goals_for, home_state.recent_goals_against, 5)
            home_form_gf_10 = self._window_mean(home_state.recent_goals_for, 10)
            home_form_ga_10 = self._window_mean(home_state.recent_goals_against, 10)
            home_form_pts_10 = self._window_mean(home_state.recent_points, 10)
            home_form_goal_diff_10 = home_form_gf_10 - home_form_ga_10
            home_form_win_rate_10 = self._window_win_rate(home_state.recent_points, 10)
            home_form_draw_rate_10 = self._window_draw_rate(home_state.recent_points, 10)
            home_form_loss_rate_10 = self._window_loss_rate(home_state.recent_points, 10)
            home_form_clean_sheet_rate_10 = self._window_clean_sheet_rate(home_state.recent_goals_against, 10)
            home_form_fail_to_score_rate_10 = self._window_fail_to_score_rate(home_state.recent_goals_for, 10)
            home_form_btts_rate_10 = self._window_btts_rate(home_state.recent_goals_for, home_state.recent_goals_against, 10)

            away_form_gf_5 = self._window_mean(away_state.recent_goals_for, 5)
            away_form_ga_5 = self._window_mean(away_state.recent_goals_against, 5)
            away_form_pts_5 = self._window_mean(away_state.recent_points, 5)
            away_form_goal_diff_5 = away_form_gf_5 - away_form_ga_5
            away_form_win_rate_5 = self._window_win_rate(away_state.recent_points, 5)
            away_form_draw_rate_5 = self._window_draw_rate(away_state.recent_points, 5)
            away_form_loss_rate_5 = self._window_loss_rate(away_state.recent_points, 5)
            away_form_clean_sheet_rate_5 = self._window_clean_sheet_rate(away_state.recent_goals_against, 5)
            away_form_fail_to_score_rate_5 = self._window_fail_to_score_rate(away_state.recent_goals_for, 5)
            away_form_btts_rate_5 = self._window_btts_rate(away_state.recent_goals_for, away_state.recent_goals_against, 5)
            away_form_gf_10 = self._window_mean(away_state.recent_goals_for, 10)
            away_form_ga_10 = self._window_mean(away_state.recent_goals_against, 10)
            away_form_pts_10 = self._window_mean(away_state.recent_points, 10)
            away_form_goal_diff_10 = away_form_gf_10 - away_form_ga_10
            away_form_win_rate_10 = self._window_win_rate(away_state.recent_points, 10)
            away_form_draw_rate_10 = self._window_draw_rate(away_state.recent_points, 10)
            away_form_loss_rate_10 = self._window_loss_rate(away_state.recent_points, 10)
            away_form_clean_sheet_rate_10 = self._window_clean_sheet_rate(away_state.recent_goals_against, 10)
            away_form_fail_to_score_rate_10 = self._window_fail_to_score_rate(away_state.recent_goals_for, 10)
            away_form_btts_rate_10 = self._window_btts_rate(away_state.recent_goals_for, away_state.recent_goals_against, 10)

            home_home_form_gf_5 = self._window_mean(home_state.recent_home_goals_for, 5)
            home_home_form_ga_5 = self._window_mean(home_state.recent_home_goals_against, 5)
            home_home_form_pts_5 = self._window_mean(home_state.recent_home_points, 5)
            home_home_form_gf_10 = self._window_mean(home_state.recent_home_goals_for, 10)
            home_home_form_ga_10 = self._window_mean(home_state.recent_home_goals_against, 10)
            home_home_form_pts_10 = self._window_mean(home_state.recent_home_points, 10)

            away_away_form_gf_5 = self._window_mean(away_state.recent_away_goals_for, 5)
            away_away_form_ga_5 = self._window_mean(away_state.recent_away_goals_against, 5)
            away_away_form_pts_5 = self._window_mean(away_state.recent_away_points, 5)
            away_away_form_gf_10 = self._window_mean(away_state.recent_away_goals_for, 10)
            away_away_form_ga_10 = self._window_mean(away_state.recent_away_goals_against, 10)
            away_away_form_pts_10 = self._window_mean(away_state.recent_away_points, 10)

            goal_diff = float(home_score - away_score)
            competition = getattr(row, "competition", "")
            round_name = getattr(row, "round", "")

            if collect_feature_rows:
                feature_rows.append(
                    {
                        "goal_diff": goal_diff,
                        "matchup_id": "_".join(sorted([home_team, away_team])),
                        "h2h_matches_before": h2h_count,
                        "head_to_head_goal_diff": h2h_sum_home / h2h_count if h2h_count else 0.0,
                        "h2h_available": int(h2h_count > 0),
                        "home_pre_elo": home_pre_elo,
                        "away_pre_elo": away_pre_elo,
                        "elo_diff": home_pre_elo - away_pre_elo,
                        "home_form_goals_for_5": home_form_gf_5,
                        "home_form_goals_against_5": home_form_ga_5,
                        "home_form_points_5": home_form_pts_5,
                        "home_form_goal_diff_5": home_form_goal_diff_5,
                        "home_form_win_rate_5": home_form_win_rate_5,
                        "home_form_draw_rate_5": home_form_draw_rate_5,
                        "home_form_loss_rate_5": home_form_loss_rate_5,
                        "home_form_clean_sheet_rate_5": home_form_clean_sheet_rate_5,
                        "home_form_fail_to_score_rate_5": home_form_fail_to_score_rate_5,
                        "home_form_btts_rate_5": home_form_btts_rate_5,
                        "home_form_goals_for_10": home_form_gf_10,
                        "home_form_goals_against_10": home_form_ga_10,
                        "home_form_points_10": home_form_pts_10,
                        "home_form_goal_diff_10": home_form_goal_diff_10,
                        "home_form_win_rate_10": home_form_win_rate_10,
                        "home_form_draw_rate_10": home_form_draw_rate_10,
                        "home_form_loss_rate_10": home_form_loss_rate_10,
                        "home_form_clean_sheet_rate_10": home_form_clean_sheet_rate_10,
                        "home_form_fail_to_score_rate_10": home_form_fail_to_score_rate_10,
                        "home_form_btts_rate_10": home_form_btts_rate_10,
                        "away_form_goals_for_5": away_form_gf_5,
                        "away_form_goals_against_5": away_form_ga_5,
                        "away_form_points_5": away_form_pts_5,
                        "away_form_goal_diff_5": away_form_goal_diff_5,
                        "away_form_win_rate_5": away_form_win_rate_5,
                        "away_form_draw_rate_5": away_form_draw_rate_5,
                        "away_form_loss_rate_5": away_form_loss_rate_5,
                        "away_form_clean_sheet_rate_5": away_form_clean_sheet_rate_5,
                        "away_form_fail_to_score_rate_5": away_form_fail_to_score_rate_5,
                        "away_form_btts_rate_5": away_form_btts_rate_5,
                        "away_form_goals_for_10": away_form_gf_10,
                        "away_form_goals_against_10": away_form_ga_10,
                        "away_form_points_10": away_form_pts_10,
                        "away_form_goal_diff_10": away_form_goal_diff_10,
                        "away_form_win_rate_10": away_form_win_rate_10,
                        "away_form_draw_rate_10": away_form_draw_rate_10,
                        "away_form_loss_rate_10": away_form_loss_rate_10,
                        "away_form_clean_sheet_rate_10": away_form_clean_sheet_rate_10,
                        "away_form_fail_to_score_rate_10": away_form_fail_to_score_rate_10,
                        "away_form_btts_rate_10": away_form_btts_rate_10,
                        "home_team_home_form_goals_for_5": home_home_form_gf_5,
                        "home_team_home_form_goals_against_5": home_home_form_ga_5,
                        "home_team_home_form_points_5": home_home_form_pts_5,
                        "home_team_home_form_goals_for_10": home_home_form_gf_10,
                        "home_team_home_form_goals_against_10": home_home_form_ga_10,
                        "home_team_home_form_points_10": home_home_form_pts_10,
                        "away_team_away_form_goals_for_5": away_away_form_gf_5,
                        "away_team_away_form_goals_against_5": away_away_form_ga_5,
                        "away_team_away_form_points_5": away_away_form_pts_5,
                        "away_team_away_form_goals_for_10": away_away_form_gf_10,
                        "away_team_away_form_goals_against_10": away_away_form_ga_10,
                        "away_team_away_form_points_10": away_away_form_pts_10,
                        "venue_form_points_diff_5": home_home_form_pts_5 - away_away_form_pts_5,
                        "venue_form_points_diff_10": home_home_form_pts_10 - away_away_form_pts_10,
                        "home_rest_days": home_rest_days,
                        "away_rest_days": away_rest_days,
                        "rest_days_diff": home_rest_days - away_rest_days,
                        "is_ucl_match": self._is_ucl_match(competition),
                        "is_knockout_round": self._is_knockout_round(round_name),
                    }
                )

            expected_home = 1.0 / (1.0 + 10.0 ** ((away_pre_elo + self.HOME_ELO_ADVANTAGE - home_pre_elo) / 400.0))
            actual_home = 1.0 if home_score > away_score else (0.5 if home_score == away_score else 0.0)
            actual_away = 1.0 - actual_home

            new_home_elo = home_pre_elo + self.ELO_K_FACTOR * (actual_home - expected_home)
            new_away_elo = away_pre_elo + self.ELO_K_FACTOR * (actual_away - (1.0 - expected_home))

            home_points = self._points_for_result(home_score, away_score)
            away_points = self._points_for_result(away_score, home_score)

            home_state.recent_goals_for.append(home_score)
            home_state.recent_goals_against.append(away_score)
            home_state.recent_points.append(home_points)
            home_state.recent_home_goals_for.append(home_score)
            home_state.recent_home_goals_against.append(away_score)
            home_state.recent_home_points.append(home_points)
            home_state.last_match_date = date
            home_state.elo = new_home_elo

            away_state.recent_goals_for.append(away_score)
            away_state.recent_goals_against.append(home_score)
            away_state.recent_points.append(away_points)
            away_state.recent_away_goals_for.append(away_score)
            away_state.recent_away_goals_against.append(home_score)
            away_state.recent_away_points.append(away_points)
            away_state.last_match_date = date
            away_state.elo = new_away_elo

            pair_info["count"] += 1
            pair_info["sum_by_team"][home_team] = float(pair_info["sum_by_team"].get(home_team, 0.0)) + goal_diff
            pair_info["sum_by_team"][away_team] = float(pair_info["sum_by_team"].get(away_team, 0.0)) - goal_diff

        return feature_rows, team_states, pair_states

    @classmethod
    def build_current_states(cls, matches_df):
        df = matches_df.copy()
        if df.empty:
            return {}, {}

        engineer = cls(df)
        _, team_states, pair_states = engineer._simulate(collect_feature_rows=False)

        team_snapshot = {}
        for team, state in team_states.items():
            team_snapshot[team] = {
                "elo": float(state.elo),
                "form_goals_for_5": cls._window_mean(state.recent_goals_for, 5),
                "form_goals_against_5": cls._window_mean(state.recent_goals_against, 5),
                "form_points_5": cls._window_mean(state.recent_points, 5),
                "form_goal_diff_5": cls._window_mean(state.recent_goals_for, 5) - cls._window_mean(state.recent_goals_against, 5),
                "form_win_rate_5": cls._window_win_rate(state.recent_points, 5),
                "form_draw_rate_5": cls._window_draw_rate(state.recent_points, 5),
                "form_loss_rate_5": cls._window_loss_rate(state.recent_points, 5),
                "form_clean_sheet_rate_5": cls._window_clean_sheet_rate(state.recent_goals_against, 5),
                "form_fail_to_score_rate_5": cls._window_fail_to_score_rate(state.recent_goals_for, 5),
                "form_btts_rate_5": cls._window_btts_rate(state.recent_goals_for, state.recent_goals_against, 5),
                "form_goals_for_10": cls._window_mean(state.recent_goals_for, 10),
                "form_goals_against_10": cls._window_mean(state.recent_goals_against, 10),
                "form_points_10": cls._window_mean(state.recent_points, 10),
                "form_goal_diff_10": cls._window_mean(state.recent_goals_for, 10) - cls._window_mean(state.recent_goals_against, 10),
                "form_win_rate_10": cls._window_win_rate(state.recent_points, 10),
                "form_draw_rate_10": cls._window_draw_rate(state.recent_points, 10),
                "form_loss_rate_10": cls._window_loss_rate(state.recent_points, 10),
                "form_clean_sheet_rate_10": cls._window_clean_sheet_rate(state.recent_goals_against, 10),
                "form_fail_to_score_rate_10": cls._window_fail_to_score_rate(state.recent_goals_for, 10),
                "form_btts_rate_10": cls._window_btts_rate(state.recent_goals_for, state.recent_goals_against, 10),
                "home_only_form_goals_for_5": cls._window_mean(state.recent_home_goals_for, 5),
                "home_only_form_goals_against_5": cls._window_mean(state.recent_home_goals_against, 5),
                "home_only_form_points_5": cls._window_mean(state.recent_home_points, 5),
                "home_only_form_goals_for_10": cls._window_mean(state.recent_home_goals_for, 10),
                "home_only_form_goals_against_10": cls._window_mean(state.recent_home_goals_against, 10),
                "home_only_form_points_10": cls._window_mean(state.recent_home_points, 10),
                "away_only_form_goals_for_5": cls._window_mean(state.recent_away_goals_for, 5),
                "away_only_form_goals_against_5": cls._window_mean(state.recent_away_goals_against, 5),
                "away_only_form_points_5": cls._window_mean(state.recent_away_points, 5),
                "away_only_form_goals_for_10": cls._window_mean(state.recent_away_goals_for, 10),
                "away_only_form_goals_against_10": cls._window_mean(state.recent_away_goals_against, 10),
                "away_only_form_points_10": cls._window_mean(state.recent_away_points, 10),
                "last_match_date": state.last_match_date,
            }

        pair_snapshot = {}
        for pair_key, info in pair_states.items():
            pair_snapshot[pair_key] = {
                "count": int(info["count"]),
                "sum_by_team": dict(info["sum_by_team"]),
            }

        return team_snapshot, pair_snapshot

    def generate_features(self):
        required = ["date", "home_team", "away_team", "home_score", "away_score"]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in club matches data: {missing}")

        if "competition" not in self.df.columns:
            self.df["competition"] = ""
        if "round" not in self.df.columns:
            self.df["round"] = ""
        if "neutral" not in self.df.columns:
            self.df["neutral"] = 0

        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        self.df = self.df.dropna(subset=["date", "home_score", "away_score"]).copy()
        self.df = self.df.sort_values("date").reset_index(drop=True)

        feature_rows, team_states, pair_states = self._simulate(collect_feature_rows=True)
        features_df = pd.DataFrame(feature_rows)
        if features_df.empty:
            return self.df

        self.df = pd.concat([self.df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

        numeric_cols = [
            "home_pre_elo",
            "away_pre_elo",
            "elo_diff",
            "h2h_matches_before",
            "head_to_head_goal_diff",
            "home_form_goals_for_5",
            "home_form_goals_against_5",
            "home_form_points_5",
            "home_form_goals_for_10",
            "home_form_goals_against_10",
            "home_form_points_10",
            "home_form_goal_diff_5",
            "home_form_win_rate_5",
            "home_form_draw_rate_5",
            "home_form_loss_rate_5",
            "home_form_clean_sheet_rate_5",
            "home_form_fail_to_score_rate_5",
            "home_form_btts_rate_5",
            "home_form_goal_diff_10",
            "home_form_win_rate_10",
            "home_form_draw_rate_10",
            "home_form_loss_rate_10",
            "home_form_clean_sheet_rate_10",
            "home_form_fail_to_score_rate_10",
            "home_form_btts_rate_10",
            "away_form_goals_for_5",
            "away_form_goals_against_5",
            "away_form_points_5",
            "away_form_goals_for_10",
            "away_form_goals_against_10",
            "away_form_points_10",
            "away_form_goal_diff_5",
            "away_form_win_rate_5",
            "away_form_draw_rate_5",
            "away_form_loss_rate_5",
            "away_form_clean_sheet_rate_5",
            "away_form_fail_to_score_rate_5",
            "away_form_btts_rate_5",
            "away_form_goal_diff_10",
            "away_form_win_rate_10",
            "away_form_draw_rate_10",
            "away_form_loss_rate_10",
            "away_form_clean_sheet_rate_10",
            "away_form_fail_to_score_rate_10",
            "away_form_btts_rate_10",
            "home_team_home_form_goals_for_5",
            "home_team_home_form_goals_against_5",
            "home_team_home_form_points_5",
            "home_team_home_form_goals_for_10",
            "home_team_home_form_goals_against_10",
            "home_team_home_form_points_10",
            "away_team_away_form_goals_for_5",
            "away_team_away_form_goals_against_5",
            "away_team_away_form_points_5",
            "away_team_away_form_goals_for_10",
            "away_team_away_form_goals_against_10",
            "away_team_away_form_points_10",
            "venue_form_points_diff_5",
            "venue_form_points_diff_10",
            "home_rest_days",
            "away_rest_days",
            "rest_days_diff",
            "goal_diff",
        ]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0)

        self.team_states_ = team_states
        self.pair_states_ = pair_states
        return self.df
