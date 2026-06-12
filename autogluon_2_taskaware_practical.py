                       
                                                                                      
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import math
import random
import shutil
import warnings
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
NUM_ACTION_CLASSES = 19
NUM_ACTION_MAIN_CLASSES = 15                                                    
NUM_POINT_CLASSES = 10
NUM_AID_GROUP_CLASSES = 5
NUM_AID_SUB_CLASSES = 8
NUM_PID_DEPTH_CLASSES = 4
NUM_PID_SIDE_CLASSES = 4

ACTION_GROUP_MAP = np.array([0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4], dtype=np.int64)
ACTION_SUB_MAP = np.array([0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4], dtype=np.int64)
POINT_DEPTH_MAP = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int64)
POINT_SIDE_MAP = np.array([0, 1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.int64)

RAW_ID_COLS = [
    "rally_uid", "rally_id", "match", "numberGame",
    "gamePlayerId", "gamePlayerOtherId",
    "player_self_id_tmp", "player_other_id_tmp", "player_pair_key_tmp",
]
RAW_ID_SUFFIXES = ("_idx", "_id_tmp", "_key_tmp")

BASE_CAT_COLS = [
    "sex",
    "strikeId", "handId", "strengthId",
    "spinId", "sid_spin", "sid_side",
    "positionId",
    "actionId", "aid_group", "aid_sub",
    "pointId", "pid_depth", "pid_side",
    "action_point_combo_bl", "action_side_combo_bl", "action_depth_combo_bl",
    "spin_strength_combo_bl",
]

BASE_NUM_COLS = [
    "scoreSelf", "scoreOther", "strikeNumber",
    "scoreDiff_bl", "scoreSum_bl", "absScoreDiff_bl",
    "scoreDiff_norm_bl", "scorePressure_bl", "points_to_win_self_bl", "points_to_win_other_bl",
    "is_tie_score_bl", "is_leading_bl", "is_trailing_bl", "is_close_score_bl",
    "is_late_game_bl", "is_overtime_bl", "is_deuce_like_bl", "is_game_point_like_bl",
    "leader_game_point_bl", "trailer_under_pressure_bl",
    "rally_phase_bl", "strike_log1p_bl", "is_serve_phase_bl", "is_receive_phase_bl",
    "is_third_ball_phase_bl", "is_rally_phase_bl",
    "is_server_turn_bl", "is_receiver_turn_bl",
    "strength_norm_bl", "action_group_norm_bl", "spin_norm_bl", "side_spin_bl",
    "is_deep_placement_bl", "is_short_placement_bl", "is_wide_placement_bl", "is_middle_placement_bl",
    "wide_deep_pressure_bl", "placement_edge_pressure_bl",
    "spin_strength_pressure_bl", "action_spin_pressure_bl",
    "server_score_pressure_bl", "receiver_score_pressure_bl",
    "point_changed_prev_bl", "position_changed_prev_bl", "action_changed_prev_bl",
    "spin_changed_prev_bl", "strength_changed_prev_bl", "side_changed_prev_bl", "depth_changed_prev_bl",
    "side_move_bl", "depth_move_bl", "placement_move_magnitude_bl",
    "aggressive_shot_bl", "control_shot_bl",
    "serve_aggressive_bl", "receive_aggressive_bl", "third_ball_aggressive_bl", "rally_aggressive_bl",
    "pressure_aggressive_bl", "late_aggressive_bl",
    "serve_control_bl", "receive_control_bl", "third_ball_control_bl", "rally_control_bl",
    "roll3_point_change_rate_bl", "roll5_point_change_rate_bl",
    "roll5_position_change_rate_bl", "roll5_action_change_rate_bl", "roll5_spin_change_rate_bl",
    "roll5_side_change_rate_bl", "roll5_depth_change_rate_bl", "placement_volatility_bl",
    "roll3_aggressive_rate_bl", "roll5_aggressive_rate_bl", "roll3_control_rate_bl",
    "roll3_wide_deep_rate_bl", "tempo_acceleration_bl",
]

SHORT_RATIO_FEATURES = [
    ("pointId", "point", NUM_POINT_CLASSES),
    ("actionId", "action", NUM_ACTION_CLASSES),
    ("aid_group", "aid_group", NUM_AID_GROUP_CLASSES),
    ("pid_side", "side", NUM_PID_SIDE_CLASSES),
    ("pid_depth", "depth", NUM_PID_DEPTH_CLASSES),
    ("sid_spin", "spin", 4),
]

TRANSITION_SPECS_ACTION = [
    ("a_last_action", ["last_actionId"]),
    ("a_last_point", ["last_pointId"]),
    ("a_last_action_point", ["last_actionId", "last_pointId"]),
    ("a_prev2_last_action", ["prev2_actionId", "last_actionId"]),
    ("a_phase_action", ["next_rally_phase_tf", "last_actionId"]),
    ("a_serve_side_action", ["next_is_serve_side_tf", "last_actionId"]),
]

TRANSITION_SPECS_POINT = [
    ("p_last_point", ["last_pointId"]),
    ("p_last_action", ["last_actionId"]),
    ("p_last_action_point", ["last_actionId", "last_pointId"]),
    ("p_last_group_point", ["last_aid_group", "last_pointId"]),
    ("p_side_depth_action", ["last_pid_side", "last_pid_depth", "last_actionId"]),
    ("p_phase_action", ["next_rally_phase_tf", "last_actionId"]),
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="float")
    return df


def _to_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _clip_int_series(s: pd.Series, lo: int, hi: int) -> np.ndarray:
    return _to_num(s, 0).astype(np.int64).clip(lo, hi).to_numpy()


def action_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "actionId" not in df.columns:
        df["actionId"] = 0
    aid = _clip_int_series(df["actionId"], 0, 18)
    df["aid_group"] = ACTION_GROUP_MAP[aid].astype(np.int64)
    df["aid_sub"] = ACTION_SUB_MAP[aid].astype(np.int64)
    return df


def point_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pointId" not in df.columns:
        df["pointId"] = 0
    pid = _clip_int_series(df["pointId"], 0, 9)
    df["pid_depth"] = POINT_DEPTH_MAP[pid].astype(np.int64)
    df["pid_side"] = POINT_SIDE_MAP[pid].astype(np.int64)
    return df


def spin_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "spinId" not in df.columns:
        df["spinId"] = 0
    table = np.array([(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1)], dtype=np.int64)
    sid = _clip_int_series(df["spinId"], 0, 5)
    mapped = table[sid]
    df["sid_spin"] = mapped[:, 0].astype(np.int64)
    df["sid_side"] = mapped[:, 1].astype(np.int64)
    return df


def _rally_phase_from_strike(strike_number: Sequence[Any]) -> np.ndarray:
    s = pd.Series(strike_number).fillna(0).astype(float)
    return np.select([s <= 1, s == 2, s == 3, s >= 4], [1, 2, 3, 4], default=4).astype(np.float32)


def _safe_norm_series(s: pd.Series, default_max: float = 1.0) -> pd.Series:
    x = _to_num(s, 0.0).astype(float)
    denom = max(float(np.nanmax(x.values)) if len(x) else 0.0, float(default_max), 1.0)
    return (x / denom).clip(0.0, 1.0).astype(np.float32)


def _safe_group_change_rate(s: pd.Series, window: int) -> pd.Series:
    changed = s.ne(s.shift(1)).astype(float)
    if len(changed) > 0:
        changed.iloc[0] = 0.0
    return changed.rolling(window=window, min_periods=1).mean()


def add_mapping_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["sex", "strikeId", "handId", "strengthId", "spinId", "positionId", "actionId", "pointId", "scoreSelf", "scoreOther", "strikeNumber"]:
        if col not in df.columns:
            df[col] = 0
    df = action_transform(df)
    df = point_transform(df)
    df = spin_transform(df)
    return df


def add_sequence_motion_features(df: pd.DataFrame) -> pd.DataFrame:

    df = add_mapping_features(df.copy())
    for col in ["scoreSelf", "scoreOther", "strikeNumber", "actionId", "aid_group", "spinId", "sid_spin", "sid_side", "strengthId", "positionId", "pointId", "pid_side", "pid_depth"]:
        df[col] = _to_num(df[col], 0)

    df["scoreDiff_bl"] = (df["scoreSelf"] - df["scoreOther"]).astype(np.float32)
    df["scoreSum_bl"] = (df["scoreSelf"] + df["scoreOther"]).astype(np.float32)
    df["absScoreDiff_bl"] = df["scoreDiff_bl"].abs().astype(np.float32)
    df["scoreDiff_norm_bl"] = (df["scoreDiff_bl"] / 11.0).clip(-1, 1).astype(np.float32)
    df["scorePressure_bl"] = (1.0 - (df["absScoreDiff_bl"] / 11.0).clip(0, 1)).astype(np.float32)
    df["points_to_win_self_bl"] = ((11.0 - df["scoreSelf"]).clip(lower=0) / 11.0).astype(np.float32)
    df["points_to_win_other_bl"] = ((11.0 - df["scoreOther"]).clip(lower=0) / 11.0).astype(np.float32)
    df["is_tie_score_bl"] = (df["scoreDiff_bl"] == 0).astype(np.float32)
    df["is_leading_bl"] = (df["scoreDiff_bl"] > 0).astype(np.float32)
    df["is_trailing_bl"] = (df["scoreDiff_bl"] < 0).astype(np.float32)
    df["is_close_score_bl"] = (df["absScoreDiff_bl"] <= 2).astype(np.float32)
    df["is_late_game_bl"] = (df["scoreSum_bl"] >= 16).astype(np.float32)
    df["is_overtime_bl"] = ((df["scoreSelf"] >= 10) | (df["scoreOther"] >= 10)).astype(np.float32)
    df["is_deuce_like_bl"] = ((df["scoreSelf"] >= 9) & (df["scoreOther"] >= 9)).astype(np.float32)
    df["is_game_point_like_bl"] = (np.maximum(df["scoreSelf"], df["scoreOther"]) >= 10).astype(np.float32)
    df["leader_game_point_bl"] = ((df["scoreDiff_bl"] > 0) & (df["scoreSelf"] >= 10)).astype(np.float32)
    df["trailer_under_pressure_bl"] = ((df["scoreDiff_bl"] < 0) & (df["scoreOther"] >= 10)).astype(np.float32)

    strike = _to_num(df["strikeNumber"], 0).astype(float)
    phase = _rally_phase_from_strike(strike)
    df["rally_phase_bl"] = (phase / 4.0).astype(np.float32)
    df["strike_log1p_bl"] = (np.log1p(strike) / np.log1p(64.0)).clip(0, 1).astype(np.float32)
    df["is_serve_phase_bl"] = (strike <= 1).astype(np.float32)
    df["is_receive_phase_bl"] = (strike == 2).astype(np.float32)
    df["is_third_ball_phase_bl"] = (strike == 3).astype(np.float32)
    df["is_rally_phase_bl"] = (strike >= 4).astype(np.float32)
    df["is_server_turn_bl"] = ((strike.astype(int) % 2) == 1).astype(np.float32)
    df["is_receiver_turn_bl"] = ((strike.astype(int) % 2) == 0).astype(np.float32)

    df["strength_norm_bl"] = _safe_norm_series(df["strengthId"], default_max=1)
    df["action_group_norm_bl"] = _safe_norm_series(df["aid_group"], default_max=4)
    df["spin_norm_bl"] = _safe_norm_series(df["sid_spin"], default_max=3)
    df["side_spin_bl"] = (df["sid_side"] > 0).astype(np.float32)

    df["is_deep_placement_bl"] = (df["pid_depth"] >= 3).astype(np.float32)
    df["is_short_placement_bl"] = ((df["pid_depth"] == 1) & (df["pointId"] != 0)).astype(np.float32)
    df["is_wide_placement_bl"] = df["pid_side"].isin([1, 3]).astype(np.float32)
    df["is_middle_placement_bl"] = (df["pid_side"] == 2).astype(np.float32)
    df["wide_deep_pressure_bl"] = (df["is_deep_placement_bl"] * df["is_wide_placement_bl"]).astype(np.float32)
    df["placement_edge_pressure_bl"] = ((df["is_wide_placement_bl"] + df["is_deep_placement_bl"] + df["is_short_placement_bl"]) / 3.0).astype(np.float32)
    df["spin_strength_pressure_bl"] = (df["spin_norm_bl"] * df["strength_norm_bl"]).astype(np.float32)
    df["action_spin_pressure_bl"] = (df["action_group_norm_bl"] * df["spin_norm_bl"]).astype(np.float32)
    df["server_score_pressure_bl"] = (df["is_server_turn_bl"] * df["scorePressure_bl"]).astype(np.float32)
    df["receiver_score_pressure_bl"] = (df["is_receiver_turn_bl"] * df["scorePressure_bl"]).astype(np.float32)

    df["action_point_combo_bl"] = (df["actionId"].astype(int).clip(0, 18) * 10 + df["pointId"].astype(int).clip(0, 9)).astype(np.int64)
    df["action_side_combo_bl"] = (df["actionId"].astype(int).clip(0, 18) * 4 + df["pid_side"].astype(int).clip(0, 3)).astype(np.int64)
    df["action_depth_combo_bl"] = (df["actionId"].astype(int).clip(0, 18) * 4 + df["pid_depth"].astype(int).clip(0, 3)).astype(np.int64)
    df["spin_strength_combo_bl"] = (df["spinId"].astype(int).clip(0, 5) * 4 + df["strengthId"].astype(int).clip(0, 3)).astype(np.int64)

    if "rally_uid" in df.columns:
        df["_orig_order_bl"] = np.arange(len(df))
        sort_cols = ["rally_uid", "strikeNumber", "_orig_order_bl"]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        g = df.groupby("rally_uid", sort=False)

        for src, dst in [
            ("pointId", "point_changed_prev_bl"), ("positionId", "position_changed_prev_bl"),
            ("actionId", "action_changed_prev_bl"), ("spinId", "spin_changed_prev_bl"),
            ("strengthId", "strength_changed_prev_bl"), ("pid_side", "side_changed_prev_bl"),
            ("pid_depth", "depth_changed_prev_bl"),
        ]:
            prev = g[src].shift(1)
            df[dst] = ((df[src] != prev) & prev.notna()).astype(np.float32)

        prev_side = g["pid_side"].shift(1)
        prev_depth = g["pid_depth"].shift(1)
        df["side_move_bl"] = ((df["pid_side"] - prev_side).fillna(0.0) / 2.0).clip(-1, 1).astype(np.float32)
        df["depth_move_bl"] = ((df["pid_depth"] - prev_depth).fillna(0.0) / 2.0).clip(-1, 1).astype(np.float32)
        df["placement_move_magnitude_bl"] = ((df["side_move_bl"].abs() + df["depth_move_bl"].abs()).clip(0, 2) / 2.0).astype(np.float32)

        df["aggressive_shot_bl"] = (0.35 * df["strength_norm_bl"] + 0.25 * df["action_group_norm_bl"] + 0.15 * df["wide_deep_pressure_bl"] + 0.15 * df["placement_move_magnitude_bl"] + 0.10 * df["spin_changed_prev_bl"]).clip(0, 1).astype(np.float32)
        df["control_shot_bl"] = (0.35 * (1 - df["strength_norm_bl"]) + 0.25 * df["is_middle_placement_bl"] + 0.20 * (1 - df["placement_move_magnitude_bl"]) + 0.20 * (1 - df["action_changed_prev_bl"])).clip(0, 1).astype(np.float32)

        for phase_col, out_col in [
            ("is_serve_phase_bl", "serve_aggressive_bl"), ("is_receive_phase_bl", "receive_aggressive_bl"),
            ("is_third_ball_phase_bl", "third_ball_aggressive_bl"), ("is_rally_phase_bl", "rally_aggressive_bl"),
            ("is_close_score_bl", "pressure_aggressive_bl"), ("is_late_game_bl", "late_aggressive_bl"),
        ]:
            df[out_col] = (df[phase_col] * df["aggressive_shot_bl"]).astype(np.float32)
        for phase_col, out_col in [
            ("is_serve_phase_bl", "serve_control_bl"), ("is_receive_phase_bl", "receive_control_bl"),
            ("is_third_ball_phase_bl", "third_ball_control_bl"), ("is_rally_phase_bl", "rally_control_bl"),
        ]:
            df[out_col] = (df[phase_col] * df["control_shot_bl"]).astype(np.float32)

        df["roll3_point_change_rate_bl"] = g["pointId"].transform(lambda s: _safe_group_change_rate(s, 3)).astype(np.float32)
        df["roll5_point_change_rate_bl"] = g["pointId"].transform(lambda s: _safe_group_change_rate(s, 5)).astype(np.float32)
        df["roll5_position_change_rate_bl"] = g["positionId"].transform(lambda s: _safe_group_change_rate(s, 5)).astype(np.float32)
        df["roll5_action_change_rate_bl"] = g["actionId"].transform(lambda s: _safe_group_change_rate(s, 5)).astype(np.float32)
        df["roll5_spin_change_rate_bl"] = g["spinId"].transform(lambda s: _safe_group_change_rate(s, 5)).astype(np.float32)
        df["roll5_side_change_rate_bl"] = g["pid_side"].transform(lambda s: _safe_group_change_rate(s, 5)).astype(np.float32)
        df["roll5_depth_change_rate_bl"] = g["pid_depth"].transform(lambda s: _safe_group_change_rate(s, 5)).astype(np.float32)
        df["placement_volatility_bl"] = (df["roll5_point_change_rate_bl"] + df["roll5_position_change_rate_bl"] + df["roll5_action_change_rate_bl"] + df["roll5_spin_change_rate_bl"] + df["roll5_side_change_rate_bl"] + df["roll5_depth_change_rate_bl"]) / 6.0
        df["roll3_aggressive_rate_bl"] = g["aggressive_shot_bl"].transform(lambda s: s.rolling(3, min_periods=1).mean()).astype(np.float32)
        df["roll5_aggressive_rate_bl"] = g["aggressive_shot_bl"].transform(lambda s: s.rolling(5, min_periods=1).mean()).astype(np.float32)
        df["roll3_control_rate_bl"] = g["control_shot_bl"].transform(lambda s: s.rolling(3, min_periods=1).mean()).astype(np.float32)
        df["roll3_wide_deep_rate_bl"] = g["wide_deep_pressure_bl"].transform(lambda s: s.rolling(3, min_periods=1).mean()).astype(np.float32)
        df["tempo_acceleration_bl"] = (df["roll3_aggressive_rate_bl"] - df["roll3_control_rate_bl"]).astype(np.float32)
        df = df.sort_values("_orig_order_bl").drop(columns=["_orig_order_bl"]).reset_index(drop=True)

    for c in BASE_CAT_COLS:
        if c not in df.columns:
            df[c] = 0
        df[c] = _to_num(df[c], 0).astype(np.int64)
    for c in BASE_NUM_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = _to_num(df[c], 0.0).replace([np.inf, -np.inf], 0.0).astype(np.float32)
    return reduce_mem_usage(df)


def safe_change_rate(values: Sequence[Any]) -> float:
    arr = np.asarray(values)
    if len(arr) <= 1:
        return 0.0
    return float(np.mean(arr[1:] != arr[:-1]))


def safe_streak_len(values: Sequence[Any]) -> int:
    vals = list(values)
    if not vals:
        return 0
    last = vals[-1]
    streak = 1
    for v in reversed(vals[:-1]):
        if v == last:
            streak += 1
        else:
            break
    return int(streak)


def _entropy_and_dominance(vals: Sequence[int], n_classes: int) -> Tuple[float, float]:
    vals = np.asarray(vals, dtype=np.int64)
    if len(vals) == 0:
        return 0.0, 0.0
    vals = np.clip(vals, 0, n_classes - 1)
    counts = np.bincount(vals, minlength=n_classes).astype(float)
    total = counts.sum()
    if total <= 0:
        return 0.0, 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p + 1e-12)).sum()), float(counts.max() / total)


def add_ratio_features(out: Dict[str, Any], prefix: pd.DataFrame, col: str, name: str, classes: int, windows: List[Tuple[str, Optional[int]]]) -> None:
    arr_all = prefix[col].to_numpy(np.int64) if col in prefix.columns else np.zeros(len(prefix), dtype=np.int64)
    for w_name, w in windows:
        arr = arr_all if w is None else arr_all[-w:]
        denom = max(len(arr), 1)
        counts = np.bincount(np.clip(arr, 0, classes - 1), minlength=classes).astype(np.float32)[:classes]
        ratio = counts / denom
        for c in range(classes):
            out[f"{w_name}_{name}_ratio_{c}"] = float(ratio[c])


def add_change_features(out: Dict[str, Any], prefix: pd.DataFrame, col: str, name: str, windows: List[Tuple[str, Optional[int]]], include_streak: bool) -> None:
    arr_all = prefix[col].to_numpy(np.int64) if col in prefix.columns else np.zeros(len(prefix), dtype=np.int64)
    for w_name, w in windows:
        arr = arr_all if w is None else arr_all[-w:]
        out[f"{w_name}_{name}_change_rate"] = safe_change_rate(arr)
        if include_streak:
            out[f"{w_name}_{name}_streak_len"] = safe_streak_len(arr)


def build_single_prefix_features(prefix: pd.DataFrame, rally_uid: Any = None, use_long_hist: bool = False, include_streak: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if rally_uid is not None:
        out["rally_uid"] = rally_uid
    last = prefix.iloc[-1]
    prev2 = prefix.iloc[-2] if len(prefix) >= 2 else last

    for col in BASE_CAT_COLS:
        out[f"last_{col}"] = int(last[col]) if col in prefix.columns else 0
        out[f"prev2_{col}"] = int(prev2[col]) if col in prefix.columns else 0
    for col in BASE_NUM_COLS:
        out[f"last_{col}"] = float(last[col]) if col in prefix.columns else 0.0
        out[f"mean_{col}"] = float(prefix[col].mean()) if col in prefix.columns else 0.0
        out[f"max_{col}"] = float(prefix[col].max()) if col in prefix.columns else 0.0

    out["prefix_len"] = int(len(prefix))
    out["prefix_len_log1p"] = float(np.log1p(len(prefix)))

    last_strike = float(last["strikeNumber"]) if "strikeNumber" in prefix.columns else float(len(prefix))
    next_strike = int(last_strike + 1)
    out["next_strikeNumber_tf"] = next_strike
    out["next_is_serve_side_tf"] = int(next_strike % 2 == 1)
    out["next_rally_phase_tf"] = int(min(max(next_strike, 1), 5))
    s_self = float(last["scoreSelf"]) if "scoreSelf" in prefix.columns else 0.0
    s_other = float(last["scoreOther"]) if "scoreOther" in prefix.columns else 0.0
    out["total_points_tf"] = float(s_self + s_other)
    out["score_lead_abs_tf"] = float(abs(s_self - s_other))
    out["points_to_win_self_tf"] = float(max(0.0, 11.0 - s_self))
    out["points_to_win_other_tf"] = float(max(0.0, 11.0 - s_other))
    out["is_deuce_tf"] = int(s_self >= 10 and s_other >= 10)
    out["match_point_self_tf"] = int(s_self >= 10 and s_self - s_other >= 0)
    out["match_point_other_tf"] = int(s_other >= 10 and s_other - s_self >= 0)

    last_a = int(last["actionId"]) if "actionId" in prefix.columns else 0
    last_p = int(last["pointId"]) if "pointId" in prefix.columns else 0
    prev_a = int(prev2["actionId"]) if "actionId" in prefix.columns else last_a
    prev_p = int(prev2["pointId"]) if "pointId" in prefix.columns else last_p
    out["last_action_point_combo_bl"] = int(np.clip(last_a, 0, 18) * 10 + np.clip(last_p, 0, 9))
    out["prev2_action_point_combo_bl"] = int(np.clip(prev_a, 0, 18) * 10 + np.clip(prev_p, 0, 9))

    windows = [("roll3", 3), ("roll5", 5)]
    if use_long_hist:
        windows.append(("roll_all", None))

    for col, name, classes in SHORT_RATIO_FEATURES:
        add_ratio_features(out, prefix, col, name, classes, windows)
    for col, name in [
        ("pid_side", "side"), ("pid_depth", "depth"), ("aid_group", "aid_group"),
        ("actionId", "action"), ("pointId", "point"), ("positionId", "position"),
        ("spinId", "spin"), ("strengthId", "strength"),
    ]:
        add_change_features(out, prefix, col, name, windows, include_streak=include_streak)

    for col in ["actionId", "pointId", "aid_group", "pid_depth", "pid_side", "spinId", "strengthId"]:
        if col in prefix.columns:
            vals = prefix[col].to_numpy(np.int64)
            for i in range(1, 4):
                out[f"lag{i}_{col}"] = int(vals[-i]) if len(vals) >= i else 0

    if use_long_hist:
        ent, dom = _entropy_and_dominance(prefix["actionId"].to_numpy() if "actionId" in prefix.columns else [], NUM_ACTION_CLASSES)
        out["hist_action_entropy_tf"] = ent
        out["hist_action_dominance_tf"] = dom
        ent, dom = _entropy_and_dominance(prefix["pointId"].to_numpy() if "pointId" in prefix.columns else [], NUM_POINT_CLASSES)
        out["hist_point_entropy_tf"] = ent
        out["hist_point_dominance_tf"] = dom

    return out


def make_sliding_feature_df(df: pd.DataFrame, max_seq_len: Optional[int], is_train: bool, use_long_hist: bool, include_streak: bool, desc: str) -> pd.DataFrame:
    df = df.copy()
    df["row_order"] = np.arange(len(df))
    df = df.sort_values(["rally_uid", "strikeNumber", "row_order"]).reset_index(drop=True)
    rows = []
    for rally_uid, g in tqdm(df.groupby("rally_uid", sort=False), desc=desc):
        g = g.sort_values(["strikeNumber", "row_order"]).reset_index(drop=True)
        if len(g) == 0:
            continue
        if is_train:
            if len(g) < 2:
                continue
            for k in range(1, len(g)):
                prefix = g.iloc[:k]
                if max_seq_len is not None:
                    prefix = prefix.iloc[-max_seq_len:]
                feat = build_single_prefix_features(prefix, rally_uid, use_long_hist=use_long_hist, include_streak=include_streak)
                target = g.iloc[k]
                feat["target_action"] = int(np.clip(target.get("actionId", 0), 0, 18))
                feat["target_aid_group"] = int(ACTION_GROUP_MAP[feat["target_action"]])
                feat["target_aid_sub"] = int(ACTION_SUB_MAP[feat["target_action"]])
                feat["target_point"] = int(np.clip(target.get("pointId", 0), 0, 9))
                feat["target_point_valid"] = int(feat["target_point"] != 0)
                feat["target_pid_depth"] = int(POINT_DEPTH_MAP[feat["target_point"]])
                feat["target_pid_side"] = int(POINT_SIDE_MAP[feat["target_point"]])
                if "serverGetPoint" in g.columns:
                    feat["target_win"] = int(g["serverGetPoint"].iloc[0])
                if "match" in g.columns:
                    feat["_group_match"] = g["match"].iloc[0]
                else:
                    feat["_group_match"] = rally_uid
                rows.append(feat)
        else:
            prefix = g
            if max_seq_len is not None:
                prefix = prefix.iloc[-max_seq_len:]
            feat = build_single_prefix_features(prefix, rally_uid, use_long_hist=use_long_hist, include_streak=include_streak)
            rows.append(feat)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.replace([np.inf, -np.inf], np.nan)
    return out


def mirror_point_id(pid: Any) -> int:
    m = {1: 3, 3: 1, 4: 6, 6: 4, 7: 9, 9: 7}
    return int(m.get(int(pid), int(pid)))


def add_hand_flip_augmentation(train: pd.DataFrame) -> pd.DataFrame:
    aug = train.copy()
                                                                               
    aug["rally_uid"] = aug["rally_uid"].astype(str) + "_flip"
    if "handId" in aug.columns:
        aug["handId"] = aug["handId"].replace({1: 2, 2: 1})
    if "positionId" in aug.columns:
        aug["positionId"] = aug["positionId"].replace({1: 3, 3: 1})
    if "pointId" in aug.columns:
        aug["pointId"] = aug["pointId"].apply(mirror_point_id)
    return pd.concat([train, aug], ignore_index=True)


def _select_state_feature_columns(df: pd.DataFrame, max_cols: int = 80) -> List[str]:


    exclude_prefixes = ("target_", "trans_", "prob_", "ag_", "hmm_", "kmeans_")
    exclude_exact = set(RAW_ID_COLS + ["rally_uid", "_group_match"])
    preferred_tokens = [
        "last_", "prev2_", "mean_", "max_", "next_", "prefix_len",
        "score", "phase", "pressure", "point", "action", "aid_", "pid_",
        "side", "depth", "spin", "strength", "placement", "aggressive", "control",
        "tempo", "change_rate", "move",
    ]
    cols: List[str] = []
    for c in df.columns:
        if c in exclude_exact or c.startswith(exclude_prefixes):
            continue
        if c.endswith(RAW_ID_SUFFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if any(tok in c for tok in preferred_tokens):
            cols.append(c)
                                                                                    
    if not cols:
        return []
    variances = df[cols].fillna(0.0).astype(np.float32).var(axis=0).sort_values(ascending=False)
    return [c for c in variances.index[:max_cols] if float(variances[c]) > 1e-10]


def _fill_zero_hmm_features(train: pd.DataFrame, test: pd.DataFrame, n_states: int, reason: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[HMM] {reason}; filling HMM features with zeros.")
    train = train.copy()
    test = test.copy()
    train["hmm_state"] = 0
    test["hmm_state"] = 0
    for i in range(n_states):
        train[f"hmm_prob_{i}"] = 0.0
        test[f"hmm_prob_{i}"] = 0.0
    train["hmm_prob_max"] = 0.0
    test["hmm_prob_max"] = 0.0
    train["hmm_entropy"] = 0.0
    test["hmm_entropy"] = 0.0
    return train, test


def _repair_hmm_probabilities(model, n_states: int):


    try:
        start = np.asarray(model.startprob_, dtype=np.float64)
        if start.shape[0] != n_states or (not np.isfinite(start).all()) or start.sum() <= 0:
            start = np.ones(n_states, dtype=np.float64) / n_states
        else:
            start = np.maximum(start, 1e-9)
            start = start / start.sum()
        model.startprob_ = start
    except Exception:
        model.startprob_ = np.ones(n_states, dtype=np.float64) / n_states

    try:
        trans = np.asarray(model.transmat_, dtype=np.float64)
        if trans.shape != (n_states, n_states):
            trans = np.ones((n_states, n_states), dtype=np.float64) / n_states
        else:
            bad = (~np.isfinite(trans).all(axis=1)) | (trans.sum(axis=1) <= 0)
            trans[~np.isfinite(trans)] = 0.0
            trans = np.maximum(trans, 1e-9)
            row_sum = trans.sum(axis=1, keepdims=True)
            trans = trans / np.maximum(row_sum, 1e-12)
            if bad.any():
                trans[bad] = 1.0 / n_states
        model.transmat_ = trans
    except Exception:
        model.transmat_ = np.ones((n_states, n_states), dtype=np.float64) / n_states
    return model


def _init_hmm_from_kmeans(model, X: np.ndarray, lengths: List[int], n_states: int, seed: int, covariance_type: str):


    km = KMeans(n_clusters=n_states, random_state=seed, n_init=10)
    labels = km.fit_predict(X)

                                                                                          
    start_counts = np.ones(n_states, dtype=np.float64)
    trans_counts = np.ones((n_states, n_states), dtype=np.float64)
    pos = 0
    for L in lengths:
        if L <= 0:
            continue
        seq = labels[pos:pos + L]
        start_counts[int(seq[0])] += 1.0
        if len(seq) >= 2:
            for a, b in zip(seq[:-1], seq[1:]):
                trans_counts[int(a), int(b)] += 1.0
        pos += L
    model.startprob_ = start_counts / start_counts.sum()
    model.transmat_ = trans_counts / trans_counts.sum(axis=1, keepdims=True)

    model.means_ = km.cluster_centers_.astype(np.float64)
    global_var = np.var(X, axis=0).astype(np.float64) + 1e-2
    if covariance_type == "diag":
        covars = []
        for k in range(n_states):
            pts = X[labels == k]
            if len(pts) >= 2:
                covars.append(np.var(pts, axis=0).astype(np.float64) + 1e-2)
            else:
                covars.append(global_var.copy())
        model.covars_ = np.asarray(covars, dtype=np.float64)
    elif covariance_type == "spherical":
        covars = []
        for k in range(n_states):
            pts = X[labels == k]
            if len(pts) >= 2:
                covars.append(float(np.var(pts) + 1e-2))
            else:
                covars.append(float(np.mean(global_var)))
        model.covars_ = np.asarray(covars, dtype=np.float64)
    return model


def add_hmm_state_features(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    n_states: int = 8,
    seed: int = SEED,
    max_iter: int = 80,
    max_cols: int = 80,
    covariance_type: str = "diag",
) -> Tuple[pd.DataFrame, pd.DataFrame]:


    train = train_feat.copy()
    test = test_feat.copy()
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception as e:
        return _fill_zero_hmm_features(train, test, n_states, f"hmmlearn unavailable ({e}). Install: pip install hmmlearn")

    feat_cols = _select_state_feature_columns(train, max_cols=max_cols)
    if len(feat_cols) < 2 or len(train) < max(n_states * 5, 20):
        return _fill_zero_hmm_features(train, test, n_states, "not enough usable numeric state features")

    train["_hmm_orig_order"] = np.arange(len(train))
    if "rally_uid" in train.columns:
        sort_cols = ["rally_uid"]
        if "next_strikeNumber_tf" in train.columns:
            sort_cols.append("next_strikeNumber_tf")
        sorted_train = train.sort_values(sort_cols + ["_hmm_orig_order"]).reset_index(drop=True)
        group_sizes = sorted_train.groupby("rally_uid", sort=False).size().astype(int)
                                                                                                        
        good_rallies = group_sizes[group_sizes >= 2].index
        fit_train = sorted_train[sorted_train["rally_uid"].isin(good_rallies)].reset_index(drop=True)
        lengths = fit_train.groupby("rally_uid", sort=False).size().astype(int).tolist()
        if len(fit_train) < max(20, n_states * 4) or len(lengths) < 3:
            fit_train = sorted_train.copy().reset_index(drop=True)
            lengths = [len(fit_train)]
    else:
        sorted_train = train.copy().reset_index(drop=True)
        fit_train = sorted_train.copy().reset_index(drop=True)
        lengths = [len(fit_train)]

                                                                                                           
    usable_states = int(min(n_states, max(2, len(fit_train) // 20), max(2, len(lengths) * 2)))
    usable_states = max(2, usable_states)
    if usable_states < n_states:
        print(f"[HMM] reducing states {n_states} -> {usable_states} for stable fit")

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(fit_train[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32))
    X_all = scaler.transform(sorted_train[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32))
    X_test = scaler.transform(test.reindex(columns=feat_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32))
    X_fit = np.nan_to_num(X_fit, nan=0.0, posinf=0.0, neginf=0.0)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    try:
                         
                                                                                   
        model = GaussianHMM(
            n_components=usable_states,
            covariance_type=covariance_type,
            n_iter=0,
            random_state=seed,
            verbose=False,
            init_params="",
            params="",
            min_covar=1e-3,
        )
        model = _init_hmm_from_kmeans(model, X_fit, lengths, usable_states, seed, covariance_type)
        model.n_features = X_fit.shape[1]
        model = _repair_hmm_probabilities(model, usable_states)

                                                                                                 
        state_sorted = model.predict(X_all).astype(np.int64)
        prob_sorted_small = model.predict_proba(X_all).astype(np.float32)
        state_test = model.predict(X_test).astype(np.int64)
        prob_test_small = model.predict_proba(X_test).astype(np.float32)
    except Exception as e:
        return _fill_zero_hmm_features(train.drop(columns=["_hmm_orig_order"], errors="ignore"), test, n_states, f"GaussianHMM failed after robust init ({e})")

                                                                                                      
    prob_sorted = np.zeros((len(sorted_train), n_states), dtype=np.float32)
    prob_test = np.zeros((len(test), n_states), dtype=np.float32)
    prob_sorted[:, :usable_states] = prob_sorted_small
    prob_test[:, :usable_states] = prob_test_small

    sorted_train["hmm_state"] = state_sorted
    test["hmm_state"] = state_test
    for i in range(n_states):
        sorted_train[f"hmm_prob_{i}"] = prob_sorted[:, i]
        test[f"hmm_prob_{i}"] = prob_test[:, i]
    sorted_train["hmm_prob_max"] = prob_sorted.max(axis=1).astype(np.float32)
    test["hmm_prob_max"] = prob_test.max(axis=1).astype(np.float32)
    sorted_train["hmm_entropy"] = (-(prob_sorted * np.log(prob_sorted + 1e-12)).sum(axis=1)).astype(np.float32)
    test["hmm_entropy"] = (-(prob_test * np.log(prob_test + 1e-12)).sum(axis=1)).astype(np.float32)

    train = sorted_train.sort_values("_hmm_orig_order").drop(columns=["_hmm_orig_order"]).reset_index(drop=True)
    test = test.reset_index(drop=True)
    print(f"[HMM] added quiet KMeans-initialized GaussianHMM states: requested={n_states}, fitted={usable_states}, features={len(feat_cols)}")
    return train, test


def add_kmeans_state_features(train_feat: pd.DataFrame, test_feat: pd.DataFrame, n_clusters: int = 8, seed: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train = train_feat.copy()
    test = test_feat.copy()
    exclude = [c for c in train.columns if c.startswith("target_") or c == "rally_uid" or c.startswith("_group")]
    feat_cols = [c for c in train.columns if c not in exclude and pd.api.types.is_numeric_dtype(train[c])]
    if len(feat_cols) == 0 or len(train) < n_clusters:
        train["kmeans_state"] = 0
        test["kmeans_state"] = 0
        return train, test
    Xtr = train[feat_cols].fillna(0.0).astype(np.float32).to_numpy()
    Xte = test.reindex(columns=feat_cols).fillna(0.0).astype(np.float32).to_numpy()
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    train["kmeans_state"] = km.fit_predict(Xtr).astype(np.int64)
    test["kmeans_state"] = km.predict(Xte).astype(np.int64)
                                                                           
    try:
        dtr = km.transform(Xtr)
        dte = km.transform(Xte)
        for i in range(n_clusters):
            train[f"kmeans_dist_{i}"] = dtr[:, i].astype(np.float32)
            test[f"kmeans_dist_{i}"] = dte[:, i].astype(np.float32)
        train["kmeans_dist_min"] = dtr.min(axis=1).astype(np.float32)
        test["kmeans_dist_min"] = dte.min(axis=1).astype(np.float32)
    except Exception:
        pass
    return train, test


class TransitionProbabilityEncoder:
    def __init__(self, specs: List[Tuple[str, List[str]]], target_col: str, num_classes: int, alpha: float = 1.0):
        self.specs = specs
        self.target_col = target_col
        self.num_classes = num_classes
        self.alpha = alpha
        self.global_prob: Optional[np.ndarray] = None
        self.tables: Dict[str, Dict[Tuple[Any, ...], np.ndarray]] = {}

    def fit(self, df: pd.DataFrame) -> "TransitionProbabilityEncoder":
        y = df[self.target_col].astype(int).clip(0, self.num_classes - 1).to_numpy()
        counts = np.bincount(y, minlength=self.num_classes).astype(float)[: self.num_classes]
        self.global_prob = (counts + self.alpha) / (counts.sum() + self.alpha * self.num_classes)
        for name, cols in self.specs:
            table: Dict[Tuple[Any, ...], np.ndarray] = {}
            if not all(c in df.columns for c in cols):
                self.tables[name] = table
                continue
            tmp = df[cols + [self.target_col]].copy()
            for key, g in tmp.groupby(cols, sort=False):
                if not isinstance(key, tuple):
                    key = (key,)
                yy = g[self.target_col].astype(int).clip(0, self.num_classes - 1).to_numpy()
                cc = np.bincount(yy, minlength=self.num_classes).astype(float)[: self.num_classes]
                table[key] = (cc + self.alpha) / (cc.sum() + self.alpha * self.num_classes)
            self.tables[name] = table
        return self

    def transform(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if self.global_prob is None:
            raise RuntimeError("Transition encoder is not fitted")
        out = pd.DataFrame(index=df.index)
        for name, cols in self.specs:
            table = self.tables.get(name, {})
            probs = np.zeros((len(df), self.num_classes), dtype=np.float32)
            for i, (_, row) in enumerate(df.iterrows()):
                key = tuple(row[c] if c in df.columns else None for c in cols)
                probs[i] = table.get(key, self.global_prob)
            for c in range(self.num_classes):
                out[f"{prefix}_{name}_prob_{c}"] = probs[:, c]
        return out

    def predict_prior_matrix(self, df: pd.DataFrame) -> np.ndarray:
        if self.global_prob is None:
            raise RuntimeError("Transition encoder is not fitted")
        mats = []
        for name, cols in self.specs:
            table = self.tables.get(name, {})
            probs = np.zeros((len(df), self.num_classes), dtype=np.float32)
            for i, (_, row) in enumerate(df.iterrows()):
                key = tuple(row[c] if c in df.columns else None for c in cols)
                probs[i] = table.get(key, self.global_prob)
            mats.append(probs)
        if not mats:
            return np.tile(self.global_prob.astype(np.float32), (len(df), 1))
        prior = np.mean(mats, axis=0)
        prior = prior / np.maximum(prior.sum(axis=1, keepdims=True), 1e-12)
        return prior.astype(np.float32)


def add_transition_features(train_feat: pd.DataFrame, test_feat: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, TransitionProbabilityEncoder, TransitionProbabilityEncoder]:
    train = train_feat.copy()
    test = test_feat.copy()
    aenc = TransitionProbabilityEncoder(TRANSITION_SPECS_ACTION, "target_action", NUM_ACTION_CLASSES, alpha=1.0).fit(train)
    penc = TransitionProbabilityEncoder(TRANSITION_SPECS_POINT, "target_point", NUM_POINT_CLASSES, alpha=1.0).fit(train)
    train = pd.concat([train, aenc.transform(train, "tr_action"), penc.transform(train, "tr_point")], axis=1)
    test = pd.concat([test, aenc.transform(test, "tr_action"), penc.transform(test, "tr_point")], axis=1)
    return train, test, aenc, penc


def drop_raw_id_features(df: pd.DataFrame, extra_drop: Optional[Iterable[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    drop_cols = set(extra_drop or [])
    for c in df.columns:
        if c in RAW_ID_COLS or c.startswith("_group"):
            drop_cols.add(c)
        if c.endswith(RAW_ID_SUFFIXES):
            drop_cols.add(c)
                                                                                                 
        for raw in ["match", "rally_id", "numberGame", "gamePlayerId", "gamePlayerOtherId"]:
            if c == raw or c.startswith(f"last_{raw}") or c.startswith(f"prev2_{raw}") or c.startswith(f"mean_{raw}") or c.startswith(f"max_{raw}"):
                drop_cols.add(c)
    keep = [c for c in df.columns if c not in drop_cols]
    return df[keep].copy(), sorted([c for c in drop_cols if c in df.columns])


def coerce_features_for_ag(train_x: pd.DataFrame, test_x: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train_x.copy()
    test = test_x.copy()
    cols = sorted(set(train.columns).union(test.columns))
    train = train.reindex(columns=cols)
    test = test.reindex(columns=cols)
    for c in cols:
        if train[c].dtype == "O" or test[c].dtype == "O":
                                                              
            tr_num = pd.to_numeric(train[c], errors="coerce")
            te_num = pd.to_numeric(test[c], errors="coerce")
            if tr_num.notna().mean() > 0.95 and te_num.notna().mean() > 0.95:
                train[c] = tr_num.fillna(0.0).astype(np.float32)
                test[c] = te_num.fillna(0.0).astype(np.float32)
            else:
                train[c] = train[c].astype(str).fillna("__NA__")
                test[c] = test[c].astype(str).fillna("__NA__")
        else:
            train[c] = pd.to_numeric(train[c], errors="coerce").fillna(0)
            test[c] = pd.to_numeric(test[c], errors="coerce").fillna(0)
    return train, test


def get_ag_hyperparameters(
    no_catboost: bool = True,
    model_suite: str = "fast",
    n_classes: Optional[int] = None,
    task_name: str = "",
) -> Dict[str, Any]:


    gbm_variants = [
        {},
        {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
        {"learning_rate": 0.03, "num_leaves": 128, "feature_fraction": 0.90, "min_data_in_leaf": 3, "ag_args": {"name_suffix": "Large", "priority": 0}},
    ]
    rf_variants = [
        {"criterion": "gini", "ag_args": {"name_suffix": "Gini", "problem_types": ["binary", "multiclass"]}},
        {"criterion": "entropy", "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]}},
    ]
    knn_variants = [
        {"weights": "uniform", "ag_args": {"name_suffix": "Unif"}},
        {"weights": "distance", "ag_args": {"name_suffix": "Dist"}},
    ]

    suite = (model_suite or "fast").lower()
    if suite in {"fast", "original"}:
        hp: Dict[str, Any] = {
            "GBM": [{}, {"extra_trees": True, "ag_args": {"name_suffix": "XT"}}],
            "XGB": {},
            "RF": {},
            "XT": {},
            "NN_TORCH": {},
        }
        if not no_catboost:
            hp["CAT"] = {}
        return hp

    if suite in {"practical", "task_aware", "task-aware"}:
                                                                                                        
        hp: Dict[str, Any] = {
            "GBM": gbm_variants,
            "XGB": {},
            "RF": rf_variants,
            "XT": rf_variants,
            "KNN": knn_variants,
            "NN_TORCH": {},
            "FASTAI": {},
            "REALMLP": {},
        }
        if not no_catboost:
            hp["CAT"] = {}

                                                                                                    
        task_l = (task_name or "").lower()
        is_action_main = "action_main" in task_l
        if suite in {"task_aware", "task-aware"} and (n_classes is not None) and n_classes <= 10 and not is_action_main:
            hp["LR"] = {}
            hp["TABPFNMIX"] = {}
            hp["TABDPT"] = {}
        return hp

    hp = {
        "GBM": gbm_variants,
        "CAT": {},
        "XGB": {},
        "RF": rf_variants,
        "XT": rf_variants,
        "KNN": knn_variants,
        "LR": {},
        "NN_TORCH": {},
        "FASTAI": {},
        "EBM": {},
        "REALMLP": {},
        "TABM": {},
    }
    if no_catboost and suite not in {"all", "everything", "all_models"}:
        hp.pop("CAT", None)

    if suite in {"everything", "extreme", "all_models"}:
                                                                            
                                                                                                     
        hp.update({
            "MITRA": {},
            "TABICL": {},
            "TABPFNMIX": {},
            "REALTABPFN-V2": {},
            "REALTABPFN-V2.5": {},
            "TABDPT": {},
            "FT_TRANSFORMER": {},
        })
    return hp

def _prob_matrix_from_predictor(predictor: Any, df: pd.DataFrame, classes: Sequence[int]) -> np.ndarray:
    proba = predictor.predict_proba(df, as_pandas=True)
    arr = np.zeros((len(df), len(classes)), dtype=np.float32)
                                                                        
    for j, cls in enumerate(classes):
        if cls in proba.columns:
            arr[:, j] = proba[cls].to_numpy(dtype=np.float32)
        elif str(cls) in proba.columns:
            arr[:, j] = proba[str(cls)].to_numpy(dtype=np.float32)
        else:
            arr[:, j] = 0.0
    s = arr.sum(axis=1, keepdims=True)
    bad = s.squeeze() <= 0
    if bad.any():
        arr[bad] = 1.0 / len(classes)
        s = arr.sum(axis=1, keepdims=True)
    return arr / np.maximum(s, 1e-12)


def _audit_requested_vs_fitted(requested_hp: Dict[str, Any], leaderboard_df: Optional[pd.DataFrame]) -> pd.DataFrame:


    patterns = {
        "GBM": ["LightGBM"],
        "CAT": ["CatBoost"],
        "XGB": ["XGBoost"],
        "RF": ["RandomForest"],
        "XT": ["ExtraTrees"],
        "KNN": ["KNeighbors"],
        "LR": ["LinearModel", "LogisticRegression"],
        "NN_TORCH": ["NeuralNetTorch"],
        "FASTAI": ["NeuralNetFastAI"],
        "EBM": ["Explainable", "EBM"],
        "REALMLP": ["RealMLP"],
        "TABM": ["TabM"],
        "MITRA": ["Mitra", "MITRA"],
        "TABICL": ["TabICL", "TABICL"],
        "TABPFNMIX": ["TabPFNMix", "TABPFNMIX"],
        "REALTABPFN-V2": ["RealTabPFN", "REALTABPFN-V2"],
        "REALTABPFN-V2.5": ["RealTabPFN", "REALTABPFN-V2.5"],
        "TABDPT": ["TabDPT", "TABDPT"],
        "FT_TRANSFORMER": ["FTTransformer", "FT_TRANSFORMER"],
    }
    models = []
    if leaderboard_df is not None and "model" in leaderboard_df.columns:
        models = leaderboard_df["model"].astype(str).tolist()
    rows = []
    for key in requested_hp.keys():
        pats = patterns.get(key, [key])
        matched = [m for m in models if any(p in m for p in pats)]
        rows.append({
            "requested_model_key": key,
            "requested": True,
            "fitted_or_appeared_in_leaderboard": bool(matched),
            "matched_models": ";".join(matched[:20]),
            "matched_count": len(matched),
        })
    return pd.DataFrame(rows)

def train_ag_oof(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    label_col: str,
    classes: Sequence[int],
    out_dir: str,
    groups: Optional[np.ndarray],
    folds: int,
    presets: str,
    time_limit: Optional[int],
    eval_metric: str,
    no_catboost: bool,
    seed: int,
    name: str,
    model_suite: str = "fast",
    fit_strategy: str = "sequential",
    ag_num_gpus: int = 0,
    ag_max_memory_usage_ratio: float = 1.45,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    from autogluon.tabular import TabularPredictor

    ensure_dir(out_dir)
    y_all = train_feat[label_col].astype(int).to_numpy()
    valid_mask = np.isin(y_all, list(classes))
    train_sub = train_feat.loc[valid_mask].reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    y = train_sub[label_col].astype(int).to_numpy()
    if groups is not None:
        groups_sub = np.asarray(groups)[valid_mask]
    else:
        groups_sub = np.arange(len(train_sub))

    full_oof = np.zeros((len(train_feat), len(classes)), dtype=np.float32)
    test_probs_folds = []
    fold_infos = []

    n_splits = min(folds, max(2, len(np.unique(groups_sub))))
    splitter = GroupKFold(n_splits=n_splits)
    split_iter = splitter.split(train_sub, y, groups_sub)

    feature_drop = [c for c in train_sub.columns if c.startswith("target_") or c == "_orig_idx"]
    X_all, X_test = drop_raw_id_features(train_sub.drop(columns=feature_drop, errors="ignore"), extra_drop=[])[0], drop_raw_id_features(test_feat.copy(), extra_drop=[])[0]
    X_all, X_test = coerce_features_for_ag(X_all, X_test)
    feature_cols = X_all.columns.tolist()

    for fold, (tr_idx, va_idx) in enumerate(split_iter):
        fold_path = os.path.join(out_dir, f"{name}_fold{fold}")
        if os.path.exists(fold_path):
            shutil.rmtree(fold_path)
        tr_df = X_all.iloc[tr_idx].copy()
        va_df = X_all.iloc[va_idx].copy()
        te_df = X_test.copy()
        tr_df[label_col] = y[tr_idx]
        va_y = y[va_idx]
        predictor = TabularPredictor(label=label_col, eval_metric=eval_metric, path=fold_path, verbosity=2)
        requested_hp = get_ag_hyperparameters(no_catboost=no_catboost, model_suite=model_suite, n_classes=len(classes), task_name=name)
        requested_keys = list(requested_hp.keys())
        print(f"[AG-MODELS] {name} fold={fold} requested_keys={requested_keys} total_keys={len(requested_keys)} fit_strategy={fit_strategy}")
        with open(os.path.join(out_dir, f"requested_models_{name}_fold{fold}.json"), "w", encoding="utf-8") as f:
            json.dump({"name": name, "fold": int(fold), "model_suite": model_suite, "requested_model_keys": requested_keys, "hyperparameters": requested_hp}, f, ensure_ascii=False, indent=2, default=str)
                                                                                    
                                                                                       
        fit_kwargs = dict(
            train_data=tr_df,
            presets=presets,
            hyperparameters=requested_hp,
            ag_args_fit={"num_gpus": int(ag_num_gpus), "ag.max_memory_usage_ratio": float(ag_max_memory_usage_ratio)},
            fit_strategy=fit_strategy,
            dynamic_stacking=False,
            num_stack_levels=0,
            num_bag_folds=0,
            num_bag_sets=1,
            save_bag_folds=False,
        )
                                                                             
                                                                                          
        if time_limit is not None and time_limit > 0:
            fit_kwargs["time_limit"] = int(time_limit)
        predictor.fit(**fit_kwargs)
        va_prob = _prob_matrix_from_predictor(predictor, va_df, classes)
        te_prob = _prob_matrix_from_predictor(predictor, te_df, classes)
        orig_idx = train_sub.iloc[va_idx]["_orig_idx"].to_numpy()
        full_oof[orig_idx] = va_prob
        test_probs_folds.append(te_prob)
        pred = va_prob.argmax(axis=1)
        score = f1_score(va_y, pred, labels=list(range(len(classes))), average="macro", zero_division=0) if len(classes) > 2 else None
        lb = None
        try:
            lb = predictor.leaderboard(silent=True)
            lb.to_csv(os.path.join(out_dir, f"leaderboard_{name}_fold{fold}.csv"), index=False)
        except Exception:
            pass
        try:
            audit_df = _audit_requested_vs_fitted(requested_hp, lb)
            audit_df.to_csv(os.path.join(out_dir, f"model_audit_{name}_fold{fold}.csv"), index=False)
            missing = audit_df.loc[~audit_df["fitted_or_appeared_in_leaderboard"], "requested_model_key"].tolist()
            if missing:
                print(f"[AG-MODELS][WARN] {name} fold={fold} missing_or_failed={missing}")
            else:
                print(f"[AG-MODELS] {name} fold={fold} all requested model keys appeared in leaderboard.")
        except Exception as e:
            print(f"[AG-MODELS][WARN] audit failed for {name} fold={fold}: {e}")
        fold_infos.append({"name": name, "fold": fold, "n_train": int(len(tr_idx)), "n_valid": int(len(va_idx)), "macro_f1_indexed": None if score is None else float(score)})
        print(f"[AG] {name} fold={fold} done score(indexed)={score}")

                                                       
    if (~valid_mask).any():
        counts = np.bincount(y, minlength=len(classes)).astype(float)[: len(classes)]
        prior = (counts + 1.0) / (counts.sum() + len(classes))
        full_oof[~valid_mask] = prior
    test_prob = np.mean(test_probs_folds, axis=0) if test_probs_folds else np.ones((len(test_feat), len(classes)), dtype=np.float32) / len(classes)
                                     
    with open(os.path.join(out_dir, f"feature_columns_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    return full_oof, test_prob.astype(np.float32), fold_infos


def get_metric_sweep_list(label_col: str, classes: Sequence[int], base_metric: str, metric_suite: str) -> List[str]:
    suite = (metric_suite or "single").lower()
    if suite in {"single", "base", "none"}:
        return [base_metric]
                                                                                              
    if len(classes) == 2:
                                                                                                                  
        metrics = ["roc_auc", "log_loss", "balanced_accuracy", "accuracy", "f1"]
    else:
                                                                                                                    
                                                                                                                                                       
        metrics = ["f1_macro", "log_loss", "accuracy"]
    out = []
    for m in [base_metric] + metrics:
        if m not in out:
            out.append(m)
    return out


def train_ag_oof_sweep(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    label_col: str,
    classes: Sequence[int],
    out_dir: str,
    groups: Optional[np.ndarray],
    folds: int,
    presets: str,
    time_limit: Optional[int],
    eval_metric: str,
    no_catboost: bool,
    seed: int,
    name: str,
    model_suite: str = "fast",
    metric_suite: str = "single",
    fit_strategy: str = "sequential",
    ag_num_gpus: int = 0,
    ag_max_memory_usage_ratio: float = 1.45,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    metrics = get_metric_sweep_list(label_col, classes, eval_metric, metric_suite)
    if len(metrics) == 1:
        return train_ag_oof(train_feat, test_feat, label_col, classes, out_dir, groups, folds, presets, time_limit, eval_metric, no_catboost, seed, name, model_suite=model_suite, fit_strategy=fit_strategy, ag_num_gpus=ag_num_gpus, ag_max_memory_usage_ratio=ag_max_memory_usage_ratio)

    print(f"[AG-SWEEP] {name}: model_suite={model_suite}, metrics={metrics}, time_limit_each={'unlimited' if time_limit is None else time_limit}")
    oofs, tests, infos_all = [], [], []
    metric_scores = []
    y_true = train_feat[label_col].astype(int).to_numpy()
    for m in metrics:
        run_name = f"{name}__metric_{m.replace('/', '_')}"
        try:
            oof, test, infos = train_ag_oof(
                train_feat, test_feat, label_col, classes, out_dir, groups, folds, presets,
                time_limit, m, no_catboost, seed, run_name, model_suite=model_suite, fit_strategy=fit_strategy, ag_num_gpus=ag_num_gpus, ag_max_memory_usage_ratio=ag_max_memory_usage_ratio,
            )
            pred = np.asarray(classes)[oof.argmax(axis=1)]
            if len(classes) == 2 and label_col == "target_win":
                try:
                    primary = roc_auc_score(y_true, oof[:, 1])
                except Exception:
                    primary = balanced_accuracy_score(y_true, pred)
            else:
                primary = f1_score(y_true, pred, labels=list(classes), average="macro", zero_division=0)
            oofs.append(oof); tests.append(test)
            metric_scores.append(max(float(primary), 1e-6))
            for info in infos:
                info["sweep_metric"] = m
                info["sweep_primary_score"] = float(primary)
            infos_all.extend(infos)
            print(f"[AG-SWEEP] {name} metric={m} primary_score={primary:.6f}")
        except Exception as e:
            print(f"[AG-SWEEP][WARN] {name} metric={m} failed: {repr(e)}")

    if not oofs:
        print(f"[AG-SWEEP][FALLBACK] all metrics failed for {name}; retry base metric with fast suite")
        return train_ag_oof(train_feat, test_feat, label_col, classes, out_dir, groups, folds, presets, time_limit, eval_metric, no_catboost, seed, name, model_suite="fast", fit_strategy=fit_strategy, ag_num_gpus=ag_num_gpus, ag_max_memory_usage_ratio=ag_max_memory_usage_ratio)

    weights = np.asarray(metric_scores, dtype=np.float64)
    weights = weights / max(weights.sum(), 1e-12)
    oof_blend = np.zeros_like(oofs[0], dtype=np.float64)
    test_blend = np.zeros_like(tests[0], dtype=np.float64)
    for w, oof, test in zip(weights, oofs, tests):
        oof_blend += w * oof
        test_blend += w * test
    oof_blend = (oof_blend / np.maximum(oof_blend.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)
    test_blend = (test_blend / np.maximum(test_blend.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)
    summary = pd.DataFrame({"metric": metrics[:len(metric_scores)], "primary_score": metric_scores, "blend_weight": weights})
    ensure_dir(out_dir)
    summary.to_csv(os.path.join(out_dir, f"metric_sweep_{name}.csv"), index=False)
    print(f"[AG-SWEEP] {name} blended weights saved.")
    return oof_blend, test_blend, infos_all


def scaled_time_limit(base: Optional[int], ratio: float, floor: int) -> Optional[int]:

    if base is None or base <= 0:
        return None
    return max(floor, int(base * ratio))


def expand_action_15_probs(prob15: np.ndarray) -> np.ndarray:
    out = np.zeros((prob15.shape[0], NUM_ACTION_CLASSES), dtype=np.float32)
    out[:, :NUM_ACTION_MAIN_CLASSES] = prob15[:, :NUM_ACTION_MAIN_CLASSES]
    s = out.sum(axis=1, keepdims=True)
    return out / np.maximum(s, 1e-12)


def legal_action_mask(probs: np.ndarray, feat: pd.DataFrame) -> np.ndarray:
    out = probs.copy().astype(np.float32)
                                                                                                                             
    if "next_strikeNumber_tf" in feat.columns:
        mask_nonserve = feat["next_strikeNumber_tf"].fillna(0).astype(int).to_numpy() != 1
    else:
        mask_nonserve = np.ones(len(feat), dtype=bool)
    out[mask_nonserve, 15:19] = 0.0
    s = out.sum(axis=1, keepdims=True)
    bad = s.squeeze() <= 0
    if bad.any():
        out[bad, :NUM_ACTION_MAIN_CLASSES] = 1.0 / NUM_ACTION_MAIN_CLASSES
        s = out.sum(axis=1, keepdims=True)
    return out / np.maximum(s, 1e-12)


def combine_point_hierarchy(main_prob: np.ndarray, valid_prob: Optional[np.ndarray], depth_prob: Optional[np.ndarray], side_prob: Optional[np.ndarray], weight_main: float = 0.65) -> np.ndarray:
    if valid_prob is None or depth_prob is None or side_prob is None:
        return main_prob
    hier = np.zeros_like(main_prob, dtype=np.float32)
    p_invalid = valid_prob[:, 0] if valid_prob.shape[1] >= 2 else 1 - valid_prob[:, 0]
    p_valid = valid_prob[:, 1] if valid_prob.shape[1] >= 2 else valid_prob[:, 0]
    hier[:, 0] = p_invalid
    for k in range(1, NUM_POINT_CLASSES):
        d = POINT_DEPTH_MAP[k]
        s = POINT_SIDE_MAP[k]
        pd = depth_prob[:, d] if d < depth_prob.shape[1] else 0
        ps = side_prob[:, s] if s < side_prob.shape[1] else 0
        hier[:, k] = p_valid * pd * ps
    hier = hier / np.maximum(hier.sum(axis=1, keepdims=True), 1e-12)
    out = weight_main * main_prob + (1 - weight_main) * hier
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def fuse_transition(model_prob: np.ndarray, prior_prob: np.ndarray, alpha: float) -> np.ndarray:
    p = np.power(np.maximum(model_prob, 1e-12), alpha) * np.power(np.maximum(prior_prob, 1e-12), 1 - alpha)
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)


def apply_temperature(prob: np.ndarray, T: float) -> np.ndarray:
    logp = np.log(np.maximum(prob, 1e-12)) / max(T, 1e-6)
    logp = logp - logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)


def find_best_temperature(oof_prob: np.ndarray, y: np.ndarray, classes: Sequence[int], metric: str = "macro_f1") -> float:
    best_T, best_score = 1.0, -1.0
    for T in np.linspace(0.70, 1.50, 17):
        p = apply_temperature(oof_prob, float(T))
        pred_idx = p.argmax(axis=1)
        class_arr = np.asarray(classes)
        pred = class_arr[pred_idx]
        if metric == "auc" and len(classes) == 2:
            try:
                score = roc_auc_score(y, p[:, 1])
            except Exception:
                score = 0.0
        else:
            score = f1_score(y, pred, labels=list(classes), average="macro", zero_division=0)
        if score > best_score:
            best_score, best_T = score, float(T)
    return best_T


def fit_class_scales(oof_prob: np.ndarray, y: np.ndarray, classes: Sequence[int], strength_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)) -> np.ndarray:
    class_arr = np.asarray(classes)
    y = np.asarray(y).astype(int)
    counts = np.array([(y == c).sum() for c in classes], dtype=float)
    prior = (counts + 1.0) / (counts.sum() + len(classes))
    inv = np.power(1.0 / np.maximum(prior, 1e-12), 0.5)
    inv = inv / np.mean(inv)
    best_scale = np.ones(len(classes), dtype=np.float32)
    best_score = -1.0
    for s in strength_grid:
        scale = np.power(inv, s)
        p = oof_prob * scale[None, :]
        pred = class_arr[p.argmax(axis=1)]
        score = f1_score(y, pred, labels=list(classes), average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_scale = scale.astype(np.float32)
    return best_scale


def apply_scales(prob: np.ndarray, scales: np.ndarray) -> np.ndarray:
    p = prob * scales[None, :]
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)


def build_rule_tables(train_feat: pd.DataFrame, target_col: str, context_cols: List[str], min_count: int) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    table = {}
    cols = [c for c in context_cols if c in train_feat.columns]
    if not cols:
        return table
    for key, g in train_feat.groupby(cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        if len(g) < min_count:
            continue
        counts = g[target_col].astype(int).value_counts()
        table[key] = {"seen": set(counts.index.astype(int).tolist()), "mode": int(counts.idxmax()), "n": int(len(g))}
    return table


def apply_rule_override(pred: np.ndarray, feat: pd.DataFrame, table: Dict[Tuple[Any, ...], Dict[str, Any]], context_cols: List[str]) -> Tuple[np.ndarray, int]:
    if not table:
        return pred, 0
    cols = [c for c in context_cols if c in feat.columns]
    out = pred.copy()
    changed = 0
    for i, (_, row) in enumerate(feat.iterrows()):
        key = tuple(row[c] for c in cols)
        item = table.get(key)
        if item is None:
            continue
        if int(out[i]) not in item["seen"]:
            out[i] = int(item["mode"])
            changed += 1
    return out, changed


def save_distribution(out_dir: str, name: str, sub: pd.DataFrame) -> Dict[str, Any]:
    info = {"name": name}
    for col in ["actionId", "pointId"]:
        dist = sub[col].value_counts(normalize=True).sort_index().to_dict()
        info[col] = {str(k): float(v) for k, v in dist.items()}
    info["server_mean"] = float(pd.to_numeric(sub["serverGetPoint"], errors="coerce").mean())
    with open(os.path.join(out_dir, f"distribution_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


def find_best_binary_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:


    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.5
    qs = np.linspace(0.01, 0.99, 99)
    thresholds = np.unique(np.quantile(y_prob, qs))
    best_thr, best_score = 0.5, -1.0
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = float(score)
            best_thr = float(thr)
    return best_thr, best_score


def make_submission(
    rally_uid: Sequence[Any],
    action_pred: np.ndarray,
    point_pred: np.ndarray,
    win_prob: np.ndarray,
    server_threshold: float = 0.5,
    hard_server: bool = True,
) -> pd.DataFrame:
    win_prob = np.clip(win_prob.astype(float), 0.0, 1.0)
    server_col = (win_prob >= float(server_threshold)).astype(int) if hard_server else win_prob
    return pd.DataFrame({
        "rally_uid": rally_uid,
        "actionId": action_pred.astype(int),
        "pointId": point_pred.astype(int),
        "serverGetPoint": server_col,
    })


def run_pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    diag_dir = os.path.join(args.out_dir, "diagnostics")
    ensure_dir(diag_dir)

    print("[LOAD]", args.train_path, args.test_path)
    print(f"[AG] per-fit time_limit={args.ag_time_limit}s; every requested AutoGluon model should be attempted unless it fails due to dependencies/resources.")
    print(f"[AG] fit_strategy={args.fit_strategy}, ag_num_gpus={args.ag_num_gpus}")
    if args.fit_strategy == "parallel":
        print("[AG] Ray parallel fit enabled inside AutoGluon. Monitor RAM/disk because multiple models may train at the same time.")
    if args.use_hmm_state:
        print("[FE] quiet HMM enabled; using KMeans-initialized smoothed HMM state features.")
    else:
        print("[FE] HMM disabled; using KMeans/transition features instead.")
    train_raw = pd.read_csv(args.train_path)
    test_raw = pd.read_csv(args.test_path)
    if args.use_hand_flip:
        print("[AUG] hand/point mirror augmentation enabled")
        train_raw = add_hand_flip_augmentation(train_raw)

    print("[FE] row-level mapping/motion features")
    train_row = add_sequence_motion_features(train_raw)
    test_row = add_sequence_motion_features(test_raw)

    print("[FE] sliding-prefix tables")
    train_feat = make_sliding_feature_df(train_row, args.max_seq_len, True, args.use_long_hist, args.include_streak, "train prefixes")
    test_feat = make_sliding_feature_df(test_row, args.max_seq_len, False, args.use_long_hist, args.include_streak, "test prefixes")
    if train_feat.empty or test_feat.empty:
        raise RuntimeError("Empty train/test prefix feature table. Check input CSV format.")

                                                 
    test_rally_uid = test_feat["rally_uid"].copy()
    groups = train_feat["_group_match"].to_numpy() if "_group_match" in train_feat.columns else train_feat["rally_uid"].to_numpy()

                                                                
    if args.use_hmm_state:
        print("[FE] train-only GaussianHMM state features")
        train_feat, test_feat = add_hmm_state_features(
            train_feat, test_feat,
            n_states=args.hmm_states,
            seed=args.seed,
            max_iter=args.hmm_max_iter,
            max_cols=args.hmm_max_cols,
            covariance_type=args.hmm_covariance_type,
        )
    if args.use_kmeans_state:
        print("[FE] train-only KMeans state features")
        train_feat, test_feat = add_kmeans_state_features(train_feat, test_feat, n_clusters=args.kmeans_states, seed=args.seed)

    aenc = penc = None
    if args.use_transition:
        print("[FE] train-only transition probability features")
        train_feat, test_feat, aenc, penc = add_transition_features(train_feat, test_feat)

                               
    _, dropped_train = drop_raw_id_features(train_feat)
    _, dropped_test = drop_raw_id_features(test_feat)
    with open(os.path.join(diag_dir, "dropped_columns.json"), "w", encoding="utf-8") as f:
        json.dump({"train": dropped_train, "test": dropped_test, "raw_id_policy": RAW_ID_COLS}, f, ensure_ascii=False, indent=2)

                    
    task_infos = []

                                                       
    print("[AG] action main")
    action_classes = list(range(NUM_ACTION_MAIN_CLASSES if args.action_0_14 else NUM_ACTION_CLASSES))
    action_oof_small, action_test_small, infos = train_ag_oof_sweep(
        train_feat, test_feat, "target_action", action_classes,
        os.path.join(args.out_dir, "ag_models"), groups, args.folds, args.ag_presets,
        args.ag_time_limit, "f1_macro", args.ag_no_catboost, args.seed, "action_main",
        model_suite=args.ag_model_suite, metric_suite=args.ag_metric_suite,
        fit_strategy=args.fit_strategy, ag_num_gpus=args.ag_num_gpus, ag_max_memory_usage_ratio=args.ag_max_memory_usage_ratio,
    )
    task_infos += infos
    if args.action_0_14:
        action_oof = expand_action_15_probs(action_oof_small)
        action_test = expand_action_15_probs(action_test_small)
    else:
        action_oof, action_test = action_oof_small, action_test_small
    action_oof = legal_action_mask(action_oof, train_feat)
    action_test = legal_action_mask(action_test, test_feat)

                                                   
    if args.use_action_aux:
        print("[AG] action auxiliary group/sub")
        grp_oof, grp_test, infos = train_ag_oof_sweep(train_feat, test_feat, "target_aid_group", list(range(NUM_AID_GROUP_CLASSES)), os.path.join(args.out_dir, "ag_models"), groups, args.folds, args.ag_presets, scaled_time_limit(args.ag_time_limit, 1/3, 60), "f1_macro", args.ag_no_catboost, args.seed, "aid_group_aux", model_suite=args.ag_model_suite, metric_suite=args.ag_metric_suite, fit_strategy=args.fit_strategy, ag_num_gpus=args.ag_num_gpus, ag_max_memory_usage_ratio=args.ag_max_memory_usage_ratio)
        sub_oof, sub_test, infos2 = train_ag_oof_sweep(train_feat, test_feat, "target_aid_sub", list(range(NUM_AID_SUB_CLASSES)), os.path.join(args.out_dir, "ag_models"), groups, args.folds, args.ag_presets, scaled_time_limit(args.ag_time_limit, 1/3, 60), "f1_macro", args.ag_no_catboost, args.seed, "aid_sub_aux", model_suite=args.ag_model_suite, metric_suite=args.ag_metric_suite, fit_strategy=args.fit_strategy, ag_num_gpus=args.ag_num_gpus, ag_max_memory_usage_ratio=args.ag_max_memory_usage_ratio)
        task_infos += infos + infos2
        for c in range(NUM_AID_GROUP_CLASSES):
            train_feat[f"ag_aux_aid_group_prob_{c}"] = grp_oof[:, c]
            test_feat[f"ag_aux_aid_group_prob_{c}"] = grp_test[:, c]
        for c in range(NUM_AID_SUB_CLASSES):
            train_feat[f"ag_aux_aid_sub_prob_{c}"] = sub_oof[:, c]
            test_feat[f"ag_aux_aid_sub_prob_{c}"] = sub_test[:, c]

                                        
    if args.use_action_point_stack:
        for c in range(NUM_ACTION_CLASSES):
            train_feat[f"ag_stack_action_prob_{c}"] = action_oof[:, c]
            test_feat[f"ag_stack_action_prob_{c}"] = action_test[:, c]
        train_feat["ag_stack_action_conf"] = action_oof.max(axis=1)
        test_feat["ag_stack_action_conf"] = action_test.max(axis=1)
        train_feat["ag_stack_action_entropy"] = -(action_oof * np.log(np.maximum(action_oof, 1e-12))).sum(axis=1)
        test_feat["ag_stack_action_entropy"] = -(action_test * np.log(np.maximum(action_test, 1e-12))).sum(axis=1)

                               
    valid_oof = valid_test = depth_oof = depth_test = side_oof = side_test = None
    if args.use_point_hier:
        print("[AG] point valid/depth/side hierarchy")
        valid_oof, valid_test, infos = train_ag_oof_sweep(train_feat, test_feat, "target_point_valid", [0, 1], os.path.join(args.out_dir, "ag_models"), groups, args.folds, args.ag_presets, scaled_time_limit(args.ag_time_limit, 1/3, 60), "f1_macro", args.ag_no_catboost, args.seed, "point_valid", model_suite=args.ag_model_suite, metric_suite=args.ag_metric_suite, fit_strategy=args.fit_strategy, ag_num_gpus=args.ag_num_gpus, ag_max_memory_usage_ratio=args.ag_max_memory_usage_ratio)
        task_infos += infos
        valid_rows = train_feat["target_point_valid"].astype(int).to_numpy() == 1
                                                                                                                                                            
        depth_oof, depth_test, infos = train_ag_oof_sweep(train_feat.loc[valid_rows].reset_index(drop=True), test_feat, "target_pid_depth", [1, 2, 3], os.path.join(args.out_dir, "ag_models"), groups[valid_rows], args.folds, args.ag_presets, scaled_time_limit(args.ag_time_limit, 1/3, 60), "f1_macro", args.ag_no_catboost, args.seed, "point_depth", model_suite=args.ag_model_suite, metric_suite=args.ag_metric_suite, fit_strategy=args.fit_strategy, ag_num_gpus=args.ag_num_gpus, ag_max_memory_usage_ratio=args.ag_max_memory_usage_ratio)
                                                                                                            
        depth_full = np.zeros((len(train_feat), 4), dtype=np.float32); depth_full[:, 0] = 1.0
        depth_full[valid_rows, 1:4] = depth_oof; depth_full[valid_rows, 0] = 0.0
        depth_oof = depth_full
        depth_test_full = np.zeros((len(test_feat), 4), dtype=np.float32); depth_test_full[:, 1:4] = depth_test; depth_test = depth_test_full
        task_infos += infos
        side_oof, side_test, infos = train_ag_oof_sweep(train_feat.loc[valid_rows].reset_index(drop=True), test_feat, "target_pid_side", [1, 2, 3], os.path.join(args.out_dir, "ag_models"), groups[valid_rows], args.folds, args.ag_presets, scaled_time_limit(args.ag_time_limit, 1/3, 60), "f1_macro", args.ag_no_catboost, args.seed, "point_side", model_suite=args.ag_model_suite, metric_suite=args.ag_metric_suite, fit_strategy=args.fit_strategy, ag_num_gpus=args.ag_num_gpus, ag_max_memory_usage_ratio=args.ag_max_memory_usage_ratio)
        side_full = np.zeros((len(train_feat), 4), dtype=np.float32); side_full[:, 0] = 1.0
        side_full[valid_rows, 1:4] = side_oof; side_full[valid_rows, 0] = 0.0
        side_oof = side_full
        side_test_full = np.zeros((len(test_feat), 4), dtype=np.float32); side_test_full[:, 1:4] = side_test; side_test = side_test_full
        task_infos += infos
                                                                 
        for c in range(2):
            train_feat[f"ag_point_valid_prob_{c}"] = valid_oof[:, c]
            test_feat[f"ag_point_valid_prob_{c}"] = valid_test[:, c]
        for c in range(4):
            train_feat[f"ag_point_depth_prob_{c}"] = depth_oof[:, c]
            test_feat[f"ag_point_depth_prob_{c}"] = depth_test[:, c]
            train_feat[f"ag_point_side_prob_{c}"] = side_oof[:, c]
            test_feat[f"ag_point_side_prob_{c}"] = side_test[:, c]

    print("[AG] point main")
    point_oof, point_test, infos = train_ag_oof_sweep(
        train_feat, test_feat, "target_point", list(range(NUM_POINT_CLASSES)),
        os.path.join(args.out_dir, "ag_models"), groups, args.folds, args.ag_presets,
        args.ag_time_limit, "f1_macro", args.ag_no_catboost, args.seed, "point_main",
        model_suite=args.ag_model_suite, metric_suite=args.ag_metric_suite,
        fit_strategy=args.fit_strategy, ag_num_gpus=args.ag_num_gpus, ag_max_memory_usage_ratio=args.ag_max_memory_usage_ratio,
    )
    task_infos += infos
    point_oof_hier = combine_point_hierarchy(point_oof, valid_oof, depth_oof, side_oof, weight_main=args.point_main_weight)
    point_test_hier = combine_point_hierarchy(point_test, valid_test, depth_test, side_test, weight_main=args.point_main_weight)

                                                       
    if args.use_server and "target_win" in train_feat.columns:
        print("[AG] serverGetPoint")
        win_oof, win_test_prob2, infos = train_ag_oof_sweep(
            train_feat, test_feat, "target_win", [0, 1], os.path.join(args.out_dir, "ag_models"), groups,
            args.folds, args.ag_presets, scaled_time_limit(args.ag_time_limit, 1/2, 120), "roc_auc", args.ag_no_catboost, args.seed, "server_win",
            model_suite=args.ag_model_suite, metric_suite=args.ag_metric_suite,
            fit_strategy=args.fit_strategy, ag_num_gpus=args.ag_num_gpus, ag_max_memory_usage_ratio=args.ag_max_memory_usage_ratio,
        )
        task_infos += infos
        win_test = win_test_prob2[:, 1]
        win_oof_score = None
        server_threshold = 0.5
        server_threshold_score = None
        try:
            y_win = train_feat["target_win"].astype(int).to_numpy()
            win_oof_score = float(roc_auc_score(y_win, win_oof[:, 1]))
            server_threshold, server_threshold_score = find_best_binary_threshold(y_win, win_oof[:, 1])
            print(f"[SERVER] hard output threshold={server_threshold:.6f}, OOF balanced_acc={server_threshold_score:.6f}, OOF AUC={win_oof_score:.6f}")
        except Exception as e:
            print(f"[SERVER] threshold fallback 0.5 ({e})")
    else:
        print("[AG] serverGetPoint skipped; using 0.5")
        win_test = np.full(len(test_feat), 0.5, dtype=np.float32)
        win_oof_score = None
        server_threshold = 0.5
        server_threshold_score = None

                                       
    action_base = action_test.copy()
    point_base = point_test_hier.copy()

                                 
    action_trans = action_base.copy()
    point_trans = point_base.copy()
    if args.use_transition_fusion and aenc is not None and penc is not None:
        print("[POST] transition fusion")
        action_prior = aenc.predict_prior_matrix(test_feat)
        point_prior = penc.predict_prior_matrix(test_feat)
        action_trans = legal_action_mask(fuse_transition(action_base, action_prior, args.action_transition_alpha), test_feat)
        point_trans = fuse_transition(point_base, point_prior, args.point_transition_alpha)

                           
    action_cal = action_trans.copy()
    point_cal = point_trans.copy()
    calibration_info = {}
    if args.use_calibration:
        print("[POST] temperature + class-scale calibration")
        y_action = train_feat["target_action"].astype(int).to_numpy()
        action_oof_cal = legal_action_mask(action_oof, train_feat)
        if args.use_transition_fusion and aenc is not None:
            action_oof_cal = legal_action_mask(fuse_transition(action_oof_cal, aenc.predict_prior_matrix(train_feat), args.action_transition_alpha), train_feat)
        Ta = find_best_temperature(action_oof_cal, y_action, list(range(NUM_ACTION_CLASSES)))
        sa = fit_class_scales(apply_temperature(action_oof_cal, Ta), y_action, list(range(NUM_ACTION_CLASSES)))
        action_cal = legal_action_mask(apply_scales(apply_temperature(action_trans, Ta), sa), test_feat)
        y_point = train_feat["target_point"].astype(int).to_numpy()
        point_oof_cal = point_oof_hier
        if args.use_transition_fusion and penc is not None:
            point_oof_cal = fuse_transition(point_oof_cal, penc.predict_prior_matrix(train_feat), args.point_transition_alpha)
        Tp = find_best_temperature(point_oof_cal, y_point, list(range(NUM_POINT_CLASSES)))
        sp = fit_class_scales(apply_temperature(point_oof_cal, Tp), y_point, list(range(NUM_POINT_CLASSES)))
        point_cal = apply_scales(apply_temperature(point_trans, Tp), sp)
        calibration_info = {"action_T": Ta, "point_T": Tp, "action_scale": sa.tolist(), "point_scale": sp.tolist()}
        with open(os.path.join(diag_dir, "calibration.json"), "w", encoding="utf-8") as f:
            json.dump(calibration_info, f, ensure_ascii=False, indent=2)

                                    
    action_pred_base = action_base.argmax(axis=1)
    point_pred_base = point_base.argmax(axis=1)
    action_pred_trans = action_trans.argmax(axis=1)
    point_pred_trans = point_trans.argmax(axis=1)
    action_pred_full = action_cal.argmax(axis=1)
    point_pred_full = point_cal.argmax(axis=1)

    change_log = {}
    if args.use_rule_override:
        print("[POST] conservative rule override")
        action_context = ["prev2_actionId", "last_actionId", "last_pointId", "next_rally_phase_tf"]
        point_context = ["last_actionId", "last_pointId", "next_rally_phase_tf"]
        atab = build_rule_tables(train_feat, "target_action", action_context, args.rule_min_context)
        ptab = build_rule_tables(train_feat, "target_point", point_context, args.rule_min_context)
        action_pred_full, ac = apply_rule_override(action_pred_full, test_feat, atab, action_context)
        point_pred_full, pc = apply_rule_override(point_pred_full, test_feat, ptab, point_context)
        change_log["rule_action_changed"] = ac
        change_log["rule_point_changed"] = pc

                         
    prob_df = pd.DataFrame({"rally_uid": test_rally_uid})
    for c in range(NUM_ACTION_CLASSES):
        prob_df[f"prob_action_{c}"] = action_cal[:, c]
    for c in range(NUM_POINT_CLASSES):
        prob_df[f"prob_point_{c}"] = point_cal[:, c]
    prob_df["prob_serverGetPoint"] = win_test
    prob_df["serverGetPoint_hard"] = (win_test >= float(server_threshold)).astype(int)
    prob_df.to_csv(os.path.join(args.out_dir, "ag_noid_probs.csv"), index=False, encoding="utf-8-sig")

                       
    variants = {
        "base": make_submission(test_rally_uid, action_pred_base, point_pred_base, win_test, server_threshold, args.hard_server_output),
        "transition": make_submission(test_rally_uid, action_pred_trans, point_pred_trans, win_test, server_threshold, args.hard_server_output),
        "full_safe": make_submission(test_rally_uid, action_pred_full, point_pred_full, win_test, server_threshold, args.hard_server_output),
    }
    for name, sub in variants.items():
        path = os.path.join(args.out_dir, f"submission_ag_noid_{name}.csv")
        sub.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {path} shape={sub.shape}")
        save_distribution(diag_dir, name, sub)

                  
    oof_summary = []
    action_macro_f1 = None
    point_macro_f1 = None
    try:
        y_a = train_feat["target_action"].astype(int).to_numpy()
        action_macro_f1 = float(f1_score(y_a, action_oof.argmax(axis=1), labels=list(range(NUM_ACTION_CLASSES)), average="macro", zero_division=0))
        oof_summary.append({"target": "action", "metric": "macro_f1", "score": action_macro_f1})
    except Exception:
        pass
    try:
        y_p = train_feat["target_point"].astype(int).to_numpy()
        point_macro_f1 = float(f1_score(y_p, point_oof_hier.argmax(axis=1), labels=list(range(NUM_POINT_CLASSES)), average="macro", zero_division=0))
        oof_summary.append({"target": "point", "metric": "macro_f1", "score": point_macro_f1})
    except Exception:
        pass
    if win_oof_score is not None:
        oof_summary.append({"target": "server", "metric": "auc", "score": win_oof_score})
    if action_macro_f1 is not None and point_macro_f1 is not None and win_oof_score is not None:
        official_weighted_score = 0.4 * point_macro_f1 + 0.4 * action_macro_f1 + 0.2 * float(win_oof_score)
        oof_summary.append({
            "target": "official_weighted",
            "metric": "0.4*point_macro_f1 + 0.4*action_macro_f1 + 0.2*server_auc",
            "score": official_weighted_score,
        })
        print(f"[OOF] official_weighted_score = 0.4*point({point_macro_f1:.6f}) + 0.4*action({action_macro_f1:.6f}) + 0.2*server({float(win_oof_score):.6f}) = {official_weighted_score:.6f}")
    pd.DataFrame(oof_summary).to_csv(os.path.join(diag_dir, "oof_summary.csv"), index=False)
    pd.DataFrame(task_infos).to_csv(os.path.join(diag_dir, "fold_infos.csv"), index=False)
    change_log["hard_server_output"] = bool(args.hard_server_output)
    change_log["server_threshold"] = float(server_threshold)
    if server_threshold_score is not None:
        change_log["server_threshold_oof_balanced_acc"] = float(server_threshold_score)
    if win_oof_score is not None:
        change_log["server_oof_auc"] = float(win_oof_score)
    with open(os.path.join(diag_dir, "postprocess_change_log.json"), "w", encoding="utf-8") as f:
        json.dump(change_log, f, ensure_ascii=False, indent=2)

    print("[DONE] outputs saved to", args.out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", default="train.csv")
    p.add_argument("--test_path", default="test.csv")
    p.add_argument("--out_dir", default="ag_outputs_noid_full")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max_seq_len", type=int, default=15)

                       
    p.add_argument("--use_transition", action="store_true", default=True)
    p.add_argument("--no_transition", action="store_false", dest="use_transition")
    p.add_argument("--use_hmm_state", action="store_true", default=True, help="Enable quiet KMeans-initialized smoothed HMM state features.")
    p.add_argument("--no_hmm_state", action="store_false", dest="use_hmm_state", help="Disable HMM state features.")
    p.add_argument("--hmm_states", type=int, default=8)
    p.add_argument("--hmm_max_iter", type=int, default=80)
    p.add_argument("--hmm_max_cols", type=int, default=80)
    p.add_argument("--hmm_covariance_type", default="diag", choices=["diag", "full", "spherical", "tied"])
    p.add_argument("--use_kmeans_state", action="store_true")
    p.add_argument("--kmeans_states", type=int, default=8)
    p.add_argument("--use_long_hist", action="store_true")
    p.add_argument("--include_streak", action="store_true")
    p.add_argument("--use_hand_flip", action="store_true")

                               
    p.add_argument("--action_0_14", action="store_true", default=True)
    p.add_argument("--action_0_18", action="store_false", dest="action_0_14")
    p.add_argument("--use_action_aux", action="store_true")
    p.add_argument("--use_action_point_stack", action="store_true", default=True)
    p.add_argument("--no_action_point_stack", action="store_false", dest="use_action_point_stack")
    p.add_argument("--use_point_hier", action="store_true", default=True)
    p.add_argument("--no_point_hier", action="store_false", dest="use_point_hier")
    p.add_argument("--point_main_weight", type=float, default=0.65)
    p.add_argument("--use_server", action="store_true", default=True)
    p.add_argument("--no_server", action="store_false", dest="use_server")
    p.add_argument("--hard_server_output", action="store_true", default=True, help="Submit serverGetPoint as hard 0/1 using OOF-tuned threshold; probabilities are still saved in ag_noid_probs.csv.")
    p.add_argument("--prob_server_output", action="store_false", dest="hard_server_output", help="Submit serverGetPoint as probability instead of hard 0/1.")

                
    p.add_argument("--ag_time_limit", type=int, default=0, help="Seconds per AutoGluon fit. <=0 means no time limit.")
    p.add_argument("--no_time_limit", action="store_true", help="Use a very large explicit AutoGluon time budget so every requested model gets a chance to run.")
    p.add_argument("--ag_unlimited_seconds", type=int, default=86400, help="Large per-fit time budget used when --no_time_limit or --ag_time_limit<=0. Prevents AutoGluon from falling back to its default 3600s budget.")
    p.add_argument("--ag_presets", default="best_quality")
    p.add_argument("--ag_no_catboost", action="store_true", default=True)
    p.add_argument("--ag_with_catboost", action="store_false", dest="ag_no_catboost")
    p.add_argument("--ag_model_suite", default="fast", choices=["fast", "original", "all", "everything", "all_models", "practical", "task_aware"], help="AutoGluon model zoo: fast=baseline; all=broad stable zoo; everything/all_models=all plus newer/foundation/experimental tabular models.")
    p.add_argument("--ag_metric_suite", default="single", choices=["single", "all"], help="single=use the task's main metric; all=train/blend multiple AutoGluon eval_metrics per task.")
    p.add_argument("--fit_strategy", default="sequential", choices=["sequential", "parallel"], help="AutoGluon model fitting strategy. parallel uses Ray to train multiple model types concurrently; sequential is safer on low RAM/disk.")
    p.add_argument("--ag_num_gpus", type=int, default=0, help="CUDA GPU count passed to AutoGluon. Keep 0 for Intel Arc/XPU because AutoGluon Tabular does not directly use torch.xpu.")
    p.add_argument("--ag_max_memory_usage_ratio", type=float, default=1.45, help="Passed to AutoGluon ag_args_fit. 1.3~1.5 helps RF/XT pass conservative memory checks; avoid huge values.")
    p.add_argument("--try_everything", action="store_true", help="Shortcut: full_open + quiet HMM + ag_model_suite=everything + ag_metric_suite=all + CatBoost enabled.")
    p.add_argument("--try_practical", action="store_true", help="Shortcut: full_open + task-aware practical model suite + single metric. Keeps RF/XT/NN_TORCH, excludes unrealistic foundation models.")

                  
    p.add_argument("--use_transition_fusion", action="store_true", default=True)
    p.add_argument("--no_transition_fusion", action="store_false", dest="use_transition_fusion")
    p.add_argument("--action_transition_alpha", type=float, default=0.80)
    p.add_argument("--point_transition_alpha", type=float, default=0.70)
    p.add_argument("--use_calibration", action="store_true", default=True)
    p.add_argument("--no_calibration", action="store_false", dest="use_calibration")
    p.add_argument("--use_rule_override", action="store_true", default=True)
    p.add_argument("--no_rule_override", action="store_false", dest="use_rule_override")
    p.add_argument("--rule_min_context", type=int, default=80)

    p.add_argument("--full_open", action="store_true", help="Turn on most experimental safe modules: aux heads, KMeans state, long-history/streak, hand flip, etc.")
    args = p.parse_args()
    if args.no_time_limit:
                                                                                
                                                                                     
        args.ag_time_limit = int(args.ag_unlimited_seconds)
    elif args.ag_time_limit <= 0:
        args.ag_time_limit = int(args.ag_unlimited_seconds)

    if args.try_everything:
        args.full_open = True
        args.ag_model_suite = "everything"
        args.ag_metric_suite = "all"
        args.ag_no_catboost = False

    if args.try_practical:
        args.full_open = True
        args.ag_model_suite = "task_aware"
        args.ag_metric_suite = "single"
                                                                                
        args.ag_no_catboost = True

    if args.full_open:
                                                                                                             
        args.use_hmm_state = True
        args.use_kmeans_state = True
        args.use_long_hist = True
        args.include_streak = True
        args.use_hand_flip = True
        args.use_action_aux = True
        args.use_action_point_stack = True
        args.use_point_hier = True
        args.use_server = True
        args.use_transition = True
        args.use_transition_fusion = True
        args.use_calibration = True
        args.use_rule_override = True
        if args.ag_model_suite in {"all", "everything", "all_models"}:
            args.ag_no_catboost = False
    return args


if __name__ == "__main__":
    run_pipeline(parse_args())
