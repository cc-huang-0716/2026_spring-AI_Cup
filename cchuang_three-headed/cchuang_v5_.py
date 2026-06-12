                       

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                                                                                           
                                                                                  
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(os.cpu_count() or 4, 1)))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import gc
import json
import math
import time
import random
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

try:
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.cluster import KMeans
except Exception as e:
    raise RuntimeError("需要 sklearn：pip install scikit-learn") from e

VERSION_TAG = "CCHUANG_STATE_DIRICHLET_HMM_ROUTER_TCN_TRANSFORMER_TEAMFEATURES_PROBWIN_SAFEFEATURES_20260515"
SEED = 42
PAD_IDX = -1

NUM_ACTION_CLASSES = 19
NUM_POINT_CLASSES = 10
NUM_WIN_CLASSES = 2

                                                
EXCLUDED_RAW_ID_COLS = ["rally_uid", "rally_id", "match"]

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
    "aggressive_shot_bl", "control_shot_bl", "serve_aggressive_bl", "receive_aggressive_bl",
    "third_ball_aggressive_bl", "rally_aggressive_bl", "pressure_aggressive_bl", "late_aggressive_bl",
    "serve_control_bl", "receive_control_bl", "third_ball_control_bl", "rally_control_bl",
    "roll3_point_change_rate_bl", "roll5_point_change_rate_bl",
    "roll5_position_change_rate_bl", "roll5_action_change_rate_bl", "roll5_spin_change_rate_bl",
    "roll5_side_change_rate_bl", "roll5_depth_change_rate_bl", "placement_volatility_bl",
    "roll3_aggressive_rate_bl", "roll5_aggressive_rate_bl", "roll3_control_rate_bl",
    "roll3_wide_deep_rate_bl", "tempo_acceleration_bl",
]

                         
LAST_CAT_COLS = BASE_CAT_COLS
LAST_NUM_COLS = BASE_NUM_COLS

                                                       
TRANSITION_SPECS_COMMON = [
    ("last_action", ["last_actionId"]),
    ("last_point", ["last_pointId"]),
    ("last_aid_group", ["last_aid_group"]),
    ("last_pid_side", ["last_pid_side"]),
    ("last_pid_depth", ["last_pid_depth"]),
    ("last_spin", ["last_spinId"]),
    ("last_strength", ["last_strengthId"]),
    ("phase_group", ["last_rally_phase_bl", "last_aid_group"]),
    ("point_group", ["last_pointId", "last_aid_group"]),
    ("side_group", ["last_pid_side", "last_aid_group"]),
    ("depth_group", ["last_pid_depth", "last_aid_group"]),
    ("prev2_last_action", ["prev2_actionId", "last_actionId"]),
    ("prev2_last_point", ["prev2_pointId", "last_pointId"]),
    ("prev2_last_group", ["prev2_aid_group", "last_aid_group"]),
                                                  
    ("last_action_point_combo", ["last_action_point_combo_bl"]),
    ("prev2_last_action_point_combo", ["prev2_action_point_combo_bl", "last_action_point_combo_bl"]),
    ("last_action_side_combo", ["last_action_side_combo_bl"]),
    ("last_action_depth_combo", ["last_action_depth_combo_bl"]),
]


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def find_input_files(train_path: Optional[str] = None, test_path: Optional[str] = None):
    if train_path is None:
        train_path = "data/train.csv" if os.path.exists("data/train.csv") else "train.csv"
    if test_path is None:
                                                                   
                                                                           
        if os.path.exists("data/test_new.csv"):
            test_path = "data/test_new.csv"
        elif os.path.exists("test_new.csv"):
            test_path = "test_new.csv"
        elif os.path.exists("data/test.csv"):
            test_path = "data/test.csv"
        elif os.path.exists("test.csv"):
            test_path = "test.csv"
        else:
            test_path = None
    return train_path, test_path


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="float")
    return df


def point_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pointId" not in df.columns:
        df["pointId"] = 0
    table = np.array([
        [0, 0],
        [1, 1], [1, 2], [1, 3],
        [2, 1], [2, 2], [2, 3],
        [3, 1], [3, 2], [3, 3],
    ], dtype=np.int64)
    pid = pd.to_numeric(df["pointId"], errors="coerce").fillna(0).astype(np.int64).clip(0, 9).to_numpy()
    mapped = table[pid]
    df["pid_depth"] = mapped[:, 0].astype(np.int64)
    df["pid_side"] = mapped[:, 1].astype(np.int64)
    return df


def action_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "actionId" not in df.columns:
        df["actionId"] = 0
    table = np.array([
        (0, 0),
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 1), (3, 2), (3, 3),
        (4, 1), (4, 2), (4, 3), (4, 4),
    ], dtype=np.int64)
    aid = pd.to_numeric(df["actionId"], errors="coerce").fillna(0).astype(np.int64).clip(0, 18).to_numpy()
    mapped = table[aid]
    df["aid_group"] = mapped[:, 0].astype(np.int64)
    df["aid_sub"] = mapped[:, 1].astype(np.int64)
    return df


def spin_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "spinId" not in df.columns:
        df["spinId"] = 0
    table = np.array([
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (1, 1),
        (2, 1),
    ], dtype=np.int64)
    sid = pd.to_numeric(df["spinId"], errors="coerce").fillna(0).astype(np.int64).clip(0, 5).to_numpy()
    mapped = table[sid]
    df["sid_spin"] = mapped[:, 0].astype(np.int64)
    df["sid_side"] = mapped[:, 1].astype(np.int64)
    return df


def _safe_norm_series(s, default_max=1.0):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    denom = max(float(np.nanmax(s.values)) if len(s) else 0.0, float(default_max), 1.0)
    return (s / denom).clip(0.0, 1.0).astype(np.float32)


def _safe_group_change_rate(s: pd.Series, window: int):
    changed = s.ne(s.shift(1)).astype(float)
    if len(changed) > 0:
        changed.iloc[0] = 0.0
    return changed.rolling(window=window, min_periods=1).mean()


def _rally_phase_from_strike(strike_number):
    s = pd.Series(strike_number).fillna(0).astype(float)
    return np.select(
        [s <= 1, s == 2, s == 3, s >= 4],
        [1, 2, 3, 4],
        default=4,
    ).astype(np.float32)


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


    df = df.copy()
    df = add_mapping_features(df)

    required = [
        "scoreSelf", "scoreOther", "strikeNumber", "actionId", "aid_group",
        "spinId", "sid_spin", "sid_side", "strengthId", "positionId",
        "pointId", "pid_side", "pid_depth",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["scoreDiff_bl"] = (df["scoreSelf"] - df["scoreOther"]).astype(np.float32)
    df["scoreSum_bl"] = (df["scoreSelf"] + df["scoreOther"]).astype(np.float32)
    df["absScoreDiff_bl"] = df["scoreDiff_bl"].abs().astype(np.float32)
                                                                                      
    df["scoreDiff_norm_bl"] = (df["scoreDiff_bl"] / 11.0).clip(-1.0, 1.0).astype(np.float32)
    df["scorePressure_bl"] = (1.0 - (df["absScoreDiff_bl"] / 11.0).clip(0.0, 1.0)).astype(np.float32)
    df["points_to_win_self_bl"] = ((11.0 - df["scoreSelf"]).clip(lower=0.0) / 11.0).astype(np.float32)
    df["points_to_win_other_bl"] = ((11.0 - df["scoreOther"]).clip(lower=0.0) / 11.0).astype(np.float32)
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

    strike = pd.to_numeric(df["strikeNumber"], errors="coerce").fillna(0).astype(float)
    phase = _rally_phase_from_strike(strike)
    df["rally_phase_bl"] = (phase / 4.0).astype(np.float32)
    df["strike_log1p_bl"] = (np.log1p(strike) / np.log1p(64.0)).clip(0.0, 1.0).astype(np.float32)
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
        df = df.sort_values(["rally_uid", "strikeNumber", "_orig_order_bl"]).reset_index(drop=True)
        g = df.groupby("rally_uid", sort=False)

        for src_col, new_col in [
            ("pointId", "point_changed_prev_bl"),
            ("positionId", "position_changed_prev_bl"),
            ("actionId", "action_changed_prev_bl"),
            ("spinId", "spin_changed_prev_bl"),
            ("strengthId", "strength_changed_prev_bl"),
            ("pid_side", "side_changed_prev_bl"),
            ("pid_depth", "depth_changed_prev_bl"),
        ]:
            prev = g[src_col].shift(1)
            df[new_col] = ((df[src_col] != prev) & prev.notna()).astype(np.float32)

        prev_side = g["pid_side"].shift(1)
        prev_depth = g["pid_depth"].shift(1)
        df["side_move_bl"] = ((df["pid_side"] - prev_side).fillna(0.0) / 2.0).clip(-1.0, 1.0).astype(np.float32)
        df["depth_move_bl"] = ((df["pid_depth"] - prev_depth).fillna(0.0) / 2.0).clip(-1.0, 1.0).astype(np.float32)
        df["placement_move_magnitude_bl"] = ((df["side_move_bl"].abs() + df["depth_move_bl"].abs()).clip(0.0, 2.0) / 2.0).astype(np.float32)

        df["aggressive_shot_bl"] = (
            0.35 * df["strength_norm_bl"]
            + 0.25 * df["action_group_norm_bl"]
            + 0.15 * df["wide_deep_pressure_bl"]
            + 0.15 * df["placement_move_magnitude_bl"]
            + 0.10 * df["spin_changed_prev_bl"]
        ).clip(0.0, 1.0).astype(np.float32)
        df["control_shot_bl"] = (
            0.35 * (1.0 - df["strength_norm_bl"])
            + 0.25 * df["is_middle_placement_bl"]
            + 0.20 * (1.0 - df["placement_move_magnitude_bl"])
            + 0.20 * (1.0 - df["action_changed_prev_bl"])
        ).clip(0.0, 1.0).astype(np.float32)

        for phase_col, out_col in [
            ("is_serve_phase_bl", "serve_aggressive_bl"),
            ("is_receive_phase_bl", "receive_aggressive_bl"),
            ("is_third_ball_phase_bl", "third_ball_aggressive_bl"),
            ("is_rally_phase_bl", "rally_aggressive_bl"),
            ("is_close_score_bl", "pressure_aggressive_bl"),
            ("is_late_game_bl", "late_aggressive_bl"),
        ]:
            df[out_col] = (df[phase_col] * df["aggressive_shot_bl"]).astype(np.float32)
        for phase_col, out_col in [
            ("is_serve_phase_bl", "serve_control_bl"),
            ("is_receive_phase_bl", "receive_control_bl"),
            ("is_third_ball_phase_bl", "third_ball_control_bl"),
            ("is_rally_phase_bl", "rally_control_bl"),
        ]:
            df[out_col] = (df[phase_col] * df["control_shot_bl"]).astype(np.float32)

        df["roll3_point_change_rate_bl"] = g["pointId"].transform(lambda s: _safe_group_change_rate(s, 3)).astype(np.float32)
        df["roll5_point_change_rate_bl"] = g["pointId"].transform(lambda s: _safe_group_change_rate(s, 5)).astype(np.float32)
        df["roll5_position_change_rate_bl"] = g["positionId"].transform(lambda s: _safe_group_change_rate(s, 5)).astype(np.float32)
        df["roll5_action_change_rate_bl"] = g["actionId"].transform(lambda s: _safe_group_change_rate(s, 5)).astype(np.float32)
        df["roll5_spin_change_rate_bl"] = g["spinId"].transform(lambda s: _safe_group_change_rate(s, 5)).astype(np.float32)
        df["roll5_side_change_rate_bl"] = g["pid_side"].transform(lambda s: _safe_group_change_rate(s, 5)).astype(np.float32)
        df["roll5_depth_change_rate_bl"] = g["pid_depth"].transform(lambda s: _safe_group_change_rate(s, 5)).astype(np.float32)
        df["placement_volatility_bl"] = (
            df["roll5_point_change_rate_bl"] + df["roll5_position_change_rate_bl"] +
            df["roll5_action_change_rate_bl"] + df["roll5_spin_change_rate_bl"] +
            df["roll5_side_change_rate_bl"] + df["roll5_depth_change_rate_bl"]
        ).astype(np.float32) / 6.0
        df["roll3_aggressive_rate_bl"] = g["aggressive_shot_bl"].transform(lambda s: s.rolling(3, min_periods=1).mean()).astype(np.float32)
        df["roll5_aggressive_rate_bl"] = g["aggressive_shot_bl"].transform(lambda s: s.rolling(5, min_periods=1).mean()).astype(np.float32)
        df["roll3_control_rate_bl"] = g["control_shot_bl"].transform(lambda s: s.rolling(3, min_periods=1).mean()).astype(np.float32)
        df["roll3_wide_deep_rate_bl"] = g["wide_deep_pressure_bl"].transform(lambda s: s.rolling(3, min_periods=1).mean()).astype(np.float32)
        df["tempo_acceleration_bl"] = (df["roll3_aggressive_rate_bl"] - df["roll3_control_rate_bl"]).astype(np.float32)
        df = df.sort_values("_orig_order_bl").drop(columns=["_orig_order_bl"]).reset_index(drop=True)
    else:
        for c in BASE_NUM_COLS:
            if c not in df.columns:
                df[c] = 0.0

    for c in BASE_CAT_COLS:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int64)
    for c in BASE_NUM_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

    return reduce_mem_usage(df)


def safe_change_rate(values):
    values = np.asarray(values)
    if len(values) <= 1:
        return 0.0
    return float(np.mean(values[1:] != values[:-1]))


def safe_streak_len(values):
    values = list(values)
    if not values:
        return 0
    last = values[-1]
    streak = 1
    for v in reversed(values[:-1]):
        if v == last:
            streak += 1
        else:
            break
    return streak


def add_ratio_features(out: Dict[str, Any], prefix: pd.DataFrame, col: str, prefix_name: str, classes: int, windows: List[Tuple[str, Optional[int]]]):
    arr_all = prefix[col].to_numpy(np.int64) if col in prefix.columns else np.zeros(len(prefix), dtype=np.int64)
    for w_name, w in windows:
        arr = arr_all if w is None else arr_all[-w:]
        denom = max(len(arr), 1)
        counts = np.bincount(np.clip(arr, 0, classes - 1), minlength=classes).astype(np.float32)[:classes]
        ratio = counts / denom
        for c in range(classes):
            out[f"{w_name}_{prefix_name}_ratio_{c}"] = float(ratio[c])


def add_change_features(out: Dict[str, Any], prefix: pd.DataFrame, col: str, prefix_name: str, windows: List[Tuple[str, Optional[int]]]):
    arr_all = prefix[col].to_numpy(np.int64) if col in prefix.columns else np.zeros(len(prefix), dtype=np.int64)
    for w_name, w in windows:
        arr = arr_all if w is None else arr_all[-w:]
        out[f"{w_name}_{prefix_name}_change_rate"] = safe_change_rate(arr)
        out[f"{w_name}_{prefix_name}_streak_len"] = safe_streak_len(arr)


def build_single_prefix_features(prefix: pd.DataFrame, rally_uid=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if rally_uid is not None:
        out["rally_uid"] = rally_uid
    last = prefix.iloc[-1]
    prev2 = prefix.iloc[-2] if len(prefix) >= 2 else last

    for col in LAST_CAT_COLS:
        out[f"last_{col}"] = int(last[col]) if col in prefix.columns else 0
        out[f"prev2_{col}"] = int(prev2[col]) if col in prefix.columns else 0
    for col in LAST_NUM_COLS:
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

    def _entropy_from_counts(vals, n_classes):
        vals = np.asarray(vals, dtype=np.int64)
        vals = np.clip(vals, 0, n_classes - 1)
        counts = np.bincount(vals, minlength=n_classes).astype(np.float64)
        total = counts.sum()
        if total <= 0:
            return 0.0, 0.0
        probs = counts[counts > 0] / total
        return float(-(probs * np.log(probs + 1e-12)).sum()), float(counts.max() / total)

    if "actionId" in prefix.columns:
        ent, dom = _entropy_from_counts(prefix["actionId"].to_numpy(), NUM_ACTION_CLASSES)
        out["hist_action_entropy_tf"] = ent
        out["hist_action_dominance_tf"] = dom
        out["hist_nunique_action_tf"] = int(pd.Series(prefix["actionId"].to_numpy()).nunique())
    else:
        out["hist_action_entropy_tf"] = 0.0; out["hist_action_dominance_tf"] = 0.0; out["hist_nunique_action_tf"] = 0
    if "pointId" in prefix.columns:
        ent, dom = _entropy_from_counts(prefix["pointId"].to_numpy(), NUM_POINT_CLASSES)
        out["hist_point_entropy_tf"] = ent
        out["hist_point_dominance_tf"] = dom
        out["hist_nunique_point_tf"] = int(pd.Series(prefix["pointId"].to_numpy()).nunique())
    else:
        out["hist_point_entropy_tf"] = 0.0; out["hist_point_dominance_tf"] = 0.0; out["hist_nunique_point_tf"] = 0

    windows = [("roll3", 3), ("roll5", 5), ("roll_all", None)]
    add_ratio_features(out, prefix, "pointId", "point", NUM_POINT_CLASSES, windows)
    add_ratio_features(out, prefix, "actionId", "action", NUM_ACTION_CLASSES, [("roll3", 3), ("roll5", 5)])
    add_ratio_features(out, prefix, "aid_group", "aid_group", 5, windows)
    add_ratio_features(out, prefix, "pid_side", "side", 4, windows)
    add_ratio_features(out, prefix, "pid_depth", "depth", 4, windows)
    add_ratio_features(out, prefix, "sid_spin", "spin", 4, [("roll3", 3), ("roll5", 5)])

    for col, name in [
        ("pid_side", "side"), ("pid_depth", "depth"), ("aid_group", "aid_group"),
        ("actionId", "action"), ("pointId", "point"), ("positionId", "position"),
        ("spinId", "spin"), ("strengthId", "strength"),
    ]:
        add_change_features(out, prefix, col, name, windows)

                                              
    for col in ["actionId", "pointId", "aid_group", "pid_side", "pid_depth", "spinId", "strengthId", "action_point_combo_bl"]:
        vals = prefix[col].tail(3).to_list() if col in prefix.columns else []
        vals = [0] * max(0, 3 - len(vals)) + vals
        for i, v in enumerate(vals, start=1):
            out[f"last3_{col}_{i}"] = int(v)

    return out


def make_sliding_feature_df(df: pd.DataFrame, max_seq_len: int = 15, is_train: bool = True, desc: str = "features") -> pd.DataFrame:
    rows = []
    df = df.copy()
    if "rally_uid" not in df.columns:
        df["rally_uid"] = np.arange(len(df))
    df["row_order"] = np.arange(len(df))
    grouped = df.groupby("rally_uid", sort=False)
    for rally_uid, g in tqdm(grouped, desc=desc):
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
                feat = build_single_prefix_features(prefix, rally_uid=rally_uid)
                target = g.iloc[k]
                feat["target_action"] = int(target["actionId"]) if "actionId" in g.columns else 0
                feat["target_point"] = int(target["pointId"]) if "pointId" in g.columns else 0
                                                                               
                feat["target_win"] = int(g["serverGetPoint"].iloc[0]) if "serverGetPoint" in g.columns else 0
                rows.append(feat)
        else:
            prefix = g
            if max_seq_len is not None:
                prefix = prefix.iloc[-max_seq_len:]
            feat = build_single_prefix_features(prefix, rally_uid=rally_uid)
            rows.append(feat)
    out = pd.DataFrame(rows)
    return reduce_mem_usage(out)


class DirichletTransitionEncoder:
    def __init__(self, specs, target_col: str, num_classes: int, alpha: float = 1.0, prefix: str = "dir"):
        self.specs = specs
        self.target_col = target_col
        self.num_classes = num_classes
        self.alpha = alpha
        self.prefix = prefix
        self.global_prob = None
        self.tables = {}

    def fit(self, feat_df: pd.DataFrame):
        y = feat_df[self.target_col].astype(int).to_numpy()
        counts = np.bincount(np.clip(y, 0, self.num_classes - 1), minlength=self.num_classes).astype(np.float64)[:self.num_classes]
        self.global_prob = (counts + self.alpha) / (counts.sum() + self.alpha * self.num_classes)

        for name, cols in self.specs:
            use_cols = [c for c in cols if c in feat_df.columns]
            if not use_cols:
                continue
            tmp = feat_df[use_cols + [self.target_col]].copy()
            tmp[self.target_col] = tmp[self.target_col].astype(int).clip(0, self.num_classes - 1)
            count_df = tmp.groupby(use_cols + [self.target_col]).size().unstack(self.target_col, fill_value=0)
            count_df = count_df.reindex(columns=list(range(self.num_classes)), fill_value=0).astype(np.float64)
            prob_df = count_df + self.alpha
            prob_df = prob_df.div(prob_df.sum(axis=1), axis=0)
            prob_cols = [f"{self.prefix}_{self.target_col}_{name}_to_{c}" for c in range(self.num_classes)]
            prob_df.columns = prob_cols
            prob_df = prob_df.reset_index()
            self.tables[name] = (use_cols, prob_df, prob_cols)
        return self

    def transform(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        out = feat_df.copy()
        for name, (cols, prob_df, prob_cols) in self.tables.items():
            out = out.merge(prob_df, how="left", on=cols)
            for c, col in enumerate(prob_cols):
                out[col] = out[col].fillna(float(self.global_prob[c])).astype(np.float32)
            probs = out[prob_cols].to_numpy(dtype=np.float32)
            out[f"{self.prefix}_{self.target_col}_{name}_entropy"] = (-(probs * np.log(probs + 1e-12)).sum(axis=1)).astype(np.float32)
            out[f"{self.prefix}_{self.target_col}_{name}_maxprob"] = probs.max(axis=1).astype(np.float32)
        return reduce_mem_usage(out)

    def fit_transform(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(feat_df)
        return self.transform(feat_df)


class HMMStateEncoder:


    def __init__(self, n_states: int = 7, random_state: int = SEED):
        self.n_states = n_states
        self.random_state = random_state
        self.feature_cols: List[str] = []
        self.scaler = None
        self.model = None
        self.kind = None

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        candidates = [
            "last_actionId", "last_pointId", "last_aid_group", "last_pid_side", "last_pid_depth",
            "last_spinId", "last_strengthId", "last_positionId", "prefix_len",
            "last_roll5_action_change_rate_bl", "last_roll5_point_change_rate_bl", "last_roll5_side_change_rate_bl",
            "last_scoreDiff_bl", "last_scorePressure_bl", "last_rally_phase_bl", "last_aggressive_shot_bl",
            "last_control_shot_bl", "last_placement_volatility_bl", "last_tempo_acceleration_bl",
        ]
        return [c for c in candidates if c in df.columns]

    def fit(self, df: pd.DataFrame):
        from sklearn.preprocessing import StandardScaler
        self.feature_cols = self._select_features(df)
        if not self.feature_cols:
            self.feature_cols = [c for c in df.columns if c.startswith("last_") and pd.api.types.is_numeric_dtype(df[c])][:12]
        X = df[self.feature_cols].fillna(0).astype(np.float32).to_numpy()
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        try:
            from hmmlearn import hmm
            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=80,
                random_state=self.random_state,
                verbose=False,
            )
            self.model.fit(Xs)
            self.kind = "hmmlearn"
        except Exception as e_hmm:
                                                                                
                                                                                        
            try:
                self.model = KMeans(
                    n_clusters=self.n_states,
                    random_state=self.random_state,
                    n_init=1,
                    algorithm="lloyd",
                )
                self.model.fit(Xs)
                self.kind = "kmeans_fallback"
            except Exception as e_km:
                print(f"[WARN] HMM/KMeans failed; using quantile_state_fallback. hmm={type(e_hmm).__name__}, kmeans={type(e_km).__name__}")
                score = Xs[:, 0].astype(np.float64) if Xs.shape[1] else np.zeros(len(Xs), dtype=np.float64)
                if Xs.shape[1] > 1:
                                                                                        
                    weights = np.linspace(1.0, 0.2, Xs.shape[1], dtype=np.float64)
                    score = Xs.dot(weights) / np.clip(np.abs(weights).sum(), 1e-12, None)
                qs = np.linspace(0, 100, self.n_states + 1)[1:-1]
                self.model = {"quantiles": np.percentile(score, qs), "score_weights": None}
                if Xs.shape[1] > 1:
                    self.model["score_weights"] = weights
                self.kind = "quantile_fallback"
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        X = out[self.feature_cols].fillna(0).astype(np.float32).to_numpy()
        Xs = self.scaler.transform(X)
        if self.kind == "hmmlearn":
            states = self.model.predict(Xs)
            try:
                probs = self.model.predict_proba(Xs)
            except Exception:
                probs = np.eye(self.n_states)[states]
        elif self.kind == "kmeans_fallback":
            states = self.model.predict(Xs)
            centers = self.model.cluster_centers_
            dist = ((Xs[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            logits = -dist
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        else:
            weights = self.model.get("score_weights") if isinstance(self.model, dict) else None
            if weights is not None and Xs.shape[1] == len(weights):
                score = Xs.dot(weights) / np.clip(np.abs(weights).sum(), 1e-12, None)
            else:
                score = Xs[:, 0].astype(np.float64) if Xs.shape[1] else np.zeros(len(Xs), dtype=np.float64)
            states = np.searchsorted(self.model["quantiles"], score, side="right").astype(np.int64)
            states = np.clip(states, 0, self.n_states - 1)
            probs = np.full((len(states), self.n_states), 1e-3 / max(self.n_states - 1, 1), dtype=np.float64)
            probs[np.arange(len(states)), states] = 1.0 - 1e-3
        out["hmm_state"] = states.astype(np.int16)
        out["hmm_state_prob_max"] = probs.max(axis=1).astype(np.float32)
        out["hmm_state_entropy"] = (-(probs * np.log(probs + 1e-12)).sum(axis=1)).astype(np.float32)
        for s in range(self.n_states):
            out[f"hmm_prob_{s}"] = probs[:, s].astype(np.float32)
        return reduce_mem_usage(out)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


def choose_cv_group_column(train_raw: pd.DataFrame) -> str:
    if "match" in train_raw.columns:
        return "match"
    return "rally_uid"


def get_model_feature_columns(df: pd.DataFrame) -> List[str]:
    drop = set([
        "rally_uid", "target_action", "target_point", "target_win",
        "match", "rally_id", "numberGame", "cv_group", "row_order",
    ])
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def align_features(train_df: pd.DataFrame, valid_or_test_df: pd.DataFrame, feature_cols: List[str]):
    X1 = train_df.reindex(columns=feature_cols, fill_value=0).replace([np.inf, -np.inf], 0).fillna(0)
    X2 = valid_or_test_df.reindex(columns=feature_cols, fill_value=0).replace([np.inf, -np.inf], 0).fillna(0)
    return X1, X2


def fit_cpu_model(model_name: str, X_train, y_train, X_valid=None, y_valid=None, task="point", seed=SEED):
    model_name = model_name.lower()
    n_classes = int(np.max(y_train)) + 1 if len(y_train) else 2
    if model_name == "logreg":
        if task == "win":
            clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, n_jobs=1, random_state=seed, class_weight="balanced"))
        else:
            clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, n_jobs=1, random_state=seed, class_weight="balanced", multi_class="auto"))
        clf.fit(X_train, y_train)
        return clf
    if model_name == "lgbm":
        try:
            import lightgbm as lgb
        except Exception as e:
            raise ImportError("找不到 lightgbm：pip install lightgbm") from e
        objective = "binary" if task == "win" else "multiclass"
        clf = lgb.LGBMClassifier(
            objective=objective,
            n_estimators=900,
            learning_rate=0.035,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
        clf.fit(X_train, y_train)
        return clf
    if model_name == "xgb":
        try:
            import xgboost as xgb
        except Exception as e:
            raise ImportError("找不到 xgboost：pip install xgboost") from e
        params = dict(
            n_estimators=700,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            eval_metric="logloss" if task == "win" else "mlogloss",
        )
        if task != "win":
            params.update(objective="multi:softprob", num_class=n_classes)
        else:
            params.update(objective="binary:logistic")
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train)
        return clf
    if model_name == "cat":
        try:
            from catboost import CatBoostClassifier
        except Exception as e:
            raise ImportError("找不到 catboost：pip install catboost") from e
        clf = CatBoostClassifier(
            iterations=800,
            learning_rate=0.035,
            depth=6,
            loss_function="Logloss" if task == "win" else "MultiClass",
            random_seed=seed,
            verbose=False,
            thread_count=1,
            allow_writing_files=False,
        )
        clf.fit(X_train, y_train)
        return clf
    raise ValueError(f"Unknown CPU model: {model_name}")


def predict_proba_safe(model, X, n_classes: int, task: str):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    else:
        pred = model.predict(X)
        proba = np.eye(n_classes)[pred]
    if task == "win":
        if isinstance(proba, list):
            proba = proba[0]
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(np.float32)
        return proba.reshape(-1).astype(np.float32)
    proba = np.asarray(proba)
    if proba.ndim == 1:
        proba = np.eye(n_classes)[proba.astype(int)]
    if proba.shape[1] < n_classes:
        fixed = np.zeros((len(proba), n_classes), dtype=np.float32)
        fixed[:, :proba.shape[1]] = proba
        proba = fixed
    elif proba.shape[1] > n_classes:
        proba = proba[:, :n_classes]
    return proba.astype(np.float32)


def compute_task_score(y_action, p_action, y_point, p_point, y_win, p_win):
    score_action = f1_score(y_action, np.argmax(p_action, axis=1), labels=list(range(NUM_ACTION_CLASSES)), average="macro", zero_division=0) if p_action is not None else 0.0
    score_point = f1_score(y_point, np.argmax(p_point, axis=1), labels=list(range(NUM_POINT_CLASSES)), average="macro", zero_division=0) if p_point is not None else 0.0
    if p_win is not None and len(np.unique(y_win)) >= 2:
        try:
            score_win = roc_auc_score(y_win, p_win)
        except Exception:
            score_win = 0.5
    else:
        score_win = 0.5
    overall = (score_action + score_point + score_win) / 3.0
    return {"action_f1": float(score_action), "point_f1": float(score_point), "win_auc": float(score_win), "overall": float(overall)}


def train_cpu_fold_job(job: Dict[str, Any]) -> Dict[str, Any]:
    set_seed(job.get("seed", SEED) + int(job["fold"]))
    model_name = job["model_name"]
    fold = int(job["fold"])
    out_dir = job["out_dir"]
    pred_path = os.path.join(out_dir, f"pred_cpu_v5prob_{model_name}_fold{fold}.npz")
    if job.get("resume", True) and os.path.exists(pred_path):
        return {"model": model_name, "fold": fold, "device": "cpu", "path": pred_path, "status": "skipped"}

    train_feat = _read_table_cache(job["train_feat_path"])
    test_feat = _read_table_cache(job["test_feat_path"])
    valid_mask = train_feat["fold"].astype(int) == fold
    train_mask = ~valid_mask
    feature_cols = get_model_feature_columns(train_feat)
    X_tr, X_va = align_features(train_feat.loc[train_mask], train_feat.loc[valid_mask], feature_cols)
    _, X_te = align_features(train_feat.loc[train_mask], test_feat, feature_cols)

    y_action_tr = train_feat.loc[train_mask, "target_action"].astype(int).clip(0, NUM_ACTION_CLASSES - 1).to_numpy()
    y_action_va = train_feat.loc[valid_mask, "target_action"].astype(int).clip(0, NUM_ACTION_CLASSES - 1).to_numpy()
    y_point_tr = train_feat.loc[train_mask, "target_point"].astype(int).clip(0, NUM_POINT_CLASSES - 1).to_numpy()
    y_point_va = train_feat.loc[valid_mask, "target_point"].astype(int).clip(0, NUM_POINT_CLASSES - 1).to_numpy()
    y_win_tr = train_feat.loc[train_mask, "target_win"].astype(int).clip(0, 1).to_numpy()
    y_win_va = train_feat.loc[valid_mask, "target_win"].astype(int).clip(0, 1).to_numpy()

    print(f"[CPU] train {model_name} fold={fold} features={len(feature_cols)}")
    m_action = fit_cpu_model(model_name, X_tr, y_action_tr, X_va, y_action_va, task="action", seed=job.get("seed", SEED) + 11)
    p_action_va = predict_proba_safe(m_action, X_va, NUM_ACTION_CLASSES, "action")
    p_action_te = predict_proba_safe(m_action, X_te, NUM_ACTION_CLASSES, "action")

    m_point = fit_cpu_model(model_name, X_tr, y_point_tr, X_va, y_point_va, task="point", seed=job.get("seed", SEED) + 22)
    p_point_va = predict_proba_safe(m_point, X_va, NUM_POINT_CLASSES, "point")
    p_point_te = predict_proba_safe(m_point, X_te, NUM_POINT_CLASSES, "point")

    m_win = fit_cpu_model(model_name, X_tr, y_win_tr, X_va, y_win_va, task="win", seed=job.get("seed", SEED) + 33)
    p_win_va = predict_proba_safe(m_win, X_va, NUM_WIN_CLASSES, "win")
    p_win_te = predict_proba_safe(m_win, X_te, NUM_WIN_CLASSES, "win")

    metrics = compute_task_score(y_action_va, p_action_va, y_point_va, p_point_va, y_win_va, p_win_va)
    print(f"[CPU DONE] {model_name} fold={fold} {metrics}")

    np.savez_compressed(
        pred_path,
        val_index=np.where(valid_mask.to_numpy())[0],
        test_rally_uid=test_feat["rally_uid"].to_numpy(),
        val_action=p_action_va,
        val_point=p_point_va,
        val_win=p_win_va,
        test_action=p_action_te,
        test_point=p_point_te,
        test_win=p_win_te,
        metrics=json.dumps(metrics),
    )
    del train_feat, test_feat, X_tr, X_va, X_te
    gc.collect()
    return {"model": model_name, "fold": fold, "device": "cpu", "path": pred_path, "status": "done", "metrics": metrics}


def _torch_import():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence
    return torch, nn, F, Dataset, DataLoader, pad_sequence


class SequencePrefixDataset:                                                                                          
    pass


def get_device_string():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass
    return "cpu"


def _read_table_cache(path: str) -> pd.DataFrame:
    if path.endswith((".pkl", ".pickle")):
        return pd.read_pickle(path)
    try:
        return pd.read_parquet(path)
    except Exception:
        alt = os.path.splitext(path)[0] + ".pkl"
        if os.path.exists(alt):
            return pd.read_pickle(alt)
        raise


def train_sequence_fold_job(job: Dict[str, Any]) -> Dict[str, Any]:


    torch, nn, F, Dataset, DataLoader, pad_sequence = _torch_import()
    model_name = str(job.get("model_name", "gru")).lower()
    if model_name in {"causal", "causal_transformer", "gpt", "mini_gpt"}:
        model_name = "transformer"
    if model_name not in {"gru", "tcn", "transformer"}:
        raise ValueError(f"Unknown sequence model: {model_name}")

    set_seed(job.get("seed", SEED) + int(job["fold"]))
    fold = int(job["fold"])
    out_dir = job["out_dir"]
    pred_path = os.path.join(out_dir, f"pred_gpu_v5prob_{model_name}_fold{fold}.npz")
    if job.get("resume", True) and os.path.exists(pred_path):
        return {"model": model_name, "fold": fold, "device": "gpu", "path": pred_path, "status": "skipped"}

    device = torch.device(get_device_string())
    print(f"[GPU] train {model_name} fold={fold} device={device}")

    train_seq = _read_table_cache(job["train_seq_path"])
    test_seq = _read_table_cache(job["test_seq_path"])
    valid_groups = set(job["fold_group_values"][str(fold)])

    cat_cols = BASE_CAT_COLS + (["hmm_state"] if "hmm_state" in train_seq.columns else [])
    num_cols = BASE_NUM_COLS + [c for c in train_seq.columns if c.startswith("hmm_prob_") or c in ["hmm_state_prob_max", "hmm_state_entropy"]]
                                                                           
    num_cols += [c for c in ["action_point_combo_bl", "action_side_combo_bl", "action_depth_combo_bl", "spin_strength_combo_bl"] if c in train_seq.columns and c not in cat_cols]
    for c in cat_cols:
        if c not in train_seq.columns:
            train_seq[c] = 0
        if c not in test_seq.columns:
            test_seq[c] = 0
        train_seq[c] = pd.to_numeric(train_seq[c], errors="coerce").fillna(0).astype(np.int64)
        test_seq[c] = pd.to_numeric(test_seq[c], errors="coerce").fillna(0).astype(np.int64)
    for c in num_cols:
        if c not in train_seq.columns:
            train_seq[c] = 0.0
        if c not in test_seq.columns:
            test_seq[c] = 0.0
        train_seq[c] = pd.to_numeric(train_seq[c], errors="coerce").fillna(0.0).astype(np.float32)
        test_seq[c] = pd.to_numeric(test_seq[c], errors="coerce").fillna(0.0).astype(np.float32)

    max_seq_len = int(job.get("max_seq_len", 15))

    class TorchSeqDataset(Dataset):
        def __init__(self, df, is_train=True, groups_keep=None):
            self.samples = []
            df = df.copy()
            df["row_order"] = np.arange(len(df))
            for rally_uid, g in df.groupby("rally_uid", sort=False):
                if groups_keep is not None:
                    group_val = g[job["group_col"]].iloc[0] if job["group_col"] in g.columns else rally_uid
                    if group_val not in groups_keep:
                        continue
                g = g.sort_values(["strikeNumber", "row_order"]).reset_index(drop=True)
                if is_train:
                    if len(g) < 2:
                        continue
                    for k in range(1, len(g)):
                        prefix = g.iloc[max(0, k - max_seq_len):k]
                        target = g.iloc[k]
                        self.samples.append({
                            "cat": prefix[cat_cols].to_numpy(np.int64),
                            "num": prefix[num_cols].to_numpy(np.float32),
                            "y_action": int(target["actionId"]),
                            "y_point": int(target["pointId"]),
                            "y_win": int(g["serverGetPoint"].iloc[0]) if "serverGetPoint" in g.columns else 0,
                            "rally_uid": rally_uid,
                        })
                else:
                    prefix = g.iloc[-max_seq_len:]
                    self.samples.append({
                        "cat": prefix[cat_cols].to_numpy(np.int64),
                        "num": prefix[num_cols].to_numpy(np.float32),
                        "rally_uid": rally_uid,
                    })
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    def collate(batch):
        cats = [torch.tensor(x["cat"], dtype=torch.long) for x in batch]
        nums = [torch.tensor(x["num"], dtype=torch.float32) for x in batch]
        cat_pad = pad_sequence(cats, batch_first=True, padding_value=PAD_IDX)
        num_pad = pad_sequence(nums, batch_first=True, padding_value=0.0)
        pad_mask = cat_pad[:, :, 0].eq(PAD_IDX)
        out = {"cat": cat_pad, "num": num_pad, "pad_mask": pad_mask, "rally_uid": [x["rally_uid"] for x in batch]}
        if "y_action" in batch[0]:
            out["y_action"] = torch.tensor([x["y_action"] for x in batch], dtype=torch.long).clamp(0, NUM_ACTION_CLASSES - 1)
            out["y_point"] = torch.tensor([x["y_point"] for x in batch], dtype=torch.long).clamp(0, NUM_POINT_CLASSES - 1)
            out["y_win"] = torch.tensor([x["y_win"] for x in batch], dtype=torch.float32).clamp(0, 1)
        return out

    train_groups = set(v for k, vals in job["fold_group_values"].items() if int(k) != fold for v in vals)
    tr_ds = TorchSeqDataset(train_seq, is_train=True, groups_keep=train_groups)
    va_ds = TorchSeqDataset(train_seq, is_train=True, groups_keep=valid_groups)
    te_ds = TorchSeqDataset(test_seq, is_train=False, groups_keep=None)

    if len(tr_ds) == 0 or len(va_ds) == 0:
        raise RuntimeError("Sequence dataset is empty; check fold/group columns.")

    batch_size = int(job.get("batch_size", 16))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate, pin_memory=(device.type == "cuda"))
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate, pin_memory=(device.type == "cuda"))
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate, pin_memory=(device.type == "cuda"))

    cardinalities = []
    for c in cat_cols:
        m = max(int(train_seq[c].max()) if c in train_seq.columns else 0, int(test_seq[c].max()) if c in test_seq.columns else 0)
        cardinalities.append(max(m + 1, 2))

    class SequenceBase(nn.Module):
        def __init__(self, cardinalities, n_num, emb_dim=8, hidden=128, dropout=0.18):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(int(card), emb_dim) for card in cardinalities])
            self.num_proj = nn.Sequential(nn.Linear(n_num, 32), nn.LayerNorm(32), nn.GELU())
            self.in_dim = len(cardinalities) * emb_dim + 32
            self.input = nn.Sequential(nn.Linear(self.in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
            self.hidden = hidden
            self.dropout = dropout
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
            self.action = nn.Linear(hidden, NUM_ACTION_CLASSES)
            self.point = nn.Linear(hidden, NUM_POINT_CLASSES)
            self.win = nn.Linear(hidden, 1)
        def embed_inputs(self, cat, num):
            cat = cat.clone()
            cat[cat < 0] = 0
            emb_list = []
            for i, emb in enumerate(self.embs):
                x = cat[:, :, i].clamp(0, emb.num_embeddings - 1)
                emb_list.append(emb(x))
            num_x = self.num_proj(num)
            x = torch.cat(emb_list + [num_x], dim=-1)
            return self.input(x)
        def heads(self, last):
            h = self.head(last)
            return self.action(h), self.point(h), self.win(h).squeeze(-1)

    class GRUMultiTask(SequenceBase):
        def __init__(self, cardinalities, n_num, emb_dim=8, hidden=128, dropout=0.18):
            super().__init__(cardinalities, n_num, emb_dim, hidden, dropout)
            self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True, num_layers=1)
            self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout))
        def forward(self, cat, num, pad_mask):
            x = self.embed_inputs(cat, num)
            out, _ = self.gru(x)
            lengths = (~pad_mask).sum(dim=1).clamp(min=1) - 1
            idx = torch.arange(out.size(0), device=out.device)
            return self.heads(out[idx, lengths])

    class Chomp1d(nn.Module):
        def __init__(self, chomp_size):
            super().__init__(); self.chomp_size = int(chomp_size)
        def forward(self, x):
            if self.chomp_size <= 0:
                return x
            return x[:, :, :-self.chomp_size].contiguous()

    class TCNBlock(nn.Module):
        def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.18):
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
                Chomp1d(padding), nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
                Chomp1d(padding), nn.GELU(), nn.Dropout(dropout),
            )
            self.norm = nn.LayerNorm(channels)
        def forward(self, x):
                      
            y = self.net(x.transpose(1, 2)).transpose(1, 2)
            return self.norm(x + y)

    class TCNMultiTask(SequenceBase):
        def __init__(self, cardinalities, n_num, emb_dim=8, hidden=128, dropout=0.18):
            super().__init__(cardinalities, n_num, emb_dim, hidden, dropout)
            self.blocks = nn.ModuleList([TCNBlock(hidden, dilation=d, dropout=dropout) for d in [1, 2, 4, 8]])
        def forward(self, cat, num, pad_mask):
            x = self.embed_inputs(cat, num)
            for b in self.blocks:
                x = b(x)
            lengths = (~pad_mask).sum(dim=1).clamp(min=1) - 1
            idx = torch.arange(x.size(0), device=x.device)
            return self.heads(x[idx, lengths])

    class CausalTransformerMultiTask(SequenceBase):
        def __init__(self, cardinalities, n_num, emb_dim=8, hidden=128, dropout=0.18, nhead=4, layers=2, max_len=64):
            super().__init__(cardinalities, n_num, emb_dim, hidden, dropout)
            nhead = max(1, min(nhead, hidden))
            while hidden % nhead != 0 and nhead > 1:
                nhead -= 1
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=nhead, dim_feedforward=hidden * 3,
                dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
            nn.init.normal_(self.pos, std=0.02)
        def forward(self, cat, num, pad_mask):
            x = self.embed_inputs(cat, num)
            T = x.size(1)
            x = x + self.pos[:, :T, :]
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            out = self.encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)
            lengths = (~pad_mask).sum(dim=1).clamp(min=1) - 1
            idx = torch.arange(out.size(0), device=out.device)
            return self.heads(out[idx, lengths])

    if model_name == "gru":
        model = GRUMultiTask(cardinalities, len(num_cols), hidden=int(job.get("hidden", 128))).to(device)
    elif model_name == "tcn":
        model = TCNMultiTask(cardinalities, len(num_cols), hidden=int(job.get("hidden", 128))).to(device)
    else:
        model = CausalTransformerMultiTask(cardinalities, len(num_cols), hidden=int(job.get("hidden", 128)), max_len=max_seq_len + 2).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(job.get("lr", 8e-4)), weight_decay=1e-4)

    y_act_all = np.array([s["y_action"] for s in tr_ds.samples], dtype=int)
    y_point_all = np.array([s["y_point"] for s in tr_ds.samples], dtype=int)
    y_win_all = np.array([s["y_win"] for s in tr_ds.samples], dtype=int)
    def class_weight(y, n):
        cnt = np.bincount(np.clip(y, 0, n - 1), minlength=n).astype(np.float32) + 1.0
        w = cnt.sum() / (n * cnt)
                                                                                           
        w = np.power(w, 0.5)
        w = w / max(np.mean(w), 1e-6)
        return torch.tensor(w, dtype=torch.float32, device=device)
    w_action = class_weight(y_act_all, NUM_ACTION_CLASSES)
    w_point = class_weight(y_point_all, NUM_POINT_CLASSES)
    pos = max(y_win_all.sum(), 1)
    neg = max(len(y_win_all) - y_win_all.sum(), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    best_score = -1
    best_state = None
    epochs = int(job.get("epochs", 8))
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for b in tr_loader:
            cat = b["cat"].to(device)
            num = b["num"].to(device)
            mask = b["pad_mask"].to(device)
            ya = b["y_action"].to(device)
            yp = b["y_point"].to(device)
            yw = b["y_win"].to(device)
            la, lp, lw = model(cat, num, mask)
            loss = F.cross_entropy(la, ya, weight=w_action) + F.cross_entropy(lp, yp, weight=w_point) + F.binary_cross_entropy_with_logits(lw, yw, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        pva = predict_sequence_loader(model, va_loader, device)
        metrics = compute_task_score(pva["y_action"], pva["action"], pva["y_point"], pva["point"], pva["y_win"], pva["win"])
        print(f"[{model_name.upper()}] fold={fold} epoch={ep}/{epochs} loss={np.mean(losses):.4f} {metrics}")
        if metrics["overall"] > best_score:
            best_score = metrics["overall"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    pva = predict_sequence_loader(model, va_loader, device)
    pte = predict_sequence_loader(model, te_loader, device, has_y=False)
    metrics = compute_task_score(pva["y_action"], pva["action"], pva["y_point"], pva["point"], pva["y_win"], pva["win"])
    print(f"[GPU DONE] {model_name} fold={fold} {metrics}")

    np.savez_compressed(
        pred_path,
        val_rally_uid=np.array(pva["rally_uid"], dtype=object),
        test_rally_uid=np.array(pte["rally_uid"], dtype=object),
        val_action=pva["action"], val_point=pva["point"], val_win=pva["win"],
        test_action=pte["action"], test_point=pte["point"], test_win=pte["win"],
        y_action=pva["y_action"], y_point=pva["y_point"], y_win=pva["y_win"],
        metrics=json.dumps(metrics),
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    del model, train_seq, test_seq, tr_ds, va_ds, te_ds
    gc.collect()
    return {"model": model_name, "fold": fold, "device": "gpu", "path": pred_path, "status": "done", "metrics": metrics}


def train_gru_fold_job(job: Dict[str, Any]) -> Dict[str, Any]:
    job = dict(job)
    job["model_name"] = "gru"
    return train_sequence_fold_job(job)


def predict_sequence_loader(model, loader, device, has_y=True):
    import torch
    model.eval()
    out = {"rally_uid": [], "action": [], "point": [], "win": []}
    if has_y:
        out.update({"y_action": [], "y_point": [], "y_win": []})
    with torch.no_grad():
        for b in loader:
            cat = b["cat"].to(device)
            num = b["num"].to(device)
            mask = b["pad_mask"].to(device)
            la, lp, lw = model(cat, num, mask)
            out["action"].append(torch.softmax(la, dim=1).cpu().numpy())
            out["point"].append(torch.softmax(lp, dim=1).cpu().numpy())
            out["win"].append(torch.sigmoid(lw).cpu().numpy())
            out["rally_uid"].extend(b["rally_uid"])
            if has_y:
                out["y_action"].append(b["y_action"].cpu().numpy())
                out["y_point"].append(b["y_point"].cpu().numpy())
                out["y_win"].append(b["y_win"].cpu().numpy())
    for k in ["action", "point", "win"]:
        out[k] = np.concatenate(out[k], axis=0) if out[k] else np.array([])
    if has_y:
        for k in ["y_action", "y_point", "y_win"]:
            out[k] = np.concatenate(out[k], axis=0) if out[k] else np.array([])
    return out


def predict_gru_loader(model, loader, device, has_y=True):
    return predict_sequence_loader(model, loader, device, has_y=has_y)

                                                              
def build_all_features(args) -> Tuple[str, str, str, str, Dict[str, List[Any]], str]:
    ensure_dir(args.out_dir)
                                                                                     
                                                             
    train_feat_path = os.path.join(args.out_dir, "train_prefix_features_v5_probwin.pkl")
    test_feat_path = os.path.join(args.out_dir, "test_prefix_features_v5_probwin.pkl")
    train_seq_path = os.path.join(args.out_dir, "train_sequence_features_v5_probwin.pkl")
    test_seq_path = os.path.join(args.out_dir, "test_sequence_features_v5_probwin.pkl")
    fold_json_path = os.path.join(args.out_dir, "fold_groups_v5_probwin.json")

    if args.resume and all(os.path.exists(p) for p in [train_feat_path, test_feat_path, train_seq_path, test_seq_path, fold_json_path]):
        with open(fold_json_path, "r", encoding="utf-8") as f:
            fold_group_values = json.load(f)
        return train_feat_path, test_feat_path, train_seq_path, test_seq_path, fold_group_values, args.group_col_cached or "match"

    train_csv, test_csv = find_input_files(args.train_path, args.test_path)
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"找不到 train csv: {train_csv}")
    if test_csv is None or not os.path.exists(test_csv):
        raise FileNotFoundError(f"找不到 test csv: {test_csv}")

    print(f"[LOAD] train={train_csv}, test={test_csv}")
    train_raw = pd.read_csv(train_csv)
    test_raw = pd.read_csv(test_csv)
    if "rally_uid" not in train_raw.columns:
        train_raw["rally_uid"] = np.arange(len(train_raw))
    if "rally_uid" not in test_raw.columns:
        test_raw["rally_uid"] = np.arange(len(test_raw))

                                                             
    print("[FE] mapping + motion features")
    train_seq = add_sequence_motion_features(train_raw)
    test_seq = add_sequence_motion_features(test_raw)

    group_col = args.group_col or choose_cv_group_column(train_raw)
    if group_col not in train_seq.columns:
        group_col = "rally_uid"
    print(f"[CV] group_col={group_col}")
    unique_groups = train_seq[group_col].drop_duplicates().to_numpy()
    gkf = GroupKFold(n_splits=args.folds)
    fold_group_values: Dict[str, List[Any]] = {}
    for fold, (_, va_idx) in enumerate(gkf.split(unique_groups, groups=unique_groups), start=0):
        vals = unique_groups[va_idx].tolist()
                                                  
        fold_group_values[str(fold)] = [int(x) if isinstance(x, (np.integer,)) else x for x in vals]
    with open(fold_json_path, "w", encoding="utf-8") as f:
        json.dump(fold_group_values, f, ensure_ascii=False)

                                                                  
    if args.use_hmm:
        print("[FE] row-level HMM/KMeans state features for GRU")
        row_for_hmm = train_seq.copy()
                                                                
        row_feat_cols = [c for c in ["actionId", "pointId", "aid_group", "pid_side", "pid_depth", "spinId", "strengthId", "positionId", "scoreDiff_bl", "rally_phase_bl", "aggressive_shot_bl", "placement_volatility_bl"] if c in row_for_hmm.columns]
        tmp_train = pd.DataFrame({f"last_{c}": train_seq[c] for c in row_feat_cols})
        tmp_test = pd.DataFrame({f"last_{c}": test_seq[c] for c in row_feat_cols})
        hmm_encoder_row = HMMStateEncoder(n_states=args.hmm_states, random_state=args.seed)
        tmp_train2 = hmm_encoder_row.fit_transform(tmp_train)
        tmp_test2 = hmm_encoder_row.transform(tmp_test)
        for c in ["hmm_state", "hmm_state_prob_max", "hmm_state_entropy"] + [f"hmm_prob_{i}" for i in range(args.hmm_states)]:
            if c in tmp_train2.columns:
                train_seq[c] = tmp_train2[c].values
                test_seq[c] = tmp_test2[c].values

    print("[FE] build sliding-prefix tabular features")
    train_feat_base = make_sliding_feature_df(train_seq, max_seq_len=args.max_seq_len, is_train=True, desc="train prefix")
    test_feat_base = make_sliding_feature_df(test_seq, max_seq_len=args.max_seq_len, is_train=False, desc="test prefix")

                                                               
    group_lookup = train_seq.groupby("rally_uid")[group_col].first().to_dict()
    train_feat_base["cv_group"] = train_feat_base["rally_uid"].map(group_lookup)
    train_feat_base["fold"] = -1
    for fold, vals in fold_group_values.items():
        train_feat_base.loc[train_feat_base["cv_group"].isin(vals), "fold"] = int(fold)
    train_feat_base["fold"] = train_feat_base["fold"].astype(int)
    if (train_feat_base["fold"] < 0).any():
        raise RuntimeError("Some prefix rows were not assigned to folds.")

                                                
    if args.use_hmm:
        print("[FE] prefix-level HMM/KMeans state features")
        hmm_encoder = HMMStateEncoder(n_states=args.hmm_states, random_state=args.seed)
                                                                                                                   
        train_feat_base = hmm_encoder.fit_transform(train_feat_base)
        test_feat_base = hmm_encoder.transform(test_feat_base)

                                                                                                               
    if args.use_dirichlet:
        print("[FE] Dirichlet transition features OOF + test full")
        train_feat = train_feat_base.copy()
        test_feat = test_feat_base.copy()
        for target_col, n_cls in [("target_action", NUM_ACTION_CLASSES), ("target_point", NUM_POINT_CLASSES), ("target_win", NUM_WIN_CLASSES)]:
            oof_parts = []
            for fold in range(args.folds):
                tr = train_feat_base[train_feat_base["fold"] != fold].copy()
                va = train_feat_base[train_feat_base["fold"] == fold].copy()
                enc = DirichletTransitionEncoder(TRANSITION_SPECS_COMMON, target_col=target_col, num_classes=n_cls, alpha=args.dirichlet_alpha, prefix="dir")
                enc.fit(tr)
                va_enc = enc.transform(va)
                oof_parts.append(va_enc)
            train_feat = pd.concat(oof_parts, axis=0).sort_index()
                                     
            enc_full = DirichletTransitionEncoder(TRANSITION_SPECS_COMMON, target_col=target_col, num_classes=n_cls, alpha=args.dirichlet_alpha, prefix="dir")
            enc_full.fit(train_feat_base)
            test_feat = enc_full.transform(test_feat)
                                                                                                                      
                                                                                                              
        train_feat = train_feat.sort_index().reset_index(drop=True)
        test_feat = test_feat.reset_index(drop=True)
    else:
        train_feat = train_feat_base
        test_feat = test_feat_base

                  
    print("[SAVE] feature caches")
    train_feat.to_pickle(train_feat_path)
    test_feat.to_pickle(test_feat_path)
    train_seq.to_pickle(train_seq_path)
    test_seq.to_pickle(test_seq_path)
    return train_feat_path, test_feat_path, train_seq_path, test_seq_path, fold_group_values, group_col


def run_jobs(jobs: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    backend = args.backend
    if backend == "auto":
        try:
            import ray        
            backend = "ray"
        except Exception:
            backend = "serial"
    print(f"[RUN] backend={backend}, jobs={len(jobs)}")

    if backend == "ray":
        import ray
        if ray.is_initialized():
            ray.shutdown()
        ray.init(num_cpus=args.n_cpus, num_gpus=args.n_gpus, include_dashboard=False, ignore_reinit_error=True)
        cpu_remote = ray.remote(num_cpus=args.cpu_per_job, num_gpus=0)(train_cpu_fold_job)
        gpu_remote = ray.remote(num_cpus=args.gpu_cpu_per_job, num_gpus=args.gpu_per_job)(train_sequence_fold_job)
        futures = []
        for job in jobs:
            if job["device_type"] == "gpu":
                futures.append(gpu_remote.remote(job))
            else:
                futures.append(cpu_remote.remote(job))
        results = []
        remaining = futures
        while remaining:
            done, remaining = ray.wait(remaining, num_returns=1)
            res = ray.get(done[0])
            results.append(res)
            print(f"[JOB FINISH] {res.get('device')} {res.get('model')} fold={res.get('fold')} status={res.get('status')}")
        ray.shutdown()
        return results

                                                                            
    results = []
    for job in jobs:
        if job["device_type"] == "gpu":
            res = train_sequence_fold_job(job)
        else:
            res = train_cpu_fold_job(job)
        print(f"[JOB FINISH] {res.get('device')} {res.get('model')} fold={res.get('fold')} status={res.get('status')}")
        results.append(res)
    return results


def _read_feature_cache(path: str) -> pd.DataFrame:


    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"feature cache not found: {path}")
    if path.endswith(".pkl") or path.endswith(".pickle"):
        return pd.read_pickle(path)
    try:
        return pd.read_parquet(path)
    except Exception:
        alt = os.path.splitext(path)[0] + ".pkl"
        if os.path.exists(alt):
            return pd.read_pickle(alt)
        raise


def _safe_prob_matrix(proba, eps: float = 1e-12):
    proba = np.asarray(proba, dtype=np.float64)
    proba = np.nan_to_num(proba, nan=eps, posinf=1.0, neginf=eps)
    proba = np.clip(proba, eps, 1.0)
    s = proba.sum(axis=1, keepdims=True)
    return (proba / np.clip(s, eps, None)).astype(np.float64)


def _softmax_from_losses(losses, temperature: float = 0.15):
    losses = np.asarray(losses, dtype=np.float64)
    if losses.size == 0:
        return losses
    finite = np.isfinite(losses)
    if not finite.any():
        return np.ones_like(losses, dtype=np.float64) / len(losses)
    fill = np.nanmedian(losses[finite]) + 1.0
    losses = np.where(finite, losses, fill)
    temp = max(float(temperature), 1e-6)
    score = -losses / temp
    score = score - np.max(score)
    w = np.exp(score)
    w = w / np.clip(w.sum(), 1e-12, None)
    return w


def _multiclass_nll_vector(y, proba, n_classes):
    y = np.asarray(y, dtype=np.int64).clip(0, n_classes - 1)
    p = _safe_prob_matrix(proba)
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))


def _binary_logloss_vector(y, p):
    y = np.asarray(y, dtype=np.float64).clip(0, 1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = np.clip(np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _phase_group_from_feature_df(df: pd.DataFrame) -> np.ndarray:


    n = len(df)
    if n == 0:
        return np.array([], dtype=np.int16)
    if "last_rally_phase_bl" in df.columns:
        val = pd.to_numeric(df["last_rally_phase_bl"], errors="coerce").fillna(1.0).to_numpy(float)
                                                                  
        phase = np.rint(val * 4).astype(np.int16)
        return np.clip(phase, 1, 4)
    phase = np.full(n, 4, dtype=np.int16)
    candidates = [
        ("last_is_serve_phase_bl", 1),
        ("last_is_receive_phase_bl", 2),
        ("last_is_third_ball_phase_bl", 3),
        ("last_is_rally_phase_bl", 4),
    ]
    for col, code in candidates:
        if col in df.columns:
            mask = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(float) > 0.5
            phase[mask] = code
    return phase


def _prediction_distribution_table(name, values):
    print(f"{name} distribution:")
    print(pd.Series(values).value_counts(normalize=True).sort_index())


def _load_prediction_records(result_paths: List[str], train_feat_path: Optional[str], test_feat_path: Optional[str]):

    train_feat = None
    test_feat = None
    if train_feat_path is not None and os.path.exists(train_feat_path):
        train_feat = _read_feature_cache(train_feat_path)
    if test_feat_path is not None and os.path.exists(test_feat_path):
        test_feat = _read_feature_cache(test_feat_path)

    test_uids = None
    test_phase = None
    if test_feat is not None and "rally_uid" in test_feat.columns:
        test_uids = test_feat["rally_uid"].to_numpy()
        test_phase = _phase_group_from_feature_df(test_feat)

    records = []
    for path in result_paths:
        if not os.path.exists(path):
            continue
        d = np.load(path, allow_pickle=True)
        if "test_rally_uid" not in d:
            continue
        uid = d["test_rally_uid"]
        if test_uids is None:
            test_uids = uid
            test_phase = np.full(len(test_uids), 4, dtype=np.int16)
        if len(uid) != len(test_uids) or np.any(uid.astype(str) != test_uids.astype(str)):
            order = pd.Series(np.arange(len(uid)), index=uid.astype(str))
            idx = np.array([int(order.loc[str(x)]) for x in test_uids], dtype=int)
        else:
            idx = np.arange(len(uid))

        name = os.path.splitext(os.path.basename(path))[0]
        rec = {"name": name, "path": path}
        if "test_action" in d:
            arr = d["test_action"][idx]
            if arr.ndim == 2 and arr.shape[1] == NUM_ACTION_CLASSES:
                rec["test_action"] = _safe_prob_matrix(arr).astype(np.float32)
        if "test_point" in d:
            arr = d["test_point"][idx]
            if arr.ndim == 2 and arr.shape[1] == NUM_POINT_CLASSES:
                rec["test_point"] = _safe_prob_matrix(arr).astype(np.float32)
        if "test_win" in d:
            rec["test_win"] = np.asarray(d["test_win"][idx]).reshape(-1).astype(np.float32)

                                
        if "val_index" in d and train_feat is not None:
            val_index = np.asarray(d["val_index"], dtype=int)
            rec["val_phase"] = _phase_group_from_feature_df(train_feat.iloc[val_index])
            if "target_action" in train_feat.columns:
                rec["y_action"] = train_feat.iloc[val_index]["target_action"].astype(int).clip(0, NUM_ACTION_CLASSES - 1).to_numpy()
            if "target_point" in train_feat.columns:
                rec["y_point"] = train_feat.iloc[val_index]["target_point"].astype(int).clip(0, NUM_POINT_CLASSES - 1).to_numpy()
            if "target_win" in train_feat.columns:
                rec["y_win"] = train_feat.iloc[val_index]["target_win"].astype(int).clip(0, 1).to_numpy()
            if "val_action" in d:
                rec["val_action"] = _safe_prob_matrix(d["val_action"]).astype(np.float32)
            if "val_point" in d:
                rec["val_point"] = _safe_prob_matrix(d["val_point"]).astype(np.float32)
            if "val_win" in d:
                rec["val_win"] = np.asarray(d["val_win"]).reshape(-1).astype(np.float32)
        else:
                                                                                         
            if "y_action" in d:
                rec["y_action"] = np.asarray(d["y_action"], dtype=int).clip(0, NUM_ACTION_CLASSES - 1)
            if "y_point" in d:
                rec["y_point"] = np.asarray(d["y_point"], dtype=int).clip(0, NUM_POINT_CLASSES - 1)
            if "y_win" in d:
                rec["y_win"] = np.asarray(d["y_win"], dtype=int).clip(0, 1)
            if "val_action" in d:
                rec["val_action"] = _safe_prob_matrix(d["val_action"]).astype(np.float32)
            if "val_point" in d:
                rec["val_point"] = _safe_prob_matrix(d["val_point"]).astype(np.float32)
            if "val_win" in d:
                rec["val_win"] = np.asarray(d["val_win"]).reshape(-1).astype(np.float32)
                                                                                          
            if "val_point" in rec:
                rec["val_phase"] = np.full(len(rec["val_point"]), 0, dtype=np.int16)

        records.append(rec)

    if test_uids is None:
        raise RuntimeError("No prediction files found for ensemble.")
    if test_phase is None:
        test_phase = np.full(len(test_uids), 4, dtype=np.int16)
    return records, test_uids, test_phase


def _global_task_weight(records, task: str, n_classes: int, temperature: float):
    losses = []
    usable = []
    for rec in records:
        val_key = f"val_{task}"
        y_key = f"y_{task}"
        test_key = f"test_{task}"
        if val_key not in rec or y_key not in rec or test_key not in rec:
            continue
        if task == "win":
            loss = float(np.mean(_binary_logloss_vector(rec[y_key], rec[val_key])))
        else:
            loss = float(np.mean(_multiclass_nll_vector(rec[y_key], rec[val_key], n_classes)))
        losses.append(loss)
        usable.append(rec)
    if not usable:
        return [], np.array([], dtype=np.float64), []
    w = _softmax_from_losses(losses, temperature=temperature)
    return usable, w, losses


def _classwise_weight(records, task: str, n_classes: int, temperature: float, shrink: float, phase: Optional[int] = None):


    usable = []
    loss_mat = []
    supports = []
    for rec in records:
        val_key = f"val_{task}"
        y_key = f"y_{task}"
        test_key = f"test_{task}"
        if val_key not in rec or y_key not in rec or test_key not in rec:
            continue
        y = np.asarray(rec[y_key], dtype=int).clip(0, n_classes - 1)
        proba = _safe_prob_matrix(rec[val_key])
        if phase is not None and "val_phase" in rec:
            ph = np.asarray(rec["val_phase"])
            mask_phase = ph == int(phase)
                                                                             
            if mask_phase.sum() >= max(30, n_classes * 3):
                y_use = y[mask_phase]
                p_use = proba[mask_phase]
            else:
                y_use = y
                p_use = proba
        else:
            y_use = y
            p_use = proba
        global_loss = float(np.mean(_multiclass_nll_vector(y_use, p_use, n_classes)))
        losses_c = []
        supp_c = []
        for c in range(n_classes):
            m = y_use == c
            supp = int(m.sum())
            supp_c.append(supp)
            if supp == 0:
                losses_c.append(global_loss)
            else:
                cls_loss = float(np.mean(-np.log(np.clip(p_use[m, c], 1e-12, 1.0))))
                                                                      
                lam = supp / (supp + max(float(shrink), 0.0))
                losses_c.append(lam * cls_loss + (1.0 - lam) * global_loss)
        usable.append(rec)
        loss_mat.append(losses_c)
        supports.append(supp_c)
    if not usable:
        return [], np.zeros((0, n_classes)), np.zeros((0, n_classes)), np.zeros((0, n_classes), dtype=int)
    loss_mat = np.asarray(loss_mat, dtype=np.float64)
    weights = np.zeros_like(loss_mat)
    for c in range(n_classes):
        weights[:, c] = _softmax_from_losses(loss_mat[:, c], temperature=temperature)
    return usable, weights, loss_mat, np.asarray(supports, dtype=int)


def _blend_multiclass(records, task: str, n_classes: int, test_phase: np.ndarray, method: str, temperature: float, classwise: bool, shrink: float):
    if method == "average":
        arrs = [rec[f"test_{task}"] for rec in records if f"test_{task}" in rec]
        if not arrs:
            return np.zeros((len(test_phase), n_classes), dtype=np.float32), {}
        return np.mean(arrs, axis=0).astype(np.float32), {"method": "average", "models": [rec["name"] for rec in records if f"test_{task}" in rec]}

    usable_global, w_global, losses_global = _global_task_weight(records, task, n_classes, temperature)
    if not usable_global:
        arrs = [rec[f"test_{task}"] for rec in records if f"test_{task}" in rec]
        return np.mean(arrs, axis=0).astype(np.float32), {"method": "fallback_average"}

    if method == "router":
                                                                                              
        out = np.zeros((len(test_phase), n_classes), dtype=np.float64)
        phase_info = {}
        for ph in sorted(set(np.asarray(test_phase).astype(int).tolist())):
            mask = np.asarray(test_phase).astype(int) == ph
            if not mask.any():
                continue
            usable, w_cls, loss_mat, support_mat = _classwise_weight(records, task, n_classes, temperature, shrink, phase=ph)
            if not usable:
                continue
            part = np.zeros((mask.sum(), n_classes), dtype=np.float64)
            for mi, rec in enumerate(usable):
                part += rec[f"test_{task}"][mask] * w_cls[mi].reshape(1, -1)
            out[mask] = part
            phase_info[str(ph)] = {
                "models": [r["name"] for r in usable],
                "classwise_weights": {str(c): {usable[i]["name"]: float(w_cls[i, c]) for i in range(len(usable))} for c in range(n_classes)},
                "classwise_losses": {str(c): {usable[i]["name"]: float(loss_mat[i, c]) for i in range(len(usable))} for c in range(n_classes)},
            }
        out = _safe_prob_matrix(out).astype(np.float32)
        return out, {"method": "router", "phase_groups": phase_info}

    if classwise:
        usable, w_cls, loss_mat, support_mat = _classwise_weight(records, task, n_classes, temperature, shrink, phase=None)
        out = np.zeros((len(test_phase), n_classes), dtype=np.float64)
        for mi, rec in enumerate(usable):
            out += rec[f"test_{task}"] * w_cls[mi].reshape(1, -1)
        out = _safe_prob_matrix(out).astype(np.float32)
        info = {
            "method": "learned_loss_classwise",
            "models": [r["name"] for r in usable],
            "classwise_weights": {str(c): {usable[i]["name"]: float(w_cls[i, c]) for i in range(len(usable))} for c in range(n_classes)},
            "classwise_losses": {str(c): {usable[i]["name"]: float(loss_mat[i, c]) for i in range(len(usable))} for c in range(n_classes)},
        }
        return out, info

    out = np.zeros((len(test_phase), n_classes), dtype=np.float64)
    for rec, w in zip(usable_global, w_global):
        out += rec[f"test_{task}"] * float(w)
    out = _safe_prob_matrix(out).astype(np.float32)
    info = {
        "method": "learned_loss_taskwise",
        "models": [r["name"] for r in usable_global],
        "weights": {rec["name"]: float(w) for rec, w in zip(usable_global, w_global)},
        "losses": {rec["name"]: float(l) for rec, l in zip(usable_global, losses_global)},
    }
    return out, info


def _blend_binary(records, test_phase: np.ndarray, method: str, temperature: float):
    if method == "average":
        arrs = [rec["test_win"] for rec in records if "test_win" in rec]
        if not arrs:
            return np.zeros(len(test_phase), dtype=np.float32), {"method": "average"}
        return np.mean(arrs, axis=0).astype(np.float32), {"method": "average", "models": [rec["name"] for rec in records if "test_win" in rec]}

    usable, w, losses = _global_task_weight(records, "win", 2, temperature)
    if not usable:
        arrs = [rec["test_win"] for rec in records if "test_win" in rec]
        return np.mean(arrs, axis=0).astype(np.float32), {"method": "fallback_average"}

                                                                                                 
    if method == "router":
        out = np.zeros(len(test_phase), dtype=np.float64)
        phase_info = {}
        for ph in sorted(set(np.asarray(test_phase).astype(int).tolist())):
            mask = np.asarray(test_phase).astype(int) == ph
            phase_losses = []
            phase_usable = []
            for rec in usable:
                if "val_phase" in rec and np.any(np.asarray(rec["val_phase"]) == ph):
                    m = np.asarray(rec["val_phase"]) == ph
                    if m.sum() >= 30:
                        loss = float(np.mean(_binary_logloss_vector(rec["y_win"][m], rec["val_win"][m])))
                    else:
                        loss = float(np.mean(_binary_logloss_vector(rec["y_win"], rec["val_win"])))
                else:
                    loss = float(np.mean(_binary_logloss_vector(rec["y_win"], rec["val_win"])))
                phase_losses.append(loss)
                phase_usable.append(rec)
            ww = _softmax_from_losses(phase_losses, temperature=temperature)
            for rec, wi in zip(phase_usable, ww):
                out[mask] += rec["test_win"][mask] * float(wi)
            phase_info[str(ph)] = {rec["name"]: float(wi) for rec, wi in zip(phase_usable, ww)}
        return out.astype(np.float32), {"method": "router", "phase_weights": phase_info}

    out = np.zeros(len(test_phase), dtype=np.float64)
    for rec, wi in zip(usable, w):
        out += rec["test_win"] * float(wi)
    return out.astype(np.float32), {
        "method": "learned_loss_taskwise",
        "weights": {rec["name"]: float(wi) for rec, wi in zip(usable, w)},
        "losses": {rec["name"]: float(li) for rec, li in zip(usable, losses)},
    }


def _build_zero_prob_override_table(raw_train: pd.DataFrame, target_col: str, min_context_samples: int = 30) -> dict:


    df = raw_train.sort_values(["rally_uid", "strikeNumber"]).copy()
    df["prev_actionId"] = df.groupby("rally_uid")["actionId"].shift(1)
    df[f"next_{target_col}"] = df.groupby("rally_uid")[target_col].shift(-1)
    df = df.dropna(subset=["prev_actionId", f"next_{target_col}"])
    if df.empty:
        return {}
    df["prev_actionId"] = df["prev_actionId"].astype(int)
    df[f"next_{target_col}"] = df[f"next_{target_col}"].astype(int)
    table = {}
    grouped = df.groupby(["prev_actionId", "actionId", "pointId"])[f"next_{target_col}"]
    for key, vals in grouped:
        if len(vals) < int(min_context_samples):
            continue
        probs = vals.value_counts(normalize=True).to_dict()
        mode = int(vals.value_counts().idxmax())
        table[tuple(int(k) for k in key)] = (probs, mode, int(len(vals)))
    return table


def _test_context_prev_last(test_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in test_raw.groupby("rally_uid"):
        g = g.sort_values("strikeNumber").reset_index(drop=True)
        if len(g) < 2:
            continue
        prev = g.iloc[-2]
        last = g.iloc[-1]
        rows.append({
            "rally_uid": int(rid),
            "prev_actionId": int(prev.get("actionId", 0)),
            "last_actionId": int(last.get("actionId", 0)),
            "last_pointId": int(last.get("pointId", 0)),
        })
    if not rows:
        return pd.DataFrame(columns=["prev_actionId", "last_actionId", "last_pointId"], index=pd.Index([], name="rally_uid"))
    return pd.DataFrame(rows).set_index("rally_uid")


def apply_zero_prob_rule_override(submission: pd.DataFrame, train_raw: pd.DataFrame, test_raw: pd.DataFrame, min_context_samples: int = 30):
    action_tbl = _build_zero_prob_override_table(train_raw, "actionId", min_context_samples=min_context_samples)
    point_tbl = _build_zero_prob_override_table(train_raw, "pointId", min_context_samples=min_context_samples)
    test_ctx = _test_context_prev_last(test_raw)
    out = submission.copy()
    changes = []
    for i, row in out.iterrows():
        rid = int(row["rally_uid"])
        if rid not in test_ctx.index:
            continue
        ctx = test_ctx.loc[rid]
        key = (int(ctx["prev_actionId"]), int(ctx["last_actionId"]), int(ctx["last_pointId"]))
        if key in action_tbl:
            probs, mode, n = action_tbl[key]
            pred = int(row["actionId"])
            if probs.get(pred, 0.0) == 0.0 and mode != pred:
                out.at[i, "actionId"] = mode
                changes.append({"rally_uid": rid, "target": "actionId", "from": pred, "to": mode, "context_n": n})
        if key in point_tbl:
            probs, mode, n = point_tbl[key]
            pred = int(row["pointId"])
            if probs.get(pred, 0.0) == 0.0 and mode != pred:
                out.at[i, "pointId"] = mode
                changes.append({"rally_uid": rid, "target": "pointId", "from": pred, "to": mode, "context_n": n})
    return out, changes


def ensemble_test_predictions(
    result_paths: List[str],
    out_dir: str,
    threshold: float = 0.5,
    train_feat_path: Optional[str] = None,
    test_feat_path: Optional[str] = None,
    method: str = "router",
    temperature: float = 0.15,
    classwise: bool = True,
    class_shrink: float = 80.0,
    use_rule_override: bool = False,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    rule_min_context: int = 30,
):


    records, test_uids, test_phase = _load_prediction_records(result_paths, train_feat_path, test_feat_path)
    method = (method or "router").lower()
    if method not in {"average", "learned_loss", "router"}:
        print(f"[WARN] unknown ensemble method={method}; fallback to router")
        method = "router"

    action_avg, action_info = _blend_multiclass(records, "action", NUM_ACTION_CLASSES, test_phase, method, temperature, classwise, class_shrink)
    point_avg, point_info = _blend_multiclass(records, "point", NUM_POINT_CLASSES, test_phase, method, temperature, classwise, class_shrink)
    win_avg, win_info = _blend_binary(records, test_phase, method, temperature)

                                                                        
    sub = pd.DataFrame({
        "rally_uid": test_uids,
        "actionId": np.argmax(action_avg, axis=1).astype(int),
        "pointId": np.argmax(point_avg, axis=1).astype(int),
        "serverGetPoint": np.clip(win_avg, 0.0, 1.0).astype(float),
    })

    rule_changes = []
    if use_rule_override:
        try:
            tr_csv, te_csv = find_input_files(train_path, test_path)
            raw_train = pd.read_csv(tr_csv)
            raw_test = pd.read_csv(te_csv)
            sub, rule_changes = apply_zero_prob_rule_override(sub, raw_train, raw_test, min_context_samples=rule_min_context)
            print(f"[RULE] zero-prob override changes={len(rule_changes)} min_context={rule_min_context}")
        except Exception as e:
            print(f"[WARN] rule override failed and was skipped: {e}")

                               
    sub["actionId"] = sub["actionId"].astype(int).clip(0, NUM_ACTION_CLASSES - 1)
    sub["pointId"] = sub["pointId"].astype(int).clip(0, NUM_POINT_CLASSES - 1)
    sub["serverGetPoint"] = pd.to_numeric(sub["serverGetPoint"], errors="coerce").fillna(0.5).clip(0.0, 1.0).astype(float)

    out_csv = os.path.join(out_dir, "submission_state_dirichlet_hmm_ensemble_probwin.csv")
    sub.to_csv(out_csv, index=False)

                                                                                                
    hard_sub = sub.copy()
    hard_sub["serverGetPoint"] = (np.asarray(win_avg) >= threshold).astype(int)
    hard_csv = os.path.join(out_dir, "submission_state_dirichlet_hmm_ensemble_hardwin_backup.csv")
    hard_sub.to_csv(hard_csv, index=False)

    prob_df = pd.DataFrame({"rally_uid": test_uids})
    for c in range(NUM_ACTION_CLASSES):
        prob_df[f"prob_action_{c}"] = action_avg[:, c]
    for c in range(NUM_POINT_CLASSES):
        prob_df[f"prob_point_{c}"] = point_avg[:, c]
    prob_df["prob_serverGetPoint"] = win_avg
    prob_df["router_phase"] = test_phase.astype(int)
    prob_csv = os.path.join(out_dir, "submission_state_dirichlet_hmm_ensemble_probs.csv")
    prob_df.to_csv(prob_csv, index=False)

    weights_info = {
        "method": method,
        "temperature": float(temperature),
        "classwise": bool(classwise),
        "class_shrink": float(class_shrink),
        "phase_meaning": {"1": "serve", "2": "receive", "3": "third_ball", "4": "rally"},
        "models": [rec["name"] for rec in records],
        "rule_override": {"enabled": bool(use_rule_override), "changes": rule_changes[:200], "n_changes": len(rule_changes), "min_context": int(rule_min_context)},
        "action": action_info,
        "point": point_info,
        "win": win_info,
        "test_phase_distribution": pd.Series(test_phase).value_counts(normalize=True).sort_index().to_dict(),
    }
    weights_path = os.path.join(out_dir, "ensemble_router_weights.json" if method == "router" else "ensemble_learned_loss_weights.json")
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(weights_info, f, ensure_ascii=False, indent=2)

    print(f"[DONE] saved {out_csv}  # submit this one")
    print(f"[DONE] saved {hard_csv}  # backup only")
    print(f"[DONE] saved {prob_csv}")
    print(f"[DONE] saved {weights_path}")
    print("Models blended:")
    for rec in records:
        print("  -", rec["name"])
    print("Prediction distribution:")
    print(sub[["actionId", "pointId"]].apply(lambda s: s.value_counts(normalize=True).sort_index()))
    print("serverGetPoint probability summary:")
    print(pd.Series(sub["serverGetPoint"]).describe().round(6).to_string())
    _prediction_distribution_table("Router/test phase", test_phase)
    return out_csv, prob_csv


def average_test_predictions(result_paths: List[str], out_dir: str, threshold: float = 0.5):
    return ensemble_test_predictions(result_paths, out_dir, threshold=threshold, method="average")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, default=None)
    p.add_argument("--test_path", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs_state_ensemble")
    p.add_argument("--backend", type=str, default="auto", choices=["auto", "ray", "serial"])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max_seq_len", type=int, default=15)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--group_col", type=str, default=None, help="CV group column, default match if exists else rally_uid")
    p.add_argument("--group_col_cached", type=str, default=None)

    p.add_argument("--use_dirichlet", action="store_true", default=True)
    p.add_argument("--no_dirichlet", action="store_true")
    p.add_argument("--dirichlet_alpha", type=float, default=1.0)
    p.add_argument("--use_hmm", action="store_true", default=True)
    p.add_argument("--no_hmm", action="store_true")
    p.add_argument("--hmm_states", type=int, default=7)

    p.add_argument("--cpu_models", type=str, default="lgbm,xgb,cat,logreg")
    p.add_argument("--gpu_models", type=str, default="gru")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=8e-4)

    p.add_argument("--n_cpus", type=int, default=max(os.cpu_count() - 2 if os.cpu_count() else 4, 2))
    p.add_argument("--n_gpus", type=float, default=1.0)
    p.add_argument("--cpu_per_job", type=int, default=2)
    p.add_argument("--gpu_cpu_per_job", type=int, default=2)
    p.add_argument("--gpu_per_job", type=float, default=1.0)
    p.add_argument("--win_threshold", type=float, default=0.5)
    p.add_argument("--ensemble_method", type=str, default="router", choices=["average", "learned_loss", "router"], help="average=simple mean; learned_loss=OOF-loss task/class weights; router=phase-specific learned-loss weights")
    p.add_argument("--ensemble_temperature", type=float, default=0.15, help="softmax temperature for loss-based ensemble weights; larger = closer to average")
    p.add_argument("--ensemble_class_shrink", type=float, default=80.0, help="shrink rare-class loss estimates toward global model loss")
    p.add_argument("--no_ensemble_classwise", action="store_true", help="disable class-wise weights for multiclass tasks")
    p.add_argument("--use_rule_override", action="store_true", help="apply conservative zero-probability context override from teammate pipeline")
    p.add_argument("--rule_min_context", type=int, default=30, help="minimum train samples for rule override context")
    args = p.parse_args()
    if args.no_resume:
        args.resume = False
    if args.no_dirichlet:
        args.use_dirichlet = False
    if args.no_hmm:
        args.use_hmm = False
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    print("VERSION_TAG:", VERSION_TAG)
    print("args:", vars(args))

    train_feat_path, test_feat_path, train_seq_path, test_seq_path, fold_group_values, group_col = build_all_features(args)
    cpu_models = [m.strip().lower() for m in args.cpu_models.split(",") if m.strip() and m.strip().lower() != "none"]
    gpu_models = [m.strip().lower() for m in args.gpu_models.split(",") if m.strip() and m.strip().lower() != "none"]

    jobs = []
    for model_name in cpu_models:
        for fold in range(args.folds):
            jobs.append({
                "device_type": "cpu",
                "model_name": model_name,
                "fold": fold,
                "out_dir": args.out_dir,
                "train_feat_path": train_feat_path,
                "test_feat_path": test_feat_path,
                "seed": args.seed,
                "resume": args.resume,
            })
    for model_name in gpu_models:
        model_name = {"causal_transformer": "transformer", "gpt": "transformer", "mini_gpt": "transformer"}.get(model_name, model_name)
        if model_name not in {"gru", "tcn", "transformer"}:
            print(f"[WARN] unknown gpu model {model_name}; supported: gru,tcn,transformer")
            continue
        for fold in range(args.folds):
            jobs.append({
                "device_type": "gpu",
                "model_name": model_name,
                "fold": fold,
                "out_dir": args.out_dir,
                "train_seq_path": train_seq_path,
                "test_seq_path": test_seq_path,
                "fold_group_values": fold_group_values,
                "group_col": group_col,
                "seed": args.seed,
                "resume": args.resume,
                "max_seq_len": args.max_seq_len,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "hidden": args.hidden,
                "lr": args.lr,
            })

    results = run_jobs(jobs, args)
    result_paths = [r["path"] for r in results if "path" in r]
    ensemble_test_predictions(
        result_paths,
        args.out_dir,
        threshold=args.win_threshold,
        train_feat_path=train_feat_path,
        test_feat_path=test_feat_path,
        method=args.ensemble_method,
        temperature=args.ensemble_temperature,
        classwise=not args.no_ensemble_classwise,
        class_shrink=args.ensemble_class_shrink,
        use_rule_override=args.use_rule_override,
        train_path=args.train_path,
        test_path=args.test_path,
        rule_min_context=args.rule_min_context,
    )


if __name__ == "__main__":
    main()
