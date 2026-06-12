import os
import random
from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError(
        "找不到 xgboost。請先安裝：pip install xgboost"
    ) from e

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight


NUM_POINT_CLASSES = 10
SEED = 42
VERSION_TAG = "XGBPROB_V1_2_LABELKEEP_BUSINESS_MATCHSPLIT_20260514"

                        
USE_ID_FEATURES = True
STATE_ONLY_MODE = False
USE_TRANSITION_PROB_FEATURES = True
USE_MATCH_LEVEL_SPLIT = True

ID_COLS = ["numberGame", "gamePlayerId", "gamePlayerOtherId"]

BASE_CAT_COLS = [
    "sex",
    "numberGame_idx",
    "gamePlayerId_idx",
    "gamePlayerOtherId_idx",
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "sid_spin",
    "sid_side",
    "positionId",
    "actionId",
    "aid_group",
    "aid_sub",
    "pointId",
    "pid_depth",
    "pid_side",
]

BASE_NUM_COLS = [
    "strikeNumber",
    "scoreSelf",
    "scoreOther",
    "scoreDiff",
]

                            
EXCLUDED_MODEL_FEATURES = ["rally_uid", "rally_id", "match"]

                      
LAST_ROW_CAT_COLS = BASE_CAT_COLS
LAST_ROW_NUM_COLS = BASE_NUM_COLS

                                      
TRANSITION_SPECS = [
    ("prev_point", ["last_pointId"]),
    ("prev_action", ["last_actionId"]),
    ("prev_aid_group", ["last_aid_group"]),
    ("prev_pid_side", ["last_pid_side"]),
    ("prev_pid_depth", ["last_pid_depth"]),
    ("prev_point_aid_group", ["last_pointId", "last_aid_group"]),
    ("prev_side_aid_group", ["last_pid_side", "last_aid_group"]),
    ("prev_depth_aid_group", ["last_pid_depth", "last_aid_group"]),
    ("prev_point_action", ["last_pointId", "last_actionId"]),
]

                                                
SECOND_ORDER_TRANSITION_SPECS = [
    ("prev2_last_point", ["prev2_pointId", "last_pointId"]),
    ("prev2_last_action", ["prev2_actionId", "last_actionId"]),
    ("prev2_last_aid_group", ["prev2_aid_group", "last_aid_group"]),
    ("prev2_last_side", ["prev2_pid_side", "last_pid_side"]),
    ("prev2_last_depth", ["prev2_pid_depth", "last_pid_depth"]),
    ("prev2_point_last_group", ["prev2_pointId", "last_pointId", "last_aid_group"]),
    ("prev2_side_last_group", ["prev2_pid_side", "last_pid_side", "last_aid_group"]),
]

TRANSITION_SPECS = TRANSITION_SPECS + SECOND_ORDER_TRANSITION_SPECS

                                                                  
if not USE_TRANSITION_PROB_FEATURES:
    TRANSITION_SPECS = []


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)


def prediction_distribution(name, values):
    values = np.asarray(values)
    print(f"{name} distribution:")
    print(pd.Series(values).value_counts(normalize=True).sort_index())


def predict_with_class_bias(proba, bias, eps=1e-12):


    proba = np.asarray(proba, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    scores = np.log(np.clip(proba, eps, 1.0)) + bias.reshape(1, -1)
    return np.argmax(scores, axis=1).astype(int)


def label_distribution_array(values, num_classes=NUM_POINT_CLASSES):

    values = np.asarray(values, dtype=int)
    counts = np.bincount(values, minlength=num_classes).astype(np.float64)[:num_classes]
    return counts / max(counts.sum(), 1.0)


def kl_divergence(p, q, eps=1e-12):

    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def evaluate_bias_score(y_true, proba, bias, num_classes=NUM_POINT_CLASSES, lambda_l2=0.02, mu_kl=0.10):


    pred = predict_with_class_bias(proba, bias)
    macro_f1 = f1_score(y_true, pred, labels=list(range(num_classes)), average="macro", zero_division=0)
    bias_l2 = float(np.sum(np.asarray(bias, dtype=np.float64) ** 2))
    pred_dist = label_distribution_array(pred, num_classes=num_classes)
    true_dist = label_distribution_array(y_true, num_classes=num_classes)
    kl = kl_divergence(pred_dist, true_dist)
    score = macro_f1 - lambda_l2 * bias_l2 - mu_kl * kl
    return {
        "score": float(score),
        "macro_f1": float(macro_f1),
        "bias_l2": bias_l2,
        "kl": float(kl),
        "pred": pred,
        "pred_dist": pred_dist,
        "bias": np.asarray(bias, dtype=np.float32),
    }


def conservative_bias_grid_search(y_true, proba, num_classes=NUM_POINT_CLASSES, lambda_l2=0.02, mu_kl=0.10):


    from itertools import product

    bias_grid = {
        0: [-0.10, -0.05, 0.00],
        2: [-0.10, -0.05, 0.00],
        7: [0.00, 0.05, 0.10, 0.15],
        8: [0.00, 0.05, 0.10, 0.15],
        9: [-0.10, -0.05, 0.00],
    }

    classes = list(bias_grid.keys())
    values = [bias_grid[c] for c in classes]
    best_result = None
    all_results = []

    for combo in product(*values):
        bias = np.zeros(num_classes, dtype=np.float32)
        for c, v in zip(classes, combo):
            bias[c] = v

        result = evaluate_bias_score(
            y_true=y_true,
            proba=proba,
            bias=bias,
            num_classes=num_classes,
            lambda_l2=lambda_l2,
            mu_kl=mu_kl,
        )
        all_results.append(result)
        if best_result is None or result["score"] > best_result["score"]:
            best_result = result

    return best_result, all_results


def apply_class_bias_to_prediction_df(pred_df, class_bias, num_classes=NUM_POINT_CLASSES):

    out = pred_df.copy()
    proba_cols = [f"prob_point_{c}" for c in range(num_classes)]
    proba = out[proba_cols].to_numpy(dtype=np.float64)
    out["pointId"] = predict_with_class_bias(proba, class_bias).astype(int)
    return out


def find_input_files():
    train_csv = "data/train.csv"
    test_csv = "data/test.csv"

    if not os.path.exists(train_csv):
        train_csv = "train.csv"
    if not os.path.exists(test_csv):
        test_csv = "test.csv"
    if not os.path.exists(test_csv) and os.path.exists("test_new.csv"):
        test_csv = "test_new.csv"

    return train_csv, test_csv


def point_transform(df, is_train=True):
    df = df.copy()
    if "pointId" not in df.columns:
        df["pointId"] = 0

    table = np.array([
        [0, 0],
        [1, 1], [1, 2], [1, 3],
        [2, 1], [2, 2], [2, 3],
        [3, 1], [3, 2], [3, 3],
    ], dtype=np.int64)

    pid = df["pointId"].fillna(0).astype(np.int64).to_numpy()
    if pid.min() < 0 or pid.max() > 9:
        raise ValueError("pointId out of range 0-9")

    ptuple = table[pid]
    df["pid_depth"] = ptuple[:, 0]
    df["pid_side"] = ptuple[:, 1]
    return df


def action_transform(df):
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

    aid = df["actionId"].fillna(0).astype(np.int64).to_numpy()
    if aid.min() < 0 or aid.max() > 18:
        raise ValueError("actionId out of range 0-18")

    atuple = table[aid]
    df["aid_group"] = atuple[:, 0]
    df["aid_sub"] = atuple[:, 1]
    return df


def spin_transform(df):
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

    sid = df["spinId"].fillna(0).astype(np.int64).to_numpy()
    if sid.min() < 0 or sid.max() > 5:
        raise ValueError("spinId out of range 0-5")

    stuple = table[sid]
    df["sid_spin"] = stuple[:, 0]
    df["sid_side"] = stuple[:, 1]
    return df


def add_numeric_features(df):
    df = df.copy()
    for col in ["scoreSelf", "scoreOther", "strikeNumber"]:
        if col not in df.columns:
            df[col] = 0
    df["scoreDiff"] = df["scoreSelf"] - df["scoreOther"]
    return df


def build_id_maps(df):
    id_maps = {}
    for col in ID_COLS:
        if col in df.columns:
            values = sorted(df[col].dropna().unique())
            id_maps[col] = {v: i + 1 for i, v in enumerate(values)}
        else:
            id_maps[col] = {}
    return id_maps


def apply_id_maps(df, id_maps):
    df = df.copy()
    for col in ID_COLS:
                                                                                          
                                                                                        
        if not USE_ID_FEATURES:
            df[f"{col}_idx"] = 0
            continue

        mapping = id_maps.get(col, {})
        if col in df.columns:
            df[f"{col}_idx"] = df[col].map(mapping).fillna(0).astype(np.int64)
        else:
            df[f"{col}_idx"] = 0
    return df


def preprocess_df(df, id_maps=None, is_train=True):
    df = df.copy()
    if is_train:
        id_maps = build_id_maps(df)

    df = point_transform(df, is_train=is_train)
    df = action_transform(df)
    df = spin_transform(df)
    df = add_numeric_features(df)
    df = apply_id_maps(df, id_maps)
    return df, id_maps


def ensure_feature_columns(df):
    df = df.copy()
    for col in BASE_CAT_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(np.int64)
    for col in BASE_NUM_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0).astype(np.float32)
    return df


def safe_change_rate(values):
    values = np.asarray(values)
    if len(values) <= 1:
        return 0.0
    return float(np.mean(values[1:] != values[:-1]))


def safe_streak_len(values):
    values = list(values)
    if len(values) == 0:
        return 0
    last = values[-1]
    streak = 1
    for v in reversed(values[:-1]):
        if v == last:
            streak += 1
        else:
            break
    return streak


def add_ratio_features(out, prefix, col, prefix_name, classes, windows):
    arr_all = prefix[col].to_numpy(np.int64)
    for w_name, w in windows:
        arr = arr_all if w is None else arr_all[-w:]
        denom = max(len(arr), 1)
        counts = np.bincount(arr, minlength=classes).astype(np.float32)[:classes]
        ratio = counts / denom
        for c in range(classes):
            out[f"{w_name}_{prefix_name}_ratio_{c}"] = float(ratio[c])


def add_change_features(out, prefix, col, prefix_name, windows):
    arr_all = prefix[col].to_numpy(np.int64)
    for w_name, w in windows:
        arr = arr_all if w is None else arr_all[-w:]
        out[f"{w_name}_{prefix_name}_change_rate"] = safe_change_rate(arr)
        out[f"{w_name}_{prefix_name}_streak_len"] = safe_streak_len(arr)


def safe_get_from_end(prefix, col, n_from_end, default=-1):


    if len(prefix) < n_from_end or col not in prefix.columns:
        return default
    val = prefix.iloc[-n_from_end][col]
    if pd.isna(val):
        return default
    return int(val)


def add_prev2_basic_features(out, prefix):

    prev2_cols = [
        "pointId",
        "actionId",
        "aid_group",
        "aid_sub",
        "pid_side",
        "pid_depth",
        "sid_spin",
        "sid_side",
        "positionId",
        "spinId",
        "strengthId",
    ]
    for col in prev2_cols:
        out[f"prev2_{col}"] = safe_get_from_end(prefix, col, n_from_end=2, default=-1)
    return out


def add_prev2_change_features(out, prefix):

    pairs = [
        ("pointId", "point_changed_last2"),
        ("actionId", "action_changed_last2"),
        ("aid_group", "aid_group_changed_last2"),
        ("pid_side", "side_changed_last2"),
        ("pid_depth", "depth_changed_last2"),
        ("sid_spin", "serve_spin_changed_last2"),
        ("positionId", "position_changed_last2"),
    ]
    for col, new_col in pairs:
        prev2_val = safe_get_from_end(prefix, col, n_from_end=2, default=-1)
        last_val = safe_get_from_end(prefix, col, n_from_end=1, default=-1)
        out[new_col] = 0 if prev2_val == -1 or last_val == -1 else int(prev2_val != last_val)
    return out


def encode_pair(a, b, base_b, default=-1):

    if a == default or b == default:
        return default
    return int(a * base_b + b)


def add_prev2_pair_features(out, prefix):

    prev2_point = safe_get_from_end(prefix, "pointId", 2, default=-1)
    last_point = safe_get_from_end(prefix, "pointId", 1, default=-1)

    prev2_action = safe_get_from_end(prefix, "actionId", 2, default=-1)
    last_action = safe_get_from_end(prefix, "actionId", 1, default=-1)

    prev2_group = safe_get_from_end(prefix, "aid_group", 2, default=-1)
    last_group = safe_get_from_end(prefix, "aid_group", 1, default=-1)

    prev2_side = safe_get_from_end(prefix, "pid_side", 2, default=-1)
    last_side = safe_get_from_end(prefix, "pid_side", 1, default=-1)

    prev2_depth = safe_get_from_end(prefix, "pid_depth", 2, default=-1)
    last_depth = safe_get_from_end(prefix, "pid_depth", 1, default=-1)

    out["pair_prev2_last_point"] = encode_pair(prev2_point, last_point, base_b=10)
    out["pair_prev2_last_action"] = encode_pair(prev2_action, last_action, base_b=19)
    out["pair_prev2_last_aid_group"] = encode_pair(prev2_group, last_group, base_b=5)
    out["pair_prev2_last_side"] = encode_pair(prev2_side, last_side, base_b=4)
    out["pair_prev2_last_depth"] = encode_pair(prev2_depth, last_depth, base_b=4)
    return out


def add_prev2_pattern_features(out, prefix):


    out = add_prev2_basic_features(out, prefix)
    out = add_prev2_change_features(out, prefix)
    out = add_prev2_pair_features(out, prefix)
    return out


def get_rally_phase(strike_number):


    try:
        strike_number = int(strike_number)
    except Exception:
        strike_number = 0
    if strike_number <= 1:
        return 1
    if strike_number == 2:
        return 2
    if strike_number == 3:
        return 3
    return 4


def _last_array(prefix, col, window=None):
    if col not in prefix.columns or len(prefix) == 0:
        return np.asarray([], dtype=np.int64)
    arr = prefix[col].fillna(0).astype(np.int64).to_numpy()
    if window is not None:
        arr = arr[-window:]
    return arr


def _unique_count(prefix, col, window=3):
    arr = _last_array(prefix, col, window=window)
    if len(arr) == 0:
        return 0
    return int(len(np.unique(arr)))


def _alternating_last3(prefix, col):

    arr = _last_array(prefix, col, window=3)
    if len(arr) < 3:
        return 0
    return int(arr[0] == arr[2] and arr[0] != arr[1])


def add_business_logic_features(out, prefix):


    last = prefix.iloc[-1]

    score_self = float(last.get("scoreSelf", 0.0))
    score_other = float(last.get("scoreOther", 0.0))
    score_diff = score_self - score_other
    score_sum = score_self + score_other
    abs_diff = abs(score_diff)
    strike_number = int(last.get("strikeNumber", len(prefix)))
    phase = get_rally_phase(strike_number)

                                   
    out["scoreDiff_bl"] = score_diff
    out["scoreSum_bl"] = score_sum
    out["absScoreDiff_bl"] = abs_diff
    out["is_tie_score_bl"] = int(score_self == score_other)
    out["is_leading_bl"] = int(score_self > score_other)
    out["is_trailing_bl"] = int(score_self < score_other)
    out["is_close_score_bl"] = int(abs_diff <= 2)
    out["is_big_lead_bl"] = int(score_diff >= 4)
    out["is_big_trail_bl"] = int(score_diff <= -4)
    out["is_late_game_bl"] = int(score_sum >= 16)
    out["is_deuce_like_bl"] = int(score_self >= 9 and score_other >= 9)
    out["is_game_point_like_bl"] = int(max(score_self, score_other) >= 10)

                                                 
    out["rally_phase_bl"] = phase
    out["strike_parity_bl"] = strike_number % 2
    out["strike_mod3_bl"] = strike_number % 3
    out["is_serve_phase_bl"] = int(phase == 1)
    out["is_receive_phase_bl"] = int(phase == 2)
    out["is_third_ball_phase_bl"] = int(phase == 3)
    out["is_rally_phase_bl"] = int(phase == 4)

                                                                                          
    last_action = int(last.get("actionId", 0))
    last_group = int(last.get("aid_group", 0))
    last_spin = int(last.get("spinId", 0))
    last_sid_spin = int(last.get("sid_spin", 0))
    last_strength = int(last.get("strengthId", 0))
    last_hand = int(last.get("handId", 0))
    last_position = int(last.get("positionId", 0))
    last_side = int(last.get("pid_side", 0))
    last_depth = int(last.get("pid_depth", 0))

    out["action_spin_bl"] = encode_pair(last_action, last_spin, base_b=10, default=-1)
    out["action_sid_spin_bl"] = encode_pair(last_action, last_sid_spin, base_b=10, default=-1)
    out["action_strength_bl"] = encode_pair(last_action, last_strength, base_b=10, default=-1)
    out["hand_action_bl"] = encode_pair(last_hand, last_action, base_b=25, default=-1)
    out["position_action_bl"] = encode_pair(last_position, last_action, base_b=25, default=-1)
    out["action_side_bl"] = encode_pair(last_action, last_side, base_b=10, default=-1)
    out["action_depth_bl"] = encode_pair(last_action, last_depth, base_b=10, default=-1)
    out["group_strength_bl"] = encode_pair(last_group, last_strength, base_b=10, default=-1)

                                                                                             
    out["point_changed_prev_bl"] = int(safe_change_rate(_last_array(prefix, "pointId", window=2)) > 0)
    out["position_changed_prev_bl"] = int(safe_change_rate(_last_array(prefix, "positionId", window=2)) > 0)
    out["action_changed_prev_bl"] = int(safe_change_rate(_last_array(prefix, "actionId", window=2)) > 0)
    out["side_changed_prev_bl"] = int(safe_change_rate(_last_array(prefix, "pid_side", window=2)) > 0)
    out["depth_changed_prev_bl"] = int(safe_change_rate(_last_array(prefix, "pid_depth", window=2)) > 0)

    out["side_unique_last3_bl"] = _unique_count(prefix, "pid_side", window=3)
    out["depth_unique_last3_bl"] = _unique_count(prefix, "pid_depth", window=3)
    out["point_unique_last3_bl"] = _unique_count(prefix, "pointId", window=3)
    out["action_unique_last3_bl"] = _unique_count(prefix, "actionId", window=3)
    out["side_alternating_last3_bl"] = _alternating_last3(prefix, "pid_side")
    out["depth_alternating_last3_bl"] = _alternating_last3(prefix, "pid_depth")
    out["point_alternating_last3_bl"] = _alternating_last3(prefix, "pointId")

    roll3_point_change = safe_change_rate(_last_array(prefix, "pointId", window=3))
    roll5_point_change = safe_change_rate(_last_array(prefix, "pointId", window=5))
    roll5_position_change = safe_change_rate(_last_array(prefix, "positionId", window=5))
    roll5_action_change = safe_change_rate(_last_array(prefix, "actionId", window=5))
    roll5_side_change = safe_change_rate(_last_array(prefix, "pid_side", window=5))
    roll5_depth_change = safe_change_rate(_last_array(prefix, "pid_depth", window=5))

    placement_volatility = float(np.mean([
        roll5_point_change,
        roll5_position_change,
        roll5_side_change,
        roll5_depth_change,
    ]))
    out["roll3_point_change_rate_bl"] = roll3_point_change
    out["roll5_point_change_rate_bl"] = roll5_point_change
    out["roll5_position_change_rate_bl"] = roll5_position_change
    out["roll5_action_change_rate_bl"] = roll5_action_change
    out["roll5_side_change_rate_bl"] = roll5_side_change
    out["roll5_depth_change_rate_bl"] = roll5_depth_change
    out["placement_volatility_bl"] = placement_volatility
    out["is_stable_placement_bl"] = int(placement_volatility <= 0.30)
    out["is_chaotic_placement_bl"] = int(placement_volatility >= 0.70)

    return out


def build_single_prefix_features(prefix, rally_uid=None):

    out = {}
    if rally_uid is not None:
        out["rally_uid"] = rally_uid

    last = prefix.iloc[-1]

                                              
    for col in LAST_ROW_CAT_COLS:
        out[f"last_{col}"] = int(last[col])
    for col in LAST_ROW_NUM_COLS:
        out[f"last_{col}"] = float(last[col])

                                                                               
    out = add_prev2_pattern_features(out, prefix)

                                                           
    out = add_business_logic_features(out, prefix)

    out["prefix_len"] = int(len(prefix))

    windows = [("roll3", 3), ("roll5", 5), ("roll_all", None)]

                            
    add_ratio_features(out, prefix, "pointId", "point", 10, windows)
    add_ratio_features(out, prefix, "aid_group", "aid_group", 5, windows)
    add_ratio_features(out, prefix, "actionId", "action", 19, [("roll3", 3), ("roll5", 5)])
    add_ratio_features(out, prefix, "pid_side", "side", 4, windows)
    add_ratio_features(out, prefix, "pid_depth", "depth", 4, windows)
    add_ratio_features(out, prefix, "sid_spin", "spin", 4, [("roll3", 3), ("roll5", 5)])

                               
    add_change_features(out, prefix, "pid_side", "side", windows)
    add_change_features(out, prefix, "pid_depth", "depth", windows)
    add_change_features(out, prefix, "aid_group", "aid_group", windows)
    add_change_features(out, prefix, "pointId", "point", windows)

                                                       
    out["abs_score_diff"] = abs(out.get("last_scoreDiff", 0.0))
    out["score_sum"] = out.get("last_scoreSelf", 0.0) + out.get("last_scoreOther", 0.0)
    out["is_early_rally"] = 1 if len(prefix) <= 3 else 0
    out["is_late_rally"] = 1 if len(prefix) >= 8 else 0

    return out


def make_sliding_feature_df(df, max_seq_len=15, is_train=True, desc="make sliding features"):


    df = ensure_feature_columns(df)
    df = df.copy()
    df["row_order"] = np.arange(len(df))
    df = df.sort_values(["rally_uid", "strikeNumber", "row_order"]).reset_index(drop=True)

    rows = []
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
                feat["target_point"] = int(target["pointId"])
                rows.append(feat)
        else:
            prefix = g
            if max_seq_len is not None:
                prefix = prefix.iloc[-max_seq_len:]
            feat = build_single_prefix_features(prefix, rally_uid=rally_uid)
            rows.append(feat)

    return pd.DataFrame(rows)


class TransitionProbabilityEncoder:
    def __init__(self, specs, target_col="target_point", num_classes=10, alpha=1.0):
        self.specs = specs
        self.target_col = target_col
        self.num_classes = num_classes
        self.alpha = alpha
        self.global_prob = None
        self.tables = {}

    def fit(self, feat_df):
        y = feat_df[self.target_col].astype(int).to_numpy()
        counts = np.bincount(y, minlength=self.num_classes).astype(np.float64)[:self.num_classes]
        self.global_prob = (counts + self.alpha) / (counts.sum() + self.alpha * self.num_classes)

        for name, cols in self.specs:
            use_cols = list(cols)
            tmp = feat_df[use_cols + [self.target_col]].copy()
            tmp[self.target_col] = tmp[self.target_col].astype(int)

            count_df = (
                tmp.groupby(use_cols + [self.target_col])
                .size()
                .unstack(self.target_col, fill_value=0)
            )
            count_df = count_df.reindex(columns=list(range(self.num_classes)), fill_value=0).astype(np.float64)

            prob_df = (count_df + self.alpha)
            prob_df = prob_df.div(prob_df.sum(axis=1), axis=0)
            prob_df.columns = [f"trans_{name}_to_{c}" for c in range(self.num_classes)]
            self.tables[name] = (use_cols, prob_df.reset_index())

        return self

    def transform(self, feat_df):
        out = feat_df.copy()
        for name, (cols, prob_df) in self.tables.items():
            proba_cols = [f"trans_{name}_to_{c}" for c in range(self.num_classes)]
            out = out.merge(prob_df, how="left", on=cols)
            for c, col in enumerate(proba_cols):
                if col not in out.columns:
                    out[col] = self.global_prob[c]
                else:
                    out[col] = out[col].fillna(self.global_prob[c]).astype(np.float32)
        return out

    def fit_transform(self, feat_df):
        self.fit(feat_df)
        return self.transform(feat_df)


def business_feature_diagnostics(feat_train, feat_valid=None, y_train=None, fold=None, save_dir="diagnostic_plots_xgbprob"):


    business_cols = [c for c in feat_train.columns if c.endswith("_bl")]
    if not business_cols:
        print("[Business features] no *_bl columns found")
        return

    Xb_train = feat_train[business_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    Xb_valid = None
    if feat_valid is not None:
        Xb_valid = feat_valid[business_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    name = f"Fold {fold} business features" if fold is not None else "Business features"
    feature_matrix_diagnostics(
        Xb_train,
        X_valid=Xb_valid,
        y_train=y_train,
        max_rows=5000,
        name=name,
    )

    summary_rows = []
    watch_cols = [
        "rally_phase_bl",
        "is_serve_phase_bl",
        "is_receive_phase_bl",
        "is_third_ball_phase_bl",
        "is_rally_phase_bl",
        "is_close_score_bl",
        "is_late_game_bl",
        "is_deuce_like_bl",
        "is_game_point_like_bl",
        "placement_volatility_bl",
        "is_stable_placement_bl",
        "is_chaotic_placement_bl",
    ]
    for col in watch_cols:
        if col in Xb_train.columns:
            row = {
                "feature": col,
                "train_mean": float(Xb_train[col].mean()),
                "train_std": float(Xb_train[col].std()),
            }
            if Xb_valid is not None and col in Xb_valid.columns:
                row["valid_mean"] = float(Xb_valid[col].mean())
                row["valid_std"] = float(Xb_valid[col].std())
                row["abs_mean_shift"] = abs(row["valid_mean"] - row["train_mean"])
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print(f"[{name}] business feature summary:")
    if len(summary_df) > 0:
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("  no watched business columns found")

                                                                                 
    if "rally_phase_bl" in Xb_train.columns:
        print(f"[{name}] train rally_phase distribution:")
        print(Xb_train["rally_phase_bl"].value_counts(normalize=True).sort_index().to_string(float_format=lambda x: f"{x:.4f}"))
        if Xb_valid is not None and "rally_phase_bl" in Xb_valid.columns:
            print(f"[{name}] valid rally_phase distribution:")
            print(Xb_valid["rally_phase_bl"].value_counts(normalize=True).sort_index().to_string(float_format=lambda x: f"{x:.4f}"))

                                                                  
    try:
        os.makedirs(save_dir, exist_ok=True)
        suffix = f"fold{fold}" if fold is not None else "all"
        summary_df.to_csv(os.path.join(save_dir, f"xgbprob_business_summary_{suffix}.csv"), index=False)
    except Exception as e:
        print(f"[Business features] failed to save summary: {e}")


def is_state_only_excluded_feature(col):


    if not STATE_ONLY_MODE:
        return False

                                                             
    if col.endswith("_bl"):
        return False

    keep_exact = {
        "last_strikeNumber",
        "last_scoreSelf",
        "last_scoreOther",
        "last_scoreDiff",
        "prefix_len",
        "abs_score_diff",
        "score_sum",
        "is_early_rally",
        "is_late_rally",
    }
    if col in keep_exact:
        return False

                                    
    if col.endswith("_idx") or "numberGame" in col or "gamePlayer" in col:
        return True

                                                                                         
    if col.startswith("last_"):
        return True

                                                                      
    if col.startswith("prev2_") or col.startswith("pair_prev2_"):
        return True

                                                                      
    if "_ratio_" in col:
        return True

                                                                                  
    if col.startswith("trans_"):
        return True

    return False


def get_model_feature_columns(df):
    drop_cols = set(["target_point", "rally_uid"] + EXCLUDED_MODEL_FEATURES)
    feat_cols = [c for c in df.columns if c not in drop_cols]
                                                                  
    feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
    if STATE_ONLY_MODE:
        before = len(feat_cols)
        feat_cols = [c for c in feat_cols if not is_state_only_excluded_feature(c)]
        print(f"STATE_ONLY_MODE feature filter: {before} -> {len(feat_cols)} features")
        raw_like_left = [
            c for c in feat_cols
            if c.startswith(("last_", "prev2_", "pair_prev2_", "trans_")) or "_ratio_" in c or c.endswith("_idx")
        ]
        if raw_like_left:
            print("[Warning] raw-like features still included:", raw_like_left[:30])
    return feat_cols


def align_features(train_df, valid_df, feature_cols):
    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    X_valid = valid_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return X_train, X_valid


def _safe_prob(proba, eps=1e-12):
    proba = np.asarray(proba, dtype=np.float64)
    proba = np.clip(proba, eps, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return proba


def probability_diagnostics(name, proba, y_true=None, num_classes=NUM_POINT_CLASSES):


    proba = _safe_prob(proba)
    entropy = -np.sum(proba * np.log(proba), axis=1)
    norm_entropy = entropy / np.log(num_classes)
    sorted_p = np.sort(proba, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]

    print(f"[{name}] probability diagnostics:")
    print(
        f"  entropy_norm mean={norm_entropy.mean():.4f}, "
        f"p10={np.percentile(norm_entropy, 10):.4f}, p90={np.percentile(norm_entropy, 90):.4f}"
    )
    print(
        f"  top1_top2_margin mean={margin.mean():.4f}, "
        f"p10={np.percentile(margin, 10):.4f}, p90={np.percentile(margin, 90):.4f}"
    )

    if y_true is None:
        return

    y_true = np.asarray(y_true, dtype=np.int64)
    ic_rows = []
    for c in range(num_classes):
        indicator = (y_true == c).astype(np.float64)
        support = int(indicator.sum())
        if support == 0 or support == len(indicator):
            spearman_ic = np.nan
            pearson_ic = np.nan
        else:
            spearman_ic = pd.Series(proba[:, c]).corr(pd.Series(indicator), method="spearman")
            pearson_ic = pd.Series(proba[:, c]).corr(pd.Series(indicator), method="pearson")
        ic_rows.append({
            "class": c,
            "support": support,
            "prob_spearman_ic": spearman_ic,
            "prob_pearson_ic": pearson_ic,
            "mean_prob_when_true": float(proba[y_true == c, c].mean()) if support > 0 else np.nan,
            "mean_prob_all": float(proba[:, c].mean()),
        })

    ic_df = pd.DataFrame(ic_rows)
    print(f"[{name}] class-wise probability IC:")
    print(ic_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        f"[{name}] mean abs Spearman IC="
        f"{np.nanmean(np.abs(ic_df['prob_spearman_ic'].to_numpy())):.4f}"
    )


def feature_matrix_diagnostics(X_train, X_valid=None, y_train=None, max_rows=5000, name="feature matrix"):


    X = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)
    if len(X) > max_rows:
        Xs = X.sample(max_rows, random_state=SEED)
    else:
        Xs = X

                                                        
    Xv = Xs.to_numpy(dtype=np.float64)
    mu = Xv.mean(axis=0)
    sd = Xv.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (Xv - mu) / sd

    constant_cols = int(np.sum(Xv.std(axis=0) < 1e-12))
    try:
        singular_values = np.linalg.svd(Z, compute_uv=False)
        if singular_values.size == 0 or singular_values[0] <= 1e-12:
            effective_rank = 0.0
            condition_number = np.inf
            top10_energy = 0.0
        else:
            energy = singular_values ** 2
            p = energy / energy.sum()
            effective_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
            condition_number = float(singular_values[0] / max(singular_values[-1], 1e-12))
            top10_energy = float(energy[:10].sum() / energy.sum())
    except np.linalg.LinAlgError:
        effective_rank = np.nan
        condition_number = np.nan
        top10_energy = np.nan

    print(f"[{name}] linear algebra diagnostics:")
    print(f"  rows_used={len(Xs)}, n_features={X.shape[1]}, constant_cols={constant_cols}")
    print(f"  effective_rank≈{effective_rank:.2f}, condition_number≈{condition_number:.2e}, top10_energy={top10_energy:.4f}")

    if X_valid is not None:
        train_mean = X_train.mean(axis=0)
        valid_mean = X_valid.mean(axis=0)
        train_std = X_train.std(axis=0).replace(0, 1.0)
        mean_shift = ((valid_mean - train_mean).abs() / train_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        top_shift = mean_shift.sort_values(ascending=False).head(15)
        print(f"[{name}] top train-valid standardized mean shifts:")
        print(top_shift.to_string(float_format=lambda x: f"{x:.4f}"))

    if y_train is not None:
                                                                       
                                                                             
        y = np.asarray(y_train, dtype=np.int64)
        ic_summary = []
                                                           
        for c in range(NUM_POINT_CLASSES):
            indicator = pd.Series((y == c).astype(float))
            vals = []
            for col in X_train.columns:
                corr = pd.Series(X_train[col].to_numpy()).corr(indicator, method="spearman")
                if pd.notna(corr):
                    vals.append(abs(float(corr)))
            ic_summary.append({
                "class": c,
                "support": int((y == c).sum()),
                "mean_abs_feature_ic": float(np.mean(vals)) if vals else np.nan,
                "max_abs_feature_ic": float(np.max(vals)) if vals else np.nan,
            })
        print(f"[{name}] feature→target Spearman IC summary:")
        print(pd.DataFrame(ic_summary).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

def xgb_params(seed=SEED):
    return {
        "objective": "multi:softprob",
        "num_class": NUM_POINT_CLASSES,
        "eval_metric": "mlogloss",
        "max_depth": 4,
        "eta": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "lambda": 1.0,
        "alpha": 0.0,
        "tree_method": "hist",
        "seed": seed,
        "verbosity": 1,
    }


def compute_smoothed_sample_weight(y, power=0.35):


    base_weight = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)
    weight = np.power(base_weight, power)
    weight = weight / np.mean(weight)
    return weight.astype(np.float32)


def train_xgb_fold(X_train, y_train, X_valid, y_valid, fold):
    sample_weight = compute_smoothed_sample_weight(y_train, power=0.35)
    print(
        f"sample weight smoothing power=0.35 | "
        f"min={sample_weight.min():.4f}, mean={sample_weight.mean():.4f}, max={sample_weight.max():.4f}"
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight, feature_names=list(X_train.columns))
    dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=list(X_valid.columns))

    params = xgb_params(seed=SEED + fold)
    watchlist = [(dtrain, "train"), (dvalid, "valid")]

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1200,
        evals=watchlist,
        early_stopping_rounds=80,
        verbose_eval=50,
    )
    return model


def predict_proba(model, X):


    dmat = xgb.DMatrix(X, feature_names=list(X.columns))

    try:
        best_iter = model.best_iteration
    except AttributeError:
        best_iter = None

    if best_iter is None:
        return model.predict(dmat)

    return model.predict(dmat, iteration_range=(0, best_iter + 1))

def evaluate_fold(model, X_valid, y_valid, fold):
    proba = predict_proba(model, X_valid)
    pred = proba.argmax(axis=1)
    f1 = f1_score(y_valid, pred, labels=list(range(NUM_POINT_CLASSES)), average="macro", zero_division=0)

    print(f"[XGBProb] Fold {fold} Point Macro F1: {f1:.4f}")
    print(classification_report(y_valid, pred, labels=list(range(NUM_POINT_CLASSES)), digits=4, zero_division=0))
    prediction_distribution(f"Fold {fold} valid true", y_valid)
    prediction_distribution(f"Fold {fold} valid pred", pred)
    probability_diagnostics(f"Fold {fold} valid", proba, y_valid)
    return f1, pred, proba


def choose_cv_group_column(train_df):
    if USE_MATCH_LEVEL_SPLIT and "match" in train_df.columns:
        return "match"
    if USE_MATCH_LEVEL_SPLIT and "numberGame" in train_df.columns:
        return "numberGame"
    return "rally_uid"


def train_kfold_xgbprob(train_df, test_df=None, n_splits=5, max_seq_len=15):
    group_col = choose_cv_group_column(train_df)
    group_ids = train_df[group_col].drop_duplicates().to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    models = []
    encoders = []
    feature_cols_per_fold = []
    scores = []

    oof_rows = []
    oof_true = []
    oof_pred = []

    for fold, (tr_i, va_i) in enumerate(gkf.split(group_ids, groups=group_ids), start=1):
        tr_groups = set(group_ids[tr_i])
        va_groups = set(group_ids[va_i])

        df_tr = train_df[train_df[group_col].isin(tr_groups)].copy()
        df_va = train_df[train_df[group_col].isin(va_groups)].copy()

        tr_rallies = set(df_tr["rally_uid"].drop_duplicates())
        va_rallies = set(df_va["rally_uid"].drop_duplicates())
        rally_overlap = len(tr_rallies & va_rallies)
        if "match" in train_df.columns:
            match_overlap = len(set(df_tr["match"].drop_duplicates()) & set(df_va["match"].drop_duplicates()))
        else:
            match_overlap = None

        print("=" * 80)
        print(f"Fold {fold}/{n_splits}")
        print(f"CV group column: {group_col}")
        print(f"train groups: {len(tr_groups)}, valid groups: {len(va_groups)}")
        print(f"train rallies: {len(tr_rallies)}, valid rallies: {len(va_rallies)}")
        print(f"rally_uid overlap: {rally_overlap}")
        if match_overlap is not None:
            print(f"match overlap: {match_overlap}")

        feat_tr_base = make_sliding_feature_df(
            df_tr,
            max_seq_len=max_seq_len,
            is_train=True,
            desc=f"Fold {fold} train sliding-prefix features",
        )
        feat_va_base = make_sliding_feature_df(
            df_va,
            max_seq_len=max_seq_len,
            is_train=True,
            desc=f"Fold {fold} valid sliding-prefix features",
        )
        print(f"train sliding-prefix samples: {len(feat_tr_base)}")
        print(f"valid sliding-prefix samples: {len(feat_va_base)}")

                                                                   
        encoder = TransitionProbabilityEncoder(
            specs=TRANSITION_SPECS,
            target_col="target_point",
            num_classes=NUM_POINT_CLASSES,
            alpha=1.0,
        )
        feat_tr = encoder.fit_transform(feat_tr_base)
        feat_va = encoder.transform(feat_va_base)

        feature_cols = get_model_feature_columns(feat_tr)
        X_train, X_valid = align_features(feat_tr, feat_va, feature_cols)
        y_train = feat_tr["target_point"].astype(int).to_numpy()
        y_valid = feat_va["target_point"].astype(int).to_numpy()

        print(f"num features: {len(feature_cols)}")
        print("target counts train:", Counter(y_train))
        print("target counts valid:", Counter(y_valid))
        if fold == 1:
            feature_matrix_diagnostics(
                X_train,
                X_valid=X_valid,
                y_train=y_train,
                max_rows=5000,
                name="Fold 1 engineered features",
            )

        business_feature_diagnostics(
            feat_tr,
            feat_valid=feat_va,
            y_train=y_train,
            fold=fold,
        )

        model = train_xgb_fold(X_train, y_train, X_valid, y_valid, fold)
        f1, pred, proba = evaluate_fold(model, X_valid, y_valid, fold)

        model_path = f"best_xgbprob_v1_2_labelkeep_matchsplit_fold{fold}.json"
        model.save_model(model_path)
        print(f"saved {model_path}")

                                                      
        fold_oof = pd.DataFrame({
            "rally_uid": feat_va["rally_uid"].values,
            "target_point": y_valid,
            "pred_point": pred,
        })
        for c in range(NUM_POINT_CLASSES):
            fold_oof[f"prob_point_{c}"] = proba[:, c]
        fold_oof["fold"] = fold
        oof_rows.append(fold_oof)

        oof_true.extend(y_valid.tolist())
        oof_pred.extend(pred.tolist())
        scores.append(f1)
        models.append(model)
        encoders.append(encoder)
        feature_cols_per_fold.append(feature_cols)

    print("=" * 80)
    print("XGBProb CV scores:", scores)
    print("XGBProb mean F1:", float(np.mean(scores)))
    print("XGBProb std  F1:", float(np.std(scores)))
    print("XGBProb OOF Macro F1:", f1_score(oof_true, oof_pred, labels=list(range(NUM_POINT_CLASSES)), average="macro", zero_division=0))
    prediction_distribution("OOF true", oof_true)
    prediction_distribution("OOF pred", oof_pred)

    if oof_rows:
        oof_df = pd.concat(oof_rows, ignore_index=True)
        oof_path = "xgbprob_v1_2_labelkeep_matchsplit_oof_point_probs.csv"
        oof_df.to_csv(oof_path, index=False)
        print(f"saved {oof_path}")
        oof_proba = oof_df[[f"prob_point_{c}" for c in range(NUM_POINT_CLASSES)]].to_numpy()
        probability_diagnostics("OOF", oof_proba, oof_df["target_point"].to_numpy())

        best_bias_result, _ = conservative_bias_grid_search(
            y_true=oof_df["target_point"].to_numpy(),
            proba=oof_proba,
            num_classes=NUM_POINT_CLASSES,
            lambda_l2=0.02,
            mu_kl=0.10,
        )
        best_bias = best_bias_result["bias"]
        np.save("xgbprob_v1_2_labelkeep_matchsplit_best_class_bias.npy", best_bias)
        pd.DataFrame({"class": np.arange(NUM_POINT_CLASSES), "bias": best_bias}).to_csv(
            "xgbprob_v1_2_labelkeep_matchsplit_best_class_bias.csv", index=False
        )
        print("Conservative class-wise bias search result:")
        print(f"  score={best_bias_result['score']:.6f}")
        print(f"  macro_f1={best_bias_result['macro_f1']:.6f}")
        print(f"  bias_l2={best_bias_result['bias_l2']:.6f}")
        print(f"  kl={best_bias_result['kl']:.6f}")
        print(f"  bias={best_bias.tolist()}")
        prediction_distribution("OOF pred with conservative bias", best_bias_result["pred"])
    else:
        best_bias = np.zeros(NUM_POINT_CLASSES, dtype=np.float32)

    return models, encoders, feature_cols_per_fold, scores, best_bias


def predict_test_xgbprob(models, encoders, feature_cols_per_fold, test_df, max_seq_len=15, class_bias=None):
    feat_test_base = make_sliding_feature_df(
        test_df,
        max_seq_len=max_seq_len,
        is_train=False,
        desc="test prefix features",
    )

    prob_sum = None
    for fold, (model, encoder, feature_cols) in enumerate(zip(models, encoders, feature_cols_per_fold), start=1):
        feat_test = encoder.transform(feat_test_base)
        X_test = feat_test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        proba = predict_proba(model, X_test)
        prob_sum = proba if prob_sum is None else prob_sum + proba
        print(f"Fold {fold} test proba done")

    avg_proba = prob_sum / len(models)
    probability_diagnostics("Test CV avg", avg_proba, y_true=None)
    if class_bias is None:
        pred = avg_proba.argmax(axis=1)
    else:
        pred = predict_with_class_bias(avg_proba, class_bias)

    pred_df = pd.DataFrame({
        "rally_uid": feat_test_base["rally_uid"].values,
        "pointId": pred.astype(int),
    })
    for c in range(NUM_POINT_CLASSES):
        pred_df[f"prob_point_{c}"] = avg_proba[:, c]

    pred_df = pred_df.sort_values("rally_uid").reset_index(drop=True)
    return pred_df


def train_full_and_predict(train_df, test_df, max_seq_len=15, class_bias=None):

    print("=" * 80)
    print("Training final full-data XGBProb model")

    feat_train_base = make_sliding_feature_df(
        train_df,
        max_seq_len=max_seq_len,
        is_train=True,
        desc="full train sliding-prefix features",
    )
    feat_test_base = make_sliding_feature_df(
        test_df,
        max_seq_len=max_seq_len,
        is_train=False,
        desc="test prefix features for full model",
    )

    encoder = TransitionProbabilityEncoder(
        specs=TRANSITION_SPECS,
        target_col="target_point",
        num_classes=NUM_POINT_CLASSES,
        alpha=1.0,
    )
    feat_train = encoder.fit_transform(feat_train_base)
    feat_test = encoder.transform(feat_test_base)

    feature_cols = get_model_feature_columns(feat_train)
    X_train = feat_train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    X_test = feat_test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    y_train = feat_train["target_point"].astype(int).to_numpy()

    sample_weight = compute_smoothed_sample_weight(y_train, power=0.35)
    print(
        f"full model sample weight smoothing power=0.35 | "
        f"min={sample_weight.min():.4f}, mean={sample_weight.mean():.4f}, max={sample_weight.max():.4f}"
    )
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight, feature_names=list(X_train.columns))

    params = xgb_params(seed=SEED + 999)
    final_model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=600,
        evals=[(dtrain, "train")],
        verbose_eval=50,
    )
    final_model.save_model("best_xgbprob_v1_2_labelkeep_matchsplit_full.json")
    print("saved best_xgbprob_v1_2_labelkeep_matchsplit_full.json")

    proba = predict_proba(final_model, X_test)
    probability_diagnostics("Test full model", proba, y_true=None)
    if class_bias is None:
        pred = proba.argmax(axis=1)
    else:
        pred = predict_with_class_bias(proba, class_bias)
    pred_df = pd.DataFrame({
        "rally_uid": feat_test_base["rally_uid"].values,
        "pointId": pred.astype(int),
    })
    for c in range(NUM_POINT_CLASSES):
        pred_df[f"prob_point_{c}"] = proba[:, c]
    pred_df = pred_df.sort_values("rally_uid").reset_index(drop=True)
    return pred_df


if __name__ == "__main__":
    set_seed(SEED)
    print("VERSION_TAG:", VERSION_TAG)

    train_csv, test_csv = find_input_files()
    print("Train CSV:", train_csv)
    print("Test CSV:", test_csv)

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"找不到訓練集: {train_csv}")

    train_raw = pd.read_csv(train_csv)
    test_raw = pd.read_csv(test_csv) if os.path.exists(test_csv) else None

    train_df, id_maps = preprocess_df(train_raw, id_maps=None, is_train=True)
    train_df = ensure_feature_columns(train_df)

    test_df = None
    if test_raw is not None:
        test_df, _ = preprocess_df(test_raw, id_maps=id_maps, is_train=False)
        test_df = ensure_feature_columns(test_df)

    print("BASE_CAT_COLS:", BASE_CAT_COLS)
    print("BASE_NUM_COLS:", BASE_NUM_COLS)
    print("USE_ID_FEATURES:", USE_ID_FEATURES)
    print("STATE_ONLY_MODE:", STATE_ONLY_MODE)
    print("USE_TRANSITION_PROB_FEATURES:", USE_TRANSITION_PROB_FEATURES)
    print("USE_MATCH_LEVEL_SPLIT:", USE_MATCH_LEVEL_SPLIT)
    print("Excluded model features:", ", ".join(EXCLUDED_MODEL_FEATURES))
    print("Train/valid mode: sliding prefix, input rows[:k], target row[k].pointId")
    print("Test mode: input all given rows, predict next pointId")
    print("Transition specs:")
    for name, cols in TRANSITION_SPECS:
        print(f"  {name}: {cols}")

    models, encoders, feature_cols_per_fold, scores, best_bias = train_kfold_xgbprob(
        train_df=train_df,
        test_df=test_df,
        n_splits=5,
        max_seq_len=15,
    )

    if test_df is not None:
                                                                                                               
        cv_pred_df = predict_test_xgbprob(
            models=models,
            encoders=encoders,
            feature_cols_per_fold=feature_cols_per_fold,
            test_df=test_df,
            max_seq_len=15,
        )
        cv_out_path = "pointid_test_pred_xgbprob_v1_2_labelkeep_matchsplit_cvavg.csv"
        cv_pred_df.to_csv(cv_out_path, index=False)
        print(f"{cv_out_path} saved")
        print(cv_pred_df.head())
        print(cv_pred_df.shape)
        prediction_distribution("Test pred pointId CV avg", cv_pred_df["pointId"].values)

                                                                                       
        cv_bias_pred_df = apply_class_bias_to_prediction_df(cv_pred_df, best_bias)
        cv_bias_out_path = "pointid_test_pred_xgbprob_v1_2_labelkeep_matchsplit_cvavg_bias.csv"
        cv_bias_pred_df.to_csv(cv_bias_out_path, index=False)
        print(f"{cv_bias_out_path} saved")
        print(cv_bias_pred_df.head())
        print(cv_bias_pred_df.shape)
        prediction_distribution("Test pred pointId CV avg conservative bias", cv_bias_pred_df["pointId"].values)

                                                                               
        full_pred_df = train_full_and_predict(train_df, test_df, max_seq_len=15)
        full_out_path = "pointid_test_pred_xgbprob_v1_2_labelkeep_matchsplit_full.csv"
        full_pred_df.to_csv(full_out_path, index=False)
        print(f"{full_out_path} saved")
        print(full_pred_df.head())
        print(full_pred_df.shape)
        prediction_distribution("Test pred pointId full", full_pred_df["pointId"].values)

        full_bias_pred_df = apply_class_bias_to_prediction_df(full_pred_df, best_bias)
        full_bias_out_path = "pointid_test_pred_xgbprob_v1_2_labelkeep_matchsplit_full_bias.csv"
        full_bias_pred_df.to_csv(full_bias_out_path, index=False)
        print(f"{full_bias_out_path} saved")
        print(full_bias_pred_df.head())
        print(full_bias_pred_df.shape)
        prediction_distribution("Test pred pointId full conservative bias", full_bias_pred_df["pointId"].values)
    else:
        print(f"找不到測試集: {test_csv}")
