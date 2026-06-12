import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from catboost import CatBoostClassifier, Pool


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


ID_COLS = ["match", "numberGame", "rally_id"]


def build_id_maps(df):
    id_maps = {}

    for col in ID_COLS:
        values = sorted(df[col].dropna().unique())
        id_maps[col] = {v: i + 1 for i, v in enumerate(values)}

    all_players = pd.concat([
        df["gamePlayerId"],
        df["gamePlayerOtherId"],
    ]).dropna().unique()

    player_map = {v: i + 1 for i, v in enumerate(sorted(all_players))}
    id_maps["gamePlayerId"] = player_map
    id_maps["gamePlayerOtherId"] = player_map

    return id_maps


def apply_id_maps(df, id_maps):
    df = df.copy()

    for col, mapping in id_maps.items():
        df[f"{col}_idx"] = df[col].map(mapping).fillna(0).astype(np.int64)

    return df


def spin_transform(df):
    df = df.copy()

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


def action_transform(df):
    df = df.copy()

    if "actionId" not in df.columns:
        df["actionId"] = 0
        df["aid_group"] = 0
        df["aid_sub"] = 0
        return df

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


def point_transform(df):
    df = df.copy()

    if "pointId" not in df.columns:
        df["pointId"] = 0
        df["pid_depth"] = 0
        df["pid_side"] = 0
        return df

    table = np.array([
        (0, 0),
        (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3),
        (3, 1), (3, 2), (3, 3),
    ], dtype=np.int64)

    pid = df["pointId"].fillna(0).astype(np.int64).to_numpy()

    if pid.min() < 0 or pid.max() > 9:
        raise ValueError("pointId out of range 0-9")

    ptuple = table[pid]
    df["pid_depth"] = ptuple[:, 0]
    df["pid_side"] = ptuple[:, 1]

    return df


def add_numeric_features(df):
    df = df.copy()

    if "scoreSelf" in df.columns and "scoreOther" in df.columns:
        df["scoreDiff"] = df["scoreSelf"] - df["scoreOther"]
    else:
        df["scoreSelf"] = 0
        df["scoreOther"] = 0
        df["scoreDiff"] = 0

    return df


def preprocess_server_df(df, id_maps=None, is_train=True):
    df = df.copy()

    if is_train:
        id_maps = build_id_maps(df)

    df = apply_id_maps(df, id_maps)
    df = spin_transform(df)
    df = action_transform(df)
    df = point_transform(df)
    df = add_numeric_features(df)

    return df, id_maps


def check_server_label_consistency(df):
    check = df.groupby("rally_uid")["serverGetPoint"].nunique()
    bad = check[check > 1]

    if len(bad) > 0:
        print("有 rally_uid 內 serverGetPoint 不一致")
        print(bad.head())
    else:
        print("每個 rally_uid 內 serverGetPoint 都一致")


def check_score_change_within_rally(df):
    score_check = df.groupby("rally_uid").agg(
        scoreSelf_nunique=("scoreSelf", "nunique"),
        scoreOther_nunique=("scoreOther", "nunique"),
        scoreSelf_first=("scoreSelf", "first"),
        scoreSelf_last=("scoreSelf", "last"),
        scoreOther_first=("scoreOther", "first"),
        scoreOther_last=("scoreOther", "last"),
        serverGetPoint=("serverGetPoint", "first") if "serverGetPoint" in df.columns else ("rally_uid", "first"),
    ).reset_index()

    score_check["score_changed"] = (
        (score_check["scoreSelf_first"] != score_check["scoreSelf_last"]) |
        (score_check["scoreOther_first"] != score_check["scoreOther_last"])
    )

    print(score_check["score_changed"].value_counts(normalize=True))
    print(score_check[score_check["score_changed"]].head(20))


def check_unseen_id_rate(df, idx_cols):
    for col in idx_cols:
        if col not in df.columns:
            continue

        rate = (df[col] == 0).mean()
        count = int((df[col] == 0).sum())
        print(f"{col}: unseen rate = {rate:.4f}, count = {count}")


def _safe_last_n_values(g_prefix, col, n=3, fill_value=0):
    values = g_prefix[col].tail(n).to_list()
    values = [fill_value] * max(0, n - len(values)) + values
    return values


def _build_one_prefix_row(rally_uid, g_prefix, is_train, target_value=None, sample_weight=None):

    row = {
        "rally_uid": rally_uid,
        "match_idx": int(g_prefix["match_idx"].iloc[0]),
        "numberGame_idx": int(g_prefix["numberGame_idx"].iloc[0]),
        "rally_id_idx": int(g_prefix["rally_id_idx"].iloc[0]),
        "gamePlayerId_idx": int(g_prefix["gamePlayerId_idx"].iloc[0]),
        "gamePlayerOtherId_idx": int(g_prefix["gamePlayerOtherId_idx"].iloc[0]),

        "sex": int(g_prefix["sex"].iloc[0]),
        "scoreSelf_first": float(g_prefix["scoreSelf"].iloc[0]),
        "scoreOther_first": float(g_prefix["scoreOther"].iloc[0]),
        "scoreDiff_first": float(g_prefix["scoreDiff"].iloc[0]),

        "prefix_len": int(len(g_prefix)),
        "strikeNumber_max": float(g_prefix["strikeNumber"].max()),
        "strikeNumber_mean": float(g_prefix["strikeNumber"].mean()),

        "strikeId_first": int(g_prefix["strikeId"].iloc[0]),
        "strikeId_last": int(g_prefix["strikeId"].iloc[-1]),
        "strikeId_nunique": int(g_prefix["strikeId"].nunique()),

        "handId_first": int(g_prefix["handId"].iloc[0]),
        "handId_last": int(g_prefix["handId"].iloc[-1]),
        "handId_nunique": int(g_prefix["handId"].nunique()),

        "strength_mean": float(g_prefix["strengthId"].mean()),
        "strength_max": float(g_prefix["strengthId"].max()),
        "strength_last": int(g_prefix["strengthId"].iloc[-1]),
        "strength_nunique": int(g_prefix["strengthId"].nunique()),

        "spinId_first": int(g_prefix["spinId"].iloc[0]),
        "spinId_last": int(g_prefix["spinId"].iloc[-1]),
        "spinId_nunique": int(g_prefix["spinId"].nunique()),

        "sid_spin_first": int(g_prefix["sid_spin"].iloc[0]),
        "sid_spin_last": int(g_prefix["sid_spin"].iloc[-1]),
        "sid_spin_nunique": int(g_prefix["sid_spin"].nunique()),
        "sid_side_first": int(g_prefix["sid_side"].iloc[0]),
        "sid_side_last": int(g_prefix["sid_side"].iloc[-1]),
        "sid_side_nunique": int(g_prefix["sid_side"].nunique()),

        "positionId_first": int(g_prefix["positionId"].iloc[0]),
        "positionId_last": int(g_prefix["positionId"].iloc[-1]),
        "positionId_nunique": int(g_prefix["positionId"].nunique()),

        "serve_count": int((g_prefix["strikeId"] == 1).sum()),
        "receive_count": int((g_prefix["strikeId"] == 2).sum()),
        "rally_strike_count": int((g_prefix["strikeId"] == 4).sum()),
        "unknown_strike_count": int((g_prefix["strikeId"] == 8).sum()),
        "stop_count": int((g_prefix["strikeId"] == 16).sum()),
        "strong_count": int((g_prefix["strengthId"] > 0).sum()),
    }

    optional_cols = [
        "actionId", "aid_group", "aid_sub",
        "pointId", "pid_depth", "pid_side",
    ]

    for col in optional_cols:
        if col in g_prefix.columns:
            row[f"{col}_first"] = int(g_prefix[col].iloc[0])
            row[f"{col}_last"] = int(g_prefix[col].iloc[-1])
            row[f"{col}_nunique"] = int(g_prefix[col].nunique())

            last3 = _safe_last_n_values(g_prefix, col, n=3, fill_value=0)
            row[f"{col}_lag3"] = int(last3[0])
            row[f"{col}_lag2"] = int(last3[1])
            row[f"{col}_lag1"] = int(last3[2])

    if is_train:
        row["serverGetPoint"] = int(target_value)
        row["sample_weight"] = float(sample_weight if sample_weight is not None else 1.0)

    return row


def build_servergetpoint_prefix_features(df, is_train=True, mode="train"):
    rows = []
    df = df.copy()
    df["row_order"] = np.arange(len(df))

    for rally_uid, g in df.groupby("rally_uid", sort=False):
        g = g.sort_values(["strikeNumber", "row_order"]).reset_index(drop=True)

        if mode == "test":
            if len(g) == 0:
                continue

            rows.append(_build_one_prefix_row(
                rally_uid=rally_uid,
                g_prefix=g,
                is_train=False,
            ))
            continue

        if len(g) < 2:
            continue

        target_value = int(g["serverGetPoint"].iloc[0])
        prefix_count = len(g) - 1
        prefix_weight = 1.0 / prefix_count

        for k in range(1, len(g)):
            g_prefix = g.iloc[:k]
            rows.append(_build_one_prefix_row(
                rally_uid=rally_uid,
                g_prefix=g_prefix,
                is_train=is_train,
                target_value=target_value,
                sample_weight=prefix_weight,
            ))

    return pd.DataFrame(rows)


RAW_LABEL_FEATURES_TO_DROP = {
    "match_idx",
    "numberGame_idx",
    "rally_id_idx",
    "gamePlayerId_idx",
    "gamePlayerOtherId_idx",

    "strikeId_first", "strikeId_last",
    "handId_first", "handId_last",
    "strength_last",
    "spinId_first", "spinId_last",
    "positionId_first", "positionId_last",

    "actionId_first", "actionId_last", "actionId_nunique",
    "actionId_lag3", "actionId_lag2", "actionId_lag1",
    "pointId_first", "pointId_last", "pointId_nunique",
    "pointId_lag3", "pointId_lag2", "pointId_lag1",
}


def get_feature_columns(server_df):
    target_col = "serverGetPoint"
    drop_cols = {
        "rally_uid",
        target_col,
        "sample_weight",
    } | RAW_LABEL_FEATURES_TO_DROP

    feature_cols = [c for c in server_df.columns if c not in drop_cols]

    cat_features = [
        "sex",
        "sid_spin_first", "sid_spin_last",
        "sid_side_first", "sid_side_last",
        "aid_group_first", "aid_group_last",
        "aid_sub_first", "aid_sub_last",
        "pid_depth_first", "pid_depth_last",
        "pid_side_first", "pid_side_last",
    ]

    for base in ["aid_group", "aid_sub", "pid_depth", "pid_side"]:
        cat_features += [f"{base}_lag3", f"{base}_lag2", f"{base}_lag1"]

    cat_features = [c for c in cat_features if c in feature_cols]

    dropped = [c for c in RAW_LABEL_FEATURES_TO_DROP if c in server_df.columns]
    print(f"Dropped raw label / ID features: {len(dropped)}")
    print(sorted(dropped))

    return feature_cols, cat_features


def make_catboost_model(fold):
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5,
        random_seed=42 + fold,
        od_type="Iter",
        od_wait=100,
        verbose=100,
    )


def find_best_threshold(y_true, prob, sample_weight=None):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)

    best_t = 0.5
    best_score = -1.0

    for t in np.linspace(0.35, 0.75, 81):
        pred = (prob >= t).astype(int)
        score = balanced_accuracy_score(
            y_true,
            pred,
            sample_weight=sample_weight,
        )

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, best_score


def run_servergetpoint_catboost_cv(train_df, n_splits=5):
    rally_ids = train_df["rally_uid"].drop_duplicates().to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    aucs = []
    weighted_aucs = []
    models = []
    final_feature_cols = None
    final_cat_features = None

    oof_true = []
    oof_prob = []
    oof_weight = []

    for fold, (tr_i, va_i) in enumerate(gkf.split(rally_ids, groups=rally_ids), start=1):
        tr_rallies = set(rally_ids[tr_i])
        va_rallies = set(rally_ids[va_i])

        df_tr = train_df[train_df["rally_uid"].isin(tr_rallies)].copy()
        df_va = train_df[train_df["rally_uid"].isin(va_rallies)].copy()

        tr_server_df = build_servergetpoint_prefix_features(df_tr, is_train=True, mode="train")
        va_server_df = build_servergetpoint_prefix_features(df_va, is_train=True, mode="train")

        feature_cols, cat_features = get_feature_columns(tr_server_df)

        missing_in_valid = [c for c in feature_cols if c not in va_server_df.columns]
        if missing_in_valid:
            raise ValueError(f"valid 缺少 feature: {missing_in_valid}")

        X_tr = tr_server_df[feature_cols]
        y_tr = tr_server_df["serverGetPoint"].astype(int).to_numpy()
        w_tr = tr_server_df["sample_weight"].astype(float).to_numpy()

        X_va = va_server_df[feature_cols]
        y_va = va_server_df["serverGetPoint"].astype(int).to_numpy()
        w_va = va_server_df["sample_weight"].astype(float).to_numpy()

        print(f"Fold {fold} train prefix samples: {len(X_tr)}")
        print(f"Fold {fold} valid prefix samples: {len(X_va)}")
        print(f"Fold {fold} train weighted target rate: {np.average(y_tr, weights=w_tr):.4f}")
        print(f"Fold {fold} valid weighted target rate: {np.average(y_va, weights=w_va):.4f}")
        print(f"Fold {fold} feature count: {len(feature_cols)}")
        print(f"Fold {fold} cat feature count: {len(cat_features)}")

        train_pool = Pool(
            data=X_tr,
            label=y_tr,
            cat_features=cat_features,
            weight=w_tr,
        )

        valid_pool = Pool(
            data=X_va,
            label=y_va,
            cat_features=cat_features,
            weight=w_va,
        )

        model = make_catboost_model(fold)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
        )

        pred = model.predict_proba(valid_pool)[:, 1]

        if len(set(y_va)) < 2:
            auc = 0.5
            w_auc = 0.5
        else:
            auc = roc_auc_score(y_va, pred)
            w_auc = roc_auc_score(y_va, pred, sample_weight=w_va)

        print(f"Fold {fold} AUC unweighted: {auc:.4f}")
        print(f"Fold {fold} AUC weighted  : {w_auc:.4f}")

        aucs.append(auc)
        weighted_aucs.append(w_auc)
        models.append(model)

        oof_true.extend(y_va.tolist())
        oof_prob.extend(pred.tolist())
        oof_weight.extend(w_va.tolist())

        final_feature_cols = feature_cols
        final_cat_features = cat_features

    oof_true = np.asarray(oof_true, dtype=int)
    oof_prob = np.asarray(oof_prob, dtype=float)
    oof_weight = np.asarray(oof_weight, dtype=float)

    best_threshold, best_bal_acc = find_best_threshold(
        oof_true,
        oof_prob,
        sample_weight=oof_weight,
    )

    print(f"Mean AUC unweighted: {np.mean(aucs):.4f}")
    print(f"Std  AUC unweighted: {np.std(aucs):.4f}")
    print(f"Mean AUC weighted  : {np.mean(weighted_aucs):.4f}")
    print(f"Std  AUC weighted  : {np.std(weighted_aucs):.4f}")
    print(f"Best threshold by weighted balanced accuracy: {best_threshold:.3f}")
    print(f"Best weighted balanced accuracy: {best_bal_acc:.4f}")

    return models, aucs, weighted_aucs, final_feature_cols, final_cat_features, best_threshold


def compare_train_test_features(train_server_df, test_server_df, feature_cols):
    missing_in_test = [c for c in feature_cols if c not in test_server_df.columns]

    print(f"Train server_df shape: {train_server_df.shape}")
    print(f"Test  server_df shape: {test_server_df.shape}")
    print(f"Feature count: {len(feature_cols)}")

    if missing_in_test:
        print("Test 缺少以下 training features:")
        print(missing_in_test)
    else:
        print("OK: test 沒有缺 training features")

    test_nan = test_server_df[feature_cols].isna().sum()
    test_nan = test_nan[test_nan > 0]

    if len(test_nan) > 0:
        print("\nTest features 有 NaN:")
        print(test_nan)
    else:
        print("OK: test features 沒有 NaN")

    numeric_cols = test_server_df[feature_cols].select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(test_server_df[numeric_cols].to_numpy()).sum()

    if inf_count > 0:
        print(f"test numeric features 有 inf,數量 = {inf_count}")
    else:
        print("OK: test numeric features 沒有 inf")


def predict_servergetpoint_test(models, test_server_df, feature_cols, threshold=0.5):
    X_test = test_server_df[feature_cols]
    prob_sum = np.zeros(len(X_test), dtype=np.float64)

    for model in models:
        prob = model.predict_proba(X_test)[:, 1]
        prob_sum += prob

    prob_avg = prob_sum / len(models)

    pred_df = test_server_df[["rally_uid"]].copy()
    pred_df["serverGetPoint_prob"] = prob_avg
    pred_df["serverGetPoint"] = (prob_avg >= threshold).astype(int)

    return pred_df


if __name__ == "__main__":
    set_seed(42)

    train_csv = "data/train.csv"
    test_csv = "data/test.csv"

    if not os.path.exists(train_csv):
        train_csv = "train.csv"
    if not os.path.exists(test_csv):
        test_csv = "test.csv"

    print(f"Train CSV: {train_csv}")
    print(f"Test  CSV: {test_csv}")

    train_df_raw = pd.read_csv(train_csv)

    check_score_change_within_rally(train_df_raw)
    check_server_label_consistency(train_df_raw)

    train_df, id_maps = preprocess_server_df(
        train_df_raw,
        id_maps=None,
        is_train=True,
    )

    models, aucs, weighted_aucs, feature_cols, cat_features, best_threshold = run_servergetpoint_catboost_cv(
        train_df,
        n_splits=5,
    )

    full_train_server_df = build_servergetpoint_prefix_features(
        train_df,
        is_train=True,
        mode="train",
    )

    print(full_train_server_df.head())
    print("Weighted target rate:", np.average(
        full_train_server_df["serverGetPoint"].astype(int),
        weights=full_train_server_df["sample_weight"].astype(float),
    ))
    print(full_train_server_df["serverGetPoint"].value_counts(normalize=True))

    test_df_raw = pd.read_csv(test_csv)

    test_df, _ = preprocess_server_df(
        test_df_raw,
        id_maps=id_maps,
        is_train=False,
    )

    check_unseen_id_rate(
        test_df,
        idx_cols=[
            "match_idx",
            "numberGame_idx",
            "rally_id_idx",
            "gamePlayerId_idx",
            "gamePlayerOtherId_idx",
        ],
    )

    test_server_df = build_servergetpoint_prefix_features(
        test_df,
        is_train=False,
        mode="test",
    )

    compare_train_test_features(
        train_server_df=full_train_server_df,
        test_server_df=test_server_df,
        feature_cols=feature_cols,
    )

    rally_pred_df = predict_servergetpoint_test(
        models=models,
        test_server_df=test_server_df,
        feature_cols=feature_cols,
        threshold=best_threshold,
    )

    out_df = rally_pred_df[["rally_uid", "serverGetPoint"]].copy()
    out_df.to_csv("servergetpoint_test_pred_v31_clean.csv", index=False)

    rally_pred_df.to_csv("servergetpoint_test_pred_v31_clean_with_prob.csv", index=False)

    print("servergetpoint_test_pred_v31_clean.csv")
    print(out_df.head())
    print(out_df["serverGetPoint"].value_counts(normalize=True).sort_index())

    print("servergetpoint_test_pred_v31_clean_with_prob.csv")
    print(rally_pred_df.head())
    print(f"Threshold used: {best_threshold:.3f}")
