import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

PAD_IDX = -1
VERSION_TAG = "CCHUANG_POINT_V3_3_BUSINESS_FEATURES_20260514"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def _safe_group_change_rate(s, window):

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


def add_numeric_features(df):


    df = df.copy()
    for col in ["scoreSelf", "scoreOther", "strikeNumber"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

                        
    df["scoreDiff"] = df["scoreSelf"] - df["scoreOther"]
    df["scoreSum_bl"] = df["scoreSelf"] + df["scoreOther"]
    df["absScoreDiff_bl"] = df["scoreDiff"].abs()

                     
    df["is_tie_score_bl"] = (df["scoreDiff"] == 0).astype(np.float32)
    df["is_leading_bl"] = (df["scoreDiff"] > 0).astype(np.float32)
    df["is_trailing_bl"] = (df["scoreDiff"] < 0).astype(np.float32)
    df["is_close_score_bl"] = (df["scoreDiff"].abs() <= 2).astype(np.float32)
    df["is_big_lead_bl"] = (df["scoreDiff"] >= 4).astype(np.float32)
    df["is_big_trail_bl"] = (df["scoreDiff"] <= -4).astype(np.float32)
    df["is_late_game_bl"] = (df["scoreSum_bl"] >= 16).astype(np.float32)
    df["is_deuce_like_bl"] = ((df["scoreSelf"] >= 9) & (df["scoreOther"] >= 9)).astype(np.float32)
    df["is_game_point_like_bl"] = (df[["scoreSelf", "scoreOther"]].max(axis=1) >= 10).astype(np.float32)

                                                        
    strike = df["strikeNumber"].fillna(0).astype(float)
    df["rally_phase_bl"] = _rally_phase_from_strike(strike)
    df["strike_parity_bl"] = (strike.astype(int) % 2).astype(np.float32)
    df["strike_mod3_bl"] = (strike.astype(int) % 3).astype(np.float32)
    df["is_serve_phase_bl"] = (strike <= 1).astype(np.float32)
    df["is_receive_phase_bl"] = (strike == 2).astype(np.float32)
    df["is_third_ball_phase_bl"] = (strike == 3).astype(np.float32)
    df["is_rally_phase_bl"] = (strike >= 4).astype(np.float32)

                                                                                    
    for col in ["actionId", "aid_group", "spinId", "sid_spin", "strengthId", "handId", "positionId", "pointId", "pid_side", "pid_depth"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    df["action_spin_bl"] = (df["aid_group"] * 10 + df["sid_spin"]).astype(np.float32)
    df["action_strength_bl"] = (df["aid_group"] * 10 + df["strengthId"]).astype(np.float32)
    df["hand_action_bl"] = (df["handId"] * 20 + df["aid_group"]).astype(np.float32)
    df["position_action_bl"] = (df["positionId"] * 20 + df["aid_group"]).astype(np.float32)

                                                                                               
    sort_cols = [c for c in ["rally_uid", "strikeNumber"] if c in df.columns]
    if "rally_uid" in df.columns:
        row_order = np.arange(len(df))
        df["_orig_order_bl"] = row_order
        df = df.sort_values(["rally_uid", "strikeNumber", "_orig_order_bl"]).reset_index(drop=True)

        for src_col, new_col in [
            ("pointId", "point_changed_prev_bl"),
            ("positionId", "position_changed_prev_bl"),
            ("actionId", "action_changed_prev_bl"),
            ("pid_side", "side_changed_prev_bl"),
            ("pid_depth", "depth_changed_prev_bl"),
        ]:
            df[new_col] = (
                df.groupby("rally_uid", sort=False)[src_col]
                .transform(lambda s: s.ne(s.shift(1)).astype(float))
                .fillna(0.0)
                .astype(np.float32)
            )
            first_mask = df.groupby("rally_uid", sort=False).cumcount() == 0
            df.loc[first_mask, new_col] = 0.0

        df["roll3_point_change_rate_bl"] = df.groupby("rally_uid", sort=False)["pointId"].transform(lambda s: _safe_group_change_rate(s, 3)).astype(np.float32)
        df["roll5_point_change_rate_bl"] = df.groupby("rally_uid", sort=False)["pointId"].transform(lambda s: _safe_group_change_rate(s, 5)).astype(np.float32)
        df["roll5_position_change_rate_bl"] = df.groupby("rally_uid", sort=False)["positionId"].transform(lambda s: _safe_group_change_rate(s, 5)).astype(np.float32)
        df["roll5_action_change_rate_bl"] = df.groupby("rally_uid", sort=False)["actionId"].transform(lambda s: _safe_group_change_rate(s, 5)).astype(np.float32)
        df["roll5_side_change_rate_bl"] = df.groupby("rally_uid", sort=False)["pid_side"].transform(lambda s: _safe_group_change_rate(s, 5)).astype(np.float32)
        df["roll5_depth_change_rate_bl"] = df.groupby("rally_uid", sort=False)["pid_depth"].transform(lambda s: _safe_group_change_rate(s, 5)).astype(np.float32)

        df = df.sort_values("_orig_order_bl").drop(columns=["_orig_order_bl"]).reset_index(drop=True)
    else:
        for col in [
            "point_changed_prev_bl", "position_changed_prev_bl", "action_changed_prev_bl",
            "side_changed_prev_bl", "depth_changed_prev_bl",
            "roll3_point_change_rate_bl", "roll5_point_change_rate_bl",
            "roll5_position_change_rate_bl", "roll5_action_change_rate_bl",
            "roll5_side_change_rate_bl", "roll5_depth_change_rate_bl",
        ]:
            df[col] = 0.0

    df["placement_volatility_bl"] = (
        df["roll5_point_change_rate_bl"]
        + df["roll5_position_change_rate_bl"]
        + df["roll5_side_change_rate_bl"]
        + df["roll5_depth_change_rate_bl"]
    ) / 4.0
    df["is_stable_placement_bl"] = (df["placement_volatility_bl"] <= 0.30).astype(np.float32)
    df["is_chaotic_placement_bl"] = (df["placement_volatility_bl"] >= 0.70).astype(np.float32)

                                                                       
    for col in [c for c in df.columns if c.endswith("_bl")]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    return df


ID_COLS = ["numberGame", "gamePlayerId", "gamePlayerOtherId"]


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

CAT_COLS = [
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

NUM_COLS = [
    "strikeNumber",
    "scoreSelf",
    "scoreOther",
    "scoreDiff",

                                      
    "scoreSum_bl",
    "absScoreDiff_bl",
    "is_tie_score_bl",
    "is_leading_bl",
    "is_trailing_bl",
    "is_close_score_bl",
    "is_big_lead_bl",
    "is_big_trail_bl",
    "is_late_game_bl",
    "is_deuce_like_bl",
    "is_game_point_like_bl",

    "rally_phase_bl",
    "strike_parity_bl",
    "strike_mod3_bl",
    "is_serve_phase_bl",
    "is_receive_phase_bl",
    "is_third_ball_phase_bl",
    "is_rally_phase_bl",

    "action_spin_bl",
    "action_strength_bl",
    "hand_action_bl",
    "position_action_bl",

    "point_changed_prev_bl",
    "position_changed_prev_bl",
    "action_changed_prev_bl",
    "side_changed_prev_bl",
    "depth_changed_prev_bl",
    "roll3_point_change_rate_bl",
    "roll5_point_change_rate_bl",
    "roll5_position_change_rate_bl",
    "roll5_action_change_rate_bl",
    "roll5_side_change_rate_bl",
    "roll5_depth_change_rate_bl",
    "placement_volatility_bl",
    "is_stable_placement_bl",
    "is_chaotic_placement_bl",
]


def ensure_feature_columns(df):
    df = df.copy()
    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(np.int64)
    for col in NUM_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0).astype(np.float32)
    return df


def get_cat_cardinalities(train_df, test_df=None):
    cardinalities = []
    for col in CAT_COLS:
        max_train = int(train_df[col].max()) if col in train_df.columns else 0
        max_test = int(test_df[col].max()) if test_df is not None and col in test_df.columns else 0
        cardinalities.append(max(max_train, max_test) + 1)
    return cardinalities


class SlidingPrefixPointDataset(Dataset):

    def __init__(self, df, max_seq_len=15, skip_target_zero=False):
        self.samples = []
        self.max_seq_len = max_seq_len
        self.skip_target_zero = skip_target_zero

        df = ensure_feature_columns(df)
        df["row_order"] = np.arange(len(df))
        df = df.sort_values(["rally_uid", "strikeNumber", "row_order"]).reset_index(drop=True)

        for rally_uid, g in df.groupby("rally_uid", sort=False):
            g = g.sort_values(["strikeNumber", "row_order"]).reset_index(drop=True)
            if len(g) < 2:
                continue

            for k in range(1, len(g)):
                target = g.iloc[k]
                if self.skip_target_zero and int(target["pointId"]) == 0:
                    continue

                prefix = g.iloc[:k]
                if self.max_seq_len is not None:
                    prefix = prefix.iloc[-self.max_seq_len:]

                self.samples.append({
                    "rally_uid": rally_uid,
                    "cat": prefix[CAT_COLS].to_numpy(np.int64),
                    "num": prefix[NUM_COLS].to_numpy(np.float32),
                    "target_point": int(target["pointId"]),
                    "target_depth": int(target["pid_depth"]),
                    "target_side": int(target["pid_side"]),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TestPrefixPointDataset(Dataset):

    def __init__(self, df, max_seq_len=15):
        self.samples = []
        self.max_seq_len = max_seq_len

        df = ensure_feature_columns(df)
        df["row_order"] = np.arange(len(df))
        df = df.sort_values(["rally_uid", "strikeNumber", "row_order"]).reset_index(drop=True)

        for rally_uid, g in df.groupby("rally_uid", sort=False):
            g = g.sort_values(["strikeNumber", "row_order"]).reset_index(drop=True)
            prefix = g
            if self.max_seq_len is not None:
                prefix = prefix.iloc[-self.max_seq_len:]

            self.samples.append({
                "rally_uid": rally_uid,
                "cat": prefix[CAT_COLS].to_numpy(np.int64),
                "num": prefix[NUM_COLS].to_numpy(np.float32),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_point_batch(batch):
    cat_list = [torch.tensor(x["cat"], dtype=torch.long) for x in batch]
    num_list = [torch.tensor(x["num"], dtype=torch.float32) for x in batch]

    cat_pad = pad_sequence(cat_list, batch_first=True, padding_value=PAD_IDX)
    num_pad = pad_sequence(num_list, batch_first=True, padding_value=0.0)
    pad_mask = cat_pad[:, :, 0].eq(PAD_IDX)

    out = {
        "cat": cat_pad,
        "num": num_pad,
        "pad_mask": pad_mask,
        "rally_uid": [x["rally_uid"] for x in batch],
    }

    if "target_point" in batch[0]:
        out["target_point"] = torch.tensor([x["target_point"] for x in batch], dtype=torch.long)
        out["target_depth"] = torch.tensor([x["target_depth"] for x in batch], dtype=torch.long)
        out["target_side"] = torch.tensor([x["target_side"] for x in batch], dtype=torch.long)

    return out


class PointLabelEmbeddingModel(nn.Module):
    def __init__(
        self,
        cat_cardinalities,
        num_numeric,
        emb_dim=8,
        hidden_dim=128,
        num_heads=4,
        num_layers=1,
        dropout=0.15,
        max_strike_number=64,
        max_seq_len=128,
        label_dim=128,
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(max(1, cardinality), emb_dim, padding_idx=0)
            for cardinality in cat_cardinalities
        ])
        cat_dim = len(cat_cardinalities) * emb_dim

        self.num_proj = nn.Sequential(
            nn.Linear(num_numeric, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        self.input_proj = nn.Sequential(
            nn.Linear(cat_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.strike_pos_emb = nn.Embedding(max_strike_number + 1, hidden_dim, padding_idx=0)
        self.seq_pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.sample_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, label_dim),
            nn.LayerNorm(label_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        point_attr = torch.tensor([
            [0, 0],
            [1, 1], [1, 2], [1, 3],
            [2, 1], [2, 2], [2, 3],
            [3, 1], [3, 2], [3, 3],
        ], dtype=torch.long)
        self.register_buffer("point_attr", point_attr)

        self.label_depth_emb = nn.Embedding(4, label_dim)
        self.label_side_emb = nn.Embedding(4, label_dim)
        self.label_point_emb = nn.Embedding(10, label_dim)
        self.label_mlp = nn.Sequential(
            nn.Linear(label_dim, label_dim),
            nn.LayerNorm(label_dim),
            nn.GELU(),
        )

        self.logit_scale = nn.Parameter(torch.tensor(10.0))
        self.point_bias = nn.Parameter(torch.zeros(10))

        self.head_depth = nn.Linear(label_dim, 4)
        self.head_side = nn.Linear(label_dim, 4)

    def build_label_embeddings(self):
        depth_id = self.point_attr[:, 0]
        side_id = self.point_attr[:, 1]
        point_id = torch.arange(10, device=depth_id.device)

        label_emb = (
            self.label_depth_emb(depth_id)
            + self.label_side_emb(side_id)
            + self.label_point_emb(point_id)
        )
        label_emb = self.label_mlp(label_emb)
        return F.normalize(label_emb, dim=-1)

    def forward(self, cat, num, pad_mask, return_aux=True):
        B, T, _ = cat.shape

        cat_input = cat.clone()
        cat_input[cat_input < 0] = 0

        emb_list = []
        for i, emb in enumerate(self.embeddings):
            col = cat_input[:, :, i].clamp(min=0, max=emb.num_embeddings - 1)
            emb_list.append(emb(col))
        cat_emb = torch.cat(emb_list, dim=-1)

        num_feat = self.num_proj(num)
        x = torch.cat([cat_emb, num_feat], dim=-1)
        x = self.input_proj(x)

        strike_number = num[:, :, 0].long()
        strike_number = torch.clamp(strike_number, min=0, max=self.strike_pos_emb.num_embeddings - 1)
        x = x + self.strike_pos_emb(strike_number)

        seq_pos = torch.arange(T, device=x.device)
        seq_pos = torch.clamp(seq_pos, max=self.seq_pos_emb.num_embeddings - 1)
        seq_pos = seq_pos.unsqueeze(0).expand(B, T)
        x = x + self.seq_pos_emb(seq_pos)

        h = self.encoder(x, src_key_padding_mask=pad_mask)

        valid_len = (~pad_mask).sum(dim=1)
        last_idx = torch.clamp(valid_len - 1, min=0)
        batch_idx = torch.arange(B, device=h.device)
        last_h = h[batch_idx, last_idx]

        valid_mask = (~pad_mask).unsqueeze(-1).float()
        mean_h = (h * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)

        feat = torch.cat([last_h, mean_h], dim=-1)
        z = self.sample_proj(feat)
        z_norm = F.normalize(z, dim=-1)

        label_emb = self.build_label_embeddings()
        point_logits = self.logit_scale.clamp(1.0, 50.0) * (z_norm @ label_emb.T) + self.point_bias

        if return_aux:
            depth_logits = self.head_depth(z)
            side_logits = self.head_side(z)
            return point_logits, depth_logits, side_logits

        return point_logits


def compute_class_weights(samples, key, num_classes, smooth_power=0.5):
    targets = np.array([s[key] for s in samples], dtype=np.int64)
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    weights = np.ones(num_classes, dtype=np.float64)
    present = counts > 0
    if present.sum() == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    total = counts[present].sum()
    n_present = present.sum()
    weights[present] = total / (n_present * counts[present])
    weights = weights ** smooth_power
    weights = weights / weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32)


def point_label_embedding_loss(
    point_logits,
    depth_logits,
    side_logits,
    y_point,
    y_depth,
    y_side,
    point_weight=None,
    depth_weight=None,
    side_weight=None,
    aux_weight=0.2,
):
    loss_point = F.cross_entropy(point_logits, y_point, weight=point_weight)
    loss_depth = F.cross_entropy(depth_logits, y_depth, weight=depth_weight)
    loss_side = F.cross_entropy(side_logits, y_side, weight=side_weight)
    return loss_point + aux_weight * (loss_depth + loss_side)


def compute_point_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=list(range(10)), average="macro", zero_division=0)


def prediction_distribution(name, values):
    s = pd.Series(values)
    print(f"{name} distribution:")
    print(s.value_counts(normalize=True).sort_index())


def numeric_feature_rank_np(X, max_rows=5000, eps=1e-8):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] < 2:
        return 0.0
    if X.shape[0] > max_rows:
        idx = np.linspace(0, X.shape[0] - 1, max_rows).astype(int)
        X = X[idx]
    X = X - np.nanmean(X, axis=0, keepdims=True)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        s = np.linalg.svd(X, compute_uv=False)
        p = s / (s.sum() + eps)
        ent = -(p * np.log(p + eps)).sum()
        return float(np.exp(ent))
    except Exception:
        return 0.0


def business_feature_diagnostics(df, name):
    biz_cols = [c for c in NUM_COLS if c.endswith("_bl") and c in df.columns]
    print(f"[{name}] business feature count: {len(biz_cols)}")
    if not biz_cols:
        return
    X = df[biz_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    print(f"[{name}] business numeric effective_rank≈{numeric_feature_rank_np(X):.2f}")
    summary_cols = [
        "rally_phase_bl", "is_serve_phase_bl", "is_receive_phase_bl",
        "is_third_ball_phase_bl", "is_rally_phase_bl", "is_close_score_bl",
        "is_late_game_bl", "is_deuce_like_bl", "is_game_point_like_bl",
        "placement_volatility_bl", "is_stable_placement_bl", "is_chaotic_placement_bl",
    ]
    summary_cols = [c for c in summary_cols if c in df.columns]
    if summary_cols:
        print(f"[{name}] business feature means:")
        print(df[summary_cols].mean().round(4).to_string())
    if "rally_phase_bl" in df.columns:
        print(f"[{name}] rally_phase distribution:")
        print(df["rally_phase_bl"].value_counts(normalize=True).sort_index())


def train_one_epoch(model, loader, optimizer, device, point_weight, depth_weight, side_weight):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train point sliding-prefix", leave=False):
        cat = batch["cat"].to(device)
        num = batch["num"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        y_point = batch["target_point"].to(device)
        y_depth = batch["target_depth"].to(device)
        y_side = batch["target_side"].to(device)

        optimizer.zero_grad()
        point_logits, depth_logits, side_logits = model(cat, num, pad_mask)
        loss = point_label_embedding_loss(
            point_logits,
            depth_logits,
            side_logits,
            y_point,
            y_depth,
            y_side,
            point_weight=point_weight,
            depth_weight=depth_weight,
            side_weight=side_weight,
            aux_weight=0.2,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_point(model, loader, device, point_weight, depth_weight, side_weight):
    model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []

    for batch in tqdm(loader, desc="Valid point sliding-prefix", leave=False):
        cat = batch["cat"].to(device)
        num = batch["num"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        y_point = batch["target_point"].to(device)
        y_depth = batch["target_depth"].to(device)
        y_side = batch["target_side"].to(device)

        point_logits, depth_logits, side_logits = model(cat, num, pad_mask)
        loss = point_label_embedding_loss(
            point_logits,
            depth_logits,
            side_logits,
            y_point,
            y_depth,
            y_side,
            point_weight=point_weight,
            depth_weight=depth_weight,
            side_weight=side_weight,
            aux_weight=0.2,
        )
        pred = point_logits.argmax(dim=-1)

        all_true.extend(y_point.detach().cpu().numpy().tolist())
        all_pred.extend(pred.detach().cpu().numpy().tolist())
        total_loss += loss.item()

    f1 = compute_point_macro_f1(all_true, all_pred)
    return total_loss / max(len(loader), 1), f1, all_true, all_pred


def make_loaders(df_tr, df_va, batch_size=32, max_seq_len=15, skip_target_zero=False):
    train_dataset = SlidingPrefixPointDataset(df_tr, max_seq_len=max_seq_len, skip_target_zero=skip_target_zero)
    valid_dataset = SlidingPrefixPointDataset(df_va, max_seq_len=max_seq_len, skip_target_zero=False)

    print(f"train sliding-prefix samples: {len(train_dataset)}")
    print(f"valid sliding-prefix samples: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_point_batch,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_point_batch,
        num_workers=0,
    )
    return train_loader, valid_loader, train_dataset


def train_kfold(
    train_df,
    test_df,
    device,
    n_splits=5,
    num_epochs=30,
    batch_size=32,
    lr=1e-4,
    max_seq_len=15,
    skip_target_zero=False,
):
    rally_ids = train_df["rally_uid"].drop_duplicates().to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    cat_cardinalities = get_cat_cardinalities(train_df, test_df)
    num_numeric = len(NUM_COLS)

    models = []
    scores = []
    oof_true = []
    oof_pred = []

    for fold, (tr_i, va_i) in enumerate(gkf.split(rally_ids, groups=rally_ids), start=1):
        tr_rallies = set(rally_ids[tr_i])
        va_rallies = set(rally_ids[va_i])

        df_tr = train_df[train_df["rally_uid"].isin(tr_rallies)].copy()
        df_va = train_df[train_df["rally_uid"].isin(va_rallies)].copy()

        business_feature_diagnostics(df_tr, f"Fold {fold} train")
        business_feature_diagnostics(df_va, f"Fold {fold} valid")

        train_loader, valid_loader, train_dataset = make_loaders(
            df_tr,
            df_va,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            skip_target_zero=skip_target_zero,
        )

        point_weight = compute_class_weights(train_dataset.samples, "target_point", 10).to(device)
        depth_weight = compute_class_weights(train_dataset.samples, "target_depth", 4).to(device)
        side_weight = compute_class_weights(train_dataset.samples, "target_side", 4).to(device)
        print("point CE weights:", point_weight.detach().cpu().numpy())

        model = PointLabelEmbeddingModel(
            cat_cardinalities=cat_cardinalities,
            num_numeric=num_numeric,
            emb_dim=8,
            hidden_dim=128,
            num_heads=4,
            num_layers=1,
            dropout=0.15,
            max_strike_number=64,
            max_seq_len=128,
            label_dim=128,
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        best_score = -1.0
        best_path = f"best_point_label_embedding_slidingprefix_fold{fold}.pth"
        best_true = []
        best_pred = []

        for epoch in range(num_epochs):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                point_weight,
                depth_weight,
                side_weight,
            )
            val_loss, f1, y_true, y_pred = validate_point(
                model,
                valid_loader,
                device,
                point_weight,
                depth_weight,
                side_weight,
            )

            print(
                f"[Point LabelEmbedding SlidingPrefix AllowedHistLabels] Fold {fold} Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
                f"Point Macro F1 {f1:.4f}"
            )

            if f1 > best_score:
                best_score = f1
                best_true = y_true
                best_pred = y_pred
                torch.save(model.state_dict(), best_path)
                print(f"Fold {fold} best F1 {best_score:.4f}, saved {best_path}")

        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
        models.append(model)
        scores.append(best_score)
        oof_true.extend(best_true)
        oof_pred.extend(best_pred)

        prediction_distribution(f"Fold {fold} valid true", best_true)
        prediction_distribution(f"Fold {fold} valid pred", best_pred)

    print("Point LabelEmbedding SlidingPrefix CV scores:", scores)
    print("Point LabelEmbedding SlidingPrefix mean F1:", np.mean(scores))
    print("Point LabelEmbedding SlidingPrefix std  F1:", np.std(scores))
    prediction_distribution("OOF true", oof_true)
    prediction_distribution("OOF pred", oof_pred)

    return models, scores, cat_cardinalities


@torch.no_grad()
def predict_test(models, test_df, device, batch_size=32, max_seq_len=15):
    dataset = TestPrefixPointDataset(test_df, max_seq_len=max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_point_batch, num_workers=0)

    rows = []
    for model in models:
        model.eval()

    for batch in tqdm(loader, desc="Predict point sliding-prefix"):
        cat = batch["cat"].to(device)
        num = batch["num"].to(device)
        pad_mask = batch["pad_mask"].to(device)

        prob_sum = None
        for model in models:
            logits = model(cat, num, pad_mask, return_aux=False)
            prob = torch.softmax(logits, dim=-1)
            prob_sum = prob if prob_sum is None else prob_sum + prob

        pred = (prob_sum / len(models)).argmax(dim=-1).detach().cpu().numpy()

        for rally_uid, point_id in zip(batch["rally_uid"], pred):
            rows.append({"rally_uid": rally_uid, "pointId": int(point_id)})

    pred_df = pd.DataFrame(rows)
    pred_df = pred_df.sort_values("rally_uid").reset_index(drop=True)
    return pred_df


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Device:", device)
    print("VERSION_TAG:", VERSION_TAG)

    train_csv = "data/train.csv"
    test_csv = "data/test.csv"

    if not os.path.exists(train_csv):
        train_csv = "train.csv"
    if not os.path.exists(test_csv):
        test_csv = "test.csv"
    if not os.path.exists(test_csv) and os.path.exists("test_new.csv"):
        test_csv = "test_new.csv"

    train_raw = pd.read_csv(train_csv)
    test_raw = pd.read_csv(test_csv) if os.path.exists(test_csv) else None

    train_df, id_maps = preprocess_df(train_raw, id_maps=None, is_train=True)
    train_df = ensure_feature_columns(train_df)

    test_df = None
    if test_raw is not None:
        test_df, _ = preprocess_df(test_raw, id_maps=id_maps, is_train=False)
        test_df = ensure_feature_columns(test_df)

    print("CAT_COLS:", CAT_COLS)
    print("NUM_COLS:", NUM_COLS)

    models, scores, _ = train_kfold(
        train_df=train_df,
        test_df=test_df,
        device=device,
        n_splits=5,
        num_epochs=30,
        batch_size=32,
        lr=1e-4,
        max_seq_len=15,
        skip_target_zero=False,
    )

    if test_df is not None:
        pred_df = predict_test(models, test_df, device, batch_size=32, max_seq_len=15)
        out_path = "pointid_test_pred_v33_label_embedding_slidingprefix_business.csv"
        pred_df.to_csv(out_path, index=False)
        print(f"{out_path} saved")
        print(pred_df.head())
        print(pred_df.shape)
        prediction_distribution("Test pred pointId", pred_df["pointId"].values)
    else:
        print(f"找不到測試集: {test_csv}")
