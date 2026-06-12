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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def point_transform(df):
    df = df.copy()

                   
    table = np.array([
        [0, 0],
        [1, 1], [1, 2], [1, 3],
        [2, 1], [2, 2], [2, 3],
        [3, 1], [3, 2], [3, 3],
    ], dtype=np.int64)

    if "pointId" not in df.columns:
        df["pointId"] = 0

    pid = df["pointId"].fillna(0).astype(np.int64).to_numpy()

    if pid.min() < 0 or pid.max() > 9:
        raise ValueError("pointId out of range 0-9")

    ptuple = table[pid]
    df["pid_depth"] = ptuple[:, 0]
    df["pid_side"] = ptuple[:, 1]

    return df


def action_transform(df):
    df = df.copy()
    table = np.array([
        (0, 0),
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 1), (3, 2), (3, 3),
        (4, 1), (4, 2), (4, 3), (4, 4),
    ], dtype=np.int64)

    if "actionId" not in df.columns:
        df["actionId"] = 0

    aid = df["actionId"].fillna(0).astype(np.int64).to_numpy()

    if aid.min() < 0 or aid.max() > 18:
        raise ValueError("actionId out of range 0-18")

    atuple = table[aid]
    df["aid_group"] = atuple[:, 0]
    df["aid_sub"] = atuple[:, 1]

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

    if "spinId" not in df.columns:
        df["spinId"] = 0

    sid = df["spinId"].fillna(0).astype(np.int64).to_numpy()

    if sid.min() < 0 or sid.max() > 5:
        raise ValueError("spinId out of range 0-5")

    stuple = table[sid]
    df["sid_spin"] = stuple[:, 0]
    df["sid_side"] = stuple[:, 1]

    return df


ID_COLS = ["match", "numberGame", "rally_id"]


def build_id_maps(df):
    id_maps = {}

    for col in ID_COLS:
        if col in df.columns:
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
        if col in df.columns:
            df[f"{col}_idx"] = df[col].map(mapping).fillna(0).astype(np.int64)
        else:
            df[f"{col}_idx"] = 0

    return df


def preprocess_common_df(df, id_maps=None, is_train=True):

    df = df.copy()

    if is_train:
        id_maps = build_id_maps(df)

    df = apply_id_maps(df, id_maps)
    df = spin_transform(df)
    df = action_transform(df)
    df = point_transform(df)

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
]


def get_cat_cardinalities(train_df, test_df=None):
    cardinalities = []

    for col in CAT_COLS:
        max_train = int(train_df[col].max()) if col in train_df.columns else 0
        max_test = int(test_df[col].max()) if test_df is not None and col in test_df.columns else 0
        cardinalities.append(max(max_train, max_test) + 1)

    return cardinalities

class SlidingPrefixPointDataset(Dataset):

    def __init__(self, df):
        self.samples = []

        df = df.copy()
        df["row_order"] = np.arange(len(df))

        df = df.sort_values(
            ["rally_uid", "strikeNumber", "row_order"]
        ).reset_index(drop=True)

        for rally_uid, g in df.groupby("rally_uid", sort=False):
            g = g.sort_values(["strikeNumber", "row_order"]).reset_index(drop=True)

            if len(g) < 2:
                continue

            for k in range(1, len(g)):
                prefix = g.iloc[:k]
                target = g.iloc[k]

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

    def __init__(self, df):
        self.samples = []

        df = df.copy()
        df["row_order"] = np.arange(len(df))

        df = df.sort_values(
            ["rally_uid", "strikeNumber", "row_order"]
        ).reset_index(drop=True)

        for rally_uid, g in df.groupby("rally_uid", sort=False):
            g = g.sort_values(["strikeNumber", "row_order"]).reset_index(drop=True)

            self.samples.append({
                "rally_uid": rally_uid,
                "cat": g[CAT_COLS].to_numpy(np.int64),
                "num": g[NUM_COLS].to_numpy(np.float32),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_point_prefix_batch(batch):
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

class PrefixPointModel(nn.Module):
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
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim, padding_idx=0)
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

        self.strike_pos_emb = nn.Embedding(
            max_strike_number + 1,
            hidden_dim,
            padding_idx=0,
        )

        self.seq_pos_emb = nn.Embedding(
            max_seq_len,
            hidden_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

                                                           
        self.head_point = nn.Linear(hidden_dim * 2, 10)

    def forward(self, cat, num, pad_mask):
        B, T, _ = cat.shape

        cat_input = cat.clone()
        cat_input[cat_input < 0] = 0

        emb_list = []
        for i, emb in enumerate(self.embeddings):
            emb_list.append(emb(cat_input[:, :, i]))

        cat_emb = torch.cat(emb_list, dim=-1)
        num_feat = self.num_proj(num)

        x = torch.cat([cat_emb, num_feat], dim=-1)
        x = self.input_proj(x)

        strike_number = num[:, :, 0].long()
        strike_number = torch.clamp(
            strike_number,
            min=0,
            max=self.strike_pos_emb.num_embeddings - 1,
        )
        x = x + self.strike_pos_emb(strike_number)

        seq_pos = torch.arange(T, device=x.device)
        seq_pos = torch.clamp(seq_pos, max=self.seq_pos_emb.num_embeddings - 1)
        seq_pos = seq_pos.unsqueeze(0).expand(B, T)
        x = x + self.seq_pos_emb(seq_pos)

        h = self.encoder(
            x,
            src_key_padding_mask=pad_mask,
        )

        valid_len = (~pad_mask).sum(dim=1)
        last_idx = torch.clamp(valid_len - 1, min=0)
        batch_idx = torch.arange(h.size(0), device=h.device)
        last_h = h[batch_idx, last_idx]

        valid_mask = (~pad_mask).unsqueeze(-1).float()
        mean_h = (h * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)

        feat = torch.cat([last_h, mean_h], dim=-1)
        point_logits = self.head_point(feat)

        return point_logits

POINT_TUPLE_TO_ID = {
    (0, 0): 0,
    (1, 1): 1,
    (1, 2): 2,
    (1, 3): 3,
    (2, 1): 4,
    (2, 2): 5,
    (2, 3): 6,
    (3, 1): 7,
    (3, 2): 8,
    (3, 3): 9,
}


def decode_point_tuple(depth: int, side: int):
    depth = int(depth)
    side = int(side)

    if depth == 0 or side == 0:
        return 0

    return POINT_TUPLE_TO_ID.get((depth, side), 0)


def decode_point_batch(depth_pred, side_pred):
    depth_np = depth_pred.detach().cpu().numpy()
    side_np = side_pred.detach().cpu().numpy()

    out = []
    for d, s in zip(depth_np, side_np):
        out.append(decode_point_tuple(d, s))

    return np.array(out, dtype=np.int64)

def train_point_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train point", leave=False):
        cat = batch["cat"].to(device)
        num = batch["num"].to(device)
        pad_mask = batch["pad_mask"].to(device)

        y_point = batch["target_point"].to(device)

        optimizer.zero_grad()

        point_logits = model(cat, num, pad_mask)

                                                                                             
        loss = F.cross_entropy(point_logits, y_point)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_point(model, loader, device):
    model.eval()

    all_true = []
    all_pred = []
    total_loss = 0.0

    for batch in tqdm(loader, desc="Valid point", leave=False):
        cat = batch["cat"].to(device)
        num = batch["num"].to(device)
        pad_mask = batch["pad_mask"].to(device)

        y_point = batch["target_point"].to(device)

        point_logits = model(cat, num, pad_mask)
        loss = F.cross_entropy(point_logits, y_point)

        point_pred = point_logits.argmax(dim=-1)

        all_true.extend(y_point.detach().cpu().numpy().tolist())
        all_pred.extend(point_pred.detach().cpu().numpy().tolist())

        total_loss += loss.item()

    f1 = f1_score(
        all_true,
        all_pred,
        labels=list(range(10)),
        average="macro",
        zero_division=0,
    )

    return total_loss / max(len(loader), 1), f1


def make_point_loaders(df_tr, df_va, batch_size=32):
    train_dataset = SlidingPrefixPointDataset(df_tr)
    valid_dataset = SlidingPrefixPointDataset(df_va)

    print(f"train prefix samples: {len(train_dataset)}")
    print(f"valid prefix samples: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_point_prefix_batch,
        num_workers=0,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_point_prefix_batch,
        num_workers=0,
    )

    return train_loader, valid_loader


def train_point_kfold(
    train_df,
    cat_cardinalities,
    num_numeric,
    device,
    n_splits=5,
    num_epochs=30,
    batch_size=32,
    lr=1e-4,
):
    rally_ids = train_df["rally_uid"].drop_duplicates().to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    models = []
    scores = []

    for fold, (tr_i, va_i) in enumerate(gkf.split(rally_ids, groups=rally_ids), start=1):
        tr_rallies = set(rally_ids[tr_i])
        va_rallies = set(rally_ids[va_i])

        df_tr = train_df[train_df["rally_uid"].isin(tr_rallies)].copy()
        df_va = train_df[train_df["rally_uid"].isin(va_rallies)].copy()

        train_loader, valid_loader = make_point_loaders(df_tr, df_va, batch_size=batch_size)

        model = PrefixPointModel(
            cat_cardinalities=cat_cardinalities,
            num_numeric=num_numeric,
            emb_dim=8,
            hidden_dim=128,
            num_heads=4,
            num_layers=1,
            dropout=0.15,
            max_strike_number=64,
            max_seq_len=128,
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        best_score = -1.0
        best_path = f"best_point_prefix_v31_direct_fold{fold}.pth"

        for epoch in range(num_epochs):
            train_loss = train_point_one_epoch(model, train_loader, optimizer, device)
            val_loss, f1 = validate_point(model, valid_loader, device)

            print(
                f"[Point v3.1 Prefix Direct] Fold {fold} Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss {train_loss:.4f} | "
                f"Val Loss {val_loss:.4f} | "
                f"Point Macro F1 {f1:.4f}"
            )

            if f1 > best_score:
                best_score = f1
                torch.save(model.state_dict(), best_path)
                print(f"Fold {fold} best F1 {best_score:.4f}, saved {best_path}")

        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()

        models.append(model)
        scores.append(best_score)

    print("Point v3.1 Prefix Direct CV scores:", scores)
    print("Point v3.1 Prefix Direct mean F1:", np.mean(scores))
    print("Point v3.1 Prefix Direct std  F1:", np.std(scores))
    return models, scores

@torch.no_grad()
def predict_point_test(models, test_df, device, batch_size=32):
    dataset = TestPrefixPointDataset(test_df)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_point_prefix_batch,
        num_workers=0,
    )

    rows = []

    for model in models:
        model.eval()

    for batch in tqdm(loader, desc="Predict point"):
        cat = batch["cat"].to(device)
        num = batch["num"].to(device)
        pad_mask = batch["pad_mask"].to(device)

        point_prob_sum = None

        for model in models:
            point_logits = model(cat, num, pad_mask)
            point_prob = torch.softmax(point_logits, dim=-1)

            if point_prob_sum is None:
                point_prob_sum = point_prob
            else:
                point_prob_sum += point_prob

        point_pred = (point_prob_sum / len(models)).argmax(dim=-1)
        point_pred = point_pred.detach().cpu().numpy()

        for rally_uid, point_id in zip(batch["rally_uid"], point_pred):
            rows.append({
                "rally_uid": rally_uid,
                "pointId": int(point_id),
            })

    pred_df = pd.DataFrame(rows)
    pred_df = pred_df.sort_values("rally_uid").reset_index(drop=True)

    return pred_df

if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Device:", device)

    train_csv = "data/train.csv"
    test_csv = "data/test.csv"

    if not os.path.exists(train_csv):
        train_csv = "train.csv"

    if not os.path.exists(test_csv):
        test_csv = "test.csv"

    train_raw = pd.read_csv(train_csv)
    test_raw = pd.read_csv(test_csv)

    train_df, id_maps = preprocess_common_df(train_raw, id_maps=None, is_train=True)
    test_df, _ = preprocess_common_df(test_raw, id_maps=id_maps, is_train=False)

    cat_cardinalities = get_cat_cardinalities(train_df, test_df)
    num_numeric = len(NUM_COLS)

    print("CAT_COLS:")
    for col, card in zip(CAT_COLS, cat_cardinalities):
        print(f"{col}: {card}")

    print("NUM_COLS:", NUM_COLS)

    point_models, point_scores = train_point_kfold(
        train_df=train_df,
        cat_cardinalities=cat_cardinalities,
        num_numeric=num_numeric,
        device=device,
        n_splits=5,
        num_epochs=30,
        batch_size=32,
        lr=1e-4,
    )

    point_pred_df = predict_point_test(
        models=point_models,
        test_df=test_df,
        device=device,
        batch_size=32,
    )

    point_pred_df.to_csv("pointid_test_pred_v31_prefix.csv", index=False)

    print("pointid_test_pred_v31_prefix.csv saved")
    print(point_pred_df.head())
    print(point_pred_df.shape)
    print(point_pred_df["pointId"].value_counts(normalize=True).sort_index())
