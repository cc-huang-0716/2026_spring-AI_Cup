from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def vif(x):
    x = x.select_dtypes(include=['number'])
    x = x.fillna(x.mean())
    vif_data = pd.DataFrame()
                 
    x = x.loc[:, ~x.columns.duplicated()]
    vif_data["feature"] = x.columns
    vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
    print(vif_data.sort_values(by="VIF", ascending=False))


import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gdown
import yaml
import os


warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "params.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def top_k_accuracy(y_true, y_prob, k = config["params"]["top-k"]) :
    topk = np.argsort(-y_prob, axis=1)[:, :k]
    hit = np.any(topk == y_true.reshape(-1, 1), axis=1)
    return float(np.mean(hit))

def build_multitask_table(df, group_col, sort_cols, action_col, point_col, win_col) :

    df = df.copy()
    df = df.sort_values(list(sort_cols)).reset_index(drop=True)

    df["y_next_action"] = df.groupby(group_col)[action_col].shift(-1)
    df["y_next_point"] = df.groupby(group_col)[point_col].shift(-1)
    df["y_rally_win"] = df[win_col]

    df = df.dropna(subset=["y_next_action", "y_next_point", "y_rally_win"]).reset_index(drop=True)

    df["y_next_action"] = df["y_next_action"].astype(int)
    df["y_next_point"] = df["y_next_point"].astype(int)
    df["y_rally_win"] = df["y_rally_win"].astype(int)

    return df

def infer_feature_groups(df):
    candidate_drop_cols = [
        "rally_uid", "match", "rally_id",
        "gamePlayerId", "gamePlayerOtherId",
        "y_next_action", "y_next_point", "y_rally_win"
    ]

    candidate_numeric_cols = [
        "scoreSelf", "scoreOther", "strikeNumber", "numberGame"
    ]

    candidate_categorical_cols = [
        "sex", "actionId", "pointId", "spinId", "strengthId",
        "positionId", "handId", "strikeId"
    ]

    drop_cols = [c for c in candidate_drop_cols if c in df.columns]
    numeric_cols = [c for c in candidate_numeric_cols if c in df.columns]
    categorical_cols = [c for c in candidate_categorical_cols if c in df.columns]

    auto_cat = [
        c for c in df.columns
        if c not in drop_cols + numeric_cols + categorical_cols
        and str(df[c].dtype) in ["object", "category", "string"]
    ]
    categorical_cols += auto_cat

    categorical_cols = list(dict.fromkeys(categorical_cols))
    numeric_cols = list(dict.fromkeys(numeric_cols))
    drop_cols = list(dict.fromkeys(drop_cols))

    return categorical_cols, numeric_cols, drop_cols


def fit_category_maps(train_df, test_df, categorical_cols):
    train_cat_idx = pd.DataFrame(index=train_df.index)
    test_cat_idx = pd.DataFrame(index=test_df.index)
    cardinalities = {}

    for col in categorical_cols:
        vocab = ["__UNK__"] + sorted(train_df[col].unique().tolist())
        mapping = {v: i for i, v in enumerate(vocab)}

        train_cat_idx[col] = train_df[col].map(lambda x: mapping.get(x, 0)).astype(int)
        test_cat_idx[col] = test_df[col].map(lambda x: mapping.get(x, 0)).astype(int)
        cardinalities[col] = len(vocab)

    return train_cat_idx, test_cat_idx, cardinalities


def build_numeric_arrays(train_df, test_df, numeric_cols):
    if not numeric_cols:
        return np.zeros((len(train_df), 0), dtype=np.float32), np.zeros((len(test_df), 0), dtype=np.float32)

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imp.fit_transform(train_df[numeric_cols])
    X_test = imp.transform(test_df[numeric_cols])

    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_test


def compute_embedding_dim(cardinality):
    return max(2, min(16, int(np.ceil(np.sqrt(cardinality)))))


class MultiTaskDataset(Dataset):
    def __init__(self,X_cat,X_num,y_action,y_point,y_win,):
        
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y_action = torch.tensor(y_action, dtype=torch.long)
        self.y_point = torch.tensor(y_point, dtype=torch.long)
        self.y_win = torch.tensor(y_win, dtype=torch.float32)

    def __len__(self):
        return len(self.y_action)

    def __getitem__(self, idx):
        return (
            self.X_cat[idx],
            self.X_num[idx],
            self.y_action[idx],
            self.y_point[idx],
            self.y_win[idx],
        )


class MultiTaskEmbeddingModel(nn.Module):
    def __init__(
        self,
        cardinalities: Dict[str, int],
        num_numeric: int,
        num_action_classes: int,
        num_point_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.cat_cols = list(cardinalities.keys())
        self.embeddings = nn.ModuleDict()
        emb_dims = []

        for col in self.cat_cols:
            card = cardinalities[col]
            emb_dim = compute_embedding_dim(card)
            self.embeddings[col] = nn.Embedding(card, emb_dim)
            emb_dims.append(emb_dim)

        input_dim = sum(emb_dims) + num_numeric

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.action_head = nn.Linear(hidden_dim, num_action_classes)
        self.point_head = nn.Linear(hidden_dim, num_point_classes)
        self.win_head = nn.Linear(hidden_dim, 1)

    def forward(self, x_cat, x_num):
        embs = []
        for i, col in enumerate(self.cat_cols):
            embs.append(self.embeddings[col](x_cat[:, i]))

        x = torch.cat(embs + [x_num], dim=1)
        h = self.backbone(x)

        action_logits = self.action_head(h)
        point_logits = self.point_head(h)
        win_logit = self.win_head(h).squeeze(1)

        return action_logits, point_logits, win_logit


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader
):
    
    criterion_action = nn.CrossEntropyLoss()
    criterion_point = nn.CrossEntropyLoss()
    criterion_win = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["params"]["lr"],
        weight_decay=config["params"]["weight_decay"],
    )

    model.train()
    for epoch in range(config["params"]["epochs"]):
        total_loss = 0.0
        for x_cat, x_num, y_action, y_point, y_win in train_loader:
            x_cat = x_cat.to(config["params"]["device"])
            x_num = x_num.to(config["params"]["device"])
            y_action = y_action.to(config["params"]["device"])
            y_point = y_point.to(config["params"]["device"])
            y_win = y_win.to(config["params"]["device"])

            optimizer.zero_grad()
            action_logits, point_logits, win_logit = model(x_cat, x_num)

            loss_action = criterion_action(action_logits, y_action)
            loss_point = criterion_point(point_logits, y_point)
            loss_win = criterion_win(win_logit, y_win)

            loss = (
                config["params"]["lambda_action"] * loss_action
                + config["params"]["lambda_point"] * loss_point
                + config["params"]["lambda_win"] * loss_win
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch+1:02d}/{config["params"]["epochs"]} - train_loss: {avg_loss:.4f}")


def predict_model(model: nn.Module, test_loader: DataLoader):
    model.eval()

    all_action_prob = []
    all_point_prob = []
    all_win_prob = []

    all_y_action = []
    all_y_point = []
    all_y_win = []

    with torch.no_grad():
        for x_cat, x_num, y_action, y_point, y_win in test_loader:
            x_cat = x_cat.to(config["params"]["device"])
            x_num = x_num.to(config["params"]["device"])

            action_logits, point_logits, win_logit = model(x_cat, x_num)

            action_prob = torch.softmax(action_logits, dim=1).cpu().numpy()
            point_prob = torch.softmax(point_logits, dim=1).cpu().numpy()
            win_prob = torch.sigmoid(win_logit).cpu().numpy()

            all_action_prob.append(action_prob)
            all_point_prob.append(point_prob)
            all_win_prob.append(win_prob)

            all_y_action.append(y_action.numpy())
            all_y_point.append(y_point.numpy())
            all_y_win.append(y_win.numpy())

    return {
        "action_prob": np.vstack(all_action_prob),
        "point_prob": np.vstack(all_point_prob),
        "win_prob": np.concatenate(all_win_prob),
        "y_action": np.concatenate(all_y_action),
        "y_point": np.concatenate(all_y_point),
        "y_win": np.concatenate(all_y_win),
    }


def evaluate_multitask(pred_dict, action_le: LabelEncoder, point_le: LabelEncoder):
    y_action = pred_dict["y_action"]
    y_point = pred_dict["y_point"]
    y_win = pred_dict["y_win"]

    action_prob = pred_dict["action_prob"]
    point_prob = pred_dict["point_prob"]
    win_prob = pred_dict["win_prob"]

    action_pred = np.argmax(action_prob, axis=1)
    point_pred = np.argmax(point_prob, axis=1)

    s1 = f1_score(y_action, action_pred, average="macro")
    s2 = f1_score(y_point, point_pred, average="macro")

    try:
        s3 = roc_auc_score(y_win, win_prob)
    except ValueError:
        s3 = np.nan

    overall_score = 0.4 * s1 + 0.4 * s2 + 0.2 * s3 if not np.isnan(s3) else np.nan

    metrics = {
        "S1_action_macro_f1": s1,
        "S2_point_macro_f1": s2,
        "S3_win_auc": s3,
        "overall_score": overall_score,
        "action_top3_acc": top_k_accuracy(y_action, action_prob, k=min(3, action_prob.shape[1])),
        "point_top3_acc": top_k_accuracy(y_point, point_prob, k=min(3, point_prob.shape[1])),
    }

    action_report = classification_report(
        y_action,
        action_pred,
        labels=np.arange(len(action_le.classes_)),
        target_names=[str(c) for c in action_le.classes_],
        output_dict=True,
        zero_division=0,
    )

    point_report = classification_report(
        y_point,
        point_pred,
        labels=np.arange(len(point_le.classes_)),
        target_names=[str(c) for c in point_le.classes_],
        output_dict=True,
        zero_division=0,
    )

    return metrics, action_report, point_report


def run():
    train_url = config["train_data"]["train_url"]
    train_output = config["train_data"]["train_output"]
    test_url = config["test_data"]["test_url"]
    test_output = config["test_data"]["test_output"]

    train_url = f"https://drive.google.com/uc?id={train_url}"
    test_url = f"https://drive.google.com/uc?id={test_url}"

    gdown.download(train_url, train_output, quiet=False)
    gdown.download(test_url, test_output, quiet=False)

    train_df = pd.read_csv(train_output)
    test_df = pd.read_csv(test_output)
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    print(train_df.head())
    print(train_df.columns.tolist())
    print(test_df.columns.tolist())

    set_seed(config["params"]["random_state"])

    train_df = build_multitask_table(
        df=train_df,
        group_col=config["column"]["group_col"],
        sort_cols=config["column"]["sort_cols"],
        action_col=config["column"]["action_col"],
        point_col=config["column"]["point_col"],
        win_col=config["column"]["win_col"]
    )

    test_df = build_multitask_table(
        df=test_df,
        group_col=config["column"]["group_col"],
        sort_cols=config["column"]["sort_cols"],
        action_col=config["column"]["action_col"],
        point_col=config["column"]["point_col"],
        win_col=config["column"]["win_col"]
    )

    categorical_cols, numeric_cols, drop_cols = infer_feature_groups(train_df)

    for bad_col in [config["column"]["action_col"],config["column"]["point_col"],config["column"]["win_col"], "y_next_action", "y_next_point", "y_rally_win"]:
        if bad_col in categorical_cols:
            categorical_cols.remove(bad_col)
        if bad_col in numeric_cols:
            numeric_cols.remove(bad_col)

    feature_cols = categorical_cols + numeric_cols

    needed_cols = feature_cols + [config["column"]["group_col"], "y_next_action", "y_next_point", "y_rally_win"]
    train_df = train_df[needed_cols].copy()
    test_df = test_df[needed_cols].copy()

    action_le = LabelEncoder()
    point_le = LabelEncoder()

    y_action_train = action_le.fit_transform(train_df["y_next_action"].astype(int))
    y_action_test = action_le.transform(test_df["y_next_action"].astype(int))

    y_point_train = point_le.fit_transform(train_df["y_next_point"].astype(int))
    y_point_test = point_le.transform(test_df["y_next_point"].astype(int))

    y_win_train = train_df["y_rally_win"].astype(int).values
    y_win_test = test_df["y_rally_win"].astype(int).values

    train_cat_idx, test_cat_idx, cardinalities = fit_category_maps(
        train_df[feature_cols], test_df[feature_cols], categorical_cols
    )
    Xn_train, Xn_test = build_numeric_arrays(
        train_df[feature_cols], test_df[feature_cols], numeric_cols
    )

    train_ds = MultiTaskDataset(
        X_cat=train_cat_idx.values.astype(np.int64),
        X_num=Xn_train,
        y_action=y_action_train,
        y_point=y_point_train,
        y_win=y_win_train,
    )

    test_ds = MultiTaskDataset(
        X_cat=test_cat_idx.values.astype(np.int64),
        X_num=Xn_test,
        y_action=y_action_test,
        y_point=y_point_test,
        y_win=y_win_test,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["params"]["batch_size"],
        shuffle=True,
        num_workers=config["params"]["num_workers"],
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config["params"]["batch_size"],
        shuffle=False,
        num_workers=config["params"]["num_workers"],
    )

    model = MultiTaskEmbeddingModel(
        cardinalities=cardinalities,
        num_numeric=Xn_train.shape[1],
        num_action_classes=len(action_le.classes_),
        num_point_classes=len(point_le.classes_),
        hidden_dim=config["params"]["hidden_dim"],
        dropout=config["params"]["dropout"],
    ).to(config["params"]["device"])

    train_one_model(model, train_loader)
    pred_dict = predict_model(model, test_loader)
    metrics, action_report, point_report = evaluate_multitask(pred_dict, action_le, point_le)

    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, (int, float, np.floating)) and not np.isnan(v) else f"{k}: {v}")

    summary_df = pd.DataFrame([metrics])
    summary_df.to_excel("multitask_baseline_summary.xlsx", index=False)

    with pd.ExcelWriter("multitask_baseline_reports.xlsx", engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(action_report).T.to_excel(writer, sheet_name="action_report")
        pd.DataFrame(point_report).T.to_excel(writer, sheet_name="point_report")

    print("\n[INFO] Saved:")
    print("  - multitask_baseline_summary.xlsx")
    print("  - multitask_baseline_reports.xlsx")


