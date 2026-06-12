import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")

class SimpleMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + d_state + d_state)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1))
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_conv, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_conv.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_dbl = self.x_proj(x_conv)
        dt, B, C = torch.split(
            x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log) 
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device)
        y_steps = []

        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1)      
            B_t = B[:, t, :].unsqueeze(1)            
            C_t = C[:, t, :]                          
            x_t = x_conv[:, t, :].unsqueeze(-1)      

            h = torch.exp(dt_t * A) * h + (dt_t * B_t) * x_t
            y_t = torch.einsum('bds,bs->bd', h, C_t)  
            y_steps.append(y_t)

        y = torch.stack(y_steps, dim=1)             
        y_gated = (y + x_conv * self.D.view(1, 1, -1)) * F.silu(z)
        return self.out_proj(y_gated)


class MambaTransformer(nn.Module):
    def __init__(self, num_actions, num_positions, hidden_dim=128, mamba_layers=2, max_rally_len=100):
        super().__init__()
        self.embed_action = nn.Embedding(num_actions, 64, padding_idx=0)
        self.embed_pos = nn.Embedding(num_positions, 32, padding_idx=0)
        self.proj_num = nn.Linear(2, 16)
        self.pos_embedding = nn.Embedding(max_rally_len, hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(64 + 32 + 16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.mamba_blocks = nn.ModuleList([SimpleMamba(hidden_dim) for _ in range(mamba_layers)])
        self.transformer_neck = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True),
            num_layers=1,
        )
        self.task_queries = nn.Parameter(torch.randn(3, hidden_dim))
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=4, batch_first=True),
            num_layers=1,
        )
        self.head_action = nn.Linear(hidden_dim, num_actions)
        self.head_point = nn.Linear(hidden_dim, num_positions)
        self.head_win = nn.Linear(hidden_dim, 1)

    def forward(self, action_seq, pos_seq, num_feats, strike_indices, pad_mask=None):
        b, _ = action_seq.size()
        x = torch.cat(
            [
                self.embed_action(action_seq),
                self.embed_pos(pos_seq),
                self.proj_num(num_feats),
            ],
            dim=-1,
        )
        x = self.input_proj(x) + self.pos_embedding(strike_indices)

        for mamba in self.mamba_blocks:
            x = mamba(x)

        memory = self.transformer_neck(x, src_key_padding_mask=pad_mask)
        queries = self.task_queries.unsqueeze(0).expand(b, -1, -1)
        task_features = self.transformer_decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=pad_mask,
        )
        return (
            self.head_action(task_features[:, 0, :]),
            self.head_point(task_features[:, 1, :]),
            self.head_win(task_features[:, 2, :]),
        )


class CompetitionSequenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        act_p, pos_p, win_p = preds
        act_t, pos_t, win_t = targets
        l_act = F.cross_entropy(act_p, act_t)
        l_pos = F.cross_entropy(pos_p, pos_t)
        l_win = self.bce(win_p.squeeze(-1), win_t.float())
        return 0.4 * l_act + 0.4 * l_pos + 0.2 * l_win


class RallyDataset(Dataset):
    def __init__(self, df, is_test=False, max_seq_len=15):
        self.df = df
        self.is_test = is_test
        self.max_seq_len = max_seq_len
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        for _, group in self.df.groupby('rally_id'):
            group = group.sort_values('strikeNumber').reset_index(drop=True)
            actions = group['actionId'].values
            positions = group['positionId'].values
            num_feats = np.stack(
                [
                    (group['scoreSelf'] - group['scoreOther']).values,
                    group['strikeNumber'].values,
                ],
                axis=-1,
            )
            winner = (
                group['serverGetPoint'].values[0]
                if (not self.is_test and 'serverGetPoint' in group.columns)
                else 0
            )

            for t in range(1, len(group)):
                start = max(0, t - self.max_seq_len)
                samples.append(
                    {
                        'action_seq': actions[start:t],
                        'pos_seq': positions[start:t],
                        'num_feats': num_feats[start:t],
                        'target_action': actions[t],
                        'target_pos': positions[t],
                        'winner': winner,
                    }
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s['action_seq'], dtype=torch.long),
            torch.tensor(s['pos_seq'], dtype=torch.long),
            torch.tensor(s['num_feats'], dtype=torch.float32),
            torch.tensor(s['target_action'], dtype=torch.long),
            torch.tensor(s['target_pos'], dtype=torch.long),
            torch.tensor([s['winner']], dtype=torch.float32),
        )


class InferenceDataset(Dataset):
    def __init__(self, df, max_seq_len=15):
        self.df = df
        self.max_seq_len = max_seq_len
        self.samples = self._build_inference_samples()

    def _build_inference_samples(self):
        samples = []
        for _, group in self.df.groupby('rally_id'):
            group = group.sort_values('strikeNumber')
            actions = group['actionId'].values
            positions = group['positionId'].values
            num_feats = np.stack(
                [
                    (group['scoreSelf'] - group['scoreOther']).values,
                    group['strikeNumber'].values,
                ],
                axis=-1,
            )
            orig_indices = group.index.values
            for t in range(len(group)):
                start = max(0, t + 1 - self.max_seq_len)
                samples.append(
                    {
                        'action_seq': actions[start:t + 1],
                        'pos_seq': positions[start:t + 1],
                        'num_feats': num_feats[start:t + 1],
                        'orig_index': orig_indices[t],
                    }
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s['action_seq'], dtype=torch.long),
            torch.tensor(s['pos_seq'], dtype=torch.long),
            torch.tensor(s['num_feats'], dtype=torch.float32),
            s['orig_index'],
        )


def collate_fn_pad(batch):
    return (
        pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0),
        pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0),
        pad_sequence([item[2] for item in batch], batch_first=True, padding_value=0.0),
        torch.stack([item[3] for item in batch]),
        torch.stack([item[4] for item in batch]),
        torch.stack([item[5] for item in batch]),
    )


def collate_fn_infer(batch):
    return (
        pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0),
        pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0),
        pad_sequence([item[2] for item in batch], batch_first=True, padding_value=0.0),
        [item[3] for item in batch],
    )


def reserve_zero_for_padding(df):
    df = df.copy()

    if df['actionId'].min() == 0:
        df['actionId'] = df['actionId'] + 1
    if df['positionId'].min() == 0:
        df['positionId'] = df['positionId'] + 1

    return df


def split_by_rally_id(train_df, val_ratio=0.1, seed=42):
    rally_ids = train_df['rally_id'].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(rally_ids)

    n_val = max(1, int(len(rally_ids) * val_ratio))
    val_ids = set(rally_ids[:n_val])

    df_train = train_df[~train_df['rally_id'].isin(val_ids)].copy()
    df_val = train_df[train_df['rally_id'].isin(val_ids)].copy()
    return df_train, df_val


def train_model(train_df, num_actions, num_positions, max_rally_len):
    df_train, df_val = split_by_rally_id(train_df, val_ratio=0.1, seed=42)

    train_loader = DataLoader(
        RallyDataset(df_train),
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn_pad,
    )
    val_loader = DataLoader(
        RallyDataset(df_val),
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn_pad,
    )

    model = MambaTransformer(num_actions, num_positions, max_rally_len=max_rally_len).to(device)
    criterion = CompetitionSequenceLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    best_val_score = -float('inf') 
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for act, pos, num, t_act, t_pos, t_win in pbar:
            act = act.to(device)
            pos = pos.to(device)
            num = num.to(device)
            t_act = t_act.to(device)
            t_pos = t_pos.to(device)
            t_win = t_win.to(device)

            optimizer.zero_grad()
            strike_idx = torch.arange(act.size(1), device=device).unsqueeze(0).expand(act.size(0), -1)
            preds = model(act, pos, num, strike_idx, pad_mask=(act == 0))
            loss = criterion(preds, (t_act, t_pos, t_win))
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        model.eval()
        val_loss = 0.0


        all_act_preds, all_pos_preds, all_win_preds = [], [], []
        all_act_targets, all_pos_targets, all_win_targets = [], [], []

        with torch.no_grad():
            for act, pos, num, t_act, t_pos, t_win in val_loader:
                act = act.to(device)
                pos = pos.to(device)
                num = num.to(device)
                t_act = t_act.to(device)
                t_pos = t_pos.to(device)
                t_win = t_win.to(device)

                strike_idx = torch.arange(act.size(1), device=device).unsqueeze(0).expand(act.size(0), -1)
                preds = model(act, pos, num, strike_idx, pad_mask=(act == 0))
                l_act, l_pos, l_win = preds
                
                val_loss += criterion(preds, (t_act, t_pos, t_win)).item()

                all_act_preds.append(l_act.cpu().numpy())
                all_pos_preds.append(l_pos.cpu().numpy())
                all_win_preds.append(l_win.cpu().numpy())

                all_act_targets.append(t_act.cpu().numpy())
                all_pos_targets.append(t_pos.cpu().numpy())
                all_win_targets.append(t_win.cpu().numpy())

        act_preds_np = np.concatenate(all_act_preds, axis=0)
        pos_preds_np = np.concatenate(all_pos_preds, axis=0)
        win_preds_np = np.concatenate(all_win_preds, axis=0).squeeze(-1)

        act_targets_np = np.concatenate(all_act_targets, axis=0)
        pos_targets_np = np.concatenate(all_pos_targets, axis=0)
        win_targets_np = np.concatenate(all_win_targets, axis=0).squeeze(-1)

        act_pred_classes = np.argmax(act_preds_np, axis=1)
        pos_pred_classes = np.argmax(pos_preds_np, axis=1)
        
        win_pred_probs = 1 / (1 + np.exp(-win_preds_np))

        f1_act = f1_score(act_targets_np, act_pred_classes, average='macro', zero_division=0)
        f1_pos = f1_score(pos_targets_np, pos_pred_classes, average='macro', zero_division=0)
        try:
            auc_win = roc_auc_score(win_targets_np, win_pred_probs)
        except ValueError:
            auc_win = np.nan

        overall_score = 0.4 * f1_act + 0.4 * f1_pos + 0.2 * auc_win if not np.isnan(auc_win) else 0

        avg_val = val_loss / max(len(val_loader), 1)
        print(f"Epoch {epoch + 1} 驗證 Loss: {avg_val:.4f} | Act F1: {f1_act:.4f} | Pos F1: {f1_pos:.4f} | Win AUC: {auc_win:.4f} | 總分: {overall_score:.4f}")

        if overall_score > best_val_score:
            best_val_score = overall_score
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"分數 {best_val_score:.4f}，儲存 best_model.pth")

def generate_submission(test_df, test_csv_path, num_actions, num_positions, max_rally_len):
    print("推論")
    model = MambaTransformer(num_actions, num_positions, max_rally_len=max_rally_len).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
    model.eval()

    loader = DataLoader(
        InferenceDataset(test_df),
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn_infer,
    )
    results = {'orig_index': [], 'pred_action': [], 'pred_position': [], 'pred_winner_prob': []}

    with torch.no_grad():
        for act, pos, num, indices in tqdm(loader, desc="Inference"):
            act = act.to(device)
            pos = pos.to(device)
            num = num.to(device)

            strike_idx = torch.arange(act.size(1), device=device).unsqueeze(0).expand(act.size(0), -1)
            l_act, l_pos, l_win = model(act, pos, num, strike_idx, pad_mask=(act == 0))

            results['orig_index'].extend(indices)
            results['pred_action'].extend(torch.argmax(l_act, dim=-1).cpu().numpy())
            results['pred_position'].extend(torch.argmax(l_pos, dim=-1).cpu().numpy())
            results['pred_winner_prob'].extend(torch.sigmoid(l_win).squeeze(-1).cpu().numpy())

    sub_df = test_df.copy()
    pred_df = pd.DataFrame(results).set_index('orig_index')
    sub_df['Next_Action'] = pred_df['pred_action']
    sub_df['Next_Position'] = pred_df['pred_position']
    sub_df['Winner_Prob'] = pred_df['pred_winner_prob']

    os.makedirs('data', exist_ok=True)
    sub_df.to_csv('data/submission.csv', index=False)
    print("生成 data/submission.csv")


if __name__ == "__main__":

    set_seed(42)
    device = get_device()
    print(f"設備: {device}")

    train_csv = 'data/train.csv'
    test_csv = 'data/test.csv'

    if os.path.exists(train_csv):
        df_train = pd.read_csv(train_csv)
        df_train["serverGetPoint"] = df_train["serverGetPoint"].shift(1).fillna(0)
        df_train = reserve_zero_for_padding(df_train)

        NUM_ACTIONS = int(df_train['actionId'].max()) + 1
        NUM_POSITIONS = int(df_train['positionId'].max()) + 1
        MAX_LEN = int(df_train['strikeNumber'].max()) + 20

        train_model(df_train, NUM_ACTIONS, NUM_POSITIONS, MAX_LEN)

        if os.path.exists(test_csv):
            df_test = pd.read_csv(test_csv)
            df_test["serverGetpoint"] = df_test["serverGetpoint"].shift(1).fillna(0)
            df_test = reserve_zero_for_padding(df_test)
            generate_submission(df_test, test_csv, NUM_ACTIONS, NUM_POSITIONS, MAX_LEN)
        else:
            print(f"找不到測試集: {test_csv}")
    else:
        print(f"找不到訓練集: {train_csv}")
