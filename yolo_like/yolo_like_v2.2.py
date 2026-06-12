import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def point_transform(df):
    df = df.copy()
    table = np.array([
        [0, 0],
        [1, 1], [1, 2], [1, 3],
        [2, 1], [2, 2], [2, 3],
        [3, 1], [3, 2], [3, 3],
    ], dtype=np.int64)
    pid = df["pointId"].fillna(0).astype(np.int64).to_numpy()
    if pid.min() < 0 or pid.max() > 9:
        raise ValueError("pointId out of range 0-9")
    mapped = table[pid]
    df["pid_depth"] = mapped[:, 0]
    df["pid_side"] = mapped[:, 1]
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
    aid = df["actionId"].fillna(0).astype(np.int64).to_numpy()
    if aid.min() < 0 or aid.max() > 18:
        raise ValueError("actionId out of range 0-18")
    mapped = table[aid]
    df["aid_group"] = mapped[:, 0]
    df["aid_sub"] = mapped[:, 1]
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
    mapped = table[sid]
    df["sid_spin"] = mapped[:, 0]
    df["sid_side"] = mapped[:, 1]
    return df


def add_mapping_features(df):
    df = action_transform(df)
    df = point_transform(df)
    df = spin_transform(df)
    return df

def compute_class_weights(df, col_name, num_classes):
    counts = df[col_name].value_counts().sort_index()
    weights = np.ones(num_classes, dtype=np.float32)

    total = counts.sum()
    n_present = len(counts)

    for cls_idx, cnt in counts.items():
        if 0 <= cls_idx < num_classes and cnt > 0:
            weights[cls_idx] = total / (n_present * cnt)

    return torch.tensor(weights, dtype=torch.float32)


def find_best_threshold_by_balanced_accuracy(y_true, y_prob, sample_weight=None):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    sample_weight = None if sample_weight is None else np.asarray(sample_weight).astype(float)

    if len(np.unique(y_true)) < 2 or len(y_prob) == 0:
        return 0.5, 0.5

    thresholds = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 99)))

    best_thr = 0.5
    best_ba = -1.0

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        ba = balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        if ba > best_ba:
            best_ba = float(ba)
            best_thr = float(thr)

    return best_thr, best_ba

class SimpleMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):

        batch, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_conv, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_conv.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        x_dbl = self.x_proj(x_conv)
        dt, B, C = torch.split(x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        y = torch.zeros(batch, seq_len, self.d_inner, device=x.device)
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device)
        
        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1)

            h = torch.exp(dt_t * A) * h + (dt_t * B[:, t, :].unsqueeze(1)) * x_conv[:, t, :].unsqueeze(-1)
            y[:, t, :] = torch.einsum('bds,bs->bd', h, C[:, t, :])
            
        y_gated = (y + x_conv * self.D) * F.silu(z)
        return self.out_proj(y_gated)

class MambaTransformer(nn.Module):
    def __init__(
        self,
        num_actions,
        num_positions,
        num_points,
        num_aid_groups,
        num_aid_subs,
        num_pid_depths,
        num_pid_sides,
        num_sid_spins,
        num_sid_sides,
        num_spins,
        num_strengths,
        num_hands,
        num_strikes,
        num_sexes,
        num_matches,
        num_number_games,
        num_rally_ids,
        num_players,
        hidden_dim=128,
        mamba_layers=2,
        max_rally_len=100
    ):
        super().__init__()

        self.embed_action = nn.Embedding(num_actions, 64)
        self.embed_pos = nn.Embedding(num_positions, 32)
        self.embed_spin = nn.Embedding(num_spins, 8)
        self.embed_strength = nn.Embedding(num_strengths, 8)
        self.embed_hand = nn.Embedding(num_hands, 4)
        self.embed_strike = nn.Embedding(num_strikes, 8)
        self.embed_point = nn.Embedding(num_points, 16)
        self.embed_aid_group = nn.Embedding(num_aid_groups, 4)
        self.embed_aid_sub = nn.Embedding(num_aid_subs, 8)
        self.embed_pid_depth = nn.Embedding(num_pid_depths, 4)
        self.embed_pid_side = nn.Embedding(num_pid_sides, 4)
        self.embed_sid_spin = nn.Embedding(num_sid_spins, 4)
        self.embed_sid_side = nn.Embedding(num_sid_sides, 4)

        self.embed_sex = nn.Embedding(num_sexes, 4)
        self.embed_match = nn.Embedding(num_matches, 32)
        self.embed_number_game = nn.Embedding(num_number_games, 8)
        self.embed_rally_id = nn.Embedding(num_rally_ids, 16)
        self.embed_player = nn.Embedding(num_players, 32)
        self.embed_other_player = nn.Embedding(num_players, 32)

        self.proj_num = nn.Linear(3, 16)

        self.pos_embedding = nn.Embedding(max_rally_len, hidden_dim)

        total_input_dim = (
            64 +   
            32 +   
            8 +  
            8 +  
            4 +    
            8 +    
            16 +
            4 +               
            8 +             
            4 +               
            4 +              
            4 +              
            4 +              
            4 +    
            32 +   
            8 +   
            16 +   
            32 +  
            32 +   
            16  
        )

        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.mamba_blocks = nn.ModuleList([SimpleMamba(hidden_dim) for _ in range(mamba_layers)])

        self.transformer_neck = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True),
            num_layers=1
        )

        self.task_queries = nn.Parameter(torch.randn(3, hidden_dim))

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=4, batch_first=True),
            num_layers=1
        )

        self.head_action = nn.Linear(hidden_dim, num_actions)
        self.head_point = nn.Linear(hidden_dim, num_points)
        self.head_win = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        action_seq,
        pos_seq,
        spin_seq,
        strength_seq,
        hand_seq,
        strike_seq,
        point_seq,
        aid_group_seq,
        aid_sub_seq,
        pid_depth_seq,
        pid_side_seq,
        sid_spin_seq,
        sid_side_seq,
        sex_seq,
        match_seq,
        number_game_seq,
        rally_id_seq,
        player_id_seq,
        other_player_id_seq,
        num_feats,
        strike_indices,
        pad_mask=None,
        return_debug=False
    ):
        b, l = action_seq.size()

        action_input = action_seq.clone()
        pos_input = pos_seq.clone()
        spin_input = spin_seq.clone()
        strength_input = strength_seq.clone()
        hand_input = hand_seq.clone()
        strike_input = strike_seq.clone()
        point_input = point_seq.clone()
        aid_group_input = aid_group_seq.clone()
        aid_sub_input = aid_sub_seq.clone()
        pid_depth_input = pid_depth_seq.clone()
        pid_side_input = pid_side_seq.clone()
        sid_spin_input = sid_spin_seq.clone()
        sid_side_input = sid_side_seq.clone()
        sex_input = sex_seq.clone()
        match_input = match_seq.clone()
        number_game_input = number_game_seq.clone()
        rally_id_input = rally_id_seq.clone()
        player_input = player_id_seq.clone()
        other_player_input = other_player_id_seq.clone()

        for item in [
            action_input,
            pos_input,
            spin_input,
            strength_input,
            hand_input,
            strike_input,
            point_input,
            aid_group_input,
            aid_sub_input,
            pid_depth_input,
            pid_side_input,
            sid_spin_input,
            sid_side_input,
            sex_input,
            match_input,
            number_game_input,
            rally_id_input,
            player_input,
            other_player_input
        ]:
            item[item == PAD_IDX] = 0

        x_input = self.input_proj(torch.cat([
            self.embed_action(action_input),
            self.embed_pos(pos_input),
            self.embed_spin(spin_input),
            self.embed_strength(strength_input),
            self.embed_hand(hand_input),
            self.embed_strike(strike_input),
            self.embed_point(point_input),
            self.embed_aid_group(aid_group_input),
            self.embed_aid_sub(aid_sub_input),
            self.embed_pid_depth(pid_depth_input),
            self.embed_pid_side(pid_side_input),
            self.embed_sid_spin(sid_spin_input),
            self.embed_sid_side(sid_side_input),
            self.embed_sex(sex_input),
            self.embed_match(match_input),
            self.embed_number_game(number_game_input),
            self.embed_rally_id(rally_id_input),
            self.embed_player(player_input),
            self.embed_other_player(other_player_input),
            self.proj_num(num_feats)
        ], dim=-1)) + self.pos_embedding(strike_indices)

        x = x_input
        mamba_layer_outputs = []
        for mamba in self.mamba_blocks:
            x = mamba(x)
            if return_debug:
                mamba_layer_outputs.append(x)
        x_mamba = x

        memory = self.transformer_neck(x, src_key_padding_mask=pad_mask)

        queries = self.task_queries.unsqueeze(0).repeat(b, 1, 1)

        task_features = self.transformer_decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=pad_mask
        )

        act_logits = self.head_action(task_features[:, 0, :])
        point_logits = self.head_point(task_features[:, 1, :])
        win_logits = self.head_win(task_features[:, 2, :])

        if return_debug:
            debug = {
                "x_input": x_input.detach(),
                "x_mamba": x_mamba.detach(),
                "memory": memory.detach(),
                "task_action": task_features[:, 0, :].detach(),
                "task_point": task_features[:, 1, :].detach(),
                "task_win": task_features[:, 2, :].detach(),
            }
            for i, layer_out in enumerate(mamba_layer_outputs, start=1):
                debug[f"mamba_layer_{i}"] = layer_out.detach()
            return act_logits, point_logits, win_logits, debug

        return act_logits, point_logits, win_logits

def effective_rank_torch(x, pad_mask=None, max_rows=4096, eps=1e-8):
    if x is None:
        return 0.0
    with torch.no_grad():
        x = x.detach().float()
        if x.dim() == 3:
            if pad_mask is not None and pad_mask.shape[:2] == x.shape[:2]:
                mask = (~pad_mask).reshape(-1).detach().bool()
                x = x.reshape(-1, x.size(-1))[mask]
            else:
                x = x.reshape(-1, x.size(-1))
        if x.size(0) < 2:
            return 0.0
        if x.size(0) > max_rows:
            idx = torch.linspace(0, x.size(0) - 1, steps=max_rows, device=x.device).long()
            x = x.index_select(0, idx)
        x = x - x.mean(dim=0, keepdim=True)
        try:
            s = torch.linalg.svdvals(x)
            p = s / (s.sum() + eps)
            ent = -(p * torch.log(p + eps)).sum()
            return float(torch.exp(ent).detach().cpu().item())
        except Exception:
            return 0.0


def logits_stats(logits):
    with torch.no_grad():
        logits = logits.detach().float()
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        counts = torch.bincount(pred, minlength=prob.size(1)).float()
        top1_ratio = float((counts.max() / max(1, pred.numel())).detach().cpu().item())
        entropy = float((-(prob * torch.log(prob + 1e-8)).sum(dim=1).mean()).detach().cpu().item())
        max_prob = float(prob.max(dim=1).values.mean().detach().cpu().item())
        unique_classes = int((counts > 0).sum().detach().cpu().item())
        return {
            "entropy": entropy,
            "max_prob": max_prob,
            "top1_ratio": top1_ratio,
            "unique_classes": unique_classes,
        }


def macro_recall_np(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(np.unique(y_true).tolist())
    vals = []
    for c in labels:
        mask = y_true == c
        if mask.sum() == 0:
            vals.append(0.0)
        else:
            vals.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(vals)) if vals else 0.0


def minority_recall_np(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    values, counts = np.unique(y_true, return_counts=True)
    if len(values) == 0:
        return 0.0
    threshold = np.median(counts)
    minority_classes = values[counts <= threshold]
    return macro_recall_np(y_true, y_pred, labels=minority_classes.tolist())


def head_diagnostics(head):
    with torch.no_grad():
        W = head.weight.detach().float()
        norms = torch.norm(W, dim=1)
        norm_cv = float((norms.std() / (norms.mean() + 1e-8)).cpu().item())
        if head.bias is not None:
            b = head.bias.detach().float()
            bias_std = float(b.std().cpu().item())
            bias_range = float((b.max() - b.min()).cpu().item())
        else:
            bias_std = 0.0
            bias_range = 0.0
        Wn = F.normalize(W, dim=1)
        sim = Wn @ Wn.T
        sim = sim - torch.eye(sim.size(0), device=sim.device) * 2.0
        nearest_cos = float(sim.max(dim=1).values.mean().cpu().item())
        return {
            "weight_norm_cv": norm_cv,
            "bias_std": bias_std,
            "bias_range": bias_range,
            "nearest_weight_cos": nearest_cos,
        }


def save_diagnostic_history(history, fold_id, save_dir="diagnostic_plots"):
    os.makedirs(save_dir, exist_ok=True)
    hist_df = pd.DataFrame(history)
    csv_path = os.path.join(save_dir, f"fold{fold_id}_diagnostics.csv")
    hist_df.to_csv(csv_path, index=False)

    def _plot(cols, title, filename):
        plt.figure(figsize=(10, 6))
        for c in cols:
            if c in hist_df.columns:
                plt.plot(hist_df["epoch"], hist_df[c], marker="o", linewidth=1.4, label=c)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=150)
        plt.close()

    _plot(
        ["train_loss", "val_loss", "act_f1", "point_f1", "win_auc", "total_score"],
        f"Fold {fold_id} Main Metrics",
        f"fold{fold_id}_main_metrics.png",
    )

    _plot(
        ["rank_input", "rank_mamba", "rank_memory", "rank_task_action", "rank_task_point", "rank_task_win"],
        f"Fold {fold_id} Effective Rank / Representation Collapse",
        f"fold{fold_id}_effective_rank.png",
    )

    _plot(
        ["action_entropy", "action_max_prob", "point_entropy", "point_max_prob"],
        f"Fold {fold_id} Softmax Smoothness",
        f"fold{fold_id}_softmax.png",
    )

    _plot(
        ["point_pred_top1_ratio", "point_pred_unique_classes", "point_macro_recall", "point_minority_recall"],
        f"Fold {fold_id} Point Collapse Signals",
        f"fold{fold_id}_point_collapse.png",
    )

    _plot(
        ["action_pred_top1_ratio", "action_pred_unique_classes", "action_macro_recall", "action_minority_recall"],
        f"Fold {fold_id} Action Collapse Signals",
        f"fold{fold_id}_action_collapse.png",
    )

    _plot(
        ["point_head_norm_cv", "point_head_bias_std", "point_head_nearest_cos",
         "action_head_norm_cv", "action_head_bias_std", "action_head_nearest_cos"],
        f"Fold {fold_id} Classifier Head Geometry",
        f"fold{fold_id}_head_geometry.png",
    )

    print(f"[Diagnostics] saved CSV and plots to: {save_dir}")


class CompetitionSequenceLoss(nn.Module):
    def __init__(self, win_pos_weight=None, ce_ratio=0.15, eps=1e-8):
        super().__init__()
        self.register_buffer("win_pos_weight", win_pos_weight if win_pos_weight is not None else None)
        self.ce_ratio = ce_ratio
        self.eps = eps

        self.bce = nn.BCEWithLogitsLoss(
            reduction='none',
            pos_weight=self.win_pos_weight
        )

    def soft_macro_f1_loss(self, logits, target):
        num_classes = logits.size(1)

        probs = torch.softmax(logits, dim=1)
        y_true = F.one_hot(target, num_classes=num_classes).float()

        tp = (probs * y_true).sum(dim=0)
        fp = (probs * (1.0 - y_true)).sum(dim=0)
        fn = ((1.0 - probs) * y_true).sum(dim=0)

        soft_f1 = (2.0 * tp + self.eps) / (2.0 * tp + fp + fn + self.eps)

        return 1.0 - soft_f1.mean()

    def forward(self, preds, targets, winner_weight=None):
        act_p, point_p, win_p = preds
        act_t, point_t, win_t = targets

        l_act_f1 = self.soft_macro_f1_loss(act_p, act_t)
        l_point_f1 = self.soft_macro_f1_loss(point_p, point_t)

        l_act_ce = F.cross_entropy(act_p, act_t)
        l_point_ce = F.cross_entropy(point_p, point_t)

        l_act = (1.0 - self.ce_ratio) * l_act_f1 + self.ce_ratio * l_act_ce
        l_point = (1.0 - self.ce_ratio) * l_point_f1 + self.ce_ratio * l_point_ce

        l_win_each = self.bce(
            win_p.squeeze(-1),
            win_t.float().squeeze(-1)
        )

        if winner_weight is not None:
            l_win_each = l_win_each * winner_weight.squeeze(-1)

        l_win = l_win_each.mean()

        return 0.4 * l_act + 0.4 * l_point + 0.2 * l_win
    
class RallyDataset(Dataset):

    def __init__(self, df, is_test=False, max_seq_len=15):
        self.df = df.copy()
        self.is_test = is_test
        self.max_seq_len = max_seq_len
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        df = self.df.copy()
        df["row_order"] = np.arange(len(df))

        for _, group in df.groupby('rally_uid', sort=False):
            group = group.sort_values(['strikeNumber', 'row_order']).reset_index(drop=True)

            if len(group) < 2:
                continue

            actions = group['actionId'].values
            positions = group['positionId'].values
            points = group['pointId'].values
            aid_groups = group['aid_group'].values
            aid_subs = group['aid_sub'].values
            pid_depths = group['pid_depth'].values
            pid_sides = group['pid_side'].values
            sid_spins = group['sid_spin'].values
            sid_sides = group['sid_side'].values
            sexs = group['sex'].values
            matches = group['match'].values
            number_games = group['numberGame'].values
            rally_ids = group['rally_id'].values
            player_ids = group['gamePlayerId'].values
            other_player_ids = group['gamePlayerOtherId'].values
            spins = group['spinId'].values
            strengths = group['strengthId'].values
            hands = group['handId'].values
            strikes = group['strikeId'].values

            num_feats = np.stack([
                group['scoreSelf'].values,
                group['scoreOther'].values,
                group['strikeNumber'].values
            ], axis=-1)

            winner = group['serverGetPoint'].values[0] if (not self.is_test and 'serverGetPoint' in group.columns) else 0

            num_prefix_samples = max(1, len(group) - 1)
            winner_weight = 1.0 / num_prefix_samples

            for t in range(1, len(group)):
                start = max(0, t - self.max_seq_len)

                assert start < t

                samples.append({
                    'action_seq': actions[start:t],
                    'pos_seq': positions[start:t],
                    'spin_seq': spins[start:t],
                    'strength_seq': strengths[start:t],
                    'hand_seq': hands[start:t],
                    'strike_seq': strikes[start:t],
                    'point_seq': points[start:t],
                    'aid_group_seq': aid_groups[start:t],
                    'aid_sub_seq': aid_subs[start:t],
                    'pid_depth_seq': pid_depths[start:t],
                    'pid_side_seq': pid_sides[start:t],
                    'sid_spin_seq': sid_spins[start:t],
                    'sid_side_seq': sid_sides[start:t],
                    'num_feats': num_feats[start:t],
                    'target_action': actions[t],
                    'target_point': points[t],
                    'sex_seq': sexs[start:t],
                    'match_seq': matches[start:t],
                    'number_game_seq': number_games[start:t],
                    'rally_id_seq': rally_ids[start:t],
                    'player_id_seq': player_ids[start:t],
                    'other_player_id_seq': other_player_ids[start:t],
                    'winner': winner,
                    'winner_weight': winner_weight
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s['action_seq'], dtype=torch.long),
            torch.tensor(s['pos_seq'], dtype=torch.long),
            torch.tensor(s['spin_seq'], dtype=torch.long),
            torch.tensor(s['strength_seq'], dtype=torch.long),
            torch.tensor(s['hand_seq'], dtype=torch.long),
            torch.tensor(s['strike_seq'], dtype=torch.long),
            torch.tensor(s['point_seq'], dtype=torch.long),
            torch.tensor(s['aid_group_seq'], dtype=torch.long),
            torch.tensor(s['aid_sub_seq'], dtype=torch.long),
            torch.tensor(s['pid_depth_seq'], dtype=torch.long),
            torch.tensor(s['pid_side_seq'], dtype=torch.long),
            torch.tensor(s['sid_spin_seq'], dtype=torch.long),
            torch.tensor(s['sid_side_seq'], dtype=torch.long),
            torch.tensor(s['num_feats'], dtype=torch.float32),
            torch.tensor(s['sex_seq'], dtype=torch.long),
            torch.tensor(s['match_seq'], dtype=torch.long),
            torch.tensor(s['number_game_seq'], dtype=torch.long),
            torch.tensor(s['rally_id_seq'], dtype=torch.long),
            torch.tensor(s['player_id_seq'], dtype=torch.long),
            torch.tensor(s['other_player_id_seq'], dtype=torch.long),
            torch.tensor(s['target_action'], dtype=torch.long),
            torch.tensor(s['target_point'], dtype=torch.long),
            torch.tensor([s['winner']], dtype=torch.float32),
            torch.tensor([s['winner_weight']], dtype=torch.float32)
        )

class InferenceDataset(Dataset):

    def __init__(self, df, max_seq_len=15):
        self.df = df.copy()
        self.max_seq_len = max_seq_len
        self.samples = self._build_inference_samples()

    def _build_inference_samples(self):
        samples = []
        df = self.df.copy()
        df["row_order"] = np.arange(len(df))

        for rally_uid, group in df.groupby('rally_uid', sort=False):
            group = group.sort_values(['strikeNumber', 'row_order']).reset_index(drop=True)

            actions = group['actionId'].values
            positions = group['positionId'].values
            points = group['pointId'].values
            aid_groups = group['aid_group'].values
            aid_subs = group['aid_sub'].values
            pid_depths = group['pid_depth'].values
            pid_sides = group['pid_side'].values
            sid_spins = group['sid_spin'].values
            sid_sides = group['sid_side'].values
            sexs = group['sex'].values
            matches = group['match'].values
            number_games = group['numberGame'].values
            rally_ids = group['rally_id'].values
            player_ids = group['gamePlayerId'].values
            other_player_ids = group['gamePlayerOtherId'].values
            spins = group['spinId'].values
            strengths = group['strengthId'].values
            hands = group['handId'].values
            strikes = group['strikeId'].values

            num_feats = np.stack([
                group['scoreSelf'].values,
                group['scoreOther'].values,
                group['strikeNumber'].values
            ], axis=-1)

            start = max(0, len(group) - self.max_seq_len)

            samples.append({
                'rally_uid': rally_uid,
                'action_seq': actions[start:],
                'pos_seq': positions[start:],
                'spin_seq': spins[start:],
                'strength_seq': strengths[start:],
                'hand_seq': hands[start:],
                'strike_seq': strikes[start:],
                'point_seq': points[start:],
                'aid_group_seq': aid_groups[start:],
                'aid_sub_seq': aid_subs[start:],
                'pid_depth_seq': pid_depths[start:],
                'pid_side_seq': pid_sides[start:],
                'sid_spin_seq': sid_spins[start:],
                'sid_side_seq': sid_sides[start:],
                'num_feats': num_feats[start:],
                'sex_seq': sexs[start:],
                'match_seq': matches[start:],
                'number_game_seq': number_games[start:],
                'rally_id_seq': rally_ids[start:],
                'player_id_seq': player_ids[start:],
                'other_player_id_seq': other_player_ids[start:]
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        return (
            torch.tensor(s['action_seq'], dtype=torch.long),
            torch.tensor(s['pos_seq'], dtype=torch.long),
            torch.tensor(s['spin_seq'], dtype=torch.long),
            torch.tensor(s['strength_seq'], dtype=torch.long),
            torch.tensor(s['hand_seq'], dtype=torch.long),
            torch.tensor(s['strike_seq'], dtype=torch.long),
            torch.tensor(s['point_seq'], dtype=torch.long),
            torch.tensor(s['aid_group_seq'], dtype=torch.long),
            torch.tensor(s['aid_sub_seq'], dtype=torch.long),
            torch.tensor(s['pid_depth_seq'], dtype=torch.long),
            torch.tensor(s['pid_side_seq'], dtype=torch.long),
            torch.tensor(s['sid_spin_seq'], dtype=torch.long),
            torch.tensor(s['sid_side_seq'], dtype=torch.long),
            torch.tensor(s['num_feats'], dtype=torch.float32),
            torch.tensor(s['sex_seq'], dtype=torch.long),
            torch.tensor(s['match_seq'], dtype=torch.long),
            torch.tensor(s['number_game_seq'], dtype=torch.long),
            torch.tensor(s['rally_id_seq'], dtype=torch.long),
            torch.tensor(s['player_id_seq'], dtype=torch.long),
            torch.tensor(s['other_player_id_seq'], dtype=torch.long),
            s['rally_uid']
        )


PAD_IDX = -1

def collate_fn_pad(batch):
    return (
        pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_IDX),   
        pad_sequence([item[1] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[2] for item in batch], batch_first=True, padding_value=PAD_IDX),   
        pad_sequence([item[3] for item in batch], batch_first=True, padding_value=PAD_IDX),   
        pad_sequence([item[4] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[5] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[6] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[7] for item in batch], batch_first=True, padding_value=PAD_IDX),   
        pad_sequence([item[8] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[9] for item in batch], batch_first=True, padding_value=PAD_IDX), 
        pad_sequence([item[10] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[12] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[13] for item in batch], batch_first=True, padding_value=0.0),      
        pad_sequence([item[15] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[16] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[17] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[18] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[19] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        torch.stack([item[20] for item in batch]),
        torch.stack([item[21] for item in batch]),
        torch.stack([item[22] for item in batch]),
        torch.stack([item[23] for item in batch]),
    )


def collate_fn_infer(batch):
    return (
        pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_IDX), 
        pad_sequence([item[1] for item in batch], batch_first=True, padding_value=PAD_IDX),
        pad_sequence([item[2] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[3] for item in batch], batch_first=True, padding_value=PAD_IDX),   
        pad_sequence([item[4] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[5] for item in batch], batch_first=True, padding_value=PAD_IDX),   
        pad_sequence([item[6] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[7] for item in batch], batch_first=True, padding_value=PAD_IDX),
        pad_sequence([item[8] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[9] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[10] for item in batch], batch_first=True, padding_value=PAD_IDX), 
        pad_sequence([item[11] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[12] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[13] for item in batch], batch_first=True, padding_value=0.0),   
        pad_sequence([item[14] for item in batch], batch_first=True, padding_value=PAD_IDX), 
        pad_sequence([item[15] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[16] for item in batch], batch_first=True, padding_value=PAD_IDX), 
        pad_sequence([item[17] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[18] for item in batch], batch_first=True, padding_value=PAD_IDX), 
        pad_sequence([item[19] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        [item[20] for item in batch]
    )

    
def train_one_fold(
    df_train,
    df_val,
    num_actions,
    num_positions,
    num_points,
    num_aid_groups,
    num_aid_subs,
    num_pid_depths,
    num_pid_sides,
    num_sid_spins,
    num_sid_sides,
    num_spins,
    num_strengths,
    num_hands,
    num_strikes,
    num_sexes,
    num_matches,
    num_number_games,
    num_rally_ids,
    num_players,
    max_rally_len,
    fold_id
):

    print(f"train match數: {df_train['match'].nunique()}, val match數: {df_val['match'].nunique()}")
    print(f"train rally_uid數: {df_train['rally_uid'].nunique()}, val rally_uid數: {df_val['rally_uid'].nunique()}")
    print(f"train rows: {len(df_train)}, val rows: {len(df_val)}")
    print("match overlap:", len(set(df_train['match']) & set(df_val['match'])))
    print("rally_uid overlap:", len(set(df_train['rally_uid']) & set(df_val['rally_uid'])))

    train_dataset = RallyDataset(df_train, max_seq_len=15)
    val_dataset = RallyDataset(df_val, max_seq_len=15)
    print(f"train prefix samples: {len(train_dataset)}, val prefix samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn_pad
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn_pad
    )

    model = MambaTransformer(
        num_actions=num_actions,
        num_positions=num_positions,
        num_points=num_points,
        num_aid_groups=num_aid_groups,
        num_aid_subs=num_aid_subs,
        num_pid_depths=num_pid_depths,
        num_pid_sides=num_pid_sides,
        num_sid_spins=num_sid_spins,
        num_sid_sides=num_sid_sides,
        num_spins=num_spins,
        num_strengths=num_strengths,
        num_hands=num_hands,
        num_strikes=num_strikes,
        num_sexes=num_sexes,
        num_matches=num_matches,
        num_number_games=num_number_games,
        num_rally_ids=num_rally_ids,
        num_players=num_players,
        max_rally_len=max_rally_len
    ).to(device)

    win_counts = df_train[['rally_uid', 'serverGetPoint']].drop_duplicates()['serverGetPoint'].value_counts()
    if 0 in win_counts and 1 in win_counts and win_counts[1] > 0:
        win_pos_weight = torch.tensor([win_counts[0] / win_counts[1]], dtype=torch.float32).to(device)
    else:
        win_pos_weight = None

    criterion = CompetitionSequenceLoss(
        win_pos_weight=win_pos_weight,
        ce_ratio=0.15
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    num_epochs = 50
    best_score = -float("inf")
    best_threshold = 0.5
    best_model_path = f"best_model_mapping_prefix_fold{fold_id}.pth"

    history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Fold {fold_id} Epoch {epoch+1}/{num_epochs}")

        for (
            act, pos, spin, strength, hand, strike, point,
            aid_group, aid_sub, pid_depth, pid_side, sid_spin, sid_side,
            num, sex, match_id, number_game, rally_id, player_id, other_player_id,
            t_act, t_point, t_win, t_win_weight
        ) in pbar:
            act = act.to(device)
            pos = pos.to(device)
            spin = spin.to(device)
            strength = strength.to(device)
            hand = hand.to(device)
            strike = strike.to(device)
            point = point.to(device)
            aid_group = aid_group.to(device)
            aid_sub = aid_sub.to(device)
            pid_depth = pid_depth.to(device)
            pid_side = pid_side.to(device)
            sid_spin = sid_spin.to(device)
            sid_side = sid_side.to(device)
            num = num.to(device)

            sex = sex.to(device)
            match_id = match_id.to(device)
            number_game = number_game.to(device)
            rally_id = rally_id.to(device)
            player_id = player_id.to(device)
            other_player_id = other_player_id.to(device)

            t_act = t_act.to(device)
            t_point = t_point.to(device)
            t_win = t_win.to(device)
            t_win_weight = t_win_weight.to(device)

            optimizer.zero_grad()

            strike_idx = torch.arange(
                act.size(1), device=device
            ).unsqueeze(0).expand(act.size(0), -1)

            preds = model(
                act,
                pos,
                spin,
                strength,
                hand,
                strike,
                point,
                aid_group,
                aid_sub,
                pid_depth,
                pid_side,
                sid_spin,
                sid_side,
                sex,
                match_id,
                number_game,
                rally_id,
                player_id,
                other_player_id,
                num,
                strike_idx,
                pad_mask=(act == PAD_IDX)
            )

            loss = criterion(preds, (t_act, t_point, t_win), winner_weight=t_win_weight)

            if torch.isnan(loss):
                print("NaN detected")
                print("act min/max:", act.min().item(), act.max().item())
                print("pos min/max:", pos.min().item(), pos.max().item())
                print("spin min/max:", spin.min().item(), spin.max().item())
                print("strength min/max:", strength.min().item(), strength.max().item())
                print("hand min/max:", hand.min().item(), hand.max().item())
                print("strike min/max:", strike.min().item(), strike.max().item())
                print("point min/max:", point.min().item(), point.max().item())
                print("aid_group min/max:", aid_group.min().item(), aid_group.max().item())
                print("aid_sub min/max:", aid_sub.min().item(), aid_sub.max().item())
                print("pid_depth min/max:", pid_depth.min().item(), pid_depth.max().item())
                print("pid_side min/max:", pid_side.min().item(), pid_side.max().item())
                print("sid_spin min/max:", sid_spin.min().item(), sid_spin.max().item())
                print("sid_side min/max:", sid_side.min().item(), sid_side.max().item())
                print("sex min/max:", sex.min().item(), sex.max().item())
                print("match_id min/max:", match_id.min().item(), match_id.max().item())
                print("number_game min/max:", number_game.min().item(), number_game.max().item())
                print("rally_id min/max:", rally_id.min().item(), rally_id.max().item())
                print("player_id min/max:", player_id.min().item(), player_id.max().item())
                print("other_player_id min/max:", other_player_id.min().item(), other_player_id.max().item())
                print("num has nan:", torch.isnan(num).any().item())
                print("t_act min/max:", t_act.min().item(), t_act.max().item())
                print("t_point min/max:", t_point.min().item(), t_point.max().item())
                print("t_win unique:", torch.unique(t_win))
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        model.eval()
        val_loss = 0.0

        all_act_true, all_act_pred = [], []
        all_point_true, all_point_pred = [], []
        all_win_true, all_win_prob, all_win_weight = [], [], []
        all_act_logits, all_point_logits = [], []
        debug_snapshot = None
        debug_pad_mask = None

        with torch.no_grad():
            for (
                act, pos, spin, strength, hand, strike, point,
                aid_group, aid_sub, pid_depth, pid_side, sid_spin, sid_side,
                num, sex, match_id, number_game, rally_id, player_id, other_player_id,
                t_act, t_point, t_win, t_win_weight
            ) in val_loader:
                act = act.to(device)
                pos = pos.to(device)
                spin = spin.to(device)
                strength = strength.to(device)
                hand = hand.to(device)
                strike = strike.to(device)
                point = point.to(device)
                aid_group = aid_group.to(device)
                aid_sub = aid_sub.to(device)
                pid_depth = pid_depth.to(device)
                pid_side = pid_side.to(device)
                sid_spin = sid_spin.to(device)
                sid_side = sid_side.to(device)
                num = num.to(device)

                sex = sex.to(device)
                match_id = match_id.to(device)
                number_game = number_game.to(device)
                rally_id = rally_id.to(device)
                player_id = player_id.to(device)
                other_player_id = other_player_id.to(device)

                t_act = t_act.to(device)
                t_point = t_point.to(device)
                t_win = t_win.to(device)
                t_win_weight = t_win_weight.to(device)

                strike_idx = torch.arange(
                    act.size(1), device=device
                ).unsqueeze(0).expand(act.size(0), -1)

                if debug_snapshot is None:
                    act_logits, point_logits, win_logits, debug_snapshot = model(
                        act,
                        pos,
                        spin,
                        strength,
                        hand,
                        strike,
                        point,
                        aid_group,
                        aid_sub,
                        pid_depth,
                        pid_side,
                        sid_spin,
                        sid_side,
                        sex,
                        match_id,
                        number_game,
                        rally_id,
                        player_id,
                        other_player_id,
                        num,
                        strike_idx,
                        pad_mask=(act == PAD_IDX),
                        return_debug=True
                    )
                    debug_pad_mask = (act == PAD_IDX).detach()
                else:
                    act_logits, point_logits, win_logits = model(
                        act,
                        pos,
                        spin,
                        strength,
                        hand,
                        strike,
                        point,
                        aid_group,
                        aid_sub,
                        pid_depth,
                        pid_side,
                        sid_spin,
                        sid_side,
                        sex,
                        match_id,
                        number_game,
                        rally_id,
                        player_id,
                        other_player_id,
                        num,
                        strike_idx,
                        pad_mask=(act == PAD_IDX)
                    )

                preds = (act_logits, point_logits, win_logits)
                loss = criterion(preds, (t_act, t_point, t_win), winner_weight=t_win_weight)
                val_loss += loss.item()
                all_act_logits.append(act_logits.detach().cpu())
                all_point_logits.append(point_logits.detach().cpu())

                all_act_true.extend(t_act.cpu().numpy())
                all_act_pred.extend(torch.argmax(act_logits, dim=1).cpu().numpy())

                all_point_true.extend(t_point.cpu().numpy())
                all_point_pred.extend(torch.argmax(point_logits, dim=1).cpu().numpy())

                all_win_true.extend(t_win.squeeze(-1).cpu().numpy())
                all_win_prob.extend(torch.sigmoid(win_logits).squeeze(-1).cpu().numpy())
                all_win_weight.extend(t_win_weight.squeeze(-1).cpu().numpy())

        avg_val = val_loss / len(val_loader)

        act_f1 = f1_score(all_act_true, all_act_pred, average="macro", zero_division=0)
        point_f1 = f1_score(all_point_true, all_point_pred, average="macro", zero_division=0)

        try:
            win_auc = roc_auc_score(all_win_true, all_win_prob)
        except ValueError:
            win_auc = 0.5

        win_threshold, win_bal_acc = find_best_threshold_by_balanced_accuracy(
            all_win_true,
            all_win_prob,
            sample_weight=all_win_weight
        )

        total_score = 0.4 * act_f1 + 0.4 * point_f1 + 0.2 * win_auc

        act_logits_all = torch.cat(all_act_logits, dim=0) if all_act_logits else torch.empty(0, num_actions)
        point_logits_all = torch.cat(all_point_logits, dim=0) if all_point_logits else torch.empty(0, num_points)
        act_stats = logits_stats(act_logits_all) if act_logits_all.numel() else {"entropy": 0.0, "max_prob": 0.0, "top1_ratio": 0.0, "unique_classes": 0}
        point_stats = logits_stats(point_logits_all) if point_logits_all.numel() else {"entropy": 0.0, "max_prob": 0.0, "top1_ratio": 0.0, "unique_classes": 0}

        all_act_true_np = np.asarray(all_act_true)
        all_act_pred_np = np.asarray(all_act_pred)
        all_point_true_np = np.asarray(all_point_true)
        all_point_pred_np = np.asarray(all_point_pred)

        act_head_diag = head_diagnostics(model.head_action)
        point_head_diag = head_diagnostics(model.head_point)

        if debug_snapshot is not None:
            rank_input = effective_rank_torch(debug_snapshot.get("x_input"), debug_pad_mask)
            rank_mamba = effective_rank_torch(debug_snapshot.get("x_mamba"), debug_pad_mask)
            rank_memory = effective_rank_torch(debug_snapshot.get("memory"), debug_pad_mask)
            rank_task_action = effective_rank_torch(debug_snapshot.get("task_action"))
            rank_task_point = effective_rank_torch(debug_snapshot.get("task_point"))
            rank_task_win = effective_rank_torch(debug_snapshot.get("task_win"))
        else:
            rank_input = rank_mamba = rank_memory = rank_task_action = rank_task_point = rank_task_win = 0.0

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(len(train_loader), 1),
            "val_loss": avg_val,
            "act_f1": act_f1,
            "point_f1": point_f1,
            "win_auc": win_auc,
            "win_threshold": win_threshold,
            "win_bal_acc": win_bal_acc,
            "total_score": total_score,

            "rank_input": rank_input,
            "rank_mamba": rank_mamba,
            "rank_memory": rank_memory,
            "rank_task_action": rank_task_action,
            "rank_task_point": rank_task_point,
            "rank_task_win": rank_task_win,

            "action_entropy": act_stats["entropy"],
            "action_max_prob": act_stats["max_prob"],
            "action_pred_top1_ratio": act_stats["top1_ratio"],
            "action_pred_unique_classes": act_stats["unique_classes"],
            "action_macro_recall": macro_recall_np(all_act_true_np, all_act_pred_np, labels=list(range(num_actions))),
            "action_minority_recall": minority_recall_np(all_act_true_np, all_act_pred_np),

            "point_entropy": point_stats["entropy"],
            "point_max_prob": point_stats["max_prob"],
            "point_pred_top1_ratio": point_stats["top1_ratio"],
            "point_pred_unique_classes": point_stats["unique_classes"],
            "point_macro_recall": macro_recall_np(all_point_true_np, all_point_pred_np, labels=list(range(num_points))),
            "point_minority_recall": minority_recall_np(all_point_true_np, all_point_pred_np),

            "action_head_norm_cv": act_head_diag["weight_norm_cv"],
            "action_head_bias_std": act_head_diag["bias_std"],
            "action_head_bias_range": act_head_diag["bias_range"],
            "action_head_nearest_cos": act_head_diag["nearest_weight_cos"],
            "point_head_norm_cv": point_head_diag["weight_norm_cv"],
            "point_head_bias_std": point_head_diag["bias_std"],
            "point_head_bias_range": point_head_diag["bias_range"],
            "point_head_nearest_cos": point_head_diag["nearest_weight_cos"],
        }
        history.append(epoch_record)

        print(
            f"Fold {fold_id} Epoch {epoch+1} 驗證 Loss: {avg_val:.4f} | "
            f"Act F1: {act_f1:.4f} | "
            f"Point F1: {point_f1:.4f} | "
            f"Win AUC: {win_auc:.4f} | "
            f"Win Thr: {win_threshold:.4f} | "
            f"Win BA: {win_bal_acc:.4f} | "
            f"總分: {total_score:.4f} | "
            f"P_top1: {point_stats['top1_ratio']:.2f} | "
            f"P_unique: {point_stats['unique_classes']} | "
            f"R_mamba: {rank_mamba:.1f}"
        )

        if total_score > best_score:
            best_score = total_score
            best_threshold = win_threshold
            torch.save(model.state_dict(), best_model_path)
            threshold_path = best_model_path.replace(".pth", "_threshold.txt")
            with open(threshold_path, "w", encoding="utf-8") as f:
                f.write(str(best_threshold))
            print(
                f"Fold {fold_id} best score {best_score:.4f}，"
                f"best threshold {best_threshold:.4f}，儲存 {best_model_path}"
            )

    save_diagnostic_history(history, fold_id)
    return best_score, best_threshold

def run_group_kfold(
    train_df,
    num_actions,
    num_positions,
    num_points,
    num_aid_groups,
    num_aid_subs,
    num_pid_depths,
    num_pid_sides,
    num_sid_spins,
    num_sid_sides,
    num_spins,
    num_strengths,
    num_hands,
    num_strikes,
    num_sexes,
    num_matches,
    num_number_games,
    num_rally_ids,
    num_players,
    max_rally_len,
    n_splits=5
):
    gkf = GroupKFold(n_splits=n_splits)
    rally_ids = train_df["rally_uid"].drop_duplicates().to_numpy()

    fold_scores = []
    fold_thresholds = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(rally_ids, groups=rally_ids), start=1):
        train_rallies = set(rally_ids[train_idx])
        val_rallies = set(rally_ids[val_idx])

        df_train = train_df[train_df["rally_uid"].isin(train_rallies)].copy()
        df_val = train_df[train_df["rally_uid"].isin(val_rallies)].copy()

        score, threshold = train_one_fold(
            df_train=df_train,
            df_val=df_val,
            num_actions=num_actions,
            num_positions=num_positions,
            num_points=num_points,
            num_aid_groups=num_aid_groups,
            num_aid_subs=num_aid_subs,
            num_pid_depths=num_pid_depths,
            num_pid_sides=num_pid_sides,
            num_sid_spins=num_sid_spins,
            num_sid_sides=num_sid_sides,
            num_spins=num_spins,
            num_strengths=num_strengths,
            num_hands=num_hands,
            num_strikes=num_strikes,
            num_sexes=num_sexes,
            num_matches=num_matches,
            num_number_games=num_number_games,
            num_rally_ids=num_rally_ids,
            num_players=num_players,
            max_rally_len=max_rally_len,
            fold_id=fold
        )
        fold_scores.append(score)
        fold_thresholds.append(threshold)

    for i, (score, threshold) in enumerate(zip(fold_scores, fold_thresholds), start=1):
        print(f"Fold {i}: {score:.4f}, threshold={threshold:.4f}")
    print(f"Mean CV Score: {np.mean(fold_scores):.4f}")
    print(f"Std CV Score: {np.std(fold_scores):.4f}")

    best_fold = int(np.argmax(fold_scores)) + 1
    best_threshold = fold_thresholds[best_fold - 1]
    print(f"Best fold: {best_fold}")
    print(f"Best threshold: {best_threshold:.4f}")

    return fold_scores, best_fold, best_threshold

def generate_submission(
    test_df,
    test_csv_path,
    num_actions,
    num_positions,
    num_points,
    num_aid_groups,
    num_aid_subs,
    num_pid_depths,
    num_pid_sides,
    num_sid_spins,
    num_sid_sides,
    num_spins,
    num_strengths,
    num_hands,
    num_strikes,
    num_sexes,
    num_matches,
    num_number_games,
    num_rally_ids,
    num_players,
    max_rally_len,
    model_path,
    win_threshold=0.5
):
    print("推論")

    model = MambaTransformer(
        num_actions=num_actions,
        num_positions=num_positions,
        num_points=num_points,
        num_aid_groups=num_aid_groups,
        num_aid_subs=num_aid_subs,
        num_pid_depths=num_pid_depths,
        num_pid_sides=num_pid_sides,
        num_sid_spins=num_sid_spins,
        num_sid_sides=num_sid_sides,
        num_spins=num_spins,
        num_strengths=num_strengths,
        num_hands=num_hands,
        num_strikes=num_strikes,
        num_sexes=num_sexes,
        num_matches=num_matches,
        num_number_games=num_number_games,
        num_rally_ids=num_rally_ids,
        num_players=num_players,
        max_rally_len=max_rally_len
    ).to(device)

    model.load_state_dict(torch.load(
        model_path,
        map_location=device,
        weights_only=True
    ))

    model.eval()

    loader = DataLoader(
        InferenceDataset(test_df),
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn_infer
    )

    results = {
        'rally_uid': [],
        'actionId': [],
        'pointId': [],
        'serverGetPoint': []
    }

    with torch.no_grad():
        for (
            act, pos, spin, strength, hand, strike, point,
            aid_group, aid_sub, pid_depth, pid_side, sid_spin, sid_side,
            num, sex, match_id, number_game, rally_id, player_id, other_player_id,
            rally_uids
        ) in tqdm(loader, desc="Inference"):
            act = act.to(device)
            pos = pos.to(device)
            spin = spin.to(device)
            strength = strength.to(device)
            hand = hand.to(device)
            strike = strike.to(device)
            point = point.to(device)
            aid_group = aid_group.to(device)
            aid_sub = aid_sub.to(device)
            pid_depth = pid_depth.to(device)
            pid_side = pid_side.to(device)
            sid_spin = sid_spin.to(device)
            sid_side = sid_side.to(device)
            num = num.to(device)

            sex = sex.to(device)
            match_id = match_id.to(device)
            number_game = number_game.to(device)
            rally_id = rally_id.to(device)
            player_id = player_id.to(device)
            other_player_id = other_player_id.to(device)

            def replace_oob_with_zero(x, num_classes, name):
                bad = (x != PAD_IDX) & ((x < 0) | (x >= num_classes))
                if bad.any():
                    print(
                        f"[Warning] {name} out of range:",
                        "min =", x[bad].min().item(),
                        "max =", x[bad].max().item(),
                        "allowed = 0 ~", num_classes - 1,
                        "count =", bad.sum().item(),
                    )
                    x = x.clone()
                    x[bad] = 0
                return x

            act = replace_oob_with_zero(act, num_actions, "actionId")
            pos = replace_oob_with_zero(pos, num_positions, "positionId")
            spin = replace_oob_with_zero(spin, num_spins, "spinId")
            strength = replace_oob_with_zero(strength, num_strengths, "strengthId")
            hand = replace_oob_with_zero(hand, num_hands, "handId")
            strike = replace_oob_with_zero(strike, num_strikes, "strikeId")
            point = replace_oob_with_zero(point, num_points, "pointId")
            aid_group = replace_oob_with_zero(aid_group, num_aid_groups, "aid_group")
            aid_sub = replace_oob_with_zero(aid_sub, num_aid_subs, "aid_sub")
            pid_depth = replace_oob_with_zero(pid_depth, num_pid_depths, "pid_depth")
            pid_side = replace_oob_with_zero(pid_side, num_pid_sides, "pid_side")
            sid_spin = replace_oob_with_zero(sid_spin, num_sid_spins, "sid_spin")
            sid_side = replace_oob_with_zero(sid_side, num_sid_sides, "sid_side")
            sex = replace_oob_with_zero(sex, num_sexes, "sex")
            match_id = replace_oob_with_zero(match_id, num_matches, "match")
            number_game = replace_oob_with_zero(number_game, num_number_games, "numberGame")
            rally_id = replace_oob_with_zero(rally_id, num_rally_ids, "rally_id")
            player_id = replace_oob_with_zero(player_id, num_players, "gamePlayerId")
            other_player_id = replace_oob_with_zero(other_player_id, num_players, "gamePlayerOtherId")

            strike_idx = torch.arange(
                act.size(1), device=device
            ).unsqueeze(0).expand(act.size(0), -1)

            l_act, l_pos, l_win = model(
                act,
                pos,
                spin,
                strength,
                hand,
                strike,
                point,
                aid_group,
                aid_sub,
                pid_depth,
                pid_side,
                sid_spin,
                sid_side,
                sex,
                match_id,
                number_game,
                rally_id,
                player_id,
                other_player_id,
                num,
                strike_idx,
                pad_mask=(act == PAD_IDX)
            )

            pred_action = torch.argmax(l_act, dim=-1).cpu().numpy()
            pred_point = torch.argmax(l_pos, dim=-1).cpu().numpy()
            pred_win = (torch.sigmoid(l_win).squeeze(-1) >= win_threshold).long().cpu().numpy()

            results['rally_uid'].extend(rally_uids)
            results['actionId'].extend(pred_action)
            results['pointId'].extend(pred_point)
            results['serverGetPoint'].extend(pred_win)

    sub_df = pd.DataFrame(results)
    print("[Test prediction distribution] actionId")
    print(sub_df["actionId"].value_counts(normalize=True).sort_index())
    print("[Test prediction distribution] pointId")
    print(sub_df["pointId"].value_counts(normalize=True).sort_index())
    print("[Test prediction distribution] serverGetPoint")
    print(sub_df["serverGetPoint"].value_counts(normalize=True).sort_index())
    sub_df.to_csv('data/submission_test_new.csv', index=False)
    print("生成 data/submission_test_new.csv")

if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"設備: {device}")

    train_csv = "data/train.csv"
    test_csv = "data/test.csv"

    if not os.path.exists(train_csv):
        train_csv = "train.csv"
    if not os.path.exists(test_csv):
        test_csv = "test.csv"
    if not os.path.exists(test_csv) and os.path.exists("test_new.csv"):
        test_csv = "test_new.csv"

    if os.path.exists(train_csv):
        df_train = pd.read_csv(train_csv)
        df_train = add_mapping_features(df_train)

        NUM_ACTIONS = int(df_train['actionId'].max()) + 1
        NUM_POSITIONS = int(df_train['positionId'].max()) + 1
        NUM_POINTS = int(df_train['pointId'].max()) + 1
        NUM_AID_GROUPS = 5                     
        NUM_AID_SUBS = 8                     
        NUM_PID_DEPTHS = 4                     
        NUM_PID_SIDES = 4                     
        NUM_SID_SPINS = 4                     
        NUM_SID_SIDES = 2                     
        NUM_SPINS = int(df_train['spinId'].max()) + 1
        NUM_STRENGTHS = int(df_train['strengthId'].max()) + 1
        NUM_HANDS = int(df_train['handId'].max()) + 1
        NUM_STRIKES = int(df_train['strikeId'].max()) + 1
        NUM_SEXES = int(df_train['sex'].max()) + 1
        NUM_MATCHES = int(df_train['match'].max()) + 1
        NUM_NUMBER_GAMES = int(df_train['numberGame'].max()) + 1
        NUM_RALLY_IDS = int(df_train['rally_id'].max()) + 1
        NUM_PLAYERS = max(
            int(df_train['gamePlayerId'].max()),
            int(df_train['gamePlayerOtherId'].max())
        ) + 1

        MAX_LEN = int(df_train['strikeNumber'].max()) + 20

        fold_scores, best_fold, best_threshold = run_group_kfold(
            train_df=df_train,
            num_actions=NUM_ACTIONS,
            num_positions=NUM_POSITIONS,
            num_points=NUM_POINTS,
            num_aid_groups=NUM_AID_GROUPS,
            num_aid_subs=NUM_AID_SUBS,
            num_pid_depths=NUM_PID_DEPTHS,
            num_pid_sides=NUM_PID_SIDES,
            num_sid_spins=NUM_SID_SPINS,
            num_sid_sides=NUM_SID_SIDES,
            num_spins=NUM_SPINS,
            num_strengths=NUM_STRENGTHS,
            num_hands=NUM_HANDS,
            num_strikes=NUM_STRIKES,
            num_sexes=NUM_SEXES,
            num_matches=NUM_MATCHES,
            num_number_games=NUM_NUMBER_GAMES,
            num_rally_ids=NUM_RALLY_IDS,
            num_players=NUM_PLAYERS,
            max_rally_len=MAX_LEN,
            n_splits=5
        )

        if os.path.exists(test_csv):
            df_test = pd.read_csv(test_csv)
            df_test = add_mapping_features(df_test)

            generate_submission(
                test_df=df_test,
                test_csv_path=test_csv,
                num_actions=NUM_ACTIONS,
                num_positions=NUM_POSITIONS,
                num_points=NUM_POINTS,
                num_aid_groups=NUM_AID_GROUPS,
                num_aid_subs=NUM_AID_SUBS,
                num_pid_depths=NUM_PID_DEPTHS,
                num_pid_sides=NUM_PID_SIDES,
                num_sid_spins=NUM_SID_SPINS,
                num_sid_sides=NUM_SID_SIDES,
                num_spins=NUM_SPINS,
                num_strengths=NUM_STRENGTHS,
                num_hands=NUM_HANDS,
                num_strikes=NUM_STRIKES,
                num_sexes=NUM_SEXES,
                num_matches=NUM_MATCHES,
                num_number_games=NUM_NUMBER_GAMES,
                num_rally_ids=NUM_RALLY_IDS,
                num_players=NUM_PLAYERS,
                max_rally_len=MAX_LEN,
                model_path=f"best_model_mapping_prefix_fold{best_fold}.pth",
                win_threshold=best_threshold
            )
        else:
            print(f"找不到測試集: {test_csv}")
    else:
        print(f"找不到訓練集: {train_csv}")
