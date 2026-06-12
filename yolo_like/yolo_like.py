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

def compute_class_weights(df, col_name, num_classes):
    counts = df[col_name].value_counts().sort_index()
    weights = np.ones(num_classes, dtype=np.float32)

    total = counts.sum()
    n_present = len(counts)

    for cls_idx, cnt in counts.items():
        if 0 <= cls_idx < num_classes and cnt > 0:
            weights[cls_idx] = total / (n_present * cnt)

    return torch.tensor(weights, dtype=torch.float32)


def find_best_binary_threshold(y_true, y_prob, sample_weight=None, metric="balanced_accuracy"):


    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight).astype(float)
        
    quantiles = np.linspace(0.01, 0.99, 99)
    dist_thresholds = np.quantile(y_prob, quantiles)
    grid_thresholds = np.linspace(0.05, 0.95, 181)
    candidates = np.unique(np.concatenate([dist_thresholds, grid_thresholds]))

    best_thr = 0.5
    best_score = -1.0

    for thr in candidates:
        pred = (y_prob >= thr).astype(int)
        if pred.min() == pred.max():
            continue

        if metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, pred, sample_weight=sample_weight)
        else:
            raise ValueError(f"Unknown threshold metric: {metric}")

        if score > best_score:
            best_score = score
            best_thr = float(thr)

    if best_score < 0:
        best_score = balanced_accuracy_score(
            y_true,
            (y_prob >= 0.5).astype(int),
            sample_weight=sample_weight
        )
        best_thr = 0.5

    return best_thr, float(best_score)

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
        sex_seq,
        match_seq,
        number_game_seq,
        rally_id_seq,
        player_id_seq,
        other_player_id_seq,
        num_feats,
        strike_indices,
        pad_mask=None
    ):
        b, l = action_seq.size()

        action_input = action_seq.clone()
        pos_input = pos_seq.clone()
        spin_input = spin_seq.clone()
        strength_input = strength_seq.clone()
        hand_input = hand_seq.clone()
        strike_input = strike_seq.clone()
        point_input = point_seq.clone()
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
            sex_input,
            match_input,
            number_game_input,
            rally_id_input,
            player_input,
            other_player_input
        ]:
            item[item == PAD_IDX] = 0

        x = self.input_proj(torch.cat([
            self.embed_action(action_input),
            self.embed_pos(pos_input),
            self.embed_spin(spin_input),
            self.embed_strength(strength_input),
            self.embed_hand(hand_input),
            self.embed_strike(strike_input),
            self.embed_point(point_input),
            self.embed_sex(sex_input),
            self.embed_match(match_input),
            self.embed_number_game(number_game_input),
            self.embed_rally_id(rally_id_input),
            self.embed_player(player_input),
            self.embed_other_player(other_player_input),
            self.proj_num(num_feats)
        ], dim=-1)) + self.pos_embedding(strike_indices)

        for mamba in self.mamba_blocks:
            x = mamba(x)

        memory = self.transformer_neck(x, src_key_padding_mask=pad_mask)

        queries = self.task_queries.unsqueeze(0).repeat(b, 1, 1)

        task_features = self.transformer_decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=pad_mask
        )

        return (
            self.head_action(task_features[:, 0, :]),
            self.head_point(task_features[:, 1, :]),
            self.head_win(task_features[:, 2, :])
        )

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
        self.df = df
        self.is_test = is_test
        self.max_seq_len = max_seq_len
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        for _, group in self.df.groupby('rally_uid'):
            group = group.sort_values('strikeNumber').reset_index(drop=True)

            actions = group['actionId'].values
            positions = group['positionId'].values
            points = group['pointId'].values
            sexs = group['sex'].values
            matches = group['match'].values
            number_games = group['numberGame'].values
            rally_ids = group['rally_id'].values
            player_ids = group['gamePlayerId'].values
            other_player_ids = group['gamePlayerOtherId'].values

            num_feats = np.stack([
                group['scoreSelf'].values,
                group['scoreOther'].values,
                group['strikeNumber'].values
            ], axis=-1)

            winner = group['serverGetPoint'].values[0] if not self.is_test and 'serverGetPoint' in group.columns else 0
            spins = group['spinId'].values
            strengths = group['strengthId'].values
            hands = group['handId'].values
            strikes = group['strikeId'].values
            points = group['pointId'].values

            num_prefix_samples = max(1, len(group) - 1)
            winner_weight = 1.0 / num_prefix_samples

            for t in range(1, len(group)):
                start = max(0, t - self.max_seq_len)
                samples.append({
                    'action_seq': actions[start:t],
                    'pos_seq': positions[start:t],
                    'spin_seq': spins[start:t],
                    'strength_seq': strengths[start:t],
                    'hand_seq': hands[start:t],
                    'strike_seq': strikes[start:t],
                    'point_seq': points[start:t],
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

    def __len__(self): return len(self.samples)
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
        self.df = df
        self.max_seq_len = max_seq_len
        self.samples = self._build_inference_samples()

    def _build_inference_samples(self):
        samples = []

        for rally_uid, group in self.df.groupby('rally_uid'):
            group = group.sort_values('strikeNumber').reset_index(drop=True)

            actions = group['actionId'].values
            positions = group['positionId'].values
            points = group['pointId'].values
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
        pad_sequence([item[7] for item in batch], batch_first=True, padding_value=0.0),      
        pad_sequence([item[8] for item in batch], batch_first=True, padding_value=PAD_IDX),   
        pad_sequence([item[9] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[10] for item in batch], batch_first=True, padding_value=PAD_IDX), 
        pad_sequence([item[11] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[12] for item in batch], batch_first=True, padding_value=PAD_IDX), 
        pad_sequence([item[13] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        torch.stack([item[14] for item in batch]),                                           
        torch.stack([item[15] for item in batch]),                                          
        torch.stack([item[16] for item in batch]),
        torch.stack([item[17] for item in batch])                                          
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
        pad_sequence([item[7] for item in batch], batch_first=True, padding_value=0.0),       
        pad_sequence([item[8] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[9] for item in batch], batch_first=True, padding_value=PAD_IDX),   
        pad_sequence([item[10] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[11] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[12] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        pad_sequence([item[13] for item in batch], batch_first=True, padding_value=PAD_IDX),  
        [item[14] for item in batch]                                                          
    )

    
def train_one_fold(
    df_train,
    df_val,
    num_actions,
    num_positions,
    num_points,
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

    train_loader = DataLoader(
        RallyDataset(df_train),
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn_pad
    )
    val_loader = DataLoader(
        RallyDataset(df_val),
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn_pad
    )

    model = MambaTransformer(
        num_actions=num_actions,
        num_positions=num_positions,
        num_points=num_points,
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

    num_epochs = 30
    best_score = -float("inf")
    best_model_path = f"best_model_fold{fold_id}.pth"

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Fold {fold_id} Epoch {epoch+1}/{num_epochs}")

        for (
            act, pos, spin, strength, hand, strike, point, num,
            sex, match_id, number_game, rally_id, player_id, other_player_id,
            t_act, t_point, t_win, t_win_weight
        ) in pbar:
            act = act.to(device)
            pos = pos.to(device)
            spin = spin.to(device)
            strength = strength.to(device)
            hand = hand.to(device)
            strike = strike.to(device)
            point = point.to(device)
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

        with torch.no_grad():
            for (
                act, pos, spin, strength, hand, strike, point, num,
                sex, match_id, number_game, rally_id, player_id, other_player_id,
                t_act, t_point, t_win, t_win_weight
            ) in val_loader:
                act = act.to(device)
                pos = pos.to(device)
                spin = spin.to(device)
                strength = strength.to(device)
                hand = hand.to(device)
                strike = strike.to(device)
                point = point.to(device)
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

                preds = model(
                    act,
                    pos,
                    spin,
                    strength,
                    hand,
                    strike,
                    point,
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
                val_loss += loss.item()

                act_logits, point_logits, win_logits = preds

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

        win_threshold, win_threshold_score = find_best_binary_threshold(
            all_win_true,
            all_win_prob,
            sample_weight=all_win_weight,
            metric="balanced_accuracy"
        )

        total_score = 0.4 * act_f1 + 0.4 * point_f1 + 0.2 * win_auc

        print(
            f"Fold {fold_id} Epoch {epoch+1} 驗證 Loss: {avg_val:.4f} | "
            f"Act F1: {act_f1:.4f} | "
            f"Point F1: {point_f1:.4f} | "
            f"Win AUC: {win_auc:.4f} | "
            f"Win thr: {win_threshold:.3f} | "
            f"Win hardAUC/BA: {win_threshold_score:.4f} | "
            f"總分: {total_score:.4f}"
        )

        if total_score > best_score:
            best_score = total_score
            best_win_threshold = win_threshold
            best_win_threshold_score = win_threshold_score
            torch.save(model.state_dict(), best_model_path)
            with open(best_threshold_path, "w", encoding="utf-8") as f:
                f.write(f"{best_win_threshold:.8f}\n")
            print(
                f"Fold {fold_id} best score {best_score:.4f}，儲存 {best_model_path} | "
                f"serverGetPoint threshold={best_win_threshold:.3f}, "
                f"hardAUC/BA={best_win_threshold_score:.4f}"
            )

    return best_score, best_win_threshold, best_win_threshold_score

def run_group_kfold(
    train_df,
    num_actions,
    num_positions,
    num_points,
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
    groups = train_df["match"]

    fold_scores = []
    fold_thresholds = []
    fold_threshold_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=groups), start=1):
        df_train = train_df.iloc[train_idx].copy()
        df_val = train_df.iloc[val_idx].copy()

        score, fold_threshold, fold_threshold_score = train_one_fold(
            df_train=df_train,
            df_val=df_val,
            num_actions=num_actions,
            num_positions=num_positions,
            num_points=num_points,
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
        fold_thresholds.append(fold_threshold)
        fold_threshold_scores.append(fold_threshold_score)

    for i, score in enumerate(fold_scores, start=1):
        print(
            f"Fold {i}: {score:.4f} | "
            f"thr={fold_thresholds[i-1]:.3f} | "
            f"hardAUC/BA={fold_threshold_scores[i-1]:.4f}"
        )
    print(f"Mean CV Score: {np.mean(fold_scores):.4f}")
    print(f"Std CV Score: {np.std(fold_scores):.4f}")
    print(f"Mean threshold: {np.mean(fold_thresholds):.3f}")

    best_fold = int(np.argmax(fold_scores)) + 1
    best_threshold = fold_thresholds[best_fold - 1]
    print(f"Best fold: {best_fold}")
    print(f"Threshold used for submission: {best_threshold:.3f}")

    return fold_scores, best_fold, best_threshold

def generate_submission(
    test_df,
    test_csv_path,
    num_actions,
    num_positions,
    num_points,
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
    print(f"serverGetPoint threshold: {win_threshold:.3f}")

    model = MambaTransformer(
        num_actions=num_actions,
        num_positions=num_positions,
        num_points=num_points,
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
            act, pos, spin, strength, hand, strike, point, num,
            sex, match_id, number_game, rally_id, player_id, other_player_id,
            rally_uids
        ) in tqdm(loader, desc="Inference"):
            act = act.to(device)
            pos = pos.to(device)
            spin = spin.to(device)
            strength = strength.to(device)
            hand = hand.to(device)
            strike = strike.to(device)
            point = point.to(device)
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
                        "count =", bad.sum().item()
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
    sub_df.to_csv('data/submission_test_new.csv', index=False)
    print("生成 data/submission_test_new.csv")

if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"設備: {device}")

    train_csv = "data/train.csv"
    test_csv = "data/test.csv"
    model_path = "best_model_fold1.pth"

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"找不到訓練集: {train_csv}")

    if not os.path.exists(test_csv):
        raise FileNotFoundError(
            f"找不到測試集: {test_csv}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔: {model_path}")

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    NUM_ACTIONS = int(df_train["actionId"].max()) + 1
    NUM_POSITIONS = int(df_train["positionId"].max()) + 1
    NUM_POINTS = int(df_train["pointId"].max()) + 1
    NUM_SPINS = int(df_train["spinId"].max()) + 1
    NUM_STRENGTHS = int(df_train["strengthId"].max()) + 1
    NUM_HANDS = int(df_train["handId"].max()) + 1
    NUM_STRIKES = int(df_train["strikeId"].max()) + 1
    NUM_SEXES = int(df_train["sex"].max()) + 1

    NUM_MATCHES = int(df_train["match"].max()) + 1
    NUM_NUMBER_GAMES = int(df_train["numberGame"].max()) + 1
    NUM_RALLY_IDS = int(df_train["rally_id"].max()) + 1
    NUM_PLAYERS = max(
        int(df_train["gamePlayerId"].max()),
        int(df_train["gamePlayerOtherId"].max())
    ) + 1

    MAX_LEN = int(df_train["strikeNumber"].max()) + 20

    print("使用模型:", model_path)

    generate_submission(
        test_df=df_test,
        test_csv_path=test_csv,
        num_actions=NUM_ACTIONS,
        num_positions=NUM_POSITIONS,
        num_points=NUM_POINTS,
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
        model_path=model_path
    )
