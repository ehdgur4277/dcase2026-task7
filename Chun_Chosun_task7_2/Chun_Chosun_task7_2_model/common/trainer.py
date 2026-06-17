"""
공통 학습 모듈 — Dataset, LoRA, EWC, Evaluation, Train Loop
각 variant의 train.py에서 import하여 사용.
"""
import os, sys, time, json, copy, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as c
from backbone import CNN14Backbone


# ============================================================
# Dataset
# ============================================================
def pad_trunc(x, tlen):
    return np.concatenate([x, np.zeros(tlen - len(x))]) if len(x) < tlen else x[:tlen]


class AudioDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.is_train = is_train
        self.data, self.labels, self.fnames, self.domains = [], [], [], []
        for _, row in df.iterrows():
            y, _ = librosa.load(row['full_path'], sr=c.SAMPLE_RATE, mono=True)
            self.data.append(pad_trunc(y, c.CLIP_SAMPLES).astype(np.float32))
            self.labels.append(int(row['new_target']))
            self.fnames.append(row['filename'])
            self.domains.append(c.DOMAIN_TO_IDX[row['domain']])
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx]
        if self.is_train:
            x = x * np.random.uniform(0.7, 1.3)
        onehot = np.zeros(c.NUM_CLASSES, dtype=np.float32)
        onehot[self.labels[idx]] = 1.0
        return x, onehot, self.fnames[idx], self.domains[idx]

    def get_balanced_sampler(self):
        labels = np.array(self.labels)
        counts = np.bincount(labels, minlength=c.NUM_CLASSES) + 1
        w = 1.0 / counts
        if c.USE_BALANCED_SAMPLING:
            for hc in c.HARD_CLASSES:
                w[hc] *= c.HARD_CLASS_MULTIPLIER
        sample_w = w[labels]
        return WeightedRandomSampler(sample_w, len(sample_w), replacement=True)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# ============================================================
# LoRA
# ============================================================
class LoRAConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, rank=64, alpha=64.0):
        super().__init__()
        k = 3
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_ch * k * k) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_ch, rank))
        self.in_ch, self.out_ch = in_ch, out_ch

    def get_delta(self):
        return (self.lora_B @ self.lora_A).view(self.out_ch, self.in_ch, 3, 3) * self.scaling

    def forward(self, x, base_conv):
        return base_conv(x) + F.conv2d(x, self.get_delta(), padding=1)


class LoRACNN14(nn.Module):
    def __init__(self, backbone, rank=64, alpha=64.0):
        super().__init__()
        self.backbone = backbone
        block_configs = [(1, 64), (64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048)]
        self.lora_list = nn.ModuleList()
        for in_ch, out_ch in block_configs:
            self.lora_list.append(LoRAConv2d(in_ch, out_ch, rank, alpha))
            self.lora_list.append(LoRAConv2d(out_ch, out_ch, rank, alpha))

    def forward(self, x, use_spec_aug=False):
        b = self.backbone
        x = b.spectrogram_extractor(x)
        x = b.logmel_extractor(x)
        x = x.transpose(1, 3); x = b.bn0[0](x); x = x.transpose(1, 3)
        if use_spec_aug and self.training:
            x = b.spec_augmenter(x)
        blocks = [b.conv_block1, b.conv_block2, b.conv_block3,
                  b.conv_block4, b.conv_block5, b.conv_block6]
        for i, block in enumerate(blocks):
            x = self.lora_list[i*2](x, block.conv1)
            x = F.relu_(block.bnF[0](x))
            x = self.lora_list[i*2+1](x, block.conv2)
            x = F.relu_(block.bnS[0](x))
            x = F.avg_pool2d(x, (2, 2))
            x = F.dropout(x, 0.2, training=self.training)
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2); x2 = torch.mean(x, dim=2)
        feat = x1 + x2
        return b.fc(feat), feat

    def get_all_lora_A(self):
        return [m.lora_A for m in self.lora_list]


# ============================================================
# EWC
# ============================================================
def compute_fisher(model, train_df, device, n_samples=500):
    model.eval()
    ds = AudioDataset(train_df, is_train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=c.NUM_WORKERS)
    criterion = nn.CrossEntropyLoss()

    fisher = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            fisher[n] = torch.zeros_like(p)

    count = 0
    for audio, target, _, _ in loader:
        if count >= n_samples:
            break
        audio = audio.float().to(device)
        target_idx = target.float().to(device).argmax(-1)
        model.zero_grad()
        logits, _ = model(audio)
        loss = criterion(logits, target_idx)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data ** 2
        count += 1

    for n in fisher:
        fisher[n] /= count
    return fisher


def ewc_loss(model, fisher, d2_params, lam):
    loss = 0.0
    for n, p in model.named_parameters():
        if n in fisher:
            loss = loss + (fisher[n] * (p - d2_params[n]) ** 2).sum()
    return lam * loss


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(model, test_df, device, criterion=None):
    """
    criterion=None  → float (macro accuracy only)
    criterion=<fn>  → (macro_accuracy, avg_loss)
    """
    model.eval()
    ds     = AudioDataset(test_df, is_train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=c.NUM_WORKERS)
    tta_gains = c.TTA_GAINS if c.USE_TTA else [1.0]
    preds, targets = [], []
    total_loss, n  = 0.0, 0

    for audio, target, _, _ in loader:
        audio      = audio.float().to(device)
        target_idx = target.float().to(device).argmax(-1)

        probs_list = []
        for g in tta_gains:
            logits, _ = model(audio * g)
            probs_list.append(F.softmax(logits, -1))
        final = torch.stack(probs_list).mean(0)
        preds.append(final.argmax(-1).item())
        targets.append(target_idx.item())

        if criterion is not None:
            logits_clean, _ = model(audio)
            total_loss += criterion(logits_clean, target_idx).item()
        n += 1

    preds, targets = np.array(preds), np.array(targets)
    # Macro accuracy: 클래스별 accuracy 평균 (DCASE Task7 공식 지표)
    class_accs = []
    for cls in range(c.NUM_CLASSES):
        mask = targets == cls
        if mask.sum() > 0:
            class_accs.append(float((preds[mask] == targets[mask]).mean()))
    acc = float(np.mean(class_accs) * 100)
    if criterion is not None:
        return acc, total_loss / n
    return acc


# ============================================================
# Training Loop
# ============================================================
def train_phase(model, train_df, device, log_fn, phase_name='',
                lr=1e-3, ortho_pairs=None, teacher_model=None,
                fisher=None, d2_params=None, mode_cfg=None):
    log_fn(f"\n========== Phase: {phase_name} ==========")
    ds = AudioDataset(train_df, is_train=True)
    log_fn(f"  Train samples: {len(ds)}")

    if c.USE_BALANCED_SAMPLING:
        loader = DataLoader(ds, batch_size=c.BATCH_SIZE, sampler=ds.get_balanced_sampler(),
                            num_workers=c.NUM_WORKERS, pin_memory=True, drop_last=True)
    else:
        loader = DataLoader(ds, batch_size=c.BATCH_SIZE, shuffle=True,
                            num_workers=c.NUM_WORKERS, pin_memory=True, drop_last=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    log_fn(f"  Trainable params: {n_params:,}, LR: {lr}")
    log_fn(f"  EWC: {'ON' if fisher else 'OFF'}, KD: {'ON' if teacher_model else 'OFF'}")

    optim = torch.optim.AdamW(trainable, lr=lr, weight_decay=c.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=c.EPOCHS, eta_min=1e-6)
    criterion = FocalLoss(gamma=c.FOCAL_GAMMA) if c.USE_FOCAL_LOSS else nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(1, c.EPOCHS + 1):
        model.train()
        model.backbone.set_bn_eval()
        sum_cls, sum_ewc, sum_orth, sum_kd, sum_feat, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0

        for audio, target, _, _ in loader:
            audio = audio.float().to(device)
            target_idx = target.float().to(device).argmax(-1)

            # Mixup
            if c.USE_MIXUP and np.random.random() < 0.5:
                lam = np.random.beta(c.MIXUP_ALPHA, c.MIXUP_ALPHA)
                idx = torch.randperm(audio.size(0), device=device)
                audio_mixed = lam * audio + (1 - lam) * audio[idx]
                ya, yb = target_idx, target_idx[idx]
            else:
                audio_mixed = audio; ya = yb = target_idx; lam = 1.0

            logits, feat = model(audio_mixed, use_spec_aug=c.USE_SPEC_AUG)
            loss_cls = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            loss = loss_cls

            # Orthogonal loss
            if ortho_pairs is not None:
                ortho_lam = mode_cfg.get('ortho_lambda', c.ORTHO_LAMBDA) if mode_cfg else c.ORTHO_LAMBDA
                orth = 0.0
                for cur_A, old_A in ortho_pairs:
                    orth = orth + torch.abs(cur_A @ old_A.detach().T).sum()
                loss = loss + ortho_lam * orth
                sum_orth += float(orth)

            # KD loss (logit + feature)
            if teacher_model is not None:
                kd_alpha = mode_cfg.get('kd_alpha', c.KD_ALPHA) if mode_cfg else c.KD_ALPHA
                kd_temp = mode_cfg.get('kd_temp', c.KD_TEMP) if mode_cfg else c.KD_TEMP
                kd_fw = mode_cfg.get('kd_feat_weight', c.KD_FEAT_WEIGHT) if mode_cfg else c.KD_FEAT_WEIGHT
                teacher_model.eval()
                with torch.no_grad():
                    t_logits, t_feat = teacher_model(audio_mixed)
                kd_loss = F.kl_div(
                    F.log_softmax(logits / kd_temp, dim=-1),
                    F.softmax(t_logits / kd_temp, dim=-1),
                    reduction='batchmean') * (kd_temp ** 2)
                loss = loss + kd_alpha * kd_loss
                sum_kd += float(kd_loss)
                feat_loss = F.mse_loss(feat, t_feat)
                loss = loss + kd_fw * feat_loss
                sum_feat += float(feat_loss)

            # EWC loss
            if fisher is not None and d2_params is not None:
                ewc_lam = mode_cfg.get('ewc_lambda', c.EWC_LAMBDA) if mode_cfg else c.EWC_LAMBDA
                l_ewc = ewc_loss(model, fisher, d2_params, ewc_lam)
                loss = loss + l_ewc
                sum_ewc += float(l_ewc)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            sum_cls += loss_cls.item(); n += 1

        sched.step()
        if epoch % 10 == 0 or epoch == 1 or epoch == c.EPOCHS:
            extra = ''
            if ortho_pairs: extra += f' orth={sum_orth/n:.4f}'
            if teacher_model: extra += f' kd={sum_kd/n:.4f} feat={sum_feat/n:.4f}'
            if fisher: extra += f' ewc={sum_ewc/n:.2f}'
            log_fn(f"  Epoch {epoch:3d}/{c.EPOCHS} | cls={sum_cls/n:.4f}{extra} "
                   f"lr={optim.param_groups[0]['lr']:.6f}")


# ============================================================
# Early Stopping (maximize val_acc)
# ============================================================
class EarlyStopping:
    """
    val_acc 기반 early stopping (최대화).
    min_epochs 이전에는 stop 신호를 내지 않음.
    best weights를 CPU deepcopy로 보존하여 restore 지원.
    """
    def __init__(self, patience=20, min_epochs=50, min_delta=1e-4):
        self.patience   = patience
        self.min_epochs = min_epochs
        self.min_delta  = min_delta
        self.best_score = -float('inf')
        self.best_epoch = 0
        self.counter    = 0
        self._best_state = None

    def step(self, val_acc, model, epoch, meta=None):
        improved = val_acc > self.best_score + self.min_delta
        if improved:
            self.best_score  = val_acc
            self.best_epoch  = epoch
            self.counter     = 0
            self._best_state = copy.deepcopy(
                {k: v.cpu() for k, v in model.state_dict().items()}
            )
        else:
            self.counter += 1
        stop = (epoch >= self.min_epochs) and (self.counter >= self.patience)
        return improved, stop

    def restore(self, model):
        if self._best_state is not None:
            model.load_state_dict(self._best_state, strict=True)


# ============================================================
# K-Fold / Val Split
# ============================================================
def make_kfold_split(df, fold_k, k_total=None, seed=None):
    """
    StratifiedKFold로 df 분할. fold_k를 val, 나머지를 train으로 반환.
    클래스별 독립 분할 → 모든 fold가 동일한 클래스 분포를 가짐.
    """
    if k_total is None: k_total = c.KFOLD_K
    if seed    is None: seed    = c.KFOLD_SEED
    rng = np.random.RandomState(seed)

    fold_assignments = np.full(len(df), -1, dtype=int)
    for cls in range(c.NUM_CLASSES):
        idx = np.where(df['new_target'].values == cls)[0]
        idx = idx[rng.permutation(len(idx))]
        for i, pos in enumerate(idx):
            fold_assignments[pos] = i % k_total

    val_mask = fold_assignments == fold_k
    tr_idx   = np.where(~val_mask)[0]
    val_idx  = np.where(val_mask)[0]
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def stratified_val_split(df, val_ratio=None, seed=None):
    """new_target 기준 stratified train/val split."""
    if val_ratio is None: val_ratio = 0.15
    if seed      is None: seed      = c.KFOLD_SEED
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for cls in range(c.NUM_CLASSES):
        idx = df.index[df['new_target'] == cls].tolist()
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return df.loc[train_idx].reset_index(drop=True), df.loc[val_idx].reset_index(drop=True)
