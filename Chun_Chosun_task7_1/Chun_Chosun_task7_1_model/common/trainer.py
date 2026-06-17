"""
공통 학습 모듈 — Dataset, LoRA, Train Loop
각 variant의 train.py에서 import하여 사용.
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as c


# ============================================================
# 디코딩 캐시 (path → waveform)
# ============================================================
_AUDIO_CACHE = {}


def _load_audio(path):
    y = _AUDIO_CACHE.get(path)
    if y is None:
        y, _ = librosa.load(path, sr=c.SAMPLE_RATE, mono=True)
        y = y.astype(np.float32)
        _AUDIO_CACHE[path] = y
    return y


# ============================================================
# Dataset
# ============================================================
def pad_trunc(x, tlen):
    return np.concatenate([x, np.zeros(tlen - len(x))]) if len(x) < tlen else x[:tlen]


def chunk_audio(y, sr=None, chunk_seconds=None, min_seconds=None):
    """
    non-overlap chunk_seconds 분할. 마지막 chunk가 min_seconds 미만이면 폐기,
    이상이면 zero-pad. 1차원 waveform → [chunk, samples] 리스트.
    """
    sr = sr or c.SAMPLE_RATE
    chunk_seconds = c.CHUNK_SECONDS if chunk_seconds is None else chunk_seconds
    min_seconds = c.CHUNK_MIN_SECONDS if min_seconds is None else min_seconds
    chunk_len = int(round(chunk_seconds * sr))
    min_len = int(round(min_seconds * sr))

    chunks = []
    for start in range(0, len(y), chunk_len):
        seg = y[start:start + chunk_len]
        if len(seg) < min_len:
            continue
        if len(seg) < chunk_len:
            seg = pad_trunc(seg, chunk_len)
        chunks.append(seg.astype(np.float32))
    if not chunks:
        chunks.append(pad_trunc(y, chunk_len).astype(np.float32))
    return chunks


class AudioDataset(Dataset):
    """
    is_train=True  : chunk_audio로 4s non-overlap 분할 → 각 chunk 독립 sample.
    is_train=False : 가변 길이(truncate 없음) → eval loader batch_size=1 필수.
    """
    def __init__(self, df, is_train=True):
        self.is_train = is_train
        self.labels, self.fnames, self.domains = [], [], []
        if is_train:
            chunks_all = []
            for _, row in df.iterrows():
                y = _load_audio(row['full_path'])
                for seg in chunk_audio(y):
                    chunks_all.append(seg)
                    self.labels.append(int(row['new_target']))
                    self.fnames.append(row['filename'])
                    self.domains.append(c.DOMAIN_TO_IDX[row['domain']])
            self._samples = (np.stack(chunks_all).astype(np.float32)
                             if chunks_all else np.zeros((0, c.CLIP_SAMPLES), np.float32))
        else:
            self.paths = []
            for _, row in df.iterrows():
                _load_audio(row['full_path'])
                self.paths.append(row['full_path'])
                self.labels.append(int(row['new_target']))
                self.fnames.append(row['filename'])
                self.domains.append(c.DOMAIN_TO_IDX[row['domain']])

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        if self.is_train:
            x = self._samples[idx].copy()
            x = x * np.random.uniform(c.GAIN_AUG_LOW, c.GAIN_AUG_HIGH)
        else:
            y = _load_audio(self.paths[idx])
            x = y if len(y) >= c.CLIP_SAMPLES else pad_trunc(y, c.CLIP_SAMPLES)
            x = np.asarray(x, dtype=np.float32).copy()
        onehot = np.zeros(c.NUM_CLASSES, dtype=np.float32)
        onehot[self.labels[idx]] = 1.0
        return x, onehot, self.fnames[idx], self.domains[idx]

    def get_balanced_sampler(self):
        labels = np.array(self.labels)
        counts = np.bincount(labels, minlength=c.NUM_CLASSES).astype(np.float64)
        w = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
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


# ============================================================
# Training Loop
# ============================================================
def train_phase(model, train_df, device, log_fn, phase_name='',
                lr=1e-3, ortho_pairs=None, teacher_model=None,
                mode_cfg=None):
    log_fn(f"\n========== Phase: {phase_name} ==========")
    ds = AudioDataset(train_df, is_train=True)
    log_fn(f"  Train samples: {len(ds)}")

    if c.USE_BALANCED_SAMPLING:
        loader = DataLoader(ds, batch_size=c.BATCH_SIZE, sampler=ds.get_balanced_sampler(),
                            num_workers=4, pin_memory=True, drop_last=True)
    else:
        loader = DataLoader(ds, batch_size=c.BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    log_fn(f"  Trainable params: {n_params:,}, LR: {lr}")
    log_fn(f"  KD: {'ON' if teacher_model else 'OFF'}")

    optim = torch.optim.AdamW(trainable, lr=lr, weight_decay=c.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=c.EPOCHS, eta_min=1e-6)
    criterion = FocalLoss(gamma=c.FOCAL_GAMMA) if c.USE_FOCAL_LOSS else nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(1, c.EPOCHS + 1):
        model.train()
        model.backbone.set_bn_eval()
        sum_cls, sum_orth, sum_kd, sum_feat, n = 0.0, 0.0, 0.0, 0.0, 0

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
            log_fn(f"  Epoch {epoch:3d}/{c.EPOCHS} | cls={sum_cls/n:.4f}{extra} "
                   f"lr={optim.param_groups[0]['lr']:.6f}")
