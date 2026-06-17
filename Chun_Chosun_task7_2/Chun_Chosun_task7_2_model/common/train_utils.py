"""
common/train_utils.py — D2/D3 단일 phase 학습 로직 + 학습 고정 설정.

exp04 (COND_NO_GAIN=1) 설정으로 고정:
  COND_CLASSES(baby_cry, telephone_ringing) 에만 gain aug 제거,
  mixup / spec-aug 는 전체 클래스 유지.
  D2 rank=128, D3 rank=64, P_SVD=0.99, D2/D3 epochs=150.

진입점(main.py, single_main.py)은 데이터프레임을 준비해 train_d2 / train_d3 를
호출하기만 하고, 체크포인트 저장 위치는 각 진입점이 결정한다.
"""
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config as c
from backbone import CNN14Backbone
from trainer import AudioDataset, FocalLoss, evaluate, train_phase
from model import (
    DeltaOrthoMultiLoRACNN14, D2_LORA_RANK, D3_LORA_RANK, freeze_module,
)

# ── 학습 고정 설정 ──────────────────────────────────────────────────────────────
P_SVD_D1 = 0.99
P_SVD_D2 = 0.99
D2_EPOCHS = int(os.environ.get('D2_EPOCHS', '150'))
D3_EPOCHS = int(os.environ.get('D3_EPOCHS', '150'))

# D1 backbone 사전학습 체크포인트 (외부 자산) — 환경변수로 재정의 가능.
_DEFAULT_D1 = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'checkpoint', 'checkpoint_D1.pth'))
D1_CKPT = os.environ.get('D1_CKPT', _DEFAULT_D1)

COND_CLASSES = [1, 7]    # baby_cry=1, telephone_ringing=7
COND_NO_GAIN = True      # exp04
COND_NO_MIXUP = False    # exp04
COND_NO_SPECAUG = False  # exp04

D1_CFG = {'kd_alpha': 0.1, 'kd_temp': 2.0, 'kd_feat_weight': 0.0}
D2_CFG = {
    'kd_alpha':       c.MODE_CONFIG['cllora']['kd_alpha'],
    'kd_temp':        c.MODE_CONFIG['cllora']['kd_temp'],
    'kd_feat_weight': c.MODE_CONFIG['cllora']['kd_feat_weight'],
}
LR_D3 = c.MODE_CONFIG['cllora']['lr_d3']


def set_seed(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')


def build_model(device):
    """D1 backbone 로드 → freeze → LoRA 모델 생성."""
    backbone = CNN14Backbone(nb_tasks=3)
    backbone.load_d1_checkpoint(D1_CKPT)
    backbone.freeze_backbone()
    backbone.to(device)
    return DeltaOrthoMultiLoRACNN14(backbone).to(device)


# ── CondAugDataset ─────────────────────────────────────────────────────────────
class CondAugDataset(AudioDataset):
    """COND_CLASSES(baby_cry, telephone_ringing)에 대해 gain 미적용."""
    def __getitem__(self, idx):
        x = self.data[idx].copy()
        label = self.labels[idx]
        if self.is_train:
            if not (COND_NO_GAIN and label in COND_CLASSES):
                x = x * np.random.uniform(0.7, 1.3)
        onehot = np.zeros(c.NUM_CLASSES, dtype=np.float32)
        onehot[label] = 1.0
        return x, onehot, self.fnames[idx], self.domains[idx]


# ── D3 학습 루프 (CosineFeatKD + exp04 조건부 aug) ──────────────────────────────
def train_d3_cond_aug(model, train_df, device, log_fn, phase_name='',
                      lr=1e-3, teacher_model=None, mode_cfg=None):
    log_fn(f"\n========== Phase: {phase_name} ==========")
    log_fn(f"  COND_CLASSES={COND_CLASSES}  NO_GAIN={COND_NO_GAIN}  "
           f"NO_MIXUP={COND_NO_MIXUP}  NO_SPECAUG={COND_NO_SPECAUG}")

    ds = CondAugDataset(train_df, is_train=True)
    log_fn(f"  Train samples: {len(ds)}")

    num_workers = int(os.environ.get('DATALOADER_WORKERS', str(c.NUM_WORKERS)))
    if c.USE_BALANCED_SAMPLING:
        loader = DataLoader(ds, batch_size=c.BATCH_SIZE, sampler=ds.get_balanced_sampler(),
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    else:
        loader = DataLoader(ds, batch_size=c.BATCH_SIZE, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    log_fn(f"  Trainable params: {sum(p.numel() for p in trainable):,}, LR: {lr}")

    optim = torch.optim.AdamW(trainable, lr=lr, weight_decay=c.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=c.EPOCHS, eta_min=1e-6)
    criterion = FocalLoss(gamma=c.FOCAL_GAMMA) if c.USE_FOCAL_LOSS else nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(1, c.EPOCHS + 1):
        model.train()
        model.backbone.set_bn_eval()
        sum_cls, sum_kd, sum_feat, n = 0.0, 0.0, 0.0, 0

        for audio, target, _, _ in loader:
            audio      = audio.float().to(device)
            target_idx = target.float().to(device).argmax(-1)
            B = audio.size(0)

            if c.USE_MIXUP and np.random.random() < 0.5:
                lam = np.random.beta(c.MIXUP_ALPHA, c.MIXUP_ALPHA)
                idx = torch.randperm(B, device=device)
                audio_mixed = lam * audio + (1 - lam) * audio[idx]
                ya, yb = target_idx, target_idx[idx]
                # COND_NO_MIXUP=False → mixup 항상 적용 (exp04)
            else:
                audio_mixed, ya, yb, lam = audio, target_idx, target_idx, 1.0

            # COND_NO_SPECAUG=False → 전체 specaug (exp04)
            logits, feat = model(audio_mixed, use_spec_aug=c.USE_SPEC_AUG)

            loss_cls = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            loss = loss_cls

            if teacher_model is not None:
                kd_alpha = mode_cfg.get('kd_alpha', c.KD_ALPHA)   if mode_cfg else c.KD_ALPHA
                kd_temp  = mode_cfg.get('kd_temp',  c.KD_TEMP)    if mode_cfg else c.KD_TEMP
                kd_fw    = mode_cfg.get('kd_feat_weight', c.KD_FEAT_WEIGHT) if mode_cfg else c.KD_FEAT_WEIGHT
                teacher_model.eval()
                with torch.no_grad():
                    t_logits, t_feat = teacher_model(audio_mixed)
                kd_loss = F.kl_div(
                    F.log_softmax(logits / kd_temp, dim=-1),
                    F.softmax(t_logits / kd_temp, dim=-1),
                    reduction='batchmean',
                ) * (kd_temp ** 2)
                feat_loss = 1.0 - F.cosine_similarity(feat, t_feat, dim=1, eps=1e-8).mean()
                loss = loss + kd_alpha * kd_loss + kd_fw * feat_loss
                sum_kd   += float(kd_loss)
                sum_feat += float(feat_loss)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            sum_cls += loss_cls.item()
            n += 1

        sched.step()
        if epoch % 10 == 0 or epoch == 1 or epoch == c.EPOCHS:
            extra = f' kd={sum_kd/n:.4f} cos={sum_feat/n:.4f}' if teacher_model else ''
            log_fn(f"  Epoch {epoch:3d}/{c.EPOCHS} | cls={sum_cls/n:.4f}{extra} "
                   f"lr={optim.param_groups[0]['lr']:.6f}")


# ── Phase 함수 (진입점에서 호출) ─────────────────────────────────────────────────
def train_d2(model, df_d2_tr, df_d2_te, device, log, label='', eval_fn=None):
    """
    D2 LoRA 학습 (D1-KD + D1 직교투영). 학습 후 D2@D2(macro) 반환.
    model 은 in-place 로 D2 LoRA 가 학습된 상태가 된다.
    eval_fn(model, df, device): 평가 프로토콜 주입 (기본 evaluate=crop+TTA).
    """
    eval_fn = eval_fn or evaluate
    log(f'\n[train_d2] {label}  train={len(df_d2_tr)}  test={len(df_d2_te)}')
    log(f'  D2_rank={D2_LORA_RANK}  p_svd_d1={P_SVD_D1}  epochs={D2_EPOCHS}')

    d1_teacher = freeze_module(copy.deepcopy(model))
    d1_teacher.active_domain_count = 0

    model.build_d1_bases(projection_rank=D2_LORA_RANK, p_svd=P_SVD_D1)
    model.active_domain_count = 1
    model.set_d2_trainable()
    model.install_orth_hooks(domain_idx=0)

    c.EPOCHS = D2_EPOCHS
    try:
        train_phase(
            model, df_d2_tr, device, log,
            phase_name=f'D2 {label} (LoRA_D2 + D1-KD + Orth)',
            lr=c.LR_D2, teacher_model=d1_teacher, mode_cfg=D1_CFG,
        )
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            log(f'  [WARNING] OOM: {e} — 현재 상태 유지')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise

    model.clear_orth_hooks()
    model.active_domain_count = 1
    d2_at_d2 = eval_fn(model, df_d2_te, device)
    log(f'  D2@D2 (macro): {d2_at_d2:.2f}%')
    return d2_at_d2


def train_d3(model, d2_at_d2, df_d3_tr, df_d3_te, df_d2_te, device, log,
             label='', df_d3_val=None, eval_fn=None):
    """
    D3 LoRA 학습 (D2-KD + DeltaGP + CosineFeatKD + CondAug). 메트릭 dict 반환.
    df_d3_val 가 주어지면 val_macro 도 계산해 dict 에 포함.
    eval_fn(model, df, device): 평가 프로토콜 주입 (기본 evaluate=crop+TTA).
    """
    eval_fn = eval_fn or evaluate
    log(f'\n[train_d3] {label}  train={len(df_d3_tr)}')
    log(f'  D3_rank={D3_LORA_RANK}  p_svd_d2={P_SVD_D2}  epochs={D3_EPOCHS}')

    d2_teacher = freeze_module(copy.deepcopy(model))
    d2_teacher.active_domain_count = 1

    model.build_d1_bases(projection_rank=D2_LORA_RANK, p_svd=P_SVD_D1)
    model.build_d2_delta_bases(p_svd=P_SVD_D2)
    model.active_domain_count = 2
    model.set_d3_trainable()
    model.install_d1_d2_delta_hooks(domain_idx=1)

    c.EPOCHS = D3_EPOCHS
    try:
        train_d3_cond_aug(
            model, df_d3_tr, device, log,
            phase_name=f'D3 {label} (LoRA_D3 + D2-KD + DeltaGP + CosineFeatKD + CondAug)',
            lr=LR_D3, teacher_model=d2_teacher, mode_cfg=D2_CFG,
        )
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            log(f'  [WARNING] OOM: {e} — 현재 상태 유지')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise

    model.clear_orth_hooks()
    model.active_domain_count = 2
    d2_at_d3 = eval_fn(model, df_d2_te, device)
    d3_at_d3 = eval_fn(model, df_d3_te, device)
    acc = (d2_at_d3 + d3_at_d3) / 2.0
    fr = round(d2_at_d2 - d2_at_d3, 4) if d2_at_d2 is not None else None

    metrics = {
        'D2@D2': d2_at_d2, 'D2@D3': d2_at_d3, 'D3@D3': d3_at_d3,
        'Acc': acc, 'Fr': fr,
    }
    if df_d3_val is not None:
        metrics['val_macro'] = eval_fn(model, df_d3_val, device)

    log(f'  D2@D2={d2_at_d2}  D2@D3={d2_at_d3:.2f}%  D3@D3={d3_at_d3:.2f}%  '
        f'Acc={acc:.2f}%  Fr={fr}%p'
        + (f'  val_macro={metrics["val_macro"]:.2f}%' if df_d3_val is not None else ''))
    return metrics
