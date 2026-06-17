"""
covforget_kfold 공통 설정 — YAML 기반 config 모듈
exp13_CovForget 모듈(dataset.py/train.py)이 import config 로 사용하므로
exp13 이 참조하는 속성(GAIN_AUG_*, P_SVD_D2 등)을 모두 노출해야 한다.
"""
import os
import random
import datetime

import numpy as np
import pandas as pd
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CONFIG_PATH = os.environ.get('CONFIG', os.path.join(BASE_DIR, 'config.yaml'))
with open(_CONFIG_PATH, 'r') as _f:
    CFG = yaml.safe_load(_f)

# ── Experiment ────────────────────────────────────────────────────────────────
MODE       = CFG.get('mode', '')
_exp       = CFG.get('exp', {})
EXP_ID     = _exp.get('id',     None)
EXP_METHOD = _exp.get('method', None)
EXP_DATE   = _exp.get('date',   None)
EXP_NOTES  = _exp.get('notes',  None)

# ── Audio ─────────────────────────────────────────────────────────────────────
_audio       = CFG.get('audio', {})
SAMPLE_RATE  = int(_audio.get('sample_rate', 32000))
MEL_BINS     = int(_audio.get('mel_bins',    64))
FMIN         = int(_audio.get('fmin',        50))
FMAX         = int(_audio.get('fmax',        14000))
WIN_SIZE     = int(_audio.get('win_size',    1024))
HOP_SIZE     = int(_audio.get('hop_size',    320))
NUM_CLASSES  = int(_audio.get('num_classes', 10))

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED      = int(CFG.get('seed',      42))
VAL_SEED  = int(CFG.get('val_seed',  42))
VAL_RATIO = float(CFG.get('val_ratio', 0.15))

# ── LoRA / DeltaGP ranks ──────────────────────────────────────────────────────
_lora      = CFG.get('lora', {})
LORA_ALPHA = float(_lora.get('alpha',   64.0))
TUNE_BN    = bool(_lora.get('tune_bn', False))
D2_RANK_A  = list(_lora.get('d2_rank_A', [4,  39, 44, 48, 79, 124]))
D2_RANK_B  = list(_lora.get('d2_rank_B', [3,   3,  1, 12,  7,   4]))
D3_RANK_A  = list(_lora.get('d3_rank_A', [9,  29, 20, 30, 40,  42]))
D3_RANK_B  = list(_lora.get('d3_rank_B', [17,  8, 18, 20, 38,  38]))

# ── Training ──────────────────────────────────────────────────────────────────
_train       = CFG.get('train', {})
BATCH_SIZE   = int(_train.get('batch_size',   32))
WEIGHT_DECAY = float(_train.get('weight_decay', 1e-5))
LR_D2        = float(_train.get('lr_d2',       1e-3))
LR_D3        = float(_train.get('lr_d3',       2.5e-4))
D2_EPOCHS    = int(_train.get('d2_epochs',    200))
D3_EPOCHS    = int(_train.get('d3_epochs',    400))
GRAD_CLIP    = float(_train.get('grad_clip',   1.0))
LR_ETA_MIN   = float(_train.get('lr_eta_min',  1e-6))
LOG_EVERY    = int(_train.get('log_every',     10))
NUM_WORKERS  = int(os.environ.get('DATALOADER_WORKERS',
                                   str(_train.get('num_workers', 0))))

# ── Early Stopping ────────────────────────────────────────────────────────────
_es           = CFG.get('early_stop', {})
D2_PATIENCE   = int(_es.get('d2_patience',   20))
D3_PATIENCE   = int(_es.get('d3_patience',   45))
D2_MIN_EPOCHS = int(_es.get('d2_min_epochs', 30))
D3_MIN_EPOCHS = int(_es.get('d3_min_epochs', 80))
MIN_DELTA     = float(_es.get('min_delta',   1e-3))

# ── Knowledge Distillation ────────────────────────────────────────────────────
_kd   = CFG.get('kd', {})
KD_D1 = dict(_kd.get('d1', {'kd_alpha': 0.1, 'kd_temp': 2.0, 'kd_feat_weight': 0.2}))
KD_D2 = dict(_kd.get('d2', {'kd_alpha': 0.7, 'kd_temp': 1.5, 'kd_feat_weight': 0.2}))

# ── DeltaGP SVD ───────────────────────────────────────────────────────────────
_pace              = CFG.get('pace', {})
P_SVD_D1           = float(_pace.get('p_svd_d1', 0.99))
P_SVD_D2           = float(_pace.get('p_svd_d2', 0.9))
P_SVD_D3           = float(_pace.get('p_svd_d3', 0.9))
P_SVD_D2_PER_BLOCK = list(_pace.get('p_svd_d2_per_block',
                                     [0.90, 0.85, 0.88, 0.92, 0.93, 0.95]))
SVD_EPS            = float(_pace.get('svd_eps', 1e-7))

# ── Focal Loss ────────────────────────────────────────────────────────────────
_focal          = CFG.get('focal', {})
USE_FOCAL_LOSS  = bool(_focal.get('use',             True))
FOCAL_GAMMA     = float(_focal.get('gamma',           2.0))
LABEL_SMOOTHING = float(_focal.get('label_smoothing', 0.1))

# ── Mixup ─────────────────────────────────────────────────────────────────────
_mixup      = CFG.get('mixup', {})
USE_MIXUP   = bool(_mixup.get('use',   True))
MIXUP_ALPHA = float(_mixup.get('alpha', 0.3))
MIXUP_PROB  = float(_mixup.get('prob',  0.5))

# ── Spec Augmentation ─────────────────────────────────────────────────────────
USE_SPEC_AUG = bool(CFG.get('spec_aug', {}).get('use', True))

# ── Gain Augmentation (exp13 dataset.py 참조) ─────────────────────────────────
_gain_aug     = CFG.get('gain_aug', {})
GAIN_AUG_LOW  = float(_gain_aug.get('low',  0.7))
GAIN_AUG_HIGH = float(_gain_aug.get('high', 1.3))
# task7_4: 이 클래스(new_target 인덱스)들은 gain 증강 제외 (baby_cry=1, telephone_ringing=9)
COND_NO_GAIN_CLASSES = list(CFG.get('cond_no_gain_classes', []))

# ── Dropout ───────────────────────────────────────────────────────────────────
DROPOUT = float(CFG.get('dropout', 0.2))

# ── TTA ───────────────────────────────────────────────────────────────────────
_tta      = CFG.get('tta', {})
USE_TTA   = bool(_tta.get('use',   False))
TTA_GAINS = list(_tta.get('gains', [1.0, 0.85, 1.15]))

# ── Sampling ──────────────────────────────────────────────────────────────────
_sampling             = CFG.get('sampling', {})
USE_BALANCED_SAMPLING = bool(_sampling.get('use_balanced',          True))
HARD_CLASSES          = list(_sampling.get('hard_classes',          [0, 4]))
HARD_CLASS_MULTIPLIER = float(_sampling.get('hard_class_multiplier', 1.0))

# ── Audio chunk ───────────────────────────────────────────────────────────────
_chunk            = CFG.get('chunk', {})
CHUNK_SECONDS     = float(_chunk.get('seconds',     4.0))
CHUNK_MIN_SECONDS = float(_chunk.get('min_seconds', 1.0))
CHUNK_SAMPLES     = int(round(CHUNK_SECONDS * SAMPLE_RATE))
MIN_CHUNK_SAMPLES = int(round(CHUNK_MIN_SECONDS * SAMPLE_RATE))
CLIP_SAMPLES      = CHUNK_SAMPLES

# ── Domain / Class ────────────────────────────────────────────────────────────
DOMAIN_TO_IDX = {'D1': 0, 'D2': 1, 'D3': 2}
CLASS_LABELS  = {
    'alarm': 0, 'baby_cry': 1, 'dog_bark': 2, 'engine': 3, 'fire': 4,
    'footsteps': 5, 'knocking': 6, 'piano': 7, 'speech': 8, 'telephone_ringing': 9,
}

# ── K-Fold ────────────────────────────────────────────────────────────────────
KFOLD_K    = 5
KFOLD_SEED = 42

# ── D1 Checkpoint ─────────────────────────────────────────────────────────────
def _resolve_d1_ckpt():
    env = os.environ.get('D1_CKPT', None)
    if env and os.path.exists(env):
        return env
    for sub in ('checkpoint', 'checkpoints'):
        local = os.path.join(BASE_DIR, sub, 'checkpoint_D1.pth')
        if os.path.exists(local):
            return local
    workspace = '/workspace/dcase_kfold/checkpoint_D1.pth'
    if os.path.exists(workspace):
        return workspace
    yaml_ckpt = CFG.get('d1_ckpt', None)
    if yaml_ckpt and os.path.exists(yaml_ckpt):
        return yaml_ckpt
    return yaml_ckpt or workspace

D1_CKPT = _resolve_d1_ckpt()

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_DIR = os.environ.get(
    'TASK7_LORA_DATASET_DIR',
    CFG.get('data_dir', '/workspace/dcase_kfold/Dcase_Dataset/dcase_dataset')
)


def _build_meta():
    metadata_dir = os.path.join(DATASET_DIR, 'metadata')
    splits = {
        'train': {'D2': 'd2-dev-train.csv', 'D3': 'd3-dev-train.csv'},
        'test':  {'D2': 'd2-dev-test.csv',  'D3': 'd3-dev-test.csv'},
    }
    audio_dirs = {
        'train': {'D2': 'd2-dev-train', 'D3': 'd3-dev-train'},
        'test':  {'D2': 'd2-dev-test',  'D3': 'd3-dev-test'},
    }
    frames_tr, frames_te = [], []
    for split, domains in splits.items():
        for dom, fn in domains.items():
            csv_path = os.path.join(metadata_dir, fn)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f'Dataset metadata not found: {csv_path}\n'
                    f'Set TASK7_LORA_DATASET_DIR env var to the dataset root.'
                )
            df = pd.read_csv(csv_path)
            df['domain']     = dom
            df['new_target'] = df['class'].map(CLASS_LABELS)
            adir = os.path.join(DATASET_DIR, audio_dirs[split][dom])
            df['full_path']  = df['filename'].apply(lambda f: os.path.join(adir, f))
            df = df.rename(columns={'class': 'target'})
            (frames_tr if split == 'train' else frames_te).append(df)
    return pd.concat(frames_tr, ignore_index=True), pd.concat(frames_te, ignore_index=True)


DF_TRAIN, DF_TEST = _build_meta()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kst_timestamp() -> str:
    kst = datetime.timezone(datetime.timedelta(hours=9))
    return datetime.datetime.now(tz=kst).strftime('%Y%m%d-%H%M%S')
