"""
LoRA Ablation 공통 설정
  CNN14 backbone (frozen) + shared BN + Conv2d LoRA + shared FC
  EWC + 학습 트릭 (FocalLoss, Mixup, BalancedSampling, SpecAug, TTA)
"""
import os
import pandas as pd

SAMPLE_RATE = 32000
CLIP_SAMPLES = 32000 * 4
MEL_BINS = 64
FMIN = 50
FMAX = 14000
WIN_SIZE = 1024
HOP_SIZE = 320
NUM_CLASSES = 10

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATASET_DIR = os.environ.get('TASK7_LORA_DATASET_DIR', '/data2/ejkim/DCASE2026/datasets')
LOCAL_TASK7_DATA_DIR = os.environ.get('TASK7_DATA_DIR', os.path.join(REPO_ROOT, 'task7_data'))

CLASS_LABELS = {
    'alarm': 0, 'baby_cry': 1, 'dog_bark': 2, 'engine': 3, 'fire': 4,
    'footsteps': 5, 'knocking': 6, 'telephone_ringing': 7, 'piano': 8, 'speech': 9,
}

# LoRA
LORA_RANK = 64
LORA_ALPHA = 64.0

# Training
EPOCHS = 150
BATCH_SIZE = 32
LR_D2 = 1e-3
WEIGHT_DECAY = 1e-5

# === Per-mode D3 config ===
MODE_CONFIG = {
    'vanilla': {
        'lr_d3': 5e-4,
        'ewc_lambda': 5000.0,
    },
    'olora': {
        'lr_d3': 7e-4,
        'ewc_lambda': 500.0,
        'ortho_lambda': 0.2,
    },
    'inflora': {
        'lr_d3': 7e-4,
        'ewc_lambda': 500.0,
        'ortho_lambda': 0.2,
    },
    'cllora': {
        'lr_d3': 3e-4,
        'ewc_lambda': 2000.0,
        'kd_alpha': 0.5,
        'kd_temp': 1.5,
        'kd_feat_weight': 0.3,
    },
}

# Defaults
ORTHO_LAMBDA = 0.5
KD_TEMP = 1.5
KD_ALPHA = 0.7
KD_FEAT_WEIGHT = 1.0
EWC_LAMBDA = 5000.0

# 학습 트릭
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
USE_MIXUP = True
MIXUP_ALPHA = 0.3
USE_BALANCED_SAMPLING = True
HARD_CLASSES = [0, 4]          # alarm, fire
HARD_CLASS_MULTIPLIER = 5
USE_SPEC_AUG = True
USE_TTA = True
TTA_GAINS = [1.0, 0.85, 1.15]

def _build_meta():
    metadata_dir = os.path.join(DATASET_DIR, 'metadata')
    if not os.path.exists(os.path.join(metadata_dir, 'd2-dev-train.csv')):
        train_path = os.path.join(LOCAL_TASK7_DATA_DIR, 'evaluation_setup', 'development_train.txt')
        test_path = os.path.join(LOCAL_TASK7_DATA_DIR, 'evaluation_setup', 'development_test.txt')
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(
                'Dataset metadata not found. Set TASK7_LORA_DATASET_DIR for metadata/*.csv '
                'or TASK7_DATA_DIR for evaluation_setup/development_*.txt.'
            )
        cols = ['filename', 'target', 'domain', 'new_target']
        df_train = pd.read_csv(train_path, sep='\t', names=cols)
        df_test = pd.read_csv(test_path, sep='\t', names=cols)
        for df in (df_train, df_test):
            df['full_path'] = df['filename'].apply(lambda f: os.path.join(LOCAL_TASK7_DATA_DIR, f))
        return df_train, df_test

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
            df = pd.read_csv(os.path.join(DATASET_DIR, 'metadata', fn))
            df['domain'] = dom
            df['new_target'] = df['class'].map(CLASS_LABELS)
            adir = os.path.join(DATASET_DIR, audio_dirs[split][dom])
            df['full_path'] = df['filename'].apply(lambda f: os.path.join(adir, f))
            df = df.rename(columns={'class': 'target'})
            (frames_tr if split == 'train' else frames_te).append(df)
    return pd.concat(frames_tr, ignore_index=True), pd.concat(frames_te, ignore_index=True)

DF_TRAIN, DF_TEST = _build_meta()
DOMAIN_TO_IDX = {'D1': 0, 'D2': 1, 'D3': 2}

# K-Fold
KFOLD_K      = 5
KFOLD_SEED   = 42

# EarlyStopping
D2_PATIENCE   = 20
D3_PATIENCE   = 20
D2_MIN_EPOCHS = 50
D3_MIN_EPOCHS = 50
MIN_DELTA     = 1e-4

# DataLoader workers — /dev/shm 부족 환경에서는 DATALOADER_WORKERS=0 으로 설정
NUM_WORKERS = int(os.environ.get('DATALOADER_WORKERS', '4'))
