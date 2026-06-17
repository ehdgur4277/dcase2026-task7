"""
LoRA Ablation 공통 설정
  CNN14 backbone (frozen) + shared BN + Conv2d LoRA + shared FC
  학습 트릭 (FocalLoss, Mixup, BalancedSampling, SpecAug)
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

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPO_ROOT = PACKAGE_ROOT

def _default_dataset_dir():
    env_dir = os.environ.get('TASK7_LORA_DATASET_DIR')
    if env_dir:
        return env_dir
    local_dir = os.path.join(PACKAGE_ROOT, 'data')
    return local_dir

DATASET_DIR = _default_dataset_dir()
LOCAL_TASK7_DATA_DIR = os.environ.get('TASK7_DATA_DIR', os.path.join(REPO_ROOT, 'task7_data'))

# Index alignment with baseline (utils/config_task7.py:dict_class_labels)
#   baseline uses short names (phone/baby/dog/knock); long names here match
#   the metadata CSV 'class' column. Indices 7/8/9 are realigned so that
#   telephone_ringing↔7 (baseline 'phone'), piano↔8, speech↔9.
#   Insertion order == index order (visualize_tsne uses .keys() ordering).
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
    },
    'olora': {
        'lr_d3': 7e-4,
        'ortho_lambda': 0.2,
    },
    'inflora': {
        'lr_d3': 7e-4,
        'ortho_lambda': 0.2,
    },
    'cllora': {
        'lr_d3': 3e-4,
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

# 학습 트릭
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
USE_MIXUP = True
MIXUP_ALPHA = 0.3
USE_BALANCED_SAMPLING = True
HARD_CLASSES = [0, 4]          # alarm, fire
HARD_CLASS_MULTIPLIER = 1
USE_SPEC_AUG = True

# Gain augmentation
GAIN_AUG_LOW = 0.7
GAIN_AUG_HIGH = 1.3

# Dropout
DROPOUT = 0.2

# DataLoader
NUM_WORKERS = 4

# Audio chunk 분할 (train: non-overlap 4s, 마지막 chunk <2s 폐기)
CHUNK_SECONDS = 4.0
CHUNK_MIN_SECONDS = 2.0

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


# ── Label flow guard ─────────────────────────────────────────────────────
# D1 backbone의 FC head는 다음 정수 매핑으로 학습되었음 (baseline
# utils/config_task7.py:dict_class_labels). path A(txt) 의 'target' 컬럼은
# 짧은 이름(phone/baby/...), path B(metadata csv) 의 'target' 컬럼(원본 'class')은
# 긴 이름(telephone_ringing/baby_cry/...) 을 쓰므로 양쪽 모두 허용.
# 어느 경로에서든 (target_string -> new_target int) 쌍이 이 정답표와 어긋나면
# 즉시 실패시킨다 (메타데이터/매핑 변경 시 silent drift 방지).
_CANONICAL_LABEL_MAP = {
    'alarm':     0,
    'baby':      1, 'baby_cry':          1,
    'dog':       2, 'dog_bark':          2,
    'engine':    3,
    'fire':      4,
    'footsteps': 5,
    'knock':     6, 'knocking':          6,
    'phone':     7, 'telephone_ringing': 7,
    'piano':     8,
    'speech':    9,
}

def _verify_label_consistency(df, source_name):
    pairs = df[['target', 'new_target']].drop_duplicates()
    for _, row in pairs.iterrows():
        name     = str(row['target'])
        expected = _CANONICAL_LABEL_MAP.get(name)
        if expected is None:
            raise RuntimeError(
                f"[label-guard:{source_name}] unknown class name '{name}' — "
                f"add it to _CANONICAL_LABEL_MAP in common/config.py"
            )
        actual = int(row['new_target'])
        if actual != expected:
            raise RuntimeError(
                f"[label-guard:{source_name}] label drift for '{name}': "
                f"data has new_target={actual}, but D1 FC head expects {expected}. "
                f"Re-check txt/metadata mapping before training."
            )

_verify_label_consistency(DF_TRAIN, 'DF_TRAIN')
_verify_label_consistency(DF_TEST,  'DF_TEST')

DOMAIN_TO_IDX = {'D1': 0, 'D2': 1, 'D3': 2}
