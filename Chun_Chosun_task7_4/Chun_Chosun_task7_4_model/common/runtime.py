"""
common/runtime.py — 경로 상수 + 로거 + device 헬퍼 (covforget / CoLoRA 패키지).

경로는 이 파일 위치(model/common/) 기준으로 계산하므로 어디서 실행하든 동일하다.
  PKG_ROOT/                      ← 제출 패키지 루트
    ├── model/  (= MODEL_ROOT)
    │     ├── common/  ← 이 파일 (config.py, backbone.py, exp13 모듈 등)
    │     ├── script/
    │     ├── checkpoint/{single,ensemble}/  + checkpoint_D1.pth
    │     ├── result/
    │     └── config.yaml
    ├── D2_dictionary/
    ├── D3_dictionary/
    ├── output.csv
    └── meta.yaml

config.py 는 CONFIG env (없으면 MODEL_ROOT/config.yaml) 와 D1_CKPT(없으면
checkpoint/checkpoint_D1.pth 동봉본) 를 자동 해석한다. 데이터셋은
TASK7_LORA_DATASET_DIR env 로 지정한다.
"""
import os

import torch

COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.dirname(COMMON_DIR)
PKG_ROOT   = os.path.dirname(MODEL_ROOT)

# config.py 가 CONFIG/모델루트 기준으로 config.yaml 을 찾도록 보장
os.environ.setdefault('CONFIG', os.path.join(MODEL_ROOT, 'config.yaml'))

# ── 앙상블 하이퍼파라미터 ─────────────────────────────────────────────────────
# val_macro 상위 N개 D3 체크포인트만 soft-voting (전체 25개가 아니라 Top-N 통일).
ENSEMBLE_TOP_N = int(os.environ.get('ENSEMBLE_TOP_N', '20'))

RESULT_DIR     = os.path.join(MODEL_ROOT, 'result')
CKPT_SINGLE    = os.path.join(MODEL_ROOT, 'checkpoint', 'single')
CKPT_ENSEMBLE  = os.path.join(MODEL_ROOT, 'checkpoint', 'ensemble')
D2_DICTIONARY  = os.path.join(PKG_ROOT, 'Chun_Chosun_task7_4_D2_dictionary')
D3_DICTIONARY  = os.path.join(PKG_ROOT, 'Chun_Chosun_task7_4_D3_dictionary')
OUTPUT_CSV     = os.path.join(PKG_ROOT, 'Chun_Chosun_task7_4.output.csv')


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def make_logger(log_path):
    """stdout + 파일(append) 동시 기록 로거 반환."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(msg=''):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(str(msg) + '\n')
    return log
