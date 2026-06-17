"""
single_main.py — 단일 모델 학습 진입점 (covforget / CoLoRA, 앙상블 없음).

구조 (task7_2 single_main 대응):
  D2 phase : D2 학습 → checkpoint/single/d2.pth (adaptor 상태 포함)
  D3 phase : 그 D2 위에 D3 학습 → checkpoint/single/d3.pth
  active_domain_count: D2=1, D3=2.

⚠️ covforget 은 train_phase 가 early-stopping(val) 을 필요로 하므로,
   task7_2 의 "D2 full data" 와 달리 D2·D3 모두 fold-0 stratified split 으로
   train/val 을 나눈다 (val 은 early-stopping/val_macro 용).

평가: D2@D2/D2@D3/D3@D3 모두 variable + TTA-off (config tta.use=false → macro_eval).

산출:
  model/checkpoint/single/d2.pth, d3.pth
  학습 로그 → model/result/single_log.txt
  메트릭     → model/result/single_metric.json

사용:
  TASK7_LORA_DATASET_DIR=<dataset_root> python single_main.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common'))

import runtime  # noqa: F401  (CONFIG env — config import 전에)
import config as c
from runtime import get_device, make_logger, RESULT_DIR, CKPT_SINGLE
from kfold_train import make_kfold_split, set_seed, run_d2, run_d3, KFOLD_SEED


def main():
    fold_k = int(os.environ.get('SINGLE_FOLD_K', '0'))  # val 분리용 fold
    log = make_logger(os.path.join(RESULT_DIR, 'single_log.txt'))
    device = get_device()
    set_seed(KFOLD_SEED, device)
    log(f'=== single_main (covforget) === device={device}  val_fold={fold_k}')
    log('eval protocol: variable + TTA-off')

    df_d2_all = c.DF_TRAIN[c.DF_TRAIN['domain'] == 'D2'].reset_index(drop=True)
    df_d2_tr, df_d2_val = make_kfold_split(df_d2_all, fold_k=fold_k)
    df_d2_te = c.DF_TEST[c.DF_TEST['domain'] == 'D2']
    df_d3_all = c.DF_TRAIN[c.DF_TRAIN['domain'] == 'D3'].reset_index(drop=True)
    df_d3_tr, df_d3_val = make_kfold_split(df_d3_all, fold_k=fold_k)
    df_d3_te = c.DF_TEST[c.DF_TEST['domain'] == 'D3']
    log(f'D2 train={len(df_d2_tr)} val={len(df_d2_val)} | D3 train={len(df_d3_tr)} val={len(df_d3_val)}')

    d2_ckpt = os.path.join(CKPT_SINGLE, 'd2.pth')
    d3_ckpt = os.path.join(CKPT_SINGLE, 'd3.pth')

    # ── D2 ──
    run_d2(fold_k, df_d2_tr, df_d2_val, df_d2_te, d2_ckpt, device, log)

    # ── D3 (그 D2 위에) ──
    metrics = run_d3(fold_k, fold_k, d2_ckpt, df_d3_tr, df_d3_val,
                     df_d2_te, df_d3_te, d3_ckpt, device, log)

    metrics_out = {'protocol': 'variable_len, TTA-off', **metrics}
    with open(os.path.join(RESULT_DIR, 'single_metric.json'), 'w') as f:
        json.dump(metrics_out, f, indent=2)
    log(f'metrics: {json.dumps(metrics_out)}')


if __name__ == '__main__':
    main()
