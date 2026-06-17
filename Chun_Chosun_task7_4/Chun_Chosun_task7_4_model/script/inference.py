"""
inference.py — 추론 전용 진입점 (covforget / CoLoRA).

../../D2_dictionary, ../../D3_dictionary 의 체크포인트를 로드해 학습 없이
평가셋 추론만 수행한다. D3 앙상블 soft voting, variable_len, TTA-off,
active_domain_count=2.

결과:
  model/result/out.csv   (스크립트 산출)
  output.csv             (패키지 루트, 공식 제출 — --no_official 로 끔)
  둘 다 공식 DCASE TSV 양식: 헤더 없음, `<basename>.wav<TAB><full_class_name>`.

사용:
  TASK7_LORA_DATASET_DIR=<dataset_root> python inference.py --eval_dir <dir>
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common'))

import runtime  # noqa: F401  (CONFIG env 설정 — config import 보다 먼저)
import config as c
from runtime import get_device, make_logger, RESULT_DIR, D3_DICTIONARY, OUTPUT_CSV, ENSEMBLE_TOP_N
from infer_utils import (
    list_audio, list_checkpoints, select_topn_ckpts, preload_audio,
    ensemble_probs, write_output_csv,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_dir', required=True, help='추론할 오디오 디렉토리')
    ap.add_argument('--d3_dict', default=D3_DICTIONARY, help='D3 체크포인트 디렉토리')
    ap.add_argument('--out', default=os.path.join(RESULT_DIR, 'out.csv'))
    ap.add_argument('--top_n', type=int, default=ENSEMBLE_TOP_N,
                    help=f'val_macro 상위 N개만 앙상블 (기본 {ENSEMBLE_TOP_N})')
    ap.add_argument('--no_official', action='store_true', help='루트 output.csv 갱신 생략')
    args = ap.parse_args()

    log = make_logger(os.path.join(RESULT_DIR, 'log.txt'))
    device = get_device()
    paths = list_audio(args.eval_dir)
    ckpts = list_checkpoints(args.d3_dict)
    ckpts = select_topn_ckpts(ckpts, args.top_n, log=log)   # val_macro Top-N
    log(f'device={device}  eval files={len(paths)}  ensemble={len(ckpts)} (Top-{args.top_n})')
    log('protocol: variable_len, TTA-off, soft voting (active_domain_count=2)')

    waves = preload_audio(paths, log=log)
    mean_probs = ensemble_probs(ckpts, waves, device, log=log, active_domain_count=2)
    pred = mean_probs.argmax(axis=1)

    np.save(os.path.join(RESULT_DIR, 'eval_probs.npy'), mean_probs)
    write_output_csv(paths, pred, args.out)
    log(f'saved: {args.out}  ({len(paths)} rows)')
    if not args.no_official:
        write_output_csv(paths, pred, OUTPUT_CSV)
        log(f'saved: {OUTPUT_CSV}')

    inv = {v: k for k, v in c.CLASS_LABELS.items()}
    uniq, cnt = np.unique(pred, return_counts=True)
    for u, n in zip(uniq, cnt):
        log(f'  {inv[int(u)]:>18}: {n}')


if __name__ == '__main__':
    main()
