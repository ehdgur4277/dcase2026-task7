"""
inference.py — 추론 전용 (K-Fold Valid Top-N soft voting)

reorg_prompt_2.md: model/script/inference.py.
  - 체크포인트 소스는 ../../D3_dictionary/ (읽기 전용). valid_summary.json 의 d3_folds
    를 ValAcc 로 정렬해 Top-N 을 soft-voting 한다. dictionary 에는 절대 쓰지 않는다.
  - 추론 결과는 ../result/infer/ 안에만 저장한다 (out.csv + 제출 포맷 사본).
    실험 root 의 제출 파일은 기본적으로 덮어쓰지 않는다. 명시적으로 갱신하려면
    --write_submission 을 준다.
  - 출력 라벨은 metadata 클래스명(config.CLASS_LABELS)을 그대로 따른다 (단축명 변환 없음).
  - 데이터셋은 --dataset [path] 로 받는다 (미지정 시 실험 root 옆 ../data 로 폴백).

Examples:
  python inference.py
  python inference.py --dataset /data/dcase2026/data --top_ns 10,20
  python inference.py --submission_top_n 10
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import librosa
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) in ('script', 'scripts') else SCRIPT_DIR
EXP_ROOT = os.path.dirname(REPO_ROOT)          # 실험 root (Chun_Chosun_task7_1)
COMMON = os.path.join(REPO_ROOT, 'common')
sys.path.insert(0, COMMON)

DEFAULT_DATASET_DIR = os.path.abspath(os.path.join(EXP_ROOT, os.pardir, 'data'))


def _resolve_dataset_dir(argv=None):
    """--dataset → TASK7_LORA_DATASET_DIR 환경변수로 변환 (config import 전에 호출)."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--dataset', default=None)
    known, _ = ap.parse_known_args(argv)
    ds = known.dataset or os.environ.get('TASK7_LORA_DATASET_DIR') or DEFAULT_DATASET_DIR
    ds = os.path.abspath(ds)
    os.environ['TASK7_LORA_DATASET_DIR'] = ds
    return ds


DATASET_DIR = _resolve_dataset_dir()

import config as c
# 추론 모델 로더는 공용 구현(common/model.py)을 재사용한다.
from model import load_model as _load_model

# 읽기 전용 dictionary 기본 경로 (실험 root 기준). inference 는 여기서 로드만 한다.
DEFAULT_DICT_DIR = os.path.join(EXP_ROOT, 'Chun_Chosun_task7_1_D3_dictionary')
# 제출 파일명 (실험 root). inference 가 갱신한다.
SUBMISSION_CSV_NAME = 'Chun_Chosun_task7_1.output.csv'
DEFAULT_EVAL_DIR = os.path.join(DATASET_DIR, 'dil-dcase26-eval')
AUDIO_EXTS = ('.wav', '.flac', '.mp3', '.ogg', '.m4a')

# Output labels follow the metadata 'class' values verbatim (config.CLASS_LABELS
# is built from the metadata CSVs). The submission CSV therefore emits the same
# class strings as data/metadata/*.csv — no short-name remapping. Keeping a
# single source of truth (metadata -> config.CLASS_LABELS) prevents silent label
# drift between training metadata and the submission file.
def build_output_label_map():
    """index -> metadata class name (identity over config.CLASS_LABELS)."""
    return {idx: name for name, idx in c.CLASS_LABELS.items()}


def discover_audio_files(eval_dir):
    paths = []
    for root, _, files in os.walk(eval_dir):
        for name in files:
            if name.lower().endswith(AUDIO_EXTS):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def format_path(path, eval_dir, path_format):
    if path_format == 'basename':
        return os.path.basename(path)
    if path_format == 'relative':
        return os.path.relpath(path, eval_dir)
    return path


@torch.no_grad()
def predict_probs_variable_len(model, audio_paths, device):
    probs = []
    for path in audio_paths:
        y, _ = librosa.load(path, sr=c.SAMPLE_RATE, mono=True)
        y = y.astype(np.float32)
        if len(y) < c.CLIP_SAMPLES:
            y = np.concatenate([y, np.zeros(c.CLIP_SAMPLES - len(y), dtype=np.float32)])
        audio = torch.from_numpy(y).unsqueeze(0).float().to(device)
        logits, _ = model(audio)
        probs.append(F.softmax(logits, dim=-1).detach().cpu().numpy()[0])
    return np.stack(probs, axis=0)


def _abs_path(base_dir, path):
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base_dir, path))


_FOLD_RE = re.compile(r'd2f(\d+)_d3f(\d+)')


def load_ranked_summaries(base_dir, ckpt_root, rank_metric, valid_summary='valid_summary.json'):
    """Rank D3 fold checkpoints from checkpoints/valid_summary.json.

    valid_summary.json holds a "d3_folds" list, each entry carrying a `pth`
    filename (relative to ckpt_root) plus validation metrics (ValAcc, ...).
    Entries are sorted by `rank_metric` (descending) and returned with a
    resolved absolute `ckpt` path so the caller can soft-vote the Top-N.
    """
    ckpt_root_abs = _abs_path(base_dir, ckpt_root)
    summary_path = valid_summary if os.path.isabs(valid_summary) else os.path.join(ckpt_root_abs, valid_summary)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f'valid summary not found: {summary_path}')

    with open(summary_path) as f:
        data = json.load(f)
    folds = data.get('d3_folds', [])
    if not folds:
        raise ValueError(f'No "d3_folds" entries in {summary_path}')

    summaries = []
    for entry in folds:
        pth = entry.get('pth')
        if not pth:
            continue
        ckpt = pth if os.path.isabs(pth) else os.path.join(ckpt_root_abs, pth)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f'Checkpoint listed in valid summary is missing: {ckpt}')
        summary = dict(entry)
        summary['ckpt'] = ckpt
        match = _FOLD_RE.search(os.path.basename(pth))
        if match:
            summary.setdefault('d2_fold', int(match.group(1)))
            summary.setdefault('d3_fold', int(match.group(2)))
        summaries.append(summary)

    if not summaries:
        raise ValueError(f'No valid checkpoints resolved from {summary_path}')

    ranked = sorted(summaries, key=lambda x: float(x.get(rank_metric, -1.0)), reverse=True)
    return ranked, f'{summary_path}:d3_folds sorted by {rank_metric}'


def select_device(requested='auto', min_cuda_free_gb=2.0):
    if requested == 'cpu':
        return 'cpu'
    if requested == 'cuda':
        return 'cuda'
    if not torch.cuda.is_available():
        return 'cpu'
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < min_cuda_free_gb * (1024 ** 3):
            return 'cpu'
    except Exception:
        pass
    return 'cuda'


def parse_top_ns(text, total):
    raw = [int(x.strip()) for x in text.split(',') if x.strip()]
    top_ns = []
    for n in raw:
        if n <= 0:
            raise ValueError(f'top_n must be positive: {n}')
        if n > total:
            raise ValueError(f'top_n={n} exceeds available checkpoints={total}')
        if n not in top_ns:
            top_ns.append(n)
    if not top_ns:
        raise ValueError('No valid top_ns were provided')
    return top_ns


def write_csv(out_path, audio_paths, labels, eval_dir, path_format):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label'])
        for path, label in zip(audio_paths, labels):
            writer.writerow([format_path(path, eval_dir, path_format), label])


# 읽기 전용 D2 dictionary 기본 경로 (D2@D2 검증용 fold 체크포인트).
DEFAULT_D2_DICT_DIR = os.path.join(EXP_ROOT, 'Chun_Chosun_task7_1_D2_dictionary')


def load_d2_fold_summaries(d2_dict_dir, valid_summary_path):
    """valid_summary.json 의 d2_folds 를 읽어 D2_dictionary 의 pth 로 해석 (읽기 전용).

    D2@D2 검증을 위해 D2 fold 체크포인트를 D2_fold_val 내림차순으로 정렬해 반환한다.
    """
    with open(valid_summary_path) as f:
        data = json.load(f)
    folds = data.get('d2_folds', [])
    summaries = []
    for entry in folds:
        pth = entry.get('pth')
        if not pth:
            continue
        ckpt = pth if os.path.isabs(pth) else os.path.join(d2_dict_dir, pth)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f'D2 fold checkpoint missing: {ckpt}')
        summary = dict(entry)
        summary['ckpt'] = ckpt
        summaries.append(summary)
    return sorted(summaries, key=lambda x: float(x.get('D2_fold_val', -1.0)), reverse=True)


def run_validation_topk(d3_probs_d2, d3_probs_d3, d2_targets, d3_targets,
                        d2_dict_dir, valid_summary_path, base_dir,
                        d2_rank, d3_rank, device, top_n):
    """Top-N(=20) 앙상블의 검증 성능을 eval.py 와 동일 정의로 계산.

      D2@D3 / D3@D3 : Top-N D3 fold(count=2) soft-voting 을 D2/D3 test 에 적용.
      D2@D2          : 전체 D2 fold(count=1) soft-voting 을 D2 test 에 적용.
      Acc = (D2@D3 + D3@D3) / 2,  Forgetting = D2@D2 − D2@D3.
    d3_probs_d2/d3 는 메인 추론 루프에서 이미 수집한 Top-N D3 모델의 D2/D3 test 확률.
    """
    import evaluate as ev

    mean_d2 = np.mean(np.stack(d3_probs_d2, axis=0), axis=0)
    mean_d3 = np.mean(np.stack(d3_probs_d3, axis=0), axis=0)
    d2_at_d3 = ev.macro_accuracy(mean_d2.argmax(axis=1), d2_targets)
    d3_at_d3 = ev.macro_accuracy(mean_d3.argmax(axis=1), d3_targets)
    d2_at_d3_pc = ev.per_class_accuracy(mean_d2.argmax(axis=1), d2_targets)
    d3_at_d3_pc = ev.per_class_accuracy(mean_d3.argmax(axis=1), d3_targets)

    # D2@D2: D2 fold (active_domain_count=1) 전체 soft-voting
    df_d2_te = c.DF_TEST[c.DF_TEST['domain'] == 'D2']
    d2_folds = load_d2_fold_summaries(d2_dict_dir, valid_summary_path)
    print(f'  [validation] D2 folds for D2@D2: {len(d2_folds)} (read-only @ {d2_dict_dir})')
    d2d2_probs, d2d2_targets = [], None
    for i, s in enumerate(d2_folds, 1):
        print(f"    D2 fold {i}/{len(d2_folds)} D2_fold_val={float(s.get('D2_fold_val', 0.0)):.2f} {os.path.basename(s['ckpt'])}")
        m = _load_model(base_dir, s['ckpt'], d2_rank, d3_rank, 1, device)
        p, t = ev.collect_probs_variable_len(m, df_d2_te, device)
        d2d2_probs.append(p)
        d2d2_targets = t
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    mean_d2d2 = np.mean(np.stack(d2d2_probs, axis=0), axis=0)
    d2_at_d2 = ev.macro_accuracy(mean_d2d2.argmax(axis=1), d2d2_targets)

    acc = (d2_at_d3 + d3_at_d3) / 2.0
    fr = d2_at_d2 - d2_at_d3
    metric = {
        'metric': 'macro_accuracy',
        'unit': 'percent',
        'mode': 'kfold_validation',
        'top_n': int(top_n),
        'd2_folds_used': len(d2_folds),
        'D2_at_D2': d2_at_d2,
        'D2_at_D3': d2_at_d3,
        'D3_at_D3': d3_at_d3,
        'Acc': acc,
        'Forgetting': fr,
        'per_class': {
            'D2_at_D3': d2_at_d3_pc,
            'D3_at_D3': d3_at_d3_pc,
            'Acc': ev._average_per_class_accuracy(d2_at_d3_pc, d3_at_d3_pc),
        },
    }
    return metric


def main():
    parser = argparse.ArgumentParser(
        description='Top-20 eval-set inference + K-Fold validation (D2@D2/D2@D3/D3@D3/Acc/Forgetting)'
    )
    parser.add_argument('--dataset', default=None,
                        help='데이터셋 root 경로. 미지정 시 실험 root 옆 ../data 로 폴백.')
    parser.add_argument('--eval_dir', default=None,
                        help='추론 대상 오디오 폴더. 기본: <dataset>/dil-dcase26-eval')
    parser.add_argument('--ckpt_root', default=DEFAULT_DICT_DIR,
                        help='D3 체크포인트(읽기 전용) 루트. 기본: ../Chun_Chosun_task7_1_D3_dictionary')
    parser.add_argument('--d2_dict', default=DEFAULT_D2_DICT_DIR,
                        help='D2 fold 체크포인트(읽기 전용) 루트. 기본: ../Chun_Chosun_task7_1_D2_dictionary')
    parser.add_argument('--valid_summary', default='valid_summary.json',
                        help='Validation summary JSON (relative to ckpt_root) listing d2_folds/d3_folds.')
    parser.add_argument('--top_n', type=int, default=20,
                        help='추론/검증에 쓸 Top-N (고정 기본값 20).')
    parser.add_argument('--rank_metric', default='ValAcc')
    parser.add_argument('--d2_rank', type=int, default=128)
    parser.add_argument('--d3_rank', type=int, default=64)
    parser.add_argument('--path_format', default='basename', choices=['absolute', 'relative', 'basename'])
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='auto uses CUDA only when enough free memory is available')
    parser.add_argument('--out_dir', default=os.path.join(REPO_ROOT, 'result', 'infer'),
                        help='추론 결과 출력 폴더. 기본: ../result/infer (실험 root 제출 파일은 건드리지 않음)')
    parser.add_argument('--no_valid', action='store_true',
                        help='검증(D2/D3 test) 성능 산출을 생략한다.')
    parser.add_argument('--write_submission', action='store_true',
                        help='실험 root 의 제출 파일(Chun_Chosun_task7_1.output.csv)을 명시적으로 덮어쓴다.')
    args = parser.parse_args()

    base_dir = REPO_ROOT
    out_dir_abs = _abs_path(base_dir, args.out_dir)
    eval_dir = _abs_path(base_dir, args.eval_dir or DEFAULT_EVAL_DIR)
    audio_paths = discover_audio_files(eval_dir)
    if not audio_paths:
        raise FileNotFoundError(f'No audio files found under {eval_dir}')

    ckpt_root_abs = _abs_path(base_dir, args.ckpt_root)
    valid_summary_path = (args.valid_summary if os.path.isabs(args.valid_summary)
                          else os.path.join(ckpt_root_abs, args.valid_summary))
    ranked, rank_source = load_ranked_summaries(base_dir, args.ckpt_root, args.rank_metric, args.valid_summary)

    top_n = args.top_n
    if top_n <= 0 or top_n > len(ranked):
        raise ValueError(f'top_n={top_n} 은 1..{len(ranked)} 범위여야 합니다.')
    device = select_device(args.device)
    inv_labels = build_output_label_map()

    run_valid = not args.no_valid
    df_d2_te = c.DF_TEST[c.DF_TEST['domain'] == 'D2'] if run_valid else None
    df_d3_te = c.DF_TEST[c.DF_TEST['domain'] == 'D3'] if run_valid else None

    print('Top-20 inference + validation')
    print(f'  device       : {device}')
    print(f'  dataset dir  : {DATASET_DIR}')
    print(f'  eval_dir     : {eval_dir}')
    print(f'  D3 ckpt root : {ckpt_root_abs} (read-only)')
    print(f'  D2 ckpt root : {_abs_path(base_dir, args.d2_dict)} (read-only)')
    print(f'  audio files  : {len(audio_paths)}')
    print(f'  checkpoints  : Top-{top_n}/{len(ranked)} by {args.rank_metric}')
    print(f'  validation   : {"ON" if run_valid else "OFF"}')
    print('  inference    : variable_len, D3 active_domain_count=2')

    import evaluate as ev  # validation 확률 수집용 (read-only 모델 로드)

    eval_probs = []
    d3_probs_d2, d3_probs_d3 = [], []
    d2_targets = d3_targets = None
    for i, summary in enumerate(ranked[:top_n], 1):
        label = f"d2f{summary.get('d2_fold', '?')}_d3f{summary.get('d3_fold', '?')}"
        metric = float(summary.get(args.rank_metric, 0.0))
        print(f"  [{i:02d}/{top_n:02d}] {label} {args.rank_metric}={metric:.2f} {summary['ckpt']}")
        model = _load_model(base_dir, summary['ckpt'], args.d2_rank, args.d3_rank, 2, device)
        eval_probs.append(predict_probs_variable_len(model, audio_paths, device))
        if run_valid:
            p2, t2 = ev.collect_probs_variable_len(model, df_d2_te, device)
            p3, t3 = ev.collect_probs_variable_len(model, df_d3_te, device)
            d3_probs_d2.append(p2); d3_probs_d3.append(p3)
            d2_targets, d3_targets = t2, t3
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── eval-set 추론 → 단일 output.csv (Top-20, metadata 라벨) ──
    mean_probs = np.mean(np.stack(eval_probs, axis=0), axis=0)
    pred_labels = [inv_labels[int(idx)] for idx in mean_probs.argmax(axis=1).tolist()]
    out_csv = os.path.join(out_dir_abs, SUBMISSION_CSV_NAME)
    write_csv(out_csv, audio_paths, pred_labels, eval_dir, args.path_format)
    print(f'\noutput.csv saved (Top-{top_n}) -> {out_csv}')

    if args.write_submission:
        root_csv = os.path.join(EXP_ROOT, SUBMISSION_CSV_NAME)
        write_csv(root_csv, audio_paths, pred_labels, eval_dir, args.path_format)
        print(f'[--write_submission] ROOT submission overwritten -> {root_csv}')
    else:
        print(f'(root submission untouched: {os.path.join(EXP_ROOT, SUBMISSION_CSV_NAME)})')

    # ── 검증 성능 (Top-20) → valid_metric.json + 콘솔 요약 ──
    if run_valid:
        print('\n========== Validation (Top-20, D2/D3 test) ==========')
        metric = run_validation_topk(
            d3_probs_d2, d3_probs_d3, d2_targets, d3_targets,
            _abs_path(base_dir, args.d2_dict), valid_summary_path, base_dir,
            args.d2_rank, args.d3_rank, device, top_n,
        )
        metric_path = os.path.join(out_dir_abs, 'valid_metric.json')
        with open(metric_path, 'w') as f:
            json.dump(metric, f, indent=2)
        print(f"  D2@D2 = {metric['D2_at_D2']:.2f}%")
        print(f"  D2@D3 = {metric['D2_at_D3']:.2f}%")
        print(f"  D3@D3 = {metric['D3_at_D3']:.2f}%")
        print(f"  Acc   = {metric['Acc']:.2f}%   (= (D2@D3 + D3@D3) / 2)")
        print(f"  Fr    = {metric['Forgetting']:+.2f}%p (= D2@D2 - D2@D3)")
        print(f'  valid_metric.json saved -> {metric_path}')

    return out_csv


if __name__ == '__main__':
    main()
