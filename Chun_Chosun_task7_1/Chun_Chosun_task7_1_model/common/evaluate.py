"""
eval.py — 261_D3DeltaGP-CosKD 포괄 평가

평가 지점 (도메인 정보 누수 없음, active_domain_count 만으로 제어):
  D2@D2 : after-D2 모델(d2.pth, count=1)  on D2 test
  D2@D3 : after-D2 모델(d2.pth, count=1)  on D3 test
  D3@D2 : after-D3 모델(d3.pth, count=2)  on D2 test  ← Forgetting
  D3@D3 : after-D3 모델(d3.pth, count=2)  on D3 test
  Avg        = (D3@D2 + D3@D3) / 2
  Forgetting = D2@D2 − D3@D2

평가 모드:
  variable_len : 파일 전체 길이 (4s 미만 zero-pad, 초과 그대로)

측정 지표:
  Macro : 클래스별 정확도 평균 (mean of per-class acc)
  per_class : 클래스별 상세 (correct, total, acc)

누수 확인:
  - active_domain_count로만 LoRA 수 제어, 도메인 레이블 불필요
  - d3.pth(count=2)는 D2/D3 테스트 모두 동일 모델로 통과 → 누수 없음
  - test_time_domain_info: False

사용법:
  python eval.py                  # 기본: K-Fold 평가
  python eval.py --eval_scope single --d2_ckpt checkpoints/fold_d2f0/d2.pth --d3_ckpt checkpoints/fold_d2f0_d3f0/d3.pth
  python eval.py --eval_scope kfold --top_ns 5,10,15,20,25
  python eval.py --d2_rank 128 --d3_rank 64
"""

import argparse, glob, os, sys, json, re, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# evaluate.py 는 common/ 에 위치한다. script/ 든 common/ 든 한 단계 위가 model(=base) root.
REPO_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) in ('script', 'scripts', 'common') else SCRIPT_DIR
EXP_ROOT = os.path.dirname(REPO_ROOT)
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
# 모델 정의/로더는 공용 구현(common/model.py)을 재사용한다 (중복 코드 금지).
from model import load_model as _load_model

# 앙상블 Top-K 는 하이퍼파라미터로 고정한다 (기본 20). dev-test 로 K 를 고르지 않는다.
ENSEMBLE_TOP_K = int(os.environ.get('ENSEMBLE_TOP_K', 20))


# ============================================================
# Dataset
# ============================================================
class VarLenDataset(Dataset):
    """파일 전체 길이. 4s 미만 zero-pad, 초과 그대로."""
    def __init__(self, df):
        self.samples = []
        for _, row in df.iterrows():
            y, _ = librosa.load(row['full_path'], sr=c.SAMPLE_RATE, mono=True)
            y = y.astype(np.float32)
            if len(y) < c.CLIP_SAMPLES:
                y = np.concatenate([y, np.zeros(c.CLIP_SAMPLES - len(y), dtype=np.float32)])
            self.samples.append((y, int(row['new_target']), row['filename']))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        y, label, fname = self.samples[idx]
        onehot = np.zeros(c.NUM_CLASSES, dtype=np.float32)
        onehot[label] = 1.0
        return torch.from_numpy(y), onehot, fname


# ============================================================
# Metrics
# ============================================================
def macro_accuracy(preds, targets):
    p, t = np.array(preds), np.array(targets)
    return float(np.mean([
        np.sum(p[t == cl] == cl) / np.sum(t == cl)
        for cl in np.unique(t)
    ]) * 100)


def per_class_accuracy(preds, targets):
    p, t = np.array(preds), np.array(targets)
    inv = {v: k for k, v in c.CLASS_LABELS.items()}
    out = {}
    for cl in sorted(np.unique(t).tolist()):
        mask = t == cl
        n_c, n_t = int(np.sum(p[mask] == cl)), int(np.sum(mask))
        out[inv.get(cl, str(cl))] = {
            'correct': n_c, 'total': n_t,
            'acc': round(n_c / n_t * 100, 2) if n_t else 0.0,
        }
    return out


def compute_metrics(preds, targets):
    return {
        'macro':     macro_accuracy(preds, targets),
        'per_class': per_class_accuracy(preds, targets),
    }


# ============================================================
# Evaluation functions
# ============================================================
@torch.no_grad()
def eval_variable_len(model, df, device):
    """파일 전체 길이로 평가."""
    model.eval()
    ds = VarLenDataset(df)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    preds, targets = [], []
    for audio, target, _ in loader:
        logits, _ = model(audio.float().to(device))
        preds.append(int(logits.argmax(dim=1).item()))
        targets.append(int(target.float().argmax(dim=1).item()))
    return preds, targets


@torch.no_grad()
def collect_probs_variable_len(model, df, device):
    """파일 전체 길이 평가용 softmax 확률 수집."""
    model.eval()
    ds = VarLenDataset(df)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    probs, targets = [], []
    for audio, target, _ in loader:
        logits, _ = model(audio.float().to(device))
        probs.append(F.softmax(logits, dim=-1).detach().cpu().numpy()[0])
        targets.append(int(target.float().argmax(dim=1).item()))
    return np.stack(probs, axis=0), np.array(targets, dtype=np.int64)


# ============================================================
# Helpers
# ============================================================
def _resolve_path(base_dir, path):
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


_D3_FOLD_RE = re.compile(r'd2f(\d+)_d3f(\d+)')
_D2_FOLD_RE = re.compile(r'd2_fold(\d+)')


def _load_valid_summary(base_dir, ckpt_root):
    ckpt_root = _resolve_path(base_dir, ckpt_root)
    summary_path = os.path.join(ckpt_root, 'valid_summary.json')
    if not os.path.exists(summary_path):
        return None, summary_path, ckpt_root
    with open(summary_path) as f:
        return json.load(f), summary_path, ckpt_root


def _discover_kfold_summaries(base_dir, ckpt_root='checkpoints'):
    ckpt_root = _resolve_path(base_dir, ckpt_root)
    pattern = os.path.join(ckpt_root, 'fold_d2f*_d3f*', 'summary.json')
    summaries = []
    for summary_path in sorted(glob.glob(pattern)):
        with open(summary_path) as f:
            summary = json.load(f)
        ckpt = summary.get('ckpt') or os.path.join(os.path.dirname(summary_path), 'd3.pth')
        ckpt = _resolve_path(base_dir, ckpt)
        if not os.path.exists(ckpt):
            continue
        summary['ckpt'] = ckpt
        summary['summary_path'] = summary_path
        summaries.append(summary)
    if summaries:
        return summaries

    valid, summary_path, ckpt_root = _load_valid_summary(base_dir, ckpt_root)
    if not valid:
        return summaries
    for entry in valid.get('d3_folds', []):
        pth = entry.get('pth')
        if not pth:
            continue
        ckpt = _resolve_path(ckpt_root, pth)
        if not os.path.exists(ckpt):
            continue
        summary = dict(entry)
        summary['ckpt'] = ckpt
        summary['summary_path'] = summary_path
        match = _D3_FOLD_RE.search(os.path.basename(pth))
        if match:
            summary.setdefault('d2_fold', int(match.group(1)))
            summary.setdefault('d3_fold', int(match.group(2)))
        summaries.append(summary)
    return summaries


def _discover_d2_fold_summaries(base_dir, ckpt_root='checkpoints'):
    ckpt_root = _resolve_path(base_dir, ckpt_root)
    pattern = os.path.join(ckpt_root, 'fold_d2f*', 'summary.json')
    summaries = []
    for summary_path in sorted(glob.glob(pattern)):
        fold_dir = os.path.basename(os.path.dirname(summary_path))
        if '_d3f' in fold_dir:
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        ckpt = summary.get('ckpt') or os.path.join(os.path.dirname(summary_path), 'd2.pth')
        ckpt = _resolve_path(base_dir, ckpt)
        if not os.path.exists(ckpt):
            continue
        summary['ckpt'] = ckpt
        summary['summary_path'] = summary_path
        summaries.append(summary)
    if summaries:
        return summaries

    valid, summary_path, ckpt_root = _load_valid_summary(base_dir, ckpt_root)
    if not valid:
        return summaries
    for entry in valid.get('d2_folds', []):
        pth = entry.get('pth')
        if not pth:
            continue
        ckpt = _resolve_path(ckpt_root, pth)
        if not os.path.exists(ckpt):
            continue
        summary = dict(entry)
        summary['ckpt'] = ckpt
        summary['summary_path'] = summary_path
        match = _D2_FOLD_RE.search(os.path.basename(pth))
        if match:
            summary.setdefault('d2_fold', int(match.group(1)))
        summaries.append(summary)
    return summaries


def _select_device(requested='auto', min_cuda_free_gb=2.0):
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


def _parse_top_ns(text, total):
    if text:
        # 명시적 --top_ns 는 진단용 스윕 허용 (사용자가 직접 지정한 경우에 한함).
        raw = [int(x.strip()) for x in text.split(',') if x.strip()]
        top_ns = []
        for n in raw:
            if 0 < n <= total and n not in top_ns:
                top_ns.append(n)
        return top_ns
    # 기본: dev-test 로 N 을 고르지 않도록 고정 Top-K 한 값만 평가한다 (누수 방지).
    return [min(ENSEMBLE_TOP_K, total)]


def _average_per_class_accuracy(d2_per_class, d3_per_class):
    out = {}
    for cls_name in sorted(set(d2_per_class) | set(d3_per_class)):
        d2 = d2_per_class.get(cls_name, {'acc': 0.0, 'correct': 0, 'total': 0})
        d3 = d3_per_class.get(cls_name, {'acc': 0.0, 'correct': 0, 'total': 0})
        out[cls_name] = {
            'acc': (float(d2['acc']) + float(d3['acc'])) / 2.0,
            'D2_acc': float(d2['acc']),
            'D3_acc': float(d3['acc']),
            'D2_correct': int(d2['correct']),
            'D2_total': int(d2['total']),
            'D3_correct': int(d3['correct']),
            'D3_total': int(d3['total']),
        }
    return out


def _bar(acc, w=20):
    n = int(acc / 100 * w)
    return '█' * n + '░' * (w - n)


def _print_metrics(label, m):
    print(f"    {label}  macro={m['macro']:.2f}%")
    for cls_name, v in m['per_class'].items():
        print(f"      {cls_name:<22} {v['acc']:>6.1f}%  {v['correct']:>4}/{v['total']:<4}  {_bar(v['acc'])}")


# ============================================================
# Main
# ============================================================
def run_single_eval(args, base_dir, device):
    d2_ckpt = _resolve_path(base_dir, args.d2_ckpt)
    d3_ckpt = _resolve_path(base_dir, args.d3_ckpt)
    for p in (d2_ckpt, d3_ckpt):
        if not os.path.exists(p): raise FileNotFoundError(f'checkpoint 없음: {p}')

    SEP  = '─' * 66
    SEP2 = '═' * 66

    print(SEP2)
    print('  261_D3DeltaGP-CosKD — Macro + Per-class 포괄 평가')
    print(SEP2)
    print(f"  Device  : {device}")
    print(f"  D2 ckpt : {os.path.basename(d2_ckpt)}  (count=1)")
    print(f"  D3 ckpt : {os.path.basename(d3_ckpt)}  (count=2, sum D2+D3)")
    print(f"  Ranks   : D2={args.d2_rank}, D3={args.d3_rank}")
    print()

    # ── 누수 확인 ──────────────────────────────────────────────
    print(f"  [누수 확인]")
    print(f"  - D2@D2: d2.pth(count=1)  → D2 LoRA만 활성, 도메인 레이블 불필요 ✓")
    print(f"  - D2@D3: d2.pth(count=1)  → 동일, D3 LoRA 미학습 상태 ✓")
    print(f"  - D3@D2: d3.pth(count=2)  → D2+D3 합산, 도메인 레이블 불필요 ✓")
    print(f"  - D3@D3: d3.pth(count=2)  → 동일 모델로 D3 test 통과, 도메인 구분 없음 ✓")
    print(f"  → test_time_domain_info = False  (누수 없음)\n")

    df_d2_te = c.DF_TEST[c.DF_TEST['domain'] == 'D2']
    df_d3_te = c.DF_TEST[c.DF_TEST['domain'] == 'D3']
    print(f"  Test: D2={len(df_d2_te)} samples, D3={len(df_d3_te)} samples\n")

    POINTS = ['D2@D2', 'D2@D3', 'D3@D2', 'D3@D3']
    print(SEP2)
    print('  평가 모드: VARIABLE_LEN')
    print(SEP2)

    metrics = {}

    # after-D2 (count=1): D2@D2, D2@D3
    print(f"  after-D2 모델 (d2.pth, active_domain_count=1)")
    m2 = _load_model(base_dir, d2_ckpt, args.d2_rank, args.d3_rank, 1, device)
    metrics['D2@D2'] = compute_metrics(*eval_variable_len(m2, df_d2_te, device))
    metrics['D2@D3'] = compute_metrics(*eval_variable_len(m2, df_d3_te, device))
    del m2

    # after-D3 (count=2): D3@D2, D3@D3
    print(f"  after-D3 모델 (d3.pth, active_domain_count=2, D2+D3 합산)")
    m3 = _load_model(base_dir, d3_ckpt, args.d2_rank, args.d3_rank, 2, device)
    metrics['D3@D2'] = compute_metrics(*eval_variable_len(m3, df_d2_te, device))
    metrics['D3@D3'] = compute_metrics(*eval_variable_len(m3, df_d3_te, device))
    del m3

    # 파생 지표
    avg_macro = (metrics['D3@D2']['macro'] + metrics['D3@D3']['macro']) / 2
    fgt_macro = metrics['D2@D2']['macro'] - metrics['D3@D2']['macro']

    # ── 요약 출력 ──────────────────────────────────────────
    print(f"\n  {SEP}")
    print(f"  CL 지표 요약")
    print(f"  {SEP}")
    rows = [(p, metrics[p]['macro'], False) for p in POINTS]
    rows += [('Avg', avg_macro, False), ('Forgetting', fgt_macro, True)]
    print(f"  {'Metric':<14} {'Macro':>10}")
    print(f"  {'─'*26}")
    for lbl, mac, is_delta in rows:
        sa = f"{mac:+.2f}%p" if is_delta else f"{mac:.2f}%"
        print(f"  {lbl:<14} {sa:>10}")
    print()

    # ── 클래스별 정확도 ────────────────────────────────────
    print(f"  {SEP}")
    print(f"  클래스별 정확도 (Macro = 아래 per-class acc 평균)")
    print(f"  {SEP}")
    for p in POINTS:
        m = metrics[p]
        print(f"\n  [{p}]  macro={m['macro']:.2f}%")
        print(f"  {'Class':<22} {'Acc':>7}  {'Correct':>7}  {'Total':>6}")
        print(f"  {'─'*52}")
        for cls_name, v in m['per_class'].items():
            print(f"  {cls_name:<22} {v['acc']:>6.1f}%  {v['correct']:>7}  {v['total']:>6}  {_bar(v['acc'])}")

    print()
    result = {
        **{p: metrics[p] for p in POINTS},
        'avg_macro': avg_macro,
        'forgetting_macro': fgt_macro,
    }

    # ── JSON 저장 ──────────────────────────────────────────────
    result_dir = os.path.join(base_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)
    TIMESTR = time.strftime('%Y%m%d-%H%M%S')
    out_path = os.path.join(result_dir, f'eval_{TIMESTR}.json')
    save = {
        'timestamp': TIMESTR,
        'method': '261_D3DeltaGP-CosKD',
        'eval_scope': 'single',
        'test_time_domain_info': False,
        'leakage': False,
        'checkpoints': {'d2': d2_ckpt, 'd3': d3_ckpt},
        'args': {'d2_rank': args.d2_rank, 'd3_rank': args.d3_rank},
        'points': POINTS,
        'inference': 'variable_len',
        'results': result,
    }
    with open(out_path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"Results saved → {out_path}")
    return out_path


def run_kfold_eval(args, base_dir, device):
    summaries = _discover_kfold_summaries(base_dir, args.ckpt_root)
    if not summaries:
        raise FileNotFoundError(f'K-Fold D3 summary 없음: {args.ckpt_root}/fold_d2f*_d3f*/summary.json')
    d2_summaries = _discover_d2_fold_summaries(base_dir, args.ckpt_root)
    if not d2_summaries:
        raise FileNotFoundError(f'D2 fold summary 없음: {args.ckpt_root}/fold_d2f*/summary.json')

    ranked = sorted(summaries, key=lambda x: float(x.get(args.rank_metric, -1.0)), reverse=True)
    d2_ranked = sorted(d2_summaries, key=lambda x: float(x.get('D2_fold_val', -1.0)), reverse=True)
    d2_top_ns = list(range(1, len(d2_ranked) + 1))
    top_ns = _parse_top_ns(args.top_ns, len(ranked))
    if not top_ns:
        raise ValueError(f'유효한 Top-N이 없습니다. 요청값={args.top_ns}, checkpoint 수={len(ranked)}')
    df_d2_te = c.DF_TEST[c.DF_TEST['domain'] == 'D2']
    df_d3_te = c.DF_TEST[c.DF_TEST['domain'] == 'D3']
    SEP  = '─' * 80
    SEP2 = '═' * 80
    print(SEP2)
    print('  267_K-Fold_ValidTopK_Standalone — K-Fold soft-voting 평가')
    print(SEP2)
    print(f"  Device       : {device}")
    print(f"  Checkpoints  : {len(ranked)} D3 fold models")
    print(f"  D2 folds     : {len(d2_ranked)} D2 fold models, sorted by D2_fold_val")
    print(f"  Rank metric  : {args.rank_metric} (test metric used for ranking: False)")
    print(f"  Top-N        : {', '.join(map(str, top_ns))}")
    print(f"  D2 Top-N     : {', '.join(map(str, d2_top_ns))}")
    print(f"  Test         : D2={len(df_d2_te)} samples, D3={len(df_d3_te)} samples")
    print(f"  Ranks        : D2={args.d2_rank}, D3={args.d3_rank}\n")

    collect_fn = collect_probs_variable_len
    d2_fold_probs = []
    d2_fold_targets = None
    probs_d2, probs_d3 = [], []
    d2_targets = d3_targets = None

    print(SEP2)
    print('  평가 모드: VARIABLE_LEN')
    print(SEP2)

    print("  D2@D2 K-Fold ensemble (D2_fold_val 기준 정렬, active_domain_count=1)")
    for i, summary in enumerate(d2_ranked, 1):
        label = f"d2f{summary.get('fold', '?')}"
        print(f"  [D2 {i:02d}/{len(d2_ranked):02d}] load {label} | D2_fold_val={float(summary.get('D2_fold_val', 0.0)):.2f} | {summary['ckpt']}")
        model = _load_model(base_dir, summary['ckpt'], args.d2_rank, args.d3_rank, 1, device)
        p2, t2 = collect_fn(model, df_d2_te, device)
        d2_fold_probs.append(p2)
        d2_fold_targets = t2
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    d2_top_results = {}
    print(f"\n  {SEP}")
    print("  D2@D2 K-Fold ensemble summary (variable_len)")
    print(f"  {SEP}")
    print(f"  {'TopN':>5} {'D2@D2':>9} {'SelectionD2Val':>15} {'Members':>20}")
    print(f"  {'─'*68}")
    for n in d2_top_ns:
        mean_d2 = np.mean(np.stack(d2_fold_probs[:n], axis=0), axis=0)
        d2_preds = mean_d2.argmax(axis=1)
        d2_at_d2 = macro_accuracy(d2_preds, d2_fold_targets)
        d2_per_class = per_class_accuracy(d2_preds, d2_fold_targets)
        selection_d2_val = float(np.mean([s.get('D2_fold_val', 0.0) for s in d2_ranked[:n]]))
        members = [f"d2f{s.get('fold', '?')}" for s in d2_ranked[:n]]
        d2_top_results[str(n)] = {
            'D2_at_D2': d2_at_d2,
            'SelectionD2FoldVal': selection_d2_val,
            'per_class': d2_per_class,
            'members': [s['ckpt'] for s in d2_ranked[:n]],
        }
        print(f"  {n:>5} {d2_at_d2:>8.2f}% {selection_d2_val:>14.2f}% {','.join(members):>20}")
    print()

    fixed_d2_top_n = min(5, len(d2_ranked))
    fixed_d2_at_d2 = d2_top_results[str(fixed_d2_top_n)]['D2_at_D2']
    fixed_d2_selection_val = d2_top_results[str(fixed_d2_top_n)]['SelectionD2FoldVal']

    print("  D3 K-Fold ensemble (ValAcc 기준 정렬, active_domain_count=2)")
    for i, summary in enumerate(ranked, 1):
        label = f"d2f{summary.get('d2_fold', '?')}_d3f{summary.get('d3_fold', '?')}"
        print(f"  [{i:02d}/{len(ranked):02d}] load {label} | {args.rank_metric}={float(summary.get(args.rank_metric, 0.0)):.2f} | {summary['ckpt']}")
        model = _load_model(base_dir, summary['ckpt'], args.d2_rank, args.d3_rank, 2, device)
        p2, t2 = collect_fn(model, df_d2_te, device)
        p3, t3 = collect_fn(model, df_d3_te, device)
        probs_d2.append(p2)
        probs_d3.append(p3)
        d2_targets = t2
        d3_targets = t3
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = {
        'ranking_metric': args.rank_metric,
        'ranking_definition': 'validation metric from fold summary; default ValAcc=(D2_fold_val_after_D3 + D3_fold_val)/2',
        'test_metrics_used_for_ranking': False,
        'num_checkpoints': len(ranked),
        'd2_ranking_metric': 'D2_fold_val',
        'd2_ranking_definition': 'D2 fold validation macro accuracy after D2 training',
        'd2_num_checkpoints': len(d2_ranked),
        'd2_top_n': d2_top_results,
        'top_n': {},
    }

    print(f"\n  {SEP}")
    print("  K-Fold ensemble summary (variable_len)")
    print(f"  D2@D2 fixed: D2_fold_val 기준 Top-{fixed_d2_top_n} D2 ensemble = {fixed_d2_at_d2:.2f}%")
    print(f"  {SEP}")
    print(f"  {'TopN':>5} {'D2@D2':>9} {'D2@D3':>9} {'D3@D3':>9} {'Acc':>9} {'Fr':>9} {'SelVal':>9}")
    print(f"  {'─'*72}")
    for n in top_ns:
        mean_d2 = np.mean(np.stack(probs_d2[:n], axis=0), axis=0)
        mean_d3 = np.mean(np.stack(probs_d3[:n], axis=0), axis=0)
        d2_preds = mean_d2.argmax(axis=1)
        d3_preds = mean_d3.argmax(axis=1)
        d2_at_d3 = macro_accuracy(d2_preds, d2_targets)
        d3_at_d3 = macro_accuracy(d3_preds, d3_targets)
        d2_at_d2 = fixed_d2_at_d2
        d2_at_d2_selected_d3_mean = float(np.mean([s.get('D2_at_D2', 0.0) for s in ranked[:n]]))
        acc = (d2_at_d3 + d3_at_d3) / 2.0
        fr = d2_at_d2 - d2_at_d3
        selection_val = float(np.mean([s.get(args.rank_metric, 0.0) for s in ranked[:n]]))
        selection_val_fr = float(np.mean([s.get('ValFr', 0.0) for s in ranked[:n]]))
        d2_per_class = per_class_accuracy(d2_preds, d2_targets)
        d3_per_class = per_class_accuracy(d3_preds, d3_targets)
        results['top_n'][str(n)] = {
            'D2_at_D2': d2_at_d2,
            'D2_at_D2_source': f'fixed D2 fold soft-voting ensemble sorted by D2_fold_val, Top-{fixed_d2_top_n}',
            'D2_at_D2_fixed_top_n': fixed_d2_top_n,
            'D2_at_D2_fixed_selection_D2FoldVal': fixed_d2_selection_val,
            'D2_at_D2_selected_d3_scalar_mean': d2_at_d2_selected_d3_mean,
            'D2_at_D3': d2_at_d3,
            'D3_at_D3': d3_at_d3,
            'Acc': acc,
            'Fr': fr,
            'SelectionMetricMean': selection_val,
            'SelectionValFr': selection_val_fr,
            'per_class': {
                'D2_at_D3': d2_per_class,
                'D3_at_D3': d3_per_class,
                'Acc': _average_per_class_accuracy(d2_per_class, d3_per_class),
            },
            'members': [s['ckpt'] for s in ranked[:n]],
        }
        print(f"  {n:>5} {d2_at_d2:>8.2f}% {d2_at_d3:>8.2f}% {d3_at_d3:>8.2f}% {acc:>8.2f}% {fr:>+8.2f}p {selection_val:>8.2f}%")
    print()

    result_dir = os.path.join(base_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)
    TIMESTR = time.strftime('%Y%m%d-%H%M%S')
    out_path = os.path.join(result_dir, f'eval_kfold_{TIMESTR}.json')
    save = {
        'timestamp': TIMESTR,
        'method': '267_K-Fold_ValidTopK_Standalone',
        'eval_scope': 'kfold',
        'test_time_domain_info': False,
        'leakage': False,
        'ranking_metric': args.rank_metric,
        'test_metrics_used_for_ranking': False,
        'args': {
            'd2_rank': args.d2_rank,
            'd3_rank': args.d3_rank,
            'inference': 'variable_len',
            'top_ns': top_ns,
            'ckpt_root': args.ckpt_root,
        },
        'checkpoints': {
            'num_d2': len(d2_ranked),
            'num_d3': len(ranked),
            'd2_ranked_by_D2_fold_val': [s['ckpt'] for s in d2_ranked],
            'ranked_members': [s['ckpt'] for s in ranked],
        },
        'results': results,
    }
    with open(out_path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"Results saved → {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description='K-Fold/Single 포괄 평가 (Macro+Per-class)')
    ap.add_argument('--dataset', default=None,
                    help='데이터셋 root 경로. 미지정 시 실험 root 옆 ../data 로 폴백.')
    ap.add_argument('--eval_scope', default='kfold', choices=['auto', 'kfold', 'single'],
                    help='auto는 K-Fold summary가 있으면 kfold, 없으면 single 실행')
    ap.add_argument('--ckpt_root', default='checkpoint/ensemble')
    ap.add_argument('--top_ns', default=None, help='K-Fold ensemble Top-N 목록. 예: 5,10,15,20,25')
    ap.add_argument('--rank_metric', default='ValAcc',
                    help='K-Fold checkpoint 정렬 기준. 기본 ValAcc')
    ap.add_argument('--d2_ckpt', default='checkpoint/ensemble/d2.pth')
    ap.add_argument('--d3_ckpt', default='checkpoint/ensemble/d3.pth')
    ap.add_argument('--d2_rank', type=int, default=128)
    ap.add_argument('--d3_rank', type=int, default=64)
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                    help='auto uses CUDA only when enough free memory is available')
    args = ap.parse_args()

    base_dir = REPO_ROOT
    device = _select_device(args.device)

    scope = args.eval_scope
    if scope == 'auto':
        scope = 'kfold' if _discover_kfold_summaries(base_dir, args.ckpt_root) else 'single'
    if scope == 'kfold':
        return run_kfold_eval(args, base_dir, device)
    return run_single_eval(args, base_dir, device)


if __name__ == '__main__':
    main()
