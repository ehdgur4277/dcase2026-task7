"""
main.py — 통합 파이프라인 (end-to-end, 앙상블)

D1 backbone 로드 → D2 K-Fold 학습 → D3 K-Fold 학습 → Soft-Voting 앙상블 평가까지
하나로 연결한다 (reorg_prompt_2.md: model/script/main.py).

경로 규칙 (모두 스크립트 위치 기준 상대경로):
  - 체크포인트 : ../checkpoint/ensemble/
  - 로그/요약  : ../result/  (master 로그 = ensemble_log.txt)
  - 데이터셋   : --dataset [path]  (미지정 시 실험 root 옆 ../data 로 폴백)

모델 정의/로더는 common/model.py 의 공용 구현을 import 한다 (중복 코드 금지).
D1 체크포인트는 환경변수 D1_CKPT 또는 ../checkpoint/[BN/]checkpoint_D1.pth 에서 찾는다.

P_SVD 환경변수:
  - 미설정 → 모든 nonzero σ 유지 (ρ_svd ≈ 1.0)
  - 0 < P_SVD ≤ 1 → 누적 에너지 비율 컷오프 적용
"""

import os, sys, time, json, copy, traceback, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) in ('script', 'scripts') else SCRIPT_DIR
EXP_ROOT = os.path.dirname(REPO_ROOT)          # 실험 root (Chun_Chosun_task7_1)
COMMON = os.path.join(REPO_ROOT, 'common')
sys.path.insert(0, COMMON)

# 데이터셋 root 옆에 data/ 가 있는 일반적인 배치를 기본값으로 둔다 (포터블 상대경로).
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
from trainer import AudioDataset, FocalLoss, train_phase
from model import build_model as _build_model, freeze as _freeze

# 앙상블 Top-K 는 하이퍼파라미터로 고정한다 (기본 20). dev-test 로 K 를 고르지 않는다.
ENSEMBLE_TOP_K = int(os.environ.get('ENSEMBLE_TOP_K', 20))


def _macro_accuracy(preds, targets):
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    class_accs = []
    for cls in range(c.NUM_CLASSES):
        mask = targets == cls
        if np.any(mask):
            class_accs.append(float(np.mean(preds[mask] == cls)))
    return float(np.mean(class_accs) * 100.0) if class_accs else 0.0


def _per_class_accuracy(preds, targets):
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    inv_labels = {v: k for k, v in getattr(c, 'CLASS_LABELS', {}).items()}
    per_class = {}
    for cls in range(c.NUM_CLASSES):
        mask = targets == cls
        total = int(np.sum(mask))
        correct = int(np.sum(preds[mask] == cls)) if total else 0
        per_class[inv_labels.get(cls, str(cls))] = {
            'correct': correct,
            'total': total,
            'acc': float(correct / total * 100.0) if total else 0.0,
        }
    return per_class


def _average_per_class_accuracy(left, right):
    keys = sorted(set(left) | set(right))
    out = {}
    for key in keys:
        l = left.get(key, {'acc': 0.0, 'correct': 0, 'total': 0})
        r = right.get(key, {'acc': 0.0, 'correct': 0, 'total': 0})
        out[key] = {
            'acc': float((l['acc'] + r['acc']) / 2.0),
            'D2_acc': float(l['acc']),
            'D3_acc': float(r['acc']),
            'D2_correct': int(l['correct']),
            'D2_total': int(l['total']),
            'D3_correct': int(r['correct']),
            'D3_total': int(r['total']),
        }
    return out


@torch.no_grad()
def evaluate_macro(model, test_df, device):
    """DCASE Task7 공식 기준: 클래스별 accuracy 평균(macro)."""
    model.eval()
    ds = AudioDataset(test_df, is_train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=c.NUM_WORKERS)
    preds, targets = [], []
    for audio, target, _, _ in loader:
        audio = audio.float().to(device)
        logits, _ = model(audio)
        probs = F.softmax(logits, dim=-1)
        preds.append(int(probs.argmax(-1).item()))
        targets.append(int(target.argmax(-1).item()))
    return _macro_accuracy(preds, targets)


@torch.no_grad()
def collect_probs(model, test_df, device):
    """Soft-voting용 확률 수집: [N, NUM_CLASSES]."""
    model.eval()
    ds = AudioDataset(test_df, is_train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=c.NUM_WORKERS)
    probs_all, targets = [], []
    for audio, target, _, _ in loader:
        audio = audio.float().to(device)
        logits, _ = model(audio)
        probs_all.append(F.softmax(logits, dim=-1).squeeze(0).cpu().numpy())
        targets.append(int(target.argmax(-1).item()))
    return np.stack(probs_all), np.asarray(targets)


def stratified_kfold_indices(df, k=5, seed=42):
    """sklearn 없이 클래스별 index를 seed로 섞은 뒤 라운드로빈 fold 배정."""
    rng = np.random.RandomState(seed)
    fold_bins = [[] for _ in range(k)]
    for _, grp in df.groupby('new_target'):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        for pos, row_idx in enumerate(idx):
            fold_bins[pos % k].append(row_idx)

    all_idx = set(df.index.tolist())
    folds = []
    for fold in range(k):
        val_idx = list(fold_bins[fold])
        train_idx = sorted(all_idx - set(val_idx))
        folds.append((train_idx, val_idx))
    return folds


def _parse_psvd():
    p_svd_env = os.environ.get('P_SVD', '').strip()
    p_svd = float(p_svd_env) if p_svd_env else None
    if p_svd is not None and not (0.0 < p_svd <= 1.0):
        raise ValueError(f'P_SVD must be in (0, 1], got {p_svd}')

    p_svd_d1_env = os.environ.get('P_SVD_D1', '').strip()
    p_svd_d1 = float(p_svd_d1_env) if p_svd_d1_env else (p_svd if p_svd is not None else 0.99)
    p_svd_d2_env = os.environ.get('P_SVD_D2', '').strip()
    p_svd_d2 = float(p_svd_d2_env) if p_svd_d2_env else (p_svd if p_svd is not None else 0.99)
    return p_svd, p_svd_d1, p_svd_d2


def _seed_everything(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)


def _make_log(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
    return log


def _common_train_config():
    d1_cfg = {'kd_alpha': 0.1, 'kd_temp': 2.0, 'kd_feat_weight': 0.0}
    d2_cfg = {
        'kd_alpha':       c.MODE_CONFIG['cllora']['kd_alpha'],
        'kd_temp':        c.MODE_CONFIG['cllora']['kd_temp'],
        'kd_feat_weight': c.MODE_CONFIG['cllora']['kd_feat_weight'],
    }
    return d1_cfg, d2_cfg, c.MODE_CONFIG['cllora']['lr_d3']


def _maybe_clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_d2_fold(task):
    gpu_id = task.get('gpu_id')
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    base_dir = task['base_dir']
    fold = task['fold']
    out_dir = task['out_dir']
    log = _make_log(task.get('log_path') or os.path.join(out_dir, 'train.log'))
    mode = task['mode']

    try:
        os.makedirs(out_dir, exist_ok=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _seed_everything(1193 + fold, device)
        p_svd, p_svd_d1, _ = _parse_psvd()
        d2_lora_rank = task['d2_lora_rank']
        d3_lora_rank = task['d3_lora_rank']
        d2_epochs = task['d2_epochs']

        log(f"Device: {device} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        log(f"Mode: {mode} | D2 fold={fold} | p_svd={p_svd}, p_svd_d1={p_svd_d1}")

        df_d2 = c.DF_TRAIN[c.DF_TRAIN['domain'] == 'D2']
        folds = stratified_kfold_indices(df_d2, k=task['k'], seed=task['fold_seed'])
        train_idx, val_idx = folds[fold]
        df_train = df_d2.loc[train_idx]
        df_val = df_d2.loc[val_idx]
        df_d2_te = c.DF_TEST[c.DF_TEST['domain'] == 'D2']

        log(f"Fold split: train_rows={len(df_train)}, val_rows={len(df_val)}")
        model = _build_model(base_dir, device, d2_lora_rank, d3_lora_rank, log)
        d1_teacher = _freeze(copy.deepcopy(model))
        d1_teacher.active_domain_count = 0

        log("\n========== Building D1 Orthogonal Projection Bases ==========")
        model.build_d1_bases(projection_rank=d2_lora_rank, p_svd=p_svd_d1)

        model.active_domain_count = 1
        model.set_d2_trainable()
        model.install_orth_hooks(domain_idx=0)

        d1_cfg, _, _ = _common_train_config()
        c.EPOCHS = d2_epochs
        train_phase(
            model, df_train, device, log,
            phase_name=f'D2 fold {fold} (LoRA_D2 + D1-KD + Orth)',
            lr=c.LR_D2,
            teacher_model=d1_teacher,
            mode_cfg=d1_cfg,
        )
        model.clear_orth_hooks()

        d2_at_d2 = evaluate_macro(model, df_d2_te, device)
        d2_val = evaluate_macro(model, df_val, device) if len(df_val) else 0.0
        ckpt_path = os.path.join(out_dir, 'd2.pth')
        payload = {
            'model': model.state_dict(),
            'mode': mode,
            'phase': 'after_D2',
            'fold': fold,
            'D2_at_D2': d2_at_d2,
            'D2_fold_val': d2_val,
            'd2_lora_rank': d2_lora_rank,
            'd3_lora_rank': d3_lora_rank,
            'p_svd_d1': p_svd_d1,
            'fold_seed': task['fold_seed'],
        }
        torch.save(payload, ckpt_path)
        summary = {k: v for k, v in payload.items() if k != 'model'}
        summary['ckpt'] = ckpt_path
        with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        log(f"D2 fold {fold} done | D2@D2={d2_at_d2:.2f}% | val={d2_val:.2f}%")
        _maybe_clear_cuda()
        return True, f'd2f{fold}', summary
    except Exception as exc:
        log(f"FAILED d2f{fold}: {exc}")
        log(traceback.format_exc())
        _maybe_clear_cuda()
        return False, f'd2f{fold}', {'error': repr(exc)}


def train_d3_combo(task):
    gpu_id = task.get('gpu_id')
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    base_dir = task['base_dir']
    d2_fold = task['d2_fold']
    d3_fold = task['d3_fold']
    out_dir = task['out_dir']
    log = _make_log(task.get('log_path') or os.path.join(out_dir, 'train.log'))
    mode = task['mode']

    try:
        os.makedirs(out_dir, exist_ok=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _seed_everything(2193 + d2_fold * 10 + d3_fold, device)
        _, p_svd_d1, p_svd_d2 = _parse_psvd()
        d2_lora_rank = task['d2_lora_rank']
        d3_lora_rank = task['d3_lora_rank']
        d3_epochs = task['d3_epochs']

        log(f"Device: {device} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        log(f"Mode: {mode} | D2 fold={d2_fold}, D3 fold={d3_fold}")
        log(f"p_svd_d1={p_svd_d1}, p_svd_d2={p_svd_d2}")

        df_d2 = c.DF_TRAIN[c.DF_TRAIN['domain'] == 'D2']
        df_d3 = c.DF_TRAIN[c.DF_TRAIN['domain'] == 'D3']
        d2_folds = stratified_kfold_indices(df_d2, k=task['k'], seed=task['fold_seed'])
        _, d2_val_idx = d2_folds[d2_fold]
        df_d2_val = df_d2.loc[d2_val_idx]

        folds = stratified_kfold_indices(df_d3, k=task['k'], seed=task['fold_seed'])
        train_idx, val_idx = folds[d3_fold]
        df_train = df_d3.loc[train_idx]
        df_val = df_d3.loc[val_idx]
        df_d2_te = c.DF_TEST[c.DF_TEST['domain'] == 'D2']
        df_d3_te = c.DF_TEST[c.DF_TEST['domain'] == 'D3']
        log(
            f"Fold split: D3 train_rows={len(df_train)}, D3 val_rows={len(df_val)}, "
            f"D2 val_rows={len(df_d2_val)}"
        )

        model = _build_model(base_dir, device, d2_lora_rank, d3_lora_rank, log)
        log("\n========== Building D1 Orthogonal Projection Bases ==========")
        model.build_d1_bases(projection_rank=d2_lora_rank, p_svd=p_svd_d1)

        d2_payload = torch.load(task['d2_ckpt'], map_location=device, weights_only=False)
        model.load_state_dict(d2_payload['model'])
        d2_at_d2 = float(d2_payload['D2_at_D2'])
        d2_fold_val_before = float(d2_payload.get('D2_fold_val', 0.0))

        d2_teacher = _freeze(copy.deepcopy(model))
        d2_teacher.active_domain_count = 1

        log("\n========== Building D2 Delta Bases (LoRA Subtraction) ==========")
        model.build_d2_delta_bases(p_svd=p_svd_d2)

        model.active_domain_count = 2
        model.set_d3_trainable()
        model.install_d1_d2_delta_hooks(domain_idx=1)

        _, d2_cfg, lr_d3 = _common_train_config()
        c.EPOCHS = d3_epochs
        try:
            train_phase_cosine_feature_kd(
                model, df_train, device, log,
                phase_name=f'D3 d2f{d2_fold}_d3f{d3_fold} (LoRA_D3 + D2-KD + DeltaGP + CosineFeatKD)',
                lr=lr_d3,
                teacher_model=d2_teacher,
                mode_cfg=d2_cfg,
            )
        except RuntimeError as exc:
            if 'out of memory' not in str(exc).lower():
                raise
            log(f"OOM during D3 training; saving current state: {exc}")
            _maybe_clear_cuda()
        model.clear_orth_hooks()

        d2_at_d3 = evaluate_macro(model, df_d2_te, device)
        d3_at_d3 = evaluate_macro(model, df_d3_te, device)
        d2_fold_val_after = evaluate_macro(model, df_d2_val, device) if len(df_d2_val) else 0.0
        d3_val = evaluate_macro(model, df_val, device) if len(df_val) else 0.0
        acc = (d2_at_d3 + d3_at_d3) / 2.0
        fr = d2_at_d2 - d2_at_d3
        val_acc = (d2_fold_val_after + d3_val) / 2.0
        val_fr = d2_fold_val_before - d2_fold_val_after

        ckpt_path = os.path.join(out_dir, 'd3.pth')
        payload = {
            'model': model.state_dict(),
            'mode': mode,
            'phase': 'after_D3',
            'd2_fold': d2_fold,
            'd3_fold': d3_fold,
            'D2_at_D2': d2_at_d2,
            'D2_at_D3': d2_at_d3,
            'D3_at_D3': d3_at_d3,
            'D2_fold_val_before': d2_fold_val_before,
            'D2_fold_val_after_D3': d2_fold_val_after,
            'D3_fold_val': d3_val,
            'ValAcc': val_acc,
            'ValFr': val_fr,
            'Acc': acc,
            'Fr': fr,
            'd2_lora_rank': d2_lora_rank,
            'd3_lora_rank': d3_lora_rank,
            'p_svd_d1': p_svd_d1,
            'p_svd_d2': p_svd_d2,
            'fold_seed': task['fold_seed'],
        }
        torch.save(payload, ckpt_path)
        summary = {k: v for k, v in payload.items() if k != 'model'}
        summary['ckpt'] = ckpt_path
        with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        log(
            f"D3 combo done | D2@D2={d2_at_d2:.2f}% D2@D3={d2_at_d3:.2f}% "
            f"D3@D3={d3_at_d3:.2f}% Acc={acc:.2f}% Fr={fr:.2f}%p "
            f"ValAcc={val_acc:.2f}% ValFr={val_fr:.2f}%p "
            f"D2val={d2_fold_val_after:.2f}% D3val={d3_val:.2f}%"
        )
        _maybe_clear_cuda()
        return True, f'd2f{d2_fold}_d3f{d3_fold}', summary
    except Exception as exc:
        log(f"FAILED d2f{d2_fold}_d3f{d3_fold}: {exc}")
        log(traceback.format_exc())
        _maybe_clear_cuda()
        return False, f'd2f{d2_fold}_d3f{d3_fold}', {'error': repr(exc)}


def _d2_fold_ensemble_d2_at_d2(d2_summaries, base_dir, device, df_d2_te,
                               d2_lora_rank, d3_lora_rank, log):
    """D2 fold 체크포인트 전체(active_domain_count=1)를 soft-voting → D2@D2.

    evaluate.py / inference.py 와 동일한 정의로 통일한다 (스칼라 평균이 아니라
    D2 fold soft-voting 앙상블). soft-voting 은 순서 무관이라 정렬은 로그 표시용.
    """
    ranked = sorted(d2_summaries, key=lambda x: float(x.get('D2_fold_val', -1.0)), reverse=True)
    probs, targets = [], None
    for i, s in enumerate(ranked, 1):
        log(f"D2@D2 fold ensemble load {i}/{len(ranked)}: {s['ckpt']} "
            f"(D2_fold_val={float(s.get('D2_fold_val', 0.0)):.2f})")
        model = _build_model(base_dir, device, d2_lora_rank, d3_lora_rank, log)
        ckpt = torch.load(s['ckpt'], map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.active_domain_count = 1
        p, t = collect_probs(model, df_d2_te, device)
        probs.append(p)
        targets = t
        del model
        _maybe_clear_cuda()
    mean_p = np.mean(np.stack(probs, axis=0), axis=0)
    preds = mean_p.argmax(axis=1)
    return _macro_accuracy(preds, targets), _per_class_accuracy(preds, targets), len(ranked)


def ensemble_evaluate(d3_summaries, d2_summaries, base_dir, result_dir,
                      d2_lora_rank, d3_lora_rank, log, top_k=ENSEMBLE_TOP_K):
    if not d3_summaries:
        log("No D3 checkpoints for ensemble evaluation.")
        return {}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df_d2_te = c.DF_TEST[c.DF_TEST['domain'] == 'D2']
    df_d3_te = c.DF_TEST[c.DF_TEST['domain'] == 'D3']

    # 모델 선택은 validation(fold-val) 기반 ValAcc 로만 수행한다 (dev-test 미사용).
    # K 는 하이퍼파라미터로 고정(기본 20); dev-test 는 이 고정된 Top-K 가 정해진 뒤
    # 최종 점수 산출에만 1회 사용한다 → Top-K 선택에 test 누수 없음.
    ranked = sorted(d3_summaries, key=lambda x: float(x.get('ValAcc', -1.0)), reverse=True)
    n = min(top_k, len(ranked))
    selected = ranked[:n]
    log(f"Ensemble Top-{n} fixed (K={top_k}), selected by ValAcc; "
        f"dev-test used ONLY for final scoring of this fixed set.")

    probs_d2, probs_d3 = [], []
    d2_targets = d3_targets = None
    for i, summary in enumerate(selected, 1):
        log(f"Ensemble load {i}/{n}: {summary['ckpt']}")
        model = _build_model(base_dir, device, d2_lora_rank, d3_lora_rank, log)
        ckpt = torch.load(summary['ckpt'], map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.active_domain_count = 2
        p2, t2 = collect_probs(model, df_d2_te, device)
        p3, t3 = collect_probs(model, df_d3_te, device)
        probs_d2.append(p2)
        probs_d3.append(p3)
        d2_targets = t2
        d3_targets = t3
        del model
        _maybe_clear_cuda()

    # 고정된 Top-K 한 세트만 dev-test 로 최종 평가한다 (N 스윕/선택 없음 → 누수 없음).
    mean_d2 = np.mean(np.stack(probs_d2, axis=0), axis=0)
    mean_d3 = np.mean(np.stack(probs_d3, axis=0), axis=0)
    d2_preds = mean_d2.argmax(axis=1)
    d3_preds = mean_d3.argmax(axis=1)
    d2_at_d3 = _macro_accuracy(d2_preds, d2_targets)
    d3_at_d3 = _macro_accuracy(d3_preds, d3_targets)
    # D2@D2: D2 fold 체크포인트 전체(count=1) soft-voting 앙상블로 통일 (evaluate.py/inference.py 와 동일).
    if d2_summaries:
        d2_at_d2, d2_at_d2_per_class, n_d2_folds = _d2_fold_ensemble_d2_at_d2(
            d2_summaries, base_dir, device, df_d2_te, d2_lora_rank, d3_lora_rank, log)
        d2_at_d2_method = f'D2 fold soft-voting ensemble (count=1), all {n_d2_folds} folds'
    else:
        d2_at_d2 = float(np.mean([s['D2_at_D2'] for s in selected]))
        d2_at_d2_per_class, n_d2_folds = {}, 0
        d2_at_d2_method = 'scalar mean of selected D3 members\' inherited D2_at_D2 (fallback)'
    acc = (d2_at_d3 + d3_at_d3) / 2.0
    fr = d2_at_d2 - d2_at_d3
    selection_val_acc = float(np.mean([s.get('ValAcc', 0.0) for s in selected]))
    selection_val_fr = float(np.mean([s.get('ValFr', 0.0) for s in selected]))
    d2_per_class = _per_class_accuracy(d2_preds, d2_targets)
    d3_per_class = _per_class_accuracy(d3_preds, d3_targets)

    results = {
        'ranking_metric': 'ValAcc',
        'ranking_definition': '(D2_fold_val_after_D3 + D3_fold_val) / 2',
        'test_metrics_used_for_ranking': False,
        'test_metrics_used_for_topk_selection': False,
        'topk_fixed': int(top_k),
        'top_k': int(n),
        'num_checkpoints': len(ranked),
        'D2_at_D2': d2_at_d2,
        'D2_at_D2_method': d2_at_d2_method,
        'D2_at_D2_folds_used': n_d2_folds,
        'D2_at_D3': d2_at_d3,
        'D3_at_D3': d3_at_d3,
        'Acc': acc,
        'Fr': fr,
        'SelectionValAcc': selection_val_acc,
        'SelectionValFr': selection_val_fr,
        'per_class': {
            'D2_at_D2': d2_at_d2_per_class,
            'D2_at_D3': d2_per_class,
            'D3_at_D3': d3_per_class,
            'Acc': _average_per_class_accuracy(d2_per_class, d3_per_class),
        },
        'members': [s['ckpt'] for s in selected],
    }
    log(
        f"Top-{n} (fixed K={top_k}): D2@D2={d2_at_d2:.2f}% D2@D3={d2_at_d3:.2f}% "
        f"D3@D3={d3_at_d3:.2f}% Acc={acc:.2f}% Fr={fr:.2f}%p "
        f"SelectionValAcc={selection_val_acc:.2f}% SelectionValFr={selection_val_fr:.2f}%p"
    )
    log(f"Top-{n} per-class ensemble Acc:")
    for cls_name, cls_metrics in results['per_class']['Acc'].items():
        log(
            f"  {cls_name}: Acc={cls_metrics['acc']:.2f}% "
            f"D2@D3={cls_metrics['D2_acc']:.2f}% "
            f"({cls_metrics['D2_correct']}/{cls_metrics['D2_total']}), "
            f"D3@D3={cls_metrics['D3_acc']:.2f}% "
            f"({cls_metrics['D3_correct']}/{cls_metrics['D3_total']})"
        )

    out_path = os.path.join(result_dir, 'ensemble_summary.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"Ensemble summary saved: {out_path}")
    return results


def write_metric_json(ensemble, result_dir, log):
    """고정 Top-K 앙상블 결과에서 macro accuracy 기준 metric.json 을 작성.

    Top-K 는 ValAcc(validation)로 선택된 고정값이므로 dev-test 로 N 을 고르지 않는다.
    여기서 dev-test 기반 Acc 는 '선택이 끝난 뒤'의 최종 점수로만 기록된다.
    """
    if not ensemble or 'Acc' not in ensemble:
        return None
    metric = {
        'metric': 'macro_accuracy',
        'unit': 'percent',
        'selected_top_k': int(ensemble.get('top_k', 0)),
        'topk_selection': 'fixed_by_ValAcc (no test-set tuning)',
        'D2_at_D2': ensemble['D2_at_D2'],
        'D2_at_D3': ensemble['D2_at_D3'],
        'D3_at_D3': ensemble['D3_at_D3'],
        'Acc': ensemble['Acc'],
        'Forgetting': ensemble['Fr'],
        'per_class': ensemble['per_class'],
    }
    out_path = os.path.join(result_dir, 'metric.json')
    with open(out_path, 'w') as f:
        json.dump(metric, f, indent=2)
    log(f"metric.json saved: {out_path} (Top-{metric['selected_top_k']}, Acc={ensemble['Acc']:.2f}%)")
    return out_path


def train_phase_cosine_feature_kd(model, train_df, device, log_fn, phase_name='',
                                  lr=1e-3, teacher_model=None, mode_cfg=None):
    """Cosine Feature KD 학습 루프 (D3 단계)."""
    log_fn(f"\n========== Phase: {phase_name} ==========")
    ds = AudioDataset(train_df, is_train=True)
    log_fn(f"  Train samples: {len(ds)}")

    if c.USE_BALANCED_SAMPLING:
        loader = DataLoader(
            ds, batch_size=c.BATCH_SIZE, sampler=ds.get_balanced_sampler(),
            num_workers=4, pin_memory=True, drop_last=True
        )
    else:
        loader = DataLoader(
            ds, batch_size=c.BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    log_fn(f"  Trainable params: {sum(p.numel() for p in trainable):,}, LR: {lr}")
    log_fn(f"  KD: {'ON' if teacher_model else 'OFF'}, Feature KD: cosine distance")

    optim = torch.optim.AdamW(trainable, lr=lr, weight_decay=c.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=c.EPOCHS, eta_min=1e-6)
    criterion = FocalLoss(gamma=c.FOCAL_GAMMA) if c.USE_FOCAL_LOSS else nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(1, c.EPOCHS + 1):
        model.train()
        model.backbone.set_bn_eval()
        sum_cls, sum_kd, sum_feat, n = 0.0, 0.0, 0.0, 0

        for audio, target, _, _ in loader:
            audio = audio.float().to(device)
            target_idx = target.float().to(device).argmax(-1)

            if c.USE_MIXUP and np.random.random() < 0.5:
                lam = np.random.beta(c.MIXUP_ALPHA, c.MIXUP_ALPHA)
                idx = torch.randperm(audio.size(0), device=device)
                audio_mixed = lam * audio + (1 - lam) * audio[idx]
                ya, yb = target_idx, target_idx[idx]
            else:
                audio_mixed, ya, yb, lam = audio, target_idx, target_idx, 1.0

            logits, feat = model(audio_mixed, use_spec_aug=c.USE_SPEC_AUG)
            loss_cls = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            loss = loss_cls

            if teacher_model is not None:
                kd_alpha = mode_cfg.get('kd_alpha', c.KD_ALPHA) if mode_cfg else c.KD_ALPHA
                kd_temp  = mode_cfg.get('kd_temp',  c.KD_TEMP)  if mode_cfg else c.KD_TEMP
                kd_fw    = mode_cfg.get('kd_feat_weight', c.KD_FEAT_WEIGHT) if mode_cfg else c.KD_FEAT_WEIGHT
                teacher_model.eval()
                with torch.no_grad():
                    t_logits, t_feat = teacher_model(audio_mixed)
                kd_loss   = F.kl_div(
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
            log_fn(
                f"  Epoch {epoch:3d}/{c.EPOCHS} | cls={sum_cls/n:.4f}{extra} "
                f"lr={optim.param_groups[0]['lr']:.6f}"
            )


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='통합 파이프라인: D2/D3 K-Fold 학습 → Soft-Voting 앙상블 (end-to-end)'
    )
    parser.add_argument('--dataset', default=None,
                        help='데이터셋 root 경로. 미지정 시 실험 root 옆 ../data 로 폴백.')
    parser.parse_args()   # --dataset 은 모듈 로드시 이미 반영됨; --help/검증용.

    mode = 'kfold_validtopk_cllora_d2rank128_d3rank64_coskd_d3_deltaGP_psvd'
    d2_lora_rank = int(os.environ.get('D2_LORA_RANK', 128))
    d3_lora_rank = int(os.environ.get('D3_LORA_RANK', 64))
    p_svd, p_svd_d1, p_svd_d2 = _parse_psvd()
    k = int(os.environ.get('KFOLD_K', 5))
    fold_seed = int(os.environ.get('KFOLD_SEED', 42))
    # 기본 순차 실행(1개씩). GPU 경합으로 인한 idle 을 피한다.
    # 병렬을 원하면 KFOLD_MAX_WORKERS 환경변수로 늘릴 수 있다.
    max_workers = int(os.environ.get('KFOLD_MAX_WORKERS', 1))
    gpu_ids = [x.strip() for x in os.environ.get('KFOLD_GPU_IDS', '0,1').split(',') if x.strip()]
    if not gpu_ids:
        gpu_ids = [None]

    base_dir   = REPO_ROOT
    result_dir = os.path.join(base_dir, 'result')
    ckpt_dir   = os.path.join(base_dir, 'checkpoint', 'ensemble')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    TIMESTR = time.strftime('%Y%m%d-%H%M%S')
    log = _make_log(os.path.join(result_dir, 'ensemble_log.txt'))

    log(f"\n################# main.py run {TIMESTR} #################")
    log(f"Dataset dir: {DATASET_DIR}")
    log(f"Checkpoint dir (ensemble): {ckpt_dir}")
    log(f"Mode: {mode}")
    log(f"p_svd (global)={p_svd}, p_svd_d1={p_svd_d1}, p_svd_d2={p_svd_d2}")
    log(f"K-Fold: K={k}, seed={fold_seed}, max_workers={max_workers}, gpu_ids={gpu_ids}")
    log(f"Ensemble: fixed Top-K={ENSEMBLE_TOP_K}, selected by validation "
        f"ValAcc=(D2_fold_val_after_D3 + D3_fold_val)/2; dev-test ONLY for final scoring.")

    if os.environ.get('EPOCHS'):
        d2_epochs = d3_epochs = int(os.environ['EPOCHS'])
        log(f"Override D2/D3 epochs: {d2_epochs}")
    else:
        d2_epochs = int(os.environ.get('D2_EPOCHS', 150))
        d3_epochs = int(os.environ.get('D3_EPOCHS', 150))
    log(f"Phase epochs: D2={d2_epochs}, D3={d3_epochs}")
    log(f"LoRA ranks: D2={d2_lora_rank}, D3={d3_lora_rank}")

    d1_cfg, d2_cfg, lr_d3 = _common_train_config()
    log(f"D1→D2 KD: alpha={d1_cfg['kd_alpha']}, temp={d1_cfg['kd_temp']}, feat={d1_cfg['kd_feat_weight']}")
    log(f"D2→D3 KD: alpha={d2_cfg['kd_alpha']}, temp={d2_cfg['kd_temp']}, feat={d2_cfg['kd_feat_weight']}, lr_d3={lr_d3}")

    df_d2_tr = c.DF_TRAIN[c.DF_TRAIN['domain'] == 'D2']
    df_d3_tr = c.DF_TRAIN[c.DF_TRAIN['domain'] == 'D3']
    log(f"D2 train rows={len(df_d2_tr)}, D3 train rows={len(df_d3_tr)}")

    d2_tasks = []
    for fold in range(k):
        d2_tasks.append({
            'base_dir': base_dir,
            'mode': mode,
            'k': k,
            'fold': fold,
            'fold_seed': fold_seed,
            'gpu_id': gpu_ids[fold % len(gpu_ids)],
            'out_dir': os.path.join(ckpt_dir, f'fold_d2f{fold}'),
            'log_path': os.path.join(result_dir, f'ensemble_d2f{fold}.log'),
            'd2_lora_rank': d2_lora_rank,
            'd3_lora_rank': d3_lora_rank,
            'd2_epochs': d2_epochs,
        })

    log("\n========== Phase 1: D2 K-Fold Training ==========")
    d2_summaries = []
    with ProcessPoolExecutor(max_workers=min(k, max_workers)) as executor:
        futures = {executor.submit(train_d2_fold, task): f"d2f{task['fold']}" for task in d2_tasks}
        for fut in as_completed(futures):
            label = futures[fut]
            success, _, summary = fut.result()
            if success:
                d2_summaries.append(summary)
                log(f"[OK] {label}: D2@D2={summary['D2_at_D2']:.2f}%")
            else:
                log(f"[FAIL] {label}: {summary.get('error')}")

    d2_summaries = sorted(d2_summaries, key=lambda x: x['fold'])
    if not d2_summaries:
        raise RuntimeError('All D2 folds failed; cannot start D3 K-Fold.')

    log("\n========== Phase 2: D3 K-Fold Training ==========")
    d3_tasks = []
    task_idx = 0
    for d2_summary in d2_summaries:
        d2_fold = d2_summary['fold']
        for d3_fold in range(k):
            d3_tasks.append({
                'base_dir': base_dir,
                'mode': mode,
                'k': k,
                'd2_fold': d2_fold,
                'd3_fold': d3_fold,
                'fold_seed': fold_seed,
                'gpu_id': gpu_ids[task_idx % len(gpu_ids)],
                'out_dir': os.path.join(ckpt_dir, f'fold_d2f{d2_fold}_d3f{d3_fold}'),
                'log_path': os.path.join(result_dir, f'ensemble_d2f{d2_fold}_d3f{d3_fold}.log'),
                'd2_ckpt': d2_summary['ckpt'],
                'd2_lora_rank': d2_lora_rank,
                'd3_lora_rank': d3_lora_rank,
                'd3_epochs': d3_epochs,
            })
            task_idx += 1

    d3_summaries = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(train_d3_combo, task): f"d2f{task['d2_fold']}_d3f{task['d3_fold']}" for task in d3_tasks}
        for fut in as_completed(futures):
            label = futures[fut]
            success, _, summary = fut.result()
            if success:
                d3_summaries.append(summary)
                log(
                    f"[OK] {label}: D2@D3={summary['D2_at_D3']:.2f}% "
                    f"D3@D3={summary['D3_at_D3']:.2f}% Acc={summary['Acc']:.2f}% Fr={summary['Fr']:.2f}%p "
                    f"ValAcc={summary['ValAcc']:.2f}% ValFr={summary['ValFr']:.2f}%p"
                )
            else:
                log(f"[FAIL] {label}: {summary.get('error')}")

    d3_summaries = sorted(d3_summaries, key=lambda x: (x['d2_fold'], x['d3_fold']))

    log("\n========== Phase 3: Soft-Voting Ensemble Evaluation ==========")
    ensemble = ensemble_evaluate(d3_summaries, d2_summaries, base_dir, result_dir, d2_lora_rank, d3_lora_rank, log)
    write_metric_json(ensemble, result_dir, log)

    summary = {
        'mode': mode,
        'timestamp': TIMESTR,
        'dataset_dir': DATASET_DIR,
        'projection_method': 'delta_SVD_with_p_svd',
        'd2_projection': 'SVD(scaling * lora_B @ lora_A) + cumulative-energy cutoff',
        'ensemble_selection': {
            'ranking_metric': 'ValAcc',
            'ranking_definition': '(D2_fold_val_after_D3 + D3_fold_val) / 2',
            'test_metrics_used_for_ranking': False,
            'test_metrics_used_for_topk_selection': False,
            'top_k_fixed': ENSEMBLE_TOP_K,
        },
        'kfold': {
            'k': k,
            'seed': fold_seed,
            'split': 'manual_stratified_round_robin_by_new_target',
            'd2_and_d3_split_independently': True,
            'chunking': 'unchanged AudioDataset(is_train=True) chunk_audio()',
            'max_workers': max_workers,
            'gpu_ids': gpu_ids,
        },
        'd2_lora_rank': d2_lora_rank,
        'd3_lora_rank': d3_lora_rank,
        'd2_epochs': d2_epochs,
        'd3_epochs': d3_epochs,
        'feature_kd_type': 'cosine',
        'd3_extra_orth': 'd2_lora_delta_right_left_singular',
        'test_time_domain_info': False,
        'p_svd_d1': p_svd_d1,
        'p_svd_d2': p_svd_d2,
        'd1_kd_alpha': d1_cfg['kd_alpha'],
        'd1_kd_temp': d1_cfg['kd_temp'],
        'd1_kd_feat_weight': d1_cfg['kd_feat_weight'],
        'd2_kd_alpha': d2_cfg['kd_alpha'],
        'd2_kd_temp': d2_cfg['kd_temp'],
        'd2_kd_feat_weight': d2_cfg['kd_feat_weight'],
        'lr_d3': lr_d3,
        'd2_folds': d2_summaries,
        'd3_combos': d3_summaries,
        'ensemble': ensemble,
    }
    summary_path = os.path.join(result_dir, f'kfold_summary_psvd{p_svd_d2}_{TIMESTR}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log(f"\nSaved summary: {summary_path}")
    log(f"Checkpoints root: {ckpt_dir}")


if __name__ == '__main__':
    main()
