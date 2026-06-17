"""
common/eval_utils.py — covforget 개발셋 앙상블 평가 → metric.json.

프로토콜: variable + TTA-off, active_domain_count=2 (D3), =1 (D2 oracle).
val_macro 내림차순 정렬 후 Top-N(5/10/15/20/25) 비교 (test leakage 방지).
Fr = D2@D2(5개 D2 앙상블) - D2@D3.
"""
import json
import os

import numpy as np
import torch

import config as c
from infer_utils import load_model, preload_audio, predict_probs


def macro_acc(preds, targets):
    accs = []
    for cls in range(c.NUM_CLASSES):
        m = targets == cls
        if m.sum() > 0:
            accs.append(float((preds[m] == targets[m]).mean()))
    return float(np.mean(accs) * 100)


def _classwise(preds, targets):
    """클래스별 recall(%). 해당 도메인에 존재하는 클래스만 (이름은 CLASS_LABELS 키)."""
    inv = {v: k for k, v in c.CLASS_LABELS.items()}
    out = {}
    for cls in range(c.NUM_CLASSES):
        m = targets == cls
        if m.sum() > 0:
            out[inv[cls]] = round(float((preds[m] == targets[m]).mean()) * 100, 2)
    return out


def _probs_over(ckpts, waves, device, active, log):
    """각 ckpt 의 확률을 리스트로 (Top-N 누적용)."""
    out = []
    for i, ck in enumerate(ckpts, 1):
        log(f'  [{i:02d}/{len(ckpts)}] {os.path.basename(ck)}')
        m = load_model(ck, device, active)
        out.append(predict_probs(m, waves, device))
        del m
        if device == 'cuda':
            torch.cuda.empty_cache()
    return out


def compute_topn_metrics(d2_ckpts, d3_ckpts_sorted, device, log=print):
    """
    d2_ckpts: D2 oracle 앙상블용 (active=1)
    d3_ckpts_sorted: val_macro 내림차순 정렬된 D3 ckpt 경로 리스트 (active=2)
    """
    df_d2 = c.DF_TEST[c.DF_TEST['domain'] == 'D2']
    df_d3 = c.DF_TEST[c.DF_TEST['domain'] == 'D3']
    gt_d2 = np.array(df_d2['new_target'].tolist())
    gt_d3 = np.array(df_d3['new_target'].tolist())
    w_d2 = preload_audio(list(df_d2['full_path']), log)
    w_d3 = preload_audio(list(df_d3['full_path']), log)
    log(f'dev test: D2={len(gt_d2)}  D3={len(gt_d3)}  (variable, TTA-off)')

    # D2@D2: D2 ckpt 앙상블 (active=1)
    log('[D2 oracle] active=1')
    p_d2only = _probs_over(d2_ckpts, w_d2, device, 1, log)
    if p_d2only:
        s2_pred = (np.sum(p_d2only, axis=0) / len(p_d2only)).argmax(1)
        d2_at_d2 = macro_acc(s2_pred, gt_d2)
        s2_cw = _classwise(s2_pred, gt_d2)
    else:
        d2_at_d2, s2_cw = None, {}

    # D3 앙상블 (active=2): D2/D3 test 동시 누적
    log('[D3 ensemble] active=2')
    pl_d2 = _probs_over(d3_ckpts_sorted, w_d2, device, 2, log)
    pl_d3 = _probs_over(d3_ckpts_sorted, w_d3, device, 2, log)
    n = len(d3_ckpts_sorted)

    s3d2_pred = (np.sum(pl_d2, axis=0) / n).argmax(1)
    s3d3_pred = (np.sum(pl_d3, axis=0) / n).argmax(1)
    d2_at_d3 = macro_acc(s3d2_pred, gt_d2)
    d3_at_d3 = macro_acc(s3d3_pred, gt_d3)
    s3d2_cw = _classwise(s3d2_pred, gt_d2)
    s3d3_cw = _classwise(s3d3_pred, gt_d3)
    acc = (d2_at_d3 + d3_at_d3) / 2
    fr = round(d2_at_d2 - d2_at_d3, 4) if d2_at_d2 is not None else None

    topn = []
    for k in [x for x in (5, 10, 15, 20, 25) if x <= n] + ([n] if n not in (5, 10, 15, 20, 25) else []):
        a2 = macro_acc((np.sum(pl_d2[:k], axis=0) / k).argmax(1), gt_d2)
        a3 = macro_acc((np.sum(pl_d3[:k], axis=0) / k).argmax(1), gt_d3)
        topn.append({'n': k, 'D2@D2': round(d2_at_d2, 4) if d2_at_d2 else None,
                     'D2@D3': round(a2, 4), 'D3@D3': round(a3, 4),
                     'Acc': round((a2 + a3) / 2, 4),
                     'Fr': round(d2_at_d2 - a2, 4) if d2_at_d2 else None})

    return {
        'protocol': 'variable_len, TTA-off',
        'n_models': n,
        'D2@D2': d2_at_d2, 'D2@D3': d2_at_d3, 'D3@D3': d3_at_d3,
        'Acc': round(acc, 4), 'Fr': fr,
        'Step2': {'Domain2': {'average': round(d2_at_d2, 2) if d2_at_d2 is not None else None,
                              'classwise': s2_cw}},
        'Step3': {'Domain2': {'average': round(d2_at_d3, 2), 'classwise': s3d2_cw},
                  'Domain3': {'average': round(d3_at_d3, 2), 'classwise': s3d3_cw}},
        'topn_ensemble': topn,
    }


def sort_d3_by_val_macro(d3_ckpts, log=print):
    """val_macro 내림차순 정렬 (없으면 Acc fallback)."""
    def score(ck):
        s = torch.load(ck, map_location='cpu', weights_only=False)
        return s.get('val_macro') or s.get('Acc', 0.0) or 0.0
    return sorted(d3_ckpts, key=score, reverse=True)


def save_metric_json(metrics, path, log=print):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    log(f'saved: {path}')
