"""
common/eval_utils.py — 개발셋(dev) classwise 평가 → metric.json 생성.

Step2 (after D2): D2 체크포인트 앙상블(active=1)을 D2 dev-test 에서 평가.
Step3 (after D3): D3 체크포인트 앙상블(active=2)을 D2/D3 dev-test 에서 평가.
프로토콜: variable_len full clip, single forward(no TTA), batch_size=1, macro accuracy.

Forgetting rate(ensemble-to-ensemble):
    Fr = Step2 Domain2 average - Step3 Domain2 average
"""
import json
import os

import numpy as np
import torch

import config as c
from infer_utils import load_infer_model, preload_audio, predict_probs

# 클래스 short 이름 (meta.yaml / metric.json 표기용)
SHORT = {
    'alarm': 'alarm', 'baby_cry': 'baby', 'dog_bark': 'dog', 'engine': 'engine',
    'fire': 'fire', 'footsteps': 'footsteps', 'knocking': 'knock',
    'telephone_ringing': 'phone', 'piano': 'piano', 'speech': 'speech',
}
IDX_TO_SHORT = {v: SHORT[k] for k, v in c.CLASS_LABELS.items()}


def load_waves(df, log=print):
    return preload_audio(list(df['full_path']), log=log)


def classwise(preds, gt):
    res = {}
    for cls in sorted(np.unique(gt)):
        mask = gt == cls
        res[IDX_TO_SHORT[int(cls)]] = round(float((preds[mask] == cls).mean()) * 100, 2)
    res_avg = round(float(np.mean(list(res.values()))), 2)
    return res_avg, res


def _ensemble_argmax(ckpts, waves, device, active, log):
    probs = np.zeros((len(waves), c.NUM_CLASSES))
    for i, ck in enumerate(ckpts, 1):
        log(f'  [{i:02d}/{len(ckpts)}] {os.path.basename(ck)}')
        m = load_infer_model(ck, device, active)
        probs += predict_probs(m, waves, device)
        del m
        if device == 'cuda':
            torch.cuda.empty_cache()
    return (probs / len(ckpts)).argmax(1)


def compute_dev_metrics(d2_ckpts, d3_ckpts, device, log=print):
    """D2/D3 체크포인트 리스트로 Step2/Step3 dev 메트릭 dict 생성."""
    df_d2 = c.DF_TEST[c.DF_TEST['domain'] == 'D2']
    df_d3 = c.DF_TEST[c.DF_TEST['domain'] == 'D3']
    gt_d2 = np.array(df_d2['new_target'].tolist())
    gt_d3 = np.array(df_d3['new_target'].tolist())
    w_d2, w_d3 = load_waves(df_d2, log), load_waves(df_d3, log)
    log(f'dev test: D2={len(gt_d2)}  D3={len(gt_d3)}')

    log('[Step2] D2 ensemble (active=1) on D2 test')
    s2_avg, s2_cls = classwise(_ensemble_argmax(d2_ckpts, w_d2, device, 1, log), gt_d2)

    log('[Step3] D3 ensemble (active=2) on D2+D3 test')
    # D3 체크포인트는 한 번만 로드해 D2/D3 test 확률을 동시에 누적
    p_d2 = np.zeros((len(gt_d2), c.NUM_CLASSES))
    p_d3 = np.zeros((len(gt_d3), c.NUM_CLASSES))
    for i, ck in enumerate(d3_ckpts, 1):
        log(f'  [{i:02d}/{len(d3_ckpts)}] {os.path.basename(ck)}')
        m = load_infer_model(ck, device, 2)
        p_d2 += predict_probs(m, w_d2, device)
        p_d3 += predict_probs(m, w_d3, device)
        del m
        if device == 'cuda':
            torch.cuda.empty_cache()
    s3d2_avg, s3d2_cls = classwise((p_d2 / len(d3_ckpts)).argmax(1), gt_d2)
    s3d3_avg, s3d3_cls = classwise((p_d3 / len(d3_ckpts)).argmax(1), gt_d3)

    avg_acc = round((s3d2_avg + s3d3_avg) / 2, 2)
    fr = round(s2_avg - s3d2_avg, 2)
    return {
        'protocol': 'variable_len, no TTA',
        'Step2': {'Domain2': {'average': s2_avg, 'classwise': s2_cls}},
        'Step3': {'Domain2': {'average': s3d2_avg, 'classwise': s3d2_cls},
                  'Domain3': {'average': s3d3_avg, 'classwise': s3d3_cls}},
        'Acc': avg_acc,
        'Fr': fr,
    }


def save_metric_json(metrics, path, log=print):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    log(f'saved: {path}')
