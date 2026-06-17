"""
common/infer_utils.py — 평가셋 추론 + 앙상블 소프트보팅 공용 로직.

추론 프로토콜 (개발셋 평가 파이프라인과 동일):
  - variable_len: 전체 클립을 1회 forward (4s 미만만 zero-pad, truncate 없음).
  - No TTA (gain=1.0 단일 forward)
  - active_domain_count=2 (D1 backbone + LoRA_D2 + LoRA_D3 합산)
  - 모든 체크포인트에 대한 soft voting
"""
import csv
import os

import librosa
import numpy as np
import torch
import torch.nn.functional as F

import config as c
from backbone import CNN14Backbone
from model import DeltaOrthoMultiLoRACNN14

AUDIO_EXTS = ('.wav', '.flac', '.mp3', '.ogg', '.m4a')


def load_infer_model(ckpt_path, device, active_domain_count=2):
    """체크포인트(backbone 가중치 포함)를 그대로 로드 — D1 ckpt 불필요."""
    backbone = CNN14Backbone(nb_tasks=3)
    model = DeltaOrthoMultiLoRACNN14(backbone).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    model.active_domain_count = active_domain_count
    model.eval()
    return model


def list_audio(eval_dir):
    paths = sorted(
        os.path.join(r, f)
        for r, _, fs in os.walk(eval_dir) for f in fs
        if f.lower().endswith(AUDIO_EXTS)
    )
    if not paths:
        raise FileNotFoundError(f'no audio under {eval_dir}')
    return paths


def list_checkpoints(dict_dir):
    cks = sorted(
        os.path.join(dict_dir, f)
        for f in os.listdir(dict_dir) if f.endswith('.pth')
    )
    if not cks:
        raise FileNotFoundError(f'no .pth under {dict_dir}')
    return cks


def select_topn_ckpts(ckpts, n, log=print):
    """val_macro 내림차순 상위 n개 체크포인트 반환 (없으면 Acc → 0 fallback).
    n 이 전체보다 크거나 None 이면 전체 사용."""
    def score(ck):
        s = torch.load(ck, map_location='cpu', weights_only=False)
        return s.get('val_macro') or s.get('Acc', 0.0) or 0.0
    ranked = sorted(ckpts, key=score, reverse=True)
    if n is None or n >= len(ranked):
        log(f'  Top-N: 전체 {len(ranked)}개 사용 (n={n})')
        return ranked
    sel = ranked[:n]
    log(f'  Top-{n} 선택 (val_macro 기준): {[os.path.basename(p) for p in sel]}')
    return sel


def preload_audio(paths, log=print):
    """각 클립을 full length 로 1회 로드 (4s 미만만 zero-pad)."""
    out = []
    for i, p in enumerate(paths):
        y, _ = librosa.load(p, sr=c.SAMPLE_RATE, mono=True)
        y = y.astype(np.float32)
        if len(y) < c.CLIP_SAMPLES:
            y = np.concatenate([y, np.zeros(c.CLIP_SAMPLES - len(y), dtype=np.float32)])
        out.append(y)
        if (i + 1) % 500 == 0:
            log(f'  loaded {i + 1}/{len(paths)}')
    return out


@torch.no_grad()
def predict_probs(model, waves, device):
    probs = []
    for y in waves:
        audio = torch.from_numpy(y).unsqueeze(0).float().to(device)
        logits, _ = model(audio)
        probs.append(F.softmax(logits, -1).cpu().numpy()[0])
    return np.stack(probs, axis=0)


def ensemble_probs(ckpts, waves, device, log=print, active_domain_count=2):
    """체크포인트 전체에 대한 평균 확률(soft voting)."""
    sum_probs = np.zeros((len(waves), c.NUM_CLASSES), dtype=np.float64)
    for i, ck in enumerate(ckpts, 1):
        log(f'  [{i:02d}/{len(ckpts)}] {os.path.basename(ck)}')
        model = load_infer_model(ck, device, active_domain_count)
        sum_probs += predict_probs(model, waves, device)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    return sum_probs / len(ckpts)


@torch.no_grad()
def eval_macro_variable(model, df, device):
    """단일모델 평가용: variable(전체 클립) + TTA-off macro accuracy.
    active_domain_count 는 호출 측에서 미리 설정한다 (D2@D2=1, D2@D3/D3@D3=2)."""
    model.eval()
    waves = preload_audio(list(df['full_path']))
    gt = np.array(df['new_target'].tolist())
    preds = predict_probs(model, waves, device).argmax(1)
    accs = [(preds[gt == k] == k).mean() for k in range(c.NUM_CLASSES) if (gt == k).sum() > 0]
    return float(np.mean(accs) * 100)


def write_output_csv(paths, pred_idx, out_path):
    """공식 DCASE 양식: TSV, 헤더 없음, 한 줄당 `<basename>.wav<TAB><full_class_name>`.
    클래스명은 CLASS_LABELS 키(full name: alarm, baby_cry, ..., telephone_ringing)."""
    inv_labels = {v: k for k, v in c.CLASS_LABELS.items()}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        for p, idx in zip(paths, pred_idx):
            w.writerow([os.path.basename(p), inv_labels[int(idx)]])
    return out_path
