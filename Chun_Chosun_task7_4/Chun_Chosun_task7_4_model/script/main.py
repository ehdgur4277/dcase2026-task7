"""
main.py — end-to-end 통합 파이프라인 진입점 (covforget / CoLoRA K-Fold).

D1 로드 → D2 5-fold 학습 → D3 25-fold 학습 → 앙상블 dev 평가(+선택적 추론).
체크포인트는 model/checkpoint/ensemble/ 에, 로그는 model/result/ensemble_log.txt 에.
병렬 학습은 자기 자신을 worker 모드로 subprocess 호출(진입점 1개 유지).

  python main.py --worker d2 --d2_fold K
  python main.py --worker d3 --d2_fold K --d3_fold J --d2_ckpt <path>

오케스트레이터:
  TASK7_LORA_DATASET_DIR=<root> GPU_IDS=0,1 MAX_WORKERS=10 python main.py
  TASK7_LORA_DATASET_DIR=<root> python main.py --eval_dir <eval audio dir>
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common'))

import runtime  # noqa: F401  (CONFIG env — config import 전에)
import config as c
from runtime import (
    get_device, make_logger, RESULT_DIR, CKPT_ENSEMBLE, D3_DICTIONARY, OUTPUT_CSV,
    ENSEMBLE_TOP_N,
)
from kfold_train import make_kfold_split, set_seed, run_d2, run_d3, KFOLD_SEED

THIS = os.path.abspath(__file__)
ENSEMBLE_LOG = os.path.join(RESULT_DIR, 'ensemble_log.txt')
K = int(c.KFOLD_K) if hasattr(c, 'KFOLD_K') else 5


# ── Worker: 단일 fold 학습 ───────────────────────────────────────────────────────
def worker_d2(d2_fold, log, device):
    set_seed(KFOLD_SEED + d2_fold * 10, device)
    df_all = c.DF_TRAIN[c.DF_TRAIN['domain'] == 'D2'].reset_index(drop=True)
    df_tr, df_val = make_kfold_split(df_all, fold_k=d2_fold)
    df_te = c.DF_TEST[c.DF_TEST['domain'] == 'D2']
    ckpt = os.path.join(CKPT_ENSEMBLE, f'fold_d2f{d2_fold}', 'd2.pth')
    if os.path.exists(ckpt):
        log(f'D2 fold{d2_fold}: 이미 완료 (스킵)'); return
    run_d2(d2_fold, df_tr, df_val, df_te, ckpt, device, log)


def worker_d3(d2_fold, d3_fold, d2_ckpt, log, device):
    set_seed(KFOLD_SEED + d2_fold * 10 + d3_fold, device)
    df_all = c.DF_TRAIN[c.DF_TRAIN['domain'] == 'D3'].reset_index(drop=True)
    df_tr, df_val = make_kfold_split(df_all, fold_k=d3_fold)
    df_d2_te = c.DF_TEST[c.DF_TEST['domain'] == 'D2']
    df_d3_te = c.DF_TEST[c.DF_TEST['domain'] == 'D3']
    ckpt = os.path.join(CKPT_ENSEMBLE, f'fold_d2f{d2_fold}_d3f{d3_fold}', 'd3.pth')
    if os.path.exists(ckpt):
        log(f'D3 d2f{d2_fold}_d3f{d3_fold}: 이미 완료 (스킵)'); return
    run_d3(d2_fold, d3_fold, d2_ckpt, df_tr, df_val, df_d2_te, df_d3_te, ckpt, device, log)


# ── Orchestrator ────────────────────────────────────────────────────────────────
def _run_subprocess(argv, env, log_path):
    import subprocess
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as logf:
        proc = subprocess.run([sys.executable, THIS, *argv], env=env,
                              stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode == 0


def _base_env(gpu):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env.setdefault('DATALOADER_WORKERS', '0')
    return env


def orchestrate(log, eval_dir=None):
    gpu_ids = [g.strip() for g in os.environ.get('GPU_IDS', '0,1').split(',')]
    max_workers = int(os.environ.get('MAX_WORKERS', '10'))
    t0 = time.time()
    log(f'=== main orchestrator (covforget) === GPU_IDS={gpu_ids} MAX_WORKERS={max_workers} K={K}')

    # Phase 1: D2
    log('\n' + '=' * 50 + '\nPhase 1: D2 folds\n' + '=' * 50)
    tasks = []
    for k in range(K):
        if os.path.exists(os.path.join(CKPT_ENSEMBLE, f'fold_d2f{k}', 'd2.pth')):
            log(f'  D2 fold{k}: 이미 완료 (스킵)'); continue
        env = _base_env(gpu_ids[k % len(gpu_ids)])
        lp = os.path.join(RESULT_DIR, 'ensemble', f'fold_d2f{k}.log')
        tasks.append((['--worker', 'd2', '--d2_fold', str(k)], env, lp, f'd2f{k}'))
    if tasks:
        with ProcessPoolExecutor(max_workers=min(len(tasks), max_workers)) as ex:
            futs = {ex.submit(_run_subprocess, a, e, l): lab for a, e, l, lab in tasks}
            for fut in as_completed(futs):
                log(f'  {futs[fut]}: {"OK" if fut.result() else "FAILED"}')

    # Phase 2: D3
    log('\n' + '=' * 50 + '\nPhase 2: D3 folds\n' + '=' * 50)
    tasks = []
    for d2k in range(K):
        d2_ckpt = os.path.join(CKPT_ENSEMBLE, f'fold_d2f{d2k}', 'd2.pth')
        if not os.path.exists(d2_ckpt):
            log(f'  WARNING: {d2_ckpt} 없음 → d2f{d2k} D3 스킵'); continue
        for d3k in range(K):
            if os.path.exists(os.path.join(CKPT_ENSEMBLE, f'fold_d2f{d2k}_d3f{d3k}', 'd3.pth')):
                log(f'  D3 d2f{d2k}_d3f{d3k}: 이미 완료 (스킵)'); continue
            env = _base_env(gpu_ids[len(tasks) % len(gpu_ids)])
            lp = os.path.join(RESULT_DIR, 'ensemble', f'fold_d2f{d2k}_d3f{d3k}.log')
            tasks.append((['--worker', 'd3', '--d2_fold', str(d2k), '--d3_fold', str(d3k),
                           '--d2_ckpt', d2_ckpt], env, lp, f'd2f{d2k}_d3f{d3k}'))
    if tasks:
        with ProcessPoolExecutor(max_workers=min(len(tasks), max_workers)) as ex:
            futs = {ex.submit(_run_subprocess, a, e, l): lab for a, e, l, lab in tasks}
            for fut in as_completed(futs):
                log(f'  {futs[fut]}: {"OK" if fut.result() else "FAILED"}')

    # Phase 3: 앙상블 dev 평가 + 선택적 추론
    log('\n' + '=' * 50 + '\nPhase 3: ensemble eval\n' + '=' * 50)
    from eval_utils import compute_topn_metrics, save_metric_json
    from infer_utils import select_topn_ckpts
    d2_ckpts = sorted(os.path.join(CKPT_ENSEMBLE, f'fold_d2f{k}', 'd2.pth')
                      for k in range(K)
                      if os.path.exists(os.path.join(CKPT_ENSEMBLE, f'fold_d2f{k}', 'd2.pth')))
    d3_ckpts = [p for d2k in range(K) for d3k in range(K)
                if os.path.exists(p := os.path.join(CKPT_ENSEMBLE, f'fold_d2f{d2k}_d3f{d3k}', 'd3.pth'))]
    device = get_device()
    # val_macro 상위 ENSEMBLE_TOP_N(=20)개만 앙상블 — 전체 25개 대신 Top-N 통일
    d3_top = select_topn_ckpts(d3_ckpts, ENSEMBLE_TOP_N, log=log) if d3_ckpts else []
    if d2_ckpts and d3_top:
        metrics = compute_topn_metrics(d2_ckpts, d3_top, device, log=log)
        metrics['selected_top_n'] = min(ENSEMBLE_TOP_N, len(d3_ckpts))
        save_metric_json(metrics, os.path.join(RESULT_DIR, 'metric.json'), log=log)
        log(f'dev metrics (Top-{metrics["selected_top_n"]}): {json.dumps(metrics)}')
    else:
        log('  체크포인트 부족 → dev 평가 스킵')

    if eval_dir:
        from infer_utils import list_audio, preload_audio, ensemble_probs, write_output_csv
        log(f'\n앙상블 추론 (Top-{ENSEMBLE_TOP_N}): {eval_dir}')
        paths = list_audio(eval_dir)
        waves = preload_audio(paths, log=log)
        probs = ensemble_probs(d3_top, waves, device, log=log, active_domain_count=2)
        pred = probs.argmax(1)
        write_output_csv(paths, pred, os.path.join(RESULT_DIR, 'out.csv'))
        write_output_csv(paths, pred, OUTPUT_CSV)
        log(f'saved: out.csv & {OUTPUT_CSV}')

    h, m = divmod(int(time.time() - t0), 3600); m, s = divmod(m, 60)
    log(f'\n총 소요: {h}h {m}m {s}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', choices=['d2', 'd3'], default=None)
    ap.add_argument('--d2_fold', type=int, default=0)
    ap.add_argument('--d3_fold', type=int, default=0)
    ap.add_argument('--d2_ckpt', default='')
    ap.add_argument('--eval_dir', default=None)
    args = ap.parse_args()

    device = get_device()
    log = make_logger(ENSEMBLE_LOG)
    if args.worker == 'd2':
        worker_d2(args.d2_fold, log, device)
    elif args.worker == 'd3':
        worker_d3(args.d2_fold, args.d3_fold, args.d2_ckpt, log, device)
    else:
        orchestrate(log, eval_dir=args.eval_dir)


if __name__ == '__main__':
    main()
