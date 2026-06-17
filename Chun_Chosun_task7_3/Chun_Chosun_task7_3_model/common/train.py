"""
train.py — exp13_CovForget 훈련 로직 + 오케스트레이션

계보: task7_baseline_official(DSBN+routing) → 261_D3DeltaGP-CosKD(DeltaGP+CosKD)
      → exp04(CorrLoRA 가산) → exp05/07(SplitLoRA+KeepLoRA) → exp10(게이트+누적_protect_v)
      → exp13(Σ-cov 보호 + 망각 R penalty).
모듈:
  - model_adaptor.py : LoRA 구조 (CorrAdditiveLoRABlock + 블록 게이트) + conv 입력활성 캡처
  - train_adaptor.py : 훈련기법 (DeltaGP null-space + KeepLoRA + Σ-cov 보호/R penalty)
  - dataset.py       : 데이터·증강 (AudioDataset, mixup, FocalLoss, split, evaluate)
  - ddp_util.py      : DDP 플러밍
  - train.py(이 파일): train_phase + EarlyStop + main 오케스트레이션

main() 흐름 — 임의 T 증분 (DOMAIN_SPEC 루프, D4,D5… 항목 추가만):
  backbone(D1 동결) → LoRACNN14(게이트) → DeltaGPAdaptor
  build_base_protect (D1 weight SVD)
  for t in DOMAIN_SPEC:
    teacher=snapshot(count=t) → init_A(⊥누적 _protect_v) → set_domain_trainable(t)
    → install_orth_hooks(t) → [KeepLoRA init]
    → train_phase (CE + KD anchor + 망각 R penalty, 학습중 Σ 활성 수집)
    → oracle eval/ckpt → finalize_cov_basis(t) + accumulate_cov_protect(t)
       # 도메인 t 활성 공분산 Σ_t 의 top-eig 를 _protect_v 에 누적 (다음 도메인 보호, true-IL)
  final eval (count=N, 도메인 id 없음)
※ Σ 는 자기 phase 자기 데이터(학습 forward)로만 수집→동결 통계. 데이터 도메인 교차 0.
"""

import argparse
import copy
import json
import os
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.join(_HERE, '..', 'common')
sys.path.insert(0, _COMMON)
sys.path.insert(0, _HERE)

os.environ.setdefault('CONFIG', os.path.join(_HERE, 'config.yaml'))

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import config as c
from backbone import CNN14Backbone
from dataset import (build_train_loader, mixup_batch, make_criterion,
                     stratified_val_split, evaluate)
from model_adaptor import LoRACNN14, drift_to_block_configs
from train_adaptor import DeltaGPAdaptor
import ddp_util as ddp

_lora_cfg = c.CFG['lora']
_pace_cfg = c.CFG['pace']


# ── Early stopping ───────────────────────────────────────────────────────────

class MultiCriterionStopper:
    """patience(얼리스탑)와 best(복원/저장)를 **분리** 모니터.
      patience_on: 무엇이 개선되면 patience 리셋 (loss | micro | macro | any[3중-OR]) → stop 신호.
      best_on    : 복원/저장 모델 선택 기준 (macro=공식 | loss | micro).
    기본 = patience_on='loss'(안정적 정지 신호), best_on='macro'(공식 메트릭 최고 보존).
    loss·micro·macro 전부 추적."""
    def __init__(self, patience, min_epochs=0, min_delta=0.0, patience_on='loss', best_on='macro'):
        self.patience = patience; self.min_epochs = min_epochs; self.min_delta = min_delta
        self.patience_on = patience_on; self.best_on = best_on
        self.best_loss = float('inf'); self.best_micro = -1.0; self.best_macro = -1.0
        self.best_epoch = 0; self.best_state = None; self.best_meta = None
        self.counter = 0

    def step(self, loss, micro, macro, model, epoch, meta=None):
        imp = {'loss':  loss  < self.best_loss  - self.min_delta,
               'micro': micro > self.best_micro + self.min_delta,
               'macro': macro > self.best_macro + self.min_delta}
        if imp['loss']:  self.best_loss  = loss
        if imp['micro']: self.best_micro = micro
        if imp['macro']: self.best_macro = macro
        improved_best = imp[self.best_on]                 # 복원/저장 = best_on 지표 최고
        if improved_best:
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict()); self.best_meta = meta
        reset = (imp['loss'] or imp['micro'] or imp['macro']) if self.patience_on == 'any' \
                else imp[self.patience_on]                # patience 리셋 = patience_on 개선 시
        self.counter = 0 if reset else self.counter + 1
        stop = (epoch >= self.min_epochs) and (self.counter >= self.patience)
        return improved_best, stop

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

    @property
    def best_str(self):
        v = {'loss': f'loss={self.best_loss:.4f}', 'micro': f'micro={self.best_micro:.2f}',
             'macro': f'macro={self.best_macro:.2f}'}[self.best_on]
        return f'{v}@{self.best_epoch}'


# ── 훈련 phase (순수 학습 루프) ──────────────────────────────────────────────

def train_phase(
    model, train_df, device, log_fn, phase_name,
    lr, teacher_model=None, mode_cfg=None,
    epochs=None, val_df=None, patience=None, min_epochs=0,
    on_best=None,
    rank=0, world_size=1, distributed=False,
    use_warm_restarts=False,
    reg_fn=None, reg_beta=0.0,   # exp13: 망각 penalty R=β·tr(D⁻¹ ΔW Σ_prior ΔWᵀ)
    cov_collect=False, cov_every=8,   # exp13: 학습 forward 도중 활성 통계 수집(별도 패스 없음)
):
    assert epochs is not None
    is_main = (rank == 0)
    log_fn(f'\n========== Phase: {phase_name} ==========')

    # 데이터 + 로더 구성은 dataset.py 가 담당 (증강/샘플러/워커시드 포함)
    ds, loader = build_train_loader(train_df, rank, world_size, distributed)
    if is_main:
        log_fn(f'  Train chunks: {len(ds)}')
        if val_df is not None:
            log_fn(f'  Val clips: {len(val_df)} | patience={patience} min_epochs={min_epochs}')

    raw_model = ddp.raw(model)
    trainable = [p for p in raw_model.parameters() if p.requires_grad]
    if is_main:
        log_fn(f'  Trainable params: {sum(p.numel() for p in trainable):,}  LR: {lr}')
        log_fn(f'  Scheduler: {"WarmRestarts" if use_warm_restarts else "CosineAnnealing"}')

    optim = torch.optim.AdamW(trainable, lr=lr, weight_decay=c.WEIGHT_DECAY)
    if use_warm_restarts:
        T0 = max(1, epochs // 4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, T_0=T0, T_mult=2, eta_min=c.LR_ETA_MIN)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=epochs, eta_min=c.LR_ETA_MIN)

    criterion = make_criterion()
    _es = c.CFG.get('early_stop', {}) or {}
    _pat_on = str(_es.get('patience_on', 'loss')).lower()
    _best_on = str(_es.get('best_on', 'macro')).lower()
    stopper = (MultiCriterionStopper(patience, min_epochs, c.MIN_DELTA,
                                     patience_on=_pat_on, best_on=_best_on)
               if (val_df is not None and patience is not None and is_main)
               else None)
    if is_main and stopper is not None:
        log_fn(f'  EarlyStop: patience_on={_pat_on}(정지신호) | best_on={_best_on}(복원/저장)')

    if cov_collect and is_main:
        raw_model._captured_acts = {}     # exp13: 학습 중 활성 reservoir 초기화 (별도 패스 없음)

    for epoch in range(1, epochs + 1):
        if distributed and hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)

        model.train()
        if not raw_model.tune_bn:
            raw_model.backbone.set_bn_eval()
        raw_model.set_bn_train_for_tuned()  # tune_bn=True: backbone BN을 train 모드 유지

        sum_cls = sum_kd = sum_feat = sum_reg = n = 0

        for batch_i, (audio, target, _, _) in enumerate(loader):
            audio      = audio.float().to(device)
            target_idx = target.float().to(device).argmax(-1)

            audio_m, ya, yb, lam = mixup_batch(audio, target_idx, device)

            if cov_collect:   # 학습 forward 도중 활성 수집 (rank0 only, mixup 없는 배치만 → 깨끗한 Σ)
                raw_model._capture_acts = (is_main and batch_i % cov_every == 0 and lam == 1.0)

            logits, feat = model(audio_m, use_spec_aug=c.USE_SPEC_AUG)
            loss_cls = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            loss = loss_cls

            if teacher_model is not None:
                kd_alpha = mode_cfg['kd_alpha']; kd_temp = mode_cfg['kd_temp']
                kd_fw    = mode_cfg['kd_feat_weight']
                teacher_model.eval()
                with torch.no_grad():
                    t_logits, t_feat = teacher_model(audio_m)
                kd_loss = F.kl_div(
                    F.log_softmax(logits / kd_temp, dim=-1),
                    F.softmax(t_logits / kd_temp, dim=-1),
                    reduction='batchmean') * (kd_temp ** 2)
                feat_loss = 1.0 - F.cosine_similarity(feat, t_feat, dim=1, eps=1e-8).mean()
                loss = loss + kd_alpha * kd_loss + kd_fw * feat_loss
                sum_kd += float(kd_loss.detach()); sum_feat += float(feat_loss.detach())

            if reg_fn is not None and reg_beta > 0.0:    # exp13: 망각 penalty (캐시 Σ_prior 통계만 사용)
                R = reg_fn()
                if R is not None:
                    loss = loss + reg_beta * R
                    sum_reg += float(R.detach())

            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, c.GRAD_CLIP)
            optim.step(); sum_cls += float(loss_cls.detach()); n += 1

        if cov_collect:
            raw_model._capture_acts = False   # val/eval 중 캡처 방지

        avg_cls = ddp.all_reduce_mean(sum_cls / max(n, 1), device, distributed)
        sched.step()

        stop = False
        if is_main and stopper is not None:
            val_micro, val_detail, val_loss = evaluate(
                raw_model, val_df, device, return_detail=True, criterion=criterion)
            val_macro = val_detail['macro']
            improved_best, stop = stopper.step(val_loss, val_micro, val_macro, raw_model, epoch,
                                               meta={'epoch': epoch, 'val_loss': val_loss,
                                                     'val_micro': val_micro, 'val_macro': val_macro})
            if improved_best and on_best is not None:      # best(monitor 지표) 갱신 시 저장
                on_best(raw_model, val_micro, val_loss, epoch, val_detail)

        stop = ddp.broadcast_stop(stop, device, distributed)

        if is_main and (epoch % c.LOG_EVERY == 0 or epoch == 1 or epoch == epochs or stop):
            extra = ''
            if teacher_model is not None:
                extra = f' kd={sum_kd/max(n,1):.4f} cos={sum_feat/max(n,1):.4f}'
            if reg_fn is not None and reg_beta > 0.0:
                extra += f' R={sum_reg/max(n,1):.4f}'
            if stopper is not None:
                extra += (f' vloss={val_loss:.4f} vacc={val_micro:.2f}/{val_detail["macro"]:.2f}m '
                          f'best({stopper.best_str}) cnt={stopper.counter}/{patience}')
            log_fn(f'  Epoch {epoch:3d}/{epochs} | cls={avg_cls:.4f}{extra} '
                   f'lr={optim.param_groups[0]["lr"]:.6f}')

        if stop:
            if is_main:
                log_fn(f'  Early stop @ epoch {epoch} (best {stopper.best_str}, {patience}ep 정체)')
            break

    if is_main and stopper is not None:
        stopper.restore(raw_model)
        log_fn(f'  Restored best model ({stopper.best_str}, best_on={stopper.best_on}, '
               f'patience_on={stopper.patience_on})')
    ddp.sync_model(raw_model, distributed)
    return stopper


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _freeze(model):
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model


def _make_log(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f: f.write(msg + '\n')
    return log


def _save_confmat_md(confmat, path, title):
    if not confmat: return
    labels = list(confmat.keys())
    lines  = [f'# {title}', '',
              '| true \\ pred | ' + ' | '.join(labels) + ' | acc% |',
              '|' + '---|' * (len(labels) + 2)]
    for tl in labels:
        row = confmat.get(tl, {}); total = sum(row.values())
        acc   = f'{100*row.get(tl,0)/total:.1f}' if total > 0 else '—'
        cells = ' | '.join(str(row.get(pl, 0)) for pl in labels)
        lines.append(f'| {tl} | {cells} | {acc} |')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f: f.write('\n'.join(lines) + '\n')


def _write_metrics(path, d2_after_d2, d2_final, d3_final,
                   d2d2_ma=None, d2d3_ma=None, d3d3_ma=None):
    """micro + (있으면) macro 둘 다 기록. 공식 메트릭은 macro(class-wise mean)."""
    avg = (d2_final + d3_final) / 2.0; fr = d2_after_d2 - d2_final
    metrics = {
        'exp_id': c.EXP_ID or '07', 'method': c.EXP_METHOD or c.MODE,
        'date': c.EXP_DATE, 'status': 'done', 'notes': c.EXP_NOTES,
        'D2@D2': round(d2_after_d2, 2), 'D2@D3': round(d2_final, 2),
        'D3@D3': round(d3_final, 2), 'Acc': round(avg, 2), 'Fr': round(fr, 2),
        'O_D2': None, 'O_D3': None, 'O_Acc': None, 'Routing_Acc': None,
    }
    if d2d3_ma is not None:
        avg_ma = (d2d3_ma + d3d3_ma) / 2.0; fr_ma = d2d2_ma - d2d3_ma
        metrics.update({
            'D2@D2_macro': round(d2d2_ma, 2), 'D2@D3_macro': round(d2d3_ma, 2),
            'D3@D3_macro': round(d3d3_ma, 2), 'Acc_macro': round(avg_ma, 2),
            'Fr_macro': round(fr_ma, 2),
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f: json.dump(metrics, f, indent=2, ensure_ascii=False)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    cli, _ = parser.parse_known_args()
    if cli.seed is not None: c.set_seed(cli.seed)

    rank, local_rank, world_size, distributed, device = ddp.setup_distributed()
    is_main = rank == 0

    if torch.cuda.is_available():
        # 결정성 강제: 동일 config → 동일 결과 (knob 비교 신뢰성 확보)
        # benchmark=True는 conv 알고리즘을 런마다 autotuning해 ~0.8%p 노이즈를 유발했음
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_float32_matmul_precision('high')

    seed = c.SEED
    random.seed(seed + rank); np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed + rank)

    TIMESTR    = c.kst_timestamp()
    ckpt_dir   = os.path.join(_HERE, 'checkpoint')
    result_dir = os.path.join(_HERE, 'results')
    log_dir    = os.path.join(_HERE, 'logs')
    for d in (ckpt_dir, result_dir, log_dir): os.makedirs(d, exist_ok=True)

    log = _make_log(os.path.join(log_dir, f'run_{TIMESTR}.log')) if is_main else (lambda m: None)
    log(f'DDP={distributed}  world_size={world_size}  device={device}')
    log(f'Config: {os.environ["CONFIG"]}')
    log(f'SEED={seed}  VAL_SEED={c.VAL_SEED}  VAL_RATIO={c.VAL_RATIO}')
    tune_bn = bool(_lora_cfg.get('tune_bn', True))
    log(f'tune_bn={tune_bn}')
    log(f'batch/GPU={c.BATCH_SIZE}  eff_batch={c.BATCH_SIZE * world_size}')

    # ── 도메인 시퀀스 (config `domains` 구동, 임의 T) ────────────────────────────
    #   config 엔 도메인별 *백본기준 drift 리스트*만 주면 rank 자동 산출(drift_to_block_configs).
    #   보호·Σ·R·게이트 등 절차는 모든 도메인 동일. 도메인 추가 = config domains 항목 추가만.
    _dom_cfg = c.CFG.get('domains', []) or []
    _dom_def = c.CFG.get('domain_defaults', {}) or {}
    if not _dom_cfg:
        raise RuntimeError('config 에 domains 리스트(도메인별 drift_conv_bn/drift_bn)가 필요합니다.')

    def _dget(d, key, default=None):
        return d.get(key, _dom_def.get(key, default))

    DOMAIN_SPEC = []; domain_block_cfgs = []; rank_meta = {}
    for d in _dom_cfg:
        name = d['name']
        cfgs, r_A, r_B = drift_to_block_configs(d['drift_conv_bn'], d['drift_bn'], _dget(d, 'ceiling'))
        domain_block_cfgs.append(cfgs)
        rank_meta[f'{name.lower()}_rank_A'] = r_A; rank_meta[f'{name.lower()}_rank_B'] = r_B
        df_all = c.DF_TRAIN[c.DF_TRAIN['domain'] == name]
        df_te  = c.DF_TEST[c.DF_TEST['domain'] == name]
        df_tr, df_val = stratified_val_split(df_all)
        spec = dict(name=name, df_tr=df_tr, df_val=df_val, df_te=df_te,
                    lr=float(_dget(d, 'lr')), epochs=int(_dget(d, 'epochs')),
                    patience=int(_dget(d, 'patience')), min_epochs=int(_dget(d, 'min_epochs')),
                    warm=bool(_dget(d, 'warm', False)), kd=dict(_dget(d, 'kd')),
                    ckpt=f'{name.lower()}.pth', best=f'{name.lower()}_best.pth',
                    cm=f'confmat_{name.lower()}_val.md', cmtitle=f'{name} Val CM')
        DOMAIN_SPEC.append(spec)
        log(f'  [{name}] rank_A={r_A} rank_B={r_B} (auto from drift, ceil={_dget(d,"ceiling")}) | '
            f'lr={spec["lr"]} epochs={spec["epochs"]} warm={spec["warm"]} '
            f'train={len(df_tr)} val={len(df_val)}')

    def make_best_saver(phase, ckpt_name, confmat_name, title):
        """phase별 best-val 체크포인트 + confusion matrix 저장 콜백 생성."""
        def _save(m, val_micro, val_loss, epoch, detail):
            torch.save({'model': m.state_dict(), 'phase': phase, 'epoch': epoch,
                        'val_loss': val_loss, 'val_micro': val_micro, **rank_meta},
                       os.path.join(ckpt_dir, ckpt_name))
            _save_confmat_md(detail.get('confusion_matrix', {}),
                             os.path.join(result_dir, confmat_name),
                             f'{title} — epoch={epoch} vacc={val_micro:.2f}%')
        return _save

    # ── D1 보호 SVD 컷오프 (D1 weight, p_svd 누적에너지) ──────────────────────
    p_svd_d1 = _pace_cfg.get('p_svd_d1', c.P_SVD_D2)
    log(f'p_svd_d1(D1 weight SVD)={p_svd_d1}  (D2+ 보호는 Σ-cov, 아래 cov_forget)')

    # ── KeepLoRA gradient B-init (현재 도메인 데이터만) ────────────────────
    _kl_cfg  = c.CFG.get('keeplora_binit', {}) or {}
    kl_use   = bool(_kl_cfg.get('use', False))
    kl_eta   = float(_kl_cfg.get('eta', 0.1))
    kl_nb    = int(_kl_cfg.get('n_batches', 4))
    kl_align = bool(_kl_cfg.get('align_A', False))
    log(f'KeepLoRA init use={kl_use} align_A={kl_align} eta={kl_eta} n_batches={kl_nb}')

    # ── 블록 게이트 (증분 역행 방지 / 임의 T 비감소 보장) ──────────────────
    _gate_cfg = c.CFG.get('gate', {}) or {}
    gate_use  = bool(_gate_cfg.get('use', False))
    log(f'Gate use={gate_use} init_logit={_gate_cfg.get("init_logit", 4.0)} '
        f'(g=sigmoid(θ)·ΔW, off=exp07 수치동일)')

    # ── exp13: 활성 공분산 Σ_prior 기반 보호 + 망각 penalty R (true-IL) ──────────
    #   Σ 는 학습 forward 도중 매 cov_every 배치 수집(별도 패스 없음). 자기 phase 자기 데이터만.
    _cf_cfg   = c.CFG.get('cov_forget', {}) or {}
    cf_use    = bool(_cf_cfg.get('use', False))
    cf_every  = int(_cf_cfg.get('cov_every', 8))     # 학습 중 활성 수집 주기(배치)
    cf_energy = float(_cf_cfg.get('energy', 0.95))
    cf_kmax   = int(_cf_cfg.get('k_max', 256))
    cf_ep     = float(_cf_cfg.get('energy_protect', 0.9))
    cf_beta   = float(_cf_cfg.get('beta', 1.0e-3))
    log(f'CovForget use={cf_use} cov_every={cf_every} energy={cf_energy} k_max={cf_kmax} '
        f'energy_protect={cf_ep} beta={cf_beta} '
        f'(Σ 학습중 수집→동결통계, true-IL. 보호=top-eig(Σ), 망각=R penalty)')

    # ── Backbone (동결 D1 base) ────────────────────────────────────────────
    backbone = CNN14Backbone(nb_tasks=3)
    backbone.load_d1_checkpoint(c.D1_CKPT)
    backbone.freeze_backbone()
    backbone.to(device)

    # ── 모델 + 훈련기법 adaptor (domain_block_cfgs = 위 domains 리스트에서 자동) ──
    model = LoRACNN14(
        backbone, domain_block_cfgs=domain_block_cfgs,
        alpha=c.LORA_ALPHA, tune_bn=tune_bn, gate_cfg=_gate_cfg,
    ).to(device)
    adaptor = DeltaGPAdaptor(model)   # DeltaGP + KeepLoRA + CovForget(Σ) 절차 보유

    n_params = sum(p.numel() for p in model.domain_loras.parameters())
    log(f'LoRA params (T={len(DOMAIN_SPEC)} domains): {n_params:,}')
    ddp.sync_model(model, distributed)

    # ── 초기 보호기저: D1 backbone weight SVD (첫 도메인 학습용) ──────────────
    log('\n========== Building base protect basis (D1 weight SVD) ==========')
    if is_main:
        adaptor.build_base_protect(p_svd=p_svd_d1, eps=c.SVD_EPS)
    ddp.broadcast_object_attrs(adaptor, rank, distributed, device, ['_protect_v'])

    oracle = {}   # name → (micro, macro)  (각 도메인 학습 직후 self-test, count=t+1)
    for t, spec in enumerate(DOMAIN_SPEC):
        name = spec['name']
        log(f'\n========== {name} Phase (domain idx {t}, count={t+1}) ==========')
        log(f'  teacher=snapshot(count={t})  KD={spec["kd"]}  warm={spec["warm"]}  '
            f'grad ⊥ 누적 _protect_v  gate={gate_use}')

        # 직전까지 스냅샷 = teacher (count=t: D1..D(t+1) 중 새 도메인 미포함)
        teacher = _freeze(copy.deepcopy(model))
        teacher.active_domain_count = t
        teacher.to(device)

        model.active_domain_count = t + 1
        adaptor.init_A_in_nullspace(domain_idx=t)        # A_t ⊥ 누적 _protect_v
        model.set_domain_trainable(t)
        adaptor.install_orth_hooks(domain_idx=t)
        if kl_use:                                       # KeepLoRA init (현재 도메인 데이터만)
            if is_main:
                adaptor.keeplora_init_AB(domain_idx=t, df=spec['df_tr'], device=device,
                                         eta=kl_eta, n_batches=kl_nb, align_A=kl_align,
                                         log_fn=log)
            ddp.sync_model(model, distributed)

        # 망각 penalty: 이전 도메인 Σ 가 캐시돼 있을 때만(=D3+). 캐시 통계만 사용(true-IL).
        reg_fn   = (adaptor.forgetting_penalty if (cf_use and adaptor._cov_U) else None)
        reg_beta = (cf_beta if reg_fn is not None else 0.0)

        best_saver = make_best_saver(f'{name}_best', spec['best'], spec['cm'], spec['cmtitle'])
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        broadcast_buffers=True) if distributed else model
        train_phase(
            ddp_model, spec['df_tr'], device, log,
            phase_name=f'{name} (CoLoRA + KD + DeltaGP[⊥_protect_v]'
                       f'{" + R-penalty" if reg_fn else ""}'
                       f'{" + WarmRestarts" if spec["warm"] else ""})',
            lr=spec['lr'], teacher_model=teacher, mode_cfg=spec['kd'],
            epochs=spec['epochs'], val_df=spec['df_val'],
            patience=spec['patience'], min_epochs=spec['min_epochs'],
            on_best=best_saver if is_main else None,
            rank=rank, world_size=world_size, distributed=distributed,
            use_warm_restarts=spec['warm'],
            reg_fn=reg_fn, reg_beta=reg_beta,
            cov_collect=cf_use, cov_every=cf_every,   # 학습 중 활성 통계 수집(별도 패스 없음)
        )
        if distributed: del ddp_model
        adaptor.clear_orth_hooks()

        # 학습 직후 self-test (oracle, count=t+1) + 도메인 체크포인트 저장
        if is_main:
            model.active_domain_count = t + 1
            o_micro, o_det, _ = evaluate(model, spec['df_te'], device, return_detail=True)
            oracle[name] = (o_micro, o_det['macro'])
            log(f'\n  {name}@{name} (oracle, count={t+1}): '
                f'micro {o_micro:.2f}% | macro {o_det["macro"]:.2f}%')
            torch.save({'model': model.state_dict(), 'phase': f'after_{name}', **rank_meta},
                       os.path.join(ckpt_dir, spec['ckpt']))
        ddp.sync_model(model, distributed); ddp.barrier(distributed)

        # 다음 도메인 보호: 방금 학습한 도메인 t 의 활성 공분산 Σ_t 를 확정(학습중 누적 활성 SVD).
        #   ★ true-IL: 자기 phase·자기 데이터만. 이후 도메인은 동결 통계(U,Λ)만 사용(데이터 미접근).
        if cf_use:
            log(f'\n========== Finalize Σ + protect: {name} (학습중 누적 활성, true-IL) ==========')
            if is_main:
                adaptor.finalize_cov_basis(domain_idx=t, energy=cf_energy,
                                           k_max=cf_kmax, log_fn=log)
                adaptor.accumulate_cov_protect(energy_protect=cf_ep, log_fn=log)
            ddp.broadcast_object_attrs(adaptor, rank, distributed, device,
                                       ['_cov_U', '_cov_lam', '_cov_bnvar', '_protect_v'])

    # ── 최종 평가 (count=N, 모든 도메인 누적 적용 — 도메인 id 없음, true-IL) ──────
    #   임의 T 일반: 도메인별 final(count=N) self-eval + oracle(학습직후). 표준 D2/D3 면 벤치 metric 매핑.
    N = len(DOMAIN_SPEC)
    if is_main:
        model.active_domain_count = N
        names = [s['name'] for s in DOMAIN_SPEC]
        final = {}     # name → (micro, macro)  count=N
        for spec in DOMAIN_SPEC:
            fmi, fdet, _ = evaluate(model, spec['df_te'], device, return_detail=True)
            final[spec['name']] = (fmi, fdet['macro'])

        log(f'\n{"="*60}')
        log(f'FINAL: {c.MODE}  (gate={gate_use}, T={N})')
        log(f'{"="*60}')
        log(f'  DDP: {distributed} | world_size: {world_size}')
        log(f'  protect: D1=weight-SVD(p_svd {p_svd_d1}) + 이후=Σ-cov(energy_protect {cf_ep}) | R β={cf_beta}')
        log(f'  {"domain":7s}  oracle(mi/ma)     final@N(mi/ma)    forget(macro)')
        for nm in names:
            om, oM = oracle[nm]; fm, fM = final[nm]
            log(f'  {nm:7s}  {om:6.2f}/{oM:6.2f}    {fm:6.2f}/{fM:6.2f}    {oM-fM:+6.2f}')
        avg_ma = sum(final[nm][1] for nm in names) / N
        fr_ma  = (sum(oracle[nm][1] - final[nm][1] for nm in names[:-1]) / (N - 1)) if N > 1 else 0.0
        avg_mi = sum(final[nm][0] for nm in names) / N
        fr_mi  = (sum(oracle[nm][0] - final[nm][0] for nm in names[:-1]) / (N - 1)) if N > 1 else 0.0
        log(f'  Avg(final) micro {avg_mi:.2f} / macro {avg_ma:.2f}   |   '
            f'Fr(이전도메인 평균) micro {fr_mi:.2f} / macro {fr_ma:.2f}')

        torch.save({'model': model.state_dict(), 'phase': f'after_{names[-1]}',
                    'Acc': avg_mi, 'Fr': fr_mi, 'Acc_macro': avg_ma, 'Fr_macro': fr_ma,
                    'final': final, 'oracle': dict(oracle), **rank_meta},
                   os.path.join(ckpt_dir, f'{names[-1].lower()}.pth'))

        # 벤치마크(D2,D3) metrics.json 호환 매핑 (make_result.py 용)
        if 'D2' in final and 'D3' in final:
            d2_after_d2, d2d2_ma = oracle['D2']
            d2_final, d2d3_ma    = final['D2']
            d3_final, d3d3_ma    = final['D3']
            _write_metrics(os.path.join(result_dir, 'metrics.json'),
                           d2_after_d2, d2_final, d3_final, d2d2_ma, d2d3_ma, d3d3_ma)
            log(f'  Saved → results/metrics.json (벤치 D2/D3 매핑)')


if __name__ == '__main__':
    main()
