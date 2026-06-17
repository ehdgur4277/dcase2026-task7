"""
common/kfold_train.py — covforget(CoLoRA) D2/D3 단일 phase 학습 로직.

train_fold(원본) 의 D2/D3 phase 를 함수로 분리해 main.py(K-fold worker) 와
single_main.py 가 공유한다. covforget 핵심(DeltaGP 직교투영 + Σ-cov 보호 +
망각 R-penalty + KeepLoRA init)은 그대로 유지한다.

D2 phase 는 adaptor 상태(protect_v, cov_U, cov_lam, cov_bnvar)를 D2 ckpt 에
저장하고, D3 phase 는 그것을 복원해 Σ-cov 보호/망각 penalty 를 적용한다.
"""
import copy
import os
import random
import shutil
import tempfile

import numpy as np
import torch

import config as c
from backbone import CNN14Backbone
from dataset import evaluate
from model_adaptor import LoRACNN14, drift_to_block_configs
from train_adaptor import DeltaGPAdaptor
from train import train_phase

KFOLD_K    = int(c.CFG.get('kfold', {}).get('k', 5)) if isinstance(c.CFG.get('kfold'), dict) else 5
KFOLD_SEED = int(getattr(c, 'SEED', 42))


# ── 유틸 ────────────────────────────────────────────────────────────────────────
def atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.tmp')
    try:
        os.close(fd)
        torch.save(obj, tmp)
        shutil.move(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def make_kfold_split(df, fold_k, k=KFOLD_K, seed=KFOLD_SEED):
    """Stratified K-Fold(클래스별 round-robin): fold_k → val, 나머지 → train."""
    df = df.reset_index(drop=True)
    labels = df['new_target'].values
    fold_assignments = np.zeros(len(df), dtype=int)
    rng = np.random.RandomState(seed)
    for cls in range(c.NUM_CLASSES):
        idx = np.where(labels == cls)[0]
        idx = rng.permutation(idx)
        for i, pos in enumerate(idx):
            fold_assignments[pos] = i % k
    val_mask = fold_assignments == fold_k
    return df[~val_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)


def set_seed(seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_float32_matmul_precision('high')


def macro_eval(model, df, device):
    _, detail, _ = evaluate(model, df, device, return_detail=True)
    return detail['macro']


# ── config 로부터 도메인/covforget 설정 읽기 ─────────────────────────────────────
def load_specs():
    _dom_cfg = c.CFG.get('domains', [])
    _dom_def = c.CFG.get('domain_defaults', {})
    _cf_cfg  = c.CFG.get('cov_forget', {})
    _kl_cfg  = c.CFG.get('keeplora_binit', {})
    _gate_cfg = c.CFG.get('gate', {})
    _pace_cfg = c.CFG.get('pace', {})

    def _dget(d, key, default=None):
        return d.get(key, _dom_def.get(key, default))

    domain_block_cfgs = []
    rank_meta = {}
    for d in _dom_cfg:
        name = d['name']
        cfgs, r_A, r_B = drift_to_block_configs(d['drift_conv_bn'], d['drift_bn'], _dget(d, 'ceiling'))
        domain_block_cfgs.append(cfgs)
        rank_meta[f'{name.lower()}_rank_A'] = r_A
        rank_meta[f'{name.lower()}_rank_B'] = r_B

    return {
        'dom_cfg': _dom_cfg, 'dget': _dget, 'gate_cfg': _gate_cfg,
        'domain_block_cfgs': domain_block_cfgs, 'rank_meta': rank_meta,
        'cf': {
            'use': bool(_cf_cfg.get('use', False)),
            'every': int(_cf_cfg.get('cov_every', 8)),
            'energy': float(_cf_cfg.get('energy', 0.95)),
            'kmax': int(_cf_cfg.get('k_max', 256)),
            'ep': float(_cf_cfg.get('energy_protect', 0.9)),
            'beta': float(_cf_cfg.get('beta', 1e-3)),
        },
        'kl': {
            'use': bool(_kl_cfg.get('use', False)),
            'eta': float(_kl_cfg.get('eta', 0.1)),
            'nb': int(_kl_cfg.get('n_batches', 4)),
            'align': bool(_kl_cfg.get('align_A', False)),
        },
        'p_svd_d1': float(_pace_cfg.get('p_svd_d1', 0.99)),
    }


def _build_model(domain_block_cfgs, gate_cfg, device):
    backbone = CNN14Backbone(nb_tasks=3)
    backbone.load_d1_checkpoint(c.D1_CKPT)
    backbone.freeze_backbone()
    backbone.to(device)
    model = LoRACNN14(
        backbone, domain_block_cfgs=domain_block_cfgs,
        alpha=c.LORA_ALPHA, tune_bn=bool(c.CFG['lora'].get('tune_bn', False)),
        gate_cfg=gate_cfg,
    ).to(device)
    return model


# ── D2 phase ────────────────────────────────────────────────────────────────────
def run_d2(d2_fold_k, df_d2_tr, df_d2_val, df_d2_te, ckpt_path, device, log):
    """D2 LoRA 학습(CovForget + DeltaGP) → ckpt 저장(adaptor 상태 포함) → D2@D2 반환."""
    sp = load_specs()
    d2_spec = next(d for d in sp['dom_cfg'] if d['name'] == 'D2')
    cf, kl, dget = sp['cf'], sp['kl'], sp['dget']

    model = _build_model(sp['domain_block_cfgs'], sp['gate_cfg'], device)
    adaptor = DeltaGPAdaptor(model)
    log(f'LoRA params: {sum(p.numel() for p in model.domain_loras.parameters()):,}')

    d1_teacher = _freeze(copy.deepcopy(model))
    d1_teacher.active_domain_count = 0
    log('========== Building D1 base protect basis ==========')
    adaptor.build_base_protect(p_svd=sp['p_svd_d1'], eps=c.SVD_EPS)

    model.active_domain_count = 1
    adaptor.init_A_in_nullspace(domain_idx=0)
    model.set_domain_trainable(0)
    adaptor.install_orth_hooks(domain_idx=0)
    if kl['use']:
        adaptor.keeplora_init_AB(0, df_d2_tr, device, eta=kl['eta'],
                                 n_batches=kl['nb'], align_A=kl['align'], log_fn=log)

    try:
        train_phase(
            model, df_d2_tr, device, log,
            phase_name=f'D2_fold{d2_fold_k} (CovForget + DeltaGP)',
            lr=float(dget(d2_spec, 'lr')),
            teacher_model=d1_teacher, mode_cfg=dict(dget(d2_spec, 'kd')),
            epochs=int(dget(d2_spec, 'epochs')),
            val_df=df_d2_val,
            patience=int(dget(d2_spec, 'patience')),
            min_epochs=int(dget(d2_spec, 'min_epochs')),
            use_warm_restarts=bool(dget(d2_spec, 'warm', False)),
            cov_collect=cf['use'], cov_every=cf['every'],
        )
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            log(f'[WARNING] OOM: {e}')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise
    adaptor.clear_orth_hooks()

    model.active_domain_count = 1
    d2_at_d2 = macro_eval(model, df_d2_te, device)
    log(f'  D2@D2 (oracle, macro): {d2_at_d2:.2f}%')

    if cf['use']:
        log('========== Finalize Σ + protect: D2 ==========')
        adaptor.finalize_cov_basis(domain_idx=0, energy=cf['energy'], k_max=cf['kmax'], log_fn=log)
        adaptor.accumulate_cov_protect(energy_protect=cf['ep'], log_fn=log)

    atomic_save({
        'model': model.state_dict(), 'phase': 'after_D2',
        'D2_at_D2': d2_at_d2, 'd2_fold_k': d2_fold_k,
        'protect_v': adaptor._protect_v, 'cov_U': adaptor._cov_U,
        'cov_lam': adaptor._cov_lam, 'cov_bnvar': adaptor._cov_bnvar,
        **sp['rank_meta'],
    }, ckpt_path)
    log(f'Saved: {ckpt_path}')
    return d2_at_d2


# ── D3 phase ────────────────────────────────────────────────────────────────────
def run_d3(d2_fold_k, d3_fold_k, d2_ckpt_path, df_d3_tr, df_d3_val,
           df_d2_te, df_d3_te, ckpt_path, device, log, eval_fn=None):
    """D2 ckpt(+adaptor 상태) 복원 → D3 LoRA 학습(R-penalty) → ckpt 저장 → metrics 반환.
    eval_fn(model, df, device): 최종 평가 프로토콜 주입 (기본 macro_eval=variable+config TTA)."""
    eval_fn = eval_fn or macro_eval
    sp = load_specs()
    d3_spec = next(d for d in sp['dom_cfg'] if d['name'] == 'D3')
    cf, kl, dget = sp['cf'], sp['kl'], sp['dget']

    d2_ckpt = torch.load(d2_ckpt_path, map_location=device, weights_only=False)
    d2_at_d2 = d2_ckpt['D2_at_D2']

    model = _build_model(sp['domain_block_cfgs'], sp['gate_cfg'], device)
    adaptor = DeltaGPAdaptor(model)
    model.load_state_dict(d2_ckpt['model'])
    model.active_domain_count = 1
    adaptor._protect_v = d2_ckpt.get('protect_v', [])
    adaptor._cov_U     = d2_ckpt.get('cov_U', [])
    adaptor._cov_lam   = d2_ckpt.get('cov_lam', [])
    adaptor._cov_bnvar = d2_ckpt.get('cov_bnvar', [])
    log(f'Loaded D2 ckpt: D2@D2={d2_at_d2:.2f}%  '
        f'cov_U={len(adaptor._cov_U)}  protect_v={len(adaptor._protect_v)}')

    d2_teacher = _freeze(copy.deepcopy(model))
    d2_teacher.active_domain_count = 1

    model.active_domain_count = 2
    adaptor.init_A_in_nullspace(domain_idx=1)
    model.set_domain_trainable(1)
    adaptor.install_orth_hooks(domain_idx=1)
    if kl['use']:
        adaptor.keeplora_init_AB(1, df_d3_tr, device, eta=kl['eta'],
                                 n_batches=kl['nb'], align_A=kl['align'], log_fn=log)

    reg_fn   = adaptor.forgetting_penalty if (cf['use'] and adaptor._cov_U) else None
    reg_beta = cf['beta'] if reg_fn is not None else 0.0
    log(f'reg_fn={"ON" if reg_fn else "OFF"}  reg_beta={reg_beta}')

    stopper = None
    try:
        stopper = train_phase(
            model, df_d3_tr, device, log,
            phase_name=f'D3_d2f{d2_fold_k}_d3f{d3_fold_k} (CovForget + R-penalty + DeltaGP)',
            lr=float(dget(d3_spec, 'lr')),
            teacher_model=d2_teacher, mode_cfg=dict(dget(d3_spec, 'kd')),
            epochs=int(dget(d3_spec, 'epochs')),
            val_df=df_d3_val,
            patience=int(dget(d3_spec, 'patience')),
            min_epochs=int(dget(d3_spec, 'min_epochs')),
            use_warm_restarts=bool(dget(d3_spec, 'warm', False)),
            reg_fn=reg_fn, reg_beta=reg_beta,
            cov_collect=cf['use'], cov_every=cf['every'],
        )
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            log(f'[WARNING] OOM: {e}')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise
    adaptor.clear_orth_hooks()

    val_macro = stopper.best_macro if stopper is not None else 0.0
    log(f'  val_macro (D3 val fold): {val_macro:.2f}%')

    model.active_domain_count = 2
    d2_at_d3 = eval_fn(model, df_d2_te, device)
    d3_at_d3 = eval_fn(model, df_d3_te, device)
    acc = (d2_at_d3 + d3_at_d3) / 2
    fr  = d2_at_d2 - d2_at_d3
    log(f'  D2@D3={d2_at_d3:.2f}%  D3@D3={d3_at_d3:.2f}%  Acc={acc:.2f}%  Fr={fr:.2f}%p')

    metrics = {
        'D2@D2': round(d2_at_d2, 4), 'D2@D3': round(d2_at_d3, 4),
        'D3@D3': round(d3_at_d3, 4), 'Acc': round(acc, 4), 'Fr': round(fr, 4),
        'val_macro': round(val_macro, 4),
        'd2_fold_k': d2_fold_k, 'd3_fold_k': d3_fold_k,
    }
    atomic_save({'model': model.state_dict(), 'phase': 'after_D3',
                 **metrics, **sp['rank_meta']}, ckpt_path)
    log(f'Saved: {ckpt_path}')
    return metrics
