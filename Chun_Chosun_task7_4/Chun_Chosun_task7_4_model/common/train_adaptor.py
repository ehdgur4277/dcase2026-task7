"""
train_adaptor.py — baseline 대비 추가된 **훈련기법** (model 구조와 분리)

원본 baseline 의 incremental_train 은 "동결 → 일부 conv/bn/fc 만 학습 → Adam+Cosine" 가 전부다.
여기 모인 것은 그 위에 망각 방지·가소성 확보를 위해 추가한 기법들이며, 모두 모델 구조가 아니라
모델(model_adaptor.LoRACNN14) 을 인자로 받아 그 LoRA 파라미터를 조작/제약하는 '절차'다.

기법 (모두 true-IL: 이전 도메인 데이터 재접근 0):
  ① DeltaGP (null-space gradient projection) — 누적 _protect_v 단일화(임의 T)
       - build_base_protect    : D1 weight SVD → 초기 보호기저 (D1 보호 전용)
       - init_A_in_nullspace   : A/As 를 null(_protect_v) 기저로 초기화
       - install_orth_hooks    : A/As grad 를 ⊥_protect_v 로 투영
       → ΔW @ V_old = 0 보장 → 이전 도메인 출력 불변(망각 0)
  ② KeepLoRA (gradient 기반 B 초기화)
       - keeplora_init_AB : zero-init B 대신 현재 도메인 첫 grad 의 null-space 투영 방향으로 init
  ③ CovForget (exp13) — 활성 공분산 Σ_prior 기반 보호 + 망각 penalty
       - finalize_cov_basis      : 학습중 모은 활성 → 저랭크 Σ=UΛUᵀ (동결 통계)
       - accumulate_cov_protect  : top-eig(Σ) 를 _protect_v 에 QR-merge (D2+ 보호, weight-SVD 대체)
       - forgetting_penalty      : R = tr(D_out⁻¹ ΔW Σ ΔWᵀ) 망각 1차 surrogate (loss penalty)

DeltaGPAdaptor 가 _protect_v / Σ 저랭크 기저(_cov_U,_cov_lam) / hook 핸들을 보유. 모델은 구조만,
adaptor 는 기법 상태/절차만 갖는다. (DDP 브로드캐스트는 train.py 가 adaptor 속성을 동기화)
"""
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as c
from dataset import AudioDataset, FocalLoss


# ── p_svd rank 선택 헬퍼 (D1 weight SVD 누적에너지 컷오프) ─────────────────────

def _energy_ratio_rank(S, p_svd, eps=1e-7):
    if p_svd is None: return None
    energy = S.float() ** 2; total = energy.sum()
    if total < eps: return 0
    ratio = torch.cumsum(energy, dim=0) / total
    return min(int((ratio < p_svd).sum().item()) + 1, S.numel())


def _apply_p_svd(n_nonzero, S, p_svd, eps=1e-7):
    r_pe = _energy_ratio_rank(S, p_svd, eps)
    return min(n_nonzero, r_pe) if r_pe is not None else n_nonzero


def _svd_kernel_v(W, p_svd, eps=1e-7):
    """conv weight [out,in,3,3] → kernel-space V 기저 [in*9, r] (None=rank0). p_svd 누적에너지 컷오프.
    DeltaGP가 ⊥ 시킬 old 방향 (exp13: D1 보호 전용. D2+ 보호는 Σ-cov top-eig)."""
    W_2d = W.view(W.shape[0], -1)                       # [out, in*9]
    _, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)
    r = _apply_p_svd(int((S > eps).sum()), S, p_svd, eps)
    if r == 0:
        return None
    return Vh[:r].t().contiguous()                      # [in*9, r]


# ── ① DeltaGP gradient hook 팩토리 (A1/A2 [r, n_ch*9] kernel-space 전용) ───────

def _make_a_weight_hook(V):
    """A [r, n_ch*9]: grad ⊥ V [n_ch*9, r_basis] (kernel-space)."""
    V = V.cpu().contiguous()
    def proj(grad):
        V_ = V.to(grad.device, grad.dtype)
        g  = grad.view(grad.shape[0], -1)
        return (g - (g @ V_) @ V_.t()).view_as(grad)
    return proj


def _make_a_kernel_combined_hook(V1, V2):
    """A1/A2: QR([V1|V2]) kernel-space 합산 (D3 phase)."""
    Q, _ = torch.linalg.qr(torch.cat([V1.cpu(), V2.cpu()], dim=1), mode='reduced')
    Q = Q.cpu().contiguous()
    def proj(grad):
        Q_ = Q.to(grad.device, grad.dtype)
        g  = grad.view(grad.shape[0], -1)
        return (g - (g @ Q_) @ Q_.t()).view_as(grad)
    return proj


# ── DeltaGPAdaptor ─────────────────────────────────────────────────────────────

class DeltaGPAdaptor:
    """LoRACNN14 에 DeltaGP(null-space) + KeepLoRA(grad-B) + CovForget(Σ-cov 보호/R penalty) 적용.

    기저/hook 핸들을 보유(model 은 구조만). 모든 메서드는 model 의 LoRA 파라미터를
    조작하지만 model 구조에는 의존만 하고 소유하지 않는다.

    bases (per conv, 12개 = 6블록×2conv):
      _protect_v : D1 weight SVD + 이전 도메인 활성공분산 Σ 의 top-eig 를 QR-merge 한
                   누적 orthonormal [in_ch*9, r]. DeltaGP hook/init 이 ⊥ 시킬 old 방향.
                   build_base_protect(D1) → (train+Σ수집) → accumulate_cov_protect(D2+) 로 단조 성장.
                   임의 T 확장: 도메인 추가 시 finalize+accumulate 반복.
    DDP: train.py 가 broadcast_attrs() 로 _protect_v 를 rank0→전체 동기화.
    """

    broadcast_attrs = ['_protect_v']

    def __init__(self, model):
        self.model = model
        self._protect_v    = []   # 누적 보호기저 (per conv)
        self._hook_handles = []
        # ── exp13: 이전 도메인 활성 공분산 Σ_prior 저랭크 캐시 ──
        #   _cov_U/_cov_lam: flat 리스트(캐시 도메인마다 n_convs=24개씩 extend). entry e → conv = e%24.
        #     Σ_c ≈ U Λ Uᵀ (U[n,k] eigenvecs, Λ[k] eigvals). 보호기저·망각 penalty R 의 근거.
        #   _cov_bnvar: conv별 BN running_var(D_out, D1 동결) → R 의 출력측 화이트닝.
        self._cov_U     = []
        self._cov_lam   = []
        self._cov_bnvar = []

    # ── DeltaGP: 누적 SVD 보호기저 (per conv, 12개) ──────────────────────────

    @staticmethod
    def _merge_basis(V_old, V_new):
        """누적 보호기저: orthonormal QR([V_old | V_new]). None 은 통과."""
        if V_new is None: return V_old
        if V_old is None: return V_new
        Q, _ = torch.linalg.qr(torch.cat([V_old.cpu().float(), V_new.cpu().float()], dim=1),
                               mode='reduced')
        return Q.contiguous()

    @torch.no_grad()
    def build_base_protect(self, p_svd=0.99, eps=1e-7):
        """초기 보호기저 _protect_v = D1 backbone conv weight의 kernel-space SVD (D1 보호 전용).
        D2+ 보호는 finalize_cov_basis/accumulate_cov_protect 의 Σ-cov 가 담당."""
        print(f'\nBuilding base protect basis (D1 weight SVD, p_svd={p_svd})...')
        self._protect_v = []
        for i, block in enumerate(self.model._conv_blocks):
            for j, conv in enumerate([block.conv1, block.conv2]):
                V = _svd_kernel_v(conv.weight.detach().float(), p_svd, eps)
                self._protect_v.append(V)
                print(f'  block{i+1}.conv{j+1}: '
                      f'{"rank=0, skipped" if V is None else f"V{tuple(V.shape)}"}')

    # ── exp13: 활성 공분산 Σ 캐시 / 보호 / 망각 penalty ──────────────────────
    #   true-IL 절대 준수: Σ 는 *학습 forward 도중* 자기 도메인 데이터로만 수집(reservoir).
    #   finalize 산출물(저랭크 Σ=U,Λ)은 동결 통계 — 이후 도메인은 이 숫자만 사용,
    #   다른 도메인 데이터에 절대 재접근하지 않는다 (BN 통계·기학습 ΔW 와 동일 합법성).

    @torch.no_grad()
    def finalize_cov_basis(self, domain_idx, energy=0.95, k_max=256, log_fn=print):
        """*학습 forward 도중* model 이 reservoir 로 모아둔 활성(model._captured_acts)을 SVD →
        conv별 저랭크 공분산 Σ_c=U Λ Uᵀ 캐시. ★ 별도 forward 없음. 자기 phase 자기 데이터(학습배치)만.
        true-IL: 결과는 동결 통계, 다른 도메인 데이터 미접근."""
        model = self.model
        model._capture_acts = False
        acts = model._captured_acts; model._captured_acts = {}
        dev = next(model.parameters()).device          # GPU 에서 SVD (CPU full SVD 회피)

        # svd_lowrank 의 내부 randn 이 학습 RNG 를 침범하지 않게 저장→고정→복원 (결정성)
        tstate = torch.get_rng_state()
        cstate = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(c.SEED + 777)

        n_convs = len(model._conv_blocks) * 2
        U_list, lam_list, rows = [], [], []
        for ci in range(n_convs):
            A = acts.get((ci // 2, ci % 2))                # [M, n] (cpu) or None
            if A is None or A.numel() == 0:
                U_list.append(None); lam_list.append(None); rows.append(0); continue
            A = A.float().to(dev)                          # GPU
            M = A.shape[0]; rows.append(M)
            # top-(≤k_max) 만 필요 → 랜덤화 저랭크 SVD (full SVD 대비 ~10-100x). V[n,q]=우특이벡터.
            q = min(k_max + 8, M, A.shape[1])
            try:
                _, S, V = torch.svd_lowrank(A, q=q)        # A ≈ U·diag(S)·Vᵀ,  V [n, q]
            except Exception:
                U_list.append(None); lam_list.append(None); continue
            eig = (S.float() ** 2) / max(M, 1)             # 활성 공분산 고유값
            total = float(eig.sum())
            if total <= 0:
                U_list.append(None); lam_list.append(None); continue
            ratio = torch.cumsum(eig, 0) / total
            k = min(int((ratio < energy).sum().item()) + 1, k_max, V.shape[1])
            U_list.append(V[:, :k].contiguous())           # [n, k]  (GPU 상주 → R 매-step 전송 제거)
            lam_list.append(eig[:k].contiguous())          # [k]     (GPU)

        torch.set_rng_state(tstate)
        if cstate is not None:
            torch.cuda.set_rng_state_all(cstate)

        # BN 출력 분산 D_out (conv1→bnF, conv2→bnS) — R 의 출력측 화이트닝 (GPU 상주)
        self._cov_bnvar = []
        for block in model._conv_blocks:
            self._cov_bnvar.append(block.bnF[0].running_var.detach().float().to(dev))
            self._cov_bnvar.append(block.bnS[0].running_var.detach().float().to(dev))

        self._cov_U.extend(U_list); self._cov_lam.extend(lam_list)   # flat, +n_convs
        log_fn(f'  cov finalized (domain phase {domain_idx}, GPU svd_lowrank): '
               f'rows/conv={rows} k/conv={[None if u is None else u.shape[1] for u in U_list]}')

    @torch.no_grad()
    def accumulate_cov_protect(self, energy_protect=0.9, log_fn=print):
        """방금 캐시한 도메인의 top-eig(Σ)(energy_protect까지)를 _protect_v 에 QR-merge (hard 보호).
        나머지 모드는 forgetting_penalty(soft R)가 담당."""
        n_convs = len(self.model._conv_blocks) * 2
        base = len(self._cov_U) - n_convs
        kept = []
        for ci in range(n_convs):
            U = self._cov_U[base + ci]; lam = self._cov_lam[base + ci]
            if U is None:
                kept.append(0); continue
            ratio = torch.cumsum(lam, 0) / float(lam.sum())
            kp = min(int((ratio < energy_protect).sum().item()) + 1, U.shape[1])
            self._protect_v[ci] = self._merge_basis(self._protect_v[ci], U[:, :kp].cpu())
            kept.append(kp)
        log_fn(f'  cov→protect: hard 보호 모드/conv = {kept}')

    def forgetting_penalty(self, eps=1e-5):
        """R = Σ_{prior Σ}Σ_c Σ_i Λ_i ‖ D_out^{-1/2} (ΔW_c u_i) ‖²  (미분가능, 현재 학습 도메인 ΔW).
        = tr(D_out^{-1} ΔW Σ ΔWᵀ) 의 저랭크 전개 = 이전 도메인 망각의 1차 surrogate.
        ★ 캐시된 통계(U,Λ)만 사용 — 다른 도메인 데이터 미접근 (true-IL)."""
        if not self._cov_U:
            return None
        model = self.model
        t = model.active_domain_count - 1               # 현재 학습 도메인
        n_convs = len(model._conv_blocks) * 2
        R = None
        for e, (U, lam) in enumerate(zip(self._cov_U, self._cov_lam)):
            if U is None:
                continue
            ci = e % n_convs
            bi, conv = ci // 2, ci % 2
            cb = model.domain_loras[t][bi]
            dW = cb.get_delta_conv1() if conv == 0 else cb.get_delta_conv2()
            dW2 = dW.reshape(dW.shape[0], -1)            # [out, n]
            Ud  = U.to(dW2.device, dW2.dtype)            # [n, k]
            dinv = (self._cov_bnvar[ci].to(dW2.device) + eps).rsqrt().unsqueeze(1)  # [out,1]
            proj = (dW2 @ Ud) * dinv                     # [out, k]  (D_out^{-1/2} 행 스케일)
            term = (lam.to(proj.device) * (proj ** 2).sum(dim=0)).sum()
            R = term if R is None else R + term
        return R

    # ── InfLoRA식 A 초기화: null(V_old) ──────────────────────────────────────

    def _old_V_for_conv(self, ci):
        """conv ci의 누적 old-direction basis V_old [n, r_old] (orthonormal cols)."""
        return self._protect_v[ci] if (self._protect_v and ci < len(self._protect_v)) else None

    @staticmethod
    def _nullspace_rows(V, n, r):
        """null(V)의 직교기저에서 r개 행 [r, n] 반환. 부족분은 0-pad."""
        if V is None:
            g = torch.Generator().manual_seed(20260606)
            Q, _ = torch.linalg.qr(torch.randn(n, r, generator=g), mode='reduced')
            return Q.t().contiguous()
        Vc = V.cpu().float()
        r_old = Vc.shape[1]
        if r_old >= n:                                   # null-space 없음
            return torch.zeros(r, n)
        Qf, _ = torch.linalg.qr(Vc, mode='complete')     # [n, n]
        null_b = Qf[:, r_old:]                            # [n, n-r_old] ⊥ V (orthonormal)
        out = torch.zeros(r, n)
        k = min(r, null_b.shape[1])
        out[:k] = null_b[:, :k].t()
        return out

    @torch.no_grad()
    def init_A_in_nullspace(self, domain_idx):
        """A1/A2/As1/As2를 null(_protect_v) 직교기저로 초기화 (결정론, 전 rank 동일)."""
        cnt = 0
        for bi, cb in enumerate(self.model.domain_loras[domain_idx]):
            # 구조항 A1/A2 + BN항 As1/As2 모두 같은 conv의 null(V_old)로 초기화
            for A_param, ci in [(cb.A1, bi * 2), (cb.As1, bi * 2),
                                (cb.A2, bi * 2 + 1), (cb.As2, bi * 2 + 1)]:
                r, n = A_param.shape
                V = self._old_V_for_conv(ci)
                A_new = self._nullspace_rows(V, n, r)
                A_param.data.copy_(A_new.to(A_param.device, A_param.dtype))
                cnt += 1
        print(f'  A init in null-space: domain_loras[{domain_idx}] ({cnt} convs)')

    # ── DeltaGP: orth hooks (단일 메서드, 임의 도메인) ────────────────────────

    def install_orth_hooks(self, domain_idx):
        """도메인 domain_idx 의 A1/A2/As1/As2 grad 를 ⊥ 누적 _protect_v 로 투영(DeltaGP).
        _protect_v 가 이미 QR([D1 | D2..D(t)]) 누적이므로 단일 weight-hook 으로
        구 install_d2 / install_d3_combined 동작을 그대로 포함(QR 합산 동일). 임의 T 대응."""
        self.clear_orth_hooks()
        if not self._protect_v:
            raise RuntimeError('build_base_protect()를 먼저 호출하세요.')
        n = 0
        for bi, cb in enumerate(self.model.domain_loras[domain_idx]):
            ci1, ci2 = bi * 2, bi * 2 + 1
            V1 = self._protect_v[ci1] if ci1 < len(self._protect_v) else None
            V2 = self._protect_v[ci2] if ci2 < len(self._protect_v) else None
            # 풀랭크(null-space 없음) 블록은 스킵 — 보호할 자유공간 없음
            if V1 is not None and V1.shape[1] < V1.shape[0]:
                self._hook_handles.append(cb.A1.register_hook(_make_a_weight_hook(V1)));  n += 1
                self._hook_handles.append(cb.As1.register_hook(_make_a_weight_hook(V1))); n += 1
            if V2 is not None and V2.shape[1] < V2.shape[0]:
                self._hook_handles.append(cb.A2.register_hook(_make_a_weight_hook(V2)));  n += 1
                self._hook_handles.append(cb.As2.register_hook(_make_a_weight_hook(V2))); n += 1
        print(f'  orth hooks installed on domain_loras[{domain_idx}] '
              f'({n} handles, ⊥ _protect_v)')

    def clear_orth_hooks(self):
        for h in self._hook_handles: h.remove()
        self._hook_handles = []

    # ── ③ KeepLoRA: gradient 기반 B 초기화 (현재 도메인 데이터만) ─────────────

    @staticmethod
    def _align_rows(Vh, V_old, r, n, eps=1e-8):
        """SVD 우특이벡터 Vh[k,n] 상위 r행을 A로. ⊥V_old 강제 재투영 + 행정규화 + 0-pad.
        σ>0 행은 이미 ⊥V_old라 불변, rank부족분(σ≈0)만 정리 → 망각 보장 안전."""
        out = torch.zeros(r, n, device=Vh.device, dtype=Vh.dtype)
        k = min(r, Vh.shape[0])
        A = Vh[:k].clone()
        if V_old is not None:
            A = A - (A @ V_old) @ V_old.t()              # ⊥ V_old 강제
        A = A / A.norm(dim=1, keepdim=True).clamp_min(eps)
        out[:k] = A
        return out

    @torch.enable_grad()
    def keeplora_init_AB(self, domain_idx, df, device, eta, n_batches,
                         align_A=True, log_fn=print):
        """KeepLoRA(ICLR2026) gradient init. 현재 도메인 데이터로 G=∂L/∂ΔW 캡처 후:
          align_A=True : A를 G의 null(V_old)-투영 상위 특이방향으로 정렬(임의 QR basis 대체)
                         → 같은 안전공간·같은 rank에서 'D3가 가려는 방향'에 학습축 정렬, 망각 0 유지.
          align_A=False: A 기존 null basis 유지, B만 init (구 B-only와 동일).
        B = −eta·(G@Aᵀ)/scale ⇒ scale·(B@A) = −eta·proj_rowspace(A)(G). 공유 Bs는 conv1·2 BN항 합산.
        true-IL: 현재 도메인 데이터만 forward. svd_lowrank는 torch RNG 저장·복원으로 downstream 불변."""
        model = self.model
        ds = AudioDataset(df, is_train=True)
        loader = DataLoader(ds, batch_size=c.BATCH_SIZE, shuffle=False,
                            num_workers=0, drop_last=False)
        criterion = (FocalLoss(gamma=c.FOCAL_GAMMA, label_smoothing=c.LABEL_SMOOTHING)
                     if c.USE_FOCAL_LOSS
                     else nn.CrossEntropyLoss(label_smoothing=c.LABEL_SMOOTHING))
        np.random.seed(c.SEED); random.seed(c.SEED)
        model.eval()
        # ── G = ∂L/∂ΔW 누적 (conv별, n_batches 평균) ──
        G_acc = {}
        model._capture_deltas = True; model._capture_domain = domain_idx
        used = 0
        for batch in loader:
            if used >= n_batches:
                break
            model._captured_deltas = {}
            model.zero_grad(set_to_none=True)
            audio = batch[0].float().to(device)
            tgt   = batch[1].float().to(device).argmax(-1)
            logits, _ = model.forward(audio, use_spec_aug=False)
            (criterion(logits, tgt) / n_batches).backward()
            for key, delta in model._captured_deltas.items():
                if delta.grad is not None:
                    g = delta.grad.detach().reshape(delta.shape[0], -1).float()  # [out, n]
                    G_acc[key] = g if key not in G_acc else G_acc[key] + g
            used += 1
        model._capture_deltas = False; model._captured_deltas = {}
        model.zero_grad(set_to_none=True)

        # svd_lowrank 재현성 + downstream RNG 불변(깨끗한 비교): torch RNG 저장→고정→복원
        torch_state = torch.get_rng_state()
        cuda_state  = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(c.SEED + 12345)

        n_a = n_b = 0
        with torch.no_grad():
            for bi, cb in enumerate(model.domain_loras[domain_idx]):
                for conv in (0, 1):                       # conv별 A 정렬 + 구조항 B
                    G = G_acc.get((bi, conv))
                    if G is None:
                        continue
                    V  = self._old_V_for_conv(bi * 2 + conv)
                    Vd = V.to(G.device, G.dtype) if V is not None else None
                    G_null = (G - (G @ Vd) @ Vd.t()) if Vd is not None else G
                    A_p  = cb.A1 if conv == 0 else cb.A2
                    As_p = cb.As1 if conv == 0 else cb.As2
                    B_p  = cb.B1 if conv == 0 else cb.B2
                    r_A, n = A_p.shape; r_B = As_p.shape[0]
                    if align_A and float(G_null.abs().max()) > 1e-8:
                        q = min(max(r_A, r_B) + 8, min(G_null.shape))
                        _, _, Vt = torch.svd_lowrank(G_null, q=q)   # Vt [n, q]
                        Vh = Vt.t()                                  # [q, n] 우특이벡터(행)
                        A_p.data.copy_(self._align_rows(Vh, Vd, r_A, n).to(A_p.dtype))
                        As_p.data.copy_(self._align_rows(Vh, Vd, r_B, n).to(As_p.dtype))
                        n_a += 2
                    A_use = A_p.data.to(G.device, G.dtype)
                    B_p.data.copy_((-eta * (G @ A_use.t()) / cb.scale_A).to(B_p.dtype)); n_b += 1
                # 공유 Bs: conv1·conv2 BN항(As1,As2) 합산
                grad_bs = None
                for conv in (0, 1):
                    G = G_acc.get((bi, conv))
                    if G is None:
                        continue
                    As_use = (cb.As1 if conv == 0 else cb.As2).data.to(G.device, G.dtype)
                    term = G @ As_use.t()
                    grad_bs = term if grad_bs is None else grad_bs + term
                if grad_bs is not None:
                    cb.Bs.data.copy_((-eta * grad_bs / cb.scale_B).to(cb.Bs.dtype)); n_b += 1

        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        log_fn(f'  KeepLoRA init: domain_loras[{domain_idx}] align_A={align_A} eta={eta} '
               f'n_batches={used} (A정렬 {n_a}, B set {n_b})')
