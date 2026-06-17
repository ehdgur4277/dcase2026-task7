"""
model_adaptor.py — baseline(CNN14) 대비 추가된 **모델 구조** (LoRA adaptor)

이 파일은 "원본 baseline 과 비교해 모델 구조에 추가된 것" 만 담는다.
훈련기법(DeltaGP null-space 투영 / SplitLoRA k* / KeepLoRA grad-B init)은
train_adaptor.py 로 분리되어 있고, 순수 훈련 루프는 train.py 에 있다.

baseline(common/backbone.py CNN14Backbone): conv 동결, BN 공유(D1), fc.
  → 도메인 적응을 위해 각 conv 에 가산 LoRA ΔW 를 주입하는 것이 본 adaptor 의 핵심.

구조 요약 (baseline 에 없던 것):
  CorrAdditiveLoRABlock : 블록당 ΔW = 구조항 s_A·(B@A) + BN항 s_B·(Bs@As)  (가산 분해)
  LoRACNN14             : 동결 backbone forward 에 conv 별 ΔW 를 더해 주입.
                          학습 대상(phase 별 requires_grad) 설정 + KeepLoRA 용 ΔW grad 캡처 훅 제공.

LoRACNN14 가 보유한 학습기법 관련 '상태'는 forward 와 직접 결합된 것(ΔW grad 캡처)만이다.
SVD 기저/그래디언트 hook/A 초기화 등 실제 기법 로직은 train_adaptor.DeltaGPAdaptor 가 model 을
인자로 받아 수행한다 (구조 ↔ 기법 분리).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as c


# baseline CNN14 의 6 ConvBlock (in_ch, out_ch). LoRA rank 를 블록별로 매핑할 때 기준.
BLOCK_CONFIGS = [
    (1,    64),
    (64,   128),
    (128,  256),
    (256,  512),
    (512,  1024),
    (1024, 2048),
]


def ranks_to_block_configs(rank_A, rank_B):
    """(rank_A 리스트, rank_B 리스트) → 블록별 (r_A, r_B) 튜플 리스트."""
    return list(zip(rank_A, rank_B))


def drift_to_block_configs(drift_conv_bn, drift_bn, ceiling):
    """백본 기준 per-block CKA drift → 블록별 (r_A, r_B). ★도메인별 자동 rank 산출(임의 T).
      r_A ∝ 구조 drift(conv_bn),  r_B ∝ BN흡수 drift(Δ=conv_only−conv_bn, ≥0).
      단일 스케일 s 로 최고블록 r_A+r_B = ceiling. r_A 는 block in_ch*9 상한, 둘 다 최소 1.
    검증: D2/D3 의 실측 drift → 기존 검증 rank([4,39,..]/[3,3,1,..] 등)을 정확히 재현."""
    db  = [max(0.0, float(x)) for x in drift_bn]
    cb  = [float(x) for x in drift_conv_bn]
    tot = [a + b for a, b in zip(cb, db)]
    s   = float(ceiling) / max(tot)
    in_dims = [in_ch * 9 for (in_ch, _) in BLOCK_CONFIGS]      # r_A 상한 (block1=9)
    r_A = [max(1, min(int(round(s * a)), in_dims[i])) for i, a in enumerate(cb)]
    r_B = [max(1, int(round(s * b))) for b in db]
    return list(zip(r_A, r_B)), r_A, r_B


# ── CorrAdditiveLoRABlock ──────────────────────────────────────────────────────

class CorrAdditiveLoRABlock(nn.Module):
    """
    CorrLoRA식 가산 분해: ΔW = 구조항(독립, rank r_A) + BN항(공유 Bs, rank r_B).
    곱(B@G@A)은 rank=min(r_A,r_B) 병목으로 폐기 — 가산은 rank=r_A+r_B (병목 없음).

    구조(독립) 항 [rank r_A]:
      A1 [r_A, in_ch*9]   — conv1 입력투영, null(V_old) 초기화 + grad ⊥ V_old (DeltaGP)
      A2 [r_A, out_ch*9]  — conv2 입력투영, 동일
      B1/B2 [out_ch, r_A] — conv별 출력측, 완전 학습, init 0
    BN(공유) 항 [rank r_B]:
      As1 [r_B, in_ch*9]  — conv1 입력투영, null(V_old) 초기화 + grad ⊥ V_old (DeltaGP)
      As2 [r_B, out_ch*9] — conv2 입력투영, 동일
      Bs  [out_ch, r_B]   — conv1/conv2 공유 출력기저 = BN 역할, 완전 학습, init 0

    ΔW1 = s_A·(B1@A1) + s_B·(Bs@As1) → view(out,in,3,3)
    ΔW2 = s_A·(B2@A2) + s_B·(Bs@As2) → view(out,out,3,3)

    A1·As1(및 A2·As2) 모두 V_old에 직교(init + hook)하므로 두 항 합쳐
    ΔW @ V_old = 0 이 B1/B2/Bs와 무관하게 성립 → 출력측 자유 학습해도 망각 방지 (가소성).
    B·Bs init=0 → 초기 ΔW=0. dB부터 흐르고 B 자라면 A 따라옴 (LoRA dynamic).

    ※ A/As 의 null-space 초기화·직교 hook 은 구조가 아니라 '훈련기법' → train_adaptor.DeltaGPAdaptor.
       여기서는 파라미터 텐서와 ΔW 합성만 정의한다.
    """
    def __init__(self, in_ch, out_ch, r_A, r_B, alpha, gate_cfg=None):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.r_A, self.r_B = r_A, r_B
        self.scale_A = alpha / r_A
        self.scale_B = alpha / r_B
        # ── 블록 게이트 g_b = sigmoid(θ) ∈ (0,1) ──
        #   ΔW_block ← g_b·ΔW_block. 학습형 스칼라. null-space 고갈 블록은 g→0 으로 수렴해
        #   직전 누적(전이) 표현을 회수 → 증분 단계가 성능을 역행시키지 못함(단조 비감소 보장).
        #   도메인 id 아님(모든 입력 공통) → router 아님. use=False면 g≡1 (exp07 과 수치 동일).
        self.use_gate = bool(gate_cfg and gate_cfg.get('use', False))
        if self.use_gate:
            self.gate_theta = nn.Parameter(
                torch.tensor(float(gate_cfg.get('init_logit', 4.0))))   # sigmoid(4)=0.982 ≈ exp07
        # 독립(구조) 항 rank r_A: A 입력측 null-space(DeltaGP), B 출력측 자유 init0
        self.A1 = nn.Parameter(torch.randn(r_A, in_ch  * 9) * 0.02)
        self.A2 = nn.Parameter(torch.randn(r_A, out_ch * 9) * 0.02)
        self.B1 = nn.Parameter(torch.zeros(out_ch, r_A))
        self.B2 = nn.Parameter(torch.zeros(out_ch, r_A))
        # 공유(BN) 항 rank r_B: As 입력측 null-space(DeltaGP), Bs 공유 출력기저=BN 자유 init0
        self.As1 = nn.Parameter(torch.randn(r_B, in_ch  * 9) * 0.02)
        self.As2 = nn.Parameter(torch.randn(r_B, out_ch * 9) * 0.02)
        self.Bs  = nn.Parameter(torch.zeros(out_ch, r_B))

    def _gate(self):
        return torch.sigmoid(self.gate_theta) if self.use_gate else 1.0

    def get_delta_conv1(self):
        """ΔW1 [out,in,3,3] = g_b·(구조항 s_A·(B1@A1) + BN항 s_B·(Bs@As1))  (가산, rank=r_A+r_B)"""
        struct = self.scale_A * (self.B1 @ self.A1)
        bn     = self.scale_B * (self.Bs @ self.As1)
        dW = (struct + bn).view(self.out_ch, self.in_ch, 3, 3)
        return self._gate() * dW if self.use_gate else dW

    def get_delta_conv2(self):
        """ΔW2 [out,out,3,3] = g_b·(구조항 s_A·(B2@A2) + BN항 s_B·(Bs@As2))"""
        struct = self.scale_A * (self.B2 @ self.A2)
        bn     = self.scale_B * (self.Bs @ self.As2)
        dW = (struct + bn).view(self.out_ch, self.out_ch, 3, 3)
        return self._gate() * dW if self.use_gate else dW


# ── LoRACNN14 ──────────────────────────────────────────────────────────────────

class LoRACNN14(nn.Module):
    """
    동결 CNN14 backbone + 도메인별 CorrAdditiveLoRABlock 가산 주입.

    baseline 과의 차이:
      - conv weight 는 동결(D1). 대신 conv 출력에 ΔW conv2d 결과를 더한다.
      - 도메인(D2,D3)마다 LoRA set 하나. active_domain_count 로 추론 시 합산 범위 제어
        (도메인 정보 없이 누적 ΔW 적용 → true-IL).

    학습기법(DeltaGP/SplitLoRA/KeepLoRA)은 이 클래스 밖(train_adaptor.py)에 있다.
    단, KeepLoRA 가 ∂L/∂ΔW 를 필요로 하므로 forward 가 ΔW 텐서를 retain_grad 로 노출하는
    '캡처 훅'만 여기 둔다 (forward 와 분리 불가능한 최소 결합).

    tune_bn=True: backbone BN 파라미터 함께 학습. (현재 config 는 tune_bn=False → BN 완전 동결)
    """
    def __init__(self, backbone, domain_block_cfgs, alpha=64.0, tune_bn=True,
                 gate_cfg=None):
        """domain_block_cfgs: 도메인별 블록 cfg 리스트의 리스트 (임의 길이 T).
        각 원소 = ranks_to_block_configs/drift_to_block_configs 결과(블록별 (r_A,r_B))."""
        super().__init__()
        self.backbone = backbone
        self.tune_bn  = tune_bn

        self.domain_loras = nn.ModuleList()
        for block_cfgs in domain_block_cfgs:          # 임의 T 도메인 어댑터
            lora_set = nn.ModuleList()
            for (in_ch, out_ch), (r_A, r_B) in zip(BLOCK_CONFIGS, block_cfgs):
                lora_set.append(CorrAdditiveLoRABlock(in_ch, out_ch, r_A, r_B, alpha,
                                                      gate_cfg=gate_cfg))
            self.domain_loras.append(lora_set)

        self.active_domain_count = 1

        # ── KeepLoRA 용 ΔW grad 캡처 (forward 결합 최소 상태) ──
        #   train_adaptor.DeltaGPAdaptor.keeplora_init_AB 가 ∂L/∂ΔW 를 얻기 위해 켠다.
        self._capture_deltas  = False
        self._capture_domain  = None
        self._captured_deltas = {}
        # ── exp13: conv 입력 활성 캡처 (Σ_prior 저랭크 공분산용) ──
        #   ★ 별도 forward 없이 *학습 forward 도중* 모은다. train_phase 가 매 K배치 _capture_acts 토글.
        #   conv 입력을 3×3 unfold → [N*L, in*9] 샘플. conv별 reservoir(상한 _cov_cap)로 누적(cpu).
        self._capture_acts   = False
        self._captured_acts  = {}     # (block_idx, conv_in_block) → [≤cap, in*9] (cpu)
        self._cov_persample  = 256    # 배치당 보관 행 수(서브샘플)
        self._cov_cap        = 4096   # conv별 reservoir 상한
        self._cov_gen        = {}     # device → Generator. ★캡처 서브샘플 전용 → 학습 RNG 불침범(결정성)

    # ── 학습 대상(phase 별 requires_grad) 설정 ──────────────────────────────────

    def set_domain_trainable(self, t):
        """도메인 t 의 LoRA(게이트 포함)만 학습, 나머지 도메인·fc 동결. (임의 T 일반)"""
        for d, lora_set in enumerate(self.domain_loras):
            req = (d == t)
            for p in lora_set.parameters(): p.requires_grad = req
        for p in self.backbone.fc.parameters(): p.requires_grad = False
        if self.tune_bn: self._unfreeze_bn()

    def _unfreeze_bn(self):
        b = self.backbone
        for p in b.bn0[0].parameters(): p.requires_grad = True
        for block in self._conv_blocks:
            for p in block.bnF[0].parameters(): p.requires_grad = True
            for p in block.bnS[0].parameters(): p.requires_grad = True

    def set_bn_train_for_tuned(self):
        """train_phase epoch 루프에서 backbone BN을 train 모드로 유지."""
        if not self.tune_bn: return
        b = self.backbone
        b.bn0[0].train()
        for block in self._conv_blocks:
            block.bnF[0].train(); block.bnS[0].train()

    @property
    def _conv_blocks(self):
        b = self.backbone
        return [b.conv_block1, b.conv_block2, b.conv_block3,
                b.conv_block4, b.conv_block5, b.conv_block6]

    # ── forward (ΔW 가산 주입) ───────────────────────────────────────────────

    def _apply_lora(self, x_in, base_out, block_idx, conv_in_block):
        if self._capture_acts:
            # conv 입력 a = unfold(x_in) [N, in*9, L] → [N*L, in*9]. 배치당 서브샘플 후 reservoir 누적.
            #   ★ 모든 randperm 은 전용 generator 사용 → 학습 RNG(dropout/mixup) 불침범, 결정성 유지.
            def _g(dev):
                g = self._cov_gen.get(dev)
                if g is None:
                    g = torch.Generator(device=dev); g.manual_seed(20260613)
                    self._cov_gen[dev] = g
                return g
            a = F.unfold(x_in, kernel_size=3, padding=1)          # [N, in*9, L]
            a = a.permute(0, 2, 1).reshape(-1, a.shape[1])         # [N*L, in*9]
            if a.shape[0] > self._cov_persample:
                idx = torch.randperm(a.shape[0], generator=_g(a.device), device=a.device)
                a = a[idx[:self._cov_persample]]
            a = a.detach().cpu()
            key = (block_idx, conv_in_block)
            prev = self._captured_acts.get(key)
            if prev is None:
                self._captured_acts[key] = a
            else:
                cat = torch.cat([prev, a], dim=0)
                if cat.shape[0] > self._cov_cap:                   # reservoir 상한 (전 학습구간 대표샘플)
                    cat = cat[torch.randperm(cat.shape[0], generator=_g(cat.device))[:self._cov_cap]]
                self._captured_acts[key] = cat
        out = base_out
        for d in range(self.active_domain_count):
            cb = self.domain_loras[d][block_idx]
            delta = cb.get_delta_conv1() if conv_in_block == 0 else cb.get_delta_conv2()
            if self._capture_deltas and d == self._capture_domain:
                delta.retain_grad()                       # ∂L/∂ΔW 확보용 (KeepLoRA A-align)
                self._captured_deltas[(block_idx, conv_in_block)] = delta
            out = out + F.conv2d(x_in, delta, padding=1)
        return out

    def forward(self, x, use_spec_aug=False):
        b = self.backbone
        x = b.spectrogram_extractor(x)
        x = b.logmel_extractor(x)
        x = x.transpose(1, 3); x = b.bn0[0](x); x = x.transpose(1, 3)
        if use_spec_aug and self.training:
            x = b.spec_augmenter(x)
        for i, block in enumerate(self._conv_blocks):
            x1 = x
            x = self._apply_lora(x1, block.conv1(x1), i, 0)
            x = F.relu_(block.bnF[0](x))
            x2 = x
            x = self._apply_lora(x2, block.conv2(x2), i, 1)
            x = F.relu_(block.bnS[0](x))
            x = F.avg_pool2d(x, (2, 2))
            x = F.dropout(x, c.DROPOUT, training=self.training)
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2); x2 = torch.mean(x, dim=2)
        feat = x1 + x2
        return b.fc(feat), feat
