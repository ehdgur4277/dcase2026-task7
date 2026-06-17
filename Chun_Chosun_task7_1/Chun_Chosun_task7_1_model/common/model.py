"""
공용 모델 정의 — DeltaOrthoMultiLoRACNN14 + 체크포인트 로더.

기존에 train.py / eval.py 에 각각 중복 정의돼 있던 모델 클래스를 한곳으로 모은다
(reorg_prompt_2.md: "공용 로직은 반드시 common/ 으로 분리, 중복 코드 금지").

  active_domain_count:
    0: D1 (LoRA 없음)
    1: backbone + LoRA_D2
    2: backbone + LoRA_D2 + LoRA_D3

학습용(main.py / single_main.py)은 SVD 베이스/직교 hook 메서드까지 모두 사용하고,
추론/평가용(inference.py / evaluate.py)은 forward + load 만 사용한다.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as c
from backbone import CNN14Backbone
from trainer import LoRAConv2d


# ============================================================
# Energy-ratio truncation helper
# ============================================================
def energy_ratio_rank(S, p_svd, eps=1e-7):
    """누적 에너지 비율 컷오프. m = min{ m : Σσ_i² / Σσ² ≥ p_svd }."""
    if p_svd is None:
        return None
    energy = (S.float() ** 2)
    total = energy.sum()
    if total < eps:
        return 0
    ratio = torch.cumsum(energy, dim=0) / total
    m = int((ratio < p_svd).sum().item()) + 1
    return min(m, S.numel())


# Backward-compatible alias (train.py 내부 명칭).
_energy_ratio_rank = energy_ratio_rank


# ============================================================
# Delta-based Orthogonal Multi-Domain LoRA Model
# ============================================================
class DeltaOrthoMultiLoRACNN14(nn.Module):
    def __init__(self, backbone, rank=64, alpha=64.0, n_domains=2, domain_ranks=None):
        super().__init__()
        self.backbone = backbone
        if domain_ranks is None:
            domain_ranks = [rank] * n_domains
        if len(domain_ranks) != n_domains:
            raise ValueError(f'domain_ranks length mismatch: {len(domain_ranks)} != {n_domains}')
        block_configs = [
            (1, 64), (64, 128), (128, 256),
            (256, 512), (512, 1024), (1024, 2048),
        ]
        self.domain_loras = nn.ModuleList()
        for domain_rank in domain_ranks:
            lora_set = nn.ModuleList()
            for in_ch, out_ch in block_configs:
                lora_set.append(LoRAConv2d(in_ch, out_ch, domain_rank, alpha))
                lora_set.append(LoRAConv2d(out_ch, out_ch, domain_rank, alpha))
            self.domain_loras.append(lora_set)

        self.active_domain_count = 1
        self._orth_bases = []     # D1 V_r: [in_ch*k*k, r]
        self._d1_out_bases = []   # D1 U_r: [out_ch, r]
        self._d2_delta_v = []     # D2 ΔW right singular: [in_ch*k*k, r]
        self._d2_delta_u = []     # D2 ΔW left  singular: [out_ch, r]
        self._hook_handles = []

    # ----------------------------------------------------------
    def _apply_loras(self, x, base_conv, conv_idx):
        out = base_conv(x)
        for d in range(self.active_domain_count):
            out = out + F.conv2d(
                x, self.domain_loras[d][conv_idx].get_delta(), padding=1
            )
        return out

    def set_d2_trainable(self):
        for p in self.domain_loras[0].parameters():
            p.requires_grad = True
        for p in self.domain_loras[1].parameters():
            p.requires_grad = False
        for p in self.backbone.fc.parameters():
            p.requires_grad = False

    def set_d3_trainable(self):
        for p in self.domain_loras[0].parameters():
            p.requires_grad = False
        for p in self.domain_loras[1].parameters():
            p.requires_grad = True
        for p in self.backbone.fc.parameters():
            p.requires_grad = False

    # ----------------------------------------------------------
    @torch.no_grad()
    def build_d1_bases(self, projection_rank=None, p_svd=None, eps=1e-7):
        if p_svd is not None:
            print(f'Building D1 SVD bases  (projection_rank={projection_rank}, p_svd={p_svd})...')
        else:
            print(f'Building D1 SVD bases  (projection_rank={projection_rank}, p_svd=None)...')
        blocks = [
            self.backbone.conv_block1, self.backbone.conv_block2,
            self.backbone.conv_block3, self.backbone.conv_block4,
            self.backbone.conv_block5, self.backbone.conv_block6,
        ]
        self._orth_bases = []
        self._d1_out_bases = []
        for i, block in enumerate(blocks):
            for j, conv in enumerate([block.conv1, block.conv2]):
                conv_idx = i * 2 + j
                W = conv.weight.detach().float().cpu()
                W_2d = W.view(W.shape[0], -1)

                rank_cap = (
                    projection_rank if projection_rank is not None
                    else int(self.domain_loras[0][conv_idx].lora_A.shape[0])
                )
                rank_cap = max(0, min(rank_cap, min(W_2d.shape)))

                if rank_cap == 0:
                    self._orth_bases.append(None)
                    self._d1_out_bases.append(None)
                    continue

                U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)
                r_eps = int((S > eps).sum().item())
                rank = min(rank_cap, r_eps)

                r_pe = energy_ratio_rank(S, p_svd, eps)
                if r_pe is not None:
                    rank = min(rank, r_pe)

                if rank == 0:
                    self._orth_bases.append(None)
                    self._d1_out_bases.append(None)
                    continue

                V_r = Vh[:rank].t().contiguous()
                U_r = U[:, :rank].contiguous()
                self._orth_bases.append(V_r)
                self._d1_out_bases.append(U_r)
                tag = f', p_svd={p_svd}→r_pe={r_pe}' if r_pe is not None else ''
                print(
                    f'  conv_block{i+1}.conv{j+1}: W {tuple(W_2d.shape)} '
                    f'-> V_r {tuple(V_r.shape)}, U_r {tuple(U_r.shape)} '
                    f'(cap={rank_cap}, r_eps={r_eps}{tag})'
                )

    # ----------------------------------------------------------
    def install_orth_hooks(self, domain_idx):
        """D2용: lora_A gradient ⊥ D1 V_r."""
        self.clear_orth_hooks()
        if not self._orth_bases:
            raise RuntimeError('build_d1_bases()를 먼저 호출하세요.')
        lora_set = self.domain_loras[domain_idx]
        n = 0
        for conv_idx, lora_module in enumerate(lora_set):
            if conv_idx >= len(self._orth_bases) or self._orth_bases[conv_idx] is None:
                continue

            def make_hook(V_r):
                def project_grad(grad):
                    V = V_r.to(device=grad.device, dtype=grad.dtype)
                    g = grad.view(grad.shape[0], -1)
                    return (g - (g @ V) @ V.t()).view_as(grad)
                return project_grad

            self._hook_handles.append(
                lora_module.lora_A.register_hook(make_hook(self._orth_bases[conv_idx]))
            )
            n += 1
        print(f'  Installed {n} D1-orth hooks on domain_loras[{domain_idx}]')

    # ----------------------------------------------------------
    @torch.no_grad()
    def build_d2_delta_bases(self, p_svd=None, eps=1e-7):
        if p_svd is not None:
            print(f'Building D2 delta SVD bases  (p_svd={p_svd})...')
        else:
            print('Building D2 delta SVD bases  (p_svd=None, 모든 nonzero σ 유지)...')
        self._d2_delta_v = []
        self._d2_delta_u = []
        for conv_idx, lora_module in enumerate(self.domain_loras[0]):
            delta_2d = lora_module.get_delta().detach().float().cpu().view(
                lora_module.out_ch, -1
            )

            if delta_2d.numel() == 0 or delta_2d.norm() < eps:
                self._d2_delta_v.append(None)
                self._d2_delta_u.append(None)
                print(f'  D2 delta layer {conv_idx}: zero delta, skipped')
                continue

            U, S, Vh = torch.linalg.svd(delta_2d, full_matrices=False)
            r_eps = int((S > eps).sum().item())
            r = r_eps

            r_pe = energy_ratio_rank(S, p_svd, eps)
            if r_pe is not None:
                r = min(r, r_pe)

            if r == 0:
                self._d2_delta_v.append(None)
                self._d2_delta_u.append(None)
                print(f'  D2 delta layer {conv_idx}: rank=0 (r_eps={r_eps}, r_pe={r_pe})')
                continue

            V_r = Vh[:r].t().contiguous()
            U_r = U[:, :r].contiguous()
            self._d2_delta_v.append(V_r)
            self._d2_delta_u.append(U_r)
            tag = f', p_svd={p_svd}→r_pe={r_pe}' if r_pe is not None else ''
            print(
                f'  D2 delta layer {conv_idx}: ΔW {tuple(delta_2d.shape)} '
                f'-> V_r {tuple(V_r.shape)}, U_r {tuple(U_r.shape)} '
                f'(r_eps={r_eps}{tag})'
            )

    # ----------------------------------------------------------
    def install_d1_d2_delta_hooks(self, domain_idx):
        """D3용: D1 basis + D2 delta basis 동시 투영."""
        self.clear_orth_hooks()
        if not self._orth_bases:
            raise RuntimeError('build_d1_bases()를 먼저 호출하세요.')
        if not self._d2_delta_v:
            raise RuntimeError('build_d2_delta_bases()를 먼저 호출하세요.')

        lora_set = self.domain_loras[domain_idx]
        n_a = n_b = 0
        for conv_idx, lora_module in enumerate(lora_set):

            def make_input_hook(Q):
                def project_grad(grad):
                    V = Q.to(device=grad.device, dtype=grad.dtype)
                    g = grad.view(grad.shape[0], -1)
                    return (g - (g @ V) @ V.t()).view_as(grad)
                return project_grad

            def make_output_hook(Q):
                def project_grad(grad):
                    V = Q.to(device=grad.device, dtype=grad.dtype)
                    g = grad.view(grad.shape[0], -1)
                    return (g - V @ (V.t() @ g)).view_as(grad)
                return project_grad

            a_bases = []
            if conv_idx < len(self._orth_bases) and self._orth_bases[conv_idx] is not None:
                a_bases.append(self._orth_bases[conv_idx])
            if conv_idx < len(self._d2_delta_v) and self._d2_delta_v[conv_idx] is not None:
                a_bases.append(self._d2_delta_v[conv_idx])
            if a_bases:
                q, _ = torch.linalg.qr(torch.cat(a_bases, dim=1), mode='reduced')
                self._hook_handles.append(
                    lora_module.lora_A.register_hook(make_input_hook(q.contiguous()))
                )
                n_a += 1

            b_bases = []
            if conv_idx < len(self._d1_out_bases) and self._d1_out_bases[conv_idx] is not None:
                b_bases.append(self._d1_out_bases[conv_idx])
            if conv_idx < len(self._d2_delta_u) and self._d2_delta_u[conv_idx] is not None:
                b_bases.append(self._d2_delta_u[conv_idx])
            if b_bases:
                q, _ = torch.linalg.qr(torch.cat(b_bases, dim=1), mode='reduced')
                self._hook_handles.append(
                    lora_module.lora_B.register_hook(make_output_hook(q.contiguous()))
                )
                n_b += 1

        print(
            f'  Installed {n_a} lora_A and {n_b} lora_B '
            f'D1+D2_delta orth hooks on domain_loras[{domain_idx}]'
        )

    def clear_orth_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    # ----------------------------------------------------------
    def forward(self, x, use_spec_aug=False):
        b = self.backbone
        x = b.spectrogram_extractor(x)
        x = b.logmel_extractor(x)
        x = x.transpose(1, 3); x = b.bn0[0](x); x = x.transpose(1, 3)
        if use_spec_aug and self.training:
            x = b.spec_augmenter(x)
        blocks = [
            b.conv_block1, b.conv_block2, b.conv_block3,
            b.conv_block4, b.conv_block5, b.conv_block6,
        ]
        for i, block in enumerate(blocks):
            x = self._apply_loras(x, block.conv1, i * 2)
            x = F.relu_(block.bnF[0](x))
            x = self._apply_loras(x, block.conv2, i * 2 + 1)
            x = F.relu_(block.bnS[0](x))
            x = F.avg_pool2d(x, (2, 2))
            x = F.dropout(x, 0.2, training=self.training)
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        feat = x1 + x2
        return b.fc(feat), feat


# ============================================================
# Checkpoint helpers
# ============================================================
def find_d1_checkpoint(base_dir, required=True):
    """D1 backbone 체크포인트 탐색.

    우선순위: 환경변수 D1_CKPT → base_dir/checkpoint(s)/(BN/)checkpoint_D1.pth.
    추론/평가용 체크포인트(.pth)는 backbone 가중치를 통째로 포함하므로 D1 로드가
    중복이다. required=False 이면 못 찾아도 None 을 반환해 그대로 진행한다.
    """
    env_path = os.environ.get('D1_CKPT', '').strip()
    if env_path:
        if os.path.exists(env_path):
            return env_path
        raise FileNotFoundError(f'D1_CKPT not found: {env_path}')
    candidates = [
        os.path.join(base_dir, 'checkpoint', 'BN', 'checkpoint_D1.pth'),
        os.path.join(base_dir, 'checkpoint', 'checkpoint_D1.pth'),
        os.path.join(base_dir, 'checkpoints', 'BN', 'checkpoint_D1.pth'),
        os.path.join(base_dir, 'checkpoints', 'checkpoint_D1.pth'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    if required:
        raise FileNotFoundError(
            'checkpoint_D1.pth not found. Set the D1_CKPT environment variable '
            'or place it under <model>/checkpoint/[BN/]checkpoint_D1.pth.'
        )
    return None


# Backward-compatible alias.
_find_d1_checkpoint = find_d1_checkpoint


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


_freeze = freeze


def build_model(base_dir, device, d2_lora_rank, d3_lora_rank, log=print):
    """학습용 모델 생성: D1 backbone 로드(필수) → LoRA wrapper."""
    backbone = CNN14Backbone(nb_tasks=3)
    d1_ckpt = find_d1_checkpoint(base_dir, required=True)
    log(f"D1 checkpoint: {d1_ckpt}")
    backbone.load_d1_checkpoint(d1_ckpt)
    backbone.freeze_backbone()
    backbone.to(device)
    model = DeltaOrthoMultiLoRACNN14(
        backbone,
        rank=d2_lora_rank,
        alpha=c.LORA_ALPHA,
        n_domains=2,
        domain_ranks=[d2_lora_rank, d3_lora_rank],
    ).to(device)
    return model


_build_model = build_model


def load_model(base_dir, ckpt_path, d2_rank, d3_rank, active_domain_count, device):
    """추론/평가용 모델 로드.

    체크포인트(.pth)는 backbone 가중치까지 포함하므로 D1 로드는 선택사항으로 둔다
    (없어도 곧바로 state_dict 가 전체를 덮어쓴다). dictionary 의 .pth 는 읽기 전용으로
    map_location 으로만 로드한다.
    """
    backbone = CNN14Backbone(nb_tasks=3)
    d1_ckpt = find_d1_checkpoint(base_dir, required=False)
    if d1_ckpt is not None:
        backbone.load_d1_checkpoint(d1_ckpt)
    backbone.to(device)
    model = DeltaOrthoMultiLoRACNN14(
        backbone, alpha=c.LORA_ALPHA, n_domains=2,
        domain_ranks=[d2_rank, d3_rank],
    ).to(device)
    model.active_domain_count = active_domain_count
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


_load_model = load_model
