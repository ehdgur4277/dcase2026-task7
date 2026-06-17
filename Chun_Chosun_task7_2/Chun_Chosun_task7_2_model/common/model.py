"""
common/model.py — DeltaOrthoMultiLoRACNN14 모델 정의 + 공용 헬퍼.

261_D3DeltaGP-CosKD 와 동일한 구조:
  CNN14 backbone(frozen) + domain별 Conv2d LoRA(D2, D3) + shared FC
  active_domain_count: 0=D1, 1=D1+LoRA_D2, 2=D1+LoRA_D2+LoRA_D3

학습(train_utils)·추론(infer_utils)·평가(eval_utils)가 모두 여기서 import 한다.
"""
import os
import shutil
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as c
from trainer import LoRAConv2d

# ── 아키텍처 고정 설정 (exp04) ──────────────────────────────────────────────────
D2_LORA_RANK = 128
D3_LORA_RANK = 64


# ── 공용 헬퍼 ───────────────────────────────────────────────────────────────────
def energy_ratio_rank(S, p_svd, eps=1e-7):
    """특이값 S 에서 누적 에너지가 p_svd 에 도달하는 rank 반환."""
    if p_svd is None:
        return None
    energy = S.float() ** 2
    total = energy.sum()
    if total < eps:
        return 0
    ratio = torch.cumsum(energy, dim=0) / total
    m = int((ratio < p_svd).sum().item()) + 1
    return min(m, S.numel())


def freeze_module(model):
    """모든 파라미터 requires_grad=False + eval 모드."""
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def atomic_save(obj, path):
    """임시 파일에 저장 후 move — 중간에 끊겨도 부분 파일이 남지 않게."""
    dir_ = os.path.dirname(path)
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix='.tmp')
    try:
        os.close(fd)
        torch.save(obj, tmp)
        shutil.move(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ── 모델 ───────────────────────────────────────────────────────────────────────
class DeltaOrthoMultiLoRACNN14(nn.Module):
    """
    active_domain_count: 0=D1, 1=D1+LoRA_D2, 2=D1+LoRA_D2+LoRA_D3
    """

    def __init__(self, backbone, domain_ranks=None):
        super().__init__()
        self.backbone = backbone
        if domain_ranks is None:
            domain_ranks = [D2_LORA_RANK, D3_LORA_RANK]
        block_configs = [
            (1, 64), (64, 128), (128, 256),
            (256, 512), (512, 1024), (1024, 2048),
        ]
        self.domain_loras = nn.ModuleList()
        for rank in domain_ranks:
            lora_set = nn.ModuleList()
            for in_ch, out_ch in block_configs:
                lora_set.append(LoRAConv2d(in_ch, out_ch, rank, c.LORA_ALPHA))
                lora_set.append(LoRAConv2d(out_ch, out_ch, rank, c.LORA_ALPHA))
            self.domain_loras.append(lora_set)

        self.active_domain_count = 1
        self._orth_bases = []
        self._d1_out_bases = []
        self._d2_delta_v = []
        self._d2_delta_u = []
        self._hook_handles = []

    def _apply_loras(self, x, base_conv, conv_idx):
        out = base_conv(x)
        for d in range(self.active_domain_count):
            out = out + F.conv2d(x, self.domain_loras[d][conv_idx].get_delta(), padding=1)
        return out

    def set_d2_trainable(self):
        for p in self.domain_loras[0].parameters(): p.requires_grad = True
        for p in self.domain_loras[1].parameters(): p.requires_grad = False
        for p in self.backbone.fc.parameters():     p.requires_grad = False

    def set_d3_trainable(self):
        for p in self.domain_loras[0].parameters(): p.requires_grad = False
        for p in self.domain_loras[1].parameters(): p.requires_grad = True
        for p in self.backbone.fc.parameters():     p.requires_grad = False

    @torch.no_grad()
    def build_d1_bases(self, projection_rank=None, p_svd=None, eps=1e-7):
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
                self._orth_bases.append(Vh[:rank].t().contiguous())
                self._d1_out_bases.append(U[:, :rank].contiguous())

    def install_orth_hooks(self, domain_idx):
        self.clear_orth_hooks()
        lora_set = self.domain_loras[domain_idx]
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

    @torch.no_grad()
    def build_d2_delta_bases(self, p_svd=None, eps=1e-7):
        self._d2_delta_v = []
        self._d2_delta_u = []
        for lora_module in self.domain_loras[0]:
            delta_2d = lora_module.get_delta().detach().float().cpu().view(lora_module.out_ch, -1)
            if delta_2d.numel() == 0 or delta_2d.norm() < eps:
                self._d2_delta_v.append(None)
                self._d2_delta_u.append(None)
                continue
            U, S, Vh = torch.linalg.svd(delta_2d, full_matrices=False)
            r = int((S > eps).sum().item())
            r_pe = energy_ratio_rank(S, p_svd, eps)
            if r_pe is not None:
                r = min(r, r_pe)
            if r == 0:
                self._d2_delta_v.append(None)
                self._d2_delta_u.append(None)
                continue
            self._d2_delta_v.append(Vh[:r].t().contiguous())
            self._d2_delta_u.append(U[:, :r].contiguous())

    def install_d1_d2_delta_hooks(self, domain_idx):
        self.clear_orth_hooks()
        lora_set = self.domain_loras[domain_idx]
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

    def clear_orth_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

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
