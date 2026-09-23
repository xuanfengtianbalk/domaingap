"""Parameter-Efficient Fine-Tuning (PEFT) for DINOv3 ViT backbone.

Methods: SSF (Scale & Shift), LoRA (Low-Rank on attention QKV), VPT (Visual Prompt).
SSF + LoRA can be combined.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================================
# SSF
# ==========================================================================

class SSFBlock(nn.Module):
    """output = scale * block(x) + shift"""

    def __init__(self, block: nn.Module, dim: int):
        super().__init__()
        self.block = block
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, *args, **kwargs):
        return self.scale * self.block(*args, **kwargs) + self.shift


def apply_ssf(backbone: nn.Module):
    dim = backbone.embed_dim
    for i in range(len(backbone.blocks)):
        backbone.blocks[i] = SSFBlock(backbone.blocks[i], dim)
    backbone.requires_grad_(False)
    for n, p in backbone.named_parameters():
        if 'scale' in n or 'shift' in n:
            p.requires_grad_(True)


# ==========================================================================
# LoRA
# ==========================================================================

class LoRAMergedLinear(nn.Module):
    """Faithful port of microsoft/LoRA ``MergedLinear`` (loralib/layers.py)
    for fused attention qkv projections.

    y = W x + bias + (lora_dropout(x) @ merge_AB()^T) * scaling
    where scaling = lora_alpha / rank and only the chunks with
    enable_lora=True receive a low-rank delta.

    Mirrors the official code:
      - A: kaiming_uniform_(a=sqrt(5)), B: zeros (delta is zero at init)
      - W frozen, bias left trainable
      - lora_dropout applied to x (default 0)

    use_dora: DoRA (ICML 2024) weight decomposition per HF PEFT
    (``peft.tuners.lora``, ``use_dora=True``): each enabled chunk's effective
    weight is W'_c = m_c * (W_c + s*BA_c) / ||W_c + s*BA_c||_c (column-wise
    norm, clamp 1e-6), where m_c is a trainable magnitude vector initialized
    to the column norms of the pretrained W_c (identity at init since
    BA_c=0). The k chunk stays frozen pretrained weights.
    """

    def __init__(self, linear: nn.Linear, rank: int = 4, lora_alpha=None,
                 lora_dropout: float = 0.0, enable_lora=(True, False, True),
                 use_dora: bool = False):
        super().__init__()
        self.W = linear.weight
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.lora_alpha = rank if lora_alpha is None else lora_alpha
        self.scaling = self.lora_alpha / self.rank
        self.enable_lora = enable_lora
        self.use_dora = use_dora
        self.sum_en = sum(enable_lora)
        self.out_chunk = self.out_features // len(enable_lora)
        assert self.out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0. else (lambda x: x)

        self.lora_A = nn.Parameter(
            self.W.new_zeros((rank * self.sum_en, self.in_features)))
        self.lora_B = nn.Parameter(
            self.W.new_zeros((self.out_chunk * self.sum_en, rank)))
        self.lora_ind = self.W.new_zeros(
            (self.out_features,), dtype=torch.bool).view(len(enable_lora), -1)
        for i, enabled in enumerate(enable_lora):
            self.lora_ind[i, :] = enabled
        self.lora_ind = self.lora_ind.view(-1)

        if self.use_dora:
            # magnitude per enabled chunk: [sum_en, 1, in], init = column
            # norms of the pretrained chunk (HF PEFT: weight.norm(p=2, dim=0))
            W3 = self.W.detach().view(len(self.enable_lora), self.out_chunk, self.in_features)
            mag_init = W3[self.lora_ind.view(len(self.enable_lora), -1)[:, 0]] \
                .norm(p=2, dim=1, keepdim=True)  # [sum_en, 1, in]
            self.lora_magnitude = nn.Parameter(mag_init)

        # freeze the pre-trained weight matrix (official keeps bias trainable)
        self.W.requires_grad_(False)

        # official reset_parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        # equivalent of the official conv1d(groups=sum(enable_lora)): each
        # enabled chunk gets its own low-rank pair B_c @ A_c
        A3 = self.lora_A.view(self.sum_en, self.rank, self.in_features)
        B3 = self.lora_B.view(self.sum_en, self.out_chunk, self.rank)
        delta_w = torch.bmm(B3, A3).view(self.sum_en * self.out_chunk, self.in_features)
        return self.zero_pad(delta_w)

    def effective_weight(self):
        """Full effective weight [out, in] (with DoRA chunk normalization)."""
        W_eff = self.W + self.merge_AB() * self.scaling
        if self.use_dora and self.rank > 0:
            W3 = W_eff.view(len(self.enable_lora), self.out_chunk, self.in_features)
            chunk_ids = self.lora_ind.view(len(self.enable_lora), -1)[:, 0].nonzero().flatten()
            for j, cid in enumerate(chunk_ids):
                Wc = W3[cid]
                norm = Wc.norm(p=2, dim=0, keepdim=True).clamp(min=1e-6)
                W3[cid] = Wc / norm * self.lora_magnitude[j]
            W_eff = W3.view(self.out_features, self.in_features)
        return W_eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank > 0 and self.use_dora:
            # DoRA path: normalized full weight (HF PEFT applies the LoRA
            # branch through the weight decomposition)
            return F.linear(x, self.effective_weight(), self.bias)
        result = F.linear(x, self.W, self.bias)
        if self.rank > 0:
            result += self.lora_dropout(x) @ self.merge_AB().t() * self.scaling
        return result


def apply_lora(backbone: nn.Module, rank: int = 4, lora_alpha=None,
               lora_dropout: float = 0.0, use_dora: bool = False):
    for block in backbone.blocks:
        inner = block.block if isinstance(block, SSFBlock) else block
        inner.attn.qkv = LoRAMergedLinear(inner.attn.qkv, rank=rank,
                                          lora_alpha=lora_alpha,
                                          lora_dropout=lora_dropout,
                                          use_dora=use_dora)
    backbone.requires_grad_(False)
    for n, p in backbone.named_parameters():
        if ('scale' in n or 'shift' in n
                or n.endswith('lora_A') or n.endswith('lora_B')
                or n.endswith('lora_magnitude')
                or n.endswith('attn.qkv.bias')):
            p.requires_grad_(True)


# ==========================================================================
# VPT
# ==========================================================================

class VPTWrapper(nn.Module):
    """Insert learnable prompt tokens before patch tokens."""

    def __init__(self, backbone: nn.Module, n_prompts: int = 10):
        super().__init__()
        self.backbone = backbone
        dim = backbone.embed_dim
        self.prompts = nn.Parameter(torch.randn(1, n_prompts, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        prompts = self.prompts.expand(B, -1, -1)
        x = torch.cat([prompts, x], dim=1)
        for blk in self.backbone.blocks:
            x = blk(x)
        return x


# ==========================================================================
# Unified
# ==========================================================================

def apply_peft(backbone: nn.Module, peft_config: dict):
    method = peft_config.get('method', 'none')
    if method == 'none':
        return
    if 'ssf' in method:
        apply_ssf(backbone)
    if 'lora' in method:
        apply_lora(backbone, rank=peft_config.get('lora_rank', 4),
                   lora_alpha=peft_config.get('lora_alpha'),
                   lora_dropout=peft_config.get('lora_dropout', 0.0),
                   use_dora=bool(peft_config.get('use_dora', False)))
    if 'vpt' in method:
        n = peft_config.get('vpt_n_prompts', 10)
        backbone.vpt = VPTWrapper(backbone, n_prompts=n)
