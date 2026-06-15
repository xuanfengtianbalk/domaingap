"""Parameter-Efficient Fine-Tuning (PEFT) for DINOv3 ViT backbone.

Methods: SSF (Scale & Shift), LoRA (Low-Rank on attention QKV), VPT (Visual Prompt).
SSF + LoRA can be combined.
"""
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

class LoRALinear(nn.Module):
    """Wx + BAx. W frozen, A and B trainable."""

    def __init__(self, linear: nn.Linear, rank: int = 4):
        super().__init__()
        self.W = linear.weight
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.A = nn.Parameter(torch.randn(rank, self.in_features) * (0.1 / self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.W, self.bias)
        y += F.linear(F.linear(x, self.A), self.B)
        return y


def apply_lora(backbone: nn.Module, rank: int = 4):
    for block in backbone.blocks:
        inner = block.block if isinstance(block, SSFBlock) else block
        inner.attn.qkv = LoRALinear(inner.attn.qkv, rank=rank)
    backbone.requires_grad_(False)
    for n, p in backbone.named_parameters():
        if 'scale' in n or 'shift' in n or n.endswith('.A') or n.endswith('.B'):
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
        apply_lora(backbone, rank=peft_config.get('lora_rank', 4))
    if 'vpt' in method:
        n = peft_config.get('vpt_n_prompts', 10)
        backbone.vpt = VPTWrapper(backbone, n_prompts=n)
