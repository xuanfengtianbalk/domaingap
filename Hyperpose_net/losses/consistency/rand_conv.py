"""RandConvLayer – random convolution for texture randomization (Xu et al. ECCV 2020)."""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseConsistencyLayer


class RandConvLayer(BaseConsistencyLayer):
    """Per-sample random convolution with optional α-blend.

    Pipeline:
        preprocess  → denorm (inherited from base)
        transform   → random conv + optional mix
        postprocess → min-max rescale → ImageNet norm
    """

    def __init__(self, kernel_sizes=(1*5, 3*5, 5*5, 7), p=0.5, mix=False):
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.p = p
        self.mix = mix

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def transform(self, x):
        """Apply per-sample random convolution + optional α-blend."""
        B = x.shape[0]
        out = []
        for i in range(B):
            p_prime = random.random()
            if p_prime < self.p and not self.mix:
                out.append(x[i:i + 1])
                continue

            k = random.choice(self.kernel_sizes)
            weight = torch.empty(3, 3, k, k, device=x.device)
            nn.init.normal_(weight, mean=0, std=1.0 / (3 * k * k))

            x_rc = F.conv2d(x[i:i + 1], weight, padding=k // 2)

            if self.mix:
                alpha = random.random()
                x_rc = alpha * x[i:i + 1] + (1 - alpha) * x_rc

            out.append(x_rc)
        return torch.cat(out, dim=0)

    def postprocess(self, x):
        # return x
        """Min-max rescale to [0, 1] then ImageNet normalise."""
        r = x.max() - x.min()
        if r > 0:
            x = (x - x.min()) / r
        return (x - self.mean) / self.std
