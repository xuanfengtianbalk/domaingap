"""PatchMaskLayer – MAE-style random patch masking."""

import torch
import torch.nn.functional as F

from .base import BaseConsistencyLayer


class PatchMaskLayer(BaseConsistencyLayer):
    """Randomly mask image patches (zero out pixels) — image-level MAE.

    PatchEmbed uses Conv2d(kernel=16, stride=16), so masking a 16×16 pixel
    block produces a zero-vector token, equivalent to token-level masking.

    Pipeline:
        preprocess  → denorm [0,1] (inherited)
        transform   → random patch mask
        postprocess → norm (inherited)
    """

    def __init__(self, patch_size=16, mask_ratio=0.5, p=1.0):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.p = p

    def transform(self, x):
        """x: [B, 3, H, W] float32 in [0, 1]"""
        if self.mask_ratio <= 0:
            return x

        B, C, H, W = x.shape
        nH = H // self.patch_size
        nW = W // self.patch_size

        # per-sample binary mask: True = keep, False = mask
        keep = torch.rand(B, nH, nW, device=x.device) > self.mask_ratio

        # upsample to pixel level [B, 1, H, W]
        keep_grid = F.interpolate(
            keep[:, None].float(), size=(H, W), mode='nearest'
        )

        return x * keep_grid
