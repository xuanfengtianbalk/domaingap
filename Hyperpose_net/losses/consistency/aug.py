"""AugConsistencyLayer – albumentations augmentation for consistency training.

Defines augmentation groups that mirror ``space_aug._augmix()`` semantics but are
self-contained: changes here do not affect the regular training pipeline.
"""

import numpy as np
import torch
import albumentations as A

from .base import BaseConsistencyLayer

# ---------------------------------------------------------------------------
# Augmentation pools  (parameters match space_aug.py, keeping independent
# definitions so that each pipeline can evolve separately.)
# ---------------------------------------------------------------------------

_BRIGHTNESS = [
    A.RandomBrightnessContrast(contrast_limit=0.3, brightness_limit=0.3, p=1),
    A.InvertImg(p=1),
    A.MultiplicativeNoise((0.9, 1.1), p=1),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=1),
    A.HueSaturationValue(20, 30, 20, p=1),
]

_BLUR = [
    A.GaussianBlur((3, 7), p=1),
    A.MotionBlur((3, 7), p=1),
    A.Sharpen(p=1),
    A.Emboss(p=1),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1),
]

_CORRUPT = [
    A.GaussNoise((0.01, 0.05), p=1),
    A.ISONoise(p=1),
    A.RandomFog(0.2, p=1),
    A.RandomSnow(0.2, p=1),
    A.RandomSunFlare((0, 0, 1, 0.5), src_radius=200, p=1),
]

_GENERAL = [
    A.CoarseDropout(num_holes_range=(1, 8), p=1),
    A.Superpixels(p_replace=0.1, n_segments=100, p=1),
    A.PixelDropout(0.02, p=1),
]

# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def _make_mix_pipeline(brightness, blur, corrupt, general,
                       n_sample=2):
    """Build a four-stage mix augmentation pipeline.

    Each stage randomly selects *n_sample* transforms from its pool and
    applies them in order.  All four stages are always executed.

    Parameters
    ----------
    brightness, blur, corrupt, general : list of albumentations transforms
    n_sample : int
        Number of transforms to sample from each pool (default 2).
    """
    stages = []
    for pool in [brightness, blur, corrupt, general]:
        if pool:
            stages.append(A.Compose([A.SomeOf(pool, n=n_sample, p=1)], p=1))
    return A.Compose(stages)


# ---------------------------------------------------------------------------
# AugConsistencyLayer
# ---------------------------------------------------------------------------

class AugConsistencyLayer(BaseConsistencyLayer):
    """Apply albumentations-based augmentation for consistency training.

    ``aug_type`` selects a predefined augmentation pipeline.  The following
    values are supported:

    - ``'augmix'``      — full mix of all categories (4 groups)
    - ``'augmix_light'`` — brightness + blur only
    - ``'none'``         — identity (pass-through)

    Parameters
    ----------
    aug_type : str
        Which pipeline to apply.
    """

    _PIPELINES = {
        'augmix': _make_mix_pipeline(_BRIGHTNESS, _BLUR, _CORRUPT, _GENERAL, n_sample=2),
    }

    def __init__(self, aug_type: str = 'augmix'):
        super().__init__()
        if aug_type not in self._PIPELINES:
            raise ValueError(f"Unknown aug_type '{aug_type}'. "
                             f"Available: {list(self._PIPELINES.keys())}")
        self.aug_type = aug_type
        self.pipeline = self._PIPELINES[aug_type]

    # ------------------------------------------------------------------
    # Hooks  (preprocess / postprocess use the base-class defaults)
    # ------------------------------------------------------------------

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply albumentations pipeline.

        x : float32 tensor [B, 3, H, W] in [0, 1]
        Returns : float32 tensor [B, 3, H, W] in [0, 1]
        """
        arrs = (x.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        out = []
        for img_np in arrs:
            result = self.pipeline(image=img_np)['image']
            out.append(np.transpose(result, (2, 0, 1)))
        return torch.from_numpy(np.stack(out)).float().to(x.device) / 255.0
