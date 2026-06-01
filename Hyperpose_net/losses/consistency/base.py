"""Base consistency augmentation layer.

Pipeline: preprocess → transform → postprocess → ImageNet-normalized tensor.
"""

import torch
import torch.nn as nn


class BaseConsistencyLayer(nn.Module):
    """Abstract base for consistency augmentation layers.

    Input  : ImageNet-normalised tensor [B, 3, H, W]
    Output : ImageNet-normalised tensor [B, 3, H, W] (augmented)

    Subclasses implement ``transform`` and may override ``preprocess`` and
    ``postprocess`` to control the denorm / re-norm stages.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

    # ------------------------------------------------------------------
    # Hooks – override in subclasses when needed.
    # ------------------------------------------------------------------

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Default: de-normalise ImageNet tensor to raw [0, 1]."""
        return (x * self.std + self.mean).clamp(0, 1)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Core augmentation.  Input and output shapes / ranges are defined
        by the subclass."""
        raise NotImplementedError

    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Default: normalise a [0, 1] tensor back to ImageNet space."""
        return (x - self.mean) / self.std

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        return self.postprocess(self.transform(self.preprocess(x)))
