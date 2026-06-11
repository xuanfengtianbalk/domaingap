"""ConsistencyAugmentor – unified multi-branch augmentation wrapper."""

import torch.nn as nn

from .rand_conv import RandConvLayer
from .aug import AugConsistencyLayer
from .patch_mask import PatchMaskLayer


class ConsistencyAugmentor(nn.Module):
    """Create and manage multiple consistency-augmentation branches.

    Parameters
    ----------
    branch_specs : list of dict
        Each dict specifies one branch.  Supported keys:

        - ``type`` (str) – ``'randconv'``, ``'aug'``, or ``'patch_mask'``
        - ``p`` (float, default 0.5) – RandConv keep-original probability
        - ``mix`` (bool, default False) – RandConv α-blend mode
        - ``aug_type`` (str, default ``'augmix'``) – AugConsistencyLayer pipeline
        - ``mask_ratio`` (float, default 0.5) – patch mask ratio

    Examples
    --------
    >>> augmentor = ConsistencyAugmentor([
    ...     {'type': 'randconv', 'mix': True, 'p': 0.5},
    ...     {'type': 'aug', 'aug_type': 'augmix'},
    ...     {'type': 'patch_mask', 'mask_ratio': 0.5},
    ... ]).to(device)
    >>> branches = augmentor(inputs)  # list of 3 tensors
    """

    def __init__(self, branch_specs):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, spec in enumerate(branch_specs):
            t = spec['type']
            if t == 'randconv':
                self.layers.append(RandConvLayer(
                    p=spec.get('p', 0.5),
                    mix=spec.get('mix', False),
                ))
            elif t == 'aug':
                self.layers.append(AugConsistencyLayer(
                    aug_type=spec.get('aug_type', 'augmix'),
                ))
            elif t == 'patch_mask':
                self.layers.append(PatchMaskLayer(
                    mask_ratio=spec.get('mask_ratio', 0.5),
                    p=spec.get('p', 1.0),
                ))
            else:
                raise ValueError(f"Branch {i}: unknown type '{t}'")

    def forward(self, inputs):
        """Apply every branch to a single input batch."""
        return [layer(inputs) for layer in self.layers]

    def __getitem__(self, idx):
        """Delegate indexing to internal layer list."""
        return self.layers[idx]

    def __len__(self):
        return len(self.layers)
