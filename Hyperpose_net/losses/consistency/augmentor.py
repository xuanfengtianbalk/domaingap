"""ConsistencyAugmentor – unified multi-branch augmentation wrapper."""

import torch.nn as nn

from .rand_conv import RandConvLayer
from .aug import AugConsistencyLayer


class ConsistencyAugmentor(nn.Module):
    """Create and manage multiple consistency-augmentation branches.

    Parameters
    ----------
    branch_specs : list of dict
        Each dict specifies one branch.  Supported keys:

        - ``type`` (str) – ``'randconv'`` or ``'aug'``
        - ``p`` (float, default 0.5) – RandConv keep-original probability
        - ``mix`` (bool, default False) – RandConv α-blend mode
        - ``aug_type`` (str, default ``'augmix'``) – AugConsistencyLayer pipeline

    Examples
    --------
    >>> augmentor = ConsistencyAugmentor([
    ...     {'type': 'randconv', 'mix': True, 'p': 0.5},
    ...     {'type': 'randconv', 'mix': True, 'p': 0.5},
    ...     {'type': 'aug', 'aug_type': 'augmix'},
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
