"""RandConv: random convolution for texture randomization (Xu et al. ECCV 2020, Algorithm 1)."""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class RandConvLayer(nn.Module):
    """Algorithm 1 lines 1-14: RANDCONV(I, K, mix, p)

    - p:           probability of keeping the original image (only when mix=False)
    - mix=False:   with prob p return I; otherwise apply random convolution
    - mix=True:    always apply random convolution, then α-blend with original I
    - Kernel weight init: Θ ~ N(0, 1/(3·k²))
    """

    def __init__(self, kernel_sizes=(1,3,5,7), p=0.5, mix=False):
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.p = p
        self.mix = mix

    def forward(self, x):
        """x: (B, 3, H, W) – normalised image batch"""
        if not self.training:
            return x

        B = x.shape[0]
        out = []
        for i in range(B):
            p_prime = random.random()
            # line 3-4: keep original with prob p (only when not in mix mode)
            if p_prime < self.p and not self.mix:
                out.append(x[i:i+1])
                continue

            # line 6: sample kernel size
            k = random.choice(self.kernel_sizes)

            # line 7: Θ ∈ R^{3×3×k×k} ~ N(0, 1/(3k²))
            weight = torch.empty(3, 3, k, k, device=x.device)
            nn.init.normal_(weight, mean=0, std=1.0 / (3 * k * k))

            # line 8: I_rc = I ∗ Θ
            x_rc = F.conv2d(x[i:i+1], weight, padding=k // 2)

            # line 9-13: optional α-blend
            if self.mix:
                alpha = random.random()                          # line 10
                print(alpha)
                x_rc = alpha * x[i:i+1] + (1 - alpha) * x_rc    # line 11-12

            out.append(x_rc)

        return torch.cat(out, dim=0)
