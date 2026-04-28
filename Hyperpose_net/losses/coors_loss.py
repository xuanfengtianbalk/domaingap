import torch.nn as nn
import torch
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=3, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, output, target):
        """
        output: [B, 224, 224], raw logits (no sigmoid)
        target: [B, 224, 224], 0 or 1
        """
        prob = torch.sigmoid(output)
        target = target.float()

        pt = prob * target + (1 - prob) * (1 - target)
        w = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = -w * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CoorsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss