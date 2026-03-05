import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Focal loss for binary classification (2-logit softmax setup).

    Args:
        alpha: Weighting factor for the positive class (class 1).
        gamma: Focusing parameter that down-weights easy examples.
        reduction: "mean", "sum", or "none".

    Inputs:
        logits: Tensor of shape (N, 2).
        target: LongTensor of shape (N,) with values in {0,1}.

    Returns:
        Scalar loss if reduction != "none", else tensor of shape (N,).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.softmax(logits, dim=1).gather(1, target.view(-1, 1)).squeeze(1)
        alpha_t = torch.where(target == 1, self.alpha, 1.0 - self.alpha)
        loss = alpha_t * (1.0 - pt).pow(self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
