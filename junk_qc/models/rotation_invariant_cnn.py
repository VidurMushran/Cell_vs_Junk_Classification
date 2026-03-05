import torch
import torch.nn as nn


class RotationInvariantCNN(nn.Module):
    """
    Rotation-invariant CNN built by averaging predictions over 0°, 90°, 180°, and 270°
    rotations of the input tile.

    Uses a SimpleCNN-style backbone internally.

    Args:
        in_ch: Number of input channels (default: 4).
        num_classes: Number of output classes (default: 2).
    """

    def __init__(self, in_ch: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _forward_single(self, x):
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.fc(h)

    def forward(self, x):
        # x: (N, C, H, W)
        logits_list = []
        for k in range(4):
            xr = torch.rot90(x, k=k, dims=(2, 3))
            logits_list.append(self._forward_single(xr))
        logits = torch.stack(logits_list, dim=0).mean(dim=0)
        return logits
