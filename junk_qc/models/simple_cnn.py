import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Small CNN for junk-vs-rest classification on 4-channel 75x75 tiles.

    Args:
        in_ch: Number of input channels (default: 4).
        num_classes: Number of output classes (default: 2).

    Forward:
        x: (N, in_ch, H, W) -> logits of shape (N, num_classes).
    """

    def __init__(self, in_ch: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),  # 75 -> 37
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),  # 37 -> 18
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),  # 18 -> 9
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.feat(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
