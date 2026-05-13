import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Lightweight CNN for 32×32 RGB images (CIFAR-10 proxy for aerial classification).
    Architecture: 3 conv-pool blocks → 2 FC layers.

    GroupNorm is used instead of BatchNorm so that per-sample gradient computation
    via torch.func.vmap works correctly (BatchNorm cross-sample statistics break vmap).
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),   # → 32×32×32
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # → 16×16×32
            nn.Conv2d(32, 64, 3, padding=1),             # → 16×16×64
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # →  8× 8×64
            nn.Conv2d(64, 128, 3, padding=1),            # →  8× 8×128
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # →  4× 4×128
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x.flatten(1))
