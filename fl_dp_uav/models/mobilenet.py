"""
MobileNetV2 backbone (Sandler et al., 2018) adapted for 32×32 CIFAR-style
input, as the mobile-optimised alternative to SimpleCNN recommended in the
report (Section 8.5) for real UAV companion-computer deployment.

Two changes from the ImageNet-standard architecture:
  - Stem and first inverted-residual stage use stride 1 instead of 2 (32×32
    input is already small; the ImageNet stem is tuned for 224×224 and would
    downsample this away almost immediately).
  - BatchNorm2d is replaced with GroupNorm throughout, for the same reason
    SimpleCNN uses it (models/cnn.py): BatchNorm's cross-sample statistics
    are incompatible with the per-sample gradient computation DP-SGD needs
    (torch.func.vmap, federated_learning/client.py).
"""

import torch
import torch.nn as nn

# (expansion factor t, output channels c, num blocks n, stride s of first block)
_INVERTED_RESIDUAL_CFG = [
    (1,  16, 1, 1),
    (6,  24, 2, 1),   # stride 2 -> 1: keep spatial resolution on 32x32 input
    (6,  32, 3, 2),
    (6,  64, 4, 2),
    (6,  96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
]


def _group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Largest group count <= max_groups that divides `channels` evenly."""
    for g in (max_groups, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


def _conv_gn_relu6(in_ch: int, out_ch: int, kernel_size: int, stride: int, groups: int = 1) -> nn.Sequential:
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
        _group_norm(out_ch),
        nn.ReLU6(),
    )


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int):
        super().__init__()
        assert stride in (1, 2)
        hidden_ch = in_ch * expand_ratio
        self.use_residual = stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            layers.append(_conv_gn_relu6(in_ch, hidden_ch, kernel_size=1, stride=1))
        layers += [
            # depthwise
            _conv_gn_relu6(hidden_ch, hidden_ch, kernel_size=3, stride=stride, groups=hidden_ch),
            # pointwise-linear (no activation)
            nn.Conv2d(hidden_ch, out_ch, 1, 1, 0, bias=False),
            _group_norm(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return x + out if self.use_residual else out


class MobileNetV2CIFAR(nn.Module):
    """MobileNetV2 for 32x32 input. ~2.3M parameters at width_mult=1.0."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3, width_mult: float = 1.0):
        super().__init__()

        def scaled(c: int) -> int:
            return max(8, int(round(c * width_mult / 8)) * 8)

        input_ch = scaled(32)
        last_ch = scaled(1280) if width_mult > 1.0 else 1280

        self.stem = _conv_gn_relu6(in_channels, input_ch, kernel_size=3, stride=1)

        blocks = []
        in_ch = input_ch
        for t, c, n, s in _INVERTED_RESIDUAL_CFG:
            out_ch = scaled(c)
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(InvertedResidual(in_ch, out_ch, stride, expand_ratio=t))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.head = _conv_gn_relu6(in_ch, last_ch, kernel_size=1, stride=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
