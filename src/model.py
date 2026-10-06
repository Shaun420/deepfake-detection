"""Model definitions for the deepfake detection mini-project."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, pool=True):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return self.pool(x)


class DeepfakeCNN(nn.Module):
    """Compact VGG-style CNN (~1.2M params) trained from scratch.

    The last convolutional block (`features[-1]`) is the Grad-CAM target layer.
    """

    def __init__(self, num_classes: int = 2, width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, width),           # 64 -> 32
            ConvBlock(width, width * 2),   # 32 -> 16
            ConvBlock(width * 2, width * 4),  # 16 -> 8
            ConvBlock(width * 4, width * 4, pool=False),  # 8 -> 8  (CAM layer)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(width * 4, num_classes))

    def forward(self, x):
        return self.head(self.features(x))

    @property
    def cam_layer(self) -> nn.Module:
        return self.features[-1]


class ResNet18Transfer(nn.Module):
    """Optional transfer-learning baseline (needs torchvision weights available)."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        from torchvision import models
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.net = models.resnet18(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x):
        return self.net(x)

    @property
    def cam_layer(self):
        return self.net.layer4


def build_model(name: str = "cnn", pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "cnn":
        return DeepfakeCNN()
    if name in ("resnet18", "resnet"):
        return ResNet18Transfer(pretrained=pretrained)
    raise ValueError(f"unknown model '{name}' (choose: cnn, resnet18)")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = build_model("cnn")
    x = torch.randn(2, 3, 64, 64)
    print(m(x).shape, f"{count_parameters(m):,} trainable parameters")
