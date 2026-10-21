"""
ResNet backbone (timm) with one linear head per supervised target (style / genre / artist).

The embedding used for the representation comparison is the globally pooled backbone feature
(2048-D for resnet50), i.e. the input of the heads.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class MultiHeadCNN(nn.Module):
    def __init__(self, arch: str, head_sizes: dict[str, int], pretrained: bool = False, drop_rate: float = 0.1):
        super().__init__()
        self.arch = arch
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        self.feat_dim = self.backbone.num_features
        self.heads = nn.ModuleDict({k: nn.Linear(self.feat_dim, n) for k, n in head_sizes.items()})

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.backbone(x)
        return {k: h(f) for k, h in self.heads.items()}
