"""
ViT-B/16 backbone (timm) with one linear head per supervised target (style / genre / artist) -- the transformer
counterpart of ``supervised_cnn.MultiHeadCNN``.

The heads read the final-layer CLS token (after the encoder's last LayerNorm, 768-D); that CLS token is also the
embedding used for the representation comparison.  ``embed(x, pool="mean")`` gives the mean of the 196 patch tokens
instead and ``"both"`` their 1536-D concatenation.

``pretrained=True`` loads the ImageNet-1k weights ``vit_base_patch16_224.<IMAGENET_TAG>`` (timm); these expect
mean = std = 0.5 input normalisation (``VIT_MEAN`` / ``VIT_STD``), which is also used when training from scratch.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

ARCH = "vit_base_patch16_224"
IMAGENET_TAG = "augreg_in1k"          # ImageNet-1k supervised only (no ImageNet-21k), like resnet50.a1_in1k
VIT_MEAN, VIT_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
POOLS = ("cls", "mean", "both")


def create_backbone(arch: str = ARCH, pretrained: bool = False, tag: str = IMAGENET_TAG, drop_rate: float = 0.0,
                    drop_path_rate: float = 0.0) -> nn.Module:
    name = f"{arch}.{tag}" if pretrained else arch
    return timm.create_model(name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate,
                             drop_path_rate=drop_path_rate)


def embed_tokens(backbone: nn.Module, x: torch.Tensor, pool: str = "cls") -> torch.Tensor:
    """(B, 768) CLS token, (B, 768) mean patch token or (B, 1536) both, after the final LayerNorm."""
    tok = backbone.forward_features(x)                       # (B, 1 + 196, D), normalised
    cls, mean = tok[:, 0], tok[:, backbone.num_prefix_tokens:].mean(1)
    return {"cls": cls, "mean": mean, "both": torch.cat([cls, mean], 1)}[pool]


class MultiHeadViT(nn.Module):
    def __init__(self, head_sizes: dict[str, int], arch: str = ARCH, pretrained: bool = False, tag: str = IMAGENET_TAG,
                 drop_rate: float = 0.1, drop_path_rate: float = 0.1):
        super().__init__()
        self.arch = arch
        self.backbone = create_backbone(arch, pretrained, tag, drop_rate, drop_path_rate)
        self.feat_dim = self.backbone.num_features
        self.heads = nn.ModuleDict({k: nn.Linear(self.feat_dim, n) for k, n in head_sizes.items()})

    def embed(self, x: torch.Tensor, pool: str = "cls") -> torch.Tensor:
        return embed_tokens(self.backbone, x, pool)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.backbone(x)                                 # CLS token (global_pool="token")
        return {k: h(f) for k, h in self.heads.items()}


def layer_scales(backbone: nn.Module, decay: float) -> dict[str, float]:
    """Layer-wise LR decay factors per backbone parameter name (embeddings smallest, last block / norm = 1)."""
    n = len(backbone.blocks)
    out = {}
    for name, _ in backbone.named_parameters():
        if name.startswith("blocks."):
            lid = int(name.split(".")[1]) + 1
        elif name.startswith(("cls_token", "pos_embed", "patch_embed", "reg_token")):
            lid = 0
        else:
            lid = n + 1
        out[name] = decay ** (n + 1 - lid)
    return out
