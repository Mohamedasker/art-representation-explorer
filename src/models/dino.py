"""
DINO v1 (Caron et al. 2021, "Emerging Properties in Self-Supervised Vision Transformers"), self-distillation with no
labels: ViT-B/16 student and EMA teacher, 3-layer MLP projection head with a weight-normalised last layer, multi-crop,
centring + sharpening.  Follows facebookresearch/dino (main_dino.py, vision_transformer.py) but uses timm's ViT, whose
parameter names equal those of the official backbone, so the published ImageNet weights load unchanged.

Used by scripts/29_train_dino.py; the trained teacher backbone is embedded by src/models/dino_train_extractor.py.
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EMBED_DIM = 768
FULL_CKPT = "pretrained/dino_vitb16_full/dino_vitbase16_pretrain_full_checkpoint.pth"
FULL_CKPT_URL = "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_full_checkpoint.pth"


def build_backbone(drop_path: float = 0.0) -> nn.Module:
    """ViT-B/16, CLS-token output after the final LayerNorm (768-D); accepts any 16-multiple resolution
    (position embeddings are interpolated, as for the 96 px local crops)."""
    from timm.models.vision_transformer import VisionTransformer
    return VisionTransformer(img_size=224, patch_size=16, embed_dim=EMBED_DIM, depth=12, num_heads=12, mlp_ratio=4.0,
                             qkv_bias=True, num_classes=0, drop_path_rate=drop_path, dynamic_img_size=True,
                             norm_layer=partial(nn.LayerNorm, eps=1e-6))


class DINOHead(nn.Module):
    """MLP(in -> hidden -> hidden -> bottleneck), L2 normalisation, weight-normalised linear layer to out_dim prototypes."""

    def __init__(self, in_dim: int, out_dim: int, norm_last_layer: bool = True, hidden_dim: int = 2048,
                 bottleneck_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, bottleneck_dim))
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        # weight_norm by hand (w = g * v / ||v||): nn.utils.weight_norm is deprecated and its state-dict keys differ
        self.last_v = nn.Parameter(torch.empty(out_dim, bottleneck_dim))
        nn.init.kaiming_uniform_(self.last_v, a=math.sqrt(5))             # nn.Linear default init
        self.last_g = nn.Parameter(torch.ones(out_dim, 1), requires_grad=not norm_last_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(self.mlp(x), dim=-1, p=2)
        w = self.last_v * (self.last_g / self.last_v.norm(dim=1, keepdim=True))
        return F.linear(x, w.to(x.dtype))


class MultiCropWrapper(nn.Module):
    """backbone + head over a list of crops; crops of equal resolution are batched together."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, crops: list[torch.Tensor]) -> torch.Tensor:
        sizes = [c.shape[-1] for c in crops]
        outs, start = [], 0
        for end in [i + 1 for i in range(len(sizes)) if i + 1 == len(sizes) or sizes[i + 1] != sizes[i]]:
            outs.append(self.backbone(torch.cat(crops[start:end])))
            start = end
        return self.head(torch.cat(outs))


def build_model(out_dim: int, norm_last_layer: bool, drop_path: float = 0.0) -> MultiCropWrapper:
    return MultiCropWrapper(build_backbone(drop_path), DINOHead(EMBED_DIM, out_dim, norm_last_layer))


def _strip(sd: dict, prefixes=("module.",)) -> dict:
    out = {}
    for k, v in sd.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def load_official_full_checkpoint(student: MultiCropWrapper, teacher: MultiCropWrapper, path: str | Path) -> None:
    """Load Meta's DINO ViT-B/16 ImageNet-1k checkpoint (backbone + projection head of student and teacher)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for model, key in ((student, "student"), (teacher, "teacher")):
        sd = _strip(ck[key])
        w = sd.pop("head.last_layer.weight")                      # materialised weight -> v = w, g = ||w||
        sd["head.last_v"] = w
        sd["head.last_g"] = w.norm(dim=1, keepdim=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"official DINO checkpoint ({key}) does not match: missing={missing} unexpected={unexpected}")


class DINOLoss(nn.Module):
    """Cross-entropy between sharpened + centred teacher outputs of the 2 global crops and student outputs of all
    other crops; the centre is an EMA of the teacher outputs (prevents one dimension from dominating)."""

    def __init__(self, out_dim: int, n_crops: int, student_temp: float = 0.1, center_momentum: float = 0.9) -> None:
        super().__init__()
        self.n_crops, self.student_temp, self.center_momentum = n_crops, student_temp, center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out: torch.Tensor, teacher_out: torch.Tensor, teacher_temp: float, stats: bool = False):
        s = (student_out.float() / self.student_temp).chunk(self.n_crops)
        t_logits = teacher_out.float()
        t = F.softmax((t_logits - self.center) / teacher_temp, dim=-1).detach().chunk(2)
        total, n_terms = 0.0, 0
        for iq, q in enumerate(t):
            for v in range(len(s)):
                if v == iq:
                    continue
                total = total + torch.sum(-q * F.log_softmax(s[v], dim=-1), dim=-1).mean()
                n_terms += 1
        total = total / n_terms
        info = None
        if stats:
            p = torch.cat(t)
            ent = -(p * (p + 1e-12).log()).sum(-1).mean()                      # mean per-sample teacher entropy
            marg = p.mean(0)
            if dist.is_initialized():
                dist.all_reduce(marg)
                marg = marg / dist.get_world_size()
            marg_ent = -(marg * (marg + 1e-12).log()).sum()                     # entropy of the batch-mean teacher distribution
            k = p.shape[-1]
            info = {"teacher_entropy": ent.item(), "teacher_marginal_entropy": marg_ent.item(),
                    "kl_to_uniform": math.log(k) - marg_ent.item(), "max_entropy": math.log(k)}
        self._update_center(t_logits)
        return total, info

    @torch.no_grad()
    def _update_center(self, teacher_out: torch.Tensor) -> None:
        batch_center = teacher_out.sum(0, keepdim=True)
        n = len(teacher_out)
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            n *= dist.get_world_size()
        self.center.mul_(self.center_momentum).add_(batch_center / n, alpha=1 - self.center_momentum)


def cosine_schedule(start: float, end: float, steps: int, warmup_steps: int = 0, warmup_start: float = 0.0):
    """Per-step array: linear warm-up warmup_start -> start, then cosine start -> end."""
    import numpy as np
    warm = np.linspace(warmup_start, start, warmup_steps) if warmup_steps > 0 else np.array([])
    it = np.arange(steps - warmup_steps)
    cos = end + 0.5 * (start - end) * (1 + np.cos(np.pi * it / max(1, len(it))))
    return np.concatenate((warm, cos))
