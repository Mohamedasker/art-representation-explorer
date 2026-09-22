"""
Self-supervised learning feature extractor.

Supports:
- **DINOv2** (Meta AI, 2023) — loaded via ``torch.hub``.
- **DINO** (Meta AI, 2021) — loaded via ``torch.hub``.
- Any **timm** model with pretrained SSL weights (e.g. ``"vit_base_patch16_224.dino"``)
  by delegating to :class:`~src.models.cnn_extractor.CNNExtractor`.

DINOv2 features are strong general-purpose image descriptors that work
remarkably well on artistic images without any fine-tuning.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .base_extractor import BaseExtractor


# ── Available SSL models ──────────────────────────────────────────────────────


# DINOv2's `main` branch has since added `float | None` (PEP 604) type hints
# in dinov2/layers/attention.py, which crashes on Python < 3.10
# (see https://github.com/facebookresearch/dinov2/pull/539, still unmerged).
# Pin to the last commit before that break instead of tracking `main`.
_DINOV2_REF = "81b2b6419385a321287de91e00282ef7cbd26f94"

SSL_MODELS: dict[str, dict] = {
    # DINOv2 (ViT-S/14, ViT-B/14, ViT-L/14, ViT-G/14)
    "dinov2_vits14": {"hub": f"facebookresearch/dinov2:{_DINOV2_REF}", "fn": "dinov2_vits14", "dim": 384},
    "dinov2_vitb14": {"hub": f"facebookresearch/dinov2:{_DINOV2_REF}", "fn": "dinov2_vitb14", "dim": 768},
    "dinov2_vitl14": {"hub": f"facebookresearch/dinov2:{_DINOV2_REF}", "fn": "dinov2_vitl14", "dim": 1024},
    "dinov2_vitg14": {"hub": f"facebookresearch/dinov2:{_DINOV2_REF}", "fn": "dinov2_vitg14", "dim": 1536},
    # DINO (ViT-S/16, ViT-B/16 — from 2021 paper)
    "dino_vits16":   {"hub": "facebookresearch/dino:main", "fn": "dino_vits16", "dim": 384},
    "dino_vitb16":   {"hub": "facebookresearch/dino:main", "fn": "dino_vitb16", "dim": 768},
    "dino_resnet50": {"hub": "facebookresearch/dino:main", "fn": "dino_resnet50", "dim": 2048},
}


def _local_dinov2_clone(ref: str) -> Path:
    """
    Ensure a local git clone of facebookresearch/dinov2 pinned at ``ref``
    exists under the torch.hub cache dir, and return its path.

    torch.hub's github loader refuses an arbitrary commit SHA that isn't the
    tip of a branch or tag (``_validate_not_a_forked_repo``), even though the
    commit is a legitimate part of the repo's history. Cloning it ourselves
    and loading with ``source="local"`` sidesteps that check.
    """
    cache_dir = Path(torch.hub.get_dir())
    dest = cache_dir / f"facebookresearch_dinov2_{ref}"
    if dest.is_dir():
        return dest

    tmp_dest = dest.with_name(dest.name + ".tmp")
    if tmp_dest.exists():
        shutil.rmtree(tmp_dest)
    tmp_dest.mkdir(parents=True)

    try:
        subprocess.run(["git", "init", "-q"], cwd=tmp_dest, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/facebookresearch/dinov2.git"],
            cwd=tmp_dest, check=True,
        )
        subprocess.run(
            ["git", "fetch", "-q", "--depth", "1", "origin", ref],
            cwd=tmp_dest, check=True,
        )
        subprocess.run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=tmp_dest, check=True)
    except Exception:
        shutil.rmtree(tmp_dest, ignore_errors=True)
        raise

    tmp_dest.rename(dest)
    return dest


class SSLExtractor(BaseExtractor):
    """
    Extract features from a self-supervised model loaded via ``torch.hub``.

    Parameters
    ----------
    model_id:
        One of the keys in :data:`SSL_MODELS`, e.g. ``"dinov2_vitb14"``.
    normalize:
        L2-normalise the output features.
    device / batch_size / num_workers:
        Inherited from :class:`~src.models.base_extractor.BaseExtractor`.
    """

    def __init__(
        self,
        model_id: str = "dinov2_vitb14",
        normalize: bool = True,
        device: Optional[str] = None,
        batch_size: int = 32,   # DINOv2 models are large; smaller default
        num_workers: int = 4,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size, num_workers=num_workers)

        if model_id not in SSL_MODELS:
            raise ValueError(
                f"Unknown SSL model: {model_id!r}.  "
                f"Available: {list(SSL_MODELS.keys())}"
            )
        self._model_id = model_id
        self._cfg = SSL_MODELS[model_id]
        self._normalize = normalize

    # ── BaseExtractor interface ───────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def layer_name(self) -> str:
        return "cls_token"

    def _build_model(self):
        repo, _, ref = self._cfg["hub"].partition(":")
        if repo == "facebookresearch/dinov2" and ref == _DINOV2_REF:
            repo_dir = _local_dinov2_clone(ref)
            model = torch.hub.load(
                str(repo_dir),
                self._cfg["fn"],
                source="local",
                pretrained=True,
                verbose=False,
            )
        else:
            model = torch.hub.load(
                self._cfg["hub"],
                self._cfg["fn"],
                pretrained=True,
                verbose=False,
            )
        return model

    def _get_transform(self):
        """Standard ImageNet-normalised 224×224 crop used by DINO/DINOv2."""
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.model(batch)
        # DINOv2 returns a plain tensor; older DINO may return dict in some modes
        if isinstance(feats, dict):
            feats = feats.get("x_norm_clstoken", feats.get("out", feats))
        if self._normalize:
            feats = F.normalize(feats.float(), dim=-1)
        return feats.float()
