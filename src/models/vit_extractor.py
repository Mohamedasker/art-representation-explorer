"""
Embeddings from a plain ViT-B/16 (timm ``vit_base_patch16_224``, 224 px input), two kinds of weights:

* ``--model vit_imagenet``: the ImageNet-1k supervised weights ``augreg_in1k``, frozen (the counterpart of
  ``resnet50_imagenet``; downloaded once into the HF cache, so fetch them on a login node first).
* ``--model vit_supervised --checkpoint runs/<run>/vit_final.pt``: a ViT trained here on the style / genre / artist
  labels of ArtBench + WikiArt with ``scripts/19_train_vit.py`` (from scratch -> slug ``vit_supervised``, fine-tuned
  from the ImageNet weights -> slug ``vit_supervised_ft``).

Embedding = the final-layer CLS token after the encoder's last LayerNorm (``pool="cls"``, 768-D; the input of the
classification heads), identical for all three.  ``pool="mean"`` gives the mean of the 196 patch tokens and ``"both"``
their 1536-D concatenation (files get a ``_mean`` / ``_both`` suffix).  Vectors are not normalised.
Pre-processing is the training evaluation transform: resize to 224, centre crop, mean = std = 0.5.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision import transforms

from .base_extractor import BaseExtractor
from .supervised_vit import ARCH, IMAGENET_TAG, POOLS, VIT_MEAN, VIT_STD, MultiHeadViT, create_backbone, embed_tokens


class ViTExtractor(BaseExtractor):
    def __init__(self, checkpoint: str | Path | None = None, pool: str = "cls", **kwargs) -> None:
        super().__init__(**kwargs)
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.pool = pool
        self._ck: dict = {}
        if self.checkpoint is not None:
            self._run = self.checkpoint.resolve().parent
            if self._run.name == "checkpoints":
                self._run = self._run.parent

    def _build_model(self) -> torch.nn.Module:
        if self.checkpoint is None:
            return create_backbone(ARCH, pretrained=True, tag=IMAGENET_TAG)
        ck = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self._ck = {k: ck.get(k) for k in ("arch", "img_size", "step", "init", "imagenet_tag")}
        model = MultiHeadViT(ck["head_sizes"], ck["arch"], pretrained=False)
        model.backbone.load_state_dict(ck["backbone"])
        model.heads.load_state_dict(ck["heads"])
        return model.backbone

    def _get_transform(self):
        S = 224 if self.checkpoint is None else (self._ck or {}).get("img_size", 224)
        return transforms.Compose([
            transforms.Resize(S, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(S), transforms.ToTensor(), transforms.Normalize(VIT_MEAN, VIT_STD)])

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            z = embed_tokens(self.model, batch, self.pool)
        return z.float()

    @property
    def file_slug(self) -> str:
        """Used in representations/<dataset>_<slug>_<split>.npz: vit_imagenet | vit_supervised | vit_supervised_ft."""
        if self.checkpoint is None:
            base = "vit_imagenet"
        else:
            self.setup()
            base = "vit_supervised_ft" if self._ck.get("init") == "imagenet" else "vit_supervised"
        return base + ("" if self.pool == "cls" else f"_{self.pool}")

    @property
    def model_name(self) -> str:
        if self.checkpoint is None:
            return f"vit_b16_imagenet_{IMAGENET_TAG}"
        step = f"step{self._ck['step']}" if self._ck else "?"
        return f"vit_b16_supervised_{self._run.name}_{step}"

    @property
    def layer_name(self) -> str:
        return {"cls": "final_cls_token", "mean": "final_mean_patch_tokens", "both": "final_cls+mean_patch_tokens"}[self.pool]
