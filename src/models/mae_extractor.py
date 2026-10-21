"""
Embeddings from a Masked Autoencoder (``src/models/mae.py``), ViT-B/16, 224 px input.

Embedding = the encoder run on the full, UNMASKED image; the 196 patch tokens (after the encoder's final
LayerNorm) are average-pooled -> 768-D (``pool="mean"``, default; this is the global-pool feature used for linear
probing in the MAE paper).  ``pool="cls"`` gives the CLS token and ``"both"`` their 1536-D concatenation.
The vector is NOT L2-normalised.

Two kinds of models:

* ``--model mae_imagenet``: Meta's official ImageNet-1k MAE, frozen (no checkpoint needed; downloaded once
  into the torch hub cache, so fetch it on a login node first).
* ``--model mae --checkpoint runs/<run>/mae_final.pt``: an MAE trained here with ``scripts/12_train_mae.py``
  (continued from the ImageNet weights, or from scratch; the run's config says which).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision import transforms

from .base_extractor import BaseExtractor
from .mae import IMAGENET_MEAN, IMAGENET_STD, MAEConfig, MaskedAutoencoderViT, load_imagenet_mae


class MAEExtractor(BaseExtractor):
    def __init__(self, checkpoint: str | Path | None = None, pool: str = "mean", **kwargs) -> None:
        super().__init__(**kwargs)
        if pool not in ("mean", "cls", "both"):
            raise ValueError(f"pool must be mean|cls|both, got {pool!r}")
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.pool = pool
        self._step = None
        self._info: dict = {}
        if self.checkpoint is not None:
            self._run = self.checkpoint.resolve().parent
            if self._run.name == "checkpoints":
                self._run = self._run.parent

    def _build_model(self) -> torch.nn.Module:
        if self.checkpoint is None:
            return load_imagenet_mae()
        ck = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self._step = ck.get("step")
        self._info = ck.get("info", {})
        model = MaskedAutoencoderViT(MAEConfig.from_dict(ck["model_cfg"]))
        model.load_state_dict(ck["model"])
        return model

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            z = self.model.embed(batch, self.pool)
        return z.float()

    @property
    def file_slug(self) -> str:
        """Used in representations/<dataset>_<slug>_<split>.npz, e.g. mae_adapt_artbench / mae_scratch_wikiart."""
        if self.checkpoint is None:
            return "mae_imagenet"
        self.setup()
        init = {"imagenet": "adapt", "scratch": "scratch"}.get(self._info.get("init"), "trained")
        return f"mae_{init}_{self._info.get('train_dataset', self._run.name)}"

    @property
    def model_name(self) -> str:
        if self.checkpoint is None:
            return "mae_vitb16_imagenet"
        step = f"step{self._step}" if self._step is not None else "?"
        return f"mae_vitb16_{self._run.name}_{step}"

    @property
    def layer_name(self) -> str:
        return {"mean": "encoder_mean_patch_tokens", "cls": "encoder_cls_token",
                "both": "encoder_mean_patch_tokens+cls"}[self.pool]
