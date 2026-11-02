"""
Embeddings from a ViT-B/16 trained here with DINO v1 (``scripts/29_train_dino.py``): the CLS token (768-D, after the final
LayerNorm) of the TEACHER backbone at Resize 256 + CenterCrop 224 with ImageNet mean/std, L2-normalised -- exactly what
``SSLExtractor`` does for the pretrained ``dino_vitb16``, so all DINO weights in the explorer are comparable.

    python scripts/02_extract_features.py --model dino_train --checkpoint runs/<run>/dino_final.pt --dataset artbench --split all
    # -> representations/<dataset>_dino_adapt_<train dataset>_*.npz (ImageNet-initialised) / dino_scratch_<train dataset>_*.npz
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

from .base_extractor import BaseExtractor
from .dino import IMAGENET_MEAN, IMAGENET_STD, build_backbone


class DINOTrainedExtractor(BaseExtractor):
    def __init__(self, checkpoint: str | Path, normalize: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        if checkpoint is None:
            raise ValueError("--model dino_train needs --checkpoint runs/<run>/dino_final.pt")
        self.checkpoint = Path(checkpoint)
        self.normalize = normalize
        self._info: dict = {}
        self._run = self.checkpoint.resolve().parent
        if self._run.name == "checkpoints":
            self._run = self._run.parent

    def _load(self) -> dict:
        if not self._info:
            ck = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            if "backbone" not in ck:                         # resumable checkpoints/last.pt holds student + teacher
                ck = {"backbone": {k[len("backbone."):]: v for k, v in ck["teacher"].items() if k.startswith("backbone.")},
                      "train_dataset": ck["args"]["dataset"], "init": ck["args"]["init"], "step": ck["step"]}
            self._info = ck
        return self._info

    def _build_model(self) -> torch.nn.Module:
        model = build_backbone(0.0)
        model.load_state_dict(self._load()["backbone"])
        return model

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            z = self.model(batch)
        z = z.float()
        return F.normalize(z, dim=-1) if self.normalize else z

    @property
    def file_slug(self) -> str:
        """representations/<dataset>_<slug>_<split>.npz, e.g. dino_adapt_artbench / dino_scratch_wikiart."""
        info = self._load()
        init = {"imagenet": "adapt", "scratch": "scratch"}.get(info.get("init"), "trained")
        return f"dino_{init}_{info.get('train_dataset', self._run.name)}"

    @property
    def model_name(self) -> str:
        info = self._load()
        return f"dino_vitb16_{self._run.name}_step{info.get('step', '?')}"

    @property
    def layer_name(self) -> str:
        return "teacher_cls_token"
