"""
Embeddings from the supervised multi-task CNN (``scripts/10_train_cnn.py``): the globally pooled
backbone feature (2048-D for resnet50).  ``checkpoint`` is ``cnn_final.pt`` / ``checkpoints/best.pt``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision import transforms

from .base_extractor import BaseExtractor
from .supervised_cnn import MultiHeadCNN

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class SupervisedCNNExtractor(BaseExtractor):
    def __init__(self, checkpoint: str | Path, **kwargs) -> None:
        super().__init__(**kwargs)
        if checkpoint is None:
            raise ValueError("--checkpoint <run>/cnn_final.pt is required for cnn_supervised")
        self.checkpoint = Path(checkpoint)
        self._ck = None
        self._run = self.checkpoint.resolve().parent
        if self._run.name == "checkpoints":
            self._run = self._run.parent

    def _build_model(self) -> torch.nn.Module:
        ck = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self._ck = {k: ck[k] for k in ("arch", "img_size", "step")}
        model = MultiHeadCNN(ck["arch"], ck["head_sizes"], pretrained=False)
        model.backbone.load_state_dict(ck["backbone"])
        model.heads.load_state_dict(ck["heads"])
        return model.to(memory_format=torch.channels_last)

    def _get_transform(self):
        S = (self._ck or torch.load(self.checkpoint, map_location="cpu", weights_only=False))["img_size"]
        return transforms.Compose([
            transforms.Resize(S, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(S), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.contiguous(memory_format=torch.channels_last)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            f = self.model.embed(batch)
        return f.float()

    @property
    def model_name(self) -> str:
        step = f"step{self._ck['step']}" if self._ck else "?"
        arch = self._ck["arch"] if self._ck else "cnn"
        return f"supervised_{arch}_{self._run.name}_{step}"

    @property
    def layer_name(self) -> str:
        return "global_pool"
