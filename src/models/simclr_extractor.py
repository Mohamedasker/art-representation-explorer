"""
Embeddings from SimCLR ResNet-50 encoders (``src/models/simclr.py``).

Embedding = the global-average-pooled ResNet-50 output (2048-D, the encoder output BEFORE the projection head), not
L2-normalised.  Input: Resize(256) -> CenterCrop(224) -> ImageNet mean/std (the same preprocessing as for the
ImageNet ResNet-50 baseline), except for the published ImageNet SimCLR v1 weights, see ``IMAGENET_NORM``.

* ``--model simclr_imagenet``: the published ImageNet SimCLR v1 ResNet-50 (1x), frozen.
  File slug ``simclr_imagenet``.
* ``--model simclr --checkpoint runs/<run>/simclr_final.pt``: a SimCLR trained here with
  ``scripts/13_train_simclr.py`` on ArtBench or WikiArt with a given augmentation set.
  File slug ``simclr_<train dataset>_<aug set>`` (e.g. ``simclr_artbench_color_heavy``); an ``--init imagenet_simclr``
  run gets the suffix ``_ft``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision import transforms

from .base_extractor import BaseExtractor
from .simclr import SimCLR, SimCLRConfig, load_simclr_v1_imagenet

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# The converted SimCLR v1 weights were trained on [0, 1] inputs WITHOUT mean/std normalisation (original TF recipe).
IMAGENET_NORM = None


class SimCLRExtractor(BaseExtractor):
    def __init__(self, checkpoint: str | Path | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self._info: dict = {}
        self._step = None
        self.file_slug = "simclr_imagenet"
        if self.checkpoint is not None:
            ck = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            info = ck.get("info") or ck.get("args", {})
            ds, aug = info.get("dataset"), info.get("aug_set")
            self.file_slug = f"simclr_{ds}_{aug}" + ("_ft" if info.get("init") == "imagenet_simclr" else "")
            self._info, self._step = info, ck.get("step")
            self._ck = ck

    def _build_model(self) -> torch.nn.Module:
        if self.checkpoint is None:
            return load_simclr_v1_imagenet()
        m = SimCLR(SimCLRConfig.from_dict(self._ck["model_cfg"]))
        sd = {k.replace("module.", ""): v for k, v in self._ck["model"].items()}
        m.load_state_dict(sd)
        self._ck = None
        return m.encoder

    def _get_transform(self):
        tf = [transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
              transforms.CenterCrop(224), transforms.ToTensor()]
        norm = IMAGENET_NORM if self.checkpoint is None else (IMAGENET_MEAN, IMAGENET_STD)
        if norm is not None:
            tf.append(transforms.Normalize(*norm))
        return transforms.Compose(tf)

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.contiguous(memory_format=torch.channels_last)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            f = self.model(batch)
        return f.float()

    @property
    def model_name(self) -> str:
        if self.checkpoint is None:
            return "simclr_v1_r50_1x_imagenet"
        return f"{self.file_slug}_{self._info.get('run', self.checkpoint.parent.name)}_step{self._step}"

    @property
    def layer_name(self) -> str:
        return "avgpool_2048"
