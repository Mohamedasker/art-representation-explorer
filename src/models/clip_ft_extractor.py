"""
Embeddings from the image tower of a CLIP fine-tuned on one art dataset (``scripts/28_train_clip_ft.py``).

``--model clip_ft --checkpoint runs/<run>/clip_ft_final.pt [--clip-alpha a]``: the image tower is the interpolation
``a`` * fine-tuned + (1 - a) * original weights (WiSE-FT; a = 1: the fine-tuned model, a = 0 would be the original CLIP).
Embedding = L2-normalised CLIP image feature (the joint-embedding-space output), exactly like the frozen
``clip_vitb32`` extractor, with the same pre-processing, so the representations are directly comparable.

File slug (``representations/<dataset>_<slug>_<split>.npz``): ``clip_ft_<training dataset>`` and, for a < 1,
``clip_ft_<training dataset>_wise<100 a>``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .clip_extractor import CLIPExtractor
from .clip_ft import blend_visual, load_ft_visual


class CLIPFTExtractor(CLIPExtractor):
    def __init__(self, checkpoint: str | Path, alpha: float = 1.0, **kwargs) -> None:
        self.checkpoint = Path(checkpoint)
        self.alpha = float(alpha)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._ck = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        super().__init__(model_name=self._ck["arch"], pretrained=self._ck["pretrained"], **kwargs)

    def _build_model(self):
        model = super()._build_model()
        load_ft_visual(model, self._ck, self.alpha)
        return model.eval()

    @property
    def file_slug(self) -> str:
        slug = f"clip_ft_{self._ck['dataset']}"
        return slug if self.alpha >= 1.0 else f"{slug}_wise{round(self.alpha * 100):d}"

    @property
    def model_name(self) -> str:
        return f"{super().model_name}_ft_{self._ck['run']}_step{self._ck['step']}_alpha{self.alpha:g}"

    @property
    def layer_name(self) -> str:
        return "vision_encoder_ft"
