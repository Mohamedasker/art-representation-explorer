"""
CLIP feature extractor using OpenCLIP.

Supports any model from the OpenCLIP model zoo (ViT-B/32, ViT-L/14,
LAION-2B checkpoints, etc.).  By default uses ``ViT-B/32`` trained on
``openai``'s original CLIP dataset — a very strong baseline for art images.

Usage example
-------------
>>> from src.models.clip_extractor import CLIPExtractor
>>> extractor = CLIPExtractor(model_name="ViT-B/32", pretrained="openai")
>>> bundle = extractor.extract(dataset, save_path="representations/clip_vitb32.npz")
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False

from .base_extractor import BaseExtractor


# ── Recommended model / pretrained pairs ──────────────────────────────────────
CLIP_PRESETS: dict[str, tuple[str, str]] = {
    "clip_vitb32":     ("ViT-B-32",  "openai"),
    "clip_vitl14":     ("ViT-L-14",  "openai"),
    "clip_vitb32_laion": ("ViT-B-32", "laion2b_s34b_b79k"),
    "clip_vitl14_laion": ("ViT-L-14", "laion2b_s32b_b82k"),
}


class CLIPExtractor(BaseExtractor):
    """
    Extract image features from a CLIP vision encoder via OpenCLIP.

    Parameters
    ----------
    model_name:
        OpenCLIP architecture name  (e.g. ``"ViT-B-32"``).  You can also pass
        one of the short aliases defined in :data:`CLIP_PRESETS`.
    pretrained:
        OpenCLIP pretrained dataset tag  (e.g. ``"openai"``, ``"laion2b_s34b_b79k"``).
    normalize:
        L2-normalise the output features (recommended for cosine-similarity
        comparisons and most visualisation tasks).
    device / batch_size / num_workers:
        Inherited from :class:`~src.models.base_extractor.BaseExtractor`.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        normalize: bool = True,
        device: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size, num_workers=num_workers)

        if not HAS_OPEN_CLIP:
            raise ImportError(
                "open_clip_torch is required for CLIPExtractor.  "
                "Install with:  pip install open-clip-torch"
            )

        # resolve short alias
        if model_name in CLIP_PRESETS:
            model_name, pretrained = CLIP_PRESETS[model_name]

        self._clip_model_name = model_name
        self._pretrained = pretrained
        self._normalize = normalize
        self._preprocess = None   # set in _build_model

    # ── BaseExtractor interface ───────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        arch = self._clip_model_name.replace("/", "").replace("-", "_").lower()
        tag = self._pretrained.replace("/", "_").lower()
        return f"clip_{arch}_{tag}"

    @property
    def layer_name(self) -> str:
        return "vision_encoder"

    def _build_model(self):
        model, _, self._preprocess = open_clip.create_model_and_transforms(
            self._clip_model_name,
            pretrained=self._pretrained,
            device=self.device,
        )
        return model

    def _get_transform(self):
        # Ensure model (and therefore self._preprocess) is built
        if self._preprocess is None:
            self.setup()
        return self._preprocess

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(batch)
        if self._normalize:
            feats = F.normalize(feats, dim=-1)
        return feats.float()

    # ── text embedding helpers ────────────────────────────────────────────────

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """
        Encode a list of text prompts.

        Returns
        -------
        torch.Tensor
            Float32 tensor of shape ``(len(texts), D)``.
        """
        self.setup()
        tokenizer = open_clip.get_tokenizer(self._clip_model_name)
        tokens = tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        if self._normalize:
            feats = F.normalize(feats, dim=-1)
        return feats.float()

    @torch.no_grad()
    def zero_shot_similarity(
        self,
        image_features: torch.Tensor,
        class_prompts: list[str],
    ) -> torch.Tensor:
        """
        Compute softmax similarity between image features and class prompts.

        Parameters
        ----------
        image_features:
            Tensor of shape (N, D) — normalised image embeddings.
        class_prompts:
            One prompt per class (e.g. ``["a painting in baroque style", …]``).

        Returns
        -------
        torch.Tensor
            Probability matrix of shape (N, C).
        """
        text_feats = self.encode_texts(class_prompts)   # (C, D)
        logits = image_features @ text_feats.T * 100.0  # temperature=100
        return logits.softmax(dim=-1)
