"""
CNN feature extractor using ResNet-50 (and other timm models).

Two usage modes
---------------
1. **ImageNet-pretrained** — the default; works out of the box with no
   additional data.
2. **ArtBench fine-tuned** — if a checkpoint path is supplied the weights are
   loaded on top of the ImageNet backbone.  A community-fine-tuned checkpoint
   for ArtBench-10 can be downloaded from HuggingFace; see the README for
   links.

The layer from which features are extracted is configurable (default:
``"global_pool"`` — the 2048-D pooled representation before the classifier).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from .base_extractor import BaseExtractor


# ── HuggingFace hub IDs for ArtBench fine-tuned checkpoints ──────────────────
# Community checkpoints trained on ArtBench-10.  The table below maps a short
# alias to a HuggingFace model repo.  Add your own if you fine-tune a model.
ARTBENCH_CHECKPOINTS: dict[str, str] = {
    # Key → HuggingFace repo ID  (will be downloaded via timm.create_model)
    # e.g. "resnet50_artbench": "username/resnet50-artbench10",
}


class CNNExtractor(BaseExtractor):
    """
    Extract penultimate-layer features from a pretrained CNN via ``timm``.

    Parameters
    ----------
    model_id:
        A ``timm`` model name (e.g. ``"resnet50"``), *or* one of the short
        aliases in :data:`ARTBENCH_CHECKPOINTS`.
    pretrained:
        Load ImageNet pretrained weights (ignored when ``checkpoint`` is given).
    checkpoint:
        Path to a local ``.pt`` / ``.pth`` file with fine-tuned weights.
        The file may be a plain ``state_dict``, or a dict with a
        ``"state_dict"`` key (standard Lightning / torchvision checkpoint).
    extract_layer:
        Which feature map to return.  Supported values:
        - ``"global_pool"``  (default) — after global average pool, (N, D).
        - ``"layer4"``       — spatial map before pooling, flattened.
        - ``"pre_logits"``   — alias for global_pool in most timm models.
    num_classes:
        Number of output classes for the classification head.  Only relevant
        when a fine-tuned checkpoint is loaded.  Pass ``0`` to remove the head
        and use the backbone directly.
    device / batch_size / num_workers:
        Inherited from :class:`~src.models.base_extractor.BaseExtractor`.
    """

    def __init__(
        self,
        model_id: str = "resnet50",
        pretrained: bool = True,
        checkpoint: Optional[str | Path] = None,
        extract_layer: str = "global_pool",
        num_classes: int = 1000,
        device: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size, num_workers=num_workers)

        if not HAS_TIMM:
            raise ImportError(
                "timm is required for CNNExtractor.  "
                "Install with:  pip install timm"
            )

        # resolve HuggingFace alias
        if model_id in ARTBENCH_CHECKPOINTS:
            self._hf_id = ARTBENCH_CHECKPOINTS[model_id]
            self._model_id = model_id
            self._pretrained_hf = True
        else:
            self._hf_id = None
            self._model_id = model_id
            self._pretrained_hf = False

        self._pretrained = pretrained
        self._checkpoint = Path(checkpoint) if checkpoint else None
        self._extract_layer = extract_layer
        self._num_classes = num_classes
        self._timm_model = None   # built lazily in _build_model

    # ── BaseExtractor interface ───────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        suffix = ""
        if self._checkpoint:
            suffix = f"_finetuned({self._checkpoint.stem})"
        elif self._pretrained_hf:
            suffix = "_artbench"
        else:
            suffix = "_imagenet" if self._pretrained else "_random"
        return f"{self._model_id}{suffix}"

    @property
    def layer_name(self) -> str:
        return self._extract_layer

    def _build_model(self) -> nn.Module:
        if self._pretrained_hf and self._hf_id:
            # load from HuggingFace hub
            model = timm.create_model(
                f"hf_hub:{self._hf_id}",
                pretrained=True,
                num_classes=0,   # remove head → returns pool features
            )
        else:
            model = timm.create_model(
                self._model_id,
                pretrained=self._pretrained,
                num_classes=0 if self._extract_layer == "global_pool" else self._num_classes,
            )

        if self._checkpoint is not None:
            self._load_checkpoint(model, self._checkpoint)

        # For non-global-pool layers we attach a hook below in _forward
        self._timm_model = model
        self._hook_output: Optional[torch.Tensor] = None
        if self._extract_layer not in ("global_pool", "pre_logits"):
            self._register_hook(model)

        return model

    def _load_checkpoint(self, model: nn.Module, ckpt_path: Path) -> None:
        print(f"  Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # strip Lightning-style prefix if present
        state = {k.replace("model.", "", 1): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"    Missing keys  ({len(missing)}): {missing[:5]} …")
        if unexpected:
            print(f"    Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")

    def _register_hook(self, model: nn.Module) -> None:
        """Attach a forward hook to capture intermediate feature maps."""
        target = dict(model.named_modules()).get(self._extract_layer)
        if target is None:
            available = [n for n, _ in model.named_modules() if n]
            raise ValueError(
                f"Layer '{self._extract_layer}' not found in {self._model_id}.\n"
                f"Available modules: {available}"
            )

        def _hook(module, inp, output):
            # flatten spatial dimensions if needed
            if output.dim() > 2:
                self._hook_output = output.flatten(1)
            else:
                self._hook_output = output

        target.register_forward_hook(_hook)

    def _get_transform(self):
        """Return timm's recommended preprocessing transform for this model."""
        # Ensure model is built before we query its config
        if self._timm_model is None:
            self.setup()
        config = resolve_data_config({}, model=self._timm_model)
        return create_transform(**config)

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        if self._extract_layer in ("global_pool", "pre_logits"):
            # num_classes=0 → model returns pool features directly
            return self.model(batch)
        else:
            self._hook_output = None
            self.model(batch)
            assert self._hook_output is not None, "Hook did not fire."
            return self._hook_output


# ── Convenience factory ───────────────────────────────────────────────────────

def make_resnet50(
    pretrained: bool = True,
    checkpoint: Optional[str | Path] = None,
    **kwargs,
) -> CNNExtractor:
    """
    Shortcut that returns a ResNet-50 extractor.

    If *checkpoint* points to a fine-tuned ArtBench-10 model the extractor
    will use those weights; otherwise ImageNet pretrained weights are used.
    """
    return CNNExtractor(
        model_id="resnet50",
        pretrained=pretrained,
        checkpoint=checkpoint,
        **kwargs,
    )
