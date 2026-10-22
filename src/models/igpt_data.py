"""
Pixel tokens for the iGPT-style model (Chen et al. 2020): a painting is shrunk to ``res`` x ``res`` pixels (average pooling of the
fixed 128 px centre crop of ``src/models/ar_data.py``) and every pixel is replaced by the index of its nearest entry in a global
colour palette (k-means with 512 clusters on RGB pixels of training images of both datasets, i.e. a 9-bit palette).  One pixel =
one token; the sequence is the raster order, ``res * res`` tokens.

Files (``data/ar_tokens/``, gitignored): ``igpt<res>_palette.npy`` (512, 3) float32 in [0, 1] and the token arrays
``tokens_igpt<res>_<dataset>_<split>[_flip].npy`` (N, res*res) uint16, written by ``scripts/26_prepare_igpt.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .ar_data import CACHE_DIR, IMG_SIZE


def name_of(res: int) -> str:
    return f"igpt{res}"


def palette_file(res: int, root: str | Path = CACHE_DIR) -> Path:
    return Path(root) / f"{name_of(res)}_palette.npy"


def shrink(x: torch.Tensor, res: int) -> torch.Tensor:
    """(B, 3, 128, 128) in [0, 1] -> (B, 3, res, res) by average pooling."""
    return F.adaptive_avg_pool2d(x.float(), res) if IMG_SIZE % res else F.avg_pool2d(x.float(), IMG_SIZE // res)


def quantize(x: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """(B, 3, r, r) in [0, 1] -> palette indices (B, r*r) in raster order (nearest colour, squared L2)."""
    p = x.permute(0, 2, 3, 1).reshape(-1, 3)
    d = (p ** 2).sum(1, keepdim=True) - 2 * p @ palette.t() + (palette ** 2).sum(1)[None]
    return d.argmin(1).reshape(x.shape[0], -1)


def render(tokens: torch.Tensor, palette: torch.Tensor, res: int) -> torch.Tensor:
    """(B, r*r) palette indices -> (B, 3, r, r) colours in [0, 1]."""
    return palette[tokens.long()].reshape(-1, res, res, 3).permute(0, 3, 1, 2)
