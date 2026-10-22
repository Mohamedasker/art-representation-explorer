"""
Shared pieces of the autoregressive (VQ tokens + GPT) pipeline: the fixed 128 px preprocessing, the union style
vocabulary used for the GPT prefix tokens, and the locations of the cached pixel / token arrays in ``data/ar_tokens/``
(gitignored).

Preprocessing (identical for the cache, the tokenizer, the GPT and the embedding extractor): resize the shorter side to
128 px (bicubic, antialiased), centre-crop 128 x 128, RGB, pixel range [0, 1] (the tokenizer maps it to [-1, 1]).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torchvision import transforms

from src.datasets.artbench import STYLES as ARTBENCH_STYLES
from src.datasets.wikiart import STYLES as WIKIART_STYLES

IMG_SIZE = 128
DATASETS = ("artbench", "wikiart")
SPLITS = ("train", "test")
STYLE_NAMES: list[str] = sorted(set(ARTBENCH_STYLES) | set(WIKIART_STYLES))   # same union vocabulary as labeled_union
CACHE_DIR = Path("data/ar_tokens")


def eval_transform(size: int = IMG_SIZE):
    """PIL -> float tensor (3, size, size) in [0, 1]."""
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(size), transforms.ToTensor()])


def pixel_file(dataset: str, split: str, root: str | Path = CACHE_DIR) -> Path:
    return Path(root) / f"pixels128_{dataset}_{split}.npy"        # (N, 128, 128, 3) uint8


def meta_file(dataset: str, split: str, root: str | Path = CACHE_DIR) -> Path:
    return Path(root) / f"meta_{dataset}_{split}.npz"             # paths, style_id (into STYLE_NAMES)


def token_file(tokenizer_name: str, dataset: str, split: str, flip: bool = False, root: str | Path = CACHE_DIR) -> Path:
    return Path(root) / f"tokens_{tokenizer_name}_{dataset}_{split}{'_flip' if flip else ''}.npy"   # (N, 256) uint16


def style_ids(paths) -> np.ndarray:
    """Union style id of every image (the style is the image's parent directory name)."""
    idx = {n: i for i, n in enumerate(STYLE_NAMES)}
    return np.array([idx[Path(p).parent.name] for p in paths], dtype=np.int16)
