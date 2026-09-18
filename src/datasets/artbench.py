"""
ArtBench-10 dataset utilities.

ArtBench-10 contains 60,000 artwork images (50k train / 10k test) across
10 art styles.  The 256×256 px version is used by default.

Official repo: https://github.com/liaopeiyuan/artbench
HuggingFace:   https://huggingface.co/datasets/huggan/artbench10
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# ── Constants ─────────────────────────────────────────────────────────────────

STYLES: list[str] = [
    "art_nouveau",
    "baroque",
    "expressionism",
    "impressionism",
    "post_impressionism",
    "realism",
    "renaissance",
    "romanticism",
    "surrealism",
    "ukiyo_e",
]

STYLE_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(STYLES)}
IDX_TO_STYLE: dict[int, str] = {i: s for s, i in STYLE_TO_IDX.items()}


# ── Dataset class ─────────────────────────────────────────────────────────────

class ArtBenchDataset(Dataset):
    """
    Minimal PyTorch Dataset for ArtBench-10.

    Expected directory layout (produced by scripts/01_download_artbench.py)::

        data/artbench/
            train/
                art_nouveau/   *.jpg
                baroque/       *.jpg
                ...
            test/
                art_nouveau/   *.jpg
                ...

    Parameters
    ----------
    root:
        Path to ``data/artbench/``.
    split:
        ``"train"`` or ``"test"``.
    transform:
        Optional torchvision / albumentations transform applied to each PIL
        image before it is returned.
    return_path:
        If ``True`` the ``__getitem__`` tuple also includes the image path as
        a third element, which is useful when saving representations.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        return_path: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.return_path = return_path

        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                "Run  python scripts/01_download_artbench.py  first."
            )

        self.samples: list[tuple[Path, int]] = []
        for style in STYLES:
            style_dir = split_dir / style
            if not style_dir.is_dir():
                continue
            label = STYLE_TO_IDX[style]
            for img_path in sorted(style_dir.glob("*.jpg")):
                self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {split_dir}. "
                "Did the download complete successfully?"
            )

    # ── dunder helpers ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.return_path:
            return image, label, str(path)
        return image, label

    # ── helpers ───────────────────────────────────────────────────────────────

    def get_paths(self) -> list[str]:
        """Return all image paths in dataset order."""
        return [str(p) for p, _ in self.samples]

    def get_labels(self) -> np.ndarray:
        """Return integer label array (length == len(dataset))."""
        return np.array([lbl for _, lbl in self.samples], dtype=np.int64)

    def get_style_names(self) -> list[str]:
        """Return style-name string for every sample."""
        return [IDX_TO_STYLE[lbl] for _, lbl in self.samples]

    @property
    def class_names(self) -> list[str]:
        return STYLES
