"""
WikiArt dataset utilities.

WikiArt (via the ``huggan/wikiart`` HuggingFace mirror) contains ~81k artwork
images scraped from wikiart.org, labelled with 27 art styles, 129 artists,
and 11 genres.  Only the style label is used here, to match the
:class:`~src.datasets.artbench.ArtBenchDataset` interface.

The upstream dataset ships as a single ``train`` split with no held-out
test set, so ``scripts/01_download_wikiart.py`` carves out a deterministic
per-style train/test split at download time and writes it to disk using the
same ``<root>/<split>/<style>/*.jpg`` layout as ArtBench.

HuggingFace: https://huggingface.co/datasets/huggan/wikiart
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# ── Constants ─────────────────────────────────────────────────────────────────
# Canonical 27-style WikiArt taxonomy (Karayev et al., 2014 / Tan et al., 2016),
# lower-cased and snake_cased.  Only styles actually present on disk are used,
# but keeping a fixed canonical ordering makes label indices stable across
# downloads / mirrors.
STYLES: list[str] = [
    "abstract_expressionism",
    "action_painting",
    "analytical_cubism",
    "art_nouveau",
    "baroque",
    "color_field_painting",
    "contemporary_realism",
    "cubism",
    "early_renaissance",
    "expressionism",
    "fauvism",
    "high_renaissance",
    "impressionism",
    "mannerism_late_renaissance",
    "minimalism",
    "naive_art_primitivism",
    "new_realism",
    "northern_renaissance",
    "pointillism",
    "pop_art",
    "post_impressionism",
    "realism",
    "rococo",
    "romanticism",
    "symbolism",
    "synthetic_cubism",
    "ukiyo_e",
]

STYLE_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(STYLES)}
IDX_TO_STYLE: dict[int, str] = {i: s for s, i in STYLE_TO_IDX.items()}


def normalize_style_name(raw: str) -> str:
    """Map a raw HuggingFace style label (e.g. ``"Art_Nouveau_Modern"``) to a
    lower_snake_case slug (e.g. ``"art_nouveau"``)."""
    slug = "".join(c.lower() if c.isalnum() else "_" for c in raw)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


# ── Dataset class ─────────────────────────────────────────────────────────────

class WikiArtDataset(Dataset):
    """
    Minimal PyTorch Dataset for WikiArt, mirroring
    :class:`~src.datasets.artbench.ArtBenchDataset`.

    Expected directory layout (produced by scripts/01_download_wikiart.py)::

        data/wikiart/
            train/
                impressionism/   *.jpg
                baroque/         *.jpg
                ...
            test/
                impressionism/   *.jpg
                ...

    Parameters
    ----------
    root:
        Path to ``data/wikiart/``.
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
                "Run  python scripts/01_download_wikiart.py  first."
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
