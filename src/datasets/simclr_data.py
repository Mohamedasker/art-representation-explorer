"""Datasets for SimCLR training / monitoring: paths + style labels of one dataset split, decoded with PIL's JPEG
``draft`` mode (WikiArt originals are ~1400x1800 px, far larger than the 224 px views)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.datasets import build_dataset, default_data_dir


class SplitImages(Dataset):
    """``(transform(image), style_label)`` for one split of ``artbench`` / ``wikiart``."""

    def __init__(self, dataset: str, split: str, transform: Optional[Callable] = None, root: str | Path | None = None,
                 draft_size: int | None = 288) -> None:
        base = build_dataset(dataset, root=root or default_data_dir(dataset), split=split)
        self.paths: list[str] = base.get_paths()
        self.labels: np.ndarray = base.get_labels()
        self.class_names: list[str] = list(base.class_names)
        self.transform, self.draft_size = transform, draft_size

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, idx: int) -> Image.Image:
        img = Image.open(self.paths[idx])
        if self.draft_size is not None:
            img.draft("RGB", (self.draft_size, self.draft_size))
        return img.convert("RGB")

    def __getitem__(self, idx: int):
        img = self.load(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])
