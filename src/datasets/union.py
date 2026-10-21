"""
Union of ArtBench-10 and WikiArt as a label-free image dataset.

Built on top of the existing :class:`ArtBenchDataset` / :class:`WikiArtDataset`
loaders (only their ``get_paths()`` is used) so the on-disk layout stays the
single source of truth.  Images are decoded with PIL's JPEG ``draft`` mode,
which makes the (very large, ~1400x1800) WikiArt originals several times
cheaper to load when the target resolution is only 256 px.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset

from src.datasets.artbench import ArtBenchDataset
from src.datasets.wikiart import WikiArtDataset

SOURCES: list[str] = ["artbench", "wikiart"]


class UnionImageDataset(Dataset):
    """Images from ArtBench-10 and/or WikiArt for one split.

    ``__getitem__`` returns ``(image, source_id)`` where ``source_id`` indexes
    :data:`SOURCES`.
    """

    def __init__(
        self,
        artbench_root: str | Path,
        wikiart_root: str | Path,
        split: str,
        transform: Optional[Callable] = None,
        draft_size: int | None = None,
    ) -> None:
        self.transform = transform
        self.draft_size = draft_size
        self.paths: list[str] = []
        self.sources: list[int] = []
        for sid, (cls, root) in enumerate(
            [(ArtBenchDataset, artbench_root), (WikiArtDataset, wikiart_root)]
        ):
            paths = cls(root, split=split).get_paths()
            self.paths += paths
            self.sources += [sid] * len(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx])
        if self.draft_size is not None:
            image.draft("RGB", (self.draft_size, self.draft_size))
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.sources[idx]
