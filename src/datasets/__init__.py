"""Dataset registry — pick a dataset backend by name.

Both :class:`~src.datasets.artbench.ArtBenchDataset` and
:class:`~src.datasets.wikiart.WikiArtDataset` share the same constructor
signature and ``get_paths`` / ``get_labels`` / `get_style_names` / `class_names`
interface, so scripts can select between them via a single ``--dataset`` flag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from src.datasets.artbench import ArtBenchDataset
from src.datasets.wikiart import WikiArtDataset

DATASETS: dict[str, type] = {
    "artbench": ArtBenchDataset,
    "wikiart": WikiArtDataset,
}

DEFAULT_DATA_DIRS: dict[str, str] = {
    "artbench": "data/artbench",
    "wikiart": "data/wikiart",
}


def get_dataset_class(name: str) -> type:
    """Resolve a ``--dataset`` name to its Dataset class."""
    try:
        return DATASETS[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown dataset {name!r}. Choose from: {', '.join(DATASETS)}"
        ) from e


def default_data_dir(name: str) -> str:
    """Default ``--data-dir`` for a given dataset name."""
    return DEFAULT_DATA_DIRS.get(name, f"data/{name}")


def build_dataset(
    name: str,
    root: str | Path,
    split: str = "train",
    transform: Optional[Callable] = None,
    return_path: bool = False,
):
    """Instantiate the dataset registered under ``name``."""
    cls = get_dataset_class(name)
    return cls(root=root, split=split, transform=transform, return_path=return_path)


__all__ = [
    "ArtBenchDataset",
    "WikiArtDataset",
    "DATASETS",
    "get_dataset_class",
    "default_data_dir",
    "build_dataset",
]
