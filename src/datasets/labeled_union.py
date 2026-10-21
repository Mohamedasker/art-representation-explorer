"""
ArtBench-10 + WikiArt with supervised targets for multi-task classification.

Heads (index -1 = no label, ignored by the loss):
  style   union of both style vocabularies (names shared by the datasets are merged)   both datasets
  genre   WikiArt genre (10 classes; 'Unknown Genre' ignored)                          WikiArt only
  artist  WikiArt artist (129 slugs; 'Unknown Artist' ignored); ArtBench images get an  both (ArtBench: where the
          artist label when the artist slug from the file name is in that vocabulary    slug is in the vocabulary)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.datasets.artbench import STYLES as ARTBENCH_STYLES
from src.datasets.union import UnionImageDataset
from src.datasets.wikiart import STYLES as WIKIART_STYLES
from src.metadata import attributes as A

STYLE_NAMES: list[str] = sorted(set(ARTBENCH_STYLES) | set(WIKIART_STYLES))
HEADS = ("style", "genre", "artist")


def _hf_labels(wikiart_dir: Path) -> dict[str, dict]:
    """{file_stem: {genre, artist}}; the arrow files are slow to scan, so cache them as json."""
    cache = wikiart_dir / "hf_labels.json"
    if cache.exists():
        return json.loads(cache.read_text())
    labels = A.wikiart_labels(wikiart_dir)
    cache.write_text(json.dumps(labels))
    return labels


def build_vocab(wikiart_dir: str | Path) -> dict[str, list[str]]:
    lab = _hf_labels(Path(wikiart_dir))
    genres = sorted({v["genre"] for v in lab.values()} - {"Unknown Genre"})
    artists = sorted({v["artist"] for v in lab.values()} - {"Unknown Artist"})
    return {"style": STYLE_NAMES, "genre": genres, "artist": artists}


class LabeledUnionDataset(UnionImageDataset):
    """``__getitem__`` -> ``(image, targets[3] = style/genre/artist, source_id)``."""

    def __init__(self, artbench_root, wikiart_root, split, transform=None, draft_size=None, vocab=None):
        super().__init__(artbench_root, wikiart_root, split, transform, draft_size)
        vocab = vocab or build_vocab(wikiart_root)
        self.vocab = vocab
        idx = {h: {n: i for i, n in enumerate(vocab[h])} for h in HEADS}
        lab = _hf_labels(Path(wikiart_root))
        targets = np.full((len(self.paths), 3), -1, dtype=np.int64)
        for i, (p, src) in enumerate(zip(self.paths, self.sources)):
            pp = Path(p)
            targets[i, 0] = idx["style"][pp.parent.name]
            if src == 1:                                            # WikiArt
                l = lab.get(pp.stem)
                if l:
                    targets[i, 1] = idx["genre"].get(l["genre"], -1)
                    targets[i, 2] = idx["artist"].get(l["artist"], -1)
            else:                                                   # ArtBench: artist slug from the file name
                a = A.artbench_artist(pp.stem)
                if a is not None:
                    targets[i, 2] = idx["artist"].get(a, -1)
        self.targets = targets

    def __getitem__(self, i: int):
        image, src = super().__getitem__(i)
        return image, self.targets[i], src
