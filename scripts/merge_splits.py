#!/usr/bin/env python3
"""
Merge the per-split embedding files into one file per model and dataset:

    representations/<dataset>_<model>_train.npz + ..._test.npz  ->  <dataset>_<model>_all.npz

No model is run; the per-split files stay untouched.  Every bundle carries a per-image ``split`` array
(train / test), so the explorer can colour by split and the two stay distinguishable.

    python scripts/merge_splits.py            # all models that have both splits
    python scripts/merge_splits.py --force    # overwrite existing *_all.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.base_extractor import RepresentationBundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repr-dir", default="representations")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    rd = REPO_ROOT / args.repr_dir
    for train in sorted(rd.glob("*_train.npz")):
        if "color_profiles" in train.name:
            continue
        test = train.with_name(train.name.replace("_train.npz", "_test.npz"))
        out = train.with_name(train.name.replace("_train.npz", "_all.npz"))
        if not test.exists():
            print(f"skip {train.name}: no test file")
            continue
        if out.exists() and not args.force and out.stat().st_mtime > max(train.stat().st_mtime, test.stat().st_mtime):
            print(f"up to date: {out.name}")
            continue
        a, b = RepresentationBundle.load(train), RepresentationBundle.load(test)
        if a.model_name != b.model_name or a.dim != b.dim:
            print(f"skip {train.name}: train/test made by different models ({a.model_name} vs {b.model_name})")
            continue
        merged = RepresentationBundle(
            vectors=np.concatenate([a.vectors, b.vectors]), labels=np.concatenate([a.labels, b.labels]),
            label_names=np.concatenate([a.label_names, b.label_names]), paths=np.concatenate([a.paths, b.paths]),
            model_name=a.model_name, layer_name=a.layer_name,
            metadata={**a.metadata, "n_samples": a.n_samples + b.n_samples,
                      "splits": {"train": a.n_samples, "test": b.n_samples}, "merged_from": [train.name, test.name]})
        merged.save(out)
        sp = merged.splits
        print(f"wrote {out.name}: {merged.n_samples} images (train {int((sp == 'train').sum())}, test {int((sp == 'test').sum())})")


if __name__ == "__main__":
    main()
