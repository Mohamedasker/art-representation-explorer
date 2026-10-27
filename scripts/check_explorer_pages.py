#!/usr/bin/env python3
"""Print which kinds of explorer pages are still missing: per_split (10k sample + another 10k), full, all
(merged train + test).  Empty output = everything is built.  Used by scripts/slurm/orchestrate_explorer.sh."""
from pathlib import Path
import numpy as np
import sys

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from src.visualization.variants import page_path
out = root / "visualizations/explorer"
N = 10000
missing = set()
for f in sorted((root / "representations").glob("*.npz")):
    if "color_profiles" in f.name:
        continue
    n = int(np.load(f, allow_pickle=False)["labels"].shape[0])
    stem = f.stem
    is_all = stem.endswith("_all")
    is_train = stem.endswith("_train")
    pages = [(page_path(stem), "all" if is_all else "per_split")]
    if n - N >= N // 2:
        pages.append((page_path(stem, "set1"), "all" if is_all else "per_split"))
    if is_train and n > N:
        pages.append((page_path(stem, "all"), "full"))
    for page, kind in pages:
        if not (out / page).exists():
            missing.add(kind)
        elif page == page_path(stem) and not (out / "cache" / stem / "landing.json").exists():
            missing.add(kind)
print(" ".join(sorted(missing)))
