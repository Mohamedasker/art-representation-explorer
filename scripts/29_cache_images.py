#!/usr/bin/env python3
"""
Cache a 256 px copy of ArtBench-10 / WikiArt as uint8 arrays for fast self-supervised training
(scripts/29_train_dino.py): the WikiArt JPEGs are ~1400x1800 px and decoding them is the data-loading bottleneck.

Every image is resized so that its SHORTER side is 256 px (bicubic, JPEG draft decoding first) and centre-cropped to
256x256; that is the image the evaluation transform (Resize 256 + CenterCrop 224) sees, so a 224 centre crop of the
cache equals the evaluation view.  Output (gitignored under data/):

    data/cache256/<dataset>_<split>_256.npy    uint8 (N, 256, 256, 3), same order as dataset.get_paths()
    data/cache256/<dataset>_<split>_meta.json  paths, labels, class names

    python scripts/29_cache_images.py                        # both datasets, both splits
    python scripts/29_cache_images.py --datasets artbench
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image

from src.datasets import build_dataset, default_data_dir

SIZE = 256


def load_one(path: str) -> np.ndarray:
    img = Image.open(path)
    img.draft("RGB", (SIZE, SIZE))                      # JPEG decoder downscales by 2/4/8 while it decodes
    img = img.convert("RGB")
    w, h = img.size
    s = SIZE / min(w, h)
    nw, nh = max(SIZE, round(w * s)), max(SIZE, round(h * s))
    img = img.resize((nw, nh), Image.BICUBIC)           # PIL's bicubic resize is antialiased
    l, t = (nw - SIZE) // 2, (nh - SIZE) // 2
    return np.asarray(img.crop((l, t, l + SIZE, t + SIZE)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["artbench", "wikiart"])
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--out-dir", default="data/cache256")
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()
    out = REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        for split in args.splits:
            base = build_dataset(name, root=default_data_dir(name), split=split)
            paths = base.get_paths()
            f_npy, f_meta = out / f"{name}_{split}_256.npy", out / f"{name}_{split}_meta.json"
            if f_npy.exists() and f_meta.exists():
                print(f"{name}/{split}: exists, skipping", flush=True)
                continue
            tmp = out / f"{name}_{split}_256.tmp.npy"
            arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(paths), SIZE, SIZE, 3))
            with Pool(args.workers) as pool:
                for i, a in enumerate(pool.imap(load_one, paths, chunksize=64)):
                    arr[i] = a
                    if i % 5000 == 0:
                        print(f"{name}/{split}: {i}/{len(paths)}", flush=True)
            arr.flush()
            del arr
            os.replace(tmp, f_npy)
            f_meta.write_text(json.dumps({"paths": paths, "labels": base.get_labels().tolist(),
                                          "class_names": list(base.class_names), "size": SIZE}))
            print(f"{name}/{split}: done {len(paths)} -> {f_npy}", flush=True)


if __name__ == "__main__":
    main()
