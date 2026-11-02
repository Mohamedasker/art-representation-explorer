#!/usr/bin/env python3
"""
Decode every ArtBench-10 / WikiArt image once with the fixed 128 px preprocessing of the autoregressive pipeline
(``src/models/ar_data.py``) and store them as uint8 arrays under ``data/ar_tokens/`` (gitignored):

    pixels128_<dataset>_<split>.npy    (N, 128, 128, 3) uint8, in the order of the dataset's ``get_paths()``
    meta_<dataset>_<split>.npz         paths, style id (union vocabulary)

so tokenizer training (scripts/21_train_vq.py) and tokenisation (scripts/22_tokenize_ar.py) never touch the big JPEGs.

    python scripts/20_prepare_ar_data.py --workers 16
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image

from src.datasets import build_dataset, default_data_dir
from src.models.ar_data import DATASETS, SPLITS, eval_transform, meta_file, pixel_file, style_ids

_TF = eval_transform()


def load(path: str) -> np.ndarray:
    im = _TF(Image.open(path).convert("RGB"))
    return (im * 255).round().byte().permute(1, 2, 0).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-dir", default="data/ar_tokens")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        for ds in DATASETS:
            for split in SPLITS:
                out = pixel_file(ds, split, args.out_dir)
                paths = build_dataset(ds, root=default_data_dir(ds), split=split).get_paths()
                if out.exists() and np.load(out, mmap_mode="r").shape[0] == len(paths):
                    print(f"{ds}/{split}: {out} exists ({len(paths)} images)")
                    continue
                t0 = time.time()
                arr = np.lib.format.open_memmap(str(out) + ".tmp.npy", mode="w+", dtype=np.uint8,
                                                shape=(len(paths), 128, 128, 3))
                for i, a in enumerate(pool.imap(load, paths, chunksize=64)):
                    arr[i] = a
                    if i % 10000 == 0:
                        print(f"  {ds}/{split} {i}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
                arr.flush(); del arr
                os.replace(str(out) + ".tmp.npy", out)
                np.savez(meta_file(ds, split, args.out_dir), paths=np.array(paths), style_id=style_ids(paths))
                print(f"{ds}/{split}: {len(paths)} images in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
