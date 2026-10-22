#!/usr/bin/env python3
"""
Prepare the pixel tokens of the iGPT-style autoregressive model (``src/models/igpt_data.py``): fit the global 512-colour palette
(k-means on RGB pixels of ``--palette-images`` random TRAIN images of both datasets, shrunk to ``--res`` px) and convert every
cached 128 px painting (scripts/20_prepare_ar_data.py) of both datasets into ``res * res`` palette indices (train also
horizontally flipped).

    python scripts/26_prepare_igpt.py --res 32
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.models.ar_data import DATASETS, SPLITS, pixel_file, token_file
from src.models.igpt_data import name_of, palette_file, quantize, render, shrink


def to_small(arr: np.ndarray, res: int, flip: bool, device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(arr)).to(device).permute(0, 3, 1, 2).float() / 255
    return shrink(x.flip(-1) if flip else x, res)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--colors", type=int, default=512)
    ap.add_argument("--palette-images", type=int, default=6000, help="train images (half per dataset) the palette is fitted on")
    ap.add_argument("--cache-dir", default="data/ar_tokens")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pf = palette_file(args.res, args.cache_dir)
    if pf.exists() and not args.overwrite:
        palette = torch.from_numpy(np.load(pf)).to(device)
        print("palette exists:", pf)
    else:
        from sklearn.cluster import MiniBatchKMeans
        rng = np.random.default_rng(0)
        px = []
        for ds in DATASETS:
            arr = np.load(pixel_file(ds, "train", args.cache_dir), mmap_mode="r")
            sel = np.sort(rng.permutation(len(arr))[: args.palette_images // 2])
            px.append(to_small(arr[sel], args.res, False, device).permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy())
        px = np.concatenate(px)
        print(f"fitting {args.colors} colours on {len(px)} pixels ...", flush=True)
        t0 = time.time()
        km = MiniBatchKMeans(args.colors, batch_size=20000, n_init=3, max_iter=100, random_state=0).fit(px)
        c = km.cluster_centers_.astype(np.float32)
        c = c[np.lexsort((c[:, 2], c[:, 1], c[:, 0]))]
        np.save(pf, c)
        palette = torch.from_numpy(c).to(device)
        err = float(((px[:200000][:, None] - c[None]) ** 2).sum(-1).min(1).mean() ** 0.5)
        print(f"palette fitted in {time.time() - t0:.0f}s, mean RGB quantisation error {err:.4f} -> {pf}", flush=True)
    for ds in DATASETS:
        for split in SPLITS:
            arr = np.load(pixel_file(ds, split, args.cache_dir), mmap_mode="r")
            for flip in ((False, True) if split == "train" else (False,)):
                out = token_file(name_of(args.res), ds, split, flip, args.cache_dir)
                if out.exists() and not args.overwrite:
                    print(out, "exists"); continue
                t0 = time.time()
                toks = np.empty((len(arr), args.res ** 2), np.uint16)
                for i in range(0, len(arr), 512):
                    toks[i: i + 512] = quantize(to_small(arr[i: i + 512], args.res, flip, device), palette).cpu().numpy()
                np.save(out, toks)
                print(f"{ds}/{split}{'/flip' if flip else ''}: {toks.shape}, {len(np.unique(toks))} colours used, "
                      f"{time.time() - t0:.0f}s -> {out}", flush=True)
    # preview: 16 real images vs their palette versions
    try:
        from torchvision.utils import make_grid, save_image
        arr = np.load(pixel_file("artbench", "test", args.cache_dir), mmap_mode="r")[::625][:16]
        x = to_small(arr, args.res, False, device)
        q = render(quantize(x, palette), palette, args.res)
        up = lambda t: torch.nn.functional.interpolate(t, scale_factor=4, mode="nearest")
        save_image(make_grid(torch.cat([up(x), up(q)]).cpu(), nrow=16, padding=2), Path(args.cache_dir) / f"{name_of(args.res)}_preview.png")
    except Exception as e:
        print("preview skipped:", e)


if __name__ == "__main__":
    main()
