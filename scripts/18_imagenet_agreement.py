#!/usr/bin/env python3
"""
Do the embedding spaces organise paintings by ImageNet class?  For every representation (merged train + test of a
dataset) this measures, on the paintings that CLIP assigns confidently to an ImageNet group (scripts/17_label_imagenet.py):

  * kNN accuracy  : leave-one-out 10-nearest-neighbour majority vote (cosine) predicting the image's ImageNet group,
  * NMI           : agreement of a k-means clustering (k = number of groups) with the ImageNet groups,

next to the same two numbers for the ART STYLE of the same images, and the chance level of the kNN vote (share of the
largest class).  Caveat: the ImageNet groups come from CLIP ViT-L/14, so CLIP embeddings are not an independent judge.

    python scripts/18_imagenet_agreement.py            # prints a table and writes docs/imagenet_agreement.md
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "32")      # the cluster's OpenBLAS is built for at most 128 threads

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from src.metadata import imagenet_labels as L
from src.metadata.nudity import image_key
from src.visualization.variants import family_order, member_of, FAMILIES


def knn_accuracy(X: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    n_cls = int(y.max()) + 1
    counts = (y[nn][:, :, None] == np.arange(n_cls)[None, None, :]).sum(1)          # (n, n_cls) votes per class
    return float((counts.argmax(1) == y).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10000, help="images sampled per dataset")
    ap.add_argument("--out", default="docs/imagenet_agreement.md")
    ap.add_argument("--no-zscore", action="store_true", help="do not standardise every feature dimension (some models, e.g. MAE, have a few dominant dimensions)")
    args = ap.parse_args()
    lines = ["# Do the embedding spaces organise paintings by ImageNet class?", "",
             "Confident paintings only (CLIP puts >= %.2f of its ImageNet probability on one WordNet group). kNN = leave-one-out "
             "10-NN majority vote (cosine, features z-scored per dimension); NMI = k-means (k = 17 groups) vs the labels. Groups come from CLIP ViT-L/14, so the "
             "CLIP row is not an independent judge." % L.MIN_GROUP_PROB, ""]
    for ds in ("artbench", "wikiart"):
        z = np.load(L.labels_file(ds), allow_pickle=False)
        gp = z["group_prob"].astype(np.float32)
        all_keys = [str(k) for k in z["keys"]]            # (NpzFile re-reads an array on every access: load once)
        conf = np.flatnonzero(gp.max(1) >= L.MIN_GROUP_PROB)
        sel = np.sort(np.random.default_rng(0).choice(conf, size=min(args.n, len(conf)), replace=False))
        grp = gp[sel].argmax(1)
        remap = {g: i for i, g in enumerate(np.unique(grp))}
        grp = np.array([remap[g] for g in grp])
        chance_g = np.bincount(grp).max() / len(grp)
        rows = []
        for f in sorted((REPO / "representations").glob(f"{ds}_*_all.npz"), key=lambda q: (family_order(q.stem), q.stem)):
            if member_of(f.stem) is None or "baseline" not in f.stem and "simclr_artbench_" in f.stem:
                pass
            t0 = time.perf_counter()
            b = np.load(f, allow_pickle=False)
            bpaths = b["paths"]
            row = {image_key(str(p)): i for i, p in enumerate(bpaths)}
            idx = np.array([row[all_keys[i]] for i in sel])
            X = b["vectors"][idx].astype(np.float32)
            X -= X.mean(0)                                            # centre: some models have a large mean offset
            if not args.no_zscore:
                X /= X.std(0) + 1e-6                                  # per-dimension z-score: a few huge dimensions must not decide the distances
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
            style = b["labels"][idx]
            _, style = np.unique(style, return_inverse=True)
            kg, ks = knn_accuracy(X, grp), knn_accuracy(X, style)
            km = KMeans(len(np.unique(grp)), n_init=3, random_state=0).fit_predict(X)
            rows.append((f.stem.removeprefix(f"{ds}_").removesuffix("_all"), kg, normalized_mutual_info_score(grp, km),
                         ks, normalized_mutual_info_score(style, km)))
            print(f"[{time.perf_counter() - t0:.0f}s] {ds:8s} {rows[-1][0]:34s} ImageNet-group kNN {kg:.3f} NMI {rows[-1][2]:.3f} | style kNN {ks:.3f} NMI {rows[-1][4]:.3f}", flush=True)
        lines += [f"## {ds} ({len(sel):,} images; chance kNN for ImageNet groups {chance_g:.3f}, for styles "
                  f"{np.bincount(style).max() / len(style):.3f})", "",
                  "| representation | ImageNet group kNN | ImageNet group NMI | art style kNN | art style NMI |", "|---|---|---|---|---|"]
        lines += [f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} |" for r in rows] + [""]
    out = REPO / args.out
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
