#!/usr/bin/env python3
"""Do the SimCLR variants cope better with GRAYSCALE input?  For every model (ImageNet weights + each aug set) on one dataset:
style accuracy (linear probe on the 2048-D encoder output, 10k train -> 4k test images) for
  colour->colour   probe trained and tested on colour images
  gray->gray       probe trained and tested on grayscale versions of the same images
  colour->gray     probe trained on colour features, tested on grayscale test images (robustness, no retraining)
plus kNN (k=20) for colour->colour and gray->gray, and the cosine similarity between the colour and grayscale embedding.
    python scripts/19_grayscale_eval.py --dataset artbench    -> docs/simclr_grayscale_<dataset>.{json,md}"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.augmentations import list_aug_sets
from src.datasets.simclr_data import SplitImages
from src.models.simclr_extractor import SimCLRExtractor

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["artbench", "wikiart"], required=True)
ap.add_argument("--ntrain", type=int, default=10000)
ap.add_argument("--ntest", type=int, default=4000)
ap.add_argument("--out", default=None)
a = ap.parse_args()
ds, dev = a.dataset, ("cuda" if torch.cuda.is_available() else "cpu")

models = {}
ex = SimCLRExtractor(checkpoint=None, device=dev); ex.setup(); models["imagenet"] = (ex.model, True)
for s in list_aug_sets():
    runs = sorted((REPO / "runs").glob(f"*_simclr_{ds}_{s}/simclr_final.pt"))
    if runs:
        ex = SimCLRExtractor(checkpoint=runs[-1], device=dev); ex.setup(); models[s] = (ex.model, False)
for m, _ in models.values():
    m.eval()

pre = [T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(224), T.ToTensor()]
norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
tf = T.Compose(pre)


class DS(Dataset):
    def __init__(self, base, idx):
        self.base, self.idx = base, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        im = self.base.load(int(self.idx[i]))
        return tf(im), tf(ImageOps.grayscale(im).convert("RGB"))


def embed_all(split, n, seed):
    base = SplitImages(ds, split, None, draft_size=288)
    idx = np.sort(np.random.default_rng(seed).permutation(len(base))[:n])
    out = {(m, g): [] for m in models for g in (0, 1)}
    with torch.no_grad():
        for xc, xg in DataLoader(DS(base, idx), batch_size=128, num_workers=16):
            for g, x in enumerate((xc, xg)):
                x = x.to(dev)
                for name, (m, raw) in models.items():
                    inp = x if raw else norm(x)
                    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                        out[(name, g)].append(m(inp).float().cpu())
    return {k: torch.cat(v) for k, v in out.items()}, torch.as_tensor(base.labels[idx])


def fit_linear(Xtr, ytr):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = ((Xtr - mu) / sd).to(dev)
    y = ytr.to(dev)
    clf = torch.nn.Linear(A.shape[1], int(y.max()) + 1).to(dev)
    opt = torch.optim.LBFGS(clf.parameters(), lr=1.0, max_iter=100, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(clf(A), y) + 1e-4 * clf.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return lambda X: clf(((X - mu) / sd).to(dev)).argmax(1).cpu()


def knn(Xtr, ytr, Xte, yte, k=20, T_=0.07):
    A, B = F.normalize(Xtr, dim=1).to(dev), F.normalize(Xte, dim=1).to(dev)
    w, nb = (B @ A.T).topk(k, 1)
    votes = torch.zeros(B.shape[0], int(ytr.max()) + 1, device=dev).scatter_add_(1, ytr.to(dev)[nb], (w / T_).exp())
    return float((votes.argmax(1).cpu() == yte).float().mean())


Ftr, ytr = embed_all("train", a.ntrain, 11)
Fte, yte = embed_all("test", a.ntest, 12)
acc = lambda pred: float((pred == yte).float().mean())
res = {}
for name in models:
    pc, pg = fit_linear(Ftr[(name, 0)], ytr), fit_linear(Ftr[(name, 1)], ytr)
    c0, c1 = Fte[(name, 0)], Fte[(name, 1)]
    mu = Ftr[(name, 0)].mean(0, keepdim=True)
    res[name] = {
        "lin_colour_colour": acc(pc(c0)), "lin_gray_gray": acc(pg(c1)), "lin_colour_to_gray": acc(pc(c1)),
        "knn_colour_colour": knn(Ftr[(name, 0)], ytr, c0, yte), "knn_gray_gray": knn(Ftr[(name, 1)], ytr, c1, yte),
        "cos_colour_vs_gray": float(F.cosine_similarity(c0 - mu, c1 - mu).mean()),
    }
    res[name]["gray_drop_lin"] = res[name]["lin_colour_colour"] - res[name]["lin_gray_gray"]
    print(name, {k: round(v, 3) for k, v in res[name].items()}, flush=True)

out = Path(a.out or f"docs/simclr_grayscale_{ds}")
out.parent.mkdir(exist_ok=True, parents=True)
Path(f"{out}.json").write_text(json.dumps(res, indent=1))
cols = ["lin_colour_colour", "lin_gray_gray", "lin_colour_to_gray", "gray_drop_lin", "knn_colour_colour", "knn_gray_gray", "cos_colour_vs_gray"]
md = [f"# Grayscale input · {ds} (style accuracy, {a.ntrain} train -> {a.ntest} test)\n", "| model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
md += [f"| {n} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |" for n, r in res.items()]
Path(f"{out}.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
