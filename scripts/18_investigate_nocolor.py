#!/usr/bin/env python3
"""Why does the always-grayscale SimCLR (`no_color`) predict colour statistics BETTER than the others?

Hypothesis: it never saw chromatic input (R=G=B), so the first-layer weights along the chroma directions stay at their random
initialisation; at test time colour images excite those untrained directions and colour leaks into the embedding as a random
projection.  Tests (ArtBench, 6000 train / 2000 test images, ridge R2 of embedding -> colour statistics):
  1. colour R2 with COLOUR input vs the same images converted to GRAYSCALE (grayscale input should remove the leak)
  2. a randomly initialised, untrained ResNet-50 as reference (does an untrained net already encode colour this well?)
  3. conv1 weight energy in the chroma subspace (filter minus its channel mean) relative to the luma part, per model
    python scripts/18_investigate_nocolor.py --out docs/simclr_nocolor_investigation
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.datasets.simclr_data import SplitImages
from src.models.simclr_extractor import SimCLRExtractor

spec = importlib.util.spec_from_file_location("cmp", REPO / "scripts" / "17_compare_simclr.py")
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="artbench")
ap.add_argument("--ntrain", type=int, default=6000)
ap.add_argument("--ntest", type=int, default=2000)
ap.add_argument("--out", default="docs/simclr_nocolor_investigation")
ap.add_argument("--device", default=None)
a = ap.parse_args()
dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
ds = a.dataset

tr = SplitImages(ds, "train", None, draft_size=288)
te = SplitImages(ds, "test", None, draft_size=288)
rng = np.random.default_rng(0)
itr = np.sort(rng.permutation(len(tr))[:a.ntrain])
ite = np.sort(rng.permutation(len(te))[:a.ntest])
col_tr, _ = cmp.load_targets(ds, np.array([tr.paths[i] for i in itr]))
col_te, _ = cmp.load_targets(ds, np.array([te.paths[i] for i in ite]))
tf_eval = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


tf_raw = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(224), T.ToTensor()])


class DS(Dataset):
    def __init__(self, base, idx, gray, raw=False):
        self.base, self.idx, self.gray, self.tf = base, idx, gray, (tf_raw if raw else tf_eval)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        im = self.base.load(int(self.idx[i]))
        if self.gray:
            im = ImageOps.grayscale(im).convert("RGB")
        return self.tf(im)   # the published ImageNet SimCLR expects raw [0,1] pixels (raw=True)


def embed(model, base, idx, gray, raw=False):
    out = []
    with torch.no_grad():
        for x in DataLoader(DS(base, idx, gray, raw), batch_size=128, num_workers=16):
            out.append(model(x.to(dev)).float().cpu())
    return torch.cat(out).numpy()


models = {}
runs = lambda s: sorted((REPO / "runs").glob(f"*_simclr_{ds}_{s}/simclr_final.pt"))[-1]
for s in ["baseline", "color_heavy", "no_color", "crop_only"]:
    ex = SimCLRExtractor(checkpoint=runs(s), device=dev)
    ex.setup()
    models[s] = ex.model
ex = SimCLRExtractor(checkpoint=None, device=dev)
ex.setup()
models["imagenet"] = ex.model
torch.manual_seed(0)
rnd = torchvision.models.resnet50(weights=None)
rnd.fc = torch.nn.Identity()
models["random_init"] = rnd.to(dev).eval()

res = {}
for name, m in models.items():
    m.eval()
    w = m.conv1.weight.detach().cpu()                       # (64, 3, 7, 7)
    luma = w.mean(1, keepdim=True)
    chroma = w - luma
    r = {"conv1_chroma_over_luma_energy": float((chroma ** 2).sum() / (luma ** 2).sum() / 2.0)}   # /2: 2 of 3 dims are chroma
    for mode in ("colour", "gray"):
        gray = mode == "gray"
        raw = name == "imagenet"
        Ftr, Fte = embed(m, tr, itr, gray, raw), embed(m, te, ite, gray, raw)
        r2 = cmp.ridge_r2(Ftr, col_tr, Fte, col_te)
        r[f"{mode}_input_R2"] = float(r2.mean())
        r[f"{mode}_input_R2_sat"] = float(r2[0])
        r[f"{mode}_input_R2_bright"] = float(r2[1])
        r[f"{mode}_input_R2_hue"] = float(r2[2:].mean())
    res[name] = r
    print(name, {k: round(v, 3) for k, v in r.items()}, flush=True)
    m.cpu()

Path(a.out).parent.mkdir(exist_ok=True, parents=True)
Path(f"{a.out}_{ds}.json").write_text(json.dumps(res, indent=1))
cols = ["colour_input_R2", "gray_input_R2", "colour_input_R2_hue", "gray_input_R2_hue", "conv1_chroma_over_luma_energy"]
md = [f"# no_color investigation · {ds}\n", "| model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
for n, r in res.items():
    md.append(f"| {n} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
Path(f"{a.out}_{ds}.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
