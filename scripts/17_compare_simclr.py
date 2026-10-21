#!/usr/bin/env python3
"""
Quantitative comparison of the SimCLR models (ImageNet weights + one per augmentation set) on one dataset.

  static    (CPU, from representations/*_{train,test}.npz)
    colour_R2     ridge regression embedding -> colour statistics (saturation, brightness, 9 hue-family shares), held-out R2
    texture_R2    same for texture statistics (edge density, sharpness = Laplacian variance, luma contrast)
    top_pc_colour how much of the variance of the 10 leading principal components is linearly explained by the colour
                  statistics (variance-weighted R2 of "PC score ~ colour stats")
    style_acc     linear-probe style accuracy (PCA-256 features, train -> test)
    cka_baseline  linear CKA to the `baseline` model (test split, 5000 images)
  invariance (GPU recommended, needs checkpoints in runs/): embed N test images and transformed copies
    cos / top1    mean cosine similarity of (mean-centred) embeddings of original vs transformed copy, and retrieval top-1:
                  is the nearest neighbour (among the N originals) of the transformed image its own original
    transforms:   grayscale, hue shift, invert, rot90, rot180, hflip, blur, sharpen

    python scripts/17_compare_simclr.py --dataset artbench --out docs/simclr_comparison_artbench   [--skip-invariance]
Writes <out>.json, <out>.md and <out>_heatmap.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.augmentations import list_aug_sets

REP = REPO / "representations"
ATTR = REPO / "data" / "attributes"
COLOUR = ["saturation", "brightness"] + [f"h_{n}" for n in
          ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta", "neutral"]]


# ── image statistics (colour from data/attributes, texture computed once and cached) ──────────────────

def texture_stats(paths: list[str], workers: int = 32) -> np.ndarray:
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image
    from scipy import ndimage

    def one(p):
        im = Image.open(p)
        im.draft("L", (192, 192))
        g = np.asarray(im.convert("L").resize((128, 128)), dtype=np.float32) / 255.0
        lap = ndimage.laplace(g)
        gx, gy = ndimage.sobel(g, 1), ndimage.sobel(g, 0)
        mag = np.hypot(gx, gy)
        return [float((mag > 0.5).mean()), float(lap.var()), float(g.std())]
    with ThreadPoolExecutor(workers) as ex:
        return np.array(list(ex.map(one, paths)), dtype=np.float32)


def load_targets(ds: str, paths: np.ndarray):
    z = np.load(ATTR / f"pixels_{ds}_all.npz", allow_pickle=False)
    idx = {str(p): i for i, p in enumerate(z["paths"])}
    sel = np.array([idx[str(p)] for p in paths])
    col = np.stack([z[k][sel] for k in COLOUR], 1).astype(np.float32)
    cache = ATTR / f"texture_{ds}_all.npz"
    if cache.exists():
        t = np.load(cache, allow_pickle=False)
        tidx = {str(p): i for i, p in enumerate(t["paths"])}
        tex = t["tex"][np.array([tidx[str(p)] for p in paths])]
    else:
        allp = [str(p) for p in z["paths"]]
        tex_all = texture_stats(allp)
        np.savez_compressed(cache, paths=np.array(allp), tex=tex_all)
        tex = tex_all[sel]
    return col, tex


# ── static metrics ────────────────────────────────────────────────────────────

def ridge_r2(Xtr, Ytr, Xte, Yte, alpha=1000.0):
    """Closed-form ridge in the dual/primal (X standardised, Y standardised); returns held-out R2 per target."""
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-6
    ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-6
    A, B = (Xtr - xm) / xs, (Xte - xm) / xs
    Yt = (Ytr - ym) / ys
    W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1], dtype=A.dtype), A.T @ Yt)
    P = B @ W * ys + ym
    return 1 - ((Yte - P) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)


def linear_cka(X, Y):
    X, Y = X - X.mean(0), Y - Y.mean(0)
    xy = np.linalg.norm(X.T @ Y) ** 2
    return float(xy / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def linear_probe_acc(Xtr, ytr, Xte, yte):
    import torch
    import torch.nn.functional as F
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = torch.tensor((Xtr - mu) / sd), torch.tensor((Xte - mu) / sd)
    y, ye = torch.tensor(ytr), torch.tensor(yte)
    clf = torch.nn.Linear(A.shape[1], int(y.max()) + 1)
    opt = torch.optim.LBFGS(clf.parameters(), lr=1.0, max_iter=100, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(clf(A), y) + 1e-4 * clf.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return float((clf(B).argmax(1) == ye).float().mean())


def static_metrics(ds: str, slugs: dict[str, str], n_cka: int = 5000) -> dict:
    out, base_te = {}, None
    cache = {}
    for name, slug in slugs.items():
        tr, te = (np.load(REP / f"{ds}_{slug}_{s}.npz", allow_pickle=False) for s in ("train", "test"))
        Xtr, Xte = tr["vectors"].astype(np.float32), te["vectors"].astype(np.float32)
        if "col" not in cache:
            cache["col"] = (load_targets(ds, tr["paths"]), load_targets(ds, te["paths"]))
            cache["ytr"], cache["yte"] = tr["labels"], te["labels"]
            cache["rng"] = np.random.default_rng(0).permutation(len(Xte))[:n_cka]
        (ctr, ttr), (cte, tte) = cache["col"]
        r_col = ridge_r2(Xtr, ctr, Xte, cte)
        r_tex = ridge_r2(Xtr, ttr, Xte, tte)
        # leading components: PC scores (PCA on train) explained by the colour statistics
        mu = Xtr.mean(0)
        _, s, Vt = np.linalg.svd(Xtr[np.random.default_rng(1).permutation(len(Xtr))[:20000]] - mu, full_matrices=False)
        Ptr, Pte = (Xtr - mu) @ Vt[:10].T, (Xte - mu) @ Vt[:10].T
        r_pc = ridge_r2(ctr, Ptr, cte, Pte, alpha=1.0)           # PC score ~ colour stats
        var = Pte.var(0)
        # 256-d PCA features for the style probe
        Z = (Xtr - mu) @ Vt[:256].T
        Zt = (Xte - mu) @ Vt[:256].T
        out[name] = {
            "colour_R2": float(r_col.mean()), "colour_R2_sat": float(r_col[0]), "colour_R2_bright": float(r_col[1]),
            "colour_R2_hue": float(r_col[2:].mean()), "texture_R2": float(r_tex.mean()),
            "top_pc_colour": float((r_pc * var).sum() / var.sum()),
            "style_acc": linear_probe_acc(Z, cache["ytr"], Zt, cache["yte"]),
            "dim": int(Xtr.shape[1]),
        }
        cache[name] = Xte[cache["rng"]]
        print(f"  {name:16s}", {k: round(v, 3) for k, v in out[name].items() if k != "dim"}, flush=True)
    if "baseline" in cache:
        for name in slugs:
            out[name]["cka_baseline"] = linear_cka(cache[name], cache["baseline"])
    return out


# ── invariance tests ──────────────────────────────────────────────────────────

def _transforms():
    from PIL import ImageFilter, ImageOps
    from torchvision import transforms as T
    hue = T.ColorJitter(hue=(0.3, 0.3))
    return {
        "grayscale": lambda im: ImageOps.grayscale(im).convert("RGB"),
        "hue_shift": lambda im: hue(im),
        "invert": ImageOps.invert,
        "rot90": lambda im: im.rotate(90, expand=True),
        "rot180": lambda im: im.rotate(180),
        "hflip": ImageOps.mirror,
        "blur": lambda im: im.filter(ImageFilter.GaussianBlur(3)),
        "sharpen": lambda im: im.filter(ImageFilter.UnsharpMask(radius=2, percent=400, threshold=0)),
    }


def invariance_metrics(ds: str, ckpts: dict[str, Path | None], n: int, device: str, batch: int = 128) -> dict:
    import torch
    from torch.utils.data import DataLoader, Dataset

    from src.datasets.simclr_data import SplitImages
    from src.models.simclr_extractor import SimCLRExtractor

    base = SplitImages(ds, "test", None, draft_size=288)
    idx = np.sort(np.random.default_rng(5).permutation(len(base))[:n])
    tfs = _transforms()
    out = {}

    class DS(Dataset):
        def __init__(self, tf, extractor_tf):
            self.tf, self.etf = tf, extractor_tf

        def __len__(self):
            return len(idx)

        def __getitem__(self, i):
            im = base.load(int(idx[i]))
            return self.etf(self.tf(im) if self.tf else im)

    for name, ck in ckpts.items():
        ex = SimCLRExtractor(checkpoint=ck, device=device)
        ex.setup()
        etf = ex._get_transform()

        def embed(tf):
            fs = []
            with torch.no_grad():
                for x in DataLoader(DS(tf, etf), batch_size=batch, num_workers=16):
                    fs.append(ex._forward(x.to(ex.device)).cpu())
            return torch.cat(fs)
        F0 = embed(None)
        mu = F0.mean(0, keepdim=True)
        n0 = torch.nn.functional.normalize(F0 - mu, dim=1)
        res = {}
        for tn, tf in tfs.items():
            F1 = torch.nn.functional.normalize(embed(tf) - mu, dim=1)
            res[tn] = {"cos": float((n0 * F1).sum(1).mean()), "top1": float(((F1 @ n0.T).argmax(1) == torch.arange(n)).float().mean())}
        out[name] = res
        print(f"  {name:16s}", {k: round(v["cos"], 2) for k, v in res.items()}, flush=True)
        del ex
    return out


# ── report ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["artbench", "wikiart"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-invariance", action="store_true")
    ap.add_argument("--skip-static", action="store_true")
    ap.add_argument("--n-inv", type=int, default=2000)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    ds = a.dataset
    out_base = Path(a.out or f"docs/simclr_comparison_{ds}")
    out_base.parent.mkdir(exist_ok=True, parents=True)

    sets = [s for s in list_aug_sets() if (REP / f"{ds}_simclr_{ds}_{s}_test.npz").exists()]
    slugs = {"imagenet": "simclr_imagenet", **{s: f"simclr_{ds}_{s}" for s in sets}}
    ckpts = {"imagenet": None}
    for s in sets:
        runs = sorted((REPO / "runs").glob(f"*_simclr_{ds}_{s}/simclr_final.pt"))
        if runs:
            ckpts[s] = runs[-1]
    res = {"dataset": ds, "static": {}, "invariance": {}}
    if not a.skip_static:
        print("static metrics")
        res["static"] = static_metrics(ds, slugs)
    if not a.skip_invariance:
        import torch
        dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("invariance metrics on", dev)
        res["invariance"] = invariance_metrics(ds, ckpts, a.n_inv, dev)
    Path(f"{out_base}.json").write_text(json.dumps(res, indent=1))

    md = [f"# SimCLR augmentation comparison · {ds}\n"]
    if res["static"]:
        cols = ["colour_R2", "colour_R2_sat", "colour_R2_bright", "colour_R2_hue", "texture_R2", "top_pc_colour", "style_acc", "cka_baseline"]
        md += ["## Static metrics (held-out test split; R2 of ridge regression embedding -> statistic)\n",
               "| model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for m, v in res["static"].items():
            md.append(f"| {m} | " + " | ".join(f"{v.get(c, float('nan')):.3f}" for c in cols) + " |")
    if res["invariance"]:
        tns = list(next(iter(res["invariance"].values())))
        md += ["\n## Invariance: mean cosine (centred) / retrieval top-1 of transformed vs original\n",
               "| model | " + " | ".join(tns) + " |", "|---" * (len(tns) + 1) + "|"]
        for m, v in res["invariance"].items():
            md.append(f"| {m} | " + " | ".join(f"{v[t]['cos']:.2f} / {v[t]['top1']:.2f}" for t in tns) + " |")
    Path(f"{out_base}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        panels = []
        if res["static"]:
            cols = ["colour_R2", "texture_R2", "top_pc_colour", "style_acc"]
            panels.append(("static", list(res["static"]), cols, np.array([[res["static"][m][c] for c in cols] for m in res["static"]])))
        if res["invariance"]:
            panels.append(("invariance (cos)", list(res["invariance"]), tns,
                           np.array([[res["invariance"][m][t]["cos"] for t in tns] for m in res["invariance"]])))
        fig, axs = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.5), squeeze=False)
        for ax, (title, rows, cols, M) in zip(axs[0], panels):
            im = ax.imshow(M, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
            ax.set_yticks(range(len(rows)), rows)
            ax.set_title(f"{ds}: {title}")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w", fontsize=7)
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(f"{out_base}_heatmap.png", dpi=130)
    except Exception as e:  # plotting is optional
        print("heatmap skipped:", e)


if __name__ == "__main__":
    main()
