#!/usr/bin/env python3
"""
MNIST explorer: the same interactive page as for ArtBench, for the intermediate representations of a small CNN.

  1. downloads MNIST (torchvision) and trains the CNN of exercise sheet 1 (src/models/mnist_cnn.py) on the 60,000
     training images (weights + history in runs/mnist_cnn/),
  2. computes the 10,000 test images' representations in
         pixels            the raw 784 pixel values
         conv1 conv2 conv3 activations after each conv block (flattened feature maps)
         fc                the 256-unit hidden layer
         logits            the 10 class scores
     for the TRAINED network and for the same network at random initialisation (the baseline: what does the
     architecture alone give you?),
  3. for each representation: PCA, k-means, UMAP, k-NN lists, t-SNE (pixels and conv3 only) (like scripts/08_build_explorer.py) and one
     explorer page.

Output (a self-contained folder: copy it anywhere and serve it through a web server):
  <out-dir>/index.html                overview: one preview per representation
  <out-dir>/pages/<stem>.html         explorer pages (mnist_pixels, mnist_cnn_<layer>_<trained|random>)
  <out-dir>/shared/meta/mnist-*.js    labels, thumbnails (data URIs) and per-image attributes, shared by all pages
  <out-dir>/cache/<stem>/             projections / clusters (reused; not needed to view the pages)

    python scripts/30_build_mnist_explorer.py                      # everything (about 30 min on a login node)
    python scripts/30_build_mnist_explorer.py --reps pixels conv3:trained --skip-tsne
    python scripts/30_build_mnist_explorer.py --index-only         # only rewrite index.html from the cache
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):     # see scripts/08_build_explorer.py
    os.environ.setdefault(_v, "64")

import argparse
import base64
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.mnist_cnn import LAYERS, MNIST_MEAN, MNIST_STD, MnistCNN
from src.visualization.explorer import write_explorer, write_meta

DIGITS = [str(d) for d in range(10)]
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
WEIGHTS = {"trained": "trained on MNIST", "random": "random init (untrained)"}
TITLES = {"pixels": "pixels", **{k: k for k in LAYERS}}
DIMS = {"pixels": 784, **{k: v[1] for k, v in LAYERS.items()}}
DESCRIPTIONS = {"pixels": "the raw 28 x 28 image, 784 numbers", **{k: v[0] for k, v in LAYERS.items()}}


def stem_of(layer: str, weights: str | None) -> str:
    return "mnist_pixels" if layer == "pixels" else f"mnist_cnn_{layer}_{weights}"


def all_reps() -> list[tuple[str, str | None]]:
    return [("pixels", None)] + [(layer, w) for layer in LAYERS for w in WEIGHTS]


def parse_rep(s: str) -> tuple[str, str | None]:
    layer, _, w = s.partition(":")
    if layer == "pixels":
        return ("pixels", None)
    if layer not in LAYERS:
        raise SystemExit(f"unknown representation {s!r}; choose from pixels, {', '.join(LAYERS)} (each with :trained or :random)")
    return (layer, w or None) if w else (layer, "trained")


def explorer_helpers():
    """The projection / k-means / neighbour helpers of scripts/08_build_explorer.py (file name starts with a digit)."""
    spec = importlib.util.spec_from_file_location("build_explorer", REPO_ROOT / "scripts" / "08_build_explorer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── data and model ────────────────────────────────────────────────────────────

def load_mnist(root: Path):
    from torchvision.datasets import MNIST
    tr, te = MNIST(str(root), train=True, download=True), MNIST(str(root), train=False, download=True)
    return tr.data.numpy(), tr.targets.numpy(), te.data.numpy(), te.targets.numpy()


def to_input(x_u8: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(x_u8).float().div_(255).unsqueeze(1)
    return ((x - MNIST_MEAN) / MNIST_STD).to(device)


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return float(torch.cat([model(x[i:i + 2000]).argmax(1) for i in range(0, len(x), 2000)]).eq(y).float().mean())


def train_model(args, xtr, ytr, xte, yte, device) -> tuple[MnistCNN, dict]:
    run = Path(args.run_dir)
    wfile, hfile = run / "cnn.pt", run / "history.json"
    model = MnistCNN().to(device)
    if wfile.exists() and not args.retrain:
        model.load_state_dict(torch.load(wfile, map_location=device, weights_only=True))
        print(f"loaded trained CNN from {wfile}")
        return model, json.loads(hfile.read_text())
    torch.manual_seed(args.seed)
    model = MnistCNN().to(device)
    x, y, xt, yt = to_input(xtr, device), torch.from_numpy(ytr).to(device), to_input(xte, device), torch.from_numpy(yte).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    hist = {"epoch": [], "train_loss": [], "test_acc": []}
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(x), device=device)
        total = 0.0
        for i in range(0, len(x), args.batch_size):
            b = perm[i:i + args.batch_size]
            loss = nn.functional.cross_entropy(model(x[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        acc = accuracy(model, xt, yt)
        hist["epoch"].append(ep); hist["train_loss"].append(total / len(x)); hist["test_acc"].append(acc)
        print(f"  epoch {ep}/{args.epochs}: train loss {total / len(x):.4f}, test accuracy {100 * acc:.2f}%")
    run.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), wfile)
    hfile.write_text(json.dumps(hist))
    return model, hist


@torch.no_grad()
def extract(model: MnistCNN, x_u8: np.ndarray, device) -> dict[str, np.ndarray]:
    out: dict[str, list] = {k: [] for k in LAYERS}
    for i in range(0, len(x_u8), 1000):
        for k, v in model.intermediates(to_input(x_u8[i:i + 1000], device)).items():
            out[k].append(v.cpu().numpy())
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}


# ── per-image data shared by all pages ────────────────────────────────────────

def thumb_uri(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def pixel_stats(x_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ink (mean pixel value) and slant (+ = leaning right like italics) of every image."""
    I = x_u8.astype(np.float64) / 255
    yy, xx = np.mgrid[0:28, 0:28]
    m = I.sum((1, 2)) + 1e-9
    cx, cy = (I * xx).sum((1, 2)) / m, (I * yy).sum((1, 2)) / m
    var_y = (I * (yy - cy[:, None, None]) ** 2).sum((1, 2)) / m
    cov = (I * (xx - cx[:, None, None]) * (yy - cy[:, None, None])).sum((1, 2)) / m
    return I.mean((1, 2)), -cov / np.maximum(var_y, 1e-6)


def shared_fields(eh, x_u8, y, logits) -> dict:
    prob = torch.softmax(torch.from_numpy(logits), 1).numpy()
    pred = prob.argmax(1)
    ink, slant = pixel_stats(x_u8)
    defs = [
        {"key": "cnn_pred", "label": "CNN prediction", "group": "Trained CNN", "kind": "cat", "names": DIGITS},
        {"key": "cnn_ok", "label": "CNN prediction is", "group": "Trained CNN", "kind": "cat", "names": ["correct", "wrong"]},
        {"key": "cnn_conf", "label": "CNN confidence (max softmax)", "group": "Trained CNN", "kind": "num"},
        {"key": "ink", "label": "ink (mean pixel value)", "group": "Pixel statistics", "kind": "num"},
        {"key": "slant", "label": "slant (+ = leaning right)", "group": "Pixel statistics", "kind": "num"},
    ]
    arrays = {"cnn_pred": pred, "cnn_ok": (pred != y).astype(int), "cnn_conf": prob.max(1), "ink": ink, "slant": slant}
    attrs = eh.encode_attrs((defs, arrays), np.arange(len(y)))
    for a in attrs:
        if a["key"] == "cnn_conf":
            a["range"] = [0.5, 1.0]             # nearly all images are > 0.98: percentile bins would be degenerate
    return {"lab": y.tolist(), "files": [f"test/{y[i]}/{i:05d}" for i in range(len(y))],
            "thumbs": [thumb_uri(x_u8[i]) for i in range(len(y))], "attrs": attrs}


# ── one representation ────────────────────────────────────────────────────────

def knn_accuracy(X: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    """Leave-one-out k-NN accuracy (cosine similarity, X is L2-normalised)."""
    ok = 0
    for s in range(0, len(X), 1000):
        S = X[s:s + 1000] @ X.T
        S[np.arange(len(S)), np.arange(s, s + len(S))] = -2.0
        part = np.argpartition(-S, k, axis=1)[:, :k]
        votes = np.apply_along_axis(lambda v: np.bincount(v, minlength=10).argmax(), 1, y[part])
        ok += int((votes == y[s:s + len(S)]).sum())
    return ok / len(X)


def build_page(eh, args, root: Path, layer: str, weights: str | None, X_raw: np.ndarray, y: np.ndarray,
               shared: dict, meta: str) -> dict:
    stem = stem_of(layer, weights)
    n, d = X_raw.shape
    settings = dict(stem=stem, n=n, seed=args.seed, variance=args.variance, n_neighbors=args.n_neighbors,
                    min_dist=args.min_dist, perplexity=args.perplexity, max_pcs=args.max_pcs, ks=args.ks,
                    weights_mtime=Path(args.run_dir, "cnn.pt").stat().st_mtime if weights == "trained" else 0)
    cache = eh.Cache(root / "cache" / stem, settings, args.force)
    X = normalize(X_raw, norm="l2")

    zf, evf = cache.dir / "pca_Z.npy", cache.dir / "pca_ev.npy"
    if zf.exists() and evf.exists():
        Z, ev = np.load(zf), np.load(evf)
    else:
        t0 = time.perf_counter()
        pca = PCA(n_components=min(args.max_pcs, d), svd_solver="full" if d <= 1500 else "randomized",
                  random_state=args.seed).fit(X)
        ratio = pca.explained_variance_ratio_
        keep = int(min(len(ratio), np.searchsorted(np.cumsum(ratio), args.variance) + 1))
        Z = pca.transform(X)[:, :keep].astype(np.float32)
        ev = (ratio[:keep] * 100).astype(np.float32)
        np.save(zf, Z); np.save(evf, ev)
        print(f"    pca: {time.perf_counter() - t0:.1f}s")
    n_pc = Z.shape[1]
    print(f"  PCA: {n_pc} components explain {ev.sum():.1f}% of the variance (of {d} dims)")
    spaces = {"embedding": f"embedding ({d}-D)", "pca": f"PCA {ev.sum():.0f}% ({n_pc} PCs)"}
    space_X = {"embedding": X, "pca": Z}

    todo = [("umap_embedding", "UMAP · embedding", lambda: eh._umap(X, args)),
            ("umap_pca", f"UMAP · PCA {ev.sum():.0f}%", lambda: eh._umap(Z, args))]
    if not args.skip_tsne and layer in args.tsne_layers:        # t-SNE is slow: only where it is asked for
        todo.append(("tsne_pca", f"t-SNE · PCA {ev.sum():.0f}%", lambda: eh._tsne(Z, args.perplexity, args.seed)))
    proj = {key: {"name": name, "xy": cache.get(key, fn)} for key, name, fn in todo}
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        if b < n_pc:
            proj[f"pc{a + 1}{b + 1}"] = {"name": f"PCA · PC{a + 1} vs PC{b + 1} ({ev[a]:.1f}% / {ev[b]:.1f}%)", "xy": Z[:, [a, b]]}

    ks = sorted({k for k in args.ks if k < n} | {10})
    clusters = {sp: {k: cache.get(f"kmeans_{sp}_k{k}", lambda: eh._kmeans(XS, k, args.seed)).astype(int) for k in ks}
                for sp, XS in space_X.items()}
    metrics = {sp: {str(k): eh._subset_metrics(space_X[sp], v, y, n, args.seed) for k, v in dd.items()}
               for sp, dd in clusters.items()}
    acc = knn_accuracy(X, y)
    print(f"  10-NN accuracy in the full space: {100 * acc:.2f}%")

    r = lambda a, dec=3: np.round(np.asarray(a, dtype=float), dec).tolist()
    sel = np.random.default_rng(0).choice(n, size=min(900, n), replace=False)
    pv = np.asarray(proj["umap_pca"]["xy"], dtype=float)[sel]
    pv = (pv - pv.min(0)) / np.maximum(pv.max(0) - pv.min(0), 1e-9)
    mm = metrics["embedding"]["10"]
    summary = {"stem": stem, "layer": layer, "weights": weights, "dim": d, "n_pc": n_pc, "var": float(ev.sum()),
               "knn": acc, "nmi": mm["nmi"], "ari": mm["ari"],
               "preview": {"x": r(pv[:, 0]), "y": r(pv[:, 1]), "c": y[sel].tolist()}}

    # navigation: one link per representation (same weights where there are two), "Weights" drop-down for the layers
    cur_w = weights or "trained"
    siblings = [{"name": f"{TITLES[ly]} ({DIMS[ly]})", "href": f"{stem_of(ly, cur_w)}.html",
                 "current": ly == layer} for ly in ["pixels", *LAYERS]]
    wlist = [] if layer == "pixels" else [{"label": lb, "href": f"{stem_of(layer, w)}.html", "current": w == weights}
                                          for w, lb in WEIGHTS.items()]
    what = "raw pixels" if layer == "pixels" else f"{layer}, {WEIGHTS[weights]}"
    payload = {
        "n": n, "classes": DIGITS, "idx": list(range(n)), **shared,
        "proj": {k: {"name": v["name"], "xy": r(v["xy"])} for k, v in proj.items()},
        "spaces": spaces, "ks": ks,
        "clusters": {sp: {str(k): v.tolist() for k, v in dd.items()} for sp, dd in clusters.items()},
        "metrics": metrics, "nn": eh.neighbour_payload(X, Z, ev),
        "default_proj": "umap_pca", "default_k": 10, "default_color": "style",
        "subtitle": f"{what} · {DESCRIPTIONS[layer]} · {n:,} MNIST test images · 10-NN accuracy {100 * acc:.1f}%",
        "noun": "digit", "home": {"href": "../index.html", "label": "MNIST overview"}, "style_colors": TAB10,
        "siblings": siblings, "weights": wlist, "variants": [], "sets": [],
    }
    page = write_explorer(root / "pages" / f"{stem}.html", payload, f"MNIST explorer · {what}", meta=meta)
    print(f"Explorer written -> {page}")
    (cache.dir / "landing.json").write_text(json.dumps(summary))
    return summary


# ── overview page ─────────────────────────────────────────────────────────────

INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MNIST explorer</title>
<style>
:root { --bg:#111; --panel:#1a1a1a; --line:#333; --fg:#e6e6e6; --muted:#8a8a8a; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1100px; margin:0 auto; padding:20px 16px 40px; }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:15px; margin:26px 0 8px; }
p { color:var(--muted); max-width:85ch; } a { color:#8ab4f8; text-decoration:none; } a:hover { text-decoration:underline; }
table { border-collapse:collapse; width:100%; }
td, th { border-top:1px solid var(--line); padding:10px 8px; vertical-align:middle; text-align:left; }
th { color:var(--muted); font-weight:500; font-size:12px; border-top:0; }
td.name b { font-size:15px; } td.name span { display:block; color:var(--muted); font-size:12px; max-width:30ch; }
.cell { display:flex; gap:10px; align-items:center; }
.cell canvas { width:110px; height:110px; background:var(--panel); border:1px solid var(--line); border-radius:6px; }
.cell div { font-size:12px; color:var(--muted); } .cell b { color:var(--fg); font-size:15px; }
.legend { display:flex; gap:10px; flex-wrap:wrap; margin:6px 0 0; font-size:12px; color:var(--muted); }
.legend i { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; }
</style></head><body><main>
<h1>MNIST explorer</h1>
<p>The 10,000 MNIST test images as seen by a small CNN (the one of exercise sheet 1: three conv blocks and a two-layer head, trained
on the 60,000 training images: <b id="acc"></b> test accuracy). Every page is the same explorer as for the paintings: a 2-D map
(UMAP, PCA and, for the pixels and the last conv layer, t-SNE), k-means clusters, nearest neighbours and interpolation, coloured by digit or by what the network thinks.
The rows follow the data through the network, from the pixels to the class scores; the right column is the same network
<i>before</i> training, so you can see what the architecture alone does and what training adds.</p>
<div class="legend" id="legend"></div>
<table id="t"><tr><th>representation</th><th>trained CNN</th><th>random initialisation</th></tr></table>
<p style="margin-top:22px">Previews: UMAP of the PCA-reduced representation, 900 random test images. <i>10-NN</i> is the leave-one-out accuracy of a
10-nearest-neighbour vote (cosine similarity) in the full space: how well the digit classes are separated <i>before</i> any 2-D projection.
Open <a href="../index.html">the art explorer overview</a>.</p>
</main>
<script>
const E = __DATA__;
document.getElementById("acc").textContent = (100 * E.test_acc).toFixed(2) + " %";
document.getElementById("legend").innerHTML = E.colors.map((c, d) => `<span><i style="background:${c}"></i>${d}</span>`).join("");
function cell(s) {
  if (!s) return "<td></td>";
  return `<td><a class="cell" href="pages/${s.stem}.html"><canvas width="220" height="220" data-stem="${s.stem}"></canvas>
    <div><b>${(100 * s.knn).toFixed(1)} %</b> 10-NN<br>${s.n_pc} PCs = ${s.var.toFixed(0)} % variance<br>NMI of 10 clusters ${s.nmi.toFixed(2)}</div></a></td>`;
}
const t = document.getElementById("t"), by = {};
E.entries.forEach(s => { by[s.stem] = s; });
for (const r of E.rows) {
  const tr = document.createElement("tr");
  tr.innerHTML = `<td class="name"><b>${r.name}</b><span>${r.desc} (${r.dim})</span></td>` + (r.pixels ?
    `${cell(by[r.pixels])}<td></td>` : `${cell(by[r.trained])}${cell(by[r.random])}`);
  t.appendChild(tr);
}
document.querySelectorAll("canvas").forEach(cv => {
  const p = by[cv.dataset.stem].preview, g = cv.getContext("2d");
  p.x.forEach((x, i) => { g.fillStyle = E.colors[p.c[i]]; g.beginPath();
    g.arc(8 + x * 204, 212 - p.y[i] * 204, 3.2, 0, 6.2832); g.fill(); });
});
</script></body></html>
"""


def write_index(root: Path, args) -> None:
    entries = [json.loads(f.read_text()) for f in sorted((root / "cache").glob("*/landing.json"))]
    rows = [{"name": "pixels", "desc": DESCRIPTIONS["pixels"], "dim": 784, "pixels": "mnist_pixels"}]
    rows += [{"name": layer, "desc": DESCRIPTIONS[layer], "dim": DIMS[layer],
              "trained": stem_of(layer, "trained"), "random": stem_of(layer, "random")} for layer in LAYERS]
    hfile = Path(args.run_dir) / "history.json"
    acc = json.loads(hfile.read_text())["test_acc"][-1] if hfile.exists() else float("nan")
    data = {"entries": entries, "rows": rows, "colors": TAB10, "test_acc": acc}
    (root / "index.html").write_text(INDEX_HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"Overview written -> {root / 'index.html'}  ({len(entries)} representations)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="visualizations/explorer/mnist")
    ap.add_argument("--data-dir", default="data/mnist")
    ap.add_argument("--run-dir", default="runs/mnist_cnn", help="Where the trained weights and history are kept.")
    ap.add_argument("--reps", nargs="+", default=None, metavar="REP",
                    help="Representations to build: pixels, or <layer>:<trained|random> (default: all).")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--retrain", action="store_true", help="Train again even if runs/mnist_cnn/cnn.pt exists.")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 8, 10, 12, 15, 20], help="Cluster counts in the page.")
    ap.add_argument("--variance", type=float, default=0.95, help="Variance kept for the PCA space.")
    ap.add_argument("--max-pcs", type=int, default=300, help="Upper limit of PCs kept for the PCA space.")
    ap.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors.")
    ap.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist.")
    ap.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    ap.add_argument("--skip-tsne", action="store_true")
    ap.add_argument("--tsne-layers", nargs="+", default=["pixels", "conv3"], metavar="LAYER",
                    help="Representations that get a t-SNE map (it takes ~10 min each): pixels and the last conv layer.")
    ap.add_argument("--index-only", action="store_true", help="Only rewrite index.html from the cache.")
    ap.add_argument("--force", action="store_true", help="Recompute cached projections / clusters.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.out_dir)
    if args.index_only:
        write_index(root, args)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xtr, ytr, xte, yte = load_mnist(Path(args.data_dir))
    print(f"MNIST: {len(xtr):,} train, {len(xte):,} test images; device {device}")
    model, hist = train_model(args, xtr, ytr, xte, yte, device)
    print(f"trained CNN: {100 * hist['test_acc'][-1]:.2f}% test accuracy")

    torch.manual_seed(args.seed + 1)
    random_model = MnistCNN().to(device)               # same architecture, untrained
    feats = {"trained": extract(model, xte, device), "random": extract(random_model, xte, device)}
    pixels = xte.reshape(len(xte), -1).astype(np.float32) / 255

    eh = explorer_helpers()
    shared = shared_fields(eh, xte, yte, feats["trained"]["logits"])
    meta = write_meta(root, "mnist", shared)           # thumbnails, labels, attributes: once for all pages
    reps = [parse_rep(s) for s in args.reps] if args.reps else all_reps()
    for layer, w in reps:
        print(f"\n── {stem_of(layer, w)} ──")
        t0 = time.perf_counter()
        build_page(eh, args, root, layer, w, pixels if layer == "pixels" else feats[w][layer], yte, shared, meta)
        print(f"    done in {time.perf_counter() - t0:.0f}s")
    write_index(root, args)
    print(f"\nServe it with:  python -m http.server --directory {root}")


if __name__ == "__main__":
    main()
