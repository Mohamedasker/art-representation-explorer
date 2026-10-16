#!/usr/bin/env python3
"""
Build an interactive explorer (k-means clusters + 2D projections + image hover).

For each representation .npz this
  1. draws a random sample of --n-samples images (the full set is too heavy for a browser),
  2. L2-normalises the embeddings and fits PCA keeping --variance (default 95%) of the variance,
  3. runs k-means for every k in --ks, both on the normalised embeddings and on the PCA scores,
  4. computes 2D projections — UMAP and t-SNE on both spaces, plus PC1/PC2, PC1/PC3, PC2/PC3,
  5. saves thumbnails of the sampled images and writes one self-contained HTML page.

In the page you can pick the projection, the cluster space (embedding / PCA 95%), k, and the
colouring (cluster or art style); hover a point to see the original image.

Output layout (copy the whole folder to view it locally):
  <out-dir>/<stem>.html            one page per representation, linked to each other
  <out-dir>/thumbs/<hash>.jpg      thumbnails, shared by pages that sample the same images
  <out-dir>/cache/<stem>/*.npy     projections / clusters, reused when settings are unchanged

Usage examples
--------------
    python scripts/08_build_explorer.py
    python scripts/08_build_explorer.py --repr representations/artbench_clip_vitb32_train.npz \\
        --n-samples 5000 --ks 3 5 8 12
    python scripts/08_build_explorer.py --skip-tsne        # UMAP + PCA only (much faster)
"""

from __future__ import annotations

import os

# The cluster's OpenBLAS is built for at most 128 threads; on a 256-core node it would
# otherwise try to start more and abort ("allocate too many memory regions").
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "64")

import argparse
import base64
import hashlib
import json
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize

from src.coloring.palette import (assign_to_reference, compute_palettes, palette_histograms,
                                  reference_palette, theme_clusters)
from src.metadata.attributes import load_attributes
from src.models.base_extractor import RepresentationBundle
from src.reduction.dimensionality import umap
from src.visualization.explorer import write_explorer
from src.visualization.landing import write_index

_SKIP_SUBSTRINGS = ("color_profiles",)


def _find_representations(repr_dir: Path, prefer_all: bool = False, with_all: bool = False) -> list[Path]:
    """Representation bundles.  By default train and test stay separate (<dataset>_<model>_train / _test
    files; merged *_all.npz files are ignored).  With ``prefer_all`` the merged files are used instead and the
    per-split files of the same model are skipped."""
    files = [f for f in sorted(repr_dir.glob("*.npz")) if not any(s in f.name for s in _SKIP_SUBSTRINGS)]
    if with_all:
        return files
    if not prefer_all:
        return [f for f in files if not f.name.endswith("_all.npz")]
    whole = {f.name[:-len("_all.npz")] for f in files if f.name.endswith("_all.npz")}
    return [f for f in files if not (f.name.endswith(("_train.npz", "_test.npz"))
                                    and f.name.rsplit("_", 1)[0] in whole)]


# ── caching ───────────────────────────────────────────────────────────────────

class Cache:
    """.npy files under one directory, invalidated when the settings dict changes."""

    def __init__(self, directory: Path, settings: dict, force: bool):
        self.dir = directory
        fp = directory / "settings.json"
        stale = force or not fp.exists() or json.loads(fp.read_text()) != settings
        if stale and directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(settings))

    def get(self, name: str, fn):
        p = self.dir / f"{name}.npy"
        if p.exists():
            return np.load(p)
        t0 = time.perf_counter()
        out = np.asarray(fn())
        np.save(p, out)
        print(f"    {name}: {time.perf_counter() - t0:.1f}s")
        return out


# ── computations ──────────────────────────────────────────────────────────────

def _tsne(X: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=perplexity, max_iter=1000, init="pca",
                learning_rate="auto", n_jobs=-1, random_state=seed).fit_transform(X)


def _umap(X: np.ndarray, args, parallel: bool = False) -> np.ndarray:
    # X is already L2-normalised (or PCA-projected), so euclidean ~ cosine; no extra PCA step.
    return umap(X, n_components=2, n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                metric="euclidean", normalize_input=False, pca_prewhiten=None,
                random_state=None if parallel else args.seed).coords


def _kmeans(X: np.ndarray, k: int, seed: int, n_init: int = 5) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(X).astype(np.int16)


# ── thumbnails ────────────────────────────────────────────────────────────────

def _thumb_name(path: str) -> str:
    return hashlib.sha1(path.encode()).hexdigest()[:14] + ".jpg"


def _make_thumb(src: Path, dst: Path, size: int) -> None:
    from PIL import Image
    if dst.exists():
        return
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        im.save(dst, quality=82)


def make_thumbs(paths: np.ndarray, thumb_dir: Path, size: int, workers: int) -> list[str]:
    thumb_dir.mkdir(parents=True, exist_ok=True)
    names = [_thumb_name(str(p)) for p in paths]
    jobs = []
    for p, n in zip(paths, names):
        src = Path(p) if Path(p).is_absolute() else REPO_ROOT / p
        jobs.append((src, thumb_dir / n))
    todo = sum(not d.exists() for _, d in jobs)
    print(f"  thumbnails: {todo} new / {len(jobs)}")
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda sd: _make_thumb(sd[0], sd[1], size), jobs))
    return names


# ── colour palettes (from the thumbnails, so the large originals are not re-read) ──

def load_palettes(thumb_dir: Path, names: list[str], cache_file: Path,
                  n_colors: int, workers: int) -> dict[str, np.ndarray]:
    """Palettes for the given thumbnails; computed once per thumbnail and cached in one .npz."""
    fields = ("colors", "weights", "lightness", "chroma")
    store: dict[str, dict] = {}
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=False)
        for i, nm in enumerate(z["names"]):
            store[str(nm)] = {f: z[f][i] for f in fields}
    missing = [nm for nm in dict.fromkeys(names) if nm not in store]
    print(f"  palettes: {len(missing)} new / {len(names)}")
    if missing:
        res = compute_palettes([thumb_dir / nm for nm in missing], n_colors=n_colors, workers=workers)
        for i, nm in enumerate(missing):
            store[nm] = {f: res[f][i] for f in fields}
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        keys = list(store)
        np.savez_compressed(cache_file, names=np.array(keys),
                            **{f: np.stack([store[k][f] for k in keys]) for f in fields})
    return {f: np.stack([store[nm][f] for nm in names]) for f in fields}


# ── painting date from the file name ──────────────────────────────────────────

# ArtBench files are "<artist>_<title>[-<year>].jpg"; only a year at the very end is trusted
# (years elsewhere in a title are often death dates or unrelated numbers).  A trailing
# "-<1-2 digits>" is tolerated as a duplicate counter.  WikiArt files are hashes: no dates.
_YEAR_RE = re.compile(r"[-_](\d{4})(?:-\d{1,2})?$")


def year_from_name(path: str) -> int | None:
    m = _YEAR_RE.search(Path(str(path)).stem)
    if m and 1300 <= int(m.group(1)) <= 2025:
        return int(m.group(1))
    return None


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def _pack(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.int64)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


# ── extra attributes (metadata / title tags / CLIP concepts) from scripts/09_build_attributes.py ──

def _path_key(p: str) -> str:
    return "/".join(Path(str(p)).parts[-3:])          # train/<style>/<file>


def load_attrs(stem: str, paths: np.ndarray, attr_dir: Path) -> list[dict]:
    m = re.match(r"(artbench|wikiart)_.*_(train|test|all)$", stem)
    f = attr_dir / f"{m.group(1)}_{m.group(2)}.npz" if m else None
    if f is None or not f.exists():
        print("  attributes: none (run scripts/09_build_attributes.py)")
        return []
    apaths, defs, arrs = load_attributes(f)
    row = {_path_key(p): i for i, p in enumerate(apaths)}
    sel = np.array([row.get(_path_key(p), -1) for p in paths])
    if (sel < 0).any():
        print(f"  attributes: {int((sel < 0).sum())} / {len(sel)} samples missing in {f.name}; skipped")
        return []
    out = []
    for d in defs:
        v = arrs[d["key"]][sel]
        item = dict(d)
        # compact base64 typed arrays (decoded in the page): num -> uint16 over [lo, hi] (65535 = n/a),
        # cat -> int16 (-1 = unknown), colour -> int32 (-1 = n/a)
        if d["kind"] == "num":
            v = v.astype(float)
            ok = ~np.isnan(v)
            lo, hi = (float(v[ok].min()), float(v[ok].max())) if ok.any() else (0.0, 1.0)
            q = np.where(ok, np.round((np.nan_to_num(v) - lo) / ((hi - lo) or 1.0) * 65534), 65535).astype(np.uint16)
            item.update(b64=_b64(q), lo=lo, hi=hi)
            item["range"] = [float(np.percentile(v[ok], 2)), float(np.percentile(v[ok], 98))] if ok.any() else [0.0, 1.0]
        elif d["kind"] == "color":
            item["b64"] = _b64(v.astype(np.int32))
        else:
            item["b64"] = _b64(v.astype(np.int16))
        out.append(item)
    print(f"  attributes: {len(out)} from {f.name}")
    return out


# ── data for the neighbour / interpolation modes of the viewer ────────────────

def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def neighbour_payload(X: np.ndarray, Z: np.ndarray, ev: np.ndarray, k: int = 30, n_pcs: int = 64) -> dict:
    """Exact cosine k-NN lists (X is L2-normalised) and the leading PCA coordinates (for interpolation)."""
    n = len(X)
    k = min(k, n - 1)
    idx = np.zeros((n, k), dtype=np.int64)
    sim = np.zeros((n, k), dtype=np.float32)
    for s in range(0, n, 1000):
        S = X[s:s + 1000] @ X.T
        S[np.arange(len(S)), np.arange(s, s + len(S))] = -2.0          # exclude the point itself
        part = np.argpartition(-S, k, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, axis=1)
        o = np.argsort(-ps, axis=1)
        idx[s:s + len(S)] = np.take_along_axis(part, o, axis=1)
        sim[s:s + len(S)] = np.take_along_axis(ps, o, axis=1)
    p = min(n_pcs, Z.shape[1])
    Zp = Z[:, :p]
    scale = float(np.abs(Zp).max()) or 1.0
    wide = n > 65535
    return {"K": int(k), "idx": _b64(idx.astype(np.uint32 if wide else np.uint16)), "idx_t": "u4" if wide else "u2",
            "sim": _b64(np.round(np.clip(sim, 0, 1) * 255).astype(np.uint8)),
            "P": int(p), "pcs": _b64(np.round(Zp / scale * 32767).astype(np.int16)), "pcs_scale": scale,
            "pcs_var": float(ev[:p].sum() / 100.0)}


# ── per-representation pipeline ───────────────────────────────────────────────

def set_page(stem: str, kind: str, j: int = 0) -> str:
    return {"default": f"{stem}.html", "extra": f"{stem}__set{j + 1}.html", "full": f"{stem}__all.html"}[kind]


def set_label(kind: str, j: int, n: int) -> str:
    return {"default": f"{n:,} samples (set 1)", "extra": f"{n:,} other samples (set {j + 2})",
            "full": f"all {n:,} images"}[kind]


def choose_indices(n_total: int, args, kind: str, j: int = 0) -> np.ndarray:
    """default: the original random sample; extra j: a further sample disjoint from all earlier ones; full: everything."""
    n = min(args.n_samples, n_total)
    rng = np.random.default_rng(args.seed)
    first = np.sort(rng.choice(n_total, size=n, replace=False))
    if kind == "default":
        return first
    if kind == "full":
        return np.arange(n_total)
    used = first                                     # extra j (0-based) is the (j+2)-th sample overall
    for t in range(1, j + 2):
        remaining = np.setdiff1d(np.arange(n_total), used)
        pick = np.sort(np.random.default_rng(args.seed + t).choice(remaining, size=min(n, len(remaining)), replace=False))
        if t == j + 1:
            return pick
        used = np.union1d(used, pick)


def build(repr_path: Path, args, out_dir: Path, kind: str = "default", j: int = 0) -> dict:
    bundle = RepresentationBundle.load(repr_path)
    print(f"  Loaded: {bundle}  [{kind}{j or ''}]")
    n_total = bundle.n_samples
    idx = choose_indices(n_total, args, kind, j)
    n = len(idx)
    big = n > args.big_n            # large sets: parallel (unseeded) UMAP, cheaper k-means, no t-SNE

    settings = dict(repr=repr_path.name, mtime=repr_path.stat().st_mtime, n=n, seed=args.seed,
                    variance=args.variance, n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                    perplexity=args.perplexity)
    if kind != "default":           # keep the default set's settings (and thus its cache) unchanged
        settings.update(kind=kind, j=j)
    cache_name = repr_path.stem if kind == "default" else f"{repr_path.stem}__{kind}{j or ''}"
    cache = Cache(out_dir / "cache" / cache_name, settings, args.force)

    X = normalize(bundle.vectors[idx].astype(np.float32), norm="l2")
    classes = sorted(set(bundle.label_names[idx].tolist()))
    # class index in the page = position in `classes`, regardless of the bundle's label ids
    cls_of = {c: j for j, c in enumerate(classes)}
    lab = np.array([cls_of[c] for c in bundle.label_names[idx]])

    pca = PCA(n_components=args.variance, svd_solver="full", random_state=args.seed).fit(X)
    Z = pca.transform(X).astype(np.float32)
    n_pc = Z.shape[1]
    print(f"  PCA: {n_pc} components explain {pca.explained_variance_ratio_.sum() * 100:.1f}% "
          f"(of {X.shape[1]} dims)")
    spaces = {"embedding": f"embedding ({X.shape[1]}-D)", "pca": f"PCA {args.variance * 100:.0f}% ({n_pc} PCs)"}
    space_X = {"embedding": X, "pca": Z}

    # projections
    ev = pca.explained_variance_ratio_ * 100
    proj: dict[str, dict] = {}
    for key, name, fn in [
        ("umap_embedding", "UMAP · embedding", lambda: _umap(X, args, big)),
        ("umap_pca", f"UMAP · PCA {args.variance * 100:.0f}%", lambda: _umap(Z, args, big)),
        ("tsne_embedding", "t-SNE · embedding", lambda: _tsne(X, args.perplexity, args.seed)),
        ("tsne_pca", f"t-SNE · PCA {args.variance * 100:.0f}%", lambda: _tsne(Z, args.perplexity, args.seed)),
    ]:
        if key.startswith("tsne") and (args.skip_tsne or n > args.tsne_max_n):
            continue
        proj[key] = {"name": name, "xy": cache.get(key, fn)}
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        if b < n_pc:
            proj[f"pc{a + 1}{b + 1}"] = {
                "name": f"PCA · PC{a + 1} vs PC{b + 1} ({ev[a]:.1f}% / {ev[b]:.1f}%)",
                "xy": Z[:, [a, b]]}

    # k-means in both spaces
    clusters: dict[str, dict[int, np.ndarray]] = {s: {} for s in spaces}
    metrics: dict[str, dict[int, dict]] = {s: {} for s in spaces}
    for s, XS in space_X.items():
        for k in args.ks:
            if k >= n:
                continue
            lab_k = cache.get(f"kmeans_{s}_k{k}", lambda: _kmeans(XS, k, args.seed, 2 if big else 5)).astype(int)
            clusters[s][k] = lab_k
            m = cache.get(f"metrics_{s}_k{k}", lambda: np.array([
                silhouette_score(XS, lab_k, sample_size=min(3000, n), random_state=args.seed),
                adjusted_rand_score(lab, lab_k),
                normalized_mutual_info_score(lab, lab_k)]))
            metrics[s][k] = {"sil": float(m[0]), "ari": float(m[1]), "nmi": float(m[2])}

    thumbs = make_thumbs(bundle.paths[idx], out_dir / "thumbs", args.thumb_size, args.workers)
    ks = [k for k in args.ks if k < n]

    years = [year_from_name(p) for p in bundle.paths[idx]]
    n_dated = sum(y is not None for y in years)
    print(f"  dates from file names: {n_dated} / {n}")

    # dominant colours -> shared colour vocabulary -> colour-theme clusters
    pal = load_palettes(out_dir / "thumbs", thumbs,
                        out_dir / "cache" / f"palettes_c{args.palette_colors}.npz",
                        args.palette_colors, args.workers)
    ref = reference_palette(pal["colors"], pal["weights"], args.palette_refs, args.seed)
    hist = palette_histograms(pal["colors"], pal["weights"], ref["centers_lab"])
    dom_ref = assign_to_reference(pal["colors"][:, :1], ref["centers_lab"])[:, 0]
    ref_hex = [_hex(c) for c in ref["rgb"]]
    theme, theme_sw, theme_metrics = {}, {}, {}
    tag = f"c{args.palette_colors}_r{args.palette_refs}"
    for k in ks:
        lab_t = cache.get(f"theme_{tag}_k{k}", lambda: theme_clusters(hist, k, args.seed)).astype(int)
        theme[str(k)] = lab_t.tolist()
        # swatches: the 4 reference colours carrying most of the cluster's pixel share
        theme_sw[str(k)] = [
            [ref_hex[j] for j in np.argsort(-hist[lab_t == c].mean(axis=0))[:4]] if (lab_t == c).any()
            else ["#000000"] for c in range(k)]
        theme_metrics[str(k)] = {"ari": float(adjusted_rand_score(lab, lab_t)),
                                 "nmi": float(normalized_mutual_info_score(lab, lab_t))}
    r = lambda a, d=3: np.round(np.asarray(a, dtype=float), d).tolist()
    # small summary for the landing page: a preview scatter (default projection, coloured by style) and
    # how well k-means (k = number of styles, embedding space) matches the art styles
    kd = min(ks, key=lambda k: abs(k - len(classes)))
    dp = "umap_pca" if "umap_pca" in proj else next(iter(proj))
    xy = np.asarray(proj[dp]["xy"], dtype=float)
    sel = np.random.default_rng(0).choice(n, size=min(900, n), replace=False)
    xy = (xy[sel] - xy.min(0)) / np.maximum(xy.max(0) - xy.min(0), 1e-9)
    mm = metrics["embedding"].get(kd, {})
    summary = {"dim": int(bundle.dim), "n": n, "n_total": int(n_total), "n_classes": len(classes), "k": kd,
               "nmi": mm.get("nmi"), "ari": mm.get("ari"), "sil": mm.get("sil"),
               "preview": {"x": np.round(xy[:, 0], 3).tolist(), "y": np.round(xy[:, 1], 3).tolist(),
                           "c": lab[sel].tolist()}}
    return {
        "stem": repr_path.stem,
        "kind": kind, "j": j, "page": set_page(repr_path.stem, kind, j),
        "summary": summary,
        "model": bundle.model_name,
        "payload": {
            "n": n,
            "classes": classes,
            "lab": lab.tolist(),
            "idx": idx.tolist(),
            "files": [Path(str(p)).name for p in bundle.paths[idx]],
            "thumbs": [f"thumbs/{t}" for t in thumbs],
            "proj": {k: {"name": v["name"], "xy": r(v["xy"])} for k, v in proj.items()},
            "spaces": spaces,
            "ks": ks,
            "clusters": {s: {str(k): v.tolist() for k, v in d.items()} for s, d in clusters.items()},
            "metrics": {s: {str(k): v for k, v in d.items()} for s, d in metrics.items()},
            "year": years if n_dated else None,
            "pal_c": _pack(pal["colors"]).tolist(),
            "pal_w": r(pal["weights"], 2),
            "light": r(pal["lightness"], 1),
            "chroma": r(pal["chroma"], 1),
            "dom_ref": dom_ref.tolist(),
            "ref_colors": ref_hex,
            "theme": theme,
            "theme_sw": theme_sw,
            "theme_metrics": theme_metrics,
            "attrs": load_attrs(repr_path.stem, bundle.paths[idx], Path(args.attr_dir)),
            "nn": neighbour_payload(X, Z, ev),
            "default_proj": "umap_pca",
            "default_k": min(ks, key=lambda k: abs(k - len(classes))),
            "subtitle": f"{bundle.model_name} · {n:,} of {n_total:,} images · {len(classes)} art styles",
            "set_label": set_label(kind, j, n),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the interactive k-means / projection explorer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repr-dir", default="representations",
                        help="Directory to scan for .npz representation bundles.")
    parser.add_argument("--repr", nargs="+", default=None, metavar="NPZ",
                        help="Explicit list of .npz files instead of scanning --repr-dir.")
    parser.add_argument("--prefer-all", action="store_true",
                        help="Use merged <dataset>_<model>_all.npz files (train + test) instead of the separate "
                             "per-split files.")
    parser.add_argument("--with-all", action="store_true",
                        help="Build pages for the merged train + test files (<dataset>_<model>_all.npz, see "
                             "scripts/merge_splits.py) in addition to the per-split pages.")
    parser.add_argument("--out-dir", default="visualizations/explorer")
    parser.add_argument("--attr-dir", default="data/attributes",
                        help="Per-image attributes from scripts/09_build_attributes.py (genre, tags, CLIP concepts).")
    parser.add_argument("--n-samples", type=int, default=10000,
                        help="Random sample size shown in the page.")
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30],
                        help="Cluster counts available in the page.")
    parser.add_argument("--variance", type=float, default=0.95,
                        help="Fraction of variance kept for the PCA space.")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument("--skip-tsne", action="store_true", help="Skip the (slow) t-SNE projections.")
    parser.add_argument("--extra-sets", type=int, default=0,
                        help="Also build this many further random samples of --n-samples images, disjoint from "
                             "the first one and from each other (pages <stem>__set2.html, ...).")
    parser.add_argument("--full", action="store_true",
                        help="Also build a page showing ALL images of the dataset (<stem>__all.html). Large pages; "
                             "t-SNE is skipped above --tsne-max-n points.")
    parser.add_argument("--only", nargs="+", choices=["default", "extra", "full"], default=None,
                        help="Build only these kinds of pages (links to already existing pages are kept).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip pages whose .html already exists (resume an interrupted run).")
    parser.add_argument("--tsne-max-n", type=int, default=20000, help="Skip t-SNE for sets larger than this.")
    parser.add_argument("--big-n", type=int, default=20000,
                        help="Above this many points UMAP runs unseeded (parallel) and k-means uses fewer restarts.")
    parser.add_argument("--thumb-size", type=int, default=192, help="Thumbnail max side in px.")
    parser.add_argument("--workers", type=int, default=16, help="Threads for thumbnail creation.")
    parser.add_argument("--palette-colors", type=int, default=5,
                        help="Dominant colours extracted per painting.")
    parser.add_argument("--palette-refs", type=int, default=16,
                        help="Size of the shared colour vocabulary used for colour themes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Recompute cached projections/clusters.")
    args = parser.parse_args()

    if args.repr:
        repr_paths = [Path(p) for p in args.repr]
    else:
        repr_paths = _find_representations(Path(args.repr_dir), args.prefer_all, args.with_all)
        if not repr_paths:
            print(f"No representation .npz files found in {args.repr_dir}.")
            return
    out_dir = Path(args.out_dir)
    args.ks = sorted(set(args.ks))

    want = set(args.only) if args.only else {"default", "extra", "full"}

    def n_in(path: Path) -> int:
        return int(np.load(path, allow_pickle=False)["labels"].shape[0])

    def declared_sets(n_tot: int) -> list[tuple[str, int]]:
        """Pages that make sense for a bundle of n_tot images: further samples only while enough images are left,
        the full page only if it differs from the default sample (so the small test splits get one page)."""
        out = [("default", 0)]
        out += [("extra", t) for t in range(args.extra_sets)
                if n_tot - args.n_samples * (t + 1) >= args.n_samples // 2]
        if args.full and n_tot > args.n_samples:
            out.append(("full", 0))
        return out

    declared = {r.stem: declared_sets(n_in(r)) for r in repr_paths}
    by_stem = {r.stem: r for r in repr_paths}

    def group_of(stem: str) -> str:      # dataset + split, e.g. "artbench_train": models are compared within a group
        return f"{stem.split('_')[0]}_{stem.rsplit('_', 1)[-1]}"

    def finalize(b: dict) -> None:
        """Write one page right after it is built (so a job that hits its time limit keeps what it finished)."""
        pl = b["payload"]
        pl["siblings"] = [{"name": o, "href": set_page(o, b["kind"], b["j"]), "current": o == b["stem"]}
                          for o in by_stem if group_of(o) == group_of(b["stem"]) and (b["kind"], b["j"]) in declared[o]]
        n_tot = n_in(by_stem[b["stem"]])
        pl["sets"] = []
        for kind, t in declared[b["stem"]]:
            n_set = n_tot if kind == "full" else min(args.n_samples, n_tot)
            page = set_page(b["stem"], kind, t)
            pl["sets"].append({"label": set_label(kind, t, n_set), "href": page, "current": page == b["page"]})
        b["sets_links"] = pl["sets"]
        p = write_explorer(out_dir / b["page"], pl, f"Explorer · {b['stem']}")
        print(f"Explorer written -> {p}")
        if b["kind"] == "default":       # what the landing page needs, kept next to the cache
            groups: dict[str, list] = {}
            for a in pl.get("attrs", []):
                groups.setdefault(a["group"], []).append(a["label"])
            (out_dir / "cache" / b["stem"] / "landing.json").write_text(json.dumps(
                {"stem": b["stem"], "summary": b["summary"], "sets_links": b["sets_links"], "attr_groups": groups}))

    def write_landing() -> None:
        entries = []
        for stem in by_stem:
            f = out_dir / "cache" / stem / "landing.json"
            if f.exists() and (out_dir / set_page(stem, "default")).exists():
                entries.append(json.loads(f.read_text()))
        if entries:
            idx = write_index(out_dir / "index.html", entries, args.n_samples)
            print(f"Landing page written -> {idx}")

    for repr_path in repr_paths:
        for kind, t in declared[repr_path.stem]:
            if kind not in want:
                continue
            done = (out_dir / set_page(repr_path.stem, kind, t)).exists() and (
                kind != "default" or (out_dir / "cache" / repr_path.stem / "landing.json").exists())
            if args.skip_existing and done:                    # (a default page without landing.json is from an older build)
                print(f"── {repr_path.stem}  [{kind}{t or ''}]  already built, skipped")
                continue
            print(f"\n── {repr_path.stem}  [{kind}{t or ''}] ──")
            b = build(repr_path, args, out_dir, kind, t)
            finalize(b)
            write_landing()
    write_landing()
    print(f"\nCopy {out_dir}/ (pages + thumbs/) to a laptop to view, e.g.\n"
          f"  scp -r <jureca>:{out_dir.resolve()} .")


if __name__ == "__main__":
    main()
