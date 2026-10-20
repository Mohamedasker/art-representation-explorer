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
from src.metadata.nudity import load_flags
from src.models.base_extractor import RepresentationBundle
from src.reduction.dimensionality import umap
from src.visualization.explorer import write_explorer, write_meta
from src.visualization.style_colors import style_colors
from src.visualization.landing import write_index
from src.visualization.variants import variant_info

_SKIP_SUBSTRINGS = ("color_profiles",)


def _find_representations(repr_dir: Path, prefer_all: bool = True) -> list[Path]:
    """Representation bundles to explore: the merged <dataset>_<model>_all.npz files (train + test in one space,
    scripts/merge_splits.py), or with ``prefer_all=False`` the separate per-split files."""
    files = [f for f in sorted(repr_dir.glob("*.npz")) if not any(s in f.name for s in _SKIP_SUBSTRINGS)]
    if prefer_all:
        return [f for f in files if f.name.endswith("_all.npz")]
    return [f for f in files if f.name.endswith(("_train.npz", "_test.npz"))]


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
    # name -> row of the cached arrays (each array is decompressed ONCE; indexing z[f][i] inside a loop would
    # decompress the whole array for every row)
    arrays: dict[str, np.ndarray] = {}
    row: dict[str, int] = {}
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=False)
        arrays = {f: z[f] for f in fields}
        row = {str(nm): i for i, nm in enumerate(z["names"])}
    missing = [nm for nm in dict.fromkeys(names) if nm not in row]
    print(f"  palettes: {len(missing)} new / {len(names)}")
    if missing:
        res = compute_palettes([thumb_dir / nm for nm in missing], n_colors=n_colors, workers=workers)
        keys = list(row) + missing
        arrays = {f: (np.concatenate([arrays[f], np.asarray(res[f])]) if row else np.asarray(res[f])) for f in fields}
        row = {k: i for i, k in enumerate(keys)}
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():                  # another build job may have added entries meanwhile: keep them
            z = np.load(cache_file, allow_pickle=False)
            theirs = [str(nm) for nm in z["names"]]
            extra = [i for i, nm in enumerate(theirs) if nm not in row]
            if extra:
                for f in fields:
                    arrays[f] = np.concatenate([arrays[f], z[f][extra]])
                for i in extra:
                    row[theirs[i]] = len(row)
                keys = list(row)
        tmp = cache_file.with_name(f"{cache_file.stem}.{os.getpid()}.tmp.npz")   # unique per process
        np.savez_compressed(tmp, names=np.array(keys), **arrays)
        tmp.replace(cache_file)
    sel = np.array([row[nm] for nm in names])
    return {f: arrays[f][sel] for f in fields}


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


def load_attr_arrays(stem: str, paths: np.ndarray, attr_dir: Path):
    """(defs, {key: array aligned with ``paths``}) from the attribute file of this representation, or None."""
    m = re.match(r"(artbench|wikiart)_.*_(train|test|all)$", stem)
    f = attr_dir / f"{m.group(1)}_{m.group(2)}.npz" if m else None
    if f is None or not f.exists():
        print("  attributes: none (run scripts/09_build_attributes.py)")
        return None
    apaths, defs, arrs = load_attributes(f)
    row = {_path_key(p): i for i, p in enumerate(apaths)}
    sel = np.array([row.get(_path_key(p), -1) for p in paths])
    if (sel < 0).any():
        print(f"  attributes: {int((sel < 0).sum())} / {len(sel)} samples missing in {f.name}; skipped")
        return None
    print(f"  attributes: {len(defs)} from {f.name}")
    return defs, {d["key"]: arrs[d["key"]][sel] for d in defs}


def encode_attrs(attr, idx: np.ndarray) -> list[dict]:
    """Page payload for the attributes of the images ``idx``: compact base64 typed arrays (decoded in the page):
    num -> uint16 over [lo, hi] (65535 = n/a), cat -> int16 (-1 = unknown), colour -> int32 (-1 = n/a)."""
    if attr is None:
        return []
    defs, arrays = attr
    out = []
    for d in defs:
        v = arrays[d["key"]][idx]
        item = dict(d)
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
    return out


def load_attrs(stem: str, paths: np.ndarray, attr_dir: Path) -> list[dict]:
    """All attributes for ``paths`` (used by scripts/rerender_explorer.py)."""
    attr = load_attr_arrays(stem, paths, attr_dir)
    return encode_attrs(attr, np.arange(len(paths))) if attr else []


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

# ── one embedding space per representation, many views (subsets) of it ───────────────────────────────

SUBSET_ORDER = ["s1", "s2", "train", "test", "all"]


def set_page(stem: str, key: str) -> str:
    return f"{stem}.html" if key == "s1" else f"{stem}__{key}.html"


def splits_of(paths) -> np.ndarray:
    def one(p: str) -> str:
        parts = Path(str(p)).parts
        return next((sp for sp in ("train", "test") if sp in parts), "")
    return np.array([one(p) for p in paths], dtype=str)


def subset_defs(paths, args) -> list[tuple[str, str, np.ndarray]]:
    """Views of one embedding space: (key, label, indices).  Every view just selects some of the points that
    were embedded together: random samples, the train / test split, everything."""
    n = len(paths)
    ns = min(args.n_samples, n)
    first = np.sort(np.random.default_rng(args.seed).choice(n, size=ns, replace=False))
    views = {"s1": (f"{ns:,} random images (sample 1)", first)}
    rest = np.setdiff1d(np.arange(n), first)
    if len(rest) >= ns // 2 and ns < n:
        views["s2"] = (f"{min(ns, len(rest)):,} other random images (sample 2)",
                       np.sort(np.random.default_rng(args.seed + 1).choice(rest, size=min(ns, len(rest)), replace=False)))
    sp = splits_of(paths)
    for name in ("train", "test"):
        m = np.flatnonzero(sp == name)
        if 0 < len(m) < n:
            views[name] = (f"{name} split ({len(m):,} images)", m)
    if ns < n:
        views["all"] = (f"all {n:,} images", np.arange(n))
    return [(k, *views[k]) for k in SUBSET_ORDER if k in views]


def compute_space(repr_path: Path, args, out_dir: Path) -> dict:
    """Everything that depends on the whole set of embeddings (PCA, UMAP, t-SNE, k-means, colour themes), computed
    once and cached on disk.  The pages are then cheap slices of it."""
    bundle = RepresentationBundle.load(repr_path)
    n = bundle.n_samples
    big = n > args.big_n            # large sets: parallel (unseeded) UMAP, cheaper k-means
    print(f"  Loaded: {bundle}")
    settings = dict(repr=repr_path.name, mtime=repr_path.stat().st_mtime, n=n, seed=args.seed,
                    variance=args.variance, n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                    perplexity=args.perplexity, tsne_embedding=args.tsne_embedding)
    cache = Cache(out_dir / "cache" / f"{repr_path.stem}__space", settings, args.force)

    X = normalize(bundle.vectors.astype(np.float32), norm="l2")
    classes = sorted(set(bundle.label_names.tolist()))
    cls_of = {c: j for j, c in enumerate(classes)}
    lab = np.array([cls_of[c] for c in bundle.label_names])

    zf, evf = cache.dir / "pca_Z.npy", cache.dir / "pca_ev.npy"
    if zf.exists() and evf.exists():
        Z, ev = np.load(zf), np.load(evf)
    else:
        t0 = time.perf_counter()
        pca = PCA(n_components=args.variance, svd_solver="full", random_state=args.seed).fit(X)
        Z = pca.transform(X).astype(np.float32)
        ev = pca.explained_variance_ratio_ * 100
        np.save(zf, Z); np.save(evf, ev)
        print(f"    pca: {time.perf_counter() - t0:.1f}s")
    n_pc = Z.shape[1]
    print(f"  PCA: {n_pc} components explain {ev.sum():.1f}% (of {X.shape[1]} dims)")
    spaces = {"embedding": f"embedding ({X.shape[1]}-D)", "pca": f"PCA {args.variance * 100:.0f}% ({n_pc} PCs)"}
    space_X = {"embedding": X, "pca": Z}

    proj: dict[str, dict] = {}
    todo = [("umap_embedding", "UMAP · embedding", lambda: _umap(X, args, big)),
            ("umap_pca", f"UMAP · PCA {args.variance * 100:.0f}%", lambda: _umap(Z, args, big))]
    if not args.skip_tsne:
        todo.append(("tsne_pca", f"t-SNE · PCA {args.variance * 100:.0f}%", lambda: _tsne(Z, args.perplexity, args.seed)))
        if args.tsne_embedding:
            todo.append(("tsne_embedding", "t-SNE · embedding", lambda: _tsne(X, args.perplexity, args.seed)))
    for key, name, fn in todo:
        proj[key] = {"name": name, "xy": cache.get(key, fn)}
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        if b < n_pc:
            proj[f"pc{a + 1}{b + 1}"] = {
                "name": f"PCA · PC{a + 1} vs PC{b + 1} ({ev[a]:.1f}% / {ev[b]:.1f}%)", "xy": Z[:, [a, b]]}

    # k-means counts: the requested ones plus the true number of classes (the page opens on that k)
    ks = sorted({k for k in args.ks if k < n} | {len(classes)})
    clusters: dict[str, dict[int, np.ndarray]] = {s_: {} for s_ in spaces}
    for sname, XS in space_X.items():
        for k in ks:
            clusters[sname][k] = cache.get(f"kmeans_{sname}_k{k}",
                                           lambda: _kmeans(XS, k, args.seed, 2 if big else 5)).astype(int)

    thumbs = make_thumbs(bundle.paths, out_dir / "thumbs", args.thumb_size, args.workers)
    years = [year_from_name(p) for p in bundle.paths]
    print(f"  dates from file names: {sum(y is not None for y in years)} / {n}")
    pal = load_palettes(out_dir / "thumbs", thumbs, out_dir / "cache" / f"palettes_c{args.palette_colors}.npz",
                        args.palette_colors, args.workers)
    ref = reference_palette(pal["colors"], pal["weights"], args.palette_refs, args.seed)
    hist = palette_histograms(pal["colors"], pal["weights"], ref["centers_lab"])
    dom_ref = assign_to_reference(pal["colors"][:, :1], ref["centers_lab"])[:, 0]
    ref_hex = [_hex(c) for c in ref["rgb"]]
    tag = f"c{args.palette_colors}_r{args.palette_refs}"
    theme, theme_sw = {}, {}
    for k in ks:
        lab_t = cache.get(f"theme_{tag}_k{k}", lambda: theme_clusters(hist, k, args.seed)).astype(int)
        theme[k] = lab_t
        theme_sw[str(k)] = [[ref_hex[j] for j in np.argsort(-hist[lab_t == c].mean(axis=0))[:4]] if (lab_t == c).any()
                            else ["#000000"] for c in range(k)]
    attr = load_attr_arrays(repr_path.stem, bundle.paths, Path(args.attr_dir))
    nsfw = load_flags(bundle.paths)           # True = flagged as nudity: hidden by default in the page
    if nsfw is None:
        print("  nudity flags: NONE (run scripts/15_flag_nudity.py first) - explicit images will NOT be hidden in this page")
    else:
        print(f"  nudity flags: {int(nsfw.sum()):,} of {len(nsfw):,} images flagged")
    return dict(stem=repr_path.stem, bundle=bundle, n=n, X=X, Z=Z, ev=ev, n_pc=n_pc, classes=classes, lab=lab,
                spaces=spaces, space_X=space_X, proj=proj, ks=ks, clusters=clusters, thumbs=thumbs, years=years,
                pal=pal, dom_ref=dom_ref, ref_hex=ref_hex, theme=theme, theme_sw=theme_sw, attr=attr, nsfw=nsfw)


def _subset_metrics(XS, lab_k, lab, n, seed):
    try:
        sil = silhouette_score(XS, lab_k, sample_size=min(3000, n), random_state=seed) if len(set(lab_k.tolist())) > 1 else float("nan")
    except Exception:
        sil = float("nan")
    return {"sil": float(sil), "ari": float(adjusted_rand_score(lab, lab_k)),
            "nmi": float(normalized_mutual_info_score(lab, lab_k))}


def shared_fields(S: dict, idx: np.ndarray) -> dict:
    """Page fields of the images ``idx`` that do not depend on the model (the same for every representation of a
    dataset): written once per dataset by write_meta() and left out of the pages (src/visualization/explorer.py)."""
    r = lambda a, d=3: np.round(np.asarray(a, dtype=float), d).tolist()
    pal = {f: np.asarray(v)[idx] for f, v in S["pal"].items()}
    years = [S["years"][i] for i in idx]
    return {
        "lab": S["lab"][idx].tolist(),
        "files": [Path(str(p)).name for p in S["bundle"].paths[idx]],
        "thumbs": [f"thumbs/{S['thumbs'][i]}" for i in idx],
        "year": years if any(y is not None for y in years) else None,
        "pal_c": _pack(pal["colors"]).tolist(), "pal_w": r(pal["weights"], 2),
        "light": r(pal["lightness"], 1), "chroma": r(pal["chroma"], 1),
        "dom_ref": S["dom_ref"][idx].tolist(),
        "theme": {str(k): v[idx].tolist() for k, v in S["theme"].items()},
        "attrs": encode_attrs(S["attr"], idx),
        **({"nsfw": S["nsfw"][idx].astype(int).tolist()} if S["nsfw"] is not None else {}),
    }


def make_page(S: dict, key: str, label: str, idx: np.ndarray, args) -> dict:
    """A view of the shared space: the selected points only (their coordinates, clusters, colours and metadata)."""
    bundle, n, ks = S["bundle"], len(idx), S["ks"]
    lab, classes = S["lab"][idx], S["classes"]
    Xs, Zs = S["X"][idx], S["Z"][idx]
    sub_X = {"embedding": Xs, "pca": Zs}
    r = lambda a, d=3: np.round(np.asarray(a, dtype=float), d).tolist()
    proj = {k: {"name": v["name"], "xy": np.asarray(v["xy"])[idx]} for k, v in S["proj"].items()}
    clusters = {sp: {k: v[idx] for k, v in d.items()} for sp, d in S["clusters"].items()}
    metrics = {sp: {k: _subset_metrics(sub_X[sp], v, lab, n, args.seed) for k, v in d.items()}
               for sp, d in clusters.items()}
    theme = {str(k): v[idx] for k, v in S["theme"].items()}
    theme_metrics = {k: {"ari": float(adjusted_rand_score(lab, v)), "nmi": float(normalized_mutual_info_score(lab, v))}
                     for k, v in theme.items()}
    kd = min(ks, key=lambda k: abs(k - len(classes)))
    dp = "umap_pca" if "umap_pca" in proj else next(iter(proj))
    xy = np.asarray(proj[dp]["xy"], dtype=float)
    sel = np.random.default_rng(0).choice(n, size=min(900, n), replace=False)
    xy = (xy[sel] - xy.min(0)) / np.maximum(xy.max(0) - xy.min(0), 1e-9)
    mm = metrics["embedding"].get(kd, {})
    summary = {"dim": int(bundle.dim), "n": n, "n_total": int(S["n"]), "n_classes": len(classes), "k": kd,
               "nmi": mm.get("nmi"), "ari": mm.get("ari"), "sil": mm.get("sil"),
               "preview": {"x": np.round(xy[:, 0], 3).tolist(), "y": np.round(xy[:, 1], 3).tolist(), "c": lab[sel].tolist(),
                           "colors": style_colors(classes)}}
    nsfw = S["nsfw"]
    if nsfw is not None:                       # flagged samples are hidden by default (see src/metadata/nudity.py)
        summary["preview"]["f"] = nsfw[idx][sel].astype(int).tolist()
        summary.update(nsfw_flagged=int(nsfw.sum()), nsfw_total=int(len(nsfw)))
    return {
        "stem": S["stem"], "key": key, "page": set_page(S["stem"], key), "summary": summary, "model": bundle.model_name,
        "payload": {
            "n": n, "classes": classes, "idx": idx.tolist(), **shared_fields(S, idx),
            "proj": {k: {"name": v["name"], "xy": r(v["xy"])} for k, v in proj.items()},
            "spaces": S["spaces"], "ks": ks,
            "clusters": {sp: {str(k): v.tolist() for k, v in d.items()} for sp, d in clusters.items()},
            "metrics": {sp: {str(k): v for k, v in d.items()} for sp, d in metrics.items()},
            "ref_colors": S["ref_hex"], "theme_sw": S["theme_sw"], "theme_metrics": theme_metrics,
            "nn": neighbour_payload(Xs, Zs, S["ev"]),
            "default_proj": "umap_pca", "default_k": kd,
            "subtitle": f"{bundle.model_name} · {n:,} of {S['n']:,} images · {len(classes)} art styles",
            "set_label": label,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the interactive explorer: one embedding space per representation, shown through "
                    "different views (random samples, train, test, everything)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repr-dir", default="representations", help="Directory to scan for .npz bundles.")
    parser.add_argument("--repr", nargs="+", default=None, metavar="NPZ",
                        help="Explicit list of .npz files instead of scanning --repr-dir.")
    parser.add_argument("--per-split", action="store_true",
                        help="Use the separate <dataset>_<model>_train / _test files instead of the merged "
                             "<dataset>_<model>_all.npz files (scripts/merge_splits.py), which is the default.")
    parser.add_argument("--sets", nargs="+", choices=SUBSET_ORDER, default=None,
                        help="Which views to write (default: all that exist; links to existing pages are kept).")
    parser.add_argument("--out-dir", default="visualizations/explorer")
    parser.add_argument("--attr-dir", default="data/attributes",
                        help="Per-image attributes from scripts/09_build_attributes.py (genre, tags, CLIP concepts).")
    parser.add_argument("--n-samples", type=int, default=10000, help="Size of the random-sample views.")
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30],
                        help="Cluster counts available in the page.")
    parser.add_argument("--variance", type=float, default=0.95, help="Fraction of variance kept for the PCA space.")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument("--skip-tsne", action="store_true", help="Skip the (slow) t-SNE projection.")
    parser.add_argument("--tsne-embedding", action="store_true",
                        help="Also run t-SNE on the raw embedding (by default only on the PCA space: much faster).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip representations whose pages all exist.")
    parser.add_argument("--big-n", type=int, default=20000,
                        help="Above this many points UMAP runs unseeded (parallel) and k-means uses fewer restarts.")
    parser.add_argument("--thumb-size", type=int, default=192, help="Thumbnail max side in px.")
    parser.add_argument("--workers", type=int, default=16, help="Threads for thumbnail creation.")
    parser.add_argument("--palette-colors", type=int, default=5, help="Dominant colours extracted per painting.")
    parser.add_argument("--palette-refs", type=int, default=16,
                        help="Size of the shared colour vocabulary used for colour themes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Recompute cached projections/clusters.")
    args = parser.parse_args()

    if args.repr:
        repr_paths = [Path(p) for p in args.repr]
    else:
        repr_paths = _find_representations(Path(args.repr_dir), prefer_all=not args.per_split)
        if not repr_paths:
            print(f"No representation .npz files found in {args.repr_dir}.")
            return
    out_dir = Path(args.out_dir)
    args.ks = sorted(set(args.ks))
    want = set(args.sets) if args.sets else set(SUBSET_ORDER)
    by_stem = {r.stem: r for r in repr_paths}
    views = {r.stem: subset_defs(np.load(r, allow_pickle=False)["paths"], args) for r in repr_paths}

    def group_of(stem: str) -> str:          # dataset + split-kind, e.g. "artbench_all": models are compared within it
        return f"{stem.split('_')[0]}_{stem.rsplit('_', 1)[-1]}"

    def finalize(b: dict) -> None:
        """Write one page right after it is built (a job that hits its time limit keeps what it finished)."""
        pl, stem = b["payload"], b["stem"]
        vi = variant_info(stem)
        pl["siblings"] = [{"name": o, "href": set_page(o, b["key"]),
                           "current": o == stem or (vi is not None and variant_info(o) is not None
                                                    and variant_info(o)["family"] == vi["family"])} for o in by_stem
                          if group_of(o) == group_of(stem) and b["key"] in [k for k, *_ in views[o]]
                          and (variant_info(o) is None or variant_info(o)["default"])]
        if vi is not None:        # augmentation-set drop-down: the members of this variant family (default first)
            # members built in THIS run, plus members whose page already exists from an earlier run
            known = set(by_stem) | {q.stem for q in Path(args.repr_dir).glob("*.npz")}
            fam = sorted((o for o in known if (variant_info(o) or {}).get("family") == vi["family"]
                          and (b["key"] in [k for k, *_ in views[o]] if o in views
                               else (out_dir / set_page(o, b["key"])).exists())),
                         key=lambda o: variant_info(o)["order"])
            pl["variants"] = [{"label": variant_info(o)["label"], "href": set_page(o, b["key"]), "current": o == stem}
                              for o in fam]
        pl["sets"] = [{"label": lb, "href": set_page(stem, k), "current": k == b["key"]} for k, lb, _ in views[stem]]
        b["sets_links"] = pl["sets"]
        p = write_explorer(out_dir / b["page"], pl, f"Explorer · {stem}", meta=b["meta"])
        print(f"Explorer written -> {p}")
        if b["key"] == "s1":                 # what the landing page needs
            groups: dict[str, list] = {}
            for a in pl.get("attrs", []):
                groups.setdefault(a["group"], []).append(a["label"])
            (out_dir / "cache" / f"{stem}__space" / "landing.json").write_text(json.dumps(
                {"stem": stem, "summary": b["summary"], "sets_links": b["sets_links"], "attr_groups": groups}))

    def write_landing() -> None:
        entries = []                         # every representation that has a page, not only those of this run
        for f in sorted((out_dir / "cache").glob("*__space/landing.json")):
            stem = f.parent.name.removesuffix("__space")
            if (out_dir / set_page(stem, "s1")).exists():
                entries.append(json.loads(f.read_text()))
        if entries:
            print(f"Landing page written -> {write_index(out_dir / 'index.html', entries, args.n_samples)}")

    for repr_path in repr_paths:
        stem = repr_path.stem
        todo = [(k, lb, ix) for k, lb, ix in views[stem] if k in want]
        if args.skip_existing and todo and all((out_dir / set_page(stem, k)).exists() for k, _, _ in todo) \
                and (out_dir / "cache" / f"{stem}__space" / "landing.json").exists():
            print(f"── {stem}: all pages already built, skipped")
            continue
        print(f"\n── {stem} ──")
        S = compute_space(repr_path, args, out_dir)
        meta = write_meta(out_dir, group_of(stem), shared_fields(S, np.arange(S["n"])))   # shared by all its pages
        for k, lb, ix in todo:
            t0 = time.perf_counter()
            finalize({**make_page(S, k, lb, ix, args), "meta": meta})
            print(f"    page '{k}' ({len(ix):,} images): {time.perf_counter() - t0:.0f}s")
            write_landing()
        del S
    write_landing()
    print(f"\nOpen {out_dir}/index.html through a web server, e.g.  python -m http.server --directory {out_dir}")


if __name__ == "__main__":
    main()
