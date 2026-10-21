"""
Dominant-colour palettes of paintings.

For each image we cluster its pixels with k-means in CIELAB (perceptually
uniform, so clusters follow how colours look rather than raw RGB distance) and
keep the cluster means as a palette, ordered by how many pixels each covers.

A *reference palette* learned over a whole set of images turns palettes into
comparable features: every palette colour is mapped to its nearest reference
colour, giving each image a histogram over a shared colour vocabulary.  Those
histograms can be clustered into "colour themes" (e.g. dark and earthy, pastel,
vivid blues).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

_D65 = np.array([0.95047, 1.0, 1.08883])
_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]])


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 255], shape (..., 3) -> CIELAB (D65), shape (..., 3)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ _RGB2XYZ.T) / _D65
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


# ── per-image palette ─────────────────────────────────────────────────────────

def extract_palette(
    path: str | Path,
    n_colors: int = 5,
    size: int = 48,
    seed: int = 0,
) -> dict:
    """
    Dominant colours of one image.

    Returns
    -------
    dict with
        ``colors``    — uint8 (n_colors, 3) RGB cluster means, most dominant first.
        ``weights``   — float32 (n_colors,) share of pixels per colour (sums to 1).
        ``lightness`` — mean CIELAB L* (0 black .. 100 white).
        ``chroma``    — mean CIELAB chroma (0 grey .. ~100+ vivid).
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        pix = np.asarray(im, dtype=np.float64).reshape(-1, 3)
    lab = rgb_to_lab(pix)

    km = KMeans(n_clusters=n_colors, n_init=1, random_state=seed).fit(lab)
    counts = np.bincount(km.labels_, minlength=n_colors).astype(np.float64)
    colors = np.zeros((n_colors, 3))
    for c in range(n_colors):
        if counts[c] > 0:
            colors[c] = pix[km.labels_ == c].mean(axis=0)
    order = np.argsort(-counts)
    return {
        "colors": np.clip(np.rint(colors[order]), 0, 255).astype(np.uint8),
        "weights": (counts[order] / counts.sum()).astype(np.float32),
        "lightness": float(lab[:, 0].mean()),
        "chroma": float(np.hypot(lab[:, 1], lab[:, 2]).mean()),
    }


def _init_worker() -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)      # one BLAS/OpenMP thread per worker process


def _palette_job(args: tuple) -> dict:
    path, n_colors, size = args
    return extract_palette(path, n_colors=n_colors, size=size)


def compute_palettes(
    paths: Sequence[str | Path],
    n_colors: int = 5,
    size: int = 48,
    workers: int = 16,
) -> dict[str, np.ndarray]:
    """Palettes for many images, in parallel.  Arrays are stacked over images."""
    jobs = [(str(p), n_colors, size) for p in paths]
    with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
        res = list(ex.map(_palette_job, jobs, chunksize=64))
    return {
        "colors": np.stack([r["colors"] for r in res]),
        "weights": np.stack([r["weights"] for r in res]),
        "lightness": np.array([r["lightness"] for r in res], dtype=np.float32),
        "chroma": np.array([r["chroma"] for r in res], dtype=np.float32),
    }


# ── shared colour vocabulary + colour themes ──────────────────────────────────

def reference_palette(
    colors: np.ndarray,
    weights: np.ndarray,
    n_refs: int = 16,
    seed: int = 42,
    max_points: int = 60000,
) -> dict[str, np.ndarray]:
    """
    Learn ``n_refs`` reference colours from all palette colours of a set of images
    (k-means in Lab, weighted by pixel share).

    Returns ``centers_lab`` (n_refs, 3) and ``rgb`` (n_refs, 3) uint8 — the
    weighted mean RGB of each reference colour's members, for display.
    """
    flat_rgb = colors.reshape(-1, 3).astype(np.float64)
    flat_w = np.maximum(weights.reshape(-1).astype(np.float64), 1e-6)
    if len(flat_rgb) > max_points:
        sel = np.random.default_rng(seed).choice(len(flat_rgb), max_points, replace=False)
        flat_rgb, flat_w = flat_rgb[sel], flat_w[sel]
    lab = rgb_to_lab(flat_rgb)
    km = KMeans(n_clusters=n_refs, n_init=3, random_state=seed).fit(lab, sample_weight=flat_w)
    rgb = np.zeros((n_refs, 3))
    for c in range(n_refs):
        m = km.labels_ == c
        if m.any():
            rgb[c] = np.average(flat_rgb[m], axis=0, weights=flat_w[m])
    return {"centers_lab": km.cluster_centers_, "rgb": np.clip(np.rint(rgb), 0, 255).astype(np.uint8)}


def assign_to_reference(colors: np.ndarray, centers_lab: np.ndarray) -> np.ndarray:
    """Nearest reference colour for every palette colour: (N, n_colors) int array."""
    lab = rgb_to_lab(colors.reshape(-1, 3))
    d = ((lab[:, None, :] - centers_lab[None, :, :]) ** 2).sum(-1)
    return d.argmin(axis=1).reshape(colors.shape[:2])


def palette_histograms(
    colors: np.ndarray,
    weights: np.ndarray,
    centers_lab: np.ndarray,
) -> np.ndarray:
    """Per-image share of pixels falling on each reference colour: (N, n_refs)."""
    ref = assign_to_reference(colors, centers_lab)
    hist = np.zeros((len(colors), len(centers_lab)), dtype=np.float64)
    for j in range(colors.shape[1]):
        np.add.at(hist, (np.arange(len(colors)), ref[:, j]), weights[:, j])
    return hist


def theme_clusters(hist: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """
    k-means over palette histograms (square-rooted, i.e. Hellinger geometry, so
    small colour shares still count).  Returns int labels (N,).
    """
    return KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(np.sqrt(hist)).astype(np.int16)
