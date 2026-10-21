"""
Cheap image statistics from a small (<=128 px) JPEG-draft decode of each image:

  saturation     mean HSV saturation (0 = grey)                                        numeric
  brightness     mean luma (0-255)                                                     numeric
  contrast       std of luma                                                           numeric
  paper          fraction of near-white, low-saturation pixels                          numeric  (sketches on paper)
  colour_mode    greyscale / monotone (single hue, e.g. sepia) / colour                  categorical
  h_<family>     colour histogram: share of pixels in each hue family (red, orange, yellow, green, cyan, blue,
                 purple, magenta; chromatic pixels = saturation > 0.2 and value > 0.15) plus 'neutral'
                 (grey / black / white); the nine shares sum to 1                         numeric

From the histogram the explorer derives the dominant hue family, a warm-vs-cool balance and the hue diversity.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

SIZE = 128
HUE_FAMILIES = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]
HUE_NAMES = HUE_FAMILIES + ["neutral"]
_EDGES = np.array([15, 45, 70, 165, 200, 260, 300, 345])          # upper hue edges in degrees (red wraps around 0)
WARM = [0, 1, 2, 7]                                                  # red, orange, yellow, magenta
COOL = [3, 4, 5, 6]                                                  # green, cyan, blue, purple
FIELDS = ("saturation", "brightness", "contrast", "paper", "colour_mode") + tuple(f"h_{n}" for n in HUE_NAMES)
COLOUR_MODES = ["colour", "monotone (one hue / sepia)", "greyscale"]


def _load(path: str) -> np.ndarray:
    im = Image.open(path)
    im.draft("RGB", (SIZE, SIZE))
    im = im.convert("RGB")
    im.thumbnail((SIZE, SIZE))
    return np.asarray(im, dtype=np.float32)


def features(path: str) -> tuple:
    try:
        a = _load(path)
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan, -1) + (np.nan,) * len(HUE_NAMES)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(-1), a.min(-1)
    sat = np.where(mx > 1, (mx - mn) / np.maximum(mx, 1), 0.0)
    val = mx / 255.0
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    hue_rad = np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)

    # colour mode: greyscale if almost no channel spread; monotone if all hues concentrated
    spread = float(np.percentile(mx - mn, 95))
    if spread < 12:
        mode = 2
    else:
        wgt = sat * val
        conc = float(np.hypot((np.cos(hue_rad) * wgt).sum(), (np.sin(hue_rad) * wgt).sum()) / max(wgt.sum(), 1e-6))
        mode = 1 if (conc > 0.93 and sat.mean() < 0.45) else 0

    paper = float(((luma > 200) & (sat < 0.15)).mean())

    # hue histogram over the chromatic pixels; everything else is 'neutral'
    chroma = (sat > 0.2) & (val > 0.15)
    hue = np.degrees(hue_rad)[chroma] % 360.0
    idx = np.searchsorted(_EDGES, hue, side="right") % len(HUE_FAMILIES)      # >= 345 deg wraps back to red
    counts = np.bincount(idx, minlength=len(HUE_FAMILIES)).astype(np.float64)
    total = float(chroma.size)
    shares = np.concatenate([counts / total, [1.0 - chroma.sum() / total]])
    return (float(sat.mean()), float(luma.mean()), float(luma.std()), paper, mode) + tuple(float(x) for x in shares)


def compute_all(paths: list[str], workers: int = 32, chunksize: int = 256) -> dict[str, np.ndarray]:
    from multiprocessing import Pool
    with Pool(workers) as pool:
        rows = pool.map(features, paths, chunksize=chunksize)
    cols = list(zip(*rows))
    out = {}
    for name, col in zip(FIELDS, cols):
        out[name] = np.array(col, dtype=np.int32 if name == "colour_mode" else np.float32)
    return out
