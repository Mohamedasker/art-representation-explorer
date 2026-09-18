"""
Color-profile features derived from the raw images.

We compute three simple scalar summaries per image:
  * ``mean_hue``        — circular mean of the HSV hue channel.
  * ``mean_saturation`` — mean saturation (0 = grey, 1 = vivid).
  * ``mean_brightness`` — mean brightness / value (0 = dark, 1 = bright).
  * ``colorfulness``    — Hasler & Süsstrunk (2003) colorfulness metric.

Any of these can be used as the colour axis in scatter plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.models.base_extractor import RepresentationBundle


# ── Feature computation ───────────────────────────────────────────────────────

def _load_rgb(path: str, size: int = 64) -> np.ndarray:
    """Load and resize an image; return float32 RGB array [0, 1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB→HSV.  Input shape (H, W, 3), values in [0, 1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    v = maxc
    s = np.where(maxc != 0, (maxc - minc) / maxc, 0.0)
    delta = maxc - minc + 1e-8
    h = np.zeros_like(r)
    mask_r = (maxc == r) & (delta > 1e-7)
    mask_g = (maxc == g) & (delta > 1e-7)
    mask_b = (maxc == b) & (delta > 1e-7)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
    h = h / 6.0  # normalise to [0, 1]
    return np.stack([h, s, v], axis=-1)


def _colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) colorfulness metric."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, mean_rg = rg.std(), rg.mean()
    std_yb, mean_yb = yb.std(), yb.mean()
    std_rgyb = np.sqrt(std_rg**2 + std_yb**2)
    mean_rgyb = np.sqrt(mean_rg**2 + mean_yb**2)
    return float(std_rgyb + 0.3 * mean_rgyb)


def compute_color_features(paths: np.ndarray, size: int = 64) -> dict[str, np.ndarray]:
    """
    Compute per-image color features for all paths in ``paths``.

    Parameters
    ----------
    paths:
        String array of image file paths (length N).
    size:
        Thumbnail size for speed (default 64×64).

    Returns
    -------
    dict mapping feature name → float32 array of shape (N,).
    """
    mean_hue        = np.zeros(len(paths), dtype=np.float32)
    mean_saturation = np.zeros(len(paths), dtype=np.float32)
    mean_brightness = np.zeros(len(paths), dtype=np.float32)
    colorfulness    = np.zeros(len(paths), dtype=np.float32)

    for i, path in enumerate(tqdm(paths, desc="Color profiles")):
        try:
            rgb = _load_rgb(str(path), size=size)
            hsv = _rgb_to_hsv(rgb)
            # circular mean of hue
            angles = hsv[..., 0] * 2 * np.pi
            mean_hue[i] = float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()) % (2 * np.pi)) / (2 * np.pi)
            mean_saturation[i] = float(hsv[..., 1].mean())
            mean_brightness[i]  = float(hsv[..., 2].mean())
            colorfulness[i]     = _colorfulness(rgb)
        except Exception:
            pass   # leave as 0 for corrupt / missing images

    return {
        "mean_hue":        mean_hue,
        "mean_saturation": mean_saturation,
        "mean_brightness": mean_brightness,
        "colorfulness":    colorfulness,
    }


# ── ColorVar factories ────────────────────────────────────────────────────────

def _make_color_var(
    feature_name: str,
    values: np.ndarray,
    colormap: str = "viridis",
    label: Optional[str] = None,
) -> dict:
    return {
        "values": values,
        "names": values.astype(str),
        "colormap": colormap,
        "label": label or feature_name.replace("_", " ").title(),
        "is_categorical": False,
    }


def from_hue(features: dict[str, np.ndarray]) -> dict:
    return _make_color_var("mean_hue", features["mean_hue"], "hsv", "Mean hue")


def from_saturation(features: dict[str, np.ndarray]) -> dict:
    return _make_color_var("mean_saturation", features["mean_saturation"], "plasma", "Mean saturation")


def from_brightness(features: dict[str, np.ndarray]) -> dict:
    return _make_color_var("mean_brightness", features["mean_brightness"], "magma", "Mean brightness")


def from_colorfulness(features: dict[str, np.ndarray]) -> dict:
    return _make_color_var("colorfulness", features["colorfulness"], "inferno", "Colorfulness")
