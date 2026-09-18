"""
Texture-based color values derived from the raw images.

We use two complementary approaches:
1. **Local Binary Patterns (LBP)** — a classic, fast texture descriptor.
2. **Gray-Level Co-occurrence Matrix (GLCM)** — measures image smoothness,
   contrast, homogeneity and energy via scikit-image.

The scalar summaries (entropy, contrast, homogeneity, energy) can each be
used as a colour axis in the scatter visualiser to reveal how texture
correlates with the learned representations.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from tqdm import tqdm


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_gray(path: str, size: int = 128) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def _lbp_entropy(gray: np.ndarray) -> float:
    """
    Compute the entropy of an LBP histogram — higher = more varied texture.
    Uses the uniform LBP implementation from scikit-image.
    """
    try:
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59), density=True)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))
    except ImportError:
        return float("nan")


def _glcm_features(gray: np.ndarray) -> dict[str, float]:
    """Compute GLCM contrast, dissimilarity, homogeneity, energy, correlation."""
    try:
        from skimage.feature import graycomatrix, graycoprops
        glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        return {
            "contrast":     float(graycoprops(glcm, "contrast")[0, 0]),
            "dissimilarity":float(graycoprops(glcm, "dissimilarity")[0, 0]),
            "homogeneity":  float(graycoprops(glcm, "homogeneity")[0, 0]),
            "energy":       float(graycoprops(glcm, "energy")[0, 0]),
            "correlation":  float(graycoprops(glcm, "correlation")[0, 0]),
        }
    except ImportError:
        return {k: float("nan") for k in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")}


def compute_texture_features(paths: np.ndarray, size: int = 128) -> dict[str, np.ndarray]:
    """
    Compute per-image texture features.

    Parameters
    ----------
    paths:
        String array of image paths.
    size:
        Thumbnail size (default 128×128).

    Returns
    -------
    dict mapping feature name → float32 array of shape (N,).
    """
    n = len(paths)
    results: dict[str, list[float]] = {
        "lbp_entropy":   [],
        "contrast":      [],
        "dissimilarity": [],
        "homogeneity":   [],
        "energy":        [],
        "correlation":   [],
    }

    for path in tqdm(paths, desc="Texture features"):
        try:
            gray = _load_gray(str(path), size=size)
            results["lbp_entropy"].append(_lbp_entropy(gray))
            glcm = _glcm_features(gray)
            for k in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
                results[k].append(glcm[k])
        except Exception:
            for k in results:
                results[k].append(float("nan"))

    return {k: np.array(v, dtype=np.float32) for k, v in results.items()}


# ── ColorVar factories ────────────────────────────────────────────────────────

def _make_color_var(feature: str, values: np.ndarray, colormap: str = "viridis") -> dict:
    return {
        "values": values,
        "names": values.astype(str),
        "colormap": colormap,
        "label": feature.replace("_", " ").title(),
        "is_categorical": False,
    }


def from_lbp_entropy(features: dict) -> dict:
    return _make_color_var("LBP entropy", features["lbp_entropy"], "plasma")

def from_contrast(features: dict) -> dict:
    return _make_color_var("GLCM contrast", features["contrast"], "magma")

def from_homogeneity(features: dict) -> dict:
    return _make_color_var("GLCM homogeneity", features["homogeneity"], "viridis")

def from_energy(features: dict) -> dict:
    return _make_color_var("GLCM energy", features["energy"], "cividis")
