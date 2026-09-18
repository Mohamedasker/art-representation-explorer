"""
Dimensionality reduction wrappers.

All reducers accept a float32 array of shape (N, D) and return (N, 2) or
(N, 3) projections together with a metadata dict.

Supported methods:
  * PCA   — fast, deterministic, linear baseline.
  * UMAP  — non-linear, topology-preserving, fast on large N.
  * t-SNE — non-linear, neighbourhood-preserving, slower.

Each function returns a :class:`ProjectionResult` that bundles the
low-dimensional coordinates with the hyperparameters used, making it
straightforward to reproduce or save results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from sklearn.preprocessing import normalize


# ── ProjectionResult ──────────────────────────────────────────────────────────

@dataclass
class ProjectionResult:
    """
    Output of a dimensionality-reduction step.

    Attributes
    ----------
    coords:
        Float32 array of shape (N, n_components).
    method:
        Name of the algorithm  (``"pca"``, ``"umap"``, ``"tsne"``).
    n_components:
        Output dimensionality (2 or 3).
    params:
        Hyperparameters used.
    elapsed_s:
        Wall-clock time in seconds.
    """

    coords: np.ndarray
    method: str
    n_components: int = 2
    params: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        np.savez_compressed(
            path,
            coords=self.coords.astype(np.float32),
            method=np.array(self.method, dtype=str),
            n_components=np.array(self.n_components, dtype=np.int64),
            params_json=np.array(json.dumps(self.params), dtype=str),
            elapsed_s=np.array(self.elapsed_s, dtype=np.float64),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ProjectionResult":
        import json
        data = np.load(path, allow_pickle=False)
        return cls(
            coords=data["coords"],
            method=str(data["method"]),
            n_components=int(data["n_components"]),
            params=json.loads(str(data["params_json"])),
            elapsed_s=float(data["elapsed_s"]),
        )

    def __repr__(self) -> str:
        return (
            f"ProjectionResult(method={self.method!r}, "
            f"shape={self.coords.shape}, elapsed={self.elapsed_s:.1f}s)"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _preprocess(
    X: np.ndarray,
    normalize_input: bool,
    pca_prewhiten: Optional[int],
) -> np.ndarray:
    """
    Optionally L2-normalise and PCA-prewhiten the feature matrix.

    Parameters
    ----------
    X:
        Input array (N, D).
    normalize_input:
        L2-normalise rows.
    pca_prewhiten:
        If given, reduce to this many PCA dimensions before the main method.
        Speeds up UMAP/t-SNE dramatically for high-D inputs.
    """
    if normalize_input:
        X = normalize(X, norm="l2")
    if pca_prewhiten is not None and pca_prewhiten < X.shape[1]:
        from sklearn.decomposition import PCA
        X = PCA(n_components=pca_prewhiten, random_state=42).fit_transform(X)
    return X.astype(np.float32)


# ── PCA ───────────────────────────────────────────────────────────────────────

def pca(
    X: np.ndarray,
    n_components: int = 2,
    normalize_input: bool = False,
    **kwargs,
) -> ProjectionResult:
    """
    Principal Component Analysis.

    Parameters
    ----------
    X:
        Feature matrix (N, D).
    n_components:
        Output dimensionality (2 or 3).
    normalize_input:
        L2-normalise rows before PCA.
    **kwargs:
        Extra keyword arguments forwarded to ``sklearn.decomposition.PCA``.

    Returns
    -------
    ProjectionResult
    """
    from sklearn.decomposition import PCA as _PCA

    X = _preprocess(X, normalize_input=normalize_input, pca_prewhiten=None)
    params = {"n_components": n_components, **kwargs}
    t0 = time.perf_counter()
    model = _PCA(n_components=n_components, random_state=42, **kwargs)
    coords = model.fit_transform(X)
    elapsed = time.perf_counter() - t0
    print(
        f"  PCA  explained variance: "
        f"{model.explained_variance_ratio_.sum()*100:.1f}%  "
        f"({elapsed:.1f}s)"
    )
    return ProjectionResult(
        coords=coords.astype(np.float32),
        method="pca",
        n_components=n_components,
        params={**params, "explained_variance_ratio": model.explained_variance_ratio_.tolist()},
        elapsed_s=elapsed,
    )


# ── UMAP ──────────────────────────────────────────────────────────────────────

def umap(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    normalize_input: bool = True,
    pca_prewhiten: Optional[int] = 50,
    random_state: int = 42,
    **kwargs,
) -> ProjectionResult:
    """
    Uniform Manifold Approximation and Projection (UMAP).

    Parameters
    ----------
    X:
        Feature matrix (N, D).
    n_components:
        Output dimensionality (2 or 3).
    n_neighbors:
        UMAP neighbourhood size — controls local vs. global structure.
    min_dist:
        Minimum distance between points in the embedding.
    metric:
        Distance metric (``"cosine"`` works well for normalised embeddings).
    normalize_input:
        L2-normalise rows before UMAP.
    pca_prewhiten:
        Reduce to this many PCA dimensions first (speeds up UMAP, default 50).
        Set to ``None`` to skip.
    random_state:
        Seed for reproducibility.
    **kwargs:
        Extra keyword arguments forwarded to ``umap.UMAP``.

    Returns
    -------
    ProjectionResult
    """
    try:
        import umap as _umap
    except ImportError:
        raise ImportError(
            "umap-learn is required.  Install with:  pip install umap-learn"
        )

    X = _preprocess(X, normalize_input=normalize_input, pca_prewhiten=pca_prewhiten)
    params = dict(n_components=n_components, n_neighbors=n_neighbors,
                  min_dist=min_dist, metric=metric, random_state=random_state, **kwargs)
    t0 = time.perf_counter()
    reducer = _umap.UMAP(**params)
    coords = reducer.fit_transform(X)
    elapsed = time.perf_counter() - t0
    print(f"  UMAP  ({elapsed:.1f}s)")
    return ProjectionResult(
        coords=coords.astype(np.float32),
        method="umap",
        n_components=n_components,
        params=params,
        elapsed_s=elapsed,
    )


# ── t-SNE ─────────────────────────────────────────────────────────────────────

def tsne(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float | str = "auto",
    n_iter: int = 1000,
    metric: str = "cosine",
    normalize_input: bool = True,
    pca_prewhiten: Optional[int] = 50,
    random_state: int = 42,
    **kwargs,
) -> ProjectionResult:
    """
    t-distributed Stochastic Neighbour Embedding (t-SNE).

    Parameters
    ----------
    X:
        Feature matrix (N, D).
    n_components:
        Output dimensionality (2 or 3).
    perplexity:
        Balances local vs. global structure; typical values 5–50.
    learning_rate:
        Step size; ``"auto"`` sets it to max(200, N/12).
    n_iter:
        Maximum optimisation steps.
    metric:
        Distance metric.
    normalize_input:
        L2-normalise rows before t-SNE.
    pca_prewhiten:
        Reduce to this many PCA dimensions first (greatly speeds up t-SNE).
    random_state:
        Seed for reproducibility.
    **kwargs:
        Extra keyword arguments forwarded to ``sklearn.manifold.TSNE``.

    Returns
    -------
    ProjectionResult
    """
    from sklearn.manifold import TSNE as _TSNE

    X = _preprocess(X, normalize_input=normalize_input, pca_prewhiten=pca_prewhiten)
    params = dict(n_components=n_components, perplexity=perplexity,
                  learning_rate=learning_rate, n_iter=n_iter, metric=metric,
                  random_state=random_state, **kwargs)
    t0 = time.perf_counter()
    model = _TSNE(**params)
    coords = model.fit_transform(X)
    elapsed = time.perf_counter() - t0
    print(f"  t-SNE  (KL-div={model.kl_divergence_:.4f}, {elapsed:.1f}s)")
    return ProjectionResult(
        coords=coords.astype(np.float32),
        method="tsne",
        n_components=n_components,
        params={k: v for k, v in params.items()},
        elapsed_s=elapsed,
    )


# ── Unified runner ────────────────────────────────────────────────────────────

Method = Literal["pca", "umap", "tsne"]


def reduce(
    X: np.ndarray,
    method: Method = "umap",
    n_components: int = 2,
    **kwargs,
) -> ProjectionResult:
    """
    Convenience wrapper — dispatch to the correct reduction function.

    Parameters
    ----------
    X:
        Feature matrix (N, D).
    method:
        ``"pca"``, ``"umap"``, or ``"tsne"``.
    n_components:
        Output dimensionality.
    **kwargs:
        Forwarded to the chosen method.

    Returns
    -------
    ProjectionResult
    """
    dispatch = {"pca": pca, "umap": umap, "tsne": tsne}
    if method not in dispatch:
        raise ValueError(f"Unknown method {method!r}.  Choose from {list(dispatch)}")
    return dispatch[method](X, n_components=n_components, **kwargs)
