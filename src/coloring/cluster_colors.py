"""
Color values from unsupervised cluster assignments.

Clusters are computed from the representation vectors in a
:class:`~src.models.base_extractor.RepresentationBundle`.

Supported algorithms: K-Means (default), Gaussian Mixture, DBSCAN.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

from src.models.base_extractor import RepresentationBundle


Algorithm = Literal["kmeans", "gmm", "dbscan"]


def compute_clusters(
    bundle: RepresentationBundle,
    n_clusters: int = 10,
    algorithm: Algorithm = "kmeans",
    normalize_vectors: bool = True,
    random_state: int = 42,
    **algo_kwargs,
) -> np.ndarray:
    """
    Assign cluster labels to every sample in *bundle*.

    Parameters
    ----------
    bundle:
        Source representations.
    n_clusters:
        Number of clusters (ignored for DBSCAN).
    algorithm:
        Clustering algorithm.
    normalize_vectors:
        L2-normalise vectors before clustering (recommended for cosine-based
        representations such as CLIP).
    random_state:
        Random seed for reproducibility.
    **algo_kwargs:
        Extra keyword arguments forwarded to the sklearn estimator.

    Returns
    -------
    np.ndarray
        Integer cluster assignment array of shape (N,).
    """
    X = bundle.vectors.astype(np.float64)
    if normalize_vectors:
        X = normalize(X, norm="l2")

    if algorithm == "kmeans":
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
            **algo_kwargs,
        )
        labels = model.fit_predict(X)
    elif algorithm == "gmm":
        model = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state,
            **algo_kwargs,
        )
        labels = model.fit_predict(X)
    elif algorithm == "dbscan":
        model = DBSCAN(**algo_kwargs)
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")

    return labels.astype(np.int64)


def from_clusters(
    bundle: RepresentationBundle,
    n_clusters: int = 10,
    algorithm: Algorithm = "kmeans",
    **kwargs,
) -> dict:
    """
    Compute cluster assignments and return a colorvar dict.

    Parameters
    ----------
    bundle:
        Representation bundle.
    n_clusters:
        Number of clusters.
    algorithm:
        Clustering algorithm (``"kmeans"``, ``"gmm"``, or ``"dbscan"``).
    **kwargs:
        Forwarded to :func:`compute_clusters`.

    Returns
    -------
    dict
        Colorvar dict ready for the scatter visualiser.
    """
    labels = compute_clusters(bundle, n_clusters=n_clusters, algorithm=algorithm, **kwargs)
    unique = np.unique(labels)
    n_found = len(unique)
    cmap = "tab20" if n_found <= 20 else "turbo"
    return {
        "values": labels,
        "names": np.array([f"Cluster {l}" for l in labels], dtype=str),
        "colormap": cmap,
        "label": f"K-Means clusters (k={n_clusters})" if algorithm == "kmeans" else f"{algorithm.upper()} clusters",
        "is_categorical": True,
    }
