"""
Static matplotlib plots that summarise the leading principal components.

Three views, all driven by the same PCA fit:
  * scree plot        — per-component and cumulative explained variance.
  * PC pair grid      — scatter of every pair among the first k PCs, with
                        per-class histograms on the diagonal.
  * class-mean heatmap — mean score of each class along each PC (z-scored),
                        showing which classes each component separates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _class_colors(labels: np.ndarray):
    """Return ({class id: rgba}, unique class ids) with a palette sized to the class count."""
    import matplotlib.pyplot as plt

    unique = np.unique(labels)
    cmap_name = "tab10" if len(unique) <= 10 else "tab20" if len(unique) <= 20 else "turbo"
    cmap = plt.get_cmap(cmap_name, len(unique))
    return {int(u): cmap(i) for i, u in enumerate(unique)}, unique


def _class_name(labels: np.ndarray, names: np.ndarray, cls: int) -> str:
    return str(names[labels == cls][0])


def plot_scree(
    explained_variance_ratio: np.ndarray,
    title: str,
    out_path: str | Path,
    dpi: int = 150,
) -> Path:
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ratio = np.asarray(explained_variance_ratio) * 100
    idx = np.arange(1, len(ratio) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(idx, ratio, color="tab:blue", alpha=0.8, label="Per component")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_xticks(idx)

    ax2 = ax.twinx()
    ax2.plot(idx, np.cumsum(ratio), color="tab:red", marker="o", label="Cumulative")
    ax2.set_ylabel("Cumulative explained variance (%)")
    ax2.set_ylim(0, 100)

    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(handles=handles, loc="center right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_pc_grid(
    scores: np.ndarray,
    labels: np.ndarray,
    label_names: np.ndarray,
    explained_variance_ratio: np.ndarray,
    title: str,
    out_path: str | Path,
    n_pcs: int = 5,
    point_size: float = 3.0,
    alpha: float = 0.5,
    dpi: int = 130,
) -> Path:
    """Pairwise scatter of the first ``n_pcs`` components, colored by class."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pcs = min(n_pcs, scores.shape[1])
    colors, unique = _class_colors(labels)

    fig, axes = plt.subplots(n_pcs, n_pcs, figsize=(2.6 * n_pcs, 2.6 * n_pcs), squeeze=False)
    for i in range(n_pcs):          # row  -> y-axis PC
        for j in range(n_pcs):      # col  -> x-axis PC
            ax = axes[i, j]
            if i == j:
                for u in unique:
                    ax.hist(scores[labels == u, i], bins=40, histtype="step",
                            color=colors[int(u)], linewidth=1.0)
                ax.set_yticks([])
            else:
                for u in unique:
                    m = labels == u
                    ax.scatter(scores[m, j], scores[m, i], s=point_size, alpha=alpha,
                               color=colors[int(u)], linewidths=0, rasterized=True)
            if i == n_pcs - 1:
                ax.set_xlabel(f"PC{j + 1} ({explained_variance_ratio[j] * 100:.1f}%)")
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(f"PC{i + 1} ({explained_variance_ratio[i] * 100:.1f}%)")
            elif j != 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors[int(u)],
                   label=_class_name(labels, label_names, int(u)), markersize=6)
        for u in unique
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=8, title="Art style")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_class_pc_heatmap(
    scores: np.ndarray,
    labels: np.ndarray,
    label_names: np.ndarray,
    explained_variance_ratio: np.ndarray,
    title: str,
    out_path: str | Path,
    n_pcs: int = 10,
    dpi: int = 150,
) -> Path:
    """Heatmap of per-class mean PC score, in units of each PC's standard deviation."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pcs = min(n_pcs, scores.shape[1])
    unique = np.unique(labels)
    means = np.stack([scores[labels == u, :n_pcs].mean(axis=0) for u in unique])
    means = means / scores[:, :n_pcs].std(axis=0)
    class_labels = [_class_name(labels, label_names, int(u)) for u in unique]

    vmax = np.abs(means).max()
    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * n_pcs, 1.0 + 0.4 * len(unique)))
    im = ax.imshow(means, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_pcs))
    ax.set_xticklabels([f"PC{k + 1}\n{explained_variance_ratio[k] * 100:.1f}%" for k in range(n_pcs)],
                       fontsize=8)
    ax.set_yticks(range(len(unique)))
    ax.set_yticklabels(class_labels, fontsize=8)
    fig.colorbar(im, ax=ax, label="Class mean (std units)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
