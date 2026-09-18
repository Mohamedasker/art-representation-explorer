#!/usr/bin/env python3
"""
Visualise image representations with PCA, UMAP, or t-SNE.

The script accepts any .npz representation file (produced by
02_extract_features.py) and any pre-computed color variable file, then
outputs both a static PNG and an interactive HTML scatter plot.

Usage examples
--------------
# UMAP of CLIP features, coloured by art style:
    python scripts/03_visualize.py \\
        --repr representations/clip_vitb32_train.npz \\
        --color label \\
        --method umap \\
        --out-dir visualizations/clip_umap_label

# t-SNE of ResNet features, coloured by colorfulness:
    python scripts/03_visualize.py \\
        --repr representations/resnet50_imagenet_train.npz \\
        --color colorfulness \\
        --color-file representations/color_profiles_train.npz \\
        --method tsne \\
        --out-dir visualizations/resnet_tsne_colorfulness

# PCA in 3D, coloured by cluster assignment:
    python scripts/03_visualize.py \\
        --repr representations/dinov2_vitb14_train.npz \\
        --color cluster \\
        --n-clusters 15 \\
        --method pca \\
        --n-components 3

# Compare multiple representations side-by-side:
    python scripts/03_visualize.py \\
        --repr representations/resnet50_imagenet_train.npz \\
                representations/clip_vitb32_train.npz \\
        --color label \\
        --method umap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.base_extractor import RepresentationBundle
from src.reduction.dimensionality import reduce
from src.visualization.scatter import visualize


# ── Color variable factory ────────────────────────────────────────────────────

def _load_colorvar(
    kind: str,
    bundle: RepresentationBundle,
    color_file: Path | None,
    n_clusters: int,
    cluster_algo: str,
    clip_model: str,
) -> dict:
    if kind == "label":
        from src.coloring.label_colors import from_labels
        return from_labels(bundle)

    if kind in ("mean_hue", "mean_saturation", "mean_brightness", "colorfulness"):
        from src.coloring import color_profile as cp
        if color_file and color_file.exists():
            import numpy as np
            raw = np.load(color_file, allow_pickle=False)
            features = {k: raw[k] for k in raw.files}
        else:
            features = cp.compute_color_features(bundle.paths)
        fn = {
            "mean_hue": cp.from_hue,
            "mean_saturation": cp.from_saturation,
            "mean_brightness": cp.from_brightness,
            "colorfulness": cp.from_colorfulness,
        }[kind]
        return fn(features)

    if kind == "cluster":
        from src.coloring.cluster_colors import from_clusters
        return from_clusters(bundle, n_clusters=n_clusters, algorithm=cluster_algo)

    if kind in ("lbp_entropy", "contrast", "dissimilarity", "homogeneity", "energy"):
        from src.coloring import texture_colors as tx
        if color_file and color_file.exists():
            import numpy as np
            raw = np.load(color_file, allow_pickle=False)
            features = {k: raw[k] for k in raw.files}
        else:
            features = tx.compute_texture_features(bundle.paths)
        fn = {
            "lbp_entropy":   tx.from_lbp_entropy,
            "contrast":      tx.from_contrast,
            "homogeneity":   tx.from_homogeneity,
            "energy":        tx.from_energy,
        }.get(kind, tx.from_lbp_entropy)
        return fn(features)

    if kind in ("semantic_subject", "semantic_mood", "semantic_period"):
        from src.coloring import semantic_colors as sc
        fn = {
            "semantic_subject": sc.from_subject,
            "semantic_mood":    sc.from_mood,
            "semantic_period":  sc.from_period,
        }[kind]
        return fn(bundle, clip_model_name=clip_model)

    raise ValueError(
        f"Unknown --color value: {kind!r}.  "
        "Choose from: label | mean_hue | mean_saturation | mean_brightness | "
        "colorfulness | cluster | lbp_entropy | contrast | homogeneity | energy | "
        "semantic_subject | semantic_mood | semantic_period"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise image representations with PCA / UMAP / t-SNE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repr",
        nargs="+",
        required=True,
        metavar="NPZ",
        help="One or more representation .npz files.",
    )
    parser.add_argument(
        "--color",
        default="label",
        help=(
            "Color variable.  Options: label | mean_hue | mean_saturation | "
            "mean_brightness | colorfulness | cluster | lbp_entropy | contrast | "
            "homogeneity | energy | semantic_subject | semantic_mood | semantic_period"
        ),
    )
    parser.add_argument(
        "--color-file",
        default=None,
        metavar="NPZ",
        help=(
            "Pre-computed color profile / texture .npz "
            "(produced by 04_compute_color_profiles.py).  "
            "If not given, features are computed on the fly."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["pca", "umap", "tsne"],
        default="umap",
        help="Dimensionality reduction method.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        choices=[2, 3],
        default=2,
        help="Output dimensionality (2 or 3).",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Number of clusters (only used when --color cluster).",
    )
    parser.add_argument(
        "--cluster-algo",
        choices=["kmeans", "gmm", "dbscan"],
        default="kmeans",
        help="Clustering algorithm (only used when --color cluster).",
    )
    parser.add_argument(
        "--out-dir",
        default="visualizations",
        help="Output directory for PNG and HTML files.",
    )
    parser.add_argument(
        "--clip-model",
        default="ViT-B-32",
        help="CLIP model for semantic coloring (only used with --color semantic_*).",
    )
    # reduction-specific
    parser.add_argument("--n-neighbors",  type=int,   default=15,   help="UMAP n_neighbors.")
    parser.add_argument("--min-dist",     type=float, default=0.1,  help="UMAP min_dist.")
    parser.add_argument("--perplexity",   type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument("--n-iter",       type=int,   default=1000, help="t-SNE iterations.")
    parser.add_argument("--metric",       default="cosine",         help="Distance metric.")
    parser.add_argument("--no-static",    action="store_true",      help="Skip PNG output.")
    parser.add_argument("--no-interactive", action="store_true",    help="Skip HTML output.")
    args = parser.parse_args()

    color_file = Path(args.color_file) if args.color_file else None

    for repr_path in args.repr:
        repr_path = Path(repr_path)
        print(f"\n── {repr_path.stem} ──")
        bundle = RepresentationBundle.load(repr_path)
        print(f"  Loaded: {bundle}")

        # ── dimensionality reduction ──
        reduce_kwargs: dict = {"metric": args.metric}
        if args.method == "umap":
            reduce_kwargs.update(n_neighbors=args.n_neighbors, min_dist=args.min_dist)
        elif args.method == "tsne":
            reduce_kwargs.update(perplexity=args.perplexity, n_iter=args.n_iter)

        proj = reduce(
            bundle.vectors,
            method=args.method,
            n_components=args.n_components,
            **reduce_kwargs,
        )
        print(f"  {proj}")

        # ── color variable ──
        colorvar = _load_colorvar(
            kind=args.color,
            bundle=bundle,
            color_file=color_file,
            n_clusters=args.n_clusters,
            cluster_algo=args.cluster_algo,
            clip_model=args.clip_model,
        )

        # ── output ──
        out_dir = Path(args.out_dir)
        stem = f"{repr_path.stem}_{args.method}_{args.color}"

        title = (
            f"{bundle.model_name}  ·  {args.method.upper()}  ·  {colorvar['label']}"
        )

        visualize(
            proj.coords,
            colorvar,
            paths=bundle.paths,
            title=title,
            out_dir=out_dir,
            stem=stem,
            static=not args.no_static,
            interactive=not args.no_interactive,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
