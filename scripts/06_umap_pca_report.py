#!/usr/bin/env python3
"""
Generate a UMAP embedding plot and a PCA report for one or more representations.

For each representation .npz this writes, into ``<out-dir>/<stem>/``:

  umap.png / umap.html          2D UMAP, colored by art style
  pca_scree.png                 explained variance per principal component
  pca_pairs.png                 pairwise scatter of the first --n-pcs components
  pca_class_means.png           per-class mean score on each leading component
  pca_3d.html                   interactive 3D scatter of PC1-PC3

Usage examples
--------------
# Every representation in representations/:
    python scripts/06_umap_pca_report.py

# One file, 6 PCs in the pair grid, custom UMAP settings:
    python scripts/06_umap_pca_report.py \\
        --repr representations/artbench_clip_vitb32_train.npz \\
        --n-pcs 6 --n-neighbors 30 --min-dist 0.05

# PCA report only (skip the slower UMAP):
    python scripts/06_umap_pca_report.py --skip-umap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from src.coloring.label_colors import from_labels
from src.models.base_extractor import RepresentationBundle
from src.reduction.dimensionality import umap
from src.visualization.pca_plots import plot_class_pc_heatmap, plot_pc_grid, plot_scree
from src.visualization.scatter import visualize

# Bundles produced by 04_compute_color_profiles.py hold color/texture
# features, not image representations — skip them when scanning a directory.
_SKIP_SUBSTRINGS = ("color_profiles",)


def _find_representations(repr_dir: Path) -> list[Path]:
    files = sorted(repr_dir.glob("*.npz"))
    return [f for f in files if not any(s in f.name for s in _SKIP_SUBSTRINGS)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UMAP + PCA visualisation report for image representations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repr-dir", default="representations",
                        help="Directory to scan for .npz representation bundles.")
    parser.add_argument("--repr", nargs="+", default=None, metavar="NPZ",
                        help="Explicit list of .npz files instead of scanning --repr-dir.")
    parser.add_argument("--out-dir", default="visualizations/umap_pca",
                        help="Output root; one subdirectory is created per representation.")
    parser.add_argument("--n-pcs", type=int, default=5,
                        help="Number of leading PCs in the pair grid.")
    parser.add_argument("--n-scree", type=int, default=20,
                        help="Number of components shown in the scree plot and class-mean heatmap.")
    parser.add_argument("--skip-umap", action="store_true", help="Skip the UMAP plot.")
    parser.add_argument("--skip-pca", action="store_true", help="Skip the PCA plots.")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--metric", default="cosine", help="UMAP distance metric.")
    parser.add_argument("--no-interactive", action="store_true", help="Skip HTML output.")
    args = parser.parse_args()

    if args.repr:
        repr_paths = [Path(p) for p in args.repr]
    else:
        repr_paths = _find_representations(Path(args.repr_dir))
        if not repr_paths:
            print(f"No representation .npz files found in {args.repr_dir}.")
            print("Run scripts/02_extract_features.py first.")
            return

    print(f"Found {len(repr_paths)} representation file(s).\n")

    for repr_path in repr_paths:
        print(f"── {repr_path.stem} ──")
        bundle = RepresentationBundle.load(repr_path)
        print(f"  Loaded: {bundle}")
        out_dir = Path(args.out_dir) / repr_path.stem
        colorvar = from_labels(bundle)

        if not args.skip_umap:
            proj = umap(bundle.vectors, n_components=2, n_neighbors=args.n_neighbors,
                        min_dist=args.min_dist, metric=args.metric)
            visualize(
                proj.coords, colorvar, paths=bundle.paths,
                title=f"{bundle.model_name}  ·  UMAP  ·  {colorvar['label']}",
                out_dir=out_dir, stem="umap", interactive=not args.no_interactive,
            )
            print(f"  UMAP -> {out_dir / 'umap.png'}")

        if not args.skip_pca:
            n_fit = min(max(args.n_scree, args.n_pcs, 3), bundle.dim, bundle.n_samples)
            # L2-normalise rows so PCA sees the same geometry UMAP does.
            X = normalize(bundle.vectors, norm="l2")
            model = PCA(n_components=n_fit, random_state=42)
            scores = model.fit_transform(X).astype(np.float32)
            evr = model.explained_variance_ratio_
            print(f"  PCA: first 3 PCs explain {evr[:3].sum() * 100:.1f}%, "
                  f"first {n_fit} explain {evr.sum() * 100:.1f}%")

            name = bundle.model_name
            plot_scree(evr, f"{name}  ·  PCA explained variance", out_dir / "pca_scree.png")
            plot_pc_grid(scores, bundle.labels, bundle.label_names, evr,
                         f"{name}  ·  leading principal components",
                         out_dir / "pca_pairs.png", n_pcs=args.n_pcs)
            plot_class_pc_heatmap(scores, bundle.labels, bundle.label_names, evr,
                                  f"{name}  ·  class means on principal components",
                                  out_dir / "pca_class_means.png", n_pcs=n_fit)
            if not args.no_interactive:
                visualize(
                    scores[:, :3], colorvar, paths=bundle.paths,
                    title=f"{name}  ·  PCA (PC1–PC3)  ·  {colorvar['label']}",
                    out_dir=out_dir, stem="pca_3d", static=False,
                )
            print(f"  PCA  -> {out_dir}/pca_*.png")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
