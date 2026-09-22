#!/usr/bin/env python3
"""
Batch-visualise representations colored by class id (art style label).

This is a thin, batch-oriented sibling of 03_visualize.py: instead of
pointing it at one .npz file with ``--color label``, it scans a directory
for every representation bundle and produces a class-id-colored PNG + HTML
for each one in a single command.

Usage examples
--------------
# Visualise every representation in representations/ with UMAP:
    python scripts/05_visualize_by_class.py

# Only t-SNE, only ArtBench representations, 3D:
    python scripts/05_visualize_by_class.py \\
        --repr-dir representations \\
        --method tsne \\
        --n-components 3

# Explicit file list instead of scanning a directory:
    python scripts/05_visualize_by_class.py \\
        --repr representations/artbench_resnet50_imagenet_train.npz \\
               representations/artbench_clip_vitb32_train.npz \\
        --method umap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.coloring.label_colors import from_labels
from src.models.base_extractor import RepresentationBundle
from src.reduction.dimensionality import reduce
from src.visualization.scatter import visualize

# Bundles produced by 04_compute_color_profiles.py hold color/texture
# features, not image representations — skip them when scanning a directory.
_SKIP_SUBSTRINGS = ("color_profiles",)


def _find_representations(repr_dir: Path) -> list[Path]:
    files = sorted(repr_dir.glob("*.npz"))
    return [f for f in files if not any(s in f.name for s in _SKIP_SUBSTRINGS)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-visualise representations colored by class id",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repr-dir",
        default="representations",
        help="Directory to scan for .npz representation bundles.",
    )
    parser.add_argument(
        "--repr",
        nargs="+",
        default=None,
        metavar="NPZ",
        help="Explicit list of .npz files to visualise instead of scanning --repr-dir.",
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
        "--out-dir",
        default="visualizations",
        help="Output directory for PNG and HTML files.",
    )
    # reduction-specific
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument("--n-iter", type=int, default=1000, help="t-SNE iterations.")
    parser.add_argument("--metric", default="cosine", help="Distance metric.")
    parser.add_argument("--no-static", action="store_true", help="Skip PNG output.")
    parser.add_argument("--no-interactive", action="store_true", help="Skip HTML output.")
    args = parser.parse_args()

    if args.repr:
        repr_paths = [Path(p) for p in args.repr]
    else:
        repr_dir = Path(args.repr_dir)
        repr_paths = _find_representations(repr_dir)
        if not repr_paths:
            print(f"No representation .npz files found in {repr_dir}.")
            print("Run scripts/02_extract_features.py first.")
            return

    print(f"Found {len(repr_paths)} representation file(s) to visualise.\n")

    for repr_path in repr_paths:
        print(f"── {repr_path.stem} ──")
        bundle = RepresentationBundle.load(repr_path)
        print(f"  Loaded: {bundle}")

        proj = reduce(
            bundle.vectors,
            method=args.method,
            n_components=args.n_components,
            metric=args.metric,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            perplexity=args.perplexity,
            n_iter=args.n_iter,
        )
        print(f"  {proj}")

        colorvar = from_labels(bundle)
        stem = f"{repr_path.stem}_{args.method}_label"
        title = f"{bundle.model_name}  ·  {args.method.upper()}  ·  {colorvar['label']}"

        visualize(
            proj.coords,
            colorvar,
            paths=bundle.paths,
            title=title,
            out_dir=Path(args.out_dir),
            stem=stem,
            static=not args.no_static,
            interactive=not args.no_interactive,
        )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
