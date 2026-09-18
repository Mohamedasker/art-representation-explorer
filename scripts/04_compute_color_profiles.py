#!/usr/bin/env python3
"""
Pre-compute color and texture profiles for an ArtBench-10 split.

These profiles (hue, saturation, brightness, colorfulness, LBP entropy,
GLCM features) are saved as an .npz file that can be reused across many
visualisation runs without re-reading the full image set each time.

Usage
-----
    python scripts/04_compute_color_profiles.py \\
        --split train \\
        --out representations/color_profiles_train.npz

    python scripts/04_compute_color_profiles.py \\
        --split test \\
        --out representations/color_profiles_test.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.artbench import ArtBenchDataset
from src.coloring.color_profile import compute_color_features
from src.coloring.texture_colors import compute_texture_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute color + texture profiles for ArtBench-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default="data/artbench",
        help="Root directory of the ArtBench-10 dataset."
    )
    parser.add_argument(
        "--split", choices=["train", "test", "both"], default="train",
        help="Dataset split to process.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output .npz path.  Derived from --split if not given.",
    )
    parser.add_argument(
        "--color-size", type=int, default=64,
        help="Thumbnail size for color feature computation.",
    )
    parser.add_argument(
        "--texture-size", type=int, default=128,
        help="Thumbnail size for texture feature computation.",
    )
    args = parser.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]

    for split in splits:
        print(f"\n── {split} split ──")
        dataset = ArtBenchDataset(
            root=args.data_dir,
            split=split,
            return_path=False,
        )
        paths = np.array(dataset.get_paths(), dtype=str)
        labels = dataset.get_labels()
        label_names = np.array(dataset.get_style_names(), dtype=str)

        print(f"  {len(paths)} images")

        print("\n  [1/2] Color profiles …")
        color_feats = compute_color_features(paths, size=args.color_size)

        print("\n  [2/2] Texture features …")
        texture_feats = compute_texture_features(paths, size=args.texture_size)

        # merge into one dict
        all_feats = {
            **color_feats,
            **texture_feats,
            "paths": paths,
            "labels": labels,
            "label_names": label_names,
        }

        out_path = Path(args.out or f"representations/color_profiles_{split}.npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **all_feats)
        print(f"\n  Saved → {out_path}")
        print(f"  Keys: {list(all_feats.keys())}")


if __name__ == "__main__":
    main()
