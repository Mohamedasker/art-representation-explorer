#!/usr/bin/env python3
"""
Build the page for the first lecture (visualizations/explorer/lecture1/index.html): nearest neighbours, interpolation, PCA, t-SNE
and UMAP on a toy data set of fish (ellipses described by length and thickness, optionally also tail length), shown like
the embedding explorer plus step-by-step walk-throughs of the three projection methods.  See src/visualization/lecture1/.

    python scripts/20_build_lecture1.py
    python scripts/20_build_lecture1.py --n-fish 200 --seed 3 --out visualizations/explorer/lecture1/index.html
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.visualization.lecture1 import build_page


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_ROOT / "visualizations/explorer/lecture1/index.html"))
    ap.add_argument("--n-fish", type=int, default=150, help="number of fish")
    ap.add_argument("--seed", type=int, default=11, help="seed of the fish generator")
    a = ap.parse_args()
    print(f"written {build_page(a.out, a.n_fish, a.seed)}")


if __name__ == "__main__":
    main()
