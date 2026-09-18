#!/usr/bin/env python3
"""
Download and unpack WikiArt (via the HuggingFace ``huggan/wikiart`` mirror).

Usage
-----
    python scripts/01_download_wikiart.py [--data-dir data/wikiart] [--test-fraction 0.1]

The script
  1. Streams the dataset from HuggingFace (tries a few known mirrors in order).
  2. Normalises each image's style label to a lower_snake_case slug.
  3. Deterministically assigns each image to train/test (hash of its HF index,
     so re-running is reproducible without needing to store extra state).
  4. Saves images into  <data-dir>/<split>/<style>/…
  5. Writes a  metadata.json  file alongside the images.

WikiArt has no official train/test split (unlike ArtBench-10), so one is
synthesised here at a fixed ratio, evenly across styles.

Run it once; subsequent calls detect existing files and skip them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tqdm import tqdm

# HuggingFace dataset IDs to try in order (first one that works is used).
# Both mirror the same underlying 81,444-image WikiArt scrape.
HF_DATASET_IDS = [
    "huggan/wikiart",
    "williamberman/wikiart",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_style_name(raw: str) -> str:
    """Map a raw HF style label (e.g. ``"Art_Nouveau_Modern"``) to a
    lower_snake_case slug (e.g. ``"art_nouveau_modern"``)."""
    slug = "".join(c.lower() if c.isalnum() else "_" for c in raw)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _assign_split(key: str, test_fraction: float) -> str:
    """Deterministically bucket an image into train/test from a stable key."""
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    return "test" if (h % 1000) < int(test_fraction * 1000) else "train"


def _download_via_huggingface(data_dir: Path, test_fraction: float) -> None:
    """
    Use the HuggingFace `datasets` library to download WikiArt.
    Tries several known dataset IDs in order.
    Saves images into  <data_dir>/<split>/<style>/  and synthesises the
    train/test split since the upstream dataset ships only one split.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise SystemExit(
            "Could not import `datasets`.  Install with:\n"
            "    pip install datasets"
        )

    ds = None
    used_id = None
    for hf_id in HF_DATASET_IDS:
        try:
            print(f"Trying HuggingFace dataset: {hf_id} …")
            ds = load_dataset(hf_id, cache_dir=str(data_dir / "_hf_cache"))
            used_id = hf_id
            print(f"  ✓ Found: {hf_id}")
            break
        except Exception as e:
            print(f"  ✗ Not available: {e}")

    if ds is None:
        raise SystemExit(
            "No HuggingFace mirror worked. Tried: " + ", ".join(HF_DATASET_IDS)
        )

    # WikiArt mirrors ship a single "train" split; concatenate whatever splits
    # exist so the logic below is robust either way.
    split_names = list(ds.keys())
    features = ds[split_names[0]].features
    style_key = "style" if "style" in features else next(
        k for k in features if "style" in k.lower()
    )
    style_names = features[style_key].names

    n_written = 0
    n_skipped_existing = 0
    for split_name in split_names:
        split = ds[split_name]
        print(f"\n  Processing {used_id}:{split_name} ({len(split)} images) …")
        for i, example in enumerate(tqdm(split, desc=split_name)):
            style_idx = example[style_key]
            if style_idx is None or style_idx < 0:
                continue  # "Unknown style" or missing label
            style = _normalize_style_name(style_names[style_idx])

            out_split = _assign_split(f"{used_id}:{split_name}:{i}", test_fraction)
            style_dir = data_dir / out_split / style
            style_dir.mkdir(parents=True, exist_ok=True)

            img = example["image"]
            img_hash = hashlib.md5(f"{used_id}:{split_name}:{i}".encode()).hexdigest()[:16]
            out_path = style_dir / f"{img_hash}.jpg"
            if out_path.exists():
                n_skipped_existing += 1
                continue
            img.convert("RGB").save(out_path, quality=95)
            n_written += 1

    print(f"\n  Done. {n_written} images written, {n_skipped_existing} already present.")


def _write_metadata(data_dir: Path) -> None:
    """Write a JSON manifest of all images present."""
    manifest: dict = {"splits": {}}
    for split_dir in sorted(data_dir.iterdir()):
        if not split_dir.is_dir() or split_dir.name.startswith("_"):
            continue
        split_name = split_dir.name
        manifest["splits"][split_name] = {}
        for style_dir in sorted(split_dir.iterdir()):
            if not style_dir.is_dir():
                continue
            imgs = sorted(style_dir.glob("*.jpg"))
            manifest["splits"][split_name][style_dir.name] = [
                str(p.relative_to(data_dir)) for p in imgs
            ]
    meta_path = data_dir / "metadata.json"
    meta_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest written → {meta_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download WikiArt dataset")
    parser.add_argument(
        "--data-dir",
        default="data/wikiart",
        help="Root directory to store the dataset (default: data/wikiart)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Fraction of images (per style) held out as the test split.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run the download/export even if images already exist.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    existing_images = list(data_dir.glob("**/*.jpg"))
    if len(existing_images) > 0 and not args.force:
        print(
            f"Found {len(existing_images)} images in {data_dir}. "
            "Dataset already downloaded. Use --force to re-run."
        )
        _write_metadata(data_dir)
        return

    _download_via_huggingface(data_dir, test_fraction=args.test_fraction)
    _write_metadata(data_dir)

    total = len(list(data_dir.glob("**/*.jpg")))
    print(f"\nWikiArt ready in  {data_dir}  ({total} images total).")


if __name__ == "__main__":
    main()
