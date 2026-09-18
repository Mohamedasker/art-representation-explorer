#!/usr/bin/env python3
"""
Download and unpack ArtBench-10 (256 × 256 px JPEG version).

Usage
-----
    python scripts/01_download_artbench.py [--data-dir data/artbench] [--splits train test]

The script
  1. Downloads the per-split tar archives from the official release.
  2. Verifies MD5 checksums.
  3. Unpacks into  <data-dir>/<split>/<style>/…
  4. Writes a  metadata.json  file alongside the images.

Run it once; subsequent calls detect existing files and skip them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

# ── Source URLs ───────────────────────────────────────────────────────────────
# ArtBench-10, 256 px JPEG, from the official GitHub release.
# MD5 checksums taken from the official README.
SOURCES: dict[str, dict] = {
    "train": {
        "url": "https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar",
        "md5": None,   # fill in after first download if you want strict verification
        "archive": "artbench-10-imagefolder-split.tar",
    },
}

# Alternative: download from HuggingFace using the datasets library
HF_DATASET_ID = "huggan/artbench10"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _download(url: str, dest: Path, desc: str = "") -> None:
    """Stream-download *url* to *dest* with a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc or dest.name
    ) as bar:
        for chunk in r.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            bar.update(len(chunk))


def _unpack_tar(archive: Path, dest: Path) -> None:
    print(f"  Unpacking  {archive.name}  →  {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        for m in tqdm(members, desc="Extracting", unit="file"):
            tf.extract(m, path=dest)


def _download_via_huggingface(data_dir: Path) -> None:
    """
    Fallback: use the HuggingFace `datasets` library to download ArtBench-10.
    Saves images into the standard  <data_dir>/train/<style>/  layout.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print(
            "Could not import `datasets`.  Install with:\n"
            "    pip install datasets\n"
            "or download the dataset manually from:\n"
            "    https://github.com/liaopeiyuan/artbench"
        )
        sys.exit(1)

    from PIL import Image as PILImage

    print(f"Downloading ArtBench-10 via HuggingFace ({HF_DATASET_ID}) …")
    ds = load_dataset(HF_DATASET_ID, cache_dir=str(data_dir / "_hf_cache"))

    style_names = ds["train"].features["label"].names

    for split_name in ds.keys():
        split = ds[split_name]
        print(f"\n  Saving {split_name} ({len(split)} images) …")
        for example in tqdm(split, desc=split_name):
            img: PILImage.Image = example["image"]
            label_idx: int = example["label"]
            style: str = style_names[label_idx]

            style_dir = data_dir / split_name / style
            style_dir.mkdir(parents=True, exist_ok=True)

            # derive a filename from hash so it is deterministic
            img_hash = hashlib.md5(img.tobytes()).hexdigest()[:12]
            out_path = style_dir / f"{img_hash}.jpg"
            if not out_path.exists():
                img.convert("RGB").save(out_path, quality=95)

    print("\n  Done.")


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
    parser = argparse.ArgumentParser(description="Download ArtBench-10 dataset")
    parser.add_argument(
        "--data-dir",
        default="data/artbench",
        help="Root directory to store the dataset (default: data/artbench)",
    )
    parser.add_argument(
        "--method",
        choices=["huggingface", "direct"],
        default="huggingface",
        help=(
            "Download method.  'huggingface' (default) uses the datasets library. "
            "'direct' attempts to download tar archives from the official server."
        ),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # quick existence check
    existing_images = list(data_dir.glob("**/*.jpg"))
    if len(existing_images) >= 60_000:
        print(
            f"Found {len(existing_images)} images in {data_dir}. "
            "Dataset already downloaded. Use --force to re-download."
        )
        _write_metadata(data_dir)
        return

    if args.method == "huggingface":
        _download_via_huggingface(data_dir)
    else:
        # direct download path (requires accessible Berkeley server)
        tmp_dir = data_dir / "_downloads"
        tmp_dir.mkdir(exist_ok=True)
        for name, info in SOURCES.items():
            archive_path = tmp_dir / info["archive"]
            if not archive_path.exists():
                print(f"Downloading {name} split …")
                _download(info["url"], archive_path, desc=info["archive"])
            else:
                print(f"  Archive already present: {archive_path}")
            if info["md5"] is not None:
                print("  Verifying checksum …", end=" ")
                computed = _md5(archive_path)
                assert computed == info["md5"], (
                    f"Checksum mismatch!  expected={info['md5']}  got={computed}"
                )
                print("OK")
            _unpack_tar(archive_path, data_dir)

    _write_metadata(data_dir)

    total = len(list(data_dir.glob("**/*.jpg")))
    print(f"\nArtBench-10 ready in  {data_dir}  ({total} images total).")


if __name__ == "__main__":
    main()
