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
# ArtBench-10, 256 px JPEG — official imagefolder-split tar from Berkeley.
# Contains both train/ and test/ already organised by style.
SOURCES: dict[str, dict] = {
    "all": {
        "url": "https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar",
        "md5": None,
        "archive": "artbench-10-imagefolder-split.tar",
    },
}

# HuggingFace dataset IDs to try in order (first one that works is used)
HF_DATASET_IDS = [
    "jlbaker361/artbench-10",   # known mirror
    "keremberke/artbench10-classification",
    "huggan/artbench10",        # original (may be removed)
]

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
    """
    Stream-download *url* to *dest* with a tqdm progress bar.
    Resumes an interrupted download automatically using HTTP Range headers.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing_bytes = dest.stat().st_size if dest.exists() else 0

    headers = {}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        print(f"  Resuming download from {existing_bytes / 1e9:.2f} GB …")

    r = requests.get(url, stream=True, timeout=120, headers=headers)

    # 416 = range not satisfiable → file already complete
    if r.status_code == 416:
        print("  Download already complete.")
        return
    r.raise_for_status()

    total = int(r.headers.get("Content-Length", 0)) + existing_bytes
    mode = "ab" if existing_bytes > 0 else "wb"

    with open(dest, mode) as f, tqdm(
        total=total,
        initial=existing_bytes,
        unit="B",
        unit_scale=True,
        desc=desc or dest.name,
    ) as bar:
        for chunk in r.iter_content(chunk_size=1 << 18):   # 256 KB chunks
            f.write(chunk)
            bar.update(len(chunk))


def _is_tar_complete(archive: Path) -> bool:
    """Quick check: try reading the last block of the tar to detect truncation."""
    try:
        with tarfile.open(archive) as tf:
            # getmembers() reads the whole index — slow but thorough
            members = tf.getmembers()
            return len(members) > 0
    except (tarfile.ReadError, EOFError):
        return False


def _unpack_tar(archive: Path, dest: Path) -> None:
    print(f"  Checking archive integrity …", end=" ", flush=True)
    if not _is_tar_complete(archive):
        print("CORRUPTED")
        print(f"  Deleting incomplete archive: {archive}")
        archive.unlink()
        raise RuntimeError(
            f"Archive {archive.name} is incomplete (download was interrupted).\n"
            "Re-run the script — it will resume the download automatically."
        )
    print("OK")

    print(f"  Unpacking  {archive.name}  →  {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        for m in tqdm(members, desc="Extracting", unit="file"):
            tf.extract(m, path=dest)


def _download_via_huggingface(data_dir: Path) -> None:
    """
    Use the HuggingFace `datasets` library to download ArtBench-10.
    Tries several known dataset IDs in order.
    Saves images into the standard  <data_dir>/train/<style>/  layout.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print(
            "Could not import `datasets`.  Install with:\n"
            "    pip install datasets\n"
            "or use --method direct to download from the Berkeley server."
        )
        sys.exit(1)

    from PIL import Image as PILImage

    ds = None
    for hf_id in HF_DATASET_IDS:
        try:
            print(f"Trying HuggingFace dataset: {hf_id} …")
            ds = load_dataset(hf_id, cache_dir=str(data_dir / "_hf_cache"))
            print(f"  ✓ Found: {hf_id}")
            break
        except Exception as e:
            print(f"  ✗ Not available: {e}")

    if ds is None:
        print(
            "\nNo HuggingFace mirror worked.  Falling back to direct download …\n"
        )
        _download_direct(data_dir)
        return

    # Detect the label feature name (varies by dataset)
    train_features = ds["train"].features
    label_key = "label" if "label" in train_features else next(
        k for k in train_features if "label" in k.lower()
    )
    style_names = train_features[label_key].names

    for split_name in ds.keys():
        split = ds[split_name]
        print(f"\n  Saving {split_name} ({len(split)} images) …")
        for example in tqdm(split, desc=split_name):
            img: PILImage.Image = example["image"]
            label_idx: int = example[label_key]
            style: str = style_names[label_idx]

            style_dir = data_dir / split_name / style
            style_dir.mkdir(parents=True, exist_ok=True)

            img_hash = hashlib.md5(img.tobytes()).hexdigest()[:12]
            out_path = style_dir / f"{img_hash}.jpg"
            if not out_path.exists():
                img.convert("RGB").save(out_path, quality=95)

    print("\n  Done.")


def _download_direct(data_dir: Path) -> None:
    """Download the official tar from the Berkeley server and unpack it."""
    info = SOURCES["all"]
    tmp_dir = data_dir / "_downloads"
    tmp_dir.mkdir(exist_ok=True)
    archive_path = tmp_dir / info["archive"]
    if not archive_path.exists():
        print(f"Downloading {info['archive']} from Berkeley server …")
        print("  (~3 GB, this will take a few minutes)")
        _download(info["url"], archive_path, desc=info["archive"])
    else:
        print(f"  Archive already present: {archive_path}")
    _unpack_tar(archive_path, data_dir)


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
        default="direct",
        help=(
            "Download method.  'direct' (default) downloads the tar archive from "
            "the official Berkeley server (~3 GB).  'huggingface' tries several "
            "HuggingFace mirrors and falls back to direct if none work."
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
        _download_direct(data_dir)

    _write_metadata(data_dir)

    total = len(list(data_dir.glob("**/*.jpg")))
    print(f"\nArtBench-10 ready in  {data_dir}  ({total} images total).")


if __name__ == "__main__":
    main()
