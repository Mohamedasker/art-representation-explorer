#!/usr/bin/env python3
"""
Pack the explorer into zip files, one per model type, to download everything or to upload it week by week.

    00_base.zip                  index.html, models.html, shared/ (thumbnails, per-image meta), lecture1/, and the shared
                                 part of the feature-map viewer (featuremaps/index.html, index.json, inputs/): upload once
    NN_<model dir>.zip           the explorer pages of one model type incl. its feature maps (<dir>/featuremaps/<model>/)
    NN_<model dir>_embeddings.zip   the embeddings of this model type (representations/<dataset>_<weights>_<split>.npz),
                                 not needed to view the pages
    99_representations_common.zip   colour profiles etc. that belong to no model
    MANIFEST.md                  what is in which zip, sizes

Paths inside the explorer zips are relative to the explorer root, so unzipping all of them (or any subset together with
00_base.zip) into one folder gives a working copy of the explorer: ``unzip -o '*.zip' -d explorer``.  The embeddings zips
unpack to ``representations/``.  ``cache/`` (build intermediates) is not packed.  Zips larger than --max-gb are split into
``..._part1.zip``, ``..._part2.zip`` ...  jpg / png / npz are stored, the rest deflated (they are compressed already).

    python scripts/24_package_explorer.py                  # -> visualizations/export/
    python scripts/24_package_explorer.py --no-embeddings
    python scripts/24_package_explorer.py --only mae simclr
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.visualization.variants import model_dir

ORDER = ["resnet50", "autoencoder", "mae", "simclr", "dino", "clip", "vit", "ar", "gan"]      # README table order
START_HERE = """Explorer export: how to view it

1. Unzip 00_base.zip and the model zips (01_..., 02_..., ...) into ONE folder (the zips are meant to be merged).
   The *_embeddings.zip files are raw data (representations/*.npz) and are not needed to view the pages.
2. Do NOT open index.html by double-click for the feature-map viewer: browsers block the local files it loads.
   Start the small web server in that folder (needs Python 3):

       python serve.py

   and open http://127.0.0.1:8877/  (feature-map viewer: http://127.0.0.1:8877/featuremaps/).
3. Models whose zip you did not unpack are simply missing (their pages, links and feature maps).
"""
STORED = {".jpg", ".jpeg", ".png", ".npz", ".gz", ".zip"}
EMB = re.compile(r"^(?:artbench|wikiart)_(?!color_profiles)(.+)_(?:all|train|test)\.npz$")


def files_under(base: Path, root: Path) -> list[tuple[Path, str]]:
    return [(f, f.relative_to(root).as_posix()) for f in sorted(base.rglob("*")) if f.is_file()]


def chunks(files: list[tuple[Path, str]], max_bytes: int) -> list[list[tuple[Path, str]]]:
    out, cur, size = [], [], 0
    for f in files:
        s = f[0].stat().st_size
        if cur and size + s > max_bytes:
            out.append(cur)
            cur, size = [], 0
        cur.append(f)
        size += s
    return out + [cur] if cur else out


def write_zip(job: tuple[str, list[tuple[Path, str]]]) -> tuple[str, int, int]:
    path, files = job
    tmp = path + ".part"
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        for f, arc in files:
            ct = zipfile.ZIP_STORED if f.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
            z.write(f, arc, compress_type=ct)
    Path(tmp).rename(path)
    return path, len(files), Path(path).stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--explorer", default="visualizations/explorer")
    ap.add_argument("--representations", default="representations")
    ap.add_argument("--out", default="visualizations/export")
    ap.add_argument("--max-gb", type=float, default=5.0, help="split zips larger than this (before compression)")
    ap.add_argument("--no-embeddings", action="store_true")
    ap.add_argument("--only", nargs="+", help="only these model directories (and 'base' for 00_base.zip)")
    ap.add_argument("--jobs", type=int, default=8, help="zips written in parallel")
    a = ap.parse_args()
    root, reps, out = Path(a.explorer), Path(a.representations), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    max_bytes = int(a.max_gb * 1024 ** 3)

    dirs = [d.name for d in root.iterdir() if d.is_dir() and d.name not in ("shared", "lecture1", "featuremaps", "cache")]
    dirs = sorted(dirs, key=lambda d: (ORDER.index(d) if d in ORDER else len(ORDER), d))
    numbers = {d: i for i, d in enumerate(dirs, 1)}                  # zip numbers do not depend on --only
    if a.only:
        dirs = [d for d in dirs if d in a.only]
    jobs: dict[str, list[tuple[Path, str]]] = {}              # zip name (without .zip) -> files

    if not a.only or "base" in a.only:
        (out / "START_HERE.txt").write_text(START_HERE)
        base = [(out / "START_HERE.txt", "START_HERE.txt"), (REPO_ROOT / "scripts" / "serve_explorer.py", "serve.py")]
        base += [(root / n, n) for n in ("index.html", "models.html") if (root / n).exists()]
        base += [x for sub in ("shared", "lecture1", "featuremaps/inputs") if (root / sub).is_dir() for x in files_under(root / sub, root)]
        base += [(root / "featuremaps" / n, f"featuremaps/{n}") for n in ("index.html", "index.json") if (root / "featuremaps" / n).exists()]
        jobs["00_base"] = base
    for d in dirs:
        jobs[f"{numbers[d]:02d}_{d}"] = files_under(root / d, root)

    if not a.no_embeddings and reps.is_dir():
        by_dir: dict[str, list[tuple[Path, str]]] = {}
        for f in sorted(reps.glob("*.npz")):
            m = EMB.match(f.name)
            d = model_dir(f"{f.name.split('_')[0]}_{m[1]}_all") if m else "common"
            by_dir.setdefault(d, []).append((f, f"{reps.name}/{f.name}"))
        for d, files in by_dir.items():
            if a.only and d not in a.only:
                continue
            if d == "common":
                jobs["99_representations_common"] = files
            elif d in dirs:
                jobs[f"{numbers[d]:02d}_{d}_embeddings"] = files
            else:
                jobs[f"98_{d}_embeddings"] = files

    todo: list[tuple[str, list[tuple[Path, str]]]] = []
    for name, files in sorted(jobs.items()):
        parts = chunks(files, max_bytes)
        for k, part in enumerate(parts, 1):
            todo.append((str(out / (f"{name}.zip" if len(parts) == 1 else f"{name}_part{k}.zip")), part))
    print(f"{len(todo)} zips for {sum(len(f) for _p, f in todo)} files -> {out}", flush=True)

    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for path, n, size in ex.map(write_zip, todo):
            print(f"  {Path(path).name}: {n} files, {size / 1e9:.2f} GB", flush=True)
            rows.append((Path(path).name, n, size))

    lines = ["# Explorer export", "",
             "Unzip everything (or the `00_base.zip` + any subset of the model zips) into one folder: "
             "`unzip -o '*.zip' -d explorer`. Open `explorer/index.html` through a web server "
             "(`python -m http.server` in that folder).", "",
             "| zip | files | size | content |", "|---|---|---|---|"]
    have = {r[0] for r in rows}                      # zips of earlier runs stay in the manifest
    for z in sorted(out.glob("*.zip")):
        if z.name not in have:
            with zipfile.ZipFile(z) as zf:
                rows.append((z.name, len(zf.namelist()), z.stat().st_size))
    for name, n, size in sorted(rows):
        kind = ("shared files: overview, thumbnails, feature-map viewer + paintings, lecture 1" if name.startswith("00_") else
                "embeddings (.npz) for `representations/`" if "embeddings" in name or name.startswith("99_") else
                "explorer pages + feature maps of one model type")
        lines.append(f"| {name} | {n} | {size / 1e9:.2f} GB | {kind} |")
    lines.append(f"\nTotal: {sum(s for _n, _c, s in rows) / 1e9:.1f} GB")
    (out / "MANIFEST.md").write_text("\n".join(lines) + "\n")
    print("wrote", out / "MANIFEST.md")


if __name__ == "__main__":
    main()
