#!/usr/bin/env python3
"""
Flag images that contain nudity, for every image of ArtBench-10 and WikiArt.

Runs two detectors (NudeNet, CLIP ViT-L/14 zero-shot) on the explorer thumbnails and combines them with
the dataset's own labels (WikiArt genre ``nude_painting``; ArtBench titles that say nude/naked/...).  The raw
scores go to ``data/nudity/scores_<dataset>.npz`` and the flags to ``data/nudity/flags.npz``, which
``scripts/08_build_explorer.py`` and ``scripts/rerender_explorer.py`` read to hide flagged images in the explorer
unless the user ticks "show explicit images".

    python scripts/15_flag_nudity.py                    # detectors + flags for both datasets
    python scripts/15_flag_nudity.py --flags-only       # only re-apply thresholds to the saved scores
    python scripts/15_flag_nudity.py --flags-only --clip-thr 0.4 --nn-thr 0.3
    python scripts/15_flag_nudity.py --flags-only --sheets /tmp/sheets   # contact sheets per score band (calibration)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "32")

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.metadata.nudity import (DEFAULT_THRESHOLDS, FLAGS_FILE, build_flags, clip_nudity_scores, image_key,
                                 nudenet_scores)

OUT = FLAGS_FILE.parent
# ArtBench file names are "<artist>_<title>.jpg"; only unambiguous words count (not "bathers" / "venus").
_TITLE_NUDE = re.compile(r"(?<![a-z])(nude|nudes|naked|nudity|odalisque|akt|desnudo|nudo|nue)(?![a-z])")


def thumb_name(path: str) -> str:
    return hashlib.sha1(path.encode()).hexdigest()[:14] + ".jpg"


def dataset_paths(dataset: str) -> list[str]:
    z = np.load(REPO / "representations" / f"{dataset}_clip_vitb32_all.npz", allow_pickle=False)
    return [str(p) for p in z["paths"]]


def metadata_flag(dataset: str, paths: list[str]) -> np.ndarray:
    if dataset == "wikiart":
        z = np.load(REPO / "data/attributes/wikiart_all.npz", allow_pickle=False)
        defs = {d["key"]: d for d in json.loads(str(z["defs"]))}
        names = defs["genre"]["names"]
        genre = z["a__genre"]
        row = {image_key(p): i for i, p in enumerate(z["paths"])}
        sel = np.array([row[image_key(p)] for p in paths])
        return genre[sel] == names.index("nude_painting")
    return np.array([bool(_TITLE_NUDE.search(Path(p).stem.split("_", 1)[-1].lower())) for p in paths])


def compute(dataset: str, args) -> None:
    paths = dataset_paths(dataset)
    thumbs = REPO / args.thumb_dir
    files = []
    for p in paths:
        t = thumbs / thumb_name(p)
        files.append(str(t if t.exists() else REPO / p))
    print(f"[{dataset}] {len(files):,} images ({sum(f.startswith(str(thumbs)) for f in files):,} via thumbnails)")
    if args.limit:
        files, paths = files[:args.limit], paths[:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    stage_clip, stage_nn = OUT / f"stage_clip_{dataset}.npz", OUT / f"stage_nn_{dataset}.npz"      # resume points
    if stage_clip.exists() and not args.limit:
        clip = np.load(stage_clip)["clip"]
        print(f"[{dataset}] CLIP stage found, reusing it")
    else:
        clip, feat = clip_nudity_scores(files, device=args.device, batch_size=args.batch_size, workers=args.workers)
        np.save(OUT / f"clipL14_features_{dataset}.npy", feat)          # for the probe (scripts/16_train_nudity_probe.py)
        if not args.limit:
            np.savez_compressed(stage_clip, clip=clip)
    if stage_nn.exists() and not args.limit:
        z = np.load(stage_nn)
        nn, nn_cls = z["nn"], z["nn_class"]
        print(f"[{dataset}] NudeNet stage found, reusing it")
    else:
        nn, nn_cls = nudenet_scores(files, workers=args.nn_workers)
        if not args.limit:
            np.savez_compressed(stage_nn, nn=nn, nn_class=nn_cls)
    meta = metadata_flag(dataset, paths)
    np.savez_compressed(OUT / f"scores_{dataset}.npz", keys=np.array([image_key(p) for p in paths]), paths=np.array(paths),
                        clip=clip, nn=nn, nn_class=nn_cls, meta=meta)
    print(f"[{dataset}] saved scores")


def clip32_scores(dataset: str, keys: np.ndarray) -> np.ndarray:
    """The explorer's CLIP ViT-B/32 'nudity' concept score (data/attributes/<dataset>_all.npz) for the given image keys."""
    f = REPO / "data" / "attributes" / f"{dataset}_all.npz"
    if not f.exists():
        print(f"[warn] {f} not found: the CLIP B/32 nudity signal is off for {dataset}")
        return np.full(len(keys), np.nan, dtype=np.float32)
    z = np.load(f, allow_pickle=False)
    row = {image_key(p): i for i, p in enumerate(z["paths"])}
    return z["a__clip_nude"].astype(np.float32)[np.array([row[str(k)] for k in keys])]


def merge(args) -> dict:
    th = {"nn": args.nn_thr, "clip": args.clip_thr, "probe": args.probe_thr, "clip32": args.clip32_thr}
    parts = []
    probe_file = OUT / "probe_scores.npz"
    pz = np.load(probe_file, allow_pickle=False) if probe_file.exists() else None
    if pz is None:
        print("[warn] data/nudity/probe_scores.npz not found (run scripts/16_train_nudity_probe.py): the probe signal is off")
    for ds in args.datasets:
        z = np.load(OUT / f"scores_{ds}.npz", allow_pickle=False)
        part = {k: z[k] for k in z.files}
        if pz is not None and f"probe_{ds}" in pz.files:
            assert (pz[f"keys_{ds}"] == part["keys"]).all(), f"probe scores of {ds} do not match the score file"
            part["probe"] = pz[f"probe_{ds}"]
        else:
            part["probe"] = np.zeros(len(part["keys"]), dtype=np.float32)
        part["clip32"] = clip32_scores(ds, part["keys"])
        parts.append(part)
    cat = lambda k: np.concatenate([p[k] for p in parts])
    keys, clip, nn, meta, probe, c32, paths = (cat("keys"), cat("clip"), cat("nn"), cat("meta"), cat("probe"),
                                               cat("clip32"), cat("paths"))
    reason = build_flags(clip, nn, meta, th, probe, c32)
    np.savez_compressed(FLAGS_FILE, keys=keys, reason=reason, clip=clip, nn=nn, meta=meta, probe=probe, clip32=c32,
                        thresholds=json.dumps(th))
    for ds, p in zip(args.datasets, parts):
        r = build_flags(p["clip"], p["nn"], p["meta"], th, p["probe"], p["clip32"])
        bit = lambda b: int(((r >> b) & 1).sum())
        print(f"[{ds}] flagged {int((r > 0).sum()):,} / {len(r):,} ({100 * (r > 0).mean():.1f}%)   NudeNet {bit(0):,} · "
              f"CLIP-L14 {bit(1):,} · metadata {bit(2):,} · probe {bit(3):,} · CLIP-B32 {bit(4):,}")
    return {"paths": paths, "clip": clip, "nn": nn, "meta": meta, "reason": reason}


def sheets(d: dict, out_dir: Path, thumb_dir: Path, n: int = 24, seed: int = 0) -> None:
    """Contact sheets of random images per score band, to eyeball the thresholds."""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    bands = {
        "clip_0.0-0.1": (d["clip"] < 0.1), "clip_0.1-0.3": (d["clip"] >= 0.1) & (d["clip"] < 0.3),
        "clip_0.3-0.5": (d["clip"] >= 0.3) & (d["clip"] < 0.5), "clip_0.5-0.8": (d["clip"] >= 0.5) & (d["clip"] < 0.8),
        "clip_0.8-1.0": d["clip"] >= 0.8,
        "nn_0.0": d["nn"] == 0, "nn_0.1-0.35": (d["nn"] >= 0.1) & (d["nn"] < 0.35), "nn_0.35-0.6": (d["nn"] >= 0.35) & (d["nn"] < 0.6),
        "nn_0.6-1.0": d["nn"] >= 0.6,
        "unflagged_clip_0.2-0.5": (d["reason"] == 0) & (d["clip"] >= 0.2),
        "only_clip": d["reason"] == 2, "only_nn": d["reason"] == 1, "only_meta": d["reason"] == 4,
    }
    for name, m in bands.items():
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
        cols, cell = 8, 112
        rows = (len(pick) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell, rows * cell), (20, 20, 20))
        for j, i in enumerate(pick):
            p = d["paths"][i]
            f = thumb_dir / thumb_name(str(p))
            with Image.open(f if f.exists() else REPO / str(p)) as im:
                im = im.convert("RGB")
                im.thumbnail((cell - 2, cell - 2))
                sheet.paste(im, ((j % cols) * cell + 1, (j // cols) * cell + 1))
        sheet.save(out_dir / f"{name}.jpg", quality=85)
        print(f"sheet {name}: {len(idx):,} images -> {out_dir / (name + '.jpg')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["artbench", "wikiart"])
    ap.add_argument("--thumb-dir", default="visualizations/explorer/thumbs")
    ap.add_argument("--flags-only", action="store_true", help="skip the detectors; reuse data/nudity/scores_*.npz")
    ap.add_argument("--nn-thr", type=float, default=DEFAULT_THRESHOLDS["nn"], help="NudeNet confidence threshold")
    ap.add_argument("--clip-thr", type=float, default=DEFAULT_THRESHOLDS["clip"], help="CLIP nude-probability threshold")
    ap.add_argument("--probe-thr", type=float, default=DEFAULT_THRESHOLDS["probe"], help="probe probability threshold")
    ap.add_argument("--clip32-thr", type=float, default=DEFAULT_THRESHOLDS["clip32"],
                    help="threshold on the CLIP ViT-B/32 'nudity' attribute (what the explorer colours by)")
    ap.add_argument("--sheets", default=None, metavar="DIR", help="write contact sheets per score band")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N images of each dataset")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16, help="data loader workers for CLIP")
    ap.add_argument("--nn-workers", type=int, default=32, help="processes for NudeNet")
    args = ap.parse_args()
    if not args.flags_only:
        for ds in args.datasets:
            compute(ds, args)
    d = merge(args)
    if args.sheets:
        sheets(d, Path(args.sheets), REPO / args.thumb_dir)
    print(f"flags -> {FLAGS_FILE}")


if __name__ == "__main__":
    main()
