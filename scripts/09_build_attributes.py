#!/usr/bin/env python3
"""
Build per-image attributes for the explorer colour maps.

  * WikiArt : genre (11) and artist (129) from the cached HuggingFace arrow files
              (file name = md5 of the HF row index, so the mapping is exact).
  * ArtBench: artist and keyword tags parsed from the file names ("artist_title-words-year.jpg").
  * both    : CLIP zero-shot content scores (woman, man, nudity, Christian, city, ...), computed
              from the existing CLIP image embeddings in representations/ (no image is re-read).

Output: data/attributes/<dataset>_<split>.npz  (read by scripts/08_build_explorer.py)

Usage
-----
    python scripts/09_build_attributes.py
    python scripts/09_build_attributes.py --datasets wikiart --clip representations/wikiart_clip_vitb32_train.npz
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.metadata import attributes as A
from src.metadata import clip_concepts as C
from src.metadata import pixel_features as P
from src.models.base_extractor import RepresentationBundle


def cat_def(key, label, group, names):
    return {"key": key, "label": label, "group": group, "kind": "cat", "names": names}


def num_def(key, label, group):
    return {"key": key, "label": label, "group": group, "kind": "num"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["artbench", "wikiart"], choices=["artbench", "wikiart"])
    ap.add_argument("--split", nargs="+", default=["train", "test"], help="train, test and/or all (merged train + test, adds a split attribute); one attribute file each")
    ap.add_argument("--clip", default=None, help="CLIP .npz for a single --datasets entry "
                    "(default representations/<dataset>_clip_vitb32_<split>.npz)")
    ap.add_argument("--top-artists", type=int, default=20)
    ap.add_argument("--out-dir", default="data/attributes")
    ap.add_argument("--no-pixels", action="store_true", help="skip the pixel statistics (B/W, skin tone, margins)")
    ap.add_argument("--workers", type=int, default=32, help="processes for the pixel statistics")
    args = ap.parse_args()

    print("encoding CLIP text prompts ...")
    concept, null, groups, pairs = C.text_embeddings("ViT-B-32", "openai")

    for ds, split in itertools.product(args.datasets, args.split):
        clip_path = Path(args.clip) if args.clip else REPO_ROOT / f"representations/{ds}_clip_vitb32_{split}.npz"
        clip = RepresentationBundle.load(clip_path)
        paths = [str(p) for p in clip.paths]
        stems = [Path(p).stem for p in paths]
        print(f"\n── {ds}: {len(paths)} images (CLIP rows from {clip_path.name}) ──")
        defs, arrays = [], {}
        # which split each image comes from (matters for the models that were trained on the train split)
        if split == "all":
            sp = np.array([0 if "/train/" in p else 1 for p in paths], dtype=np.int16)
            defs.append(cat_def("split", "split (train / test)", "Dataset", ["train", "test"]))
            arrays["split"] = sp

        if ds == "wikiart":
            lab = A.wikiart_labels(REPO_ROOT / "data" / "wikiart")
            miss = [s for s in stems if s not in lab]
            print(f"  HF labels found for {len(stems) - len(miss)} / {len(stems)}")
            genre, gnames = A.top_categories([lab.get(s, {}).get("genre") for s in stems], 20)
            artist, anames = A.top_categories([lab.get(s, {}).get("artist") for s in stems], args.top_artists)
            defs += [cat_def("genre", "genre", "WikiArt metadata", gnames),
                     cat_def("artist", f"artist (top {args.top_artists})", "WikiArt metadata", anames)]
            arrays |= {"genre": genre, "artist": artist}
        else:
            artist, anames = A.top_categories([A.artbench_artist(s) for s in stems], args.top_artists)
            defs.append(cat_def("artist", f"artist (top {args.top_artists})", "ArtBench file-name metadata", anames))
            arrays["artist"] = artist
            tags = [A.title_tags(s) for s in stems]
            for tag in A.TITLE_TAGS:
                v = np.array([t[tag] for t in tags], dtype=np.int16)
                print(f"  title tag {tag!r}: {int(v.sum())}")
                key = "tag_" + tag.replace(" / ", "_").replace(" ", "_")
                defs.append(cat_def(key, f"title has: {tag}", "ArtBench title keywords", ["no", "yes"]))
                arrays[key] = v

        num, cats, pair = C.score(clip.vectors, concept, null, groups, pairs)
        for j, (k, (label, group, _)) in enumerate(C.CONCEPTS.items()):
            defs.append(num_def(f"clip_{k}", label, group + " (similarity)"))
            arrays[f"clip_{k}"] = num[:, j]
        for j, (k, (label, group, _, _)) in enumerate(C.PAIRS.items()):
            defs.append(num_def(f"pair_{k}", label, group + " (similarity)"))
            arrays[f"pair_{k}"] = pair[:, j]
        for k, (label, group, classes, _) in C.GROUPS.items():
            defs.append(cat_def(k, label, group + " (argmax)", list(classes)))
            arrays[k] = cats[k]
            counts = np.bincount(cats[k], minlength=len(classes))
            print(f"  {label}: " + ", ".join(f"{c}={n}" for c, n in zip(classes, counts)))

        if not args.no_pixels:
            cache = REPO_ROOT / args.out_dir / f"pixels_{ds}_{split}.npz"
            parts = [REPO_ROOT / args.out_dir / f"pixels_{ds}_{sp}.npz" for sp in ("train", "test")]
            if split == "all" and not cache.exists() and all(q.exists() for q in parts) and \
                    all(all(f in np.load(q) for f in P.FIELDS) for q in parts):
                zs = [np.load(q) for q in parts]                   # reuse the per-split statistics
                row = {str(pp): (k, i) for k, z in enumerate(zs) for i, pp in enumerate(z["paths"])}
                if all(pp in row for pp in paths):
                    np.savez_compressed(cache, paths=np.array(paths), **{
                        f: np.array([zs[row[pp][0]][f][row[pp][1]] for pp in paths]) for f in P.FIELDS})
                    print("  pixel statistics: assembled from the train and test files")
            if cache.exists() and list(np.load(cache)["paths"]) == paths and all(f in np.load(cache) for f in P.FIELDS):
                z = np.load(cache)
                pix = {k: z[k] for k in P.FIELDS}
                print("  pixel statistics: loaded from", cache.name)
            else:
                print(f"  pixel statistics for {len(paths)} images ({args.workers} workers) ...")
                pix = P.compute_all([str(REPO_ROOT / p) for p in paths], workers=args.workers)
                np.savez_compressed(cache, paths=np.array(paths), **pix)
            G = "Image statistics (pixels)"
            defs += [num_def("pix_saturation", "saturation (mean)", G), num_def("pix_brightness", "brightness", G),
                     num_def("pix_contrast", "contrast", G), num_def("pix_paper", "paper-white background fraction", G),
                     cat_def("pix_colour_mode", "colour mode (greyscale / monotone / colour)", G, P.COLOUR_MODES)]
            arrays |= {"pix_saturation": pix["saturation"], "pix_brightness": pix["brightness"],
                       "pix_contrast": pix["contrast"], "pix_paper": pix["paper"],
                       "pix_colour_mode": pix["colour_mode"].astype(np.int16)}
            # colour histogram: hue-family shares and what can be derived from them
            H = "Colour histogram (hue shares, pixels)"
            shares = np.stack([pix[f"h_{n}"] for n in P.HUE_NAMES], axis=1)            # N x 9, rows sum to 1
            chromatic = 1.0 - shares[:, -1]
            for j, n in enumerate(P.HUE_NAMES):
                label = f"share of {n} pixels" if n != "neutral" else "share of neutral pixels (grey / black / white)"
                defs.append(num_def(f"hist_{n}", label, H))
                arrays[f"hist_{n}"] = shares[:, j]
            dom = shares.argmax(1)
            dom = np.where(chromatic < 0.05, len(P.HUE_NAMES) - 1, dom)               # almost no colour at all -> neutral
            defs.append(cat_def("hist_dominant", "dominant hue family", H, list(P.HUE_NAMES)))
            arrays["hist_dominant"] = dom.astype(np.int16)
            warm, cool = shares[:, P.WARM].sum(1), shares[:, P.COOL].sum(1)
            balance = np.where(chromatic > 0.05, (warm - cool) / np.maximum(warm + cool, 1e-6), np.nan)
            defs.append(num_def("hist_warm_cool", "warm \u2194 cool balance (warm = red/orange/yellow/magenta)", H))
            arrays["hist_warm_cool"] = balance.astype(np.float32)
            p = np.clip(shares, 1e-12, 1)
            defs.append(num_def("hist_entropy", "colour diversity (hue-histogram entropy, bits)", H))
            arrays["hist_entropy"] = (-(shares * np.log2(p)).sum(1)).astype(np.float32)
            print("  dominant hue family: " + ", ".join(f"{n}={int((dom == j).sum())}" for j, n in enumerate(P.HUE_NAMES)))

        out = REPO_ROOT / args.out_dir / f"{ds}_{split}.npz"
        A.save_attributes(out, paths, defs, arrays)
        print(f"  wrote {out}  ({len(defs)} attributes)")


if __name__ == "__main__":
    main()
