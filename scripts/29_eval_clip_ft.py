#!/usr/bin/env python3
"""
Fit vs. forgetting for the fine-tuned CLIPs (scripts/28_train_clip_ft.py) -> docs/clip_ft_eval.md (+ .json).

For the original CLIP and for every fine-tuned run (alpha = 1 and the WiSE-FT interpolations --alphas):
  * zero-shot style accuracy ("a painting in the X style") on the test split of ArtBench and of WikiArt; for a model
    trained on dataset A and evaluated on dataset B the test images with a near-duplicate in A's TRAIN split are left
    out (CLIP cosine >= --dup-threshold, from the frozen representations/<dataset>_clip_vitb32_qgelu_all.npz), the number of
    excluded images is reported,
  * zero-shot top-1 on CIFAR-100 / CIFAR-10 (data/generic) -- general knowledge the fine-tuning should not destroy,
  * kNN style accuracy (k = 20, cosine) of the extracted representations, if representations/<dataset>_<slug>_{train,test}.npz exist.

    python scripts/29_eval_clip_ft.py --run artbench=runs/<stamp>_clip_ft_artbench --run wikiart=runs/<stamp>_clip_ft_wikiart
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import default_data_dir
from src.models import clip_ft as C
from src.models.base_extractor import RepresentationBundle

DATASETS = ("artbench", "wikiart")


def key_of(path: str) -> str:
    p = Path(path)
    return f"{p.parent.name}/{p.name}"


def near_duplicates(test_ds: str, train_ds: str, thr: float, repr_dir: Path) -> dict | None:
    """Mask over the test split of ``test_ds`` (order of ``C.dataset_items``): has a CLIP near-duplicate in the train
    split of ``train_ds``.  None when the frozen representations are not there."""
    files = {d: repr_dir / f"{d}_clip_vitb32_qgelu_all.npz" for d in (test_ds, train_ds)}
    if not all(f.exists() for f in files.values()):
        return None
    te, tr = (RepresentationBundle.load(files[d]) for d in (test_ds, train_ds))
    te_m, tr_m = te.splits == "test", tr.splits == "train"
    a = torch.from_numpy(te.vectors[te_m]).float()
    b = torch.from_numpy(tr.vectors[tr_m]).float()
    a, b = a / a.norm(dim=1, keepdim=True), b / b.norm(dim=1, keepdim=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = b.to(dev)
    best = torch.cat([(a[i:i + 2048].to(dev) @ b.T).max(1).values.cpu() for i in range(0, len(a), 2048)])
    dup = dict(zip((key_of(p) for p in te.paths[te_m]), (best >= thr).numpy()))
    return dup


def knn_accuracy(slug: str, ds: str, repr_dir: Path, k: int = 20) -> float | None:
    f = {s: repr_dir / f"{ds}_{slug}_{s}.npz" for s in ("train", "test")}
    if not all(p.exists() for p in f.values()):
        return None
    tr, te = RepresentationBundle.load(f["train"]), RepresentationBundle.load(f["test"])
    a = torch.from_numpy(te.vectors).float()
    b = torch.from_numpy(tr.vectors).float()
    a, b = a / a.norm(dim=1, keepdim=True), b / b.norm(dim=1, keepdim=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = b.to(dev)
    ytr = torch.from_numpy(tr.labels).to(dev)
    n_cls = int(max(tr.labels.max(), te.labels.max())) + 1
    pred = []
    for i in range(0, len(a), 1024):
        idx = (a[i:i + 1024].to(dev) @ b.T).topk(k, dim=1).indices
        votes = torch.zeros(len(idx), n_cls, device=dev).scatter_add_(1, ytr[idx], torch.ones_like(idx, dtype=torch.float))
        pred.append(votes.argmax(1).cpu())
    return float((torch.cat(pred).numpy() == te.labels).mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", default=[], metavar="DATASET=RUNDIR",
                   help="fine-tuned run, named by the dataset it was trained on (repeatable)")
    p.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.5])
    p.add_argument("--arch", default=C.DEFAULT_ARCH)
    p.add_argument("--pretrained", default=C.DEFAULT_PRETRAINED)
    p.add_argument("--generic-dir", default="data/generic")
    p.add_argument("--repr-dir", default="representations")
    p.add_argument("--dup-threshold", type=float, default=0.95)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--out", default="docs/clip_ft_eval")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repr_dir = Path(args.repr_dir)

    runs = dict(r.split("=", 1) for r in args.run)
    models = [("original CLIP", None, None, 1.0, "clip_vitb32_qgelu")]
    for ds, rd in runs.items():
        rd = Path(rd)
        ck = rd / "clip_ft_final.pt"
        for a in args.alphas:
            slug = f"clip_ft_{ds}" + ("" if a >= 1 else f"_wise{round(a * 100):d}")
            models.append((f"fine-tuned on {ds}" + ("" if a >= 1 else f" (alpha {a:g})"), ds, ck, a, slug))

    base, tokenizer, preprocess = C.load_clip(args.arch, args.pretrained, device)
    data = {}
    for ds in DATASETS:
        paths, styles, _ = C.dataset_items(ds, default_data_dir(ds), "test", default_data_dir("wikiart"))
        names = sorted(set(styles))
        data[ds] = {"paths": paths, "labels": torch.tensor([names.index(s) for s in styles]), "names": names,
                    "text": C.encode_texts(base, tokenizer, [C.style_prompt(s) for s in names], device)}
    dups = {(a, b): near_duplicates(b, a, args.dup_threshold, repr_dir) for a in runs for b in DATASETS if a != b}
    generic_text: dict = {}

    rows = []
    for label, train_ds, ck, alpha, slug in models:
        model, _, _ = C.load_clip(args.arch, args.pretrained, device)
        if ck is not None:
            C.load_ft_visual(model, torch.load(ck, map_location="cpu", weights_only=False), alpha)
        model.eval()
        row = {"model": label, "slug": slug}
        for ds in DATASETS:
            d = data[ds]
            loader = DataLoader(C.CaptionedImages(d["paths"], [0] * len(d["paths"]), d["labels"].numpy(), preprocess),
                                batch_size=args.batch_size, num_workers=args.num_workers)
            feats, (_cid, sid) = C.encode_images(model, loader, device)
            correct = ((feats @ d["text"].T).argmax(1).cpu() == sid).numpy()
            row[f"zs_style/{ds}"] = float(correct.mean())
            if train_ds is not None and train_ds != ds and dups.get((train_ds, ds)) is not None:
                keep = np.array([not dups[(train_ds, ds)][key_of(p)] for p in d["paths"]])
                row[f"zs_style/{ds}_no_dups"] = float(correct[keep].mean())
                row[f"excluded/{ds}"] = int((~keep).sum())
            kn = knn_accuracy(slug, ds, repr_dir)
            if kn is not None:
                row[f"knn_style/{ds}"] = kn
        row.update({f"zs/{k}": v for k, v in C.generic_zero_shot(
            model, tokenizer, preprocess, args.generic_dir, device, args.batch_size, args.num_workers,
            text_cache=generic_text).items()})
        rows.append(row)
        print(json.dumps(row))
        del model

    cols = [("zs_style/artbench", "style zero-shot, ArtBench test"), ("zs_style/wikiart", "style zero-shot, WikiArt test"),
            ("zs_style/artbench_no_dups", "ArtBench test, no near-dups of WikiArt train"),
            ("zs_style/wikiart_no_dups", "WikiArt test, no near-dups of ArtBench train"),
            ("knn_style/artbench", "kNN style, ArtBench"), ("knn_style/wikiart", "kNN style, WikiArt"),
            ("zs/cifar100", "CIFAR-100 zero-shot"), ("zs/cifar10", "CIFAR-10 zero-shot")]
    cols = [c for c in cols if any(c[0] in r for r in rows)]
    md = ["# CLIP fine-tuning: fit vs. forgetting", "",
          f"{args.arch} / {args.pretrained}; top-1 accuracy in %. Near-duplicate threshold (CLIP cosine): {args.dup_threshold}.",
          "", "| model | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for r in rows:
        md.append(f"| {r['model']} | " + " | ".join(f"{100 * r[c[0]]:.1f}" if c[0] in r else "–" for c in cols) + " |")
    excl = sorted({(k, v) for r in rows for k, v in r.items() if k.startswith("excluded/")})
    if excl:
        md += ["", "Excluded near-duplicates: " + ", ".join(f"{k.split('/')[1]} test: {v}" for k, v in excl)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
