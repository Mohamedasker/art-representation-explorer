#!/usr/bin/env python3
"""
Label every painting with ImageNet classes by CLIP zero-shot classification, and add them to the attribute files as
colour maps for the explorer (see src/metadata/imagenet_labels.py).

Uses the CLIP ViT-L/14 image features that scripts/15_flag_nudity.py saved for all images (data/nudity/
clipL14_features_<dataset>.npy), so no image is read again; only the text side of CLIP runs: the 1000 ImageNet class
names with the 80 standard OpenAI prompt templates.

Output: data/imagenet/labels_<dataset>.npz (top-1/top-5 class, probabilities, probability mass per WordNet group) and the
imagenet_* attributes in data/attributes/<dataset>_<split>.npz.  Then run scripts/rerender_explorer.py to update the pages.

    python scripts/17_label_imagenet.py
    python scripts/17_label_imagenet.py --rebuild-groups        # redo the WordNet class -> group table (needs nltk)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.metadata import imagenet_labels as L
from src.metadata.nudity import image_key


def text_classifier(device: str) -> np.ndarray:
    """(1000, 768) unit-norm CLIP ViT-L/14 class embeddings: the mean over the OpenAI ImageNet prompt templates."""
    import open_clip
    import torch
    from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)
    model.eval()
    tok = open_clip.get_tokenizer("ViT-L-14")
    out = []
    with torch.no_grad():
        for name in IMAGENET_CLASSNAMES:
            t = model.encode_text(tok([tpl(name) for tpl in OPENAI_IMAGENET_TEMPLATES]).to(device))
            t = t / t.norm(dim=-1, keepdim=True)
            m = t.mean(dim=0)
            out.append((m / m.norm()).float().cpu())
    return torch.stack(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["artbench", "wikiart"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rebuild-groups", action="store_true", help="rebuild src/metadata/imagenet_groups.json from WordNet")
    ap.add_argument("--no-attributes", action="store_true", help="only write the label scores")
    args = ap.parse_args()
    if args.rebuild_groups or not L.GROUPS_FILE.exists():
        L.build_groups()
    meta = L.load_groups()
    class_group = np.array(meta["class_group"])
    onehot = np.zeros((1000, len(meta["groups"])), dtype=np.float32)
    onehot[np.arange(1000), class_group] = 1.0

    print("encoding the ImageNet class prompts with CLIP ViT-L/14 ...")
    T = text_classifier(args.device)
    L.LABELS_DIR.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        z = np.load(REPO / "data" / "nudity" / f"scores_{ds}.npz", allow_pickle=False)     # same image order as the features
        feats = np.load(REPO / "data" / "nudity" / f"clipL14_features_{ds}.npy")
        assert len(feats) == len(z["keys"]), f"{ds}: features and keys differ in length"
        top1, top5, prob1, gprob = [], [], [], []
        for s in range(0, len(feats), 8192):
            logits = 100.0 * feats[s:s + 8192].astype(np.float32) @ T.T
            logits -= logits.max(1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(1, keepdims=True)
            order = np.argsort(-p, axis=1)[:, :5]
            top5.append(order.astype(np.int16))
            top1.append(order[:, 0].astype(np.int16))
            prob1.append(np.take_along_axis(p, order[:, :1], 1)[:, 0].astype(np.float16))
            gprob.append((p @ onehot).astype(np.float16))
        np.savez_compressed(L.labels_file(ds), keys=z["keys"], top1=np.concatenate(top1), top5=np.concatenate(top5),
                            prob1=np.concatenate(prob1), group_prob=np.concatenate(gprob))
        p1, g = np.concatenate(prob1).astype(np.float32), np.concatenate(gprob).astype(np.float32)
        print(f"[{ds}] {len(feats):,} images; top-1 probability: median {np.median(p1):.2f}, >= {L.MIN_CLASS_PROB}: "
              f"{100 * (p1 >= L.MIN_CLASS_PROB).mean():.0f}%;  confident groups (>= {L.MIN_GROUP_PROB}): "
              f"{100 * (g.max(1) >= L.MIN_GROUP_PROB).mean():.0f}%")
        counts = np.bincount(g.argmax(1)[g.max(1) >= L.MIN_GROUP_PROB], minlength=len(meta["groups"]))
        print("   " + ", ".join(f"{n}={c}" for n, c in sorted(zip(meta["groups"], counts), key=lambda t: -t[1])[:8]))
    if not args.no_attributes:
        for ds in args.datasets:
            for f in sorted((REPO / "data" / "attributes").glob(f"{ds}_*.npz")):
                if f.name.startswith("pixels_") or "clip_text" in f.name or f.stem.endswith(".tmp"):
                    continue
                print(f"{f.name}: " + ("imagenet attributes added" if L.add_to_attribute_file(f, ds) else "skipped"))


if __name__ == "__main__":
    main()
