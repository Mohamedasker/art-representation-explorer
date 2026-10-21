#!/usr/bin/env python3
"""
Extract image representations for ArtBench-10 or WikiArt.

Usage examples
--------------
# ResNet-50 (ImageNet pretrained) on the ArtBench-10 training split:
    python scripts/02_extract_features.py \\
        --model resnet50 \\
        --dataset artbench \\
        --split train \\
        --out representations/artbench_resnet50_imagenet_train.npz

# CLIP ViT-B/32 on WikiArt:
    python scripts/02_extract_features.py \\
        --model clip_vitb32 \\
        --dataset wikiart \\
        --split train \\
        --out representations/wikiart_clip_vitb32_train.npz

# DINOv2 ViT-B/14:
    python scripts/02_extract_features.py \\
        --model dinov2_vitb14 \\
        --dataset artbench \\
        --split train \\
        --out representations/artbench_dinov2_vitb14_train.npz

# Autoencoder trained with scripts/07_train_autoencoder.py (1024-D pooled latent):
    python scripts/02_extract_features.py \\
        --model autoencoder \\
        --checkpoint runs/<run>/autoencoder_final.pt \\
        --dataset artbench \\
        --split train

# Supervised multi-task ResNet trained with scripts/10_train_cnn.py (2048-D pooled feature):
    python scripts/02_extract_features.py \\
        --model cnn_supervised \\
        --checkpoint runs/<run>/cnn_final.pt \\
        --dataset artbench \\
        --split train

# Masked autoencoder: Meta's ImageNet ViT-B/16 (frozen), and one trained with scripts/12_train_mae.py
# (mean-pooled patch tokens of the encoder on the unmasked image, 768-D):
    python scripts/02_extract_features.py --model mae_imagenet --dataset artbench --split both
    python scripts/02_extract_features.py --model mae --checkpoint runs/<run>/mae_final.pt \\
        --dataset wikiart --split both        # -> representations/wikiart_mae_adapt_artbench_{train,test}.npz

# SimCLR ResNet-50 (2048-D pooled encoder output): the published ImageNet SimCLR v1 weights (frozen), and one trained
# with scripts/13_train_simclr.py (-> representations/<dataset>_simclr_<train dataset>_<aug set>_{train,test}.npz):
    python scripts/02_extract_features.py --model simclr_imagenet --dataset artbench --split both
    python scripts/02_extract_features.py --model simclr --checkpoint runs/<run>/simclr_final.pt \\
        --dataset artbench --split both

# Custom checkpoint (ResNet fine-tuned on ArtBench):
    python scripts/02_extract_features.py \\
        --model resnet50 \\
        --checkpoint checkpoints/resnet50_artbench.pth \\
        --dataset artbench \\
        --split train \\
        --out representations/artbench_resnet50_artbench_train.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root to path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torchvision import transforms

from src.datasets import build_dataset, default_data_dir
from src.models.base_extractor import RepresentationBundle


# ── Model registry ────────────────────────────────────────────────────────────

def _get_extractor(model_id: str, checkpoint, batch_size: int, device: str | None, ae_pool: int = 4,
                   mae_pool: str = "mean"):
    """Resolve model ID string to an extractor instance."""
    from src.models.clip_extractor import CLIP_PRESETS, CLIPExtractor
    from src.models.ssl_extractor import SSL_MODELS, SSLExtractor
    from src.models.cnn_extractor import CNNExtractor

    if model_id == "autoencoder":
        from src.models.ae_extractor import AutoencoderExtractor
        return AutoencoderExtractor(checkpoint=checkpoint, pool=ae_pool, batch_size=batch_size, device=device)

    if model_id in ("mae", "mae_imagenet"):
        from src.models.mae_extractor import MAEExtractor
        return MAEExtractor(checkpoint=checkpoint if model_id == "mae" else None, pool=mae_pool,
                            batch_size=batch_size, device=device)

    if model_id in ("simclr", "simclr_imagenet"):
        from src.models.simclr_extractor import SimCLRExtractor
        return SimCLRExtractor(checkpoint=checkpoint if model_id == "simclr" else None,
                               batch_size=batch_size, device=device)

    if model_id == "cnn_supervised":
        from src.models.supervised_extractor import SupervisedCNNExtractor
        return SupervisedCNNExtractor(checkpoint=checkpoint, batch_size=batch_size, device=device)

    if model_id in CLIP_PRESETS or model_id.startswith("clip_"):
        key = model_id if model_id in CLIP_PRESETS else model_id
        return CLIPExtractor(model_name=key, batch_size=batch_size, device=device)

    if model_id in SSL_MODELS or model_id.startswith("dino"):
        return SSLExtractor(model_id=model_id, batch_size=batch_size, device=device)

    # fall through to CNN extractor
    return CNNExtractor(
        model_id=model_id,
        checkpoint=checkpoint,
        batch_size=batch_size,
        device=device,
    )


MODEL_SLUGS = {"resnet50": "resnet50_imagenet"}      # file-name slug for model ids that differ from it


def _update_manifest(out_path, bundle, args, split: str) -> None:
    """Keep representations/manifest.json up to date (what each saved .npz contains)."""
    import datetime
    import json
    mf = Path("representations/manifest.json")
    data = json.loads(mf.read_text()) if mf.exists() else {}
    data[Path(out_path).name] = {
        "dataset": args.dataset, "split": split, "n": int(bundle.n_samples), "dim": int(bundle.dim),
        "model_name": bundle.model_name, "layer": bundle.layer_name, "model_arg": args.model,
        "checkpoint": args.checkpoint, "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    mf.write_text(json.dumps(data, indent=1, sort_keys=True))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract image representations from ArtBench-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="resnet50",
        help=(
            "Model to use.  Options: resnet50 | efficientnet_b0 | vit_base_patch16_224 "
            "| clip_vitb32 | clip_vitl14 | dinov2_vitb14 | dinov2_vitl14 | … "
            "Any timm model name is also valid."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a fine-tuned checkpoint (.pt / .pth). Optional.",
    )
    parser.add_argument(
        "--ae-pool",
        type=int,
        default=4,
        help="--model autoencoder only: average-pool the 8x8 latent to NxN (0 = full 4096-D latent).",
    )
    parser.add_argument(
        "--mae-pool",
        choices=["mean", "cls", "both"],
        default="mean",
        help="--model mae / mae_imagenet only: mean-pooled patch tokens (768-D), CLS token, or both.",
    )
    parser.add_argument(
        "--dataset",
        choices=["artbench", "wikiart"],
        default="artbench",
        help="Which dataset to extract features from.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "both", "all"],
        default="train",
        help="Dataset split(s) to process.  'both' = separate train and test files (existing ones are reused); "
             "'all' = additionally merge them into one <dataset>_<model>_all.npz (not used by the explorer).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Root directory of the dataset. Defaults to data/artbench or "
            "data/wikiart depending on --dataset."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output .npz path.  If not given, a name is derived from "
            "--model and --split and saved in representations/."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Images per GPU batch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="With --split both/all: re-extract even if per-split files for this model already exist.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: cuda | mps | cpu.  Auto-detected if not given.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or default_data_dir(args.dataset)

    extractor = _get_extractor(
        args.model,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
        ae_pool=args.ae_pool,
        mae_pool=args.mae_pool,
    )

    # Build the transform from the extractor
    extractor.setup()
    transform = extractor._get_transform()

    splits = ["train", "test"] if args.split in ("both", "all") else [args.split]
    model_slug = MODEL_SLUGS.get(args.model, args.model).replace("/", "_").replace(":", "_")
    if args.checkpoint and args.model == "resnet50":
        model_slug = "resnet50_custom"
    if hasattr(extractor, "file_slug"):      # MAE: slug names the model (and the dataset it was trained on)
        model_slug = extractor.file_slug + ("" if args.mae_pool == "mean" else f"_{args.mae_pool}")
    parts: list[RepresentationBundle] = []

    for split in splits:
        out_path = (args.out if (args.out and len(splits) == 1)
                    else f"representations/{args.dataset}_{model_slug}_{split}.npz")
        existing = Path(out_path)
        if args.split in ("both", "all") and existing.exists() and not args.no_reuse:
            old = RepresentationBundle.load(existing)
            if old.model_name == extractor.model_name:
                print(f"\n── {args.dataset} / {split}: reusing {existing} ({old.n_samples} images) ──")
                _update_manifest(existing, old, args, split)
                parts.append(old)
                continue
            print(f"  {existing} was made by {old.model_name!r}, not {extractor.model_name!r}: re-extracting")
        print(f"\n── Extracting  {args.model}  |  dataset={args.dataset}  |  split={split} ──")
        dataset = build_dataset(
            args.dataset,
            root=data_dir,
            split=split,
            transform=transform,
            return_path=True,
        )
        print(f"  Dataset: {len(dataset)} images")

        bundle = extractor.extract(
            dataset,
            save_path=out_path,
            show_progress=True,
        )
        print(f"  {bundle}")
        _update_manifest(out_path, bundle, args, split)
        parts.append(bundle)

    if args.split == "all":
        import numpy as np
        merged = RepresentationBundle(
            vectors=np.concatenate([b.vectors for b in parts]),
            labels=np.concatenate([b.labels for b in parts]),
            label_names=np.concatenate([b.label_names for b in parts]),
            paths=np.concatenate([b.paths for b in parts]),
            model_name=parts[0].model_name, layer_name=parts[0].layer_name,
            metadata={**parts[0].metadata, "n_samples": int(sum(b.n_samples for b in parts)),
                      "splits": {sp: b.n_samples for sp, b in zip(splits, parts)}, "whole_dataset": True},
        )
        out_all = f"representations/{args.dataset}_{model_slug}_all.npz"
        merged.save(out_all)
        _update_manifest(out_all, merged, args, "all")
        print(f"\n  WHOLE DATASET: {merged}  ->  {out_all}")

    print("\nDone.")


if __name__ == "__main__":
    main()
