#!/usr/bin/env python3
"""
Extract image representations for the ArtBench-10 dataset.

Usage examples
--------------
# ResNet-50 (ImageNet pretrained) on the training split:
    python scripts/02_extract_features.py \\
        --model resnet50 \\
        --split train \\
        --out representations/resnet50_imagenet_train.npz

# CLIP ViT-B/32:
    python scripts/02_extract_features.py \\
        --model clip_vitb32 \\
        --split train \\
        --out representations/clip_vitb32_train.npz

# DINOv2 ViT-B/14:
    python scripts/02_extract_features.py \\
        --model dinov2_vitb14 \\
        --split train \\
        --out representations/dinov2_vitb14_train.npz

# Custom checkpoint (ResNet fine-tuned on ArtBench):
    python scripts/02_extract_features.py \\
        --model resnet50 \\
        --checkpoint checkpoints/resnet50_artbench.pth \\
        --split train \\
        --out representations/resnet50_artbench_train.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root to path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torchvision import transforms

from src.datasets.artbench import ArtBenchDataset
from src.models.base_extractor import RepresentationBundle


# ── Model registry ────────────────────────────────────────────────────────────

def _get_extractor(model_id: str, checkpoint, batch_size: int, device: str | None):
    """Resolve model ID string to an extractor instance."""
    from src.models.clip_extractor import CLIP_PRESETS, CLIPExtractor
    from src.models.ssl_extractor import SSL_MODELS, SSLExtractor
    from src.models.cnn_extractor import CNNExtractor

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
        "--split",
        choices=["train", "test", "both"],
        default="train",
        help="Dataset split(s) to process.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/artbench",
        help="Root directory of the ArtBench-10 dataset.",
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
        "--device",
        default=None,
        help="Device: cuda | mps | cpu.  Auto-detected if not given.",
    )
    args = parser.parse_args()

    extractor = _get_extractor(
        args.model,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Build the transform from the extractor
    extractor.setup()
    transform = extractor._get_transform()

    splits = ["train", "test"] if args.split == "both" else [args.split]

    for split in splits:
        print(f"\n── Extracting  {args.model}  |  split={split} ──")
        dataset = ArtBenchDataset(
            root=args.data_dir,
            split=split,
            transform=transform,
            return_path=True,
        )
        print(f"  Dataset: {len(dataset)} images")

        if args.out and len(splits) == 1:
            out_path = args.out
        else:
            model_slug = args.model.replace("/", "_").replace(":", "_")
            out_path = f"representations/{model_slug}_{split}.npz"

        bundle = extractor.extract(
            dataset,
            save_path=out_path,
            show_progress=True,
        )
        print(f"  {bundle}")

    print("\nDone.")


if __name__ == "__main__":
    main()
