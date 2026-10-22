#!/usr/bin/env python3
"""
Encode all cached 128 px paintings (scripts/20_prepare_ar_data.py) into 16 x 16 = 256 discrete tokens with a trained VQ
tokenizer (scripts/21_train_vq.py) once, so the GPT training only reads token arrays:

    data/ar_tokens/tokens_<tokenizer run name>_<dataset>_<split>.npy        (N, 256) uint16, raster order
    data/ar_tokens/tokens_<tokenizer run name>_<dataset>_train_flip.npy     the horizontally flipped train images (augmentation)

    python scripts/22_tokenize_ar.py --tokenizer runs/<stamp>_vq_artbench_wikiart/vq_final.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.models.ar_data import DATASETS, SPLITS, pixel_file, token_file
from src.models.vq_tokenizer import VQConfig, VQTokenizer


def load_tokenizer(path: str | Path, device) -> tuple[VQTokenizer, str]:
    path = Path(path).resolve()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = VQTokenizer(VQConfig.from_dict(ck["model_cfg"]))
    model.load_state_dict(ck["model"])
    run = path.parent.parent if path.parent.name == "checkpoints" else path.parent
    return model.to(device).eval(), run.name


@torch.no_grad()
def encode_array(model: VQTokenizer, arr: np.ndarray, device, flip: bool, bs: int = 256) -> np.ndarray:
    out = np.empty((len(arr), model.cfg.n_tokens), np.uint16)
    for i in range(0, len(arr), bs):
        x = torch.from_numpy(np.ascontiguousarray(arr[i: i + bs])).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1
        if flip:
            x = x.flip(-1)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            ids = model.encode(x)
        out[i: i + bs] = ids.reshape(len(x), -1).cpu().numpy().astype(np.uint16)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="vq_final.pt (or a checkpoint) of scripts/21_train_vq.py")
    ap.add_argument("--cache-dir", default="data/ar_tokens")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, name = load_tokenizer(args.tokenizer, device)
    print(f"tokenizer {name}: {model.cfg}")
    for ds in DATASETS:
        for split in SPLITS:
            arr = np.load(pixel_file(ds, split, args.cache_dir), mmap_mode="r")
            for flip in ((False, True) if split == "train" else (False,)):
                out = token_file(name, ds, split, flip, args.cache_dir)
                if out.exists() and not args.overwrite:
                    print(f"{out} exists"); continue
                t0 = time.time()
                toks = encode_array(model, arr, device, flip)
                np.save(out, toks)
                used = len(np.unique(toks))
                print(f"{ds}/{split}{'/flip' if flip else ''}: {toks.shape} in {time.time() - t0:.0f}s, "
                      f"{used}/{model.cfg.codebook_size} codes used -> {out}", flush=True)


if __name__ == "__main__":
    main()
